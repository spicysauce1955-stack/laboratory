"""SkyPilot backend — remote execution on any SkyPilot cloud (Vast.ai for us).

``submit`` spawns a detached local supervisor (``lab.sky_runner``) that performs the *blocking*
``sky.launch`` (provision + run), so submission returns immediately (FR-A1) and the job survives
the CLI/MCP process exiting (NFR-2). The supervisor records terminal state, rsyncs outputs back,
and tears the instance down (FR-C2). ``sky.launch(down=True, idle_minutes_to_autostop=…)`` is the
cost-safety guarantee even if the supervisor dies (NFR-7).

P0 limitations (tracked): artifacts are rsynced from the live instance before teardown — durable
object storage (R2/S3) is a P1 item (research/15); a wall-clock timeout surfaces as ``failed``
(the remote ``timeout`` kills the job) rather than ``timed_out``.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lab._util import infer_artifact_type, now, parse_duration
from lab.manifest import sha256_file
from lab.metrics import METRICS_FILE, read_points
from lab.models import ArtifactRecord, JobManifest, JobState
from lab.store import JobStore

if TYPE_CHECKING:
    import sky

REMOTE_RUN_DIR = "/tmp/lab_run"
DEFAULT_AUTOSTOP_MIN = 5  # safety-net teardown if the supervisor process dies
_TERMINAL = {JobState.succeeded, JobState.failed, JobState.cancelled, JobState.timed_out}

# SkyPilot JobStatus name -> lab JobState (pure; unit-tested).
_STATUS_MAP = {
    "INIT": JobState.queued,
    "PENDING": JobState.queued,
    "SETTING_UP": JobState.queued,
    "RUNNING": JobState.running,
    "SUCCEEDED": JobState.succeeded,
    "FAILED": JobState.failed,
    "FAILED_SETUP": JobState.failed,
    "FAILED_DRIVER": JobState.failed,
    "CANCELLED": JobState.cancelled,
}


def map_job_status(status_name: str) -> JobState:
    """Map a SkyPilot JobStatus name to a lab JobState (unknown -> failed)."""
    return _STATUS_MAP.get(status_name, JobState.failed)


def cluster_name_for(job_id: str) -> str:
    """SkyPilot cluster name: starts with a letter, lowercase alnum + hyphen."""
    safe = re.sub(r"[^a-z0-9-]", "-", job_id.lower()).strip("-")
    return f"lab-{safe}"[:60]


def build_setup_script() -> str:
    """Install uv and materialise the locked env on the remote (FR-B2)."""
    return (
        "set -e\n"
        "curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        "uv sync --frozen --no-dev\n"
    )


def build_run_script(manifest: JobManifest) -> str:
    """Activate the env, then run the entrypoint under a wall-clock timeout (FR-I1)."""
    timeout = parse_duration(manifest.resources.timeout)
    guard = f"timeout {int(timeout)} " if timeout else ""
    return (
        'export PATH="$HOME/.local/bin:$PATH"\n'
        "source .venv/bin/activate\n"
        f'mkdir -p "{REMOTE_RUN_DIR}"\n'
        f"{guard}{manifest.run.entrypoint_command}\n"
    )


def build_task(manifest: JobManifest, workdir: Path) -> sky.Task:
    """Translate a JobManifest into a SkyPilot Task (no cloud calls; unit-tested)."""
    import sky

    task = sky.Task(
        name=cluster_name_for(manifest.job_id),
        setup=build_setup_script(),
        run=build_run_script(manifest),
        envs={
            "LAB_RUN_ID": manifest.job_id,
            "LAB_RUN_DIR": REMOTE_RUN_DIR,
            "LAB_SEED": str(manifest.run.seed),
        },
        workdir=str(workdir),
    )
    # NOTE: GPU *type* isn't modelled in ResourceRequest yet (only a count), and a count alone
    # isn't a valid SkyPilot accelerator spec — so P0 requests CPU/memory and lets SkyPilot
    # cost-optimise the offer. Typed GPU selection is a P1 follow-up.
    task.set_resources(
        sky.Resources(
            cloud=sky.Vast(),
            cpus=manifest.resources.cpus,
            memory=manifest.resources.memory,
        )
    )
    return task


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SkyPilotBackend:
    name = "skypilot"

    def __init__(self, home: Path, repo: Path | None = None) -> None:
        self.store = JobStore(Path(home))
        self.repo = Path(repo) if repo else Path.cwd()

    def submit(self, manifest: JobManifest) -> str:
        job_dir = self.store.job_dir(manifest.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        # Supervisor's stdout/stderr (incl. sky.launch streamed logs) -> the job log file (FR-D1).
        logf = self.store.logs_path(manifest.job_id).open("w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "lab.sky_runner", str(job_dir)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(self.repo),
            start_new_session=True,
        )
        self.store.write_runtime(
            manifest.job_id, runner_pid=proc.pid, cluster=cluster_name_for(manifest.job_id)
        )
        return manifest.job_id

    def status(self, job_id: str) -> JobState:
        m = self.store.read_manifest(job_id)
        if m.status not in _TERMINAL:
            rt = self.store.read_runtime(job_id)
            if rt.get("runner_pid") and not _alive(rt["runner_pid"]):
                return self.store.update_manifest(
                    job_id,
                    status=JobState.failed,
                    ended_at=now(),
                    end_reason="supervisor exited without recording status",
                ).status
        return m.status

    def tail_logs(
        self, job_id: str, tail: int | None = None, follow: bool = False
    ) -> Iterable[str]:
        p = self.store.logs_path(job_id)
        if not p.exists():
            return []
        lines = p.read_text(errors="replace").splitlines()
        return lines[-tail:] if tail else lines

    def cancel(self, job_id: str) -> JobState:
        m = self.store.read_manifest(job_id)
        if m.status in _TERMINAL:
            return m.status
        self.store.update_manifest(
            job_id, status=JobState.cancelled, ended_at=now(), end_reason="cancelled by user"
        )
        rt = self.store.read_runtime(job_id)
        if rt.get("runner_pid"):
            try:
                os.kill(rt["runner_pid"], signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        cluster = rt.get("cluster") or cluster_name_for(job_id)
        try:
            import sky

            sky.cancel(cluster, all=True)
        except Exception:  # noqa: BLE001 - best-effort; teardown below is what matters
            pass
        try:
            import sky

            sky.down(cluster)
        except Exception:  # noqa: BLE001
            pass
        return JobState.cancelled

    def collect_artifacts(self, job_id: str, dest: str) -> list[ArtifactRecord]:
        # The supervisor rsyncs the remote run dir into output/ before teardown; read it locally.
        out = self.store.output_dir(job_id)
        records: list[ArtifactRecord] = []
        if out.exists():
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(out).as_posix()
                    records.append(
                        ArtifactRecord(
                            name=rel,
                            type=infer_artifact_type(rel),  # type: ignore[arg-type]
                            path=str(f),
                            sha256=sha256_file(f),
                            bytes=f.stat().st_size,
                        )
                    )
        self.store.update_manifest(job_id, artifacts=records)
        return records

    def read_metrics(
        self, job_id: str, names: Iterable[str] | None = None, since_step: int | None = None
    ) -> list[dict[str, Any]]:
        return read_points(
            self.store.output_dir(job_id) / METRICS_FILE, names=names, since_step=since_step
        )
