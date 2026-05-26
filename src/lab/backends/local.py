"""Local subprocess backend — the NFR-4 fallback and the P0 dev loop.

``submit`` launches a detached runner (``python -m lab.runner``) that supervises the experiment,
so the job survives the CLI/MCP process exiting (NFR-2). State/logs are read from the run dir;
``cancel`` flags the manifest then kills the process group; artifacts are already local so
``collect_artifacts`` just records them in place (FR-E2).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from lab._util import infer_artifact_type, now
from lab.manifest import sha256_file
from lab.models import ArtifactRecord, JobManifest, JobState
from lab.store import JobStore

_TERMINAL = {JobState.succeeded, JobState.failed, JobState.cancelled, JobState.timed_out}


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


def _kill(pid: int, *, group: bool) -> None:
    try:
        (os.killpg if group else os.kill)(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


class LocalBackend:
    name = "local"

    def __init__(self, home: Path, repo: Path | None = None) -> None:
        self.store = JobStore(Path(home))
        self.repo = Path(repo) if repo else Path.cwd()

    def submit(self, manifest: JobManifest) -> str:
        job_dir = self.store.job_dir(manifest.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [sys.executable, "-m", "lab.runner", str(job_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(self.repo),
            start_new_session=True,
        )
        self.store.write_runtime(manifest.job_id, runner_pid=proc.pid)
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
                    end_reason="runner exited without recording status",
                ).status
        return m.status

    def tail_logs(
        self, job_id: str, tail: int | None = None, follow: bool = False
    ) -> Iterable[str]:
        p = self.store.logs_path(job_id)
        if not p.exists():
            return []
        lines = p.read_text().splitlines()
        return lines[-tail:] if tail else lines

    def cancel(self, job_id: str) -> JobState:
        m = self.store.read_manifest(job_id)
        if m.status in _TERMINAL:
            return m.status
        # Flag first so the runner preserves "cancelled" when its command dies.
        self.store.update_manifest(
            job_id, status=JobState.cancelled, ended_at=now(), end_reason="cancelled by user"
        )
        rt = self.store.read_runtime(job_id)
        if rt.get("command_pgid"):
            _kill(rt["command_pgid"], group=True)
        if rt.get("runner_pid"):
            _kill(rt["runner_pid"], group=False)
        return JobState.cancelled

    def collect_artifacts(self, job_id: str, dest: str) -> list[ArtifactRecord]:
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
