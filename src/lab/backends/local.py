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
from typing import Any

from lab._util import infer_artifact_type, now, pid_alive, process_start_time
from lab.manifest import sha256_file
from lab.metrics import METRICS_FILE, read_points
from lab.models import ArtifactRecord, JobManifest, JobState
from lab.runner import group_alive, group_is_ours, stop_group
from lab.store import JobStore

_TERMINAL = {
    JobState.succeeded, JobState.failed, JobState.cancelled, JobState.timed_out, JobState.preempted
}




_TAIL_BLOCK = 64 * 1024
# A log with almost no '\n' (a progress bar redrawn with '\r') would otherwise be read whole while
# looking for line breaks; past this, the tail is taken from what was read.
_TAIL_MAX_BYTES = 16 * 1024 * 1024


def tail_lines(path: Path, n: int | None) -> list[str]:
    """The last ``n`` lines of ``path``: all of them for ``None``, none for ``n <= 0``.

    Reads backwards from the end in blocks, so showing 100 lines of a multi-GB log costs about 100
    lines of I/O, and decodes with ``errors="replace"``: an experiment's stdout is not promised to
    be UTF-8, and one stray byte used to crash ``lab logs``. A block boundary can split a
    multi-byte character, but only inside the first, partial line, which is always dropped.
    """
    if n is not None and n <= 0:
        return []
    with path.open("rb") as f:
        if n is None:
            return f.read().decode("utf-8", errors="replace").splitlines()
        pos = f.seek(0, os.SEEK_END)
        buf = b""
        # n+1 line breaks guarantee n whole lines after the first (possibly partial) one.
        while pos > 0 and buf.count(b"\n") <= n and len(buf) < _TAIL_MAX_BYTES:
            step = min(_TAIL_BLOCK, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
    lines = buf.decode("utf-8", errors="replace").splitlines()
    if pos > 0:
        lines = lines[1:]  # began mid-line (or mid-character)
    return lines[-n:]


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
        self.store.write_runtime(
            manifest.job_id, runner_pid=proc.pid, runner_start_time=process_start_time(proc.pid)
        )
        return manifest.job_id

    def status(self, job_id: str) -> JobState:
        m = self.store.read_manifest(job_id)
        if m.status not in _TERMINAL:
            rt = self.store.read_runtime(job_id)
            if rt.get("runner_pid") and not pid_alive(
                rt["runner_pid"], start_time=rt.get("runner_start_time")
            ):
                return self._finalize_dead_runner(job_id, rt)
        return m.status

    def _finalize_dead_runner(self, job_id: str, rt: dict[str, Any]) -> JobState:
        """Record ``failed`` for a job whose runner died without a verdict, and stop the
        experiment it was supervising if that is provably still running.

        The experiment runs in its own session (so it can be killed as a group), which also means
        it survives its runner: a SIGKILLed runner left it running with nobody enforcing its
        wall-clock cap, while ``cancel`` refused it as already terminal (stress run 2026-11-19).
        So the recorded group is stopped (:func:`~lab.runner.stop_group`: SIGTERM, a grace, then
        SIGKILL) — but only when it is provably still the job's: the leader's start time must have
        been recorded, and :func:`~lab.runner.group_is_ours` must accept it. A runtime file from
        before start times were recorded is never signalled from here; a status poll is not the
        place to guess.

        **The verdict stays ``failed`` even when the experiment had already exited.** A runner
        that dies between its experiment exiting and recording the outcome leaves outputs that may
        well be complete, but the exit code is gone and nothing else proves success. Calling it
        ``succeeded`` would be the one wrong answer that is never revisited; ``failed`` costs
        little, because ``sweep-aggregate`` still reads a failed shard's rows (as partial, stamped
        ``_shard_status``) and ``sweep-retry`` only re-buys seeds that are actually missing. The
        ``end_reason`` says which case this was, and that the outputs were kept.
        """
        base = "runner exited without recording status"
        pgid = rt.get("command_pgid")
        leader_start = rt.get("command_start_time")
        stop_pgid: int | None = None
        if not pgid:
            reason = base  # died before launching its command: nothing ran
        elif not group_alive(pgid):
            reason = (
                f"{base}; the experiment had already exited, but its exit code was never "
                "recorded — not marked succeeded (outputs kept)"
            )
        elif leader_start is None or not group_is_ours(pgid, leader_start):
            reason = (
                f"{base}; its experiment may still be running (process group {pgid}) but could "
                "not be proven to be this job's — not signalled"
            )
        else:
            stop_pgid = int(pgid)
            reason = f"{base}; its experiment was still running and has been stopped"
        final = self.store.update_manifest(
            job_id,
            only_if_active=True,  # the runner may have recorded its verdict, then exited
            status=JobState.failed,
            ended_at=now(),
            end_reason=reason,
        )
        if stop_pgid is not None:
            # After the verdict, like `cancel`: whatever the group writes on its way down lands
            # under a terminal manifest. A no-op if the runner did record a verdict first — its
            # group has exited by then, and `stop_group` re-checks the identity itself.
            stop_group(stop_pgid, leader_start_time=leader_start)
        return final.status

    def tail_logs(
        self, job_id: str, tail: int | None = None, follow: bool = False
    ) -> Iterable[str]:
        p = self.store.logs_path(job_id)
        if not p.exists():
            return []
        return tail_lines(p, tail)

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
            # SIGTERM, a grace, then SIGKILL — an experiment that ignores SIGTERM must not outlive
            # the `cancelled` just recorded. Skipped if the group id now belongs to someone else.
            stop_group(rt["command_pgid"], leader_start_time=rt.get("command_start_time"))
        runner_pid = rt.get("runner_pid")
        if runner_pid and pid_alive(runner_pid, start_time=rt.get("runner_start_time")):
            # Only the runner we launched: a bare PID may have been recycled by now, and this
            # used to SIGTERM whatever held the number.
            _kill(runner_pid, group=False)
        return JobState.cancelled

    def collect_artifacts(self, job_id: str, dest: str) -> list[ArtifactRecord]:
        out = self.store.output_dir(job_id)
        records: list[ArtifactRecord] = []
        if out.exists():
            for f in sorted(out.rglob("*")):
                if f.is_file() and not f.name.startswith("."):  # skip sentinels/hidden files
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
