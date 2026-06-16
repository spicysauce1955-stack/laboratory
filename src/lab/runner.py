"""Job runner — supervises one experiment subprocess (launched detached by LocalBackend).

Responsibilities (FR-C4, FR-I1, FR-A4): inject the lab env contract, run the entrypoint with a
wall-clock timeout, capture stdout+stderr, and record the terminal state in the manifest.

Entry point:  python -m lab.runner <job_dir>
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from lab._util import duration_seconds, now, parse_duration
from lab.models import CostInfo, JobState
from lab.store import JobStore


def _terminate_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def run_job(job_dir: Path) -> int:
    job_dir = Path(job_dir)
    store = JobStore(job_dir.parent)
    job_id = job_dir.name
    manifest = store.read_manifest(job_id)

    output = store.output_dir(job_id)
    output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LAB_RUN_ID"] = job_id
    env["LAB_RUN_DIR"] = str(output)
    env["LAB_SEED"] = str(manifest.run.seed)

    timeout = parse_duration(manifest.resources.timeout)
    started = now()
    store.update_manifest(job_id, status=JobState.running, started_at=started)

    timed_out = False
    with store.logs_path(job_id).open("w") as logf:
        proc = subprocess.Popen(
            manifest.run.entrypoint_command,
            shell=True,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # own process group → killable as a unit
        )
        store.write_runtime(job_id, command_pgid=os.getpgid(proc.pid))
        exit_code: int | None
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_group(proc.pid)
            try:
                exit_code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                exit_code = proc.wait()

    ended = now()
    cost = CostInfo(
        duration_seconds=duration_seconds(started, ended),
        hourly_usd=0.0,
        estimated_usd=0.0,
        actual_usd=0.0,
    )

    # Respect a concurrent cancel: the backend sets status=cancelled before killing the group.
    if store.read_manifest(job_id).status == JobState.cancelled:
        store.update_manifest(job_id, ended_at=ended, exit_code=exit_code, cost=cost)
        return exit_code if exit_code is not None else 1

    if timed_out:
        wall = int(timeout) if timeout else 0
        status, reason = JobState.timed_out, f"timed out after {wall}s wall-clock cap"
    elif exit_code == 0:
        status, reason = JobState.succeeded, "completed"
    else:
        status, reason = JobState.failed, f"exit code {exit_code}"

    # final_metrics is snapshotted centrally by the store on the succeeded transition (FR-B4).
    store.update_manifest(
        job_id, status=status, ended_at=ended, exit_code=exit_code, end_reason=reason, cost=cost
    )
    return exit_code if exit_code is not None else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(run_job(Path(sys.argv[1])))
