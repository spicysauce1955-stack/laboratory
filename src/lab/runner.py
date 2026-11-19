"""Job runner — supervises one experiment subprocess (launched detached by LocalBackend).

Responsibilities (FR-C4, FR-I1, FR-A4): inject the lab env contract, run the entrypoint with a
wall-clock timeout, capture stdout+stderr, and record the terminal state in the manifest.

Entry point:  python -m lab.runner <job_dir>
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from lab._util import (
    duration_seconds,
    now,
    parse_duration,
    process_start_time,
    timeout_reason,
)
from lab.models import CostInfo, JobState
from lab.store import JobStore

# How long an experiment gets between SIGTERM and SIGKILL, on a timeout and on a cancel. A
# ceiling, not a sleep: the wait ends as soon as the whole process group is gone.
KILL_GRACE_S = 10.0
_GROUP_POLL_S = 0.1


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def group_alive(pgid: int) -> bool:
    """Does any *running* process still belong to process group ``pgid``?

    ``killpg(pgid, 0)`` alone cannot answer this: it succeeds for a zombie, and the runner's own
    shell is exactly that between dying of SIGTERM and being reaped — so the grace period would
    always run to its end. On Linux the members are read off ``/proc`` (field 5 is the pgrp;
    state ``Z``/``X`` is a process that has already exited). Anywhere ``/proc`` cannot be read the
    answer degrades to the ``killpg`` one, which errs towards "alive": the cost is waiting out the
    grace, never skipping the SIGKILL.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        pids = [name for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return True
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            state, _ppid, pgrp = stat[stat.rindex(")") + 1 :].split()[:3]
        except (OSError, ValueError):
            continue  # exited mid-scan, or unreadable: not evidence either way
        if int(pgrp) == pgid and state not in ("Z", "X"):
            return True
    return False


def group_is_ours(pgid: int, leader_start_time: int | None) -> bool:
    """May ``pgid`` still be signalled as the job's command group?

    The group id is the PID of the shell that led it. While any member lives, Linux will not hand
    that number to a new process, so the group cannot have been recycled. Once it is empty the
    number is free, and a later process that takes it and calls ``setsid`` leads an unrelated group
    of the same id. So: if a process now holds the leader's PID with a *different* start time, the
    group is not ours. If nobody holds it, any remaining members can only be ours. With no recorded
    start time (a runtime file from before it was recorded) the old behaviour stands.
    """
    if leader_start_time is None:
        return True
    current = process_start_time(pgid)
    return current is None or current == leader_start_time


def stop_group(pgid: int, *, leader_start_time: int | None = None) -> bool:
    """SIGTERM process group ``pgid``, wait up to :data:`KILL_GRACE_S` for it to empty, then
    SIGKILL whatever is left. Returns ``False``, having signalled nothing, when the group fails
    :func:`group_is_ours`.

    The whole *group*, not the process we hold: the entrypoint runs under ``sh -c``, and a compound
    command keeps the shell as the parent. The shell dies of SIGTERM at once; an experiment that
    ignores SIGTERM does not, and escalating only on the shell's exit (the old ``proc.kill()``)
    left it running after ``timed_out``/``cancelled`` was already recorded.
    """
    if not group_is_ours(pgid, leader_start_time):
        return False
    _killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + KILL_GRACE_S
    while group_alive(pgid):
        if time.monotonic() >= deadline:
            # Still the same group: it has had a member throughout, so its id was never free.
            _killpg(pgid, signal.SIGKILL)
            break
        time.sleep(_GROUP_POLL_S)
    return True


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
        # The leader's start time is what lets `LocalBackend.cancel` (another process, possibly
        # much later) tell this group from one that reused its id (`group_is_ours`).
        pgid = os.getpgid(proc.pid)
        store.write_runtime(
            job_id, command_pgid=pgid, command_start_time=process_start_time(proc.pid)
        )
        exit_code: int | None
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # The shell is our unreaped child until `proc.wait()` below, so its id (the group's)
            # cannot be recycled while this runs; no identity check is needed here.
            stop_group(pgid)
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
        status, reason = JobState.timed_out, timeout_reason(wall)
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
