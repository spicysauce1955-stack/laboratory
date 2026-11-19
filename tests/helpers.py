"""Shared test helpers."""

from __future__ import annotations

import sys
import time

from lab._util import now
from lab.models import (
    BackendInfo,
    CodeRef,
    EnvInfo,
    JobManifest,
    JobState,
    ResourceRequest,
    RunSpec,
)

PYTHON = sys.executable
TERMINAL = {
    JobState.succeeded, JobState.failed, JobState.cancelled, JobState.timed_out, JobState.preempted
}


def make_manifest(
    job_id: str,
    command: str,
    *,
    seed: int = 0,
    timeout: str | None = None,
    accelerators: str | None = None,
    resources: ResourceRequest | None = None,
) -> JobManifest:
    """A minimal manifest. Pass ``resources`` for a full spec (region/zone/disk/spot); the
    ``timeout``/``accelerators`` shorthands still apply on top of it."""
    res = resources or ResourceRequest()
    update = {}
    if timeout is not None:
        update["timeout"] = timeout
    if accelerators is not None:
        update["accelerators"] = accelerators
    return JobManifest(
        job_id=job_id,
        created_at=now(),
        submitted_by="agent",
        code=CodeRef(git_commit="0" * 40, git_dirty=False),
        env=EnvInfo(uv_lock_sha256="test", python_version="3.12"),
        run=RunSpec(entrypoint_command=command, seed=seed),
        resources=res.model_copy(update=update) if update else res,
        backend=BackendInfo(provisioner="local"),
        status=JobState.queued,
    )


def pin_in_flight(store, job_id: str, status: JobState = JobState.running) -> None:
    """Make a (settled) local job read as genuinely in flight: a non-terminal manifest AND a live
    runner. Status reads refresh through the backend, which finalizes a non-terminal job whose
    runner is gone — so a bare status rewrite now reads as a dead supervisor, not in-flight.
    Our own pid, with its real start time, is a runner that is provably alive."""
    import os

    from lab._util import process_start_time

    store.write_manifest(
        store.read_manifest(job_id).model_copy(
            update={"status": status, "ended_at": None, "end_reason": None}
        )
    )
    store.write_runtime(
        job_id, runner_pid=os.getpid(), runner_start_time=process_start_time(os.getpid())
    )


def wait_terminal(backend, job_id: str, timeout: float = 30.0) -> JobState:
    """Poll status until the job reaches a terminal state (or fail)."""
    deadline = time.time() + timeout
    state = backend.status(job_id)
    while time.time() < deadline:
        state = backend.status(job_id)
        if state in TERMINAL:
            return state
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not reach terminal state (last={state})")
