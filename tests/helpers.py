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
TERMINAL = {JobState.succeeded, JobState.failed, JobState.cancelled, JobState.timed_out, JobState.preempted}


def make_manifest(
    job_id: str,
    command: str,
    *,
    seed: int = 0,
    timeout: str | None = None,
    accelerators: str | None = None,
) -> JobManifest:
    return JobManifest(
        job_id=job_id,
        created_at=now(),
        submitted_by="agent",
        code=CodeRef(git_commit="0" * 40, git_dirty=False),
        env=EnvInfo(uv_lock_sha256="test", python_version="3.12"),
        run=RunSpec(entrypoint_command=command, seed=seed),
        resources=ResourceRequest(timeout=timeout, accelerators=accelerators),
        backend=BackendInfo(provisioner="local"),
        status=JobState.queued,
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
