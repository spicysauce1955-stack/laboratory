"""Pure terminal-state classifier for the unmanaged spot path (no cloud calls).

Inferred preemption is *strictly* the lowest-precedence outcome: an explicit timeout or a
user-initiated cancel/early-kill always wins, so a deliberately-killed run is never auto-resubmitted.
A genuine terminal status reported by the cloud (succeeded/failed) is trusted over inference.
"""
from __future__ import annotations

from lab.models import JobState


def classify_terminal(
    *,
    sky_state: JobState,
    timed_out: bool,
    cancel_requested: bool,
    use_spot: bool,
    cluster_gone: bool,
    reached_terminal: bool,
) -> JobState:
    # An explicit, authoritative outcome always wins over inference.
    if timed_out:
        return JobState.timed_out
    if cancel_requested:
        return JobState.cancelled
    # The cloud actually reported a terminal status -> trust it (success or genuine failure).
    if reached_terminal and sky_state in (JobState.succeeded, JobState.failed):
        return sky_state
    # No authoritative terminal, the box vanished, and it was spot -> inferred preemption.
    if use_spot and cluster_gone:
        return JobState.preempted
    return JobState.failed
