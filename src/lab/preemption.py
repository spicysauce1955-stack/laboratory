"""Pure terminal-state classifier for the unmanaged spot path (no cloud calls).

Inferred preemption is *strictly* the lowest-precedence outcome: an explicit timeout or a
user-initiated cancel/early-kill always wins, so a deliberately-killed run is never auto-resubmitted.
A genuine terminal status reported by the cloud (succeeded/failed) is trusted over inference.
"""
from __future__ import annotations

from typing import Any

from lab.models import JobState


def gcp_terminal_state(instances: list[dict[str, Any]]) -> JobState | None:
    """GCE's own account of why a cluster's instances stopped, or ``None`` when it has none.

    On GCP the lowest-precedence *inference* (spot + box vanished + no authoritative terminal) can
    be replaced by an authoritative answer: a preempted Spot VM goes ``TERMINATED`` with
    ``scheduling.preemptible = true``. Reading it closes GCP-PREEMPT-1, where a genuinely *failed*
    spot job whose box happened to disappear classified as ``preempted``, the scheduler
    auto-resubmitted it, and the same failure was paid for twice.

    Abstains (``None``) rather than guessing when there is nothing terminal to read — no instance
    left, or one still running — so the caller keeps today's inference. Pure; the listing that
    feeds it is :func:`lab.backends.skypilot.gcp_preemption_state`.
    """
    stopped = [i for i in instances if str(i.get("status") or "").upper() == "TERMINATED"]
    if not stopped:
        return None
    if any(i.get("preemptible") for i in stopped):
        return JobState.preempted
    return JobState.failed  # GCE stopped it, and it was not a preemption


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
