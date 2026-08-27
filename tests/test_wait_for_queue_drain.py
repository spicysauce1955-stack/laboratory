"""`wait_for_queue_drain` — the safety gate a scheduler cutover waits on before pausing the
queue (docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md). A registration's
`state` already reflects its mirrored job's real terminality: `Scheduler._sync` keeps them in
lock-step while the queue is unpaused, so checking `state` alone — no separate job-status lookup
— is sufficient."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import PYTHON

from lab.models import CodeRef, JobSpec
from lab.scheduler.models import Guardrails, Registration, RegState, Triggers
from lab.scheduler.queue import LocalQueueStore, wait_for_queue_drain

T0 = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)


def _reg(reg_id: str, state: RegState) -> Registration:
    return Registration(
        reg_id=reg_id,
        created_at=T0,
        spec=JobSpec(command=f"{PYTHON} -c 'print(1)'"),
        triggers=Triggers(),
        guardrails=Guardrails(expires_at=T0 + timedelta(days=1)),
        bundle_key=f"bundles/{reg_id}.tar.gz",
        code=CodeRef(git_commit="0" * 40),
        state=state,
    )


def test_drains_immediately_when_nothing_is_in_flight(tmp_path: Path) -> None:
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("r1", RegState.pending))
    q.put_entry(_reg("r2", RegState.succeeded))

    blocking = wait_for_queue_drain(q, interval=0.01, timeout=1.0)

    assert blocking == []


def test_blocks_on_launching_and_launched(tmp_path: Path) -> None:
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("r1", RegState.launching))
    q.put_entry(_reg("r2", RegState.launched))

    blocking = wait_for_queue_drain(q, interval=0.01, timeout=0.05)

    assert {r.reg_id for r in blocking} == {"r1", "r2"}


def test_returns_empty_once_a_blocking_entry_transitions_to_terminal(tmp_path: Path) -> None:
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("r1", RegState.launched))

    # Flip it to terminal shortly after the first poll, on a real background thread —
    # exercises the actual polling loop, not a mocked clock.
    import threading

    def _finish():
        import time

        time.sleep(0.03)
        q.put_entry(_reg("r1", RegState.succeeded))

    threading.Thread(target=_finish, daemon=True).start()

    blocking = wait_for_queue_drain(q, interval=0.01, timeout=2.0)

    assert blocking == []


def test_pending_with_a_future_trigger_never_blocks(tmp_path: Path) -> None:
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("r1", RegState.pending))

    blocking = wait_for_queue_drain(q, interval=0.01, timeout=0.05)

    assert blocking == []


def test_no_timeout_means_no_timeout_arg_is_required() -> None:
    import inspect

    sig = inspect.signature(wait_for_queue_drain)
    assert sig.parameters["timeout"].default is None
