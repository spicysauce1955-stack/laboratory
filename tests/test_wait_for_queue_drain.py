"""`wait_for_queue_drain` — the safety gate a scheduler cutover waits on before pausing the
queue (docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md). A registration's
`state` already reflects its mirrored job's real terminality: `Scheduler._sync` keeps them in
lock-step while the queue is unpaused, so checking `state` alone — no separate job-status lookup
— is sufficient."""

import time
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


class _FlakyOnceStore:
    """Wraps a real store; raises on the first list_entries() call, then delegates."""

    def __init__(self, inner: LocalQueueStore) -> None:
        self._inner = inner
        self._calls = 0

    def list_entries(self) -> list[Registration]:
        self._calls += 1
        if self._calls == 1:
            raise ConnectionError("simulated transient R2 read failure")
        return self._inner.list_entries()


def test_tolerates_one_flaky_read(tmp_path: Path) -> None:
    # A real store (R2QueueStore) has no internal retry -- a single transient network error
    # during an up-to-30-minute drain wait must not crash the whole cutover before the queue is
    # even paused.
    inner = LocalQueueStore(tmp_path / "queue")
    q = _FlakyOnceStore(inner)

    blocking = wait_for_queue_drain(q, interval=0.01, timeout=1.0)

    assert blocking == []


class _AlwaysFailsStore:
    def list_entries(self) -> list[Registration]:
        raise ConnectionError("simulated persistent R2 outage")


def test_raises_at_timeout_if_no_read_ever_succeeded(tmp_path: Path) -> None:
    # A queue that was NEVER successfully read must not silently report "drained" -- that would
    # be reporting a clean drain with zero evidence, right before deploy.sh pauses production.
    q = _AlwaysFailsStore()

    import pytest

    with pytest.raises(ConnectionError):
        wait_for_queue_drain(q, interval=0.01, timeout=0.05)


class _UnreadableEntryStore:
    """A listing that skipped one entry it could not read (what R2QueueStore does)."""

    def __init__(self, inner: LocalQueueStore, unreadable: set[str], clears_after: int) -> None:
        self._inner = inner
        self._unreadable = unreadable
        self._left = clears_after
        self.last_unreadable: frozenset[str] = frozenset()

    def list_entries(self) -> list[Registration]:
        self.last_unreadable = frozenset(self._unreadable) if self._left > 0 else frozenset()
        self._left -= 1
        return self._inner.list_entries()


def test_an_unreadable_entry_is_not_a_drained_one(tmp_path: Path) -> None:
    """The drain gate is what deploy.sh pauses production on. An entry the listing skipped may be
    `launched` -- a clean drain must not be reported until it can actually be read."""
    import pytest

    from lab.scheduler.queue import UnreadableEntriesError

    q = _UnreadableEntryStore(LocalQueueStore(tmp_path / "queue"), {"reg-x"}, clears_after=10**6)
    with pytest.raises(UnreadableEntriesError) as exc:
        wait_for_queue_drain(q, interval=0.01, timeout=0.05)
    assert "reg-x" in str(exc.value)


def test_a_briefly_unreadable_entry_only_delays_the_drain(tmp_path: Path) -> None:
    q = _UnreadableEntryStore(LocalQueueStore(tmp_path / "queue"), {"reg-x"}, clears_after=2)
    assert wait_for_queue_drain(q, interval=0.01, timeout=2.0) == []


def test_does_not_overrun_timeout_when_interval_is_larger(tmp_path: Path) -> None:
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("r1", RegState.launched))

    start = time.monotonic()
    blocking = wait_for_queue_drain(q, interval=100.0, timeout=0.1)
    elapsed = time.monotonic() - start

    assert {r.reg_id for r in blocking} == {"r1"}
    assert elapsed < 2.0  # would be ~100s without a deadline-clamped sleep
