"""Successful read-only polls are rate-limited on the write path.

Incident (campaign of 2026-09-03..07): runaway shell loops made 98,654 *successful* `lab status`
calls at a flat 1,050/hour. Every one wrote a full open/close pair, the ledger hit the 50 MB byte
cap, and `compact()` — age-gated at 14 days — could not touch a single one of those fresh
successes. The only lever left was deleting a whole day file, and it took day one of the live
campaign, mid-investigation. These tests pin the write-path half of the fix; the retention half
lives in `test_events_retention.py`.

The clock is injected (`record.now` is monkeypatched), never slept on: a real-clock test of a
60-second window would either take a minute or decay into flakiness.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab import events
from lab.events import store

record_module = importlib.import_module("lab.events.record")

T0 = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)


class Clock:
    """A hand-cranked clock standing in for `lab._util.now` inside the writer."""

    def __init__(self, start: datetime) -> None:
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock(T0)
    monkeypatch.setattr(record_module, "now", c)
    return c


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    monkeypatch.delenv("LAB_EVENTS_READ_MIN_INTERVAL_S", raising=False)
    monkeypatch.setenv("LAB_SESSION_ID", "sess_test")
    monkeypatch.setattr(record_module, "_seq", 0)
    monkeypatch.setattr(record_module, "_session", None)
    monkeypatch.setattr(record_module, "_pruned", False)
    return tmp_path / "events"


def _records() -> list[dict]:
    return list(store.iter_records(store.day_files()))


def _poll(action: str = "status") -> None:
    with events.record("cli", action, {"job_id": "j-1"}):
        pass


def test_repeated_successful_polls_inside_the_window_write_exactly_one_pair(clock: Clock) -> None:
    for _ in range(20):
        clock.advance(3)  # 20 polls over a minute, the incident's cadence
        _poll()
    records = _records()
    assert [r["phase"] for r in records] == ["open", "close"]
    assert records[0]["action"] == "status"
    assert records[1]["outcome"] == "ok"


def test_the_first_poll_after_the_window_is_recorded_again(clock: Clock) -> None:
    _poll()
    clock.advance(59)
    _poll()
    assert len(_records()) == 2, "a poll inside the window must not be recorded"
    clock.advance(2)  # now 61s past the recorded one
    _poll()
    assert [r["phase"] for r in _records()] == ["open", "close", "open", "close"]


def test_a_failing_read_only_call_is_never_rate_limited(clock: Clock) -> None:
    _poll()  # takes the window
    clock.advance(1)
    with pytest.raises(RuntimeError):
        with events.record("cli", "status", {"job_id": "j-2"}):
            events.note("store.read", job_id="j-2")
            raise RuntimeError("job store unreadable")
    records = _records()
    assert [r["phase"] for r in records] == ["open", "close", "open", "close"]
    failed_open, failed_close = records[2], records[3]
    assert failed_open["action"] == "status"
    assert failed_open["params"]["job_id"] == "j-2"
    assert failed_close["outcome"] == "crash"
    assert failed_close["error"]["message"] == "job store unreadable"
    assert [n["k"] for n in failed_close["trace"]] == ["store.read"]


def test_a_failure_does_not_consume_the_window_and_hide_the_next_success(clock: Clock) -> None:
    """A failure must neither stamp nor refresh the last-recorded-success time. If it did, the
    very next success would be suppressed by a window it never opened."""
    with pytest.raises(RuntimeError):
        with events.record("cli", "status", {}):
            raise RuntimeError("boom")
    clock.advance(1)
    _poll()
    phases = [(r["phase"], r.get("outcome")) for r in _records()]
    assert phases == [("open", None), ("close", "crash"), ("open", None), ("close", "ok")]


def test_a_mutating_actions_open_is_written_at_call_start(clock: Clock) -> None:
    """The dangling-open property (CLAUDE.md, `compact()`'s docstring): a hard-killed mutating
    call must still leave a visible `open` with no close. Rate limiting must not defer these."""
    record_module.begin("cli", "submit", {"argv": ["submit"]})  # killed before finish()
    records = _records()
    assert [r["phase"] for r in records] == ["open"]
    assert records[0]["action"] == "submit"


def test_a_read_only_call_that_dies_before_close_writes_nothing(clock: Clock) -> None:
    """The accepted trade: a SIGKILLed `lab status` leaves no trace at all, because its open is
    buffered until the outcome — which the rate-limit decision needs — is known."""
    record_module.begin("cli", "status", {"argv": ["status", "j-1"]})
    assert _records() == []


def test_different_read_only_actions_do_not_share_one_window(clock: Clock) -> None:
    _poll("status")
    _poll("list")
    _poll("metrics")
    assert [r["action"] for r in _records() if r["phase"] == "open"] == [
        "status", "list", "metrics",
    ]


def test_both_queue_list_spellings_are_rate_limited(clock: Clock) -> None:
    """The ledger holds both `queue list` (CLI leaf-action naming) and `queue_list` (the MCP tool
    name) for the same read. Both are cheap reads and both must be limited."""
    _poll("queue list")
    clock.advance(1)
    _poll("queue_list")
    clock.advance(1)
    _poll("queue show")
    clock.advance(1)
    _poll("queue_show")
    assert [r["action"] for r in _records() if r["phase"] == "open"] == ["queue list", "queue show"]


def test_a_mutating_action_is_never_rate_limited(clock: Clock) -> None:
    for _ in range(3):
        with events.record("cli", "submit", {}):
            pass
        clock.advance(1)
    assert len([r for r in _records() if r["phase"] == "open"]) == 3


def test_garbage_min_interval_falls_back_to_the_default(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LAB_EVENTS_READ_MIN_INTERVAL_S", "banana")
    _poll()
    clock.advance(5)
    _poll()
    assert len(_records()) == 2  # the 60s default still applied
    clock.advance(56)
    _poll()
    assert len(_records()) == 4


def test_the_min_interval_is_configurable(clock: Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EVENTS_READ_MIN_INTERVAL_S", "5")
    _poll()
    clock.advance(2)
    _poll()
    assert len(_records()) == 2
    clock.advance(4)
    _poll()
    assert len(_records()) == 4


def test_a_zero_min_interval_disables_the_limit(clock: Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EVENTS_READ_MIN_INTERVAL_S", "0")
    _poll()
    _poll()
    assert len(_records()) == 4


def test_a_corrupt_stamp_reads_as_no_recent_success(clock: Clock, _events_dir: Path) -> None:
    """An unreadable stamp must never suppress a record (and never raise): the fallback is to
    log, which is the cheap-and-correct direction."""
    _poll()
    stamps = list((_events_dir / store.READ_STAMP_DIR).glob("*"))
    assert len(stamps) == 1, "the poll should have left exactly one per-(action, project) stamp"
    stamps[0].write_text("not a timestamp", encoding="utf-8")
    clock.advance(1)
    _poll()
    assert len(_records()) == 4


def test_an_unwritable_stamp_dir_still_records(
    clock: Clock, _events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> bool:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(store, "claim_read_slot", _boom)
    _poll()
    clock.advance(1)
    _poll()
    assert len(_records()) == 4  # a broken limiter degrades to recording everything


def test_the_stamp_files_are_not_mistaken_for_day_files(clock: Clock) -> None:
    _poll()
    assert [p.name for p in store.day_files()] == ["2026-09-03.jsonl"]


def test_rate_limiting_is_off_when_the_ledger_is_disabled(
    clock: Clock, _events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LAB_EVENTS", "0")
    _poll()
    _poll()
    assert _records() == []
    assert not _events_dir.exists()  # not even a stamp directory


POLLER = """
from lab import events
with events.record("cli", "status", {"job_id": "j-1"}):
    pass
"""


def test_concurrent_pollers_write_exactly_one_pair(tmp_path: Path) -> None:
    """The real load is ~100k separate short-lived processes, so the check-and-stamp has to be
    atomic across processes, not merely within one. Sixteen concurrent `lab status`-shaped calls
    that all succeed inside the window may leave exactly one pair between them."""
    import os
    import subprocess
    import sys

    events_dir = tmp_path / "concurrent-events"
    env = {**os.environ, "LAB_EVENTS_DIR": str(events_dir)}
    procs = [subprocess.Popen([sys.executable, "-c", POLLER], env=env) for _ in range(16)]
    for p in procs:
        assert p.wait() == 0
    lines = [
        line
        for path in sorted(events_dir.glob("????-??-??.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2, f"expected one open/close pair, got {len(lines)} lines"
