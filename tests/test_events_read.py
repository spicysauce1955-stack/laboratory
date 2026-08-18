from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab.events import store
from lab.events.read import fold, read, row

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def _open(id_: str, **over) -> dict:
    base = {"id": id_, "ts": NOW.isoformat(), "phase": "open", "session": "s", "seq": 0,
            "surface": "cli", "action": "submit", "params": {}, "project": {"name": "lab"},
            "lab_version": "0.5.1"}
    return {**base, **over}


def _close(id_: str, **over) -> dict:
    base = {"id": id_, "ts": (NOW + timedelta(seconds=2)).isoformat(), "phase": "close",
            "outcome": "ok", "exit_code": 0, "duration_ms": 2000, "refs": {}, "result": {},
            "error": None}
    return {**base, **over}


def test_a_pair_folds_into_one_event() -> None:
    (event,) = fold([_open("a"), _close("a", outcome="error", exit_code=1)])
    assert event.id == "a" and event.action == "submit"
    assert event.outcome == "error" and event.exit_code == 1
    assert event.status == "error"


def test_a_dangling_open_folds_into_a_running_or_died_row() -> None:
    (event,) = fold([_open("a")])
    assert event.outcome is None
    assert event.status == "running-or-died"
    assert event.failed is True


def test_a_close_without_an_open_is_dropped() -> None:
    assert fold([_close("orphan")]) == []


def test_pairs_split_across_day_files_still_fold() -> None:
    records = [_open("a"), _open("b"), _close("a"), _close("b")]
    assert {e.id for e in fold(records)} == {"a", "b"}


def test_trace_becomes_note_objects() -> None:
    (event,) = fold([_open("a"), _close("a", outcome="crash",
                                        trace=[{"t": 5, "k": "provision.attempt", "d": {"z": "b"}}])])
    assert event.trace[0].k == "provision.attempt"
    assert event.trace[0].d == {"z": "b"}


def test_events_are_newest_first() -> None:
    older = _open("a", ts=(NOW - timedelta(hours=1)).isoformat())
    events = fold([older, _close("a"), _open("b"), _close("b")])
    assert [e.id for e in events] == ["b", "a"]


@pytest.fixture
def _ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    store.append(_open("a", action="submit", project={"name": "capacity"}), when=NOW)
    store.append(_close("a", outcome="error", refs={"job_id": "j-1"}), when=NOW)
    store.append(_open("b", action="doctor", project={"name": "lab"},
                       ts=(NOW - timedelta(days=5)).isoformat()), when=NOW - timedelta(days=5))
    store.append(_close("b"), when=NOW - timedelta(days=5))


def test_read_filters_by_action_project_failures_and_job(_ledger: None) -> None:
    assert [e.id for e in read(action="doctor")] == ["b"]
    assert [e.id for e in read(project="capacity")] == ["a"]
    assert [e.id for e in read(failures_only=True)] == ["a"]
    assert [e.id for e in read(job="j-1")] == ["a"]


def test_read_since_uses_the_duration_parser(_ledger: None) -> None:
    assert [e.id for e in read(since="2d", now_=NOW)] == ["a"]
    assert {e.id for e in read(since="30d", now_=NOW)} == {"a", "b"}


def test_read_limit_applies_after_filtering(_ledger: None) -> None:
    assert len(read(limit=1)) == 1


def test_row_is_brief_by_default_and_detailed_with_full() -> None:
    (event,) = fold([_open("a", params={"backend": "cpu"}),
                     _close("a", outcome="error",
                            trace=[{"t": 5, "k": "provision.attempt", "d": {"z": "b"}}])])
    brief = row(event)
    assert brief["status"] == "error" and brief["action"] == "submit"
    assert "trace" not in brief and "params" not in brief
    detailed = row(event, full=True)
    assert detailed["params"] == {"backend": "cpu"}
    assert detailed["trace"][0]["k"] == "provision.attempt"
