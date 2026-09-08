from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab.events import store
from lab.events.read import crossref, fold, read, row, since_cutoff

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
    close = _close("a", outcome="crash",
                    trace=[{"t": 5, "k": "provision.attempt", "d": {"z": "b"}}])
    (event,) = fold([_open("a"), close])
    assert event.trace[0].k == "provision.attempt"
    assert event.trace[0].d == {"z": "b"}


def test_events_are_newest_first() -> None:
    older = _open("a", ts=(NOW - timedelta(hours=1)).isoformat())
    events = fold([older, _close("a"), _open("b"), _close("b")])
    assert [e.id for e in events] == ["b", "a"]


def test_a_close_arriving_before_its_open_still_folds() -> None:
    (event,) = fold([_close("a", outcome="error"), _open("a")])
    assert event.id == "a"
    assert event.outcome == "error"


def test_a_non_list_trace_is_dropped_but_the_batch_survives() -> None:
    events = fold([_open("a"), _close("a", trace="not-a-list"),
                    _open("b"), _close("b", outcome="ok")])
    by_id = {e.id: e for e in events}
    assert by_id["a"].trace == []
    assert by_id["a"].outcome == "ok"
    assert by_id["b"].outcome == "ok"


def test_a_non_dict_trace_entry_is_skipped_but_others_survive() -> None:
    close = _close("a", trace=["not-a-mapping",
                                {"t": 5, "k": "provision.attempt", "d": {}}])
    (event,) = fold([_open("a"), close])
    assert len(event.trace) == 1
    assert event.trace[0].k == "provision.attempt"


def test_a_non_int_seq_defaults_but_the_batch_survives() -> None:
    events = fold([_open("a", seq="bad"), _close("a"), _open("b"), _close("b")])
    by_id = {e.id: e for e in events}
    assert by_id["a"].seq == 0
    assert by_id["b"].id == "b"


def test_a_non_finite_seq_defaults_but_the_batch_survives() -> None:
    # json.loads accepts NaN/Infinity literals by default, so a non-finite float is reachable
    # from a real ledger line, not just a hand-built Python float. Round-trip through json.dumps
    # (allow_nan=True by default) and json.loads to prove that: the wire text really contains the
    # bareword NaN literal, exactly as a line written by json.dumps(record, default=str) would.
    raw = json.dumps(_open("a", seq=float("nan")))
    assert '"seq":NaN' in raw.replace(" ", "")
    open_a = json.loads(raw)
    assert open_a["seq"] != open_a["seq"]  # sanity: really is NaN, not the literal string
    events = fold([open_a, _close("a"), _open("b"), _close("b")])
    by_id = {e.id: e for e in events}
    assert by_id["a"].seq == 0
    assert by_id["b"].id == "b"


def test_a_non_finite_trace_t_defaults_but_the_entry_and_batch_survive() -> None:
    close_a = _close("a", trace=[{"t": float("inf"), "k": "provision.attempt", "d": {}}])
    events = fold([_open("a"), close_a, _open("b"), _close("b")])
    by_id = {e.id: e for e in events}
    assert by_id["a"].trace[0].t == 0
    assert by_id["a"].trace[0].k == "provision.attempt"
    assert by_id["b"].id == "b"


def test_non_dict_mapping_fields_are_coerced_but_the_batch_survives() -> None:
    events = fold([
        _open("a", params="nope", project=["not", "a", "dict"]),
        _close("a", refs="nope", result=5),
        _open("b"), _close("b"),
    ])
    by_id = {e.id: e for e in events}
    assert by_id["a"].params == {}
    assert by_id["a"].project == {}
    assert by_id["a"].refs == {}
    assert by_id["a"].result == {}
    assert by_id["b"].outcome == "ok"


def test_a_non_dict_error_is_dropped_and_row_stays_usable() -> None:
    (event,) = fold([_open("a"), _close("a", outcome="error", error="boom")])
    assert event.error is None
    assert row(event)["error"] is None


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


def test_read_job_filter_matches_job_ids_list_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    store.append(_open("a", action="sweep"), when=NOW)
    store.append(_close("a", refs={"job_ids": ["j-2", "j-3"]}), when=NOW)
    store.append(_open("b", action="sweep"), when=NOW)
    store.append(_close("b", refs={"job_ids": ["j-9"]}), when=NOW)
    assert [e.id for e in read(job="j-3")] == ["a"]


def test_read_since_uses_the_duration_parser(_ledger: None) -> None:
    assert [e.id for e in read(since="2d", now_=NOW)] == ["a"]
    assert {e.id for e in read(since="30d", now_=NOW)} == {"a", "b"}


def test_since_cutoff_accepts_durations_including_compounds() -> None:
    assert since_cutoff("2d", now_=NOW) == NOW - timedelta(days=2)
    assert since_cutoff("30m", now_=NOW) == NOW - timedelta(minutes=30)
    assert since_cutoff("3h30m", now_=NOW) == NOW - timedelta(hours=3, minutes=30)
    assert since_cutoff("3600", now_=NOW) == NOW - timedelta(hours=1)  # plain seconds
    assert since_cutoff(None) is None
    assert since_cutoff("") is None


def test_since_cutoff_accepts_an_absolute_date_as_utc_midnight() -> None:
    """Ledger, three times during the 2026-09 campaign: `bad since '2026-09-06': could not
    convert string to float: '2026-09-06'`. Someone investigating an incident reaches for the
    date, not for a duration.

    A bare date is the **start of that day in UTC**, never local midnight: the whole ledger is
    stamped UTC (`store.append` / `_util.now`), the incident box runs UTC+3, and a cutoff
    silently shifted by the reader's zone would drop or resurrect rows either side of a day
    boundary with nothing on screen to say so.
    """
    assert since_cutoff("2026-09-06") == datetime(2026, 9, 6, tzinfo=timezone.utc)
    assert since_cutoff("2026-09-06").tzinfo is not None  # comparable with an aware event ts


def test_since_cutoff_accepts_iso_datetimes_at_every_precision() -> None:
    assert since_cutoff("2026-09-07T02") == datetime(2026, 9, 7, 2, tzinfo=timezone.utc)  # ledger
    assert since_cutoff("2026-09-07T02:30") == datetime(2026, 9, 7, 2, 30, tzinfo=timezone.utc)
    assert since_cutoff("2026-09-07 02:30:15") == datetime(
        2026, 9, 7, 2, 30, 15, tzinfo=timezone.utc
    )
    # An explicit offset is honoured rather than reinterpreted, and normalised to UTC.
    assert since_cutoff("2026-09-07T05:30+03:00") == datetime(2026, 9, 7, 2, 30, tzinfo=timezone.utc)
    assert since_cutoff("2026-09-07T02:30:00Z") == datetime(2026, 9, 7, 2, 30, tzinfo=timezone.utc)


def test_since_cutoff_is_utc_regardless_of_the_local_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The absolute path must not go through any local-time constructor. Pinned by running the
    same input under a non-UTC `TZ` (the scheduler droplet and the 2026-08 incident box are both
    UTC+3) — `datetime.fromisoformat` on a naive string is zone-free, and this asserts nobody
    later "fixes" it with `astimezone()`/`.timestamp()`, which would silently shift the window."""
    monkeypatch.setenv("TZ", "Asia/Nicosia")
    time.tzset()
    try:
        assert since_cutoff("2026-09-06") == datetime(2026, 9, 6, tzinfo=timezone.utc)
    finally:
        monkeypatch.delenv("TZ", raising=False)
        time.tzset()


def test_since_cutoff_rejects_garbage_naming_both_accepted_forms() -> None:
    with pytest.raises(ValueError) as exc_info:
        since_cutoff("garbage")
    message = str(exc_info.value)
    assert "garbage" in message
    assert "2026-09-06" in message  # names the absolute form
    assert "30m" in message  # ...and the relative one
    assert "could not convert" not in message
    for bad in ["2026-13-06", "2026-09-32", "2026-09"]:
        with pytest.raises(ValueError, match=r"invalid since"):
            since_cutoff(bad)


def test_read_since_accepts_an_absolute_date(_ledger: None) -> None:
    """The fixture's two events are stamped NOW (2026-08-18) and NOW-5d (2026-08-13)."""
    assert [e.id for e in read(since="2026-08-18")] == ["a"]
    assert {e.id for e in read(since="2026-08-01")} == {"a", "b"}
    assert read(since="2026-08-19") == []


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


class _FakeManifest:
    def __init__(self, state: str, end_reason: str | None) -> None:
        self.status = _FakeState(state)
        self.end_reason = end_reason


class _FakeState:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeStore:
    """A minimal stand-in for `lab.store.JobStore` — proves `crossref`/`row` work against
    anything shaped like a store, not a hard import of `JobStore` itself (the module-boundary
    fix: `lab.events` takes an injected store rather than importing `lab.store`)."""

    def __init__(self, manifests: dict[str, _FakeManifest]) -> None:
        self._manifests = manifests

    def read_manifest(self, job_id: str) -> _FakeManifest:
        return self._manifests[job_id]  # KeyError on a miss, same as JobStore's file-not-found

    def logs_path(self, job_id: str) -> str:
        return f"runs/{job_id}/logs.txt"


def test_crossref_resolves_manifest_state_and_logs_path() -> None:
    (event,) = fold([_open("a"), _close("a", outcome="error", refs={"job_id": "j-1"})])
    fake = _FakeStore({"j-1": _FakeManifest("failed", "timed out after 20m wall-clock cap")})
    assert crossref(event, fake) == {
        "manifest_state": "failed",
        "manifest_end_reason": "timed out after 20m wall-clock cap",
        "logs_path": "runs/j-1/logs.txt",
    }


def test_crossref_degrades_to_empty_on_a_missing_manifest() -> None:
    """A job id in `refs` whose manifest is gone (cleaned-up runs/, or it never got that far)
    must not raise — cross-referencing is best-effort, the forensic view still has everything
    else in the row."""
    (event,) = fold([_open("a"), _close("a", outcome="error", refs={"job_id": "j-missing"})])
    assert crossref(event, _FakeStore({})) == {}


def test_crossref_is_a_no_op_without_a_job_id() -> None:
    (event,) = fold([_open("a"), _close("a", outcome="ok")])
    assert crossref(event, _FakeStore({})) == {}


def test_row_full_cross_references_when_a_store_is_supplied() -> None:
    (event,) = fold([_open("a"), _close("a", outcome="error", refs={"job_id": "j-1"})])
    fake = _FakeStore({"j-1": _FakeManifest("failed", "no capacity")})
    brief = row(event, full=True)  # no store -> no crossref fields, no crash
    assert "manifest_state" not in brief
    detailed = row(event, full=True, job_store=fake)
    assert detailed["manifest_state"] == "failed"
    assert detailed["logs_path"] == "runs/j-1/logs.txt"
