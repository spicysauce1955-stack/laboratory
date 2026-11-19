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


def test_read_action_filter_matches_either_surfaces_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lab history --action sweep-aggregate` must reach the MCP calls the stats row counted."""
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    store.append(_open("a", action="sweep-aggregate"), when=NOW)
    store.append(_close("a"), when=NOW)
    store.append(_open("b", action="sweep_aggregate", surface="mcp"), when=NOW)
    store.append(_close("b"), when=NOW)
    store.append(_open("c", action="sweep"), when=NOW)
    store.append(_close("c"), when=NOW)
    for spelling in ("sweep-aggregate", "sweep_aggregate"):
        assert {e.id for e in read(action=spelling)} == {"a", "b"}
    assert {e.action for e in read(action="sweep-aggregate")} == {
        "sweep-aggregate", "sweep_aggregate",  # rows keep what each surface recorded
    }


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


# --- a read touches only the day files it needs ------------------------------------------------
#
# Every `lab history`/`lab report` used to parse the whole ledger whatever it asked for: ~1.0 s
# and 251 MB RSS on a 33 MB ledger, per call. The fix reads newest-first and stops — skipping
# day files wholly before `since`, and every older file once `limit` is satisfied — and the
# property that makes it safe is that the output is unchanged. `_reference_read` is the old
# algorithm, kept verbatim so every query below is checked against it.

END = datetime(2026, 8, 20, 18, 0, tzinfo=timezone.utc)  # the last fixture day


def _reference_read(*, since=None, project=None, action=None, session=None, job=None,
                    failures_only=False, limit=None, now_=None):
    from lab.events.stats import action_key

    events = fold(store.iter_records(store.day_files()))
    if since:
        cutoff = since_cutoff(since, now_=now_)
        if cutoff is not None:
            events = [e for e in events if e.ts >= cutoff]
    if project:
        events = [e for e in events if e.project.get("name") == project]
    if action:
        events = [e for e in events if action_key(e.action) == action_key(action)]
    if session:
        events = [e for e in events if e.session == session]
    if job:
        events = [e for e in events
                  if job == e.refs.get("job_id") or job in (e.refs.get("job_ids") or [])]
    if failures_only:
        events = [e for e in events if e.failed]
    return events[:limit] if limit else events


def _many_days(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, days: int = 10) -> None:
    """A ledger with every shape the fold has to get right across files: calls straddling UTC
    midnight (open on day N, close on day N+1), dangling opens, a close written before its
    open, identical timestamps, two projects, both spellings of one action, and failures."""
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    first = END.replace(hour=0) - timedelta(days=days - 1)
    for d in range(days):
        day = first + timedelta(days=d)
        for i in range(6):
            id_ = f"{d:02d}-{i}"
            ts = day + timedelta(hours=1 + 3 * i)
            if i == 5:  # straddles midnight: closes in the next day's file
                ts = day + timedelta(hours=23, minutes=59, seconds=59)
            if i == 2:  # a tie with call 1
                ts = day + timedelta(hours=4)
            opened = _open(
                id_, ts=ts.isoformat(), session=f"s{d % 3}",
                action=["submit", "sweep-aggregate", "sweep_aggregate", "status"][i % 4],
                project={"name": "lab" if (d + i) % 3 else "other"},
            )
            closed = _close(
                id_, ts=(ts + timedelta(seconds=30)).isoformat(),
                outcome="error" if (d * 7 + i) % 4 == 0 else "ok",
                refs={"job_id": f"job-{d % 4}"} if i % 2 else {"job_ids": [f"job-{d % 5}"]},
                error={"type": "LabError", "message": f"boom {d}"} if (d * 7 + i) % 4 == 0
                else None,
            )
            if d == 4 and i == 3:
                store.append(opened, when=ts)  # dangling: the process died
                continue
            if i == 4:  # the close lands in the file first
                store.append(closed, when=ts)
                store.append(opened, when=ts)
                continue
            store.append(opened, when=ts)
            store.append(closed, when=ts + timedelta(seconds=30))


QUERIES = [
    {},
    {"limit": 1}, {"limit": 2}, {"limit": 5}, {"limit": 6}, {"limit": 7}, {"limit": 13},
    {"limit": 1000},
    {"since": "1d"}, {"since": "36h"}, {"since": "3d", "limit": 4}, {"since": "100d"},
    {"since": "2026-08-15"}, {"since": "2026-08-15T23:59:59"}, {"since": "2026-08-21"},
    {"project": "other", "limit": 3}, {"project": "lab", "since": "2d"},
    {"action": "sweep-aggregate", "limit": 4}, {"action": "sweep_aggregate"},
    {"session": "s1", "limit": 2}, {"job": "job-2"}, {"job": "job-3", "limit": 2},
    {"failures_only": True, "limit": 3}, {"failures_only": True, "since": "4d"},
    {"project": "lab", "failures_only": True, "action": "submit", "limit": 2},
]


@pytest.mark.parametrize("query", QUERIES, ids=[str(q) for q in QUERIES])
def test_a_bounded_read_returns_exactly_what_a_full_read_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, query: dict
) -> None:
    _many_days(tmp_path, monkeypatch)
    got = [row(e, full=True) for e in read(now_=END, **query)]
    want = [row(e, full=True) for e in _reference_read(now_=END, **query)]
    assert got == want


def _spy_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []
    real = store.iter_records

    def spy(paths):
        for path in paths:
            opened.append(Path(path).name)
            yield from real([path])

    monkeypatch.setattr(store, "iter_records", spy)
    return opened


def test_a_since_window_never_opens_a_day_file_wholly_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _many_days(tmp_path, monkeypatch)
    opened = _spy_reads(monkeypatch)
    read(since="1d", now_=END)  # cutoff 2026-08-19T18:00: the 19th and 20th can hold it, and
    # the 21st holds only the close of a call opened just before midnight on the 20th.
    assert opened == ["2026-08-21.jsonl", "2026-08-20.jsonl", "2026-08-19.jsonl"]


def test_a_satisfied_limit_stops_before_the_older_day_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _many_days(tmp_path, monkeypatch)
    opened = _spy_reads(monkeypatch)
    assert len(read(limit=3)) == 3
    # Newest first — the 21st holds only a straddling close, the 20th fills the limit — and
    # nothing older is opened.
    assert opened == ["2026-08-21.jsonl", "2026-08-20.jsonl"]


def test_a_read_with_an_unparseable_day_file_name_still_reads_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`day_files()` globs `????-??-??.jsonl`. A file whose name is not a date says nothing
    about which records it holds, so it must not be skipped — and it disables the shortcuts."""
    _many_days(tmp_path, monkeypatch)
    stray = Path(store.events_dir()) / "back-up-xx.jsonl"
    stray.write_text(json.dumps(_open("zz", ts=(END - timedelta(days=30)).isoformat())) + "\n")
    got = [e.id for e in read(since="1d", now_=END)]
    assert got == [e.id for e in _reference_read(since="1d", now_=END)]
    assert "zz" in [e.id for e in read(limit=1000)]


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
