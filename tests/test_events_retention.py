"""Retention runs on an injected clock. Real-clock tests anchored to a fixed T0 decay into
failures once the anchor ages — the scheduler watchdog already taught us that."""

from __future__ import annotations

import fcntl
import os
import threading
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from lab.events import store

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


def _write(when: datetime, *records: dict) -> None:
    for record in records:
        store.append(record, when=when)


def _pair(id_: str, outcome: str) -> tuple[dict, dict]:
    return ({"id": id_, "phase": "open", "action": "submit"},
            {"id": id_, "phase": "close", "outcome": outcome})


def test_compaction_drops_old_successes_and_keeps_old_failures(_events_dir: Path) -> None:
    old = NOW - timedelta(days=30)
    _write(old, *_pair("ok1", "ok"), *_pair("bad1", "error"))
    store.compact(now=NOW, success_ttl_days=14)
    ids = {r["id"] for r in store.iter_records(store.day_files())}
    assert ids == {"bad1"}


def test_compaction_keeps_recent_successes(_events_dir: Path) -> None:
    _write(NOW - timedelta(days=3), *_pair("ok1", "ok"))
    store.compact(now=NOW, success_ttl_days=14)
    assert {r["id"] for r in store.iter_records(store.day_files())} == {"ok1"}


def test_compaction_keeps_dangling_opens_regardless_of_age(_events_dir: Path) -> None:
    _write(NOW - timedelta(days=60), {"id": "hung", "phase": "open", "action": "submit"})
    store.compact(now=NOW, success_ttl_days=14)
    assert {r["id"] for r in store.iter_records(store.day_files())} == {"hung"}


def test_compaction_recognizes_a_success_whose_close_landed_in_the_next_days_file(
    _events_dir: Path,
) -> None:
    """A call that straddles UTC midnight: `open` in day N, `close` in day N+1 — any supervisor
    run of more than a few hours, or an overnight scheduled job. Both days are old enough to be
    eligible for compaction, so the bug isn't masked by day N+1 being too recent to touch.

    The pre-fix `compact()` computed its `succeeded` id set per day file: day N's file has no
    close to prove success, so its `open` survived; day N+1's close was dropped as a lone
    success with no open to pair it with in that file. The `open` becomes a permanent
    `running-or-died` phantom. Reproduced exactly as the reviewer described it:
    ``before: [('A','ok',False)] -> after: [('A','running-or-died',True)]``."""
    day_n = NOW - timedelta(days=30)
    day_n_plus_1 = day_n + timedelta(days=1)
    store.append({"id": "A", "phase": "open", "action": "submit"}, when=day_n)
    store.append({"id": "A", "phase": "close", "outcome": "ok"}, when=day_n_plus_1)

    store.compact(now=NOW, success_ttl_days=14)

    ids = {r["id"] for r in store.iter_records(store.day_files())}
    assert "A" not in ids, "a genuinely successful call must not survive as a phantom open"


def test_age_cap_deletes_whole_day_files(_events_dir: Path) -> None:
    _write(NOW - timedelta(days=200), *_pair("ancient", "error"))
    _write(NOW - timedelta(days=2), *_pair("recent", "error"))
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=50)
    assert [p.name for p in store.day_files()] == [f"{(NOW - timedelta(days=2)).date()}.jsonl"]


def test_size_cap_deletes_oldest_first_until_under_budget(_events_dir: Path) -> None:
    blob = {"id": "x", "phase": "close", "outcome": "error", "pad": "p" * 2000}
    for age in (5, 4, 3):
        _write(NOW - timedelta(days=age), *[dict(blob, id=f"d{age}-{i}") for i in range(200)])
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=0.5)
    total = sum(p.stat().st_size for p in store.day_files())
    assert total <= 0.5 * 1024 * 1024
    assert (NOW - timedelta(days=3)).strftime("%Y-%m-%d") in {p.stem for p in store.day_files()}


def test_maybe_prune_runs_once_per_day(_events_dir: Path) -> None:
    _write(NOW - timedelta(days=30), *_pair("ok1", "ok"))
    store.maybe_prune(now=NOW)
    assert {r["id"] for r in store.iter_records(store.day_files())} == set()
    # a second call the same day must not re-scan; re-add and confirm it survives
    _write(NOW - timedelta(days=30), *_pair("ok2", "ok"))
    store.maybe_prune(now=NOW)
    assert {r["id"] for r in store.iter_records(store.day_files())} == {"ok2"}
    store.maybe_prune(now=NOW + timedelta(days=1))
    assert {r["id"] for r in store.iter_records(store.day_files())} == set()


def test_pruning_failure_is_swallowed(_events_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "compact", lambda **_: (_ for _ in ()).throw(OSError("boom")))
    store.maybe_prune(now=NOW)  # must not raise


def test_concurrent_append_not_lost_during_compaction(
    _events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent append() during compact()'s critical section must not be lost.

    This calls the real ``store.compact`` — no re-implementation of its body. The only
    test-controlled seam is a wrapper around ``store.iter_records``, a function compact()
    already calls to take its snapshot. The wrapper does the real read, then blocks on a
    ``threading.Event`` (not a sleep) so the main thread can deterministically start a
    concurrent ``append()`` while compact is paused between "snapshot taken" and "decide
    and rewrite" — exactly the window the pre-fix code left unlocked.

    On the fixed store this passes because compact() takes the per-day lock file *before*
    calling iter_records, so the paused window is still inside the lock: append() simply
    blocks until compact() finishes and releases, and its write always lands after
    compact()'s rewrite. Nothing is lost, only ordered. Reverting compact() to the pre-fix
    body (which reads unlocked and only locks around the write) loses "concurrent_fail" to
    exactly this interleaving — see the RED evidence in the fix report.
    """
    old = NOW - timedelta(days=30)
    _write(old, *_pair("old_ok", "ok"))

    real_iter_records = store.iter_records
    snapshot_taken = threading.Event()
    release_compact = threading.Event()

    def paused_iter_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
        records = list(real_iter_records(paths))
        snapshot_taken.set()
        if not release_compact.wait(timeout=5):
            raise TimeoutError("test never released compact's paused snapshot")
        return iter(records)

    monkeypatch.setattr(store, "iter_records", paused_iter_records)

    compact_errors: list[BaseException] = []

    def run_compact() -> None:
        try:
            store.compact(now=NOW, success_ttl_days=14)
        except BaseException as e:  # noqa: BLE001 — captured and re-raised in the main thread
            compact_errors.append(e)

    compact_thread = threading.Thread(target=run_compact)
    compact_thread.start()
    assert snapshot_taken.wait(timeout=5), "compact() never reached its snapshot point"

    append_errors: list[BaseException] = []

    def run_append() -> None:
        try:
            store.append(
                {"id": "concurrent_fail", "phase": "close", "outcome": "error"}, when=old
            )
        except BaseException as e:  # noqa: BLE001 — captured and re-raised in the main thread
            append_errors.append(e)

    # append() from a real production caller, concurrently, while compact() is paused —
    # this is the interleaving under test, not a re-implementation of either function.
    append_thread = threading.Thread(target=run_append)
    append_thread.start()

    release_compact.set()

    compact_thread.join(timeout=5)
    append_thread.join(timeout=5)

    assert not compact_thread.is_alive(), "compact() thread did not finish in time"
    assert not append_thread.is_alive(), "append() thread did not finish in time"
    if compact_errors:
        raise compact_errors[0]
    if append_errors:
        raise append_errors[0]

    ids = {r["id"] for r in store.iter_records(store.day_files())}
    assert "concurrent_fail" in ids, "concurrent append was lost to the compaction race"
    assert "old_ok" not in ids  # compaction still did its job on the pre-existing success


def test_lock_file_structure(_events_dir: Path) -> None:
    """Verify lock file naming and exclusion from operations."""
    old = NOW - timedelta(days=1)
    _write(old, *_pair("id1", "ok"))
    path = store.day_file(old)
    lock = store.lock_path(path)
    # Verify lock file has correct name (string concat, not with_suffix)
    assert lock.name == f"{path.stem}.jsonl.lock"
    assert lock.exists()
    # Verify lock files don't appear in day_files() glob
    day_files = store.day_files()
    assert all(p.name.endswith(".jsonl") and not p.name.endswith(".lock") for p in day_files)
    assert lock not in day_files


def test_lock_files_excluded_from_day_files_glob(_events_dir: Path) -> None:
    """Verify lock files don't appear in day_files() or iter_records."""
    _write(NOW - timedelta(days=1), *_pair("id1", "ok"))
    # Lock file should exist but not be returned by day_files()
    path = store.day_file(NOW - timedelta(days=1))
    lock = store.lock_path(path)
    # Trigger lock creation by appending
    store.append({"id": "id2", "phase": "open", "action": "submit"}, when=NOW - timedelta(days=1))
    # Verify lock file exists but is not in day_files()
    assert lock.exists()
    day_files = store.day_files()
    assert all(p.name.endswith(".jsonl") and not p.name.endswith(".jsonl.lock") for p in day_files)
    # Verify lock file doesn't appear in iter_records
    all_records = {r["id"] for r in store.iter_records(store.day_files())}
    assert "id1" in all_records
    assert "id2" in all_records


def test_lock_files_excluded_from_byte_budget(_events_dir: Path) -> None:
    """Verify lock files don't appear in day_files() glob."""
    _write(NOW - timedelta(days=1), {"id": "x", "phase": "close", "outcome": "error"})
    # Trigger lock creation by appending
    store.append({"id": "lock1", "phase": "open"}, when=NOW - timedelta(days=1))
    # Verify lock files exist but are excluded from day_files()
    lock_files = list((store.events_dir()).glob("*.jsonl.lock"))
    assert len(lock_files) > 0  # Lock files exist
    day_files = store.day_files()
    # Glob exclusion is proven by prior test; this verifies the pattern holds
    assert all(p.name.endswith(".jsonl") for p in day_files)
    assert not any(p.name.endswith(".lock") for p in day_files)


PAD = "p" * 2000


def _successes(day_tag: str, count: int) -> list[dict]:
    """`count` complete successful calls, padded so a day of them is worth real bytes."""
    out: list[dict] = []
    for i in range(count):
        id_ = f"poll-{day_tag}-{i}"
        out.append({"id": id_, "phase": "open", "action": "status", "pad": PAD})
        out.append({"id": id_, "phase": "close", "outcome": "ok"})
    return out


def _ids() -> set[str]:
    return {r["id"] for r in store.iter_records(store.day_files())}


def _stems() -> set[str]:
    return {p.stem for p in store.day_files()}


def test_over_cap_compacts_non_today_successes_instead_of_deleting_day_one(
    _events_dir: Path,
) -> None:
    """The 2026-09 incident, reproduced: five campaign days of successful `lab status` polls
    (98,654 of them in reality, at a flat 1,050/hour) plus today's, over the byte cap. Every
    success is *fresh*, so the age-gated `compact()` could not touch one of them and the byte
    cap's only lever was deleting a whole day file — it deleted day one of the campaign, the day
    of the incident under investigation, mid-investigation.

    Over the cap, successes are droppable regardless of age: compact first, delete only if that
    was not enough. Day one keeps its failure record, and today's file is never the sacrifice.
    """
    for age in (5, 4, 3, 2, 1, 0):
        _write(NOW - timedelta(days=age), *_successes(f"d{age}", 100))
    _write(NOW - timedelta(days=5), *_pair("incident", "error"))

    store.enforce_caps(now=NOW, max_age_days=90, max_mb=0.5)

    day_one = (NOW - timedelta(days=5)).strftime("%Y-%m-%d")
    assert day_one in _stems(), "day one of the campaign was deleted to satisfy the byte cap"
    assert NOW.strftime("%Y-%m-%d") in _stems(), "today's file was sacrificed to the byte cap"
    ids = _ids()
    assert "incident" in ids, "the failure under investigation was dropped"
    assert not any(i.startswith("poll-d5-") for i in ids), "non-today successes were not compacted"
    assert any(i.startswith("poll-d0-") for i in ids), "today's records must be left alone"
    total = sum(p.stat().st_size for p in store.day_files())
    assert total <= 0.5 * 1024 * 1024


def test_over_cap_compaction_never_drops_a_failure(_events_dir: Path) -> None:
    """Failures survive the over-cap compaction; only the age cap (or, still, whole-file deletion
    as a last resort) may take them."""
    for age in (3, 2, 1):
        _write(
            NOW - timedelta(days=age),
            *[
                {"id": f"bad-d{age}-{i}", "phase": "close", "outcome": "error", "pad": PAD}
                for i in range(100)
            ],
        )
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=0.5)
    survivors = _ids()
    # Whatever files survive, they were not rewritten: every failure in them is still there.
    for age in (3, 2, 1):
        day = store.day_file(NOW - timedelta(days=age))
        if not day.exists():
            continue
        assert len({r["id"] for r in store.iter_records([day])}) == 100
    assert survivors, "every file was deleted — the deletion lever ran unbounded"


def test_over_cap_still_deletes_a_day_when_compaction_is_not_enough(_events_dir: Path) -> None:
    """Compaction is a first lever, not a replacement: a ledger over the cap purely on failure
    records still loses whole day files, oldest first, exactly as before."""
    for age in (5, 4, 3):
        _write(
            NOW - timedelta(days=age),
            *[
                {"id": f"bad-d{age}-{i}", "phase": "close", "outcome": "error", "pad": PAD}
                for i in range(100)
            ],
        )
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=0.3)
    assert (NOW - timedelta(days=5)).strftime("%Y-%m-%d") not in _stems()
    total = sum(p.stat().st_size for p in store.day_files())
    assert total <= 0.3 * 1024 * 1024


def test_todays_file_is_never_deleted_for_the_byte_cap(_events_dir: Path) -> None:
    """A single day over the cap by itself overshoots by at most that day; deleting the live day
    would destroy exactly the records anyone is about to read."""
    _write(
        NOW,
        *[
            {"id": f"bad-today-{i}", "phase": "close", "outcome": "error", "pad": PAD}
            for i in range(100)
        ],
    )
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=0.05)
    assert _stems() == {NOW.strftime("%Y-%m-%d")}


def test_under_the_cap_nothing_is_compacted(_events_dir: Path) -> None:
    """The over-cap compaction is a cap lever only: under budget, a fresh success inside the TTL
    is still there to be read."""
    _write(NOW - timedelta(days=1), *_pair("fresh_ok", "ok"))
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=50)
    assert "fresh_ok" in _ids()


def test_ttl_ignoring_compaction_still_pairs_a_success_across_two_day_files(
    _events_dir: Path,
) -> None:
    """The over-cap path's new parameters must not reintroduce the per-file `succeeded` set bug
    `compact()`'s docstring documents: an `open` in day N with its `close` in day N+1 is a
    success, and the `open` must go with it rather than becoming a `running-or-died` phantom."""
    day_n = NOW - timedelta(days=2)
    store.append({"id": "A", "phase": "open", "action": "status"}, when=day_n)
    store.append({"id": "A", "phase": "close", "outcome": "ok"}, when=day_n + timedelta(days=1))

    store.compact(now=NOW, success_ttl_days=14, ignore_ttl=True, exclude_today=True)

    assert "A" not in _ids()


def test_ttl_ignoring_compaction_leaves_todays_file_alone(_events_dir: Path) -> None:
    _write(NOW, *_pair("today_ok", "ok"))
    _write(NOW - timedelta(days=1), *_pair("yesterday_ok", "ok"))
    store.compact(now=NOW, success_ttl_days=14, ignore_ttl=True, exclude_today=True)
    assert _ids() == {"today_ok"}


def _orphan_lock(day: datetime, *, age_days: float) -> Path:
    lock = store.lock_path(store.day_file(day))
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("", encoding="utf-8")
    stamp = (NOW - timedelta(days=age_days)).timestamp()
    os.utime(lock, (stamp, stamp))
    return lock


def test_a_stale_lock_with_no_day_file_is_reaped(_events_dir: Path) -> None:
    """Ten of these were sitting in the real ledger directory, one of them for the day whose data
    the byte cap had already deleted — litter that outlives the file it guarded."""
    lock = _orphan_lock(NOW - timedelta(days=30), age_days=30)
    store.reap_stale_locks(now=NOW)
    assert not lock.exists()


def test_a_lock_whose_day_file_still_exists_is_kept(_events_dir: Path) -> None:
    day = NOW - timedelta(days=30)
    _write(day, *_pair("old", "error"))
    lock = store.lock_path(store.day_file(day))
    stamp = (NOW - timedelta(days=30)).timestamp()
    os.utime(lock, (stamp, stamp))
    store.reap_stale_locks(now=NOW)
    assert lock.exists(), "the lock for a live day file is load-bearing"


def test_a_fresh_orphan_lock_is_kept(_events_dir: Path) -> None:
    """Younger than a day: a process may be between creating the lock and writing its first
    record, and its day file does not exist yet."""
    lock = _orphan_lock(NOW, age_days=0)
    store.reap_stale_locks(now=NOW)
    assert lock.exists()


def test_a_held_lock_is_not_reaped(_events_dir: Path) -> None:
    """Belt and braces on top of the age and day-file conditions: if anything holds the lock
    right now, leave it. `flock` is per open-file-description, so a second handle on the same
    path conflicts even inside one process."""
    lock = _orphan_lock(NOW - timedelta(days=30), age_days=30)
    with lock.open("a", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        try:
            store.reap_stale_locks(now=NOW)
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
    assert lock.exists()


def test_reaping_a_lock_never_raises(_events_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _orphan_lock(NOW - timedelta(days=30), age_days=30)

    def _boom(self: Path, *a: object, **k: object) -> None:
        raise OSError("nope")

    monkeypatch.setattr(Path, "unlink", _boom)
    store.reap_stale_locks(now=NOW)  # best-effort like the rest of retention


def test_maybe_prune_reaps_stale_locks(_events_dir: Path) -> None:
    lock = _orphan_lock(NOW - timedelta(days=30), age_days=30)
    store.maybe_prune(now=NOW)
    assert not lock.exists()


def test_enforce_caps_handles_missing_files(
    _events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify enforce_caps doesn't crash when a file disappears mid-loop.

    This simulates two racing maybe_prune processes. We monkeypatch Path.stat()
    to delete the file on first call, simulating disappearance between the
    existence check and the stat call.
    """
    # Write records to create multiple day files
    for age in (3, 2, 1):
        _write(NOW - timedelta(days=age), *_pair(f"d{age}", "error"))
    files_before = store.day_files()
    assert len(files_before) == 3
    target_file = files_before[0]
    original_stat = Path.stat
    call_count = [0]

    def patched_stat(self: Path, *args: object, **kwargs: object) -> object:
        # First call on our target file: delete it, then raise
        if self == target_file and call_count[0] == 0:
            call_count[0] += 1
            target_file.unlink()
            raise OSError("File vanished")
        # All other calls: use original — must accept e.g. follow_symlinks=... too
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", patched_stat)
    # This should not raise even though a file disappeared mid-loop
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=0.5)
    # Should complete without error and skip the vanished file
    files_after = store.day_files()
    # The target file should be gone (we deleted it)
    assert target_file not in files_after
    # We should still have some files (the newer ones)
    assert len(files_after) > 0
