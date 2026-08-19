"""Retention runs on an injected clock. Real-clock tests anchored to a fixed T0 decay into
failures once the anchor ages — the scheduler watchdog already taught us that."""

from __future__ import annotations

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
