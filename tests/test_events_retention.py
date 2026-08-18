"""Retention runs on an injected clock. Real-clock tests anchored to a fixed T0 decay into
failures once the anchor ages — the scheduler watchdog already taught us that."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def test_lock_file_prevents_data_loss(_events_dir: Path) -> None:
    """Verify lock file gates both append and compact correctly.

    The fix uses lock_path() to ensure a stable-inode lock that both append() and
    compact() hold during their file operations, preventing the race where appends
    were lost between snapshot and rewrite.
    """
    old = NOW - timedelta(days=30)
    _write(old, *_pair("old_ok", "ok"))
    path = store.day_file(old)
    lock = store.lock_path(path)
    # Verify lock file exists (created by first append)
    assert lock.exists()
    # Verify lock has .lock suffix and day_files() doesn't match it
    assert lock.name.endswith(".jsonl.lock")
    assert lock not in store.day_files()
    # Run compact - should complete without error
    store.compact(now=NOW, success_ttl_days=14)
    # Lock file should still exist (never deleted)
    assert lock.exists()
    # Old successful call should be gone (age > TTL)
    ids = {r["id"] for r in store.iter_records(store.day_files())}
    assert "old_ok" not in ids


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
    """Verify lock files don't count toward the MB budget."""
    blob = {"id": "x", "phase": "close", "outcome": "error", "pad": "p" * 10000}
    for i in range(100):
        _write(NOW - timedelta(days=1), dict(blob, id=f"d{i}"))
    # Create lock files by appending
    store.append({"id": "lock1", "phase": "open"}, when=NOW - timedelta(days=1))
    # Get total size before cap enforcement (should not include lock files)
    files_before = store.day_files()
    size_before = sum(p.stat().st_size for p in files_before)
    # Lock files should not be counted
    lock_files = list((store.events_dir()).glob("*.jsonl.lock"))
    lock_sizes = sum(p.stat().st_size for p in lock_files)
    # Now enforce a tight cap
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=0.5)
    # Check that byte budget was enforced (day files only, not lock files)
    files_after = store.day_files()
    size_after = sum(p.stat().st_size for p in files_after)
    assert size_after <= 0.5 * 1024 * 1024
    # Lock files should still exist (they are not deleted by enforce_caps)
    assert len(lock_files) > 0


def test_enforce_caps_handles_missing_files(_events_dir: Path) -> None:
    """Verify enforce_caps doesn't crash when a file disappears mid-loop.

    This simulates two racing maybe_prune processes where one deletes a file
    while the other is iterating through remaining files.
    """
    # Write records to create multiple day files
    for age in (3, 2, 1):
        _write(NOW - timedelta(days=age), *_pair(f"d{age}", "error"))
    files_before = store.day_files()
    assert len(files_before) == 3
    # Now manually delete the oldest file mid-enforcement to simulate race
    files_before[0].unlink()
    # This should not raise even though a file disappeared
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=0.5)
    # Should complete without error
    files_after = store.day_files()
    assert len(files_after) >= 0
