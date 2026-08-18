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
