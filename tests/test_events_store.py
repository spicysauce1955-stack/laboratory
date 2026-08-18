from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.events import store


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


T0 = datetime(2026, 8, 18, 14, 3, 11, tzinfo=timezone.utc)


def test_append_writes_one_json_line_to_the_utc_day_file(_events_dir: Path) -> None:
    store.append({"id": "a", "phase": "open"}, when=T0)
    path = _events_dir / "2026-08-18.jsonl"
    assert path.read_text().count("\n") == 1
    assert list(store.iter_records([path])) == [{"id": "a", "phase": "open"}]


def test_appends_accumulate(_events_dir: Path) -> None:
    store.append({"id": "a"}, when=T0)
    store.append({"id": "b"}, when=T0)
    assert len(list(store.iter_records(store.day_files()))) == 2


def test_a_malformed_line_is_skipped_not_raised(_events_dir: Path) -> None:
    store.append({"id": "a"}, when=T0)
    path = _events_dir / "2026-08-18.jsonl"
    with path.open("a") as f:
        f.write("{not json\n")
    store.append({"id": "b"}, when=T0)
    assert [r["id"] for r in store.iter_records([path])] == ["a", "b"]


def test_disabled_by_env(_events_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EVENTS", "0")
    assert store.enabled() is False
    store.append({"id": "a"}, when=T0)
    assert not _events_dir.exists()


def test_an_unwritable_store_never_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    monkeypatch.setenv("LAB_EVENTS_DIR", str(blocker))
    store.append({"id": "a"}, when=T0)  # must not raise


def test_day_files_are_sorted_oldest_first(_events_dir: Path) -> None:
    store.append({"id": "a"}, when=T0)
    store.append({"id": "b"}, when=T0.replace(day=19))
    assert [p.name for p in store.day_files()] == ["2026-08-18.jsonl", "2026-08-19.jsonl"]


def test_values_that_json_cannot_encode_do_not_lose_the_record(_events_dir: Path) -> None:
    store.append({"id": "a", "params": {"p": Path("/x")}}, when=T0)
    (record,) = list(store.iter_records(store.day_files()))
    assert record["params"]["p"] == "/x"
