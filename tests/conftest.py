"""Test-wide defaults. The event ledger defaults to ~/.lab/events, so without this every test
run would write into the developer's real ledger — and `_load_env` opens a call that only
`main()` closes, so those records would be permanently dangling `running-or-died` rows."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_event_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "lab-events"))
