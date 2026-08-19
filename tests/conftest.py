"""Test-wide defaults. The event ledger defaults to ~/.lab/events, so without this every test
run would write into the developer's real ledger — and `_load_env` opens a call that only
`main()` closes, so those records would be permanently dangling `running-or-died` rows.

The same "`_load_env` opens, only `main()` closes" fact also means any test that drives a CLI
command straight through `CliRunner.invoke(app, ...)` — bypassing `main()` — leaves
`lab.events`'s current-call `ContextVar` pointing at that test's now-torn-down `Call` forever
after: a plain module-level `ContextVar` shares its value across every test in one pytest
process unless something resets it. Left alone, that dangling reference is invisible to a test
that never inspects `events.current()`, but it corrupts one that does (or that opens its own
call and expects to find nothing "current" beforehand) — a cross-test pollution bug, not
anything wrong with the polluted test itself. Reset it here so every test starts clean.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# `lab.events.__init__` re-exports the `record` *function* under the name `record`, shadowing
# the `lab.events.record` submodule as a package attribute — `from lab.events import record`
# would bind to the function, not the module. Go through `sys.modules` (via `importlib`) to
# reach the module itself, the same workaround `test_events_record.py` already documents.
_record_module = importlib.import_module("lab.events.record")


@pytest.fixture(autouse=True)
def _isolate_event_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "lab-events"))
    _record_module._current.set(None)
    yield
    _record_module._current.set(None)
