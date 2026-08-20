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
def _isolate_event_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "lab-events"))
    # Same isolation for the machine-wide job index: a test must never claim a job in the
    # real `~/.lab/jobs`, nor read another project's real claims.
    monkeypatch.setenv("LAB_JOBS_INDEX_DIR", str(tmp_path / "lab-jobs"))
    _record_module._current.set(None)
    yield
    _record_module._current.set(None)


@pytest.fixture(autouse=True)
def _no_real_cloud_clients(monkeypatch):
    """No test may reach a real provider API by *omission*.

    This dev box has doctl configured, so an unpatched DO call in a test does not fail — it
    quietly enumerates the live account, and the code under test here is teardown code whose job
    is to destroy things. A test that forgets to stub the client should break loudly rather than
    reach production; tests that legitimately need a client monkeypatch it themselves, and their
    patch (applied later) wins over this one.
    """

    def _refuse(*a, **k):
        raise AssertionError(
            "a test tried to build a real DigitalOcean client — monkeypatch "
            "lab.backends.skypilot._get_do_client in this test"
        )

    monkeypatch.setattr("lab.backends.skypilot._get_do_client", _refuse)


@pytest.fixture(autouse=True)
def _hermetic_sky_versions(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Pin the SkyPilot client/server version check to "compatible" by default.

    `Lab.reconcile(apply=True)` refuses to destroy anything under version skew, because under skew
    the client cannot decode the result of a destroy (incident 2026-08-20). Left unpatched, every
    reconcile test would depend on whether a real API server happens to be running on the
    developer's machine and which version it is -- the suite would pass or fail by accident.
    Tests that exercise the skew behaviour itself override this with their own value.
    """
    if request.node.module.__name__ == "test_skycompat":
        return  # the module under test owns this function; stubbing it would test the stub
    from lab._skycompat import SkyVersions

    monkeypatch.setattr(
        "lab._skycompat.sky_versions",
        lambda **kw: SkyVersions(client="0.12.3", server="0.12.3", compatible=True, detail="ok"),
    )
