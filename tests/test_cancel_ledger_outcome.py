"""A supervisor stopped by ``lab cancel`` must not be counted as a failure in the ledger.

Smoke run, 2026-11-19: job F was ``lab cancel``led. Its manifest correctly says ``cancelled`` (the
abort path reads the ``cancelling`` intent ``lab cancel`` writes before it SIGTERMs the
supervisor), but the supervisor's own ledger call closed as ``interrupted`` with
``SupervisorTerminated: supervisor terminated by SIGTERM`` — so ``lab history --failures`` and
``lab report`` listed a deliberate user action among the failures, beside the crashes they exist to
surface. A failure list padded with cancels is one people learn to skim (R10).

The distinction is the ``cancelling`` intent, not the signal: the same SIGTERM from anything else
— systemd stopping the unit, an OOM-killer's parent, a stray ``kill`` — is a real interruption of
a job that was meant to run, and must stay a failure.
"""

from __future__ import annotations

import signal
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from helpers import make_manifest

import lab.sky_runner as runner_mod
from lab.events import store as ledger
from lab.events.models import Event
from lab.events.read import fold
from lab.models import JobState
from lab.store import JobStore


class _FakeSky(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("sky")

    def launch(self, task: Any, cluster_name: str, **kw: Any) -> str:
        return "req-1"

    def api_cancel(self, request_id: Any) -> None:
        pass


def _supervisor_call() -> Event:
    calls = [e for e in fold(ledger.iter_records(ledger.day_files())) if e.surface == "supervisor"]
    assert calls, "the supervisor wrote no ledger record at all"
    return calls[-1]


def _sigterm_mid_provision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, user_cancel: bool
) -> JobStore:
    store = JobStore(tmp_path / "runs")
    store.create(make_manifest("j1", "python x.py"))
    monkeypatch.setitem(sys.modules, "sky", _FakeSky())
    monkeypatch.setattr(runner_mod, "build_task", lambda *a, **k: "task")
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)

    def _signalled(*a: Any, **k: Any) -> Any:
        if user_cancel:
            store.write_runtime("j1", cancelling=True)  # what `lab cancel` writes first
        raise runner_mod.SupervisorTerminated(signal.SIGTERM)

    monkeypatch.setattr(runner_mod, "provision_with_watchdog", _signalled)
    with pytest.raises(SystemExit):
        runner_mod.run_job(store.job_dir("j1"))
    return store


def test_a_user_cancel_closes_the_supervisor_call_as_a_non_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _sigterm_mid_provision(tmp_path, monkeypatch, user_cancel=True)

    call = _supervisor_call()
    assert not call.failed, f"a user cancel was recorded as {call.outcome}: {call.error}"
    assert call.result.get("cancelled_by_user") is True
    assert call.result.get("signal") == "SIGTERM"
    assert store.read_manifest("j1").status is JobState.cancelled


def test_a_sigterm_that_is_not_a_cancel_is_still_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _sigterm_mid_provision(tmp_path, monkeypatch, user_cancel=False)

    call = _supervisor_call()
    assert call.failed and call.outcome == "interrupted"
    assert "SIGTERM" in str(call.error)
    assert store.read_manifest("j1").status is JobState.failed
