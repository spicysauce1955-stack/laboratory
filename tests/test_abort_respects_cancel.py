"""A supervisor stopped by ``lab cancel`` records ``cancelled``, not ``failed`` (review finding).

Since the R9 reordering, ``SkyPilotBackend.cancel`` keeps the manifest non-terminal until its own
teardown finishes: it writes ``cancelling`` to ``_runtime.json`` and SIGTERMs the supervisor
first. The supervisor's abort path ignored that flag and stamped the job ``failed`` ("supervisor
terminated by SIGTERM"). If the cancel caller was then killed — the very scenario the reordering
exists for — ``status()`` could no longer repair it, because the job was already terminal: a
deliberate cancel read as a supervisor crash, forever.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab.backends.skypilot import SkyPilotBackend
from lab.models import JobState
from lab.store import JobStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobStore:
    store = JobStore(tmp_path / "runs")
    store.create(make_manifest("j1", "python x.py").model_copy(update={"status": JobState.running}))
    store.write_runtime("j1", runner_pid=999999999, cluster="lab-j1")
    monkeypatch.setitem(sys.modules, "sky", types.ModuleType("sky"))
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)
    return store


def test_an_abort_during_a_cancel_records_cancelled(store: JobStore) -> None:
    """The reviewer's reproduction: what `cancel` writes, then the SIGTERM it sends."""
    store.write_runtime("j1", cancelling=True)

    runner_mod._teardown_on_abort(
        store, "j1", "lab-j1", "vast",
        why="supervisor terminated by SIGTERM", launched=True, torn_down=False,
    )

    after = store.read_manifest("j1")
    assert after.status is JobState.cancelled
    assert "cancel" in (after.end_reason or "")
    # ...and if the cancel caller dies now, recovery still reads it as the cancel it was.
    assert SkyPilotBackend(home=store.home).status("j1") is JobState.cancelled


def test_an_abort_with_no_cancel_in_flight_is_still_a_failure(store: JobStore) -> None:
    runner_mod._teardown_on_abort(
        store, "j1", "lab-j1", "vast",
        why="supervisor terminated by SIGTERM", launched=True, torn_down=False,
    )

    after = store.read_manifest("j1")
    assert after.status is JobState.failed
    assert after.end_reason == "supervisor terminated by SIGTERM"
