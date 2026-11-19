"""An aborting supervisor cancels its in-flight launch request before tearing down (review finding).

``sky.launch`` returns a request id and the provisioning runs on the API server. When the
supervisor was signalled mid-provision, the abort path tore the cluster down but never
``sky.api_cancel``ed that request — so a launch still *queued* on the server could provision
afterwards, onto a cluster name nobody was supervising any more, and run unsupervised and
billing. The request id is now kept reachable and cancelled first.
"""

from __future__ import annotations

import signal
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab.models import JobState
from lab.store import JobStore


class _FakeSky(types.ModuleType):
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        super().__init__("sky")
        self.calls = calls

    def launch(self, task: Any, cluster_name: str, **kw: Any) -> str:
        return "req-1"

    def api_cancel(self, request_id: Any) -> None:
        self.calls.append(("api_cancel", request_id))


def test_a_signal_mid_provision_cancels_the_launch_then_tears_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / "runs")
    store.create(make_manifest("j1", "python x.py"))
    calls: list[tuple[str, Any]] = []
    monkeypatch.setitem(sys.modules, "sky", _FakeSky(calls))
    monkeypatch.setattr(runner_mod, "build_task", lambda *a, **k: "task")

    def _sigterm_while_provisioning(*a: Any, **k: Any) -> Any:
        raise runner_mod.SupervisorTerminated(signal.SIGTERM)

    def _teardown(sky_mod: Any, cluster: str, *a: Any, **k: Any) -> bool:
        calls.append(("teardown", cluster))
        return True

    monkeypatch.setattr(runner_mod, "provision_with_watchdog", _sigterm_while_provisioning)
    monkeypatch.setattr(runner_mod, "tear_down_and_record", _teardown)

    with pytest.raises(SystemExit):
        runner_mod.run_job(store.job_dir("j1"))

    assert [c[0] for c in calls] == ["api_cancel", "teardown"]
    assert calls[0] == ("api_cancel", "req-1")
    assert store.read_manifest("j1").status is JobState.failed


def test_a_failing_api_cancel_never_skips_the_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / "runs")
    store.create(make_manifest("j1", "python x.py").model_copy(update={"status": JobState.running}))
    sky = types.ModuleType("sky")

    def _api_cancel(request_id: Any) -> None:
        raise RuntimeError("API server unreachable")

    sky.api_cancel = _api_cancel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky", sky)
    torn: list[str] = []
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda s, c, *a, **k: torn.append(c))

    runner_mod._teardown_on_abort(
        store, "j1", "lab-j1", "vast",
        why="supervisor terminated by SIGTERM", launched=True, torn_down=False,
        request_id="req-9",
    )

    assert torn == ["lab-j1"]
