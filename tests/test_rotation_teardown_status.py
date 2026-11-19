"""After an accelerator rotation, ``teardown_status`` must not describe the abandoned attempt.

Review finding: rotation destroys attempt N (``tear_down_and_record`` writes
``teardown_status="succeeded"``) and launches attempt N+1 under the same cluster name. Nothing
reset the field, so between the supervisor writing the terminal status and its final teardown
the manifest read as a confirmed all-clear for a machine that was still billing. A supervisor
SIGKILLed in that window (OOM, ``kill -9``, a systemd stop timeout) left ``lab wait`` exiting 0
behind ``teardown_status: succeeded`` instead of the "teardown not confirmed" warning a
non-rotated job gets there. The field is cleared when a new launch attempt starts.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import lab.placement as P
import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab.backends.local import LocalBackend
from lab.backends.skypilot import ProvisionTimeout
from lab.core import Lab
from lab.models import BackendInfo, JobManifest, JobState, ResourceRequest


class _FakeSky(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("sky")
        self.launches: list[str] = []

    def launch(self, task: Any, cluster_name: str, **kw: Any) -> str:
        self.launches.append(cluster_name)
        return f"req-{len(self.launches)}"

    def api_cancel(self, request_id: Any) -> None:
        pass

    def stream_and_get(self, request_id: Any) -> tuple[int, Any]:
        return 1, types.SimpleNamespace(launched_resources=None)

    def tail_logs(self, *a: Any, **k: Any) -> None:
        return None

    def get(self, x: Any) -> Any:
        return x

    def queue(self, cluster: str, skip_finished: bool = False) -> Any:
        return [{"job_id": 1, "status": types.SimpleNamespace(name="SUCCEEDED")}]

    def status(self, *a: Any, **k: Any) -> Any:
        return []


def test_a_rotated_job_does_not_read_as_torn_down_before_its_final_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runs"
    lab = Lab(backend=LocalBackend(home=home, repo=tmp_path), repo=tmp_path, home=home)
    res = ResourceRequest(accelerators="RTX4090:1", accelerator_pool="RTX3090:1")
    lab.store.create(
        make_manifest("rot", "python x.py", resources=res).model_copy(
            update={"backend": BackendInfo(provisioner="skypilot")}
        )
    )

    sky = _FakeSky()
    provisions = {"n": 0}

    def _provision(sky_mod: Any, request_id: Any, *, timeout_s: float) -> tuple[Any, Any]:
        provisions["n"] += 1
        if provisions["n"] == 1:
            raise ProvisionTimeout("attempt 1 never came up")
        return sky_mod.stream_and_get(request_id)

    # What `lab wait` would see if the supervisor died the instant its final teardown began.
    seen_at_final_teardown: list[JobManifest] = []

    def _teardown(sky_mod: Any, cluster: str, st: Any, jid: str, cloud: str = "vast", **k: Any):
        current = st.read_manifest(jid)
        if current.status is JobState.succeeded:
            seen_at_final_teardown.append(current)
        st.update_manifest(jid, teardown_status="succeeded")
        return True

    monkeypatch.setitem(sys.modules, "sky", sky)
    monkeypatch.setattr(runner_mod, "build_task", lambda *a, **k: "task")
    monkeypatch.setattr(runner_mod, "provision_with_watchdog", _provision)
    monkeypatch.setattr(runner_mod, "tear_down_and_record", _teardown)
    monkeypatch.setattr(runner_mod, "resolve_cost", lambda *a, **k: runner_mod.CostInfo())
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "r2_enabled", lambda: False)
    monkeypatch.setattr(runner_mod, "confirm_success", lambda final, out: final)
    monkeypatch.setattr(runner_mod, "vast_machine_id", lambda cluster, client=None: None)
    monkeypatch.setattr(P, "affordable_under_cap", lambda r, a: True)

    rc = runner_mod.run_job(lab.store.job_dir("rot"))

    assert rc == 0 and len(sky.launches) == 2, "the job did not rotate; the test proves nothing"
    assert len(seen_at_final_teardown) == 1
    snapshot = seen_at_final_teardown[0]
    assert snapshot.teardown_status is None, (
        "the abandoned attempt's teardown was still standing in for the live machine's"
    )
    summary = lab._wait_summary_dict([snapshot], failed_fast=False)
    assert summary["teardown_unconfirmed"] == ["rot"]
    # ...and once the real teardown lands, the field says so.
    assert lab.store.read_manifest("rot").teardown_status == "succeeded"
