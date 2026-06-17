import sys
import types
from pathlib import Path

import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab._util import now
from lab.backends.skypilot import TIMEOUT_SENTINEL
from lab.models import CostInfo, JobState
from lab.store import JobStore


def test_remote_timeout_reason_carries_wall_and_tears_down(tmp_path: Path, monkeypatch):
    home = tmp_path / "runs"
    store = JobStore(home)
    m = make_manifest("t1", "python x.py", timeout="120s").model_copy(
        update={"status": JobState.running, "started_at": now(),
                "cost": CostInfo(hourly_usd=0.2, estimated_usd=0.2)}
    )
    store.create(m)
    store.write_runtime("t1", runner_pid=1, cluster="lab-t1")

    fake_sky = types.ModuleType("sky")
    monkeypatch.setitem(sys.modules, "sky", fake_sky)

    # The on-box `timeout` wrapper would have dropped this sentinel; promote_timeout uses it to
    # relabel the failed remote job as timed_out.
    output = store.output_dir("t1")
    output.mkdir(parents=True, exist_ok=True)
    (output / TIMEOUT_SENTINEL).write_text("1")

    monkeypatch.setattr(runner_mod, "_wait_terminal", lambda *a, **k: (JobState.failed, True))
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "_resolve_hourly", lambda *a, **k: 0.2)
    monkeypatch.setattr(runner_mod, "_cluster_up", lambda *a, **k: False)
    monkeypatch.setattr(runner_mod, "r2_enabled", lambda: False)

    teardown_calls = {"n": 0}

    def _spy(sky_mod, cluster, st, jid, cloud="vast"):
        teardown_calls["n"] += 1
        st.update_manifest(jid, teardown_status="succeeded")
        return True

    monkeypatch.setattr(runner_mod, "tear_down_and_record", _spy)

    rc = runner_mod.run_job(home / "t1", adopt=True)

    final = store.read_manifest("t1")
    assert final.status is JobState.timed_out
    assert final.end_reason == "timed out after 120s wall-clock cap"
    assert final.teardown_status == "succeeded"
    assert teardown_calls["n"] == 1
    assert rc == 0
