"""--adopt re-attaches to a live cluster: no sky.launch, waits, rsyncs, tears down."""

import sys
import types
from pathlib import Path

import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab._util import now
from lab.backends.skypilot import SUCCESS_SENTINEL
from lab.models import CostInfo, JobState
from lab.store import JobStore


def test_adopt_skips_launch_and_finishes(tmp_path: Path, monkeypatch):
    home = tmp_path / "runs"
    store = JobStore(home)
    m = make_manifest("j1", "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running, "started_at": now(),
                "cost": CostInfo(hourly_usd=0.2, estimated_usd=0.2)}
    )
    store.create(m)
    store.write_runtime("j1", runner_pid=1, cluster="lab-j1")

    fake_sky = types.ModuleType("sky")
    monkeypatch.setitem(sys.modules, "sky", fake_sky)

    launched = []
    monkeypatch.setattr(
        fake_sky, "launch", lambda *a, **k: launched.append(1), raising=False
    )
    monkeypatch.setattr(runner_mod, "_wait_terminal",
                        lambda *a, **k: JobState.succeeded)
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: 0.2)

    # Simulate a clean remote exit: write the success sentinel that the run script would drop.
    output = store.output_dir("j1")
    output.mkdir(parents=True, exist_ok=True)
    (output / SUCCESS_SENTINEL).write_text("1")

    rc = runner_mod.run_job(home / "j1", adopt=True)
    assert rc == 0
    assert launched == []  # adopt never re-launches
    final = store.read_manifest("j1")
    assert final.status is JobState.succeeded
