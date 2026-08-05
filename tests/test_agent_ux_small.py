"""Field-report #7 small items: seeds-only sweeps, mid-run cost estimate, status heartbeat."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

import lab.cli as cli_mod
from helpers import make_manifest
from lab._util import now
from lab.cli import app
from lab.core import job_status_view
from lab.models import CostInfo, JobState
from lab.store import JobStore


# ---------------------------------------------------------------------------
# --grid optional when --seeds present
# ---------------------------------------------------------------------------


def _fake_sweep_lab(seen: dict) -> MagicMock:
    fake = MagicMock()

    def _sweep(cmd, grid, **kw):
        seen["grid"] = grid
        seen["kw"] = kw
        return ("sweep-1", ["j1"])

    fake.sweep.side_effect = _sweep
    fake.store.has_sweep_plan.return_value = False
    return fake


def test_cli_sweep_seeds_only_no_grid():
    seen: dict = {}
    with patch.object(cli_mod, "_lab", return_value=_fake_sweep_lab(seen)):
        result = CliRunner().invoke(
            app, ["sweep", "-c", "python x.py", "--seeds", "0-1", "--shard-size", "1"]
        )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert seen["grid"] == {}
    assert seen["kw"]["seeds"] == "0-1"


def test_cli_sweep_requires_grid_or_seeds():
    # The guard lives in Lab.sweep (thin-shell rule: one implementation for CLI+MCP);
    # here the real core raises LabError before anything is submitted.
    result = CliRunner().invoke(app, ["sweep", "-c", "python x.py"])
    assert result.exit_code == 1
    assert "--grid" in result.output and "--seeds" in result.output


def test_core_sweep_requires_grid_or_seeds(tmp_path):
    import pytest as _pytest

    from lab.backends.local import LocalBackend
    from lab.core import Lab, LabError

    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
    with _pytest.raises(LabError, match="at least one axis"):
        lab.sweep("python x.py", {})


# ---------------------------------------------------------------------------
# estimated_running_usd + heartbeat in status
# ---------------------------------------------------------------------------


def test_status_estimated_running_usd_for_running_job(tmp_path):
    store = JobStore(tmp_path)
    m = make_manifest("jr1", "python x.py").model_copy(
        update={
            "status": JobState.running,
            "started_at": now() - timedelta(minutes=30),
            "cost": CostInfo(hourly_usd=0.2, estimated_usd=0.4),
        }
    )
    store.create(m)
    view = job_status_view(tmp_path, tmp_path, "jr1")
    assert view["state"] == "running"
    assert 0.09 <= view["estimated_running_usd"] <= 0.11  # ~0.5h * $0.2/h


def test_status_estimated_running_usd_none_when_not_running(tmp_path):
    store = JobStore(tmp_path)
    m = make_manifest("jr2", "python x.py").model_copy(
        update={
            "status": JobState.succeeded,
            "started_at": now() - timedelta(minutes=30),
            "cost": CostInfo(hourly_usd=0.2, actual_usd=0.1),
        }
    )
    store.create(m)
    view = job_status_view(tmp_path, tmp_path, "jr2")
    assert view["estimated_running_usd"] is None


def test_status_heartbeat_surfaces_last_log_line(tmp_path):
    store = JobStore(tmp_path)
    m = make_manifest("jh1", "python x.py").model_copy(update={"status": JobState.running})
    store.create(m)
    store.logs_path("jh1").write_text("starting\nepoch 3/10 loss=0.42\n")
    view = job_status_view(tmp_path, tmp_path, "jh1")
    assert view["last_log_line"] == "epoch 3/10 loss=0.42"
    assert view["last_log_at"] is not None


def test_status_heartbeat_none_for_mirrored(tmp_path, monkeypatch):
    from lab.scheduler.queue import LocalQueueStore

    qdir = tmp_path / "queue"
    q = LocalQueueStore(qdir)
    m = make_manifest("jh2", "python x.py").model_copy(update={"status": JobState.running})
    q.mirror_manifest(m)
    monkeypatch.setenv("LAB_QUEUE_DIR", str(qdir))
    view = job_status_view(tmp_path, tmp_path, "jh2")
    assert view["mirrored"] is True
    assert view["last_log_line"] is None and view["last_log_at"] is None
