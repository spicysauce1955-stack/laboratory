"""Tests for --spot / --no-fallback CLI options on submit, sweep, and register."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

import lab.cli as cli_mod
from lab.cli import app
from lab.models import JobSpec, ResourceRequest

runner = CliRunner()


def _make_fake_lab(submitted_specs: list[JobSpec]) -> MagicMock:
    """Return a fake Lab that captures submitted JobSpec objects."""
    fake = MagicMock()
    fake.find_cached.return_value = None
    fake.submit.side_effect = lambda spec: (submitted_specs.append(spec) or "job-test-1")
    fake.status.return_value = MagicMock(value="queued")
    return fake


def test_submit_spot_no_fallback():
    """--spot --no-fallback sets use_spot=True and spot_fallback=False on the ResourceRequest."""
    captured: list[JobSpec] = []
    fake_lab = _make_fake_lab(captured)

    with patch.object(cli_mod, "_lab", return_value=fake_lab):
        result = runner.invoke(
            app,
            ["submit", "--command", "echo hi", "--backend", "skypilot", "--spot", "--no-fallback"],
        )

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert len(captured) == 1
    res: ResourceRequest = captured[0].resources
    assert res.use_spot is True
    assert res.spot_fallback is False


def test_submit_spot_with_fallback():
    """--spot alone keeps spot_fallback=True (the default)."""
    captured: list[JobSpec] = []
    fake_lab = _make_fake_lab(captured)

    with patch.object(cli_mod, "_lab", return_value=fake_lab):
        result = runner.invoke(
            app,
            ["submit", "--command", "echo hi", "--backend", "skypilot", "--spot"],
        )

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert len(captured) == 1
    res: ResourceRequest = captured[0].resources
    assert res.use_spot is True
    assert res.spot_fallback is True


def test_submit_no_spot_defaults():
    """Without --spot, use_spot=False and spot_fallback=True (defaults unchanged)."""
    captured: list[JobSpec] = []
    fake_lab = _make_fake_lab(captured)

    with patch.object(cli_mod, "_lab", return_value=fake_lab):
        result = runner.invoke(app, ["submit", "--command", "echo hi"])

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert len(captured) == 1
    res: ResourceRequest = captured[0].resources
    assert res.use_spot is False
    assert res.spot_fallback is True


def test_sweep_spot_no_fallback():
    """lab sweep --spot --no-fallback passes use_spot=True and spot_fallback=False."""
    resources_seen: list[ResourceRequest] = []

    fake_lab = MagicMock()
    fake_lab.sweep.side_effect = lambda cmd, grid, seed=None, resources=None: (
        resources_seen.append(resources) or ("sweep-1", ["job-1", "job-2"])
    )

    with patch.object(cli_mod, "_lab", return_value=fake_lab):
        result = runner.invoke(
            app,
            [
                "sweep",
                "--command", "echo hi",
                "--grid", "lr=0.1,0.01",
                "--spot",
                "--no-fallback",
            ],
        )

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert len(resources_seen) == 1
    res = resources_seen[0]
    assert res.use_spot is True
    assert res.spot_fallback is False


def test_register_spot_no_fallback(tmp_path: Path):
    """lab register --spot --no-fallback stores use_spot=True and spot_fallback=False."""
    from test_scheduler_bundle import _make_repo

    repo = _make_repo(tmp_path)
    env = {"LAB_QUEUE_DIR": str(tmp_path / "queue"), "LAB_REPO_DIR": str(repo)}

    result = runner.invoke(
        app,
        [
            "register",
            "--command", "python exp.py",
            "--timeout", "1h",
            "--expires", "+3d",
            "--accelerators", "RTX_4090:1",
            "--spot",
            "--no-fallback",
        ],
        env=env,
    )

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    out = json.loads(result.output)
    reg_id = out["reg_id"]

    from lab.scheduler.queue import LocalQueueStore

    q = LocalQueueStore(tmp_path / "queue")
    reg = q.get_entry(reg_id)
    assert reg.spec.resources.use_spot is True
    assert reg.spec.resources.spot_fallback is False
