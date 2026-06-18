"""Tests for sharded sweep CLI surface (Task 7, P1-2)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

import lab.cli as cli_mod
from lab.cli import app
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import JobState

runner = CliRunner()


def _real_lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    return Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)


def test_sweep_sharded_emits_cells(tmp_path: Path) -> None:
    with patch.object(cli_mod, "_lab", return_value=_real_lab(tmp_path)):
        res = runner.invoke(
            app,
            ["sweep", "-c", "true", "-g", "N=1000", "--seeds", "0-3", "--shard-size", "2"],
        )
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["sweep_id"]
    assert len(out["cells"]) == 1
    assert out["cells"][0]["seeds_expected"] == 4
    assert len(out["cells"][0]["shard_job_ids"]) == 2


def test_sweep_unsharded_unchanged(tmp_path: Path) -> None:
    with patch.object(cli_mod, "_lab", return_value=_real_lab(tmp_path)):
        res = runner.invoke(app, ["sweep", "-c", "true", "-g", "N=1000,1500"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["count"] == 2
    assert "cells" not in out


def _write_shard_result(lab: Lab, job_id: str, seeds: list[int]) -> None:
    """Mark a shard succeeded and write a results.csv with one row per seed."""
    out = lab.store.output_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["seed,acc"] + [f"{s},0.{s}" for s in seeds]
    (out / "results.csv").write_text("\n".join(lines) + "\n")
    lab.store.update_manifest(job_id, status=JobState.succeeded)


def test_sweep_aggregate_emits_cells(tmp_path: Path) -> None:
    lab = _real_lab(tmp_path)
    # Create the sharded sweep via the CLI, reusing the same lab instance.
    with patch.object(cli_mod, "_lab", return_value=lab):
        res = runner.invoke(
            app,
            ["sweep", "-c", "true", "-g", "N=1000", "--seeds", "0-3", "--shard-size", "2"],
        )
    assert res.exit_code == 0, res.output
    sweep_id = json.loads(res.output)["sweep_id"]

    # Mark both shards succeeded with results.
    plan = lab.sweep_plan(sweep_id)
    cell = plan.cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    _write_shard_result(lab, cell.shard_job_ids[1], [2, 3])

    # Invoke sweep-aggregate against the same lab.
    with patch.object(cli_mod, "_lab", return_value=lab):
        res = runner.invoke(app, ["sweep-aggregate", sweep_id])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["sweep_id"] == sweep_id
    assert len(out["cells"]) == 1
    c = out["cells"][0]
    assert c["status"] == "complete"
    assert c["seeds_present"] == 4


def test_sweep_retry_emits_cells(tmp_path: Path) -> None:
    lab = _real_lab(tmp_path)
    # Create the sharded sweep via the CLI, reusing the same lab instance.
    with patch.object(cli_mod, "_lab", return_value=lab):
        res = runner.invoke(
            app,
            ["sweep", "-c", "true", "-g", "N=1000", "--seeds", "0-3", "--shard-size", "2"],
        )
    assert res.exit_code == 0, res.output
    sweep_id = json.loads(res.output)["sweep_id"]

    # Succeed one shard, fail the other.
    plan = lab.sweep_plan(sweep_id)
    cell = plan.cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.failed)

    # First aggregate so that missing_seeds are computed.
    lab.aggregate_sweep(sweep_id)

    # Invoke sweep-retry against the same lab.
    with patch.object(cli_mod, "_lab", return_value=lab):
        res = runner.invoke(app, ["sweep-retry", sweep_id])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["sweep_id"] == sweep_id
    assert "cells" in out
    assert len(out["cells"]) == 1
