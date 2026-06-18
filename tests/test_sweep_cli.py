"""Tests for sharded sweep CLI surface (Task 7, P1-2)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import lab.cli as cli_mod
from lab.cli import app
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root

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
