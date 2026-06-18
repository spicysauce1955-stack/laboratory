from __future__ import annotations

from pathlib import Path

import pytest

from lab.backends.local import LocalBackend
from lab.core import Lab, LabError
from lab.manifest import repo_root
from lab.store import cell_id_for


def _lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    return Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)


def test_sweep_no_seeds_writes_no_plan(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, job_ids = lab.sweep("true", {"N": [1000]})
    assert len(job_ids) == 1
    assert not lab.store.has_sweep_plan(sweep_id)


def test_sharded_sweep_partitions_and_plans(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, job_ids = lab.sweep("true", {"N": [1000]}, seeds="0-7", shard_size=2)
    # one cell, 4 shards of 2 seeds each
    assert len(job_ids) == 4
    plan = lab.sweep_plan(sweep_id)
    assert len(plan.cells) == 1
    cell = plan.cells[0]
    assert cell.coords == {"N": "1000"}
    assert cell.seeds_expected == [0, 1, 2, 3, 4, 5, 6, 7]
    assert cell.shard_seeds == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert len(cell.shard_job_ids) == 4
    assert cell.status == "pending"
    # each shard's command carries its seed subset under the seed-axis key
    cmds = [lab.manifest(j).run.entrypoint_command for j in cell.shard_job_ids]
    assert any("seeds=0,1" in c for c in cmds)
    assert any("seeds=6,7" in c for c in cmds)
    # each shard records its cell_id; the per-job singular seed anchors to the shard's first seed
    assert all(lab.manifest(j).cell_id == cell.cell_id for j in cell.shard_job_ids)
    # invariant: cell_id_for(coords) must equal the stored cell_id (fix: coords coerced before hashing)
    assert cell_id_for(cell.coords) == cell.cell_id


def test_seeds_in_both_axis_and_grid_is_rejected(tmp_path: Path):
    lab = _lab(tmp_path)
    with pytest.raises(LabError, match="both"):
        lab.sweep("true", {"seeds": [0, 1]}, seeds="0-7", shard_size=2)


def test_shard_size_ge_len_is_one_shard_per_cell(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, job_ids = lab.sweep("true", {"N": [1000, 1500]}, seeds="0-3", shard_size=8)
    assert len(job_ids) == 2  # one shard per cell
    plan = lab.sweep_plan(sweep_id)
    assert all(len(c.shard_seeds) == 1 for c in plan.cells)


def test_bad_seed_range_raises_lab_error(tmp_path: Path):
    lab = _lab(tmp_path)
    with pytest.raises(LabError):
        lab.sweep("true", {"N": [1000]}, seeds="5-2", shard_size=2)


def test_zero_shard_size_raises_lab_error(tmp_path: Path):
    lab = _lab(tmp_path)
    with pytest.raises(LabError):
        lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=0)


def test_singular_seed_grid_key_rejected(tmp_path: Path):
    lab = _lab(tmp_path)
    with pytest.raises(LabError, match="both"):
        lab.sweep("true", {"seed": [0, 1]}, seeds="0-3", shard_size=2)
