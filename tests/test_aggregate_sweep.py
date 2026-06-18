# tests/test_aggregate_sweep.py
from __future__ import annotations

from pathlib import Path

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import JobState


def _lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    return Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)


def _write_shard_result(lab: Lab, job_id: str, seeds: list[int]) -> None:
    """Mark a shard succeeded and drop a results.csv with one row per seed."""
    out = lab.store.output_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["seed,acc"] + [f"{s},0.{s}" for s in seeds]
    (out / "results.csv").write_text("\n".join(lines) + "\n")
    lab.store.update_manifest(job_id, status=JobState.succeeded)


def test_aggregate_complete_cell(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    plan = lab.sweep_plan(sweep_id)
    cell = plan.cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    _write_shard_result(lab, cell.shard_job_ids[1], [2, 3])

    updated = lab.aggregate_sweep(sweep_id)
    c = updated.cells[0]
    assert c.status == "complete"
    assert c.seeds_present == [0, 1, 2, 3]
    assert c.missing_seeds == []
    agg = Path(c.aggregate_ref).read_text()
    assert agg == "seed,acc\n0,0.0\n1,0.1\n2,0.2\n3,0.3\n"


def test_aggregate_partial_failure_is_honest(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    plan = lab.sweep_plan(sweep_id)
    cell = plan.cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.failed)  # shard 2 dies

    updated = lab.aggregate_sweep(sweep_id)
    c = updated.cells[0]
    assert c.status == "incomplete"
    assert c.seeds_present == [0, 1]
    assert c.missing_seeds == [2, 3]
    assert Path(c.aggregate_ref).read_text() == "seed,acc\n0,0.0\n1,0.1\n"


def test_aggregate_is_idempotent_and_resumable(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    first = lab.aggregate_sweep(sweep_id).cells[0]
    assert first.status == "incomplete" and first.missing_seeds == [2, 3]
    _write_shard_result(lab, cell.shard_job_ids[1], [2, 3])  # second shard finishes later
    second = lab.aggregate_sweep(sweep_id).cells[0]
    assert second.status == "complete" and second.seeds_present == [0, 1, 2, 3]
