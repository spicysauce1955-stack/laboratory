from __future__ import annotations

from pathlib import Path

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import JobState


def _lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    return Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)


def _succeed(lab: Lab, job_id: str, seeds: list[int]) -> None:
    out = lab.store.output_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.csv").write_text("seed,acc\n" + "".join(f"{s},0.{s}\n" for s in seeds))
    lab.store.update_manifest(job_id, status=JobState.succeeded)


def test_retry_resubmits_only_missing_shards(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _succeed(lab, cell.shard_job_ids[0], [0, 1])
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.failed)
    lab.aggregate_sweep(sweep_id)

    before_ids = set(lab.sweep_plan(sweep_id).cells[0].shard_job_ids)
    updated = lab.retry_sweep(sweep_id)
    c = updated.cells[0]
    new_ids = [j for j in c.shard_job_ids if j not in before_ids]
    assert len(new_ids) == 1  # only the missing shard (seeds 2,3) was resubmitted
    assert "seeds=2,3" in lab.manifest(new_ids[0]).run.entrypoint_command
    assert lab.manifest(new_ids[0]).cell_id == c.cell_id
