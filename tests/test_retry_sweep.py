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


def test_retry_sweep_no_duplicate_when_prior_retry_in_flight(tmp_path: Path):
    """A second retry_sweep call must not resubmit a shard already covered by an in-flight retry."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]

    # Succeed shard [0,1]; fail shard [2,3]
    _succeed(lab, cell.shard_job_ids[0], [0, 1])
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.failed)
    lab.aggregate_sweep(sweep_id)

    # First retry — should add exactly one new job for seeds [2,3]
    updated = lab.retry_sweep(sweep_id)
    c = updated.cells[0]
    assert len(c.shard_job_ids) == 3  # 2 original + 1 retry

    # Identify the new retry job
    new_id = c.shard_job_ids[2]
    assert "seeds=2,3" in lab.manifest(new_id).run.entrypoint_command

    # Simulate the retry job still being in-flight (non-terminal: running)
    lab.store.update_manifest(new_id, status=JobState.running)

    # Second retry — the in-flight job covers seeds [2,3], so NO additional job should be added
    updated2 = lab.retry_sweep(sweep_id)
    c2 = updated2.cells[0]
    assert len(c2.shard_job_ids) == 3  # unchanged — no new job was added
