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


def test_retry_resubmits_only_missing_seeds_of_partial_shard(tmp_path):
    """A shard that timed out with some seeds recovered is retried NARROWED to its missing
    seeds — not re-run whole (which would waste money and duplicate recovered rows)."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _succeed(lab, cell.shard_job_ids[0], [0, 1])  # shard 1 complete
    # shard 2 ([2,3]) timed out having recovered only seed 2
    out = lab.store.output_dir(cell.shard_job_ids[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.csv").write_text("seed,acc\n2,0.2\n")
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.timed_out)

    plan = lab.retry_sweep(sweep_id)
    cell = plan.cells[0]
    assert len(cell.shard_job_ids) == 3  # one narrowed retry appended
    retry_cmd = lab.manifest(cell.shard_job_ids[2]).run.entrypoint_command
    assert "seeds=3" in retry_cmd and "seeds=2,3" not in retry_cmd


def test_retry_skips_when_inflight_superset_covers_missing(tmp_path):
    """An in-flight full-shard retry (seeds=2,3) suppresses a would-be narrowed retry (3)."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _succeed(lab, cell.shard_job_ids[0], [0, 1])
    # shard 2 ([2,3]) times out with NO recovered rows -> first retry re-runs the full shard
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.timed_out)
    first = lab.retry_sweep(sweep_id)
    retry_jid = first.cells[0].shard_job_ids[-1]
    assert "seeds=2,3" in lab.manifest(retry_jid).run.entrypoint_command
    lab.store.update_manifest(retry_jid, status=JobState.running)  # pin the retry in flight
    # seed 2 now surfaces from the timed-out shard's late-rsynced partial output
    out = lab.store.output_dir(cell.shard_job_ids[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.csv").write_text("seed,acc\n2,0.2\n")

    n_before = len(first.cells[0].shard_job_ids)
    second = lab.retry_sweep(sweep_id)
    # missing is now just [3]; the in-flight {2,3} retry is a superset -> no duplicate submit
    assert second.cells[0].missing_seeds == [3]
    assert len(second.cells[0].shard_job_ids) == n_before
