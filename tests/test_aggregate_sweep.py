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
    assert agg == (
        "seed,acc,_shard_status\n0,0.0,succeeded\n1,0.1,succeeded\n"
        "2,0.2,succeeded\n3,0.3,succeeded\n"
    )


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
    assert Path(c.aggregate_ref).read_text() == (
        "seed,acc,_shard_status\n0,0.0,succeeded\n1,0.1,succeeded\n"
    )


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


def _write_partial_result(lab: Lab, job_id: str, seeds: list[int], status: JobState) -> None:
    """A shard that reached a non-succeeded terminal state but left rows on disk
    (heartbeat-rsynced partial output)."""
    out = lab.store.output_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["seed,acc"] + [f"{s},0.{s}" for s in seeds]
    (out / "results.csv").write_text("\n".join(lines) + "\n")
    lab.store.update_manifest(job_id, status=status)


def test_aggregate_includes_timed_out_shard_rows_by_default(tmp_path: Path):
    """Field-report #2: rows the user already paid for must not be discarded."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    _write_partial_result(lab, cell.shard_job_ids[1], [2], JobState.timed_out)  # 1 of 2 seeds

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.seeds_present == [0, 1, 2]
    assert c.seeds_partial == [2]
    assert c.missing_seeds == [3]
    assert c.status == "incomplete"
    agg = Path(c.aggregate_ref).read_text()
    assert "2,0.2,timed_out" in agg and "0,0.0,succeeded" in agg


def test_aggregate_strict_excludes_partial_shards(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    _write_partial_result(lab, cell.shard_job_ids[1], [2], JobState.timed_out)

    c = lab.aggregate_sweep(sweep_id, include_partial=False).cells[0]
    assert c.seeds_present == [0, 1]
    assert c.seeds_partial == []
    assert c.missing_seeds == [2, 3]


def test_aggregate_excludes_running_shard(tmp_path: Path):
    """A running shard's file is still moving under the heartbeat — never read it."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-1", shard_size=1)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0])
    out = lab.store.output_dir(cell.shard_job_ids[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.csv").write_text("seed,acc\n1,0.1\n")
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.running)

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.seeds_present == [0] and c.missing_seeds == [1]


def test_aggregate_excludes_unconsumed_config_shard(tmp_path: Path):
    """Cross-fix guard via the REAL path: a timed-out shard whose effective_config.json shows
    it never consumed a passed override has its rows excluded (wrong-config rows are what the
    fail-closed check exists to kill; the audit records on every terminal transition)."""
    import json as _json

    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true N=1000", {}, seeds="0-1", shard_size=1)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0])
    # shard 2 wrote rows + an effective config that consumed only `seeds` — the argv-passed
    # `N` went unconsumed. It then timed out; the terminal audit records unconsumed_config.
    out = lab.store.output_dir(cell.shard_job_ids[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.csv").write_text("seed,acc\n1,0.1\n")
    (out / "effective_config.json").write_text(_json.dumps({"seeds": "1"}))
    m = lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.timed_out)
    assert m.unconsumed_config == ["N"]  # recorded without flipping the timed_out verdict
    assert m.status is JobState.timed_out

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.seeds_present == [0] and c.missing_seeds == [1]


def test_old_plan_json_reads_with_default_seeds_partial(tmp_path: Path):
    """plan.json written before seeds_partial existed must still read (backward compat)."""
    import json as _json

    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-1", shard_size=1)
    p = lab.store.sweep_plan_path(sweep_id)
    raw = _json.loads(p.read_text())
    for c in raw["cells"]:
        c.pop("seeds_partial", None)
    p.write_text(_json.dumps(raw))
    plan = lab.sweep_plan(sweep_id)
    assert plan.cells[0].seeds_partial == []


def test_sweep_row_key_flows_to_aggregation(tmp_path: Path):
    """--row-key seed,alpha: multi-row-per-seed shard results aggregate instead of raising
    (verification report 2026-08-06 §2 — the layout behind the hand-stitched headline data)."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep(
        "true", {"N": [1000]}, seeds="0-1", shard_size=1, row_key="seed,alpha"
    )
    plan = lab.sweep_plan(sweep_id)
    cell = plan.cells[0]
    assert cell.row_key == ["seed", "alpha"]
    for i, jid in enumerate(cell.shard_job_ids):
        out = lab.store.output_dir(jid)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.csv").write_text(
            f"seed,alpha,acc\n{i},2.7,0.9\n{i},2.72,0.8\n"
        )
        lab.store.update_manifest(jid, status=JobState.succeeded)

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.status == "complete"
    assert c.seeds_present == [0, 1]
    agg = Path(c.aggregate_ref).read_text()
    assert agg.count("\n") == 5  # header + 4 rows (2 seeds x 2 alphas)


def test_sweep_row_key_must_contain_seed_column(tmp_path: Path):
    lab = _lab(tmp_path)
    import pytest as _pytest

    from lab.core import LabError

    with _pytest.raises(LabError, match="seed"):
        lab.sweep("true", {"N": [1]}, seeds="0-1", shard_size=1, row_key="alpha")
