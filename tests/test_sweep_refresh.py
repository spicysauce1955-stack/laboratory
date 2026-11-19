"""Sweep readers refresh non-terminal shards through their backend (dead supervisors finalize),
and sweep-retry resubmits through the backend that ran the sweep."""

from __future__ import annotations

import os
from pathlib import Path

from core_fixtures import local_lab, make_repo
from helpers import wait_terminal

from lab._util import process_start_time
from lab.models import JobState


def _sweep_with_dead_shard(tmp_path: Path, monkeypatch):
    """A 2-shard sweep: shard 0 succeeded with rows, shard 1's supervisor died at `running`."""
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    sweep_id, _ = lab.sweep("true", {}, seeds="0-1", shard_size=1)
    a, b = lab.sweep_plan(sweep_id).cells[0].shard_job_ids
    for jid in (a, b):
        wait_terminal(lab.backend, jid)
    (lab.store.output_dir(a) / "results.csv").write_text("seed,acc\n0,0.5\n")
    # Frozen at `running` with a runner that is gone: a recorded pid whose identity no longer
    # matches (our own pid, with a start time it never had) reads as dead.
    lab.store.write_manifest(
        lab.manifest(b).model_copy(
            update={"status": JobState.running, "ended_at": None, "end_reason": None}
        )
    )
    lab.store.write_runtime(
        b, runner_pid=os.getpid(), runner_start_time=(process_start_time(os.getpid()) or 0) + 7
    )
    return lab, sweep_id, a, b


def test_aggregate_finalizes_a_dead_shard(tmp_path: Path, monkeypatch):
    lab, sweep_id, _, b = _sweep_with_dead_shard(tmp_path, monkeypatch)
    lab.aggregate_sweep(sweep_id)
    m = lab.manifest(b)
    assert m.status is JobState.failed
    assert m.end_reason == "runner exited without recording status"


def test_retry_resubmits_a_shard_whose_supervisor_died(tmp_path: Path, monkeypatch):
    lab, sweep_id, a, b = _sweep_with_dead_shard(tmp_path, monkeypatch)
    plan = lab.retry_sweep(sweep_id)
    ids = plan.cells[0].shard_job_ids
    assert len(ids) == 3, "the dead shard's seed must be resubmitted, not treated as in flight"
    assert "seeds=1" in lab.manifest(ids[2]).run.entrypoint_command


def test_sweep_summary_finalizes_a_dead_shard(tmp_path: Path, monkeypatch):
    lab, sweep_id, _, b = _sweep_with_dead_shard(tmp_path, monkeypatch)
    s = lab.sweep_summary(sweep_id)
    assert s["per_point"][b]["state"] == "failed"
    assert s["failed"] == 1 and s["succeeded"] == 1
