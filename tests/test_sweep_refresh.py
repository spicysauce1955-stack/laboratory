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


class _FakeRemote:
    """Stands in for SkyPilotBackend: records what was launched, touches no cloud."""

    name = "skypilot"

    def __init__(self, home: Path) -> None:
        from lab.store import JobStore

        self.store = JobStore(home)
        self.submitted: list[str] = []

    def submit(self, manifest):
        self.submitted.append(manifest.job_id)
        return manifest.job_id

    def status(self, job_id: str) -> JobState:
        return self.store.read_manifest(job_id).status


def test_retry_resubmits_through_the_sweeps_own_backend(tmp_path: Path, monkeypatch):
    from helpers import make_manifest

    import lab.core as core_mod
    from lab._util import now
    from lab.models import BackendInfo, ResourceRequest, SweepCell, SweepPlan

    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)  # what the CLI/MCP `sweep-retry` shells build: a local-backend Lab
    fake = _FakeRemote(lab.home)
    monkeypatch.setattr(
        core_mod, "build_backend",
        lambda name, **kw: fake if name == "skypilot" else core_mod.LocalBackend(**kw),
    )
    monkeypatch.setattr(core_mod, "r2_enabled", lambda: False)
    monkeypatch.setattr("lab.doctor.preflight", lambda *a, **k: [])
    monkeypatch.setenv("LAB_SUBMIT_STAGGER_S", "0")

    res = ResourceRequest(cloud="do", cpus=4, disk_size=50, use_spot=False, spot_fallback=False)
    shard = make_manifest("shard-0", "python x.py seeds=0").model_copy(
        update={
            "status": JobState.failed,
            "sweep_id": "sweep-r",
            "cell_id": "c1",
            "resources": res,
            "backend": BackendInfo(provisioner="skypilot"),
        }
    )
    lab.store.create(shard)
    lab.store.write_sweep_plan(
        SweepPlan(
            sweep_id="sweep-r", created_at=now(), command="python x.py", seed_axis_key="seeds",
            cells=[SweepCell(
                coords={}, cell_id="c1", seeds_expected=[0], shard_seeds=[[0]],
                shard_job_ids=["shard-0"], results_file="results.csv", seed_column="seed",
                aggregate_ref=str(lab.home / "sweep-r" / "cells" / "c1" / "results.csv"),
            )],
        )
    )

    plan = lab.retry_sweep("sweep-r")

    new_id = plan.cells[0].shard_job_ids[-1]
    assert new_id != "shard-0"
    assert fake.submitted == [new_id], "the retry must launch on the sweep's backend"
    m = lab.manifest(new_id)
    assert m.backend.provisioner == "skypilot"
    assert m.resources.cloud == "do"
