"""Sweep-scoped reads take a sharded sweep's job ids from its plan, not from a project scan.

Stress run 2026-11-19: `sweep-status`, `wait --sweep`, `export <sweep>` and `jobs_in_sweep` parsed
every manifest in the project to find one sweep's shards — 0.56 s at 10k jobs, and `list` p50
1.77 s under 20x concurrency. A sharded sweep's plan.json already records every shard id (and,
since retries hold the plan lock and record each launch, every retry too). The scan stays the
fallback for unsharded sweeps, which have no plan.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import make_manifest

from lab._util import now
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.models import JobState, SweepCell, SweepPlan
from lab.store import JobStore


def _lab(tmp_path: Path) -> Lab:
    home = tmp_path / "runs"
    return Lab(backend=LocalBackend(home=home, repo=tmp_path), repo=tmp_path, home=home)


def _sharded(lab: Lab, sweep_id: str, ids: list[str]) -> None:
    for jid in ids:
        lab.store.create(
            make_manifest(jid, "python x.py").model_copy(
                update={"sweep_id": sweep_id, "status": JobState.succeeded}
            )
        )
    lab.store.write_sweep_plan(
        SweepPlan(
            sweep_id=sweep_id, created_at=now(), command="python x.py", seed_axis_key="seeds",
            cells=[
                SweepCell(
                    coords={}, cell_id="c0", seeds_expected=[0, 1], shard_seeds=[[0], [1]],
                    shard_job_ids=ids, results_file="results.csv", seed_column="seed",
                    aggregate_ref=str(lab.home / sweep_id / "cells" / "c0" / "results.csv"),
                )
            ],
        )
    )


@pytest.fixture
def no_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self: JobStore) -> list[str]:
        raise AssertionError("a sharded sweep's read scanned every manifest in the project")

    monkeypatch.setattr(JobStore, "list_job_ids", refuse)


def test_jobs_in_sweep_reads_the_plan(tmp_path: Path, no_scan) -> None:
    lab = _lab(tmp_path)
    _sharded(lab, "sweep-a", ["j2", "j1"])
    lab.store.create(make_manifest("other", "python x.py"))
    assert lab.jobs_in_sweep("sweep-a") == ["j1", "j2"]


def test_a_plan_id_whose_run_dir_is_gone_is_left_out(tmp_path: Path, no_scan) -> None:
    lab = _lab(tmp_path)
    _sharded(lab, "sweep-a", ["j1", "j2"])
    lab.store.manifest_path("j2").unlink()
    assert lab.jobs_in_sweep("sweep-a") == ["j1"]


def test_sweep_summary_reads_the_plan(tmp_path: Path, no_scan) -> None:
    lab = _lab(tmp_path)
    _sharded(lab, "sweep-a", ["j1", "j2"])
    s = lab.sweep_summary("sweep-a")
    assert s["total"] == 2 and s["succeeded"] == 2


def test_export_of_a_sharded_sweep_reads_the_plan(tmp_path: Path, no_scan) -> None:
    lab = _lab(tmp_path)
    _sharded(lab, "sweep-a", ["j1", "j2"])
    index = lab.export("sweep-a", tmp_path / "bundle")
    assert [j["job_id"] for j in index["jobs"]] == ["j1", "j2"]


def test_an_unsharded_sweep_still_falls_back_to_the_scan(tmp_path: Path) -> None:
    lab = _lab(tmp_path)
    for jid in ("u1", "u2"):
        lab.store.create(
            make_manifest(jid, "python x.py").model_copy(update={"sweep_id": "sweep-u"})
        )
    lab.store.create(make_manifest("other", "python x.py"))
    assert lab.jobs_in_sweep("sweep-u") == ["u1", "u2"]
    assert lab.sweep_summary("sweep-u")["total"] == 2
