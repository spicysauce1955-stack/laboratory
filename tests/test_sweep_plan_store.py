from __future__ import annotations

from pathlib import Path

from lab._util import now
from lab.models import SweepCell, SweepPlan
from lab.store import JobStore, cell_id_for


def test_cell_id_stable_and_order_independent():
    a = cell_id_for({"N": "1000", "alpha": "0.5"})
    b = cell_id_for({"alpha": "0.5", "N": "1000"})
    assert a == b and len(a) == 8


def test_write_read_roundtrip(tmp_path: Path):
    store = JobStore(tmp_path)
    cell = SweepCell(
        coords={"N": "1000"},
        cell_id=cell_id_for({"N": "1000"}),
        seeds_expected=[0, 1, 2, 3],
        shard_seeds=[[0, 1], [2, 3]],
        shard_job_ids=["j1", "j2"],
        results_file="results.csv",
        seed_column="seed",
        aggregate_ref=str(tmp_path / "sw1" / "cells" / "x" / "results.csv"),
    )
    plan = SweepPlan(
        sweep_id="sw1", created_at=now(), command="true", seed_axis_key="seeds", cells=[cell]
    )
    assert not store.has_sweep_plan("sw1")
    store.write_sweep_plan(plan)
    assert store.has_sweep_plan("sw1")
    got = store.read_sweep_plan("sw1")
    assert got.cells[0].status == "pending"
    assert got.cells[0].seeds_expected == [0, 1, 2, 3]
    assert got.sweep_id == "sw1"
