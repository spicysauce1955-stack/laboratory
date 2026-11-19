"""`fetch_artifacts`' R2 fallback is best-effort: one shard's R2 error never aborts the caller."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core_fixtures import local_lab, make_repo
from helpers import make_manifest

import lab.core as core_mod
from lab import events
from lab.models import BackendInfo, JobState


class _DownR2:
    calls: list[str] = []

    def download_dir(self, prefix, out):
        _DownR2.calls.append(prefix)
        raise ConnectionError("Could not connect to the endpoint URL")


def _remote_job(lab, jid: str, *, sweep_id: str | None = None, cell_id: str | None = None):
    m = make_manifest(jid, "python x.py").model_copy(
        update={
            "status": JobState.succeeded,
            "backend": BackendInfo(provisioner="skypilot"),
            "artifacts_uri": f"r2://bucket/{jid}",
            "sweep_id": sweep_id,
            "cell_id": cell_id,
        }
    )
    lab.store.create(m)
    return m


@pytest.fixture
def down_r2(monkeypatch):
    _DownR2.calls = []
    monkeypatch.setattr(core_mod, "r2_enabled", lambda: True)
    monkeypatch.setattr(core_mod.R2Store, "from_env", staticmethod(lambda: _DownR2()))


def test_r2_network_error_is_a_note_not_a_crash(tmp_path: Path, down_r2, capsys):
    lab = local_lab(make_repo(tmp_path))
    _remote_job(lab, "jr")
    call = events.begin("cli", "fetch", {})
    assert lab.fetch_artifacts("jr") == []  # must not raise
    out, err = capsys.readouterr()
    assert out == ""
    assert "jr" in err and "R2" in err
    notes = [n for n in call.notes if n["k"] == "storage.fallback_failed"]
    assert notes and notes[0]["d"]["job_id"] == "jr"
    events.finish(call, outcome="ok")


def test_local_job_never_tries_r2(tmp_path: Path, monkeypatch):
    lab = local_lab(make_repo(tmp_path))
    lab.store.create(make_manifest("jl", "python x.py").model_copy(
        update={"status": JobState.failed}
    ))
    monkeypatch.setattr(core_mod, "r2_enabled", lambda: True)

    def _refuse():
        raise AssertionError("a local-backend job has nothing in R2 to fall back to")

    monkeypatch.setattr(core_mod.R2Store, "from_env", staticmethod(_refuse))
    assert lab.fetch_artifacts("jl") == []


def test_aggregate_survives_one_shard_r2_failure(tmp_path: Path, down_r2, capsys):
    repo = make_repo(tmp_path)
    lab = local_lab(repo)
    from lab._util import now
    from lab.models import SweepCell, SweepPlan

    # Shard A has its results locally; shard B's local copy is gone and R2 is down.
    _remote_job(lab, "ja", sweep_id="sweep-x", cell_id="c1")
    _remote_job(lab, "jb", sweep_id="sweep-x", cell_id="c1")
    (lab.store.output_dir("ja") / "results.csv").write_text("seed,acc\n0,0.5\n")
    lab.store.write_sweep_plan(
        SweepPlan(
            sweep_id="sweep-x", created_at=now(), command="python x.py", seed_axis_key="seeds",
            cells=[SweepCell(
                coords={}, cell_id="c1", seeds_expected=[0, 1], shard_seeds=[[0], [1]],
                shard_job_ids=["ja", "jb"], results_file="results.csv", seed_column="seed",
                aggregate_ref=str(lab.home / "sweep-x" / "cells" / "c1" / "results.csv"),
            )],
        )
    )
    plan = lab.aggregate_sweep("sweep-x")  # must not raise
    cell = plan.cells[0]
    assert cell.seeds_present == [0]
    assert cell.missing_seeds == [1]
    assert "jb" in capsys.readouterr().err
    assert json.loads(lab.store.sweep_plan_path("sweep-x").read_text())["cells"][0][
        "missing_seeds"
    ] == [1]
