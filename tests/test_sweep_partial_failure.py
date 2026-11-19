"""A sweep refused partway (e.g. submit #3 hits the price-cap admission check) must not strand the
jobs it already launched: the caller learns the sweep_id + job ids, and a sharded sweep's partial
plan is persisted so sweep-retry can finish it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core_fixtures import local_lab, make_repo
from helpers import wait_terminal

from lab.core import Lab, LabError, SweepSubmitError

# Restored by hand rather than with `monkeypatch.undo()`, which reverts conftest's isolation too:
# the rest of the test then wrote 105 job-index entries and their ledger records into the
# developer's real ~/.lab (2026-11-19).
_REAL_SUBMIT = Lab.submit


def _refuse_on(monkeypatch, n: int) -> dict[str, int]:
    real = Lab.submit
    calls = {"n": 0}

    def flaky(self, spec, **kw):
        calls["n"] += 1
        if calls["n"] == n:
            raise LabError("simulated admission refusal")
        return real(self, spec, **kw)

    monkeypatch.setattr(Lab, "submit", flaky)
    return calls


def test_sharded_sweep_refused_partway_keeps_a_plan_and_names_its_jobs(
    tmp_path: Path, monkeypatch
):
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    _refuse_on(monkeypatch, 3)

    with pytest.raises(SweepSubmitError) as ei:
        lab.sweep("true", {}, seeds="0-7", shard_size=2)
    err = ei.value
    assert isinstance(err, LabError)  # every shell's existing `except LabError` still catches it
    assert err.sweep_id.startswith("sweep-") and len(err.job_ids) == 2
    assert err.sweep_id in str(err) and all(j in str(err) for j in err.job_ids)
    assert "simulated admission refusal" in str(err)
    assert err.details == {"sweep_id": err.sweep_id, "job_ids": err.job_ids}

    plan = lab.sweep_plan(err.sweep_id)
    assert plan.cells[0].shard_job_ids == err.job_ids
    for jid in err.job_ids:
        assert lab.manifest(jid).sweep_id == err.sweep_id

    # The launched shards land their rows; sweep-retry then submits only what never launched.
    monkeypatch.setattr(Lab, "submit", _REAL_SUBMIT)  # not undo(): see _REAL_SUBMIT
    monkeypatch.chdir(repo)
    for jid, seeds in zip(err.job_ids, ([0, 1], [2, 3])):
        wait_terminal(lab.backend, jid)
        rows = "".join(f"{s},0.{s}\n" for s in seeds)
        (lab.store.output_dir(jid) / "results.csv").write_text("seed,acc\n" + rows)
    after = lab.retry_sweep(err.sweep_id).cells[0]
    new = after.shard_job_ids[2:]
    assert len(new) == 2
    assert [lab.manifest(j).run.resolved_config["seeds"] for j in new] == ["4,5", "6,7"]


def test_retry_fills_a_cell_the_failed_sweep_never_reached(tmp_path: Path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    _refuse_on(monkeypatch, 2)
    with pytest.raises(SweepSubmitError) as ei:
        lab.sweep("true", {"N": [1, 2]}, seeds="0", shard_size=1)
    plan = lab.sweep_plan(ei.value.sweep_id)
    assert [len(c.shard_job_ids) for c in plan.cells] == [1, 0]

    monkeypatch.setattr(Lab, "submit", _REAL_SUBMIT)  # not undo(): see _REAL_SUBMIT
    monkeypatch.chdir(repo)
    after = lab.retry_sweep(ei.value.sweep_id)
    assert all(c.shard_job_ids for c in after.cells)


def test_unsharded_sweep_refused_partway_names_its_jobs(tmp_path: Path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    _refuse_on(monkeypatch, 3)
    with pytest.raises(SweepSubmitError) as ei:
        lab.sweep("true", {"N": [1, 2, 3, 4]})
    assert len(ei.value.job_ids) == 2
    assert sorted(lab.jobs_in_sweep(ei.value.sweep_id)) == sorted(ei.value.job_ids)


def test_first_submit_refused_is_the_plain_error(tmp_path: Path, monkeypatch):
    """Nothing launched means no sweep exists: the original error, no orphan plan."""
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    _refuse_on(monkeypatch, 1)
    with pytest.raises(LabError) as ei:
        lab.sweep("true", {}, seeds="0-3", shard_size=2)
    assert not isinstance(ei.value, SweepSubmitError)
    assert not any(p.name.startswith("sweep-") for p in (repo / "runs").glob("*"))


def test_retry_refused_partway_persists_what_it_launched(tmp_path: Path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    sweep_id, first = lab.sweep("true", {}, seeds="0-5", shard_size=2)
    for jid in first:
        wait_terminal(lab.backend, jid)  # no rows written: every shard reads as missing
    _refuse_on(monkeypatch, 2)
    with pytest.raises(SweepSubmitError) as ei:
        lab.retry_sweep(sweep_id)
    assert ei.value.sweep_id == sweep_id and len(ei.value.job_ids) == 1
    assert ei.value.job_ids[0] in lab.sweep_plan(sweep_id).cells[0].shard_job_ids


def test_cli_sweep_emits_the_sweep_id_and_job_ids(tmp_path: Path, monkeypatch):
    from typer.testing import CliRunner

    import lab.cli as cli_mod

    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    monkeypatch.setattr(cli_mod, "_lab", lambda *a, **k: lab)
    _refuse_on(monkeypatch, 3)
    r = CliRunner().invoke(cli_mod.app, ["sweep", "-c", "true", "--seeds", "0-7", "--shard-size", "2"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["sweep_id"].startswith("sweep-")
    assert len(payload["job_ids"]) == 2
    assert "simulated admission refusal" in payload["error"]
