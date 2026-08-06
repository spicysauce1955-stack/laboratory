"""`lab export` — the committable provenance bundle (field-report #5)."""

import json
from pathlib import Path

import pytest
from helpers import make_manifest

from lab.backends.local import LocalBackend
from lab.core import Lab, LabError
from lab.models import CostInfo, JobState


def _lab(tmp_path: Path) -> Lab:
    return Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)


def _job(lab: Lab, job_id: str, *, sweep_id: str | None = None, files: dict | None = None):
    m = make_manifest(job_id, "python x.py").model_copy(
        update={
            "status": JobState.succeeded,
            "sweep_id": sweep_id,
            "cost": CostInfo(actual_usd=0.12),
        }
    )
    lab.store.create(m)
    out = lab.store.output_dir(job_id)
    for name, content in (files or {}).items():
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)
    return m


def test_export_job_copies_manifest_config_and_tables(tmp_path):
    lab = _lab(tmp_path)
    _job(lab, "j1", files={"results.csv": "seed,acc\n0,0.9\n", "fig.png": b"\x89PNG"})
    dest = tmp_path / "bundle"

    index = lab.export("j1", dest)

    assert (dest / "j1" / "manifest.json").exists()
    assert (dest / "j1" / "resolved_config.json").exists()
    assert (dest / "j1" / "output" / "results.csv").read_text() == "seed,acc\n0,0.9\n"
    assert (dest / "j1" / "output" / "fig.png").exists()
    assert (dest / "index.json").exists()
    job_entry = index["jobs"][0]
    assert job_entry["job_id"] == "j1"
    assert job_entry["git_commit"] == "0" * 40
    assert job_entry["actual_usd"] == 0.12
    assert any(f.endswith("results.csv") for f in job_entry["files"])


def test_export_excludes_blobs_and_records_skips(tmp_path):
    lab = _lab(tmp_path)
    _job(lab, "j2", files={
        "results.csv": "seed,acc\n0,0.9\n",
        "model.ckpt": b"\x00" * 64,
        "data.npz": b"\x00" * 64,
    })
    dest = tmp_path / "bundle"
    index = lab.export("j2", dest)
    assert not (dest / "j2" / "output" / "model.ckpt").exists()
    assert not (dest / "j2" / "output" / "data.npz").exists()
    skipped = {s["file"]: s["reason"] for s in index["jobs"][0]["skipped"]}
    assert "model.ckpt" in skipped and "data.npz" in skipped


def test_export_size_threshold_skips_and_records(tmp_path):
    lab = _lab(tmp_path)
    _job(lab, "j3", files={"big.csv": "x" * 2048})
    dest = tmp_path / "bundle"
    index = lab.export("j3", dest, max_file_mb=0.000001)  # ~1 byte cap
    assert not (dest / "j3" / "output" / "big.csv").exists()
    assert index["jobs"][0]["skipped"][0]["file"] == "big.csv"
    assert "size" in index["jobs"][0]["skipped"][0]["reason"]


def test_export_logs_opt_in(tmp_path):
    lab = _lab(tmp_path)
    _job(lab, "j4", files={"results.csv": "seed,acc\n0,0.9\n"})
    lab.store.logs_path("j4").write_text("log line\n")
    dest1 = tmp_path / "b1"
    lab.export("j4", dest1)
    assert not (dest1 / "j4" / "logs.txt").exists()
    dest2 = tmp_path / "b2"
    lab.export("j4", dest2, include_logs=True)
    assert (dest2 / "j4" / "logs.txt").read_text() == "log line\n"


def test_export_includes_code_diff_when_present(tmp_path):
    lab = _lab(tmp_path)
    _job(lab, "j5", files={"results.csv": "seed,acc\n0,0.9\n"})
    (lab.store.job_dir("j5") / "code_diff.tar.gz").write_bytes(b"\x1f\x8b")
    dest = tmp_path / "bundle"
    lab.export("j5", dest)
    assert (dest / "j5" / "code_diff.tar.gz").exists()


def test_export_sweep_bundles_plan_and_cell_aggregates(tmp_path):
    from lab.manifest import repo_root

    repo = repo_root(Path.cwd())  # sweep() pins a commit, so it needs a real git repo
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1]}, seeds="0-1", shard_size=1)
    plan = lab.sweep_plan(sweep_id)
    for jid, seed in zip(plan.cells[0].shard_job_ids, [0, 1]):
        out = lab.store.output_dir(jid)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.csv").write_text(f"seed,acc\n{seed},0.9\n")
        lab.store.update_manifest(jid, status=JobState.succeeded)
    lab.aggregate_sweep(sweep_id)

    dest = tmp_path / "bundle"
    index = lab.export(sweep_id, dest)

    assert index["kind"] == "sweep"
    assert (dest / "plan.json").exists()
    cell = lab.sweep_plan(sweep_id).cells[0]
    assert (dest / "cells" / cell.cell_id / "results.csv").exists()
    assert len(index["jobs"]) == 2


def test_export_unknown_id_fails_loud(tmp_path):
    lab = _lab(tmp_path)
    with pytest.raises(LabError, match="unknown"):
        lab.export("nope", tmp_path / "bundle")
    with pytest.raises(LabError, match="no jobs"):
        lab.export("sweep-doesnotexist", tmp_path / "bundle")


def test_cli_export_prints_index(tmp_path, monkeypatch):
    from unittest.mock import patch

    from typer.testing import CliRunner

    import lab.cli as cli_mod
    from lab.cli import app

    lab = _lab(tmp_path)
    _job(lab, "j6", files={"results.csv": "seed,acc\n0,0.9\n"})
    dest = tmp_path / "bundle"
    with patch.object(cli_mod, "_lab", return_value=lab):
        result = CliRunner().invoke(app, ["export", "j6", "--to", str(dest)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["jobs"][0]["job_id"] == "j6"
    assert (dest / "index.json").exists()


def test_fetch_artifacts_survives_missing_r2_extra(tmp_path, monkeypatch):
    """LAB_R2_ENDPOINT set but boto3 not installed must not crash a local fetch
    (verification report 2026-08-06, minor finding)."""
    import lab.core as core_mod

    lab = _lab(tmp_path)
    _job(lab, "jr2", files={})  # empty output -> the R2 fallback branch is taken
    monkeypatch.setattr(core_mod, "r2_enabled", lambda: True)

    class _NoBoto:
        @staticmethod
        def from_env():
            raise ImportError("No module named 'boto3'")

    monkeypatch.setattr(core_mod, "R2Store", _NoBoto)
    arts = lab.fetch_artifacts("jr2")  # must not raise
    assert arts == []
