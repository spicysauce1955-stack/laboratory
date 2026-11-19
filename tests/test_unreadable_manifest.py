"""One unreadable manifest must not take `list`/`status`/`wait`/spend/reconcile down for the whole
project — and a NaN final metric (serialised as null) must not make a manifest unreadable."""

from __future__ import annotations

import json
import math
from pathlib import Path

from helpers import make_manifest

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.models import BackendInfo, JobManifest, JobState


def _lab(tmp_path: Path) -> Lab:
    return Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)


def test_null_final_metric_on_disk_still_loads(tmp_path: Path):
    """What a NaN final metric already wrote: pydantic serialises NaN as JSON null."""
    lab = _lab(tmp_path)
    lab.store.create(make_manifest("jn", "python x.py"))
    raw = json.loads(lab.store.manifest_path("jn").read_text())
    raw["final_metrics"] = {"loss": None, "acc": 0.5}
    lab.store.manifest_path("jn").write_text(json.dumps(raw))
    assert lab.manifest("jn").final_metrics == {"acc": 0.5}


def test_non_finite_final_metric_is_never_written(tmp_path: Path):
    m = make_manifest("jf", "python x.py").model_copy(
        update={"final_metrics": {"loss": math.nan, "acc": 0.5, "big": math.inf}}
    )
    m = JobManifest.model_validate(m.model_dump())
    assert m.final_metrics == {"acc": 0.5}
    assert json.loads(m.model_dump_json())["final_metrics"] == {"acc": 0.5}


def test_store_write_drops_a_non_finite_metric(tmp_path: Path):
    """Store writes go through model_copy(update=...), which skips validators."""
    lab = _lab(tmp_path)
    lab.store.create(make_manifest("jw", "python x.py"))
    lab.store.update_manifest("jw", final_metrics={"loss": math.nan, "acc": 0.5})
    assert json.loads(lab.store.manifest_path("jw").read_text())["final_metrics"] == {"acc": 0.5}


def _corrupt(lab: Lab, job_id: str) -> None:
    lab.store.create(make_manifest(job_id, "python x.py"))
    lab.store.manifest_path(job_id).write_text('{"job_id": "' + job_id + '", "status": ')


def test_list_jobs_skips_and_names_an_unreadable_manifest(tmp_path: Path, capsys):
    lab = _lab(tmp_path)
    lab.store.create(make_manifest("good", "python x.py"))
    _corrupt(lab, "bad")

    jobs = lab.list_jobs()

    assert [j.job_id for j in jobs] == ["good"]
    assert [u["job_id"] for u in lab.unreadable_jobs] == ["bad"]
    out, err = capsys.readouterr()
    assert out == "" and "bad" in err
    assert lab.spend()["unreadable_jobs"] == ["bad"]


def test_cli_list_survives_an_unreadable_manifest(tmp_path: Path, monkeypatch):
    from typer.testing import CliRunner

    import lab.cli as cli_mod

    lab = _lab(tmp_path)
    lab.store.create(make_manifest("good", "python x.py"))
    _corrupt(lab, "bad")
    monkeypatch.setattr(cli_mod, "_lab", lambda *a, **k: lab)

    r = CliRunner().invoke(cli_mod.app, ["list"])

    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    assert [j["job_id"] for j in payload["jobs"]] == ["good"]
    assert payload["spend"]["unreadable_jobs"] == ["bad"]


def test_reconcile_never_treats_an_unreadable_jobs_machine_as_an_orphan(tmp_path, monkeypatch):
    """Skipping an unreadable manifest must not *unprotect* its cluster: for all reconcile knows
    it is a running job, and destroying a live job's machine is the 2026-08-20 incident."""
    import sys
    import types

    from lab.backends.skypilot import cluster_name_for

    lab = _lab(tmp_path)
    lab.store.create(
        make_manifest("jlive", "python x.py").model_copy(
            update={"status": JobState.running, "backend": BackendInfo(provisioner="skypilot")}
        )
    )
    lab.store.manifest_path("jlive").write_text("{truncated")
    label = f"{cluster_name_for('jlive')}-1a2b-head"
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances", lambda *a, **k: [{"id": 7, "label": label}]
    )
    fake = types.ModuleType("sky")
    fake.get = lambda x: x
    fake.status = lambda refresh=None: []
    fake.StatusRefreshMode = types.SimpleNamespace(AUTO="AUTO", FORCE="FORCE", NONE="NONE")
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])

    report = lab.reconcile()

    assert report["orphans"] == []
    assert label not in report["unattributed"]
    assert report["unreadable_manifests"] == ["jlive"]
