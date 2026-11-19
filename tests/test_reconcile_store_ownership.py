"""A matching project slug names a *project*, not a *job store* (review finding, 2026-11).

`_ours_only` took a cluster name stamped with this project's slug as proof that this checkout's
``runs/`` owns it. Two job stores routinely share a project name:

* two clones of one project on one machine, and
* the scheduler host (``LAB_REPO_DIR=/opt/tempotron-capacity``) next to the laptop checkout of the
  same project, on one cloud account.

So ``reconcile --apply --yes`` (or the scheduler's ``auto_reconcile``) from one store destroyed
the other store's RUNNING clusters — the 2026-08-20 incident again, one level down. The
machine-wide registry (``~/.lab/jobs/index.jsonl``) records which ``runs_dir`` owns the job and
was never consulted, because the slug check returned first.

The rule pinned here: slug match + job id not in this ``runs/`` => destroyable only on positive
proof that *this* runs_dir owns it (a registry entry naming it). Otherwise the resource is
reported (``other_projects`` / ``unattributed``) and never destroyed. The registry is
per-machine, so a job the scheduler host launched is unknown to the laptop's registry and must
therefore never be destroyed from the laptop.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import lab.backends.skypilot as sp
import lab.core as core_mod
from helpers import make_manifest
from lab import attribution
from lab.attribution import Attribution
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.models import JobState

PROJECT = "tempotron-capacity"
JOB = "20261119-101010-abcdef"


def _fake_clouds(monkeypatch: pytest.MonkeyPatch, cluster: str) -> dict[str, list[Any]]:
    """One Vast rental + one sky-tracked cluster for ``cluster``; record every destroy."""
    calls: dict[str, list[Any]] = {"vast": [], "sky": []}
    monkeypatch.setattr(
        sp, "list_vast_instances", lambda *a, **k: [{"id": 77, "label": f"{cluster}-1a2b-head"}]
    )

    class _Vast:
        def destroy_instance(self, id: int) -> None:  # noqa: A002 — the SDK's own kwarg
            calls["vast"].append(int(id))

    monkeypatch.setattr(sp, "_get_vast_client", lambda: _Vast())
    fake = types.ModuleType("sky")
    fake.get = lambda x: x  # type: ignore[attr-defined]
    fake.status = lambda refresh=None: [{"name": cluster}]  # type: ignore[attr-defined]
    fake.down = lambda c: calls["sky"].append(c)  # type: ignore[attr-defined]
    fake.StatusRefreshMode = types.SimpleNamespace(AUTO="AUTO")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setattr(sp, "list_do_volumes", lambda *a, **k: [])
    not_here = sp.GcpNotConfigured("no ADC")
    monkeypatch.setattr(sp, "list_gcp_instances", lambda *a, **k: (_ for _ in ()).throw(not_here))
    monkeypatch.setattr(sp, "list_gcp_disks", lambda *a, **k: (_ for _ in ()).throw(not_here))
    return calls


def _lab(home: Path, monkeypatch: pytest.MonkeyPatch) -> Lab:
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(core_mod, "local_project", lambda: PROJECT)
    return Lab(backend=LocalBackend(home=home, repo=home.parent), repo=home.parent, home=home)


def test_a_live_job_in_another_store_of_the_same_project_is_not_destroyed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's reproduction: the registry says exactly where the job lives."""
    other_runs = tmp_path / "clone-b" / "runs"
    other_runs.mkdir(parents=True)
    attribution.record_job(JOB, project=PROJECT, runs_dir=other_runs)
    cluster = sp.cluster_name_for(JOB, project=PROJECT)
    calls = _fake_clouds(monkeypatch, cluster)
    lab = _lab(tmp_path / "clone-a" / "runs", monkeypatch)

    report = lab.reconcile(apply=True)

    assert calls == {"vast": [], "sky": []}, "destroyed another job store's live cluster"
    assert report["orphans"] == [] and report["sky_orphans"] == []
    owners = {(o["pass"], o["project"], o.get("runs_dir")) for o in report["other_projects"]}
    assert owners == {
        ("orphans", PROJECT, str(other_runs.resolve())),
        ("sky_orphans", PROJECT, str(other_runs.resolve())),
    }


def test_a_scheduler_launched_job_unknown_to_this_machine_is_never_destroyed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The laptop's registry has never heard of a job the scheduler host launched. The slug
    matches and nothing else is known: that is not ownership, so it must not die."""
    cluster = sp.cluster_name_for(JOB, project=PROJECT)
    calls = _fake_clouds(monkeypatch, cluster)
    lab = _lab(tmp_path / "laptop" / "runs", monkeypatch)

    report = lab.reconcile(apply=True)

    assert calls == {"vast": [], "sky": []}
    assert report["unattributed"] and all(cluster in u for u in report["unattributed"])
    assert report["orphans"] == [] and report["sky_orphans"] == []


def test_a_ledger_claim_by_project_name_is_not_proof_of_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two clones on one machine share one event ledger, and the ledger records only the
    project *name*. A name says nothing the slug did not already say."""
    cluster = sp.cluster_name_for(JOB, project=PROJECT)
    calls = _fake_clouds(monkeypatch, cluster)
    lab = _lab(tmp_path / "clone-a" / "runs", monkeypatch)
    monkeypatch.setattr(
        core_mod,
        "attribute_jobs",
        lambda ids: {j: Attribution(j, PROJECT, None, "ledger") for j in ids},
    )

    report = lab.reconcile(apply=True)

    assert calls == {"vast": [], "sky": []}
    assert report["orphans"] == [] and report["sky_orphans"] == []


def test_a_registry_entry_naming_this_store_is_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No regression: our own leak whose manifest is gone from runs/ is still cleaned up, as
    long as the registry says it was created by *this* store."""
    home = tmp_path / "clone-a" / "runs"
    home.mkdir(parents=True)
    attribution.record_job(JOB, project=PROJECT, runs_dir=home)
    cluster = sp.cluster_name_for(JOB, project=PROJECT)
    calls = _fake_clouds(monkeypatch, cluster)
    lab = _lab(home, monkeypatch)

    report = lab.reconcile(apply=True)

    assert calls == {"vast": [77], "sky": [cluster]}
    assert report["other_projects"] == [] and report["unattributed"] == []


def test_a_terminal_job_in_this_runs_dir_is_still_ours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No regression: a job in our own runs/ is ours by definition, registry or not."""
    home = tmp_path / "clone-a" / "runs"
    lab = _lab(home, monkeypatch)
    lab.store.create(make_manifest(JOB, "python x.py").model_copy(
        update={"status": JobState.failed}
    ))
    # A newer registry record naming another runs_dir (a copied runs/, say) must not out-rank
    # the manifest actually sitting in ours.
    attribution.record_job(JOB, project=PROJECT, runs_dir=tmp_path / "elsewhere" / "runs")
    cluster = sp.cluster_name_for(JOB, project=PROJECT)
    calls = _fake_clouds(monkeypatch, cluster)

    report = lab.reconcile(apply=True)

    assert calls == {"vast": [77], "sky": [cluster]}
    assert report["other_projects"] == []
