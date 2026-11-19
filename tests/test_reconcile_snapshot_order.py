"""Reconcile must read the provider state *before* it decides what is protected (review finding).

It built the protected set from ``list_jobs()`` first and only then listed sky.status, DO volumes
and GCP. A job submitted in that window — a sweep still staggering its shard submits, which is
exactly when someone runs a cleanup — has a live cluster in the later snapshot but no protection,
is ours by slug and by ``runs/``, and ``--apply --yes`` (or the scheduler's ``auto_reconcile``)
destroyed it.

The ordering argument that makes snapshot-first sufficient: a job's manifest is created before
its launch is requested, so any machine visible in a snapshot belongs to a manifest that already
existed when the snapshot was taken — and is therefore in a protected set read afterwards.
"""

from __future__ import annotations

import os
import sys
import types
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import pytest

import lab.backends.skypilot as sp
import lab.core as core_mod
from helpers import make_manifest
from lab._util import now
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.models import BackendInfo, JobState

PROJECT = "tempotron-capacity"
JOB = "20261119-101010-abcdef"


def _submit_live_shard(lab: Lab, cluster: str) -> None:
    """What a staggering sweep does mid-reconcile: manifest + a live supervisor."""
    if lab.store.manifest_path(JOB).exists():
        return
    lab.store.create(
        make_manifest(JOB, "python x.py").model_copy(
            update={
                "status": JobState.running,
                "started_at": now() - timedelta(seconds=5),
                "backend": BackendInfo(provisioner="skypilot"),
            }
        )
    )
    lab.store.write_runtime(JOB, runner_pid=os.getpid(), cluster=cluster)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    home = tmp_path / "runs"
    home.mkdir()
    monkeypatch.setattr(core_mod, "local_project", lambda: PROJECT)
    lab = Lab(backend=LocalBackend(home=home, repo=tmp_path), repo=tmp_path, home=home)
    cluster = sp.cluster_name_for(JOB, project=PROJECT)
    destroyed: list[Any] = []

    fake = types.ModuleType("sky")
    fake.get = lambda x: x  # type: ignore[attr-defined]
    fake.status = lambda refresh=None: [{"name": cluster}]  # type: ignore[attr-defined]
    fake.down = lambda c: destroyed.append(("sky", c))  # type: ignore[attr-defined]
    fake.StatusRefreshMode = types.SimpleNamespace(AUTO="AUTO")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setattr(sp, "list_vast_instances", lambda *a, **k: [])
    volume = {"id": "v1", "name": f"{cluster}-3dd12990-ab12-head"}
    monkeypatch.setattr(sp, "list_do_volumes", lambda *a, **k: [volume])

    class _DoVolumes:
        """pydo's shape: a missing volume reads back as DO's 404 body, not an exception —
        reconcile now verifies each delete by id."""

        alive = {"v1"}

        def get(self, volume_id: str) -> dict[str, Any]:
            return {"volume": volume} if volume_id in self.alive else {"id": "not_found"}

        def delete(self, volume_id: str) -> None:
            destroyed.append(("do", volume_id))
            self.alive = self.alive - {volume_id}

    monkeypatch.setattr(sp, "_get_do_client", lambda: types.SimpleNamespace(volumes=_DoVolumes()))
    node = {
        "name": f"{cluster[:23]}-1e-3dd12990-head-abcd1234-compute",
        "zone": "us-central1-a",
        "status": "RUNNING",
        "labels": sp.project_labels(JOB, project=PROJECT),
    }
    monkeypatch.setattr(sp, "list_gcp_instances", lambda *a, **k: [node])
    monkeypatch.setattr(sp, "list_gcp_disks", lambda *a, **k: [])
    monkeypatch.setattr(
        sp, "delete_gcp_instance", lambda name, zone, *a, **k: destroyed.append(("gcp", name))
    )
    monkeypatch.setattr(sp, "gcp_project", lambda: "p")
    return {"lab": lab, "cluster": cluster, "destroyed": destroyed, "fake": fake}


def _during(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, which: str) -> None:
    """Make the shard's submit land while ``which`` is being listed."""
    lab, cluster = world["lab"], world["cluster"]

    def _wrap(inner: Callable[..., Any]) -> Callable[..., Any]:
        def _listing(*a: Any, **k: Any) -> Any:
            _submit_live_shard(lab, cluster)
            return inner(*a, **k)

        return _listing

    if which == "sky":
        world["fake"].status = _wrap(world["fake"].status)
    else:
        target = {"vast": "list_vast_instances", "do": "list_do_volumes", "gcp": "list_gcp_instances"}
        monkeypatch.setattr(sp, target[which], _wrap(getattr(sp, target[which])))


@pytest.mark.parametrize("which", ["vast", "sky", "do", "gcp"])
def test_a_shard_submitted_mid_reconcile_is_not_destroyed(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    _during(world, monkeypatch, which)

    report = world["lab"].reconcile(apply=True)  # == --apply --yes / auto_reconcile

    assert world["destroyed"] == [], f"destroyed a live shard submitted during the {which} listing"
    assert report["sky_orphans"] == [] and report["do_volume_orphans"] == []
    assert report["gcp_orphans"] == []


def test_a_genuinely_finished_job_is_still_cleaned_up(world: dict[str, Any]) -> None:
    """No regression: a terminal job of ours still has its leftovers destroyed."""
    lab = world["lab"]
    lab.store.create(make_manifest(JOB, "python x.py").model_copy(update={"status": JobState.failed}))

    lab.reconcile(apply=True)

    kinds = sorted(kind for kind, _ in world["destroyed"])
    assert kinds == ["do", "gcp", "sky"]
