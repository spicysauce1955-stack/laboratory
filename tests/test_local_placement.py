"""A local job's status placement says where it ran: here, on no cloud.

Stress run 2026-11-19: every local job reported `placement.cloud: "vast"`, because the view
defaulted a missing cloud to SkyPilot's default. That default only means something for a job a
cloud provisioner launched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from helpers import make_manifest

from lab.core import _status_fields, job_status_view
from lab.models import BackendInfo, ResourceRequest
from lab.store import JobStore


def _view(tmp_path: Path, provisioner: str, cloud: str | None) -> dict[str, Any]:
    store = JobStore(tmp_path / "runs")
    m = make_manifest("j", "python x.py").model_copy(
        update={
            "backend": BackendInfo(provisioner=provisioner),
            "resources": ResourceRequest(cloud=cloud),
        }
    )
    store.create(m)
    if provisioner == "local":
        return job_status_view(tmp_path / "runs", tmp_path, "j")
    return _status_fields(store.read_manifest("j"), state="queued", mirrored=False)


def test_a_local_job_reports_no_cloud(tmp_path: Path) -> None:
    placement = _view(tmp_path, "local", None)["placement"]
    assert placement["cloud"] is None
    assert placement["provisioner"] == "local"


def test_a_local_job_ignores_cloud_resources_it_never_used(tmp_path: Path) -> None:
    """A retried remote shard once ran locally still carrying `cloud=do`; it was never on DO."""
    assert _view(tmp_path, "local", "do")["placement"]["cloud"] is None


def test_a_skypilot_job_keeps_the_vast_default(tmp_path: Path) -> None:
    placement = _view(tmp_path, "skypilot", None)["placement"]
    assert placement["cloud"] == "vast"
    assert placement["provisioner"] == "skypilot"


def test_a_skypilot_job_reports_its_cloud(tmp_path: Path) -> None:
    assert _view(tmp_path, "skypilot", "gcp")["placement"]["cloud"] == "gcp"
