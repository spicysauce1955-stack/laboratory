from pathlib import Path

import pytest
import sky

from helpers import make_manifest
from lab.backends.skypilot import SkyPilotBackend, _cloud_for, build_task
from lab.core import LabError, build_backend, resolve_backend_profile
from lab.models import ResourceRequest


def test_cloud_defaults_none():
    assert ResourceRequest().cloud is None


def test_cloud_roundtrips():
    r = ResourceRequest(cloud="do", cpus=8)
    assert r.cloud == "do" and r.cpus == 8
    assert ResourceRequest.model_validate_json(r.model_dump_json()).cloud == "do"


def test_profile_cpu_sets_do_and_defaults():
    provisioner, res = resolve_backend_profile("cpu", ResourceRequest())
    assert provisioner == "skypilot"
    assert res.cloud == "do"
    assert res.cpus == 8  # default
    assert res.use_spot is False and res.spot_fallback is False


def test_profile_cpu_preserves_explicit_cpus():
    _, res = resolve_backend_profile("cpu", ResourceRequest(cpus=32))
    assert res.cpus == 32 and res.cloud == "do"


def test_profile_cpu_rejects_accelerators():
    with pytest.raises(LabError, match="CPU-only"):
        resolve_backend_profile("cpu", ResourceRequest(accelerators="RTX4090:1"))


def test_profile_passthrough_for_other_backends():
    res = ResourceRequest(cpus=4)
    provisioner, out = resolve_backend_profile("skypilot", res)
    assert provisioner == "skypilot" and out is res  # unchanged identity


def test_build_backend_cpu_is_skypilot(tmp_path):
    b = build_backend("cpu", home=tmp_path, repo=tmp_path)
    assert isinstance(b, SkyPilotBackend)


def test_cloud_for_maps_names():
    assert isinstance(_cloud_for("do"), sky.clouds.DO)
    assert isinstance(_cloud_for("vast"), sky.clouds.Vast)
    assert isinstance(_cloud_for("gcp"), sky.clouds.GCP)
    assert isinstance(_cloud_for("unknown"), sky.clouds.Vast)  # fallback


def test_build_task_uses_do_cloud_no_accelerators(tmp_path: Path):
    m = make_manifest("c1", "python x.py", timeout="10m")
    m.resources.cloud = "do"
    m.resources.cpus = 8
    task = build_task(m, workdir=tmp_path)
    res = list(task.resources)[0]
    assert isinstance(res.cloud, sky.clouds.DO)
    assert res.accelerators is None


def test_build_task_rejects_do_spot(tmp_path: Path):
    m = make_manifest("c2", "python x.py", timeout="10m")
    m.resources.cloud = "do"
    m.resources.use_spot = True
    with pytest.raises(LabError, match="DigitalOcean has no spot"):
        build_task(m, workdir=tmp_path)
