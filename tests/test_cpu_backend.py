from pathlib import Path

import pytest
import sky

import lab.sky_runner as sky_runner
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


class _Launched:
    def __init__(self, cost_per_hr, instance_type, region):
        self._c = cost_per_hr
        self.instance_type = instance_type
        self.region = region
        self.use_spot = False

    def get_cost(self, seconds):  # SkyPilot Resources API
        return self._c * seconds / 3600


class _Handle:
    def __init__(self, launched):
        self.launched_resources = launched


def test_resolve_hourly_do_uses_sky_estimate(monkeypatch):
    handle = _Handle(_Launched(0.75, "g-16vcpu-64gb", "nyc3"))
    monkeypatch.setattr(
        sky_runner, "vast_hourly_for_cluster",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vast path used for DO")),
    )
    assert sky_runner._resolve_hourly("lab-x", handle, "do") == 0.75


def test_resolve_hourly_vast_prefers_dph(monkeypatch):
    monkeypatch.setattr(sky_runner, "vast_hourly_for_cluster", lambda c: 0.16)
    assert sky_runner._resolve_hourly("lab-x", _Handle(_Launched(9.9, "x", "y")), "vast") == 0.16


def test_provision_failure_reason_do_does_not_consult_vast_balance(monkeypatch):
    monkeypatch.setattr(
        sky_runner, "vast_balance",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vast_balance used for DO")),
    )
    msg = sky_runner.provision_failure_reason("launch error: boom", "do")
    assert "DigitalOcean" in msg or "doctl" in msg
