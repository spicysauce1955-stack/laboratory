import pytest

from lab.backends.skypilot import SkyPilotBackend
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
