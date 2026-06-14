from lab.models import BackendInfo, ResourceRequest


def test_resource_request_spot_defaults_off():
    r = ResourceRequest()
    assert r.use_spot is False
    assert r.spot_fallback is True  # fallback-to-on-demand is default-on


def test_backend_info_records_launched_spot():
    b = BackendInfo(provisioner="skypilot", launched_spot=True)
    assert b.launched_spot is True
    assert BackendInfo(provisioner="local").launched_spot is None
