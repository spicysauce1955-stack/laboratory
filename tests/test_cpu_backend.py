from lab.models import ResourceRequest


def test_cloud_defaults_none():
    assert ResourceRequest().cloud is None


def test_cloud_roundtrips():
    r = ResourceRequest(cloud="do", cpus=8)
    assert r.cloud == "do" and r.cpus == 8
    assert ResourceRequest.model_validate_json(r.model_dump_json()).cloud == "do"
