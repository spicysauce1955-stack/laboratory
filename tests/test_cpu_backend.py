import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sky
from typer.testing import CliRunner

import lab.cli as cli_mod
import lab.sky_runner as sky_runner
from helpers import make_manifest
from lab.backends.local import LocalBackend
from lab.backends.skypilot import SkyPilotBackend, _cloud_for, build_task, robust_teardown
from lab.cli import app
from lab.core import Lab, LabError, build_backend, resolve_backend_profile
from lab.models import JobSpec, ResourceRequest


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
    assert res.cpus == 4  # default kept within the default DO account tier
    assert res.disk_size == 50  # default volume within the default DO block-storage tier
    assert res.use_spot is False and res.spot_fallback is False


def test_profile_cpu_preserves_explicit_cpus():
    _, res = resolve_backend_profile("cpu", ResourceRequest(cpus=32))
    assert res.cpus == 32 and res.cloud == "do"


def test_profile_cpu_preserves_explicit_disk_size():
    _, res = resolve_backend_profile("cpu", ResourceRequest(disk_size=200))
    assert res.disk_size == 200 and res.cloud == "do"


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


def test_build_task_passes_disk_size_to_resources(tmp_path: Path):
    """disk_size on the request must reach sky.Resources, so SkyPilot's DO provisioner sizes the
    attached block volume within the account tier (default 256GB volume is rejected on a fresh DO
    account: '422 failed to create volume: invalid size specified')."""
    m = make_manifest("c3", "python x.py", timeout="10m")
    m.resources.cloud = "do"
    m.resources.cpus = 4
    m.resources.disk_size = 50
    task = build_task(m, workdir=tmp_path)
    res = list(task.resources)[0]
    assert res.disk_size == 50


def test_do_volume_orphans_flags_untracked_lab_volumes():
    """The volume-leak analogue of the instance orphan pass: a `lab-*` DO block volume not tied to
    any running cluster is an orphan (SkyPilot names the volume after the cluster)."""
    from lab.backends.skypilot import do_volume_orphans

    volumes = [
        {"id": "v1", "name": "lab-job-dead-3dd1-head"},   # no running cluster -> orphan
        {"id": "v2", "name": "lab-job-alive-3dd1-head"},  # tied to a running cluster -> kept
        {"id": "v3", "name": "someone-else-vol"},         # not ours -> ignored
    ]
    orphans = do_volume_orphans(volumes, running_clusters={"lab-job-alive"})
    assert orphans == [{"id": "v1", "name": "lab-job-dead-3dd1-head"}]


def test_list_do_volumes_parses_pydo_response():
    """list_do_volumes unwraps pydo's {'volumes': [...]} response into plain dicts."""
    from lab.backends.skypilot import list_do_volumes

    class _Vols:
        def list(self, **kw):  # noqa: A003 — mirrors pydo's volumes.list(**kwargs)
            return {"volumes": [{"id": "v1", "name": "lab-x"}, {"id": "v2", "name": "lab-y"}]}

    class _Client:
        volumes = _Vols()

    assert list_do_volumes(_Client()) == [
        {"id": "v1", "name": "lab-x"},
        {"id": "v2", "name": "lab-y"},
    ]


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


class _SkyDownFails:
    def down(self, cluster):
        raise RuntimeError("sky.down boom")

    def get(self, x):
        return x


def test_robust_teardown_do_skips_vast_fallback(monkeypatch):
    monkeypatch.setattr(
        "lab.backends.skypilot._vast_destroy_matching",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vast fallback used for DO")),
    )
    out = robust_teardown(_SkyDownFails(), "lab-x", backoffs=(), cloud="do")
    assert out["status"] == "failed" and out["vast_fallback_used"] is False


def test_robust_teardown_vast_uses_fallback(monkeypatch):
    monkeypatch.setattr("lab.backends.skypilot._vast_destroy_matching", lambda c: [123])
    out = robust_teardown(_SkyDownFails(), "lab-x", backoffs=(), cloud="vast")
    assert out["status"] == "succeeded" and out["vast_destroyed"] == [123]


def test_sky_status_orphans_finds_untracked_lab_clusters(tmp_path, monkeypatch):
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)

    class _FakeSky:
        def get(self, x):
            return x

        def status(self, refresh=False):
            return [{"name": "lab-abc"}, {"name": "lab-running"}, {"name": "someone-else"}]

    fake = types.ModuleType("sky")
    fake.get = _FakeSky().get
    fake.status = _FakeSky().status
    fake.StatusRefreshMode = types.SimpleNamespace(AUTO="AUTO", FORCE="FORCE", NONE="NONE")
    monkeypatch.setitem(sys.modules, "sky", fake)

    orphans = lab._sky_status_orphans(running_clusters={"lab-running"})
    assert orphans == ["lab-abc"]  # lab-* not running; non-lab ignored


def _make_fake_lab(submitted_specs: list[JobSpec]) -> MagicMock:
    """A fake Lab that captures submitted JobSpec objects (no network)."""
    fake = MagicMock()
    fake.find_cached.return_value = None
    fake.submit.side_effect = lambda spec, **kw: (submitted_specs.append(spec) or "job-1")
    fake.status.return_value = MagicMock(value="queued")
    return fake


def test_cli_submit_cpu_stamps_do_profile():
    """`lab submit --backend cpu` stamps the DO profile onto the spec without touching the
    network and resolves the provisioner to skypilot (CLI stays a thin shell)."""
    captured: list[JobSpec] = []
    fake_lab = _make_fake_lab(captured)
    seen_backends: list[str] = []

    def _fake(backend: str = "local") -> MagicMock:
        seen_backends.append(backend)
        return fake_lab

    with patch.object(cli_mod, "_lab", side_effect=_fake):
        result = CliRunner().invoke(app, ["submit", "-c", "python x.py", "--backend", "cpu"])

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert len(captured) == 1
    res = captured[0].resources
    assert res.cloud == "do" and res.cpus == 4 and res.disk_size == 50
    assert res.use_spot is False and res.spot_fallback is False
    assert seen_backends == ["skypilot"]  # cpu resolved to the skypilot provisioner


def test_cli_submit_disk_size_flows_to_resources():
    """`--disk-size` reaches the submitted spec's resources (the override the cpu profile honors)."""
    captured: list[JobSpec] = []
    fake_lab = _make_fake_lab(captured)
    with patch.object(cli_mod, "_lab", return_value=fake_lab):
        result = CliRunner().invoke(
            app, ["submit", "-c", "python x.py", "--backend", "cpu", "--disk-size", "120"]
        )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert captured[0].resources.disk_size == 120  # overrides the cpu profile default of 50


def test_cli_submit_cpu_rejects_accelerators():
    """`--backend cpu` + `--accelerators` surfaces the LabError as a structured CLI error."""
    captured: list[JobSpec] = []
    fake_lab = _make_fake_lab(captured)

    with patch.object(cli_mod, "_lab", return_value=fake_lab):
        result = CliRunner().invoke(
            app,
            ["submit", "-c", "python x.py", "--backend", "cpu", "--accelerators", "RTX4090:1"],
        )

    assert result.exit_code == 1
    assert "CPU-only" in result.output
    assert captured == []  # never submitted


def test_cli_sweep_cpu_stamps_do_profile():
    """`lab sweep --backend cpu` stamps the DO profile onto the per-job resources."""
    resources_seen: list[ResourceRequest] = []
    fake_lab = MagicMock()
    fake_lab.sweep.side_effect = lambda cmd, grid, seed=None, resources=None, **_kw: (
        resources_seen.append(resources) or ("sweep-1", ["job-1", "job-2"])
    )
    seen_backends: list[str] = []

    def _fake(backend: str = "local") -> MagicMock:
        seen_backends.append(backend)
        return fake_lab

    with patch.object(cli_mod, "_lab", side_effect=_fake):
        result = CliRunner().invoke(
            app, ["sweep", "-c", "python x.py", "--grid", "lr=0.1,0.01", "--backend", "cpu"]
        )

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert len(resources_seen) == 1
    res = resources_seen[0]
    assert res.cloud == "do" and res.cpus == 4 and res.disk_size == 50
    assert seen_backends == ["skypilot"]
