"""GCP as a compute cloud: validation, profile composition, teardown/preemption semantics,
and the `--cloud` CLI/MCP surface. Mirrors the structure of test_cpu_backend.py (no network)."""

import pytest

from lab.backends.local import LocalBackend
from lab.core import Lab, LabError, resolve_backend_profile, validate_cloud
from lab.models import ResourceRequest


# ---------------------------------------------------------------------------
# validate_cloud
# ---------------------------------------------------------------------------


def test_validate_cloud_accepts_supported_and_none():
    assert validate_cloud(None) is None
    assert validate_cloud("vast") == "vast"
    assert validate_cloud("do") == "do"
    assert validate_cloud("gcp") == "gcp"


def test_validate_cloud_rejects_unknown():
    with pytest.raises(LabError, match="aws"):
        validate_cloud("aws")


# ---------------------------------------------------------------------------
# resolve_backend_profile with a cloud override
# ---------------------------------------------------------------------------


def test_profile_cpu_cloud_override_gcp_keeps_defaults():
    provisioner, res = resolve_backend_profile("cpu", ResourceRequest(cloud="gcp"))
    assert provisioner == "skypilot"
    assert res.cloud == "gcp"
    assert res.cpus == 4 and res.disk_size == 50


def test_profile_cpu_gcp_preserves_spot_flags():
    """GCP has preemptible CPU instances — only DO forces spot off."""
    _, res = resolve_backend_profile(
        "cpu", ResourceRequest(cloud="gcp", use_spot=True, spot_fallback=True)
    )
    assert res.use_spot is True and res.spot_fallback is True


def test_profile_cpu_default_do_still_forces_spot_off():
    _, res = resolve_backend_profile("cpu", ResourceRequest(use_spot=True))
    assert res.cloud == "do"
    assert res.use_spot is False and res.spot_fallback is False


def test_profile_rejects_unknown_cloud_for_every_backend():
    for backend in ("cpu", "skypilot", "local"):
        with pytest.raises(LabError, match="aws"):
            resolve_backend_profile(backend, ResourceRequest(cloud="aws"))


# ---------------------------------------------------------------------------
# reconcile without vastai-sdk (GCP/DO-only install)
# ---------------------------------------------------------------------------


def test_reconcile_skips_vast_pass_when_sdk_missing(tmp_path, monkeypatch):
    """A GCP-only install (no vastai-sdk) must still run the cloud-agnostic sky.status pass
    instead of hard-failing."""
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("No module named 'vastai_sdk'")),
    )
    monkeypatch.setattr(Lab, "_sky_status_orphans", lambda self, running_clusters: ["lab-orphan"])
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])

    report = lab.reconcile()

    assert report["vast_pass"] == "skipped (vastai-sdk not installed)"
    assert report["orphans"] == [] and report["instances_total"] == 0
    assert report["sky_orphans"] == ["lab-orphan"]  # the agnostic pass still ran


def test_reconcile_still_raises_on_non_import_listing_failure(tmp_path, monkeypatch):
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down")),
    )
    with pytest.raises(LabError, match="could not list Vast"):
        lab.reconcile()


# ---------------------------------------------------------------------------
# GCP second teardown channel: compute-API orphan passes + robust_teardown branch
# ---------------------------------------------------------------------------


def test_gcp_instance_orphans_flags_untracked_lab_instances():
    from lab.backends.skypilot import gcp_instance_orphans

    instances = [
        {"name": "lab-job-dead-3dd1-head", "zone": "us-central1-a", "status": "RUNNING"},
        {"name": "lab-job-alive-3dd1-head", "zone": "us-central1-a", "status": "RUNNING"},
        {"name": "someone-else-vm", "zone": "us-central1-a", "status": "RUNNING"},
    ]
    orphans = gcp_instance_orphans(instances, running_clusters={"lab-job-alive"})
    assert [o["name"] for o in orphans] == ["lab-job-dead-3dd1-head"]


def test_gcp_disk_orphans_flags_unattached_lab_disks():
    """A persistent disk that outlived its VM keeps billing — the GCP analogue of the DO
    volume-leak pass. Attached disks die with their instance teardown; only unattached ones leak."""
    from lab.backends.skypilot import gcp_disk_orphans

    disks = [
        {"name": "lab-job-dead-3dd1-head", "zone": "us-central1-a", "users": []},
        {"name": "lab-job-alive-3dd1-head", "zone": "us-central1-a",
         "users": ["projects/p/zones/z/instances/lab-job-alive-3dd1-head"]},
        {"name": "someone-else-disk", "zone": "us-central1-a", "users": []},
    ]
    orphans = gcp_disk_orphans(disks, running_clusters=set())
    assert [o["name"] for o in orphans] == ["lab-job-dead-3dd1-head"]


def test_list_gcp_instances_parses_aggregated_list():
    from lab.backends.skypilot import list_gcp_instances

    class _Req:
        def execute(self):
            return {
                "items": {
                    "zones/us-central1-a": {
                        "instances": [
                            {"name": "lab-x-1a2b-head",
                             "zone": "https://.../zones/us-central1-a", "status": "RUNNING"}
                        ]
                    },
                    "zones/us-east1-b": {"warning": {"code": "NO_RESULTS_ON_PAGE"}},
                }
            }

    class _Instances:
        def aggregatedList(self, project):  # noqa: N802 — mirrors googleapiclient
            return _Req()

        def aggregatedList_next(self, previous_request, previous_response):  # noqa: N802
            return None

    class _Compute:
        def instances(self):
            return _Instances()

    out = list_gcp_instances(_Compute(), "proj")
    assert out == [{"name": "lab-x-1a2b-head", "zone": "us-central1-a", "status": "RUNNING"}]


def test_robust_teardown_gcp_uses_gcp_fallback(monkeypatch):
    from lab.backends.skypilot import robust_teardown

    class _SkyDownFails:
        def down(self, cluster):
            raise RuntimeError("sky.down boom")

        def get(self, x):
            return x

    monkeypatch.setattr(
        "lab.backends.skypilot._gcp_destroy_matching", lambda c: ["lab-x-1a2b-head"]
    )
    out = robust_teardown(_SkyDownFails(), "lab-x", backoffs=(), cloud="gcp")
    assert out["status"] == "succeeded"
    assert out["gcp_destroyed"] == ["lab-x-1a2b-head"]


def test_robust_teardown_gcp_fallback_failure_is_failed(monkeypatch):
    from lab.backends.skypilot import robust_teardown

    class _SkyDownFails:
        def down(self, cluster):
            raise RuntimeError("sky.down boom")

        def get(self, x):
            return x

    monkeypatch.setattr(
        "lab.backends.skypilot._gcp_destroy_matching",
        lambda c: (_ for _ in ()).throw(RuntimeError("no ADC")),
    )
    out = robust_teardown(_SkyDownFails(), "lab-x", backoffs=(), cloud="gcp")
    assert out["status"] == "failed"
    assert "gcp-direct" in (out["error"] or "")


def test_reconcile_gcp_pass_flags_and_destroys_orphans(tmp_path, monkeypatch):
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
    monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
    monkeypatch.setattr(Lab, "_sky_status_orphans", lambda self, running_clusters: [])
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances",
        lambda *a, **k: [{"name": "lab-leak-1a2b-head", "zone": "us-central1-a",
                          "status": "RUNNING"}],
    )
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_disks",
        lambda *a, **k: [{"name": "lab-leak-1a2b-head", "zone": "us-central1-a", "users": []}],
    )
    deleted: list[tuple] = []
    monkeypatch.setattr(
        "lab.backends.skypilot.delete_gcp_instance",
        lambda name, zone, **kw: deleted.append(("inst", name, zone)),
    )
    monkeypatch.setattr(
        "lab.backends.skypilot.delete_gcp_disk",
        lambda name, zone, **kw: deleted.append(("disk", name, zone)),
    )

    report = lab.reconcile(apply=True)
    assert [o["name"] for o in report["gcp_orphans"]] == ["lab-leak-1a2b-head"]
    assert report["gcp_destroyed"] == ["lab-leak-1a2b-head"]
    assert [o["name"] for o in report["gcp_disk_orphans"]] == ["lab-leak-1a2b-head"]
    assert report["gcp_disks_destroyed"] == ["lab-leak-1a2b-head"]
    assert ("inst", "lab-leak-1a2b-head", "us-central1-a") in deleted
    assert ("disk", "lab-leak-1a2b-head", "us-central1-a") in deleted


def test_reconcile_gcp_pass_skips_when_unconfigured(tmp_path, monkeypatch):
    """No ADC / GCP not set up: the GCP pass skips silently (best-effort), like the DO pass."""
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
    monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
    monkeypatch.setattr(Lab, "_sky_status_orphans", lambda self, running_clusters: [])
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no application default creds")),
    )
    report = lab.reconcile()
    assert report["gcp_orphans"] == [] and report["gcp_destroyed"] == []


# ---------------------------------------------------------------------------
# CLI surface: --cloud on submit/sweep
# ---------------------------------------------------------------------------


def _fake_cli_lab(captured):
    from unittest.mock import MagicMock

    fake = MagicMock()
    fake.find_cached.return_value = None
    fake.submit.side_effect = lambda spec, **kw: (captured.append(spec) or "job-1")
    fake.status.return_value = MagicMock(value="queued")
    return fake


def test_cli_submit_cloud_gcp_with_accelerators_and_spot():
    from unittest.mock import patch

    from typer.testing import CliRunner

    import lab.cli as cli_mod
    from lab.cli import app

    captured: list = []
    with patch.object(cli_mod, "_lab", return_value=_fake_cli_lab(captured)):
        result = CliRunner().invoke(
            app,
            ["submit", "-c", "python x.py", "--backend", "skypilot",
             "--cloud", "gcp", "--accelerators", "L4:1", "--spot"],
        )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    res = captured[0].resources
    assert res.cloud == "gcp" and res.accelerators == "L4:1" and res.use_spot is True


def test_cli_submit_backend_cpu_cloud_gcp_override():
    from unittest.mock import patch

    from typer.testing import CliRunner

    import lab.cli as cli_mod
    from lab.cli import app

    captured: list = []
    with patch.object(cli_mod, "_lab", return_value=_fake_cli_lab(captured)):
        result = CliRunner().invoke(
            app, ["submit", "-c", "python x.py", "--backend", "cpu", "--cloud", "gcp"]
        )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    res = captured[0].resources
    assert res.cloud == "gcp" and res.cpus == 4 and res.disk_size == 50


def test_cli_submit_unknown_cloud_exits_1():
    from unittest.mock import patch

    from typer.testing import CliRunner

    import lab.cli as cli_mod
    from lab.cli import app

    captured: list = []
    with patch.object(cli_mod, "_lab", return_value=_fake_cli_lab(captured)):
        result = CliRunner().invoke(
            app, ["submit", "-c", "python x.py", "--backend", "skypilot", "--cloud", "aws"]
        )
    assert result.exit_code == 1
    assert "unknown cloud" in result.output
    assert captured == []


def test_cli_sweep_cloud_gcp():
    from unittest.mock import MagicMock, patch

    from typer.testing import CliRunner

    import lab.cli as cli_mod
    from lab.cli import app

    resources_seen: list = []
    fake = MagicMock()
    fake.sweep.side_effect = lambda cmd, grid, seed=None, resources=None, **_kw: (
        resources_seen.append(resources) or ("sweep-1", ["job-1"])
    )
    with patch.object(cli_mod, "_lab", return_value=fake):
        result = CliRunner().invoke(
            app,
            ["sweep", "-c", "python x.py", "--grid", "lr=0.1", "--backend", "cpu",
             "--cloud", "gcp"],
        )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert resources_seen[0].cloud == "gcp"


def test_cli_reconcile_exits_3_on_sky_orphans_even_without_vast_pass():
    """With the Vast pass skipped (no vastai-sdk), sky_orphans alone must still trip the
    dry-run leak alarm (exit 3)."""
    from unittest.mock import MagicMock, patch

    from typer.testing import CliRunner

    import lab.cli as cli_mod
    from lab.cli import app

    fake = MagicMock()
    fake.reconcile.return_value = {
        "vast_pass": "skipped (vastai-sdk not installed)",
        "instances_total": 0, "orphans": [], "destroyed": [], "ghosts": [],
        "sky_orphans": ["lab-leaked"], "sky_destroyed": [],
        "do_volume_orphans": [], "do_volumes_destroyed": [], "applied": False,
    }
    with patch.object(cli_mod, "_lab", return_value=fake):
        result = CliRunner().invoke(app, ["reconcile"])
    assert result.exit_code == 3


# ---------------------------------------------------------------------------
# Scheduler: register validation + cloud-aware watchdog seams
# ---------------------------------------------------------------------------


def _reg_fixtures(tmp_path):
    from datetime import datetime, timedelta, timezone

    from test_scheduler_bundle import _make_repo

    from lab.scheduler.models import Guardrails
    from lab.scheduler.queue import LocalQueueStore

    repo = _make_repo(tmp_path)
    q = LocalQueueStore(tmp_path / "q")
    guard = Guardrails(expires_at=datetime.now(timezone.utc) + timedelta(days=1))
    return repo, q, guard


def test_register_rejects_vast_price_trigger_for_gcp(tmp_path):
    from lab.models import JobSpec
    from lab.scheduler.models import Triggers
    from lab.scheduler.register import register

    repo, q, guard = _reg_fixtures(tmp_path)
    spec = JobSpec(command="python x.py", resources=ResourceRequest(cloud="gcp"))
    with pytest.raises(LabError, match="Vast"):
        register(repo, q, spec, Triggers(max_hourly_usd=0.5), guard)
    with pytest.raises(LabError, match="Vast"):
        register(repo, q, spec, Triggers(offer_query="gpu_name=RTX_4090"), guard)
    assert q.list_entries() == []  # nothing committed


def test_register_gcp_cloud_roundtrips_through_queue(tmp_path):
    from lab.models import JobSpec
    from lab.scheduler.models import Triggers
    from lab.scheduler.register import register

    repo, q, guard = _reg_fixtures(tmp_path)
    spec = JobSpec(command="python x.py", resources=ResourceRequest(cloud="gcp"))
    reg = register(repo, q, spec, Triggers(), guard)
    assert q.get_entry(reg.reg_id).spec.resources.cloud == "gcp"


def test_register_rejects_unknown_cloud(tmp_path):
    from lab.models import JobSpec
    from lab.scheduler.models import Triggers
    from lab.scheduler.register import register

    repo, q, guard = _reg_fixtures(tmp_path)
    spec = JobSpec(command="python x.py", resources=ResourceRequest(cloud="aws"))
    with pytest.raises(LabError, match="aws"):
        register(repo, q, spec, Triggers(), guard)


def test_register_sweep_rejects_vast_price_trigger_for_gcp(tmp_path):
    from lab.scheduler.models import Triggers
    from lab.scheduler.register import register_sweep

    repo, q, guard = _reg_fixtures(tmp_path)
    with pytest.raises(LabError, match="Vast"):
        register_sweep(
            repo, q, "python x.py", {"lr": [0.1]},
            resources=ResourceRequest(cloud="gcp", timeout="1h"),
            triggers=Triggers(max_hourly_usd=0.5), guardrails=guard,
        )


def _make_sched(tmp_path):
    from lab.scheduler.queue import LocalQueueStore
    from lab.scheduler.tick import Scheduler

    q = LocalQueueStore(tmp_path / "queue")
    return Scheduler(q, home=tmp_path / "runs")


def test_scheduler_cluster_alive_gcp_uses_sky_status(tmp_path, monkeypatch):
    import sys
    import types

    sched = _make_sched(tmp_path)
    monkeypatch.setattr(
        "lab.backends.skypilot.vast_hourly_for_cluster",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vast path used for gcp")),
    )
    fake_sky = types.ModuleType("sky")
    fake_sky.get = lambda x: x  # type: ignore[attr-defined]
    fake_sky.status = lambda cluster_names=None: [  # type: ignore[attr-defined]
        {"name": "lab-x", "status": types.SimpleNamespace(name="UP")}
    ]
    monkeypatch.setitem(sys.modules, "sky", fake_sky)
    assert sched._cluster_alive("lab-x", cloud="gcp") is True

    fake_sky.status = lambda cluster_names=None: []  # type: ignore[attr-defined]
    assert sched._cluster_alive("lab-x", cloud="gcp") is False


def test_scheduler_teardown_forwards_cloud(tmp_path, monkeypatch):
    import sys
    import types

    sched = _make_sched(tmp_path)
    monkeypatch.setitem(sys.modules, "sky", types.ModuleType("sky"))
    seen: dict = {}

    def _fake_tdr(sky_mod, cluster, store, job_id, cloud="vast"):
        seen["cloud"] = cloud
        return True

    monkeypatch.setattr("lab.backends.skypilot.tear_down_and_record", _fake_tdr)
    assert sched._teardown("lab-x", "job-1", cloud="gcp") is True
    assert seen["cloud"] == "gcp"


# ---------------------------------------------------------------------------
# SkyPilot backend: strict cloud map, preemption confirm, teardown annotation
# ---------------------------------------------------------------------------


def test_cloud_for_rejects_unknown():
    from lab.backends.skypilot import _cloud_for

    with pytest.raises(LabError, match="unknown"):
        _cloud_for("aws")


def test_build_task_gcp_accelerators_pass_through(tmp_path):
    import sky

    from helpers import make_manifest
    from lab.backends.skypilot import build_task

    m = make_manifest("g1", "python x.py", timeout="10m")
    m.resources.cloud = "gcp"
    m.resources.accelerators = "L4:1"
    task = build_task(m, workdir=tmp_path)
    res = list(task.resources)[0]
    assert isinstance(res.cloud, sky.clouds.GCP)
    assert res.accelerators == {"L4": 1}


def test_build_task_gcp_spot_allowed_with_fallback(tmp_path):
    """GCP spot must not hit the DO rejection, and spot_fallback yields the two-resource
    spot -> on-demand list."""
    from helpers import make_manifest
    from lab.backends.skypilot import build_task

    m = make_manifest("g2", "python x.py", timeout="10m")
    m.resources.cloud = "gcp"
    m.resources.use_spot = True
    m.resources.spot_fallback = True
    task = build_task(m, workdir=tmp_path)
    res_list = list(task.resources)
    assert [r.use_spot for r in res_list] == [True, False]


def test_preempted_teardown_confirmed_gcp_never_touches_vast(monkeypatch):
    from lab.backends.skypilot import preempted_teardown_confirmed

    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vast listing used for gcp")),
    )
    assert preempted_teardown_confirmed("gcp", "lab-x") is True


def test_preempted_teardown_confirmed_vast_delegates(monkeypatch):
    from lab.backends.skypilot import preempted_teardown_confirmed

    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances",
        lambda *a, **k: [{"label": "lab-x still here"}],
    )
    assert preempted_teardown_confirmed("vast", "lab-x") is False


def test_provision_failure_reason_gcp_mentions_sky_check_not_vast(monkeypatch):
    import lab.sky_runner as sky_runner

    monkeypatch.setattr(
        sky_runner, "vast_balance",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vast_balance used for gcp")),
    )
    msg = sky_runner.provision_failure_reason("launch error: boom", "gcp")
    assert "sky check" in msg and "quota" in msg


def test_run_job_gcp_preemption_does_not_flag_teardown_failed(tmp_path, monkeypatch):
    """The :367 bug: a preempted non-Vast spot job whose teardown succeeded must NOT be
    marked teardown_status=failed via the Vast-only confirm_no_rental."""
    import sys
    import types

    import lab.sky_runner as runner_mod
    from helpers import make_manifest
    from lab._util import now
    from lab.models import BackendInfo, CostInfo, JobState
    from lab.store import JobStore

    home = tmp_path / "runs"
    store = JobStore(home)
    m = make_manifest("g4", "python x.py", timeout="1h").model_copy(
        update={
            "status": JobState.running,
            "started_at": now(),
            "cost": CostInfo(hourly_usd=0.2, estimated_usd=0.2),
            "backend": BackendInfo(provisioner="skypilot", launched_spot=True),
        }
    )
    m.resources.cloud = "gcp"
    m.resources.use_spot = True
    store.create(m)
    store.write_runtime("g4", runner_pid=1, cluster="lab-g4")

    fake_sky = types.ModuleType("sky")
    monkeypatch.setitem(sys.modules, "sky", fake_sky)
    monkeypatch.setattr(fake_sky, "tail_logs", lambda *a, **k: None, raising=False)
    # Cluster vanished mid-run without a terminal state -> classifier infers preemption.
    monkeypatch.setattr(runner_mod, "_wait_terminal", lambda *a, **k: (JobState.failed, False))
    monkeypatch.setattr(runner_mod, "_cluster_up", lambda *a, **k: False)
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)
    # Any touch of the Vast listing for a gcp job is the bug.
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vast listing used for gcp")),
    )

    rc = runner_mod.run_job(home / "g4", adopt=True)

    final = store.read_manifest("g4")
    assert final.status is JobState.preempted
    assert rc == 0  # teardown succeeded and stays succeeded
    assert final.end_reason != "preempted but teardown unconfirmed — see `lab reconcile`"


def test_tear_down_and_record_gcp_annotation_names_cloud(tmp_path, monkeypatch):
    """A gcp teardown failure must not tell the operator to run vastai commands."""
    from lab.backends.skypilot import tear_down_and_record
    from lab.store import JobStore

    from helpers import make_manifest

    store = JobStore(tmp_path)
    m = make_manifest("g3", "python x.py", timeout="10m")
    store.create(m)
    monkeypatch.setattr(
        "lab.backends.skypilot.robust_teardown",
        lambda *a, **k: {
            "status": "failed",
            "error": "boom",
            "attempts": 3,
            "vast_fallback_used": False,
            "vast_destroyed": [],
        },
    )
    ok = tear_down_and_record(object(), "lab-g3", store, m.job_id, cloud="gcp")
    assert ok is False
    reason = store.read_manifest(m.job_id).end_reason or ""
    assert "gcp" in reason
    assert "vastai destroy_instance" not in reason and "vast-sdk" not in reason
    assert "lab reconcile" in reason
