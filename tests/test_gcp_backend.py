"""GCP as a compute cloud: validation, profile composition, teardown/preemption semantics,
and the `--cloud` CLI/MCP surface. Mirrors the structure of test_cpu_backend.py (no network)."""

import pytest

from lab.backends.local import LocalBackend
from lab.core import Lab, LabError, resolve_backend_profile, validate_cloud
from lab.models import JobState, ResourceRequest


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
    # An orphan is only *reported as destroyable* when it is attributable to us, so the fixture
    # has to name a job this project actually ran (see `_claim_finished`).
    _claim_finished(lab, LEAKED_NODE_JOB_ID)
    orphan = f"lab-{LEAKED_NODE_JOB_ID}"
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("No module named 'vastai_sdk'")),
    )
    monkeypatch.setattr(Lab, "_sky_status_orphans", lambda self, running_clusters: [orphan])
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])

    report = lab.reconcile()

    assert report["vast_pass"] == "skipped (vastai-sdk not installed)"
    assert report["orphans"] == [] and report["instances_total"] == 0
    assert report["sky_orphans"] == [orphan]  # the agnostic pass still ran


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

# Real GCE node names, copied verbatim from the live GCP run of 2026-08-11. Do not "tidy" these:
# every segment is load-bearing, and a hand-written approximation is what made the first attempt
# at GCP-LEAK-7 match nothing at all.
#
#   lab-20260811-144501-c5b340-3dd12990-head-c0h9pkx0-compute
#   \_/ \___________________/ \______/ \__/ \______/ \_____/
#    |    cluster_name_for()   user     node  uuid8   node type
#    |    = lab-<job_id>       hash                   (compute|tpu|mig)
#    `-- our prefix
#
# The user hash comes from `make_cluster_name_on_cloud`, which *also* truncates: GCP's limit is 35
# chars and `lab-<job_id>` is 26 against a budget of 35-9=26 — it fits by exactly zero characters.
# One more character in a job id and the name becomes `lab-<trunc>-<2ch>-<userhash>-head-…`, with
# the job id mangled. That is why the predicate anchors on SkyPilot's node suffix rather than on
# the job id: the job id is not reliably present in the name at all.
REAL_NODE = "lab-20260811-144501-c5b340-3dd12990-head-c0h9pkx0-compute"
LEAKED_NODE_JOB_ID = "20260811-182037-cde576"
LEAKED_NODE = f"lab-{LEAKED_NODE_JOB_ID}-3dd12990-head-4oq7tgke-compute"
REAL_WORKER = "lab-20260811-144501-c5b340-3dd12990-worker-3z8x1v0w-compute"
REAL_TRUNCATED_NODE = "lab-85-3dd12990-head-c0h9pkx0-compute"  # the over-length fallback shape


def test_gcp_instance_orphans_flags_untracked_lab_instances():
    from lab.backends.skypilot import gcp_instance_orphans

    dead = "lab-20260812-090000-dead01-head-3dd1aaaa-compute"
    alive = "lab-20260812-100000-a11ced-head-3dd1bbbb-compute"
    instances = [
        {"name": dead, "zone": "us-central1-a", "status": "RUNNING"},
        {"name": alive, "zone": "us-central1-a", "status": "RUNNING"},
        {"name": "someone-else-vm", "zone": "us-central1-a", "status": "RUNNING"},
    ]
    orphans = gcp_instance_orphans(instances, running_clusters={"lab-20260812-100000-a11ced"})
    assert [o["name"] for o in orphans] == [dead]


def test_gcp_disk_orphans_flags_unattached_lab_disks():
    """A persistent disk that outlived its VM keeps billing — the GCP analogue of the DO
    volume-leak pass. Attached disks die with their instance teardown; only unattached ones leak."""
    from lab.backends.skypilot import gcp_disk_orphans

    dead = "lab-20260812-090000-dead01-head-3dd1aaaa-compute"
    attached = "lab-20260812-100000-a11ced-head-3dd1bbbb-compute"
    disks = [
        {"name": dead, "zone": "us-central1-a", "users": []},
        {"name": attached, "zone": "us-central1-a",
         "users": [f"projects/p/zones/z/instances/{attached}"]},
        {"name": "someone-else-disk", "zone": "us-central1-a", "users": []},
    ]
    orphans = gcp_disk_orphans(disks, running_clusters=set())
    assert [o["name"] for o in orphans] == [dead]


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
    assert out == [
        {"name": "lab-x-1a2b-head", "zone": "us-central1-a", "status": "RUNNING",
         "preemptible": False, "labels": {}}  # unlabelled instance -> unattributed, never "ours"
    ]


class _SkyDownFails:
    def down(self, cluster):
        raise RuntimeError("sky.down boom")

    def get(self, x):
        return x


def test_robust_teardown_gcp_uses_gcp_fallback(monkeypatch):
    monkeypatch.setattr(
        "lab.backends.skypilot._gcp_destroy_matching", lambda c: (["lab-x-1a2b-head"], [])
    )
    from lab.backends.skypilot import robust_teardown

    out = robust_teardown(_SkyDownFails(), "lab-x", backoffs=(), cloud="gcp")
    assert out["status"] == "succeeded"
    assert out["gcp_destroyed"] == ["lab-x-1a2b-head"]


def test_robust_teardown_gcp_reports_failed_when_a_destroy_fails(monkeypatch):
    """GCP-LEAK-4: 'destroyed-or-none-found are both safe outcomes' has a third case —
    found-and-failed-to-destroy — which is NOT safe. Reporting it as succeeded writes
    teardown_status='succeeded' onto the manifest and exits `lab wait` 0 while the VM bills on:
    the exact false-clean FR-C2 exists to prevent."""
    from lab.backends.skypilot import robust_teardown

    monkeypatch.setattr(
        "lab.backends.skypilot._gcp_destroy_matching",
        lambda c: ([], ["lab-x-1a2b-head: 403 Forbidden"]),
    )
    out = robust_teardown(_SkyDownFails(), "lab-x", backoffs=(), cloud="gcp")
    assert out["status"] == "failed"
    assert "403 Forbidden" in (out["error"] or "")


def test_robust_teardown_gcp_partial_destroy_is_failed(monkeypatch):
    """One of two instances destroyed is still a leak."""
    from lab.backends.skypilot import robust_teardown

    monkeypatch.setattr(
        "lab.backends.skypilot._gcp_destroy_matching",
        lambda c: (["lab-x-1a2b-head"], ["lab-x-1a2b-worker: still running"]),
    )
    out = robust_teardown(_SkyDownFails(), "lab-x", backoffs=(), cloud="gcp")
    assert out["status"] == "failed"
    assert out["gcp_destroyed"] == ["lab-x-1a2b-head"]  # report what DID die, still alarm


def test_gcp_destroy_matching_collects_failures(monkeypatch):
    """A delete that raises must surface as a failure, not vanish into a print()."""
    from lab.backends.skypilot import _gcp_destroy_matching

    monkeypatch.setattr(
        "lab.backends.skypilot._get_gcp_compute", lambda: (object(), "proj")
    )
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances",
        lambda *a, **k: [{"name": "lab-x-1a2b-head", "zone": "us-central1-a", "status": "RUNNING"}],
    )
    monkeypatch.setattr(
        "lab.backends.skypilot.delete_gcp_instance", _raise(RuntimeError("403 Forbidden"))
    )
    destroyed, failures = _gcp_destroy_matching("lab-x")
    assert destroyed == []
    assert len(failures) == 1 and "403 Forbidden" in failures[0]


# --- GCP-LEAK-6: a delete request is not a completed delete ----------------------------------


class _FakeOps:
    """GCE zonal operations. `wait` blocks server-side until the op is DONE, then returns it."""

    def __init__(self, result):
        self._result = result
        self.waited = []

    def wait(self, project, zone, operation):
        self.waited.append(operation)
        return _Executable(self._result)


class _Executable:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeCompute:
    def __init__(self, op_result):
        self.ops = _FakeOps(op_result)
        self.deleted = []

    def instances(self):
        return self

    def disks(self):
        return self

    def delete(self, project, zone, **kw):
        self.deleted.append(kw)
        return _Executable({"name": "operation-1", "status": "PENDING"})

    def zoneOperations(self):  # noqa: N802 — mirrors googleapiclient
        return self.ops


def test_delete_gcp_instance_raises_when_the_operation_fails(monkeypatch):
    """GCP-LEAK-6: `instances().delete().execute()` returns an Operation, not a completed delete.
    GCE deletes take 30-60s and can fail AFTER acceptance (RESOURCE_IN_USE, a stuck zonal op).
    Treating the 202 as success reports a destroyed VM that is still RUNNING — and every
    downstream signal (teardown_status, `lab wait`'s exit code, the dashboard) inherits the lie."""
    from lab.backends.skypilot import delete_gcp_instance

    compute = _FakeCompute(
        {"status": "DONE", "error": {"errors": [{"message": "RESOURCE_IN_USE_BY_ANOTHER_RESOURCE"}]}}
    )
    with pytest.raises(RuntimeError, match="RESOURCE_IN_USE"):
        delete_gcp_instance("lab-x-1a2b-head", "us-central1-a", compute, "proj")


def test_delete_gcp_instance_succeeds_when_the_operation_completes(monkeypatch):
    from lab.backends.skypilot import delete_gcp_instance

    compute = _FakeCompute({"status": "DONE"})
    delete_gcp_instance("lab-x-1a2b-head", "us-central1-a", compute, "proj")
    assert compute.ops.waited == ["operation-1"]  # it actually waited for completion


def test_delete_gcp_disk_raises_when_the_operation_fails():
    from lab.backends.skypilot import delete_gcp_disk

    compute = _FakeCompute({"status": "DONE", "error": {"errors": [{"message": "diskInUse"}]}})
    with pytest.raises(RuntimeError, match="diskInUse"):
        delete_gcp_disk("lab-x-1a2b-head", "us-central1-a", compute, "proj")


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
    _claim_finished(lab, LEAKED_NODE_JOB_ID)
    monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
    monkeypatch.setattr(Lab, "_sky_status_orphans", lambda self, running_clusters: [])
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances",
        lambda *a, **k: [{"name": LEAKED_NODE, "zone": "us-central1-a",
                          "status": "RUNNING"}],
    )
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_disks",
        lambda *a, **k: [{"name": LEAKED_NODE, "zone": "us-central1-a", "users": []}],
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
    assert [o["name"] for o in report["gcp_orphans"]] == [LEAKED_NODE]
    assert report["gcp_destroyed"] == [LEAKED_NODE]
    assert [o["name"] for o in report["gcp_disk_orphans"]] == [LEAKED_NODE]
    assert report["gcp_disks_destroyed"] == [LEAKED_NODE]
    assert ("inst", LEAKED_NODE, "us-central1-a") in deleted
    assert ("disk", LEAKED_NODE, "us-central1-a") in deleted


def _claim_finished(lab, job_id):
    """Record `job_id` as a finished job of *this* project.

    Since the 2026-08-20 incident a `lab-*` resource is only destroyable when reconcile can prove
    this project owns it; an id with no record anywhere is `unattributed` and deliberately left
    alone. So the realistic leak — and what these tests mean by "orphan" — is a job that *was*
    ours and is no longer running, whose machine or disk outlived it. `JobStore.create` claims the
    id in the machine-wide index, which is what makes the resource attributable.
    """
    from helpers import make_manifest

    lab.store.create(
        make_manifest(job_id, "python x.py", timeout="1h").model_copy(
            update={"status": JobState.succeeded}
        )
    )


def _lab_with_other_passes_clean(tmp_path, monkeypatch):
    """A Lab whose Vast/sky/DO passes are all clean, so only the GCP passes are under test."""
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
    _claim_finished(lab, LEAKED_NODE_JOB_ID)
    monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
    monkeypatch.setattr(Lab, "_sky_status_orphans", lambda self, running_clusters: [])
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
    return lab


def _raise(exc):
    def _boom(*a, **k):
        raise exc

    return _boom


def test_reconcile_gcp_pass_skips_when_unconfigured(tmp_path, monkeypatch):
    """GCP genuinely not set up on this machine: skip the pass, but SAY SO in the report."""
    from lab.backends.skypilot import GcpNotConfigured

    lab = _lab_with_other_passes_clean(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances", _raise(GcpNotConfigured("no ADC"))
    )
    report = lab.reconcile()
    assert report["gcp_orphans"] == [] and report["gcp_destroyed"] == []
    assert "skipped" in report["gcp_pass"]


def test_reconcile_gcp_api_failure_raises_instead_of_reporting_clean(tmp_path, monkeypatch):
    """GCP-LEAK-2: a revoked role / expired key / disabled API is NOT 'GCP not configured'.
    Swallowing it makes a leak-detection command report clean while blind — worse than having no
    pass at all, because the report claims coverage. Mirrors the Vast pass, which raises on
    anything that isn't a missing SDK."""
    lab = _lab_with_other_passes_clean(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances",
        _raise(RuntimeError("403 Forbidden: compute.instances.list denied")),
    )
    with pytest.raises(LabError, match="GCP"):
        lab.reconcile()


def test_reconcile_gcp_disk_pass_runs_even_when_the_instance_pass_is_skipped(
    tmp_path, monkeypatch
):
    """GCP-LEAK-3: the disk pass must not be nested inside the instance pass. An unattached disk
    is the slow, quiet leak — it survives every instance-level cleanup and bills forever."""
    from lab.backends.skypilot import GcpNotConfigured

    lab = _lab_with_other_passes_clean(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances", _raise(GcpNotConfigured("no compute scope"))
    )
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_disks",
        lambda *a, **k: [{"name": LEAKED_NODE, "zone": "us-central1-a", "users": []}],
    )
    report = lab.reconcile()
    assert [d["name"] for d in report["gcp_disk_orphans"]] == [LEAKED_NODE]


def test_reconcile_gcp_disk_api_failure_raises(tmp_path, monkeypatch):
    """A disk-listing failure must not read as 'no leaked disks'."""
    lab = _lab_with_other_passes_clean(tmp_path, monkeypatch)
    monkeypatch.setattr("lab.backends.skypilot.list_gcp_instances", lambda *a, **k: [])
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_disks", _raise(RuntimeError("500 backendError"))
    )
    with pytest.raises(LabError, match="GCP"):
        lab.reconcile()


# --- GCP-LEAK-7: `lab-` matching is too broad, and unanchored to a project -------------------


def test_a_bare_lab_prefixed_vm_is_not_an_orphan():
    """The destructive false positive. `reconcile --apply` deletes without prompting, so in a
    shared project someone's `lab-notebook` was a VM we would silently destroy. Only SkyPilot's
    real node shape — cluster + `-head|worker-<uuid>-<node type>` — is ours to delete."""
    from lab.backends.skypilot import gcp_instance_orphans

    instances = [
        {"name": "lab-notebook", "zone": "us-central1-a", "status": "RUNNING"},
        {"name": "lab-shared-jupyter", "zone": "us-central1-a", "status": "RUNNING"},
        {"name": REAL_NODE, "zone": "us-central1-a", "status": "RUNNING"},
    ]
    assert [o["name"] for o in gcp_instance_orphans(instances, set())] == [REAL_NODE]


def test_a_bare_lab_prefixed_disk_is_not_an_orphan():
    """Same narrowing for disks: a GCE boot disk inherits its instance's name, so an unattached
    disk we may delete carries the same node shape. `lab-notebook`'s data disk does not."""
    from lab.backends.skypilot import gcp_disk_orphans

    disks = [
        {"name": "lab-notebook", "zone": "us-central1-a", "users": []},
        {"name": REAL_NODE, "zone": "us-central1-a", "users": []},
    ]
    assert [o["name"] for o in gcp_disk_orphans(disks, set())] == [REAL_NODE]


def test_worker_nodes_match_too():
    from lab.backends.skypilot import gcp_instance_orphans

    instances = [{"name": REAL_WORKER, "zone": "us-central1-a", "status": "RUNNING"}]
    assert [o["name"] for o in gcp_instance_orphans(instances, set())] == [REAL_WORKER]


def test_the_predicate_accepts_names_skypilot_itself_generates():
    """The narrowing is only safe while it still recognises our own clusters, and a leak pass that
    silently matches nothing reports clean forever.

    So this builds the name the way SkyPilot really builds it — `make_cluster_name_on_cloud` at
    GCP's own length limit, then `_generate_node_name` — from a freshly generated job id, rather
    than from our belief about the shape. The first attempt at GCP-LEAK-7 hand-wrote the expected
    name, omitted the user hash, and matched none of the two real instances the live run produced;
    a self-confirming test did not catch it. This one would have.
    """
    from sky.clouds.gcp import GCP
    from sky.provision.gcp.instance_utils import GCPNodeType, _generate_node_name
    from sky.utils.common_utils import make_cluster_name_on_cloud

    from lab.backends.skypilot import cluster_name_for, is_lab_cluster_node
    from lab.core import _new_job_id

    on_cloud = make_cluster_name_on_cloud(
        cluster_name_for(_new_job_id()), max_length=GCP.max_cluster_name_length()
    )
    # Enumerate the real enum, not a hand-copied list: a node type SkyPilot adds later fails here
    # rather than silently going unmatched by the predicate.
    for node_type in GCPNodeType:
        for is_head in (True, False):
            name = _generate_node_name(on_cloud, node_type.value, is_head=is_head)
            assert is_lab_cluster_node(name), name


@pytest.mark.parametrize(
    "name",
    [
        "lab-notebook",  # the shared-project VM that started GCP-LEAK-7
        "lab-shared-jupyter",
        "lab-ml-worker-2-gpu",  # human-named and *shaped* like a node: uuid too short, bad type
        "lab-run-head-c0h9pkx0-gpu",  # `gpu` is not a GCPNodeType
        "lab-run-head-short-compute",  # uuid is not 8 chars
        "notlab-20260811-144501-c5b340-3dd12990-head-c0h9pkx0-compute",  # not our prefix
    ],
)
def test_the_predicate_rejects_things_that_are_not_ours(name):
    """Everything the predicate accepts is something `reconcile --apply` deletes without asking,
    so the near misses matter as much as the hits."""
    from lab.backends.skypilot import is_lab_cluster_node

    assert not is_lab_cluster_node(name), name


def test_the_predicate_accepts_the_real_names_from_the_live_run():
    """Belt and braces on the generator above: the exact strings GCE reported on 2026-08-11,
    including the over-length truncated shape, which drops the job id from the name entirely."""
    from lab.backends.skypilot import is_lab_cluster_node

    for name in (REAL_NODE, LEAKED_NODE, REAL_WORKER, REAL_TRUNCATED_NODE):
        assert is_lab_cluster_node(name), name


def test_unmatched_lab_names_are_reported_but_never_destroyed(tmp_path, monkeypatch):
    """The narrowing's own safety net: a `lab-*` resource we no longer claim is still surfaced,
    so a real leak in an unexpected shape is visible rather than silently dropped. It is
    advisory — it must not land in the orphan lists that `--apply` destroys."""
    lab = _lab_with_other_passes_clean(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances",
        lambda *a, **k: [{"name": "lab-notebook", "zone": "us-central1-a", "status": "RUNNING"}],
    )
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_disks",
        lambda *a, **k: [{"name": "lab-notebook-data", "zone": "us-central1-a", "users": []}],
    )
    monkeypatch.setattr(
        "lab.backends.skypilot.delete_gcp_instance", _raise(AssertionError("must not destroy"))
    )
    monkeypatch.setattr(
        "lab.backends.skypilot.delete_gcp_disk", _raise(AssertionError("must not destroy"))
    )

    report = lab.reconcile(apply=True)
    assert report["gcp_orphans"] == [] and report["gcp_disk_orphans"] == []
    assert sorted(report["gcp_unmatched"]) == ["lab-notebook", "lab-notebook-data"]


def test_reconcile_reports_the_project_it_swept(tmp_path, monkeypatch):
    """Unanchored: the project comes from ambient ADC, while SkyPilot can be pinned to a
    different one in ~/.sky/config.yaml. A reconcile of the wrong project reports clean and is
    indistinguishable from a real all-clear — unless the report says which project it swept."""
    lab = _lab_with_other_passes_clean(tmp_path, monkeypatch)
    monkeypatch.setattr("lab.backends.skypilot.list_gcp_instances", lambda *a, **k: [])
    monkeypatch.setattr("lab.backends.skypilot.list_gcp_disks", lambda *a, **k: [])
    monkeypatch.setattr(
        "lab.backends.skypilot._gcp_default_credentials", lambda: (object(), "swept-project")
    )
    assert lab.reconcile()["gcp_project"] == "swept-project"


def test_reconcile_reports_a_null_project_when_gcp_is_unconfigured(tmp_path, monkeypatch):
    from lab.backends.skypilot import GcpNotConfigured

    lab = _lab_with_other_passes_clean(tmp_path, monkeypatch)
    monkeypatch.setattr("lab.backends.skypilot.list_gcp_instances", lambda *a, **k: [])
    monkeypatch.setattr("lab.backends.skypilot.list_gcp_disks", lambda *a, **k: [])
    monkeypatch.setattr(
        "lab.backends.skypilot._gcp_default_credentials", _raise(GcpNotConfigured("no ADC"))
    )
    assert lab.reconcile()["gcp_project"] is None


def test_get_gcp_compute_raises_not_configured_without_a_project(monkeypatch):
    """The missing-project case is a configuration gap, not an API failure — it must be
    distinguishable, or the reconcile pass cannot tell 'skip me' from 'I am blind'."""
    import lab.backends.skypilot as sky_mod
    from lab.backends.skypilot import GcpNotConfigured

    monkeypatch.setattr(sky_mod, "_gcp_default_credentials", lambda: (object(), None))
    with pytest.raises(GcpNotConfigured, match="project"):
        sky_mod._get_gcp_compute()


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


def _clean_reconcile_report(**overrides: object) -> dict:
    """A reconcile report with every pass clean — overridden one list at a time below."""
    report: dict = {
        "vast_pass": "ran", "gcp_pass": "ran", "gcp_disk_pass": "ran",
        "instances_total": 0, "unsupervised": [],
        "orphans": [], "destroyed": [], "ghosts": [],
        "sky_orphans": [], "sky_destroyed": [],
        "do_volume_orphans": [], "do_volumes_destroyed": [],
        "gcp_orphans": [], "gcp_destroyed": [],
        "gcp_disk_orphans": [], "gcp_disks_destroyed": [],
        "applied": False,
    }
    report.update(overrides)
    return report


def _reconcile_exit_code(report: dict) -> int:
    from unittest.mock import MagicMock, patch

    from typer.testing import CliRunner

    import lab.cli as cli_mod
    from lab.cli import app

    fake = MagicMock()
    fake.reconcile.return_value = report
    with patch.object(cli_mod, "_lab", return_value=fake):
        return CliRunner().invoke(app, ["reconcile"]).exit_code


@pytest.mark.parametrize(
    "leak_field",
    ["orphans", "sky_orphans", "gcp_orphans", "gcp_disk_orphans", "do_volume_orphans"],
)
def test_cli_reconcile_exits_3_for_every_orphan_list(leak_field):
    """GCP-LEAK-1: the dry-run leak alarm must read EVERY orphan pass, not just the two it was
    born with. The GCP compute pass exists for the case where SkyPilot's registry lost the
    cluster — exactly when `sky_orphans` is empty and `gcp_orphans` is the only non-empty list,
    so wiring only the first two silences the alarm in the one scenario the pass was written for.
    """
    assert _reconcile_exit_code(_clean_reconcile_report(**{leak_field: ["lab-leaked"]})) == 3


def test_cli_reconcile_exits_0_when_every_pass_is_clean():
    assert _reconcile_exit_code(_clean_reconcile_report()) == 0


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


def _no_vast(monkeypatch):
    """Any touch of the Vast listing while handling a GCP job is the :367 bug."""
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances",
        _raise(AssertionError("vast listing used for gcp")),
    )


def test_preempted_teardown_confirmed_gcp_true_when_instance_is_gone(monkeypatch):
    """GCP-LEAK-5: teardown runs BEFORE this check, so a surviving instance record means the
    teardown didn't take. GCP has a provider-direct listing — the same one `reconcile` and the
    teardown fallback already use — so unlike DO it can give the second opinion Vast gets."""
    from lab.backends.skypilot import preempted_teardown_confirmed

    _no_vast(monkeypatch)
    monkeypatch.setattr("lab.backends.skypilot.list_gcp_instances", lambda *a, **k: [])
    assert preempted_teardown_confirmed("gcp", "lab-x") is True


def test_preempted_teardown_confirmed_gcp_false_when_instance_survives(monkeypatch):
    """The likeliest way a GCP box outlives its job is an unmanaged spot preemption — which is
    exactly the path that used to skip confirmation entirely and return True."""
    from lab.backends.skypilot import preempted_teardown_confirmed

    _no_vast(monkeypatch)
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances",
        lambda *a, **k: [
            {"name": "lab-x-1a2b-head", "zone": "us-central1-a", "status": "RUNNING"}
        ],
    )
    assert preempted_teardown_confirmed("gcp", "lab-x") is False


def test_preempted_teardown_confirmed_gcp_false_when_listing_fails(monkeypatch):
    """Uncertainty must read as 'still maybe billing' — the contract confirm_no_rental holds."""
    from lab.backends.skypilot import preempted_teardown_confirmed

    _no_vast(monkeypatch)
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances", _raise(RuntimeError("api down"))
    )
    assert preempted_teardown_confirmed("gcp", "lab-x") is False


def test_preempted_teardown_confirmed_do_stays_optimistic(monkeypatch):
    """DO has no provider-direct listing, so tear_down_and_record's outcome is the only answer."""
    from lab.backends.skypilot import preempted_teardown_confirmed

    _no_vast(monkeypatch)
    assert preempted_teardown_confirmed("do", "lab-x") is True


def test_preempted_teardown_confirmed_vast_delegates(monkeypatch):
    from lab.backends.skypilot import preempted_teardown_confirmed

    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances",
        lambda *a, **k: [{"label": "lab-x still here"}],
    )
    assert preempted_teardown_confirmed("vast", "lab-x") is False


# --- GCP-PROV-3: diagnose the actual failure, don't hand back a leaflet ------------------------
#
# Vast got a *dynamic* diagnosis out of LAB-BUGS §8 (consult the balance, say "top up"). GCP got a
# fixed string listing three possible causes, returned regardless of what happened. Every real
# cause is unambiguously identifiable from the error text we already hold — and the one we hit
# live (capacity) surfaced as "Failed to set up SkyPilot runtime", which reads like a lab bug.


@pytest.mark.parametrize(
    ("launch_error", "remedy"),
    [
        # Assert on the REMEDY, never on a word that also appears in the error text — otherwise
        # echoing the error back verbatim would pass the test without diagnosing anything.
        (
            "ZONE_RESOURCE_POOL_EXHAUSTED: The zone 'us-central1-a' does not have enough "
            "resources available to fulfill the request",
            "--spot",
        ),
        (
            "Quota 'N4_CPUS' exceeded. Limit: 24.0 in region us-central1",
            "quota increase",
        ),
        (
            "Compute Engine API has not been used in project 12345 before or it is disabled",
            "gcloud services enable",
        ),
        (
            "The billing account for the owning project is disabled in state absent",
            "enable billing",
        ),
    ],
)
def test_provision_failure_reason_gcp_names_the_actual_cause(launch_error, remedy):
    import lab.sky_runner as sky_runner

    msg = sky_runner.provision_failure_reason(f"launch error: {launch_error}", "gcp")
    assert remedy in msg.lower()


def test_provision_failure_reason_gcp_capacity_suggests_the_workaround():
    """The lab exposes no region/zone override (GCP-PROV-1), so the only lever a user has when a
    zone is exhausted is re-pricing the optimizer's search with --spot. Say so."""
    import lab.sky_runner as sky_runner

    msg = sky_runner.provision_failure_reason(
        "launch error: ZONE_RESOURCE_POOL_EXHAUSTED", "gcp"
    )
    assert "--spot" in msg


def test_provision_failure_reason_gcp_keeps_the_original_error():
    """The diagnosis annotates; it never swallows the message it diagnosed."""
    import lab.sky_runner as sky_runner

    msg = sky_runner.provision_failure_reason("launch error: QUOTA_EXCEEDED for L4", "gcp")
    assert "QUOTA_EXCEEDED for L4" in msg


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
    _no_vast(monkeypatch)
    # The GCP confirm consults the compute API; stub it so the test stays hermetic (an unstubbed
    # listing would reach the real project, and its answer would depend on the dev's ADC).
    monkeypatch.setattr("lab.backends.skypilot.list_gcp_instances", lambda *a, **k: [])

    rc = runner_mod.run_job(home / "g4", adopt=True)

    final = store.read_manifest("g4")
    assert final.status is JobState.preempted
    assert rc == 0  # teardown succeeded and stays succeeded
    assert final.end_reason != "preempted but teardown unconfirmed — see `lab reconcile`"


# --- GCP-PREEMPT-1: GCE reports preemption; stop inferring it --------------------------------


def test_gce_terminal_state_reads_preemption_off_a_terminated_spot_vm():
    from lab.preemption import gcp_terminal_state

    stopped_spot = [{"name": REAL_NODE, "status": "TERMINATED", "preemptible": True}]
    assert gcp_terminal_state(stopped_spot) is JobState.preempted


def test_gce_terminal_state_calls_a_non_preemptible_stop_a_failure():
    """The direction that costs money: a genuinely failed job whose box happened to stop is not
    a preemption, and must not be auto-resubmitted."""
    from lab.preemption import gcp_terminal_state

    stopped = [{"name": REAL_NODE, "status": "TERMINATED", "preemptible": False}]
    assert gcp_terminal_state(stopped) is JobState.failed


def test_a_real_gcp_preemption_leaves_nothing_for_the_probe_to_read():
    """The limit of the probe, pinned so nobody over-trusts it.

    SkyPilot's GCP spot config sets `instanceTerminationAction: DELETE` (templates/gcp-ray.yml.j2),
    so GCE **deletes** a preempted VM rather than leaving it TERMINATED — the evidence the probe
    was meant to read is destroyed by the very event it is trying to confirm. The probe therefore
    abstains and today's inference carries the classification, which lands on `preempted` anyway.

    The consequence: on GCP spot, `preemptible` is always true whenever an instance *is* readable,
    so the `failed` branch cannot fire for a spot job, and the probe cannot yet distinguish a real
    preemption from a non-preemption stop. It is safe (never worse than the inference) but it does
    not yet deliver GCP-PREEMPT-1's benefit. The authoritative record that survives the delete is
    the zone operations log (`operationType = compute.instances.preempted`) — see the follow-up in
    docs/proposals/2026-08-12-gcp-remaining-gaps.md.
    """
    from lab.preemption import gcp_terminal_state

    assert gcp_terminal_state([]) is None


def test_gce_terminal_state_says_nothing_when_there_is_nothing_to_read():
    """No instance left (already deleted) or still running: GCE has no authoritative answer, so
    the probe must abstain rather than invent one — the caller keeps today's inference."""
    from lab.preemption import gcp_terminal_state

    assert gcp_terminal_state([]) is None
    assert gcp_terminal_state([{"name": REAL_NODE, "status": "RUNNING"}]) is None


def test_list_gcp_instances_carries_the_preemptible_flag():
    from lab.backends.skypilot import list_gcp_instances

    class _Req:
        def execute(self):
            return {
                "items": {
                    "zones/us-central1-a": {
                        "instances": [
                            {
                                "name": REAL_NODE,
                                "zone": "https://.../zones/us-central1-a",
                                "status": "TERMINATED",
                                "scheduling": {"preemptible": True},
                            }
                        ]
                    }
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

    assert list_gcp_instances(_Compute(), "proj")[0]["preemptible"] is True


def _gcp_spot_job_whose_box_vanished(tmp_path, monkeypatch, job_id):
    """A gcp spot job that reached no terminal state and whose cluster is gone — the exact input
    the classifier could previously only *infer* about. Returns its JobStore."""
    import sys
    import types

    import lab.sky_runner as runner_mod
    from helpers import make_manifest
    from lab._util import now
    from lab.backends.skypilot import cluster_name_for
    from lab.models import BackendInfo, CostInfo
    from lab.store import JobStore

    store = JobStore(tmp_path / "runs")
    m = make_manifest(job_id, "python x.py", timeout="1h").model_copy(
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
    # The name actually launched under, which the supervisor now trusts over recomputing it.
    # Must agree with the instance names the tests mock, or the fixture tests nothing.
    store.write_runtime(job_id, runner_pid=1, cluster=cluster_name_for(job_id))

    fake_sky = types.ModuleType("sky")
    monkeypatch.setitem(sys.modules, "sky", fake_sky)
    monkeypatch.setattr(fake_sky, "tail_logs", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner_mod, "_wait_terminal", lambda *a, **k: (JobState.failed, False))
    monkeypatch.setattr(runner_mod, "_cluster_up", lambda *a, **k: False)
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)
    _no_vast(monkeypatch)
    return store


def test_a_non_preempted_spot_failure_is_not_resubmitted(tmp_path, monkeypatch):
    """GCP-PREEMPT-1's failure mode, end to end. Spot + box gone + no terminal used to infer
    `preempted`, which the scheduler auto-resubmits — so a job that genuinely failed got paid for
    twice. GCE says outright that this VM was not preemptible, and that answer wins."""
    import lab.sky_runner as runner_mod
    from lab.backends.skypilot import cluster_name_for

    store = _gcp_spot_job_whose_box_vanished(tmp_path, monkeypatch, "gp1")
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances",
        lambda *a, **k: [
            {"name": f"{cluster_name_for('gp1')}-head-1a2b3c4d-compute", "zone": "us-central1-a",
             "status": "TERMINATED", "preemptible": False}
        ],
    )

    runner_mod.run_job(tmp_path / "runs" / "gp1", adopt=True)

    assert store.read_manifest("gp1").status is JobState.failed


def test_a_real_preemption_is_confirmed_by_gce_rather_than_inferred(tmp_path, monkeypatch):
    import lab.sky_runner as runner_mod
    from lab.backends.skypilot import cluster_name_for

    store = _gcp_spot_job_whose_box_vanished(tmp_path, monkeypatch, "gp2")
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances",
        lambda *a, **k: [
            {"name": f"{cluster_name_for('gp2')}-head-1a2b3c4d-compute", "zone": "us-central1-a",
             "status": "TERMINATED", "preemptible": True}
        ],
    )

    runner_mod.run_job(tmp_path / "runs" / "gp2", adopt=True)

    assert store.read_manifest("gp2").status is JobState.preempted


def test_a_failed_probe_falls_back_to_inference_never_worse(tmp_path, monkeypatch):
    """The probe refines an inference; it must never remove one. A revoked role or a 500 leaves
    the old behaviour exactly as it was."""
    import lab.sky_runner as runner_mod

    store = _gcp_spot_job_whose_box_vanished(tmp_path, monkeypatch, "gp3")
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances", _raise(RuntimeError("403 Forbidden"))
    )

    runner_mod.run_job(tmp_path / "runs" / "gp3", adopt=True)

    assert store.read_manifest("gp3").status is JobState.preempted


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


def test_a_probed_preemption_survives_the_classifier(tmp_path, monkeypatch):
    """Review finding: `classify_terminal` never trusts `sky_state=preempted` directly — it
    reaches `preempted` only via `use_spot and cluster_gone`. The probe set `cluster_gone` but not
    `use_spot`, and `use_spot` comes from `launched_spot`, which the adopt path never records — so
    GCE's authoritative answer could be silently discarded and the job come out `failed`."""
    import lab.sky_runner as runner_mod
    from lab.backends.skypilot import cluster_name_for

    store = _gcp_spot_job_whose_box_vanished(tmp_path, monkeypatch, "gp4")
    # launched_spot absent AND the manifest not marked spot: the classifier would infer `failed`.
    m = store.read_manifest("gp4")
    m.resources.use_spot = False
    m.backend.launched_spot = None
    store.write_manifest(m)
    monkeypatch.setattr(
        "lab.backends.skypilot.list_gcp_instances",
        lambda *a, **k: [
            {"name": f"{cluster_name_for('gp4')}-head-1a2b3c4d-compute", "zone": "us-central1-a",
             "status": "TERMINATED", "preemptible": True}
        ],
    )

    runner_mod.run_job(tmp_path / "runs" / "gp4", adopt=True)

    assert store.read_manifest("gp4").status is JobState.preempted
