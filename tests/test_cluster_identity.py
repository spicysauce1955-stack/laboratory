"""Project identity on the cloud resources a job provisions.

`lab reconcile --apply --yes` once destroyed seven RUNNING clusters belonging to a *different*
project on the same machine (field report 2026-08-20). The root cause these tests pin down is
that a provisioned resource carried no record of who launched it: every project on the box named
its clusters `lab-<job_id>`, and a job id is a timestamp plus randomness — so attribution from
the cloud side was impossible even in principle, not merely unimplemented.

The tests below therefore assert the two durable carriers of that identity: the cluster *name*
(the only one every cloud has, since SkyPilot's name is the one string that reaches Vast, DO and
GCP alike) and, where the cloud supports it, real instance labels. They are deliberately built
against SkyPilot's own validators and naming functions rather than our belief about the shapes —
the lesson of GCP-LEAK-7, where a hand-written expected name matched none of the real instances.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from helpers import make_manifest

from lab.backends.skypilot import (
    CLUSTER_NAME_MAX,
    ClusterIdentity,
    cluster_name_for,
    parse_cluster_name,
    project_labels,
    project_slug,
)

JOB_ID = "20260820-071905-771110"  # the real shape: %Y%m%d-%H%M%S + 6 hex (core._new_job_id)
# SkyPilot cluster names: start with a letter, then lowercase alphanumerics and hyphens.
_VALID_NAME = re.compile(r"^[a-z][a-z0-9-]*$")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the cwd-derived project at a named directory.

    `LAB_REPO_DIR` is the documented machine-local override `manifest.repo_root` already honours
    (the scheduler host's reason for existing), so it is also the honest way to fake "which repo
    am I in" — faking `Path.cwd` would bypass the very resolution under test.
    """

    def _set(name: str) -> str:
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LAB_REPO_DIR", str(root))
        return name

    return _set


def test_new_style_name_round_trips(project):
    """The whole point: a name states its project *and* still yields the job id teardown needs."""
    project("laboratory")
    name = cluster_name_for(JOB_ID)
    assert name == "lab-laboratory-20260820-071905-771110"

    ident = parse_cluster_name(name)
    assert ident == ClusterIdentity(job_id=JOB_ID, project="laboratory")


def test_legacy_name_still_parses_to_its_job_id():
    """Clusters launched before this change are still running in the wild and still bill.

    A parser that only understood the new shape would make every one of them unattributable —
    i.e. it would turn a naming change into a leak.
    """
    ident = parse_cluster_name("lab-20260820-071905-771110")
    assert ident is not None
    assert ident.job_id == JOB_ID
    assert ident.project is None  # unattributable, which is not the same as "not ours"


def test_a_foreign_name_does_not_parse():
    """`lab-notebook` is a real shared-project VM (GCP-LEAK-7); it must not look like a job."""
    assert parse_cluster_name("lab-notebook") is None
    assert parse_cluster_name("notlab-20260820-071905-771110") is None
    assert parse_cluster_name("") is None


def test_long_project_name_keeps_the_cap_and_the_job_id(project):
    """The slug is the part that gives: the job id is what reconcile and teardown match on."""
    project("a-catastrophically-long-monorepo-name-that-nobody-would-choose")
    name = cluster_name_for(JOB_ID)

    assert len(name) <= CLUSTER_NAME_MAX
    assert _VALID_NAME.match(name)
    ident = parse_cluster_name(name)
    assert ident is not None and ident.job_id == JOB_ID


def test_hostile_project_names_produce_a_valid_name(project):
    """Directory names are unconstrained; cluster names are not."""
    for raw in ("My Project", "under_scored", "9lives", "実験室", "-dashes-"):
        project(raw)
        name = cluster_name_for(JOB_ID)
        assert _VALID_NAME.match(name), (raw, name)
        assert len(name) <= CLUSTER_NAME_MAX
        ident = parse_cluster_name(name)
        assert ident is not None and ident.job_id == JOB_ID, (raw, name)


def test_skypilot_accepts_the_names_we_generate(project):
    """Validated by SkyPilot's own checker, not by our reading of its docs."""
    pytest.importorskip("sky")
    from sky.utils.common_utils import check_cluster_name_is_valid

    for raw in ("laboratory", "My Project", "9lives", "実験室"):
        project(raw)
        check_cluster_name_is_valid(cluster_name_for(JOB_ID))  # raises if invalid


def test_different_projects_get_different_names_for_the_same_job_id(project):
    """Two projects can mint the same job id within the same second; they must not collide.

    Prefix-truncated slugs are the trap — `machine-learning-alpha` and `machine-learning-beta`
    share their first 12 characters, so a plain truncation would map both to one name and hand
    `reconcile` back exactly the false match this change exists to remove.
    """
    project("alpha-project")
    a = cluster_name_for(JOB_ID)
    project("beta-project")
    b = cluster_name_for(JOB_ID)
    assert a != b

    project("machine-learning-alpha")
    long_a = cluster_name_for(JOB_ID)
    project("machine-learning-beta")
    long_b = cluster_name_for(JOB_ID)
    assert long_a != long_b

    project("実験室")
    u1 = cluster_name_for(JOB_ID)
    project("研究室")
    assert u1 != cluster_name_for(JOB_ID)  # unrepresentable names must still be distinguishable


def test_explicit_project_beats_the_cwd(project):
    """The scheduler launches jobs for a project that is not its own working directory."""
    project("scheduler-host")
    name = cluster_name_for(JOB_ID, project="tempotron-capacity")
    assert parse_cluster_name(name) == ClusterIdentity(
        job_id=JOB_ID, project=project_slug("tempotron-capacity")
    )
    assert name != cluster_name_for(JOB_ID)


def test_slug_is_stable():
    """Names are matched across processes and across releases; the mapping cannot drift."""
    assert project_slug("laboratory") == "laboratory"
    assert project_slug("My Project") == "my-project"


def test_a_name_with_nothing_representable_still_gets_a_slug():
    """A slug of "" would silently produce the legacy shape and lose the attribution entirely."""
    for raw in ("   ", "実験室", "----"):
        slug = project_slug(raw)
        assert slug and re.match(r"^[a-z0-9-]+$", slug), (raw, slug)
    assert project_slug("実験室") != project_slug("研究室")


def test_a_pathological_job_id_falls_back_to_the_legacy_shape():
    """A job id long enough to eat the whole budget keeps the old behaviour rather than a
    silently truncated, unrecoverable id."""
    name = cluster_name_for("x" * 100, project="laboratory")
    assert name.startswith("lab-") and len(name) <= CLUSTER_NAME_MAX
    assert "laboratory" not in name


def test_labels_carry_the_project_and_the_job_id(project):
    """The name is truncated by the cloud; a label is not (GCP), so both are recorded."""
    project("laboratory")
    labels = project_labels(JOB_ID)
    assert labels == {"lab-project": "laboratory", "lab-job-id": JOB_ID}


def test_gcp_accepts_our_labels(project):
    """GCP is the one cloud of ours that really stores labels — ask its validator, don't assume."""
    pytest.importorskip("sky")
    from sky.clouds.gcp import GCP

    for raw in ("laboratory", "My Project", "9lives", "実験室", "a" * 80):
        project(raw)
        for key, value in project_labels(JOB_ID).items():
            valid, err = GCP.is_label_valid(key, value)
            assert valid, (raw, key, value, err)


def test_build_task_attaches_the_labels(tmp_path: Path, project):
    """`build_task` is the choke point every launch passes through — including the scheduler's,
    which never calls `resolve_backend_profile` (the same reason `effective_disk_gb` lives here).
    """
    pytest.importorskip("sky")

    from lab.backends.skypilot import build_task
    from lab.models import ResourceRequest

    project("laboratory")
    m = make_manifest(JOB_ID, "python x.py", resources=ResourceRequest(cloud="gcp"))
    task = build_task(m, tmp_path)
    for res in task.resources:
        assert res.labels == {"lab-project": "laboratory", "lab-job-id": JOB_ID}


def test_the_gcp_node_predicate_still_matches_the_new_shape(project):
    """The new name flows through SkyPilot's on-cloud mangling before reconcile ever sees it.

    GCP truncates to 35 characters, so the *instance* name loses the tail — this asserts the leak
    predicate still recognises what SkyPilot actually produces from a slugged name, built with
    SkyPilot's own functions rather than a hand-written expectation.
    """
    pytest.importorskip("sky")
    from sky.clouds.gcp import GCP
    from sky.provision.gcp.instance_utils import GCPNodeType, _generate_node_name
    from sky.utils.common_utils import make_cluster_name_on_cloud

    from lab.backends.skypilot import is_lab_cluster_node

    project("laboratory")
    on_cloud = make_cluster_name_on_cloud(
        cluster_name_for(JOB_ID), max_length=GCP.max_cluster_name_length()
    )
    for node_type in GCPNodeType:
        for is_head in (True, False):
            name = _generate_node_name(on_cloud, node_type.value, is_head=is_head)
            assert is_lab_cluster_node(name), name


def test_a_live_gcp_instance_still_matches_its_running_cluster(project):
    """The regression the slug would otherwise cause, and the one that costs money.

    GCP truncates the cluster name to 35 characters on the cloud. `lab-<job_id>` fit that budget
    by exactly zero characters, so a plain substring test used to work; a slug spends that zero.
    If the match breaks, every *live* GCP box stops looking like it backs a running job and
    `reconcile --apply` destroys it — the 2026-08-20 incident, recreated by its own fix.
    """
    pytest.importorskip("sky")
    from sky.clouds.gcp import GCP
    from sky.provision.gcp.instance_utils import _generate_node_name
    from sky.utils.common_utils import make_cluster_name_on_cloud

    from lab.backends.skypilot import gcp_instance_orphans, gcp_name_matches

    project("laboratory")
    cluster = cluster_name_for(JOB_ID)
    on_cloud = make_cluster_name_on_cloud(cluster, max_length=GCP.max_cluster_name_length())
    live = _generate_node_name(on_cloud, "compute", is_head=True)
    assert cluster not in live  # the naive substring test really is broken — that is the point
    assert gcp_name_matches(cluster, live)
    assert gcp_instance_orphans([{"name": live}], {cluster}) == []

    # ...and a *different* job in the same project, same day, is still an orphan: the fragment we
    # match on must not degrade into "anything launched today".
    other = cluster_name_for("20260820-235959-aaaaaa")
    other_live = _generate_node_name(
        make_cluster_name_on_cloud(other, max_length=GCP.max_cluster_name_length()),
        "compute",
        is_head=True,
    )
    assert not gcp_name_matches(cluster, other_live)
    assert gcp_instance_orphans([{"name": other_live}], {cluster}) == [{"name": other_live}]


def test_a_legacy_gcp_instance_still_matches_its_cluster():
    """Clusters launched before the slug are still running; their names must still resolve."""
    pytest.importorskip("sky")
    from lab.backends.skypilot import gcp_name_matches

    legacy = f"lab-{JOB_ID}"
    assert gcp_name_matches(legacy, f"lab-{JOB_ID}-3dd12990-head-c0h9pkx0-compute")


def test_gcp_listing_surfaces_the_labels():
    """Stamping identity is only half of it — a leak sweep has to be able to read it back.

    On GCP the label is the *only* full-fidelity copy (the instance name is truncated to 35
    characters), so `list_gcp_instances` has to carry it through or the stamp is write-only.
    """
    from lab.backends.skypilot import list_gcp_instances

    class _Req:
        def execute(self):
            return {
                "items": {
                    "zones/us-central1-a": {
                        "instances": [
                            {
                                "name": "lab-laboratory-20260820-ef-3dd12990-head-c0h9pkx0-compute",
                                "zone": "https://x/zones/us-central1-a",
                                "status": "RUNNING",
                                "labels": {"lab-project": "laboratory", "lab-job-id": JOB_ID},
                            },
                            {"name": "someone-else", "zone": "z/zones/us-central1-a"},
                        ]
                    }
                }
            }

    class _Instances:
        def aggregatedList(self, project):
            return _Req()

        def aggregatedList_next(self, previous_request, previous_response):
            return None

    class _Compute:
        def instances(self):
            return _Instances()

    out = list_gcp_instances(_Compute(), "proj")
    assert out[0]["labels"] == {"lab-project": "laboratory", "lab-job-id": JOB_ID}
    assert out[1]["labels"] == {}  # an unlabelled instance reads as unattributed, never as ours
