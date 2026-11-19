"""A GCE instance's ``lab-job-id`` label decides which cluster it belongs to (review finding).

GCP truncates the cluster name to 35 characters on the cloud. With a 12-char project slug,
``gcp_name_fragment`` keeps only ``YYYYMM`` of the job id plus SkyPilot's 2-hex md5 digest, so two
jobs of one project in one month collide with p~1/256. Matching on that fragment let job A's
gcp-direct teardown fallback delete sibling B's *running* VM and report ``succeeded``; the same
collision poisoned ``confirm_no_instance``, ``gcp_preemption_state`` and reconcile's
``gcp_instance_orphans``.

``build_task`` stamps ``lab-job-id`` on every GCE instance it launches (``project_labels``), and a
label is not truncated. When an instance carries it, equality with the cluster's job id is
required; the name fragment is only for unlabelled legacy instances.
"""

from __future__ import annotations

from typing import Any

import pytest

import lab.backends.skypilot as sp

pytest.importorskip("sky")
from sky.clouds.gcp import GCP  # noqa: E402
from sky.provision.gcp.instance_utils import _generate_node_name  # noqa: E402
from sky.utils.common_utils import make_cluster_name_on_cloud  # noqa: E402

PROJECT = "tempotron-capacity"


def _colliding_jobs() -> tuple[str, str]:
    """Two job ids of one project, one month, whose GCE name fragments are identical."""
    seen: dict[str, str] = {}
    for day in range(1, 29):
        for n in range(4000):
            jid = f"202611{day:02d}-{n % 24:02d}{(n // 24) % 60:02d}{n % 60:02d}-{n:06x}"
            frag = sp.gcp_name_fragment(sp.cluster_name_for(jid, project=PROJECT))
            if frag in seen and seen[frag] != jid:
                return seen[frag], jid
            seen[frag] = jid
    raise AssertionError("no fragment collision found")


def _instance(job_id: str, *, labelled: bool = True, status: str = "RUNNING") -> dict[str, Any]:
    cluster = sp.cluster_name_for(job_id, project=PROJECT)
    on_cloud = make_cluster_name_on_cloud(cluster, max_length=GCP.max_cluster_name_length())
    return {
        "name": _generate_node_name(on_cloud, "compute", is_head=True),
        "zone": "us-central1-a",
        "status": status,
        "preemptible": False,
        "labels": sp.project_labels(job_id, project=PROJECT) if labelled else {},
    }


class _NotFound(Exception):
    pass


_NotFound.__name__ = "ClusterDoesNotExist"  # matched by type name, as lab._skycompat does


class _Sky:
    def down(self, cluster: str) -> None:
        raise _NotFound(f"Cluster {cluster} does not exist")

    def get(self, x: Any) -> Any:
        return x


@pytest.fixture
def pair() -> tuple[str, str]:
    a, b = _colliding_jobs()
    ca = sp.cluster_name_for(a, project=PROJECT)
    cb = sp.cluster_name_for(b, project=PROJECT)
    assert sp.gcp_name_fragment(ca) == sp.gcp_name_fragment(cb)  # the precondition, pinned
    return a, b


def test_teardown_of_a_does_not_delete_its_labelled_sibling(
    pair: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's reproduction: A's fallback must not touch B's live VM."""
    a, b = pair
    inst_b = _instance(b)
    monkeypatch.setattr(sp, "_get_gcp_compute", lambda: (object(), "proj"))
    monkeypatch.setattr(sp, "list_gcp_instances", lambda *a_, **k: [inst_b])
    deleted: list[str] = []
    monkeypatch.setattr(
        sp, "delete_gcp_instance", lambda name, zone, *a_, **k: deleted.append(name)
    )

    out = sp.robust_teardown(_Sky(), sp.cluster_name_for(a, project=PROJECT), backoffs=(), cloud="gcp")

    assert deleted == []
    assert out["status"] == "succeeded" and out["gcp_destroyed"] == []


def test_a_teardown_still_deletes_its_own_labelled_instance(
    pair: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = pair
    inst_a, inst_b = _instance(a), _instance(b)
    monkeypatch.setattr(sp, "_get_gcp_compute", lambda: (object(), "proj"))
    monkeypatch.setattr(sp, "list_gcp_instances", lambda *a_, **k: [inst_a, inst_b])
    deleted: list[str] = []
    monkeypatch.setattr(
        sp, "delete_gcp_instance", lambda name, zone, *a_, **k: deleted.append(name)
    )

    sp.robust_teardown(_Sky(), sp.cluster_name_for(a, project=PROJECT), backoffs=(), cloud="gcp")

    assert deleted == [inst_a["name"]]


def test_confirmation_and_preemption_probes_ignore_the_sibling(
    pair: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A's post-teardown confirmation and A's preemption probe must read A's VM, not B's."""
    a, b = pair
    ca = sp.cluster_name_for(a, project=PROJECT)
    monkeypatch.setattr(
        sp, "list_gcp_instances", lambda *a_, **k: [_instance(b, status="TERMINATED")]
    )

    assert sp.confirm_no_instance(ca) is True
    assert sp.gcp_preemption_state(ca) is None  # B's TERMINATED VM says nothing about A


def test_reconcile_orphan_pass_does_not_let_a_sibling_shield_a_leak(pair: tuple[str, str]) -> None:
    """B is leaked (no running job); A is live. B's fragment collides with A's, which used to
    count B as 'backing a live job' and hide it from the sweep forever."""
    a, b = pair
    live, leaked = _instance(a), _instance(b)
    running = {sp.cluster_name_for(a, project=PROJECT)}

    assert sp.gcp_instance_orphans([live, leaked], running) == [leaked]


def test_an_unlabelled_legacy_instance_still_matches_by_name(pair: tuple[str, str]) -> None:
    """Instances launched before the labels existed have nothing else to go on."""
    a, _ = pair
    ca = sp.cluster_name_for(a, project=PROJECT)
    legacy = _instance(a, labelled=False)

    assert sp.gcp_instance_orphans([legacy], {ca}) == []
