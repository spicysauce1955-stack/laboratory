"""DigitalOcean needs a provider-direct teardown fallback, like Vast and GCP already have (F2).

When `sky.down` cannot take -- most often because SkyPilot's own registry has lost the cluster
(`ClusterDoesNotExist`) while the droplet is very much alive and billing -- Vast gets a
vastai-sdk-direct destroy and GCP gets a compute-API-direct destroy. DO got neither, so on DO a
lost registration was not "degraded but recoverable", it was a *guaranteed* orphan until a human
noticed and ran `lab reconcile --apply`.

The DO teardown also has a second half the other clouds don't: the block volume. SkyPilot's DO
provisioner attaches a volume named after the cluster, and destroying the droplet alone leaves it
behind, detached and still billing -- the exact residue found on 2026-08-20.

Names here are checked against the real observed shape (`lab-<job_id>-<uuid8>-<suffix>-head`), and
the pydo call signatures against the installed SDK: `droplets.destroy(droplet_id: int)`,
`droplets.list(per_page=..)`, `volumes.delete(volume_id: str)`.
"""

import pytest

from lab.backends import skypilot as m

CLUSTER = "lab-laboratory-20260820-071905-771110"
DROPLET = f"{CLUSTER}-3dd12990-f5bf-head"
VOLUME = f"{CLUSTER}-3dd12990-f5bf-head"


class _Droplets:
    def __init__(self, droplets, destroy_error=None):
        self._droplets = droplets
        self.destroyed: list[int] = []
        self._destroy_error = destroy_error

    def list(self, **kw):
        return {"droplets": self._droplets}

    def destroy(self, droplet_id, **kw):
        if self._destroy_error is not None:
            raise self._destroy_error
        self.destroyed.append(droplet_id)
        self._droplets = [d for d in self._droplets if d["id"] != droplet_id]


class _Volumes:
    def __init__(self, volumes, delete_error=None):
        self._volumes = volumes
        self.deleted: list[str] = []
        self._delete_error = delete_error

    def list(self, **kw):
        return {"volumes": self._volumes}

    def delete(self, volume_id, **kw):
        if self._delete_error is not None:
            raise self._delete_error
        self.deleted.append(volume_id)


class _Client:
    def __init__(self, droplets=(), volumes=(), droplet_error=None, volume_error=None):
        self.droplets = _Droplets(list(droplets), droplet_error)
        self.volumes = _Volumes(list(volumes), volume_error)


class _Sky:
    """A sky whose `down` always fails the way the live incident's did."""

    def __init__(self, exc=None):
        self.exc = exc or RuntimeError("ClusterDoesNotExist: Cluster does not exist.")
        self.downs = 0

    def get(self, x):
        return x

    def down(self, cluster):
        self.downs += 1
        raise self.exc


def _patch(monkeypatch, client):
    monkeypatch.setattr(m, "_get_do_client", lambda: client)
    return client


class TestDoDirectFallback:
    def test_a_lost_cluster_is_destroyed_through_the_do_api(self, monkeypatch):
        client = _patch(
            monkeypatch,
            _Client(
                droplets=[{"id": 593699953, "name": DROPLET}],
                volumes=[{"id": "vol-1", "name": VOLUME, "droplet_ids": []}],
            ),
        )

        out = m.robust_teardown(_Sky(), CLUSTER, cloud="do", backoffs=(0,))

        assert client.droplets.destroyed == [593699953]
        assert client.volumes.deleted == ["vol-1"]
        assert out["status"] == "succeeded"
        assert out["do_fallback_used"] is True

    def test_finding_nothing_is_success_not_failure(self, monkeypatch):
        """Nothing matching means nothing is billing -- that is the outcome we wanted."""
        client = _patch(monkeypatch, _Client(droplets=[], volumes=[]))

        out = m.robust_teardown(_Sky(), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "succeeded"
        assert client.droplets.destroyed == []

    def test_a_droplet_we_cannot_kill_must_alarm(self, monkeypatch):
        """Found-and-failed-to-destroy is a live box we know about: it must alarm (FR-C2)."""
        _patch(
            monkeypatch,
            _Client(
                droplets=[{"id": 1, "name": DROPLET}],
                droplet_error=RuntimeError("422 unprocessable"),
            ),
        )

        out = m.robust_teardown(_Sky(), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "failed"
        assert "422" in (out["error"] or "")

    def test_a_volume_that_will_not_delete_still_alarms(self, monkeypatch):
        """The detached-volume leak is quieter than a droplet but it bills all the same."""
        _patch(
            monkeypatch,
            _Client(
                droplets=[],
                volumes=[{"id": "vol-1", "name": VOLUME, "droplet_ids": []}],
                volume_error=RuntimeError("attached volume cannot be deleted"),
            ),
        )

        out = m.robust_teardown(_Sky(), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "failed"

    def test_another_projects_droplet_is_never_touched(self, monkeypatch):
        """The whole point of 2026-08-20: a `lab-*` name is not proof it is ours."""
        client = _patch(
            monkeypatch,
            _Client(
                droplets=[
                    {"id": 1, "name": DROPLET},
                    {"id": 2, "name": "lab-tempotron-capacity-20260820-084011-911bf3-abc-head"},
                    {"id": 3, "name": "lab-scheduler-scheduler"},
                ]
            ),
        )

        m.robust_teardown(_Sky(), CLUSTER, cloud="do", backoffs=(0,))

        assert client.droplets.destroyed == [1], "only the named cluster's droplet may die"

    def test_an_unreachable_do_api_alarms_rather_than_claiming_success(self, monkeypatch):
        def _boom():
            raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(m, "_get_do_client", _boom)

        out = m.robust_teardown(_Sky(), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "failed"
        assert "401" in (out["error"] or "")

    def test_sky_down_succeeding_never_reaches_the_fallback(self, monkeypatch):
        client = _patch(monkeypatch, _Client(droplets=[{"id": 1, "name": DROPLET}]))

        class _OkSky:
            def get(self, x):
                return x

            def down(self, cluster):
                return None

        out = m.robust_teardown(_OkSky(), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "succeeded"
        assert client.droplets.destroyed == [], "the fallback must stay a fallback"


class TestOtherCloudsUnaffected:
    @pytest.mark.parametrize("cloud", ["vast", "gcp"])
    def test_the_do_fallback_does_not_hijack_other_clouds(self, monkeypatch, cloud):
        _patch(
            monkeypatch,
            _Client(droplets=[{"id": 1, "name": DROPLET}]),
        )
        monkeypatch.setattr(
            m, "_vast_destroy_matching", lambda cluster: ([], [])
        )
        monkeypatch.setattr(m, "_gcp_destroy_matching", lambda cluster: ([], []))

        out = m.robust_teardown(_Sky(), CLUSTER, cloud=cloud, backoffs=(0,))

        assert out.get("do_fallback_used") is not True
        assert out["status"] == "succeeded"


# ---------------------------------------------------------------------------
# P4-a: a successful `sky.down` is not proof the storage is gone.
# ---------------------------------------------------------------------------


class _OkSky:
    """A `sky.down` that returns cleanly, so no fallback ever runs."""

    def get(self, x):
        return x

    def down(self, cluster):
        return None


class TestSuccessfulTeardownStillVerifiesTheVolume:
    """Found live on 2026-08-20, and only findable live.

    A job that failed partway through launch recorded ``teardown_status: "succeeded"`` and left a
    50 GB detached block volume behind -- still present and still billing seventeen minutes later.
    ``sky.down`` returned cleanly, so the DO-direct fallback that removes the volume alongside the
    droplet never ran, and nothing else looked.

    The volume is created *before* the droplet is fully up, so a launch that dies partway is
    precisely the case that strands one. This is the same leak F2 covers, reached from the opposite
    side: F2 is ``sky.down`` failing, this is ``sky.down`` succeeding and being incomplete.
    """

    def test_a_leftover_volume_is_removed_after_a_clean_sky_down(self, monkeypatch):
        client = _patch(
            monkeypatch,
            _Client(volumes=[{"id": "vol-1", "name": VOLUME, "droplet_ids": []}]),
        )

        out = m.robust_teardown(_OkSky(), CLUSTER, cloud="do", backoffs=(0,))

        assert client.volumes.deleted == ["vol-1"]
        assert out["status"] == "succeeded"

    def test_a_leftover_volume_that_will_not_delete_is_a_real_alarm(self, monkeypatch):
        """We found it, we know it is billing, and we could not remove it: that is `failed`."""
        _patch(
            monkeypatch,
            _Client(
                volumes=[{"id": "vol-1", "name": VOLUME, "droplet_ids": []}],
                volume_error=RuntimeError("attached volume cannot be deleted"),
            ),
        )

        out = m.robust_teardown(_OkSky(), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "failed"

    def test_an_unverifiable_volume_state_is_unknown_not_success(self, monkeypatch):
        """The droplet is confirmed gone; the storage is not. That is exactly `unknown` (R10)."""

        def _unreachable(*a, **k):
            raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(m, "_get_do_client", _unreachable)

        out = m.robust_teardown(_OkSky(), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "unknown"

    def test_nothing_left_over_is_plain_success(self, monkeypatch):
        client = _patch(monkeypatch, _Client(volumes=[]))

        out = m.robust_teardown(_OkSky(), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "succeeded"
        assert client.volumes.deleted == []

    def test_another_projects_volume_is_never_touched(self, monkeypatch):
        client = _patch(
            monkeypatch,
            _Client(
                volumes=[
                    {"id": "mine", "name": VOLUME, "droplet_ids": []},
                    {
                        "id": "theirs",
                        "name": "lab-tempotron-capacity-20260820-124800-05befa-abc-head",
                        "droplet_ids": [],
                    },
                ]
            ),
        )

        m.robust_teardown(_OkSky(), CLUSTER, cloud="do", backoffs=(0,))

        assert client.volumes.deleted == ["mine"]

    @pytest.mark.parametrize("cloud", ["vast", "gcp"])
    def test_other_clouds_do_not_pay_for_the_check(self, monkeypatch, cloud):
        """Vast has no block volumes and GCP disks have their own reconcile pass."""

        def _boom(*a, **k):
            raise AssertionError(f"DO client built for a {cloud} teardown")

        monkeypatch.setattr(m, "_get_do_client", _boom)

        assert m.robust_teardown(_OkSky(), CLUSTER, cloud=cloud, backoffs=(0,))["status"] == (
            "succeeded"
        )
