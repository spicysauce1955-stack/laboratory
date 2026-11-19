"""The DO volume-detach race, on the two paths the 2026-08-23 fix did not cover (review finding).

DO detaches a block volume from a destroyed droplet *asynchronously*, so a delete issued the
instant the droplet goes is refused ("attached volume cannot be deleted") for a volume seconds
away from deletable. ``_do_sweep_leftover_volumes`` rides that out with a re-listing ladder after
a clean ``sky.down`` (see test_do_volume_detach_race.py). Two paths still raced:

* the **DO-direct teardown fallback** destroyed the droplet and deleted its volume in the same
  instant, so a fallback that worked recorded ``failed`` — ``lab wait`` exit 3 over a volume DO
  removes moments later (R10: the alarm that is usually wrong stops being believed);
* **reconcile's DO volume pass** deletes the volume of a cluster the sky pass just destroyed in
  the same run, with the same race, producing a false exit 5.

Both now go through the same ladder: retry only the detach message, re-list every pass, and
count a volume that vanished as gone.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import lab.backends.skypilot as sp
import lab.core as core_mod
from helpers import make_manifest
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.models import JobState

CLUSTER = "lab-tempotr-5171-20261119-101010-abcdef"
VOLUME = f"{CLUSTER}-3dd12990-ab12-s-4vcpu-8gb"


class ResourceExistsError(Exception):
    """The type pydo raised on 2026-08-23."""


class _Client:
    """A DO account where a volume's delete is refused until ``settle`` more listings happen
    after its droplet is destroyed (or forever, with ``settle=None``)."""

    def __init__(self, *, droplet: bool = True, settle: int | None = 1) -> None:
        self.droplet_alive = droplet
        self.volume_alive = True
        self.lists_since_destroy: int | None = None if droplet else 0
        self.settle = settle
        self.deletes = 0
        self.droplets = types.SimpleNamespace(list=self._list_droplets, destroy=self._destroy)
        self.volumes = types.SimpleNamespace(list=self._list_volumes, delete=self._delete)

    def _list_droplets(self, per_page: int = 200) -> dict[str, Any]:
        return {"droplets": [{"id": 1, "name": VOLUME}] if self.droplet_alive else []}

    def _destroy(self, droplet_id: int) -> None:
        self.droplet_alive = False
        self.lists_since_destroy = 0

    def _list_volumes(self, per_page: int = 200) -> dict[str, Any]:
        if self.lists_since_destroy is not None:
            self.lists_since_destroy += 1
        return {"volumes": [{"id": "v1", "name": VOLUME}] if self.volume_alive else []}

    def _delete(self, volume_id: str) -> None:
        self.deletes += 1
        detached = (
            self.settle is not None
            and self.lists_since_destroy is not None
            and self.lists_since_destroy > self.settle
        )
        if not detached:
            raise ResourceExistsError("failed to delete volume: attached volume cannot be deleted")
        self.volume_alive = False


class _DownFails:
    def down(self, cluster: str) -> None:
        raise RuntimeError("sky.down: API server 503")

    def get(self, x: Any) -> Any:
        return x


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(sp.time, "sleep", calls.append)
    return calls


class TestDoDirectFallback:
    def test_a_detaching_volume_is_waited_out_not_alarmed(
        self, monkeypatch: pytest.MonkeyPatch, slept: list[float]
    ) -> None:
        """The reviewer's reproduction: the fallback works, and must say so."""
        client = _Client()
        monkeypatch.setattr(sp, "_get_do_client", lambda: client)

        out = sp.robust_teardown(_DownFails(), CLUSTER, backoffs=(), cloud="do")

        assert out["status"] == "succeeded", out["error"]
        assert not client.droplet_alive and not client.volume_alive
        assert out["do_destroyed"] == [1, "v1"]

    def test_a_detach_that_never_settles_still_alarms(
        self, monkeypatch: pytest.MonkeyPatch, slept: list[float]
    ) -> None:
        monkeypatch.setattr(sp, "_get_do_client", lambda: _Client(settle=None))

        out = sp.robust_teardown(_DownFails(), CLUSTER, backoffs=(), cloud="do")

        assert out["status"] == "failed"
        assert "attached volume cannot be deleted" in (out["error"] or "")

    def test_no_ladder_when_the_droplet_itself_would_not_die(
        self, monkeypatch: pytest.MonkeyPatch, slept: list[float]
    ) -> None:
        """A volume attached to a droplet we failed to destroy will never detach: waiting a
        minute for it only delays the real alarm."""
        client = _Client(settle=None)

        def _refuse(droplet_id: int) -> None:
            raise RuntimeError("422 unprocessable")

        client.droplets.destroy = _refuse
        monkeypatch.setattr(sp, "_get_do_client", lambda: client)

        out = sp.robust_teardown(_DownFails(), CLUSTER, backoffs=(), cloud="do")

        assert out["status"] == "failed" and "422" in (out["error"] or "")
        assert slept == []


class TestReconcileVolumePass:
    @pytest.fixture
    def lab(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lab:
        home = tmp_path / "runs"
        lab = Lab(backend=LocalBackend(home=home, repo=tmp_path), repo=tmp_path, home=home)
        job = CLUSTER.removeprefix("lab-tempotr-5171-")
        lab.store.create(make_manifest(job, "python x.py").model_copy(update={"status": JobState.failed}))
        monkeypatch.setattr(core_mod, "local_project", lambda: "tempotron-capacity")
        monkeypatch.setattr(sp, "list_vast_instances", lambda *a, **k: [])
        monkeypatch.setattr(sp, "list_gcp_instances", lambda *a, **k: [])
        monkeypatch.setattr(sp, "list_gcp_disks", lambda *a, **k: [])
        monkeypatch.setattr(sp, "gcp_project", lambda: "p")
        return lab

    def _sky_destroys_the_droplet(self, monkeypatch: pytest.MonkeyPatch, client: _Client) -> None:
        fake = types.ModuleType("sky")
        fake.get = lambda x: x  # type: ignore[attr-defined]
        fake.status = lambda refresh=None: [{"name": CLUSTER}]  # type: ignore[attr-defined]
        fake.down = lambda c: client._destroy(1)  # type: ignore[attr-defined]
        fake.StatusRefreshMode = types.SimpleNamespace(AUTO="AUTO")  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sky", fake)
        monkeypatch.setattr(sp, "_get_do_client", lambda: client)  # list_do_volumes uses it too

    def test_the_volume_of_a_cluster_just_destroyed_is_not_a_false_exit_5(
        self, lab: Lab, monkeypatch: pytest.MonkeyPatch, slept: list[float]
    ) -> None:
        client = _Client()
        self._sky_destroys_the_droplet(monkeypatch, client)

        report = lab.reconcile(apply=True)

        assert report["sky_destroyed"] == [CLUSTER]
        assert report["do_volumes_destroyed"] == ["v1"]
        assert report["destroy_outcomes"] == []

    def test_a_volume_that_stays_attached_is_still_reported(
        self, lab: Lab, monkeypatch: pytest.MonkeyPatch, slept: list[float]
    ) -> None:
        """Attached after the whole ladder means its droplet is alive: a real alarm."""
        client = _Client(settle=None)
        self._sky_destroys_the_droplet(monkeypatch, client)

        report = lab.reconcile(apply=True)

        assert report["do_volumes_destroyed"] == []
        assert [o["pass"] for o in report["destroy_outcomes"]] == ["do_volume_orphans"]
