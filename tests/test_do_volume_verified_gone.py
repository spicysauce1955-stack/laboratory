"""A DO teardown may only say ``succeeded`` once each volume it knows of is verified gone by id.

Job E (2026-11-19) hit its ``--timeout 3m`` on a DigitalOcean box, was recorded ``timed_out``, and
its teardown recorded ``succeeded`` — yet its 50 GB volume ``lab-smoke-…-88cf91-3dd12990-ea4c-head``
was still on the account, detached, about two hours later. A false ``succeeded`` is the most
dangerous outcome teardown has: nothing alarms, ``lab wait`` exits 0, and the volume bills until a
human happens to run ``lab reconcile``.

What the code shows (SkyPilot 0.13, ``sky/provision/do``): ``sky.down`` never deletes the volume
itself. It calls ``droplets.destroy_with_associated_resources_dangerous`` — DO destroys the
droplet and deletes its volume **in the background** — and returns once the droplet has left the
*droplet* listing. Nobody checks the background job (DO reports its per-resource failures on a
status endpoint SkyPilot never reads). Our only check was then a single ``volumes.list`` taken the
instant ``sky.down`` returned, and an absent name was read as "nothing is billing". DO's volume
listing is eventually consistent (seen the same day: a deleted volume still listed ~20 s later),
and a volume mid-way through DO's own deletion is precisely the one a listing is least likely to
describe truthfully. If DO's background delete then fails, the volume is left detached — and the
teardown has already recorded ``succeeded``.

The fix: absence from a listing is never evidence. Every volume we know of — remembered while the
job ran, or seen by a listing taken *before* ``sky.down`` — is verified by a direct
``volumes.get`` until DO answers 404, deleting it (through the detach ladder) on the way. A
listing still runs, but only to *discover* volumes we never saw. A known volume that cannot be
verified gone is ``failed`` (still there, refused) or ``unknown`` (cannot tell) — never
``succeeded``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fake_do import FakeDo, FakeSky, http_error
from helpers import make_manifest

from lab.backends import skypilot as m
from lab.models import JobState, ResourceRequest
from lab.store import JobStore

CLUSTER = "lab-smoke-20261119-111134-88cf91"
VOLUME = f"{CLUSTER}-3dd12990-ea4c-head"
DROPLET_ID = 594518797
FOREVER = 1e9


def _job_e(**kw: Any) -> FakeDo:
    """Job E's account: one droplet, its volume attached, and DO's own background delete of that
    volume failing after the dangerous destroy, which also hides it from listings for a while."""
    kw.setdefault("detach_s", 20.0)
    kw.setdefault("hide_s", 30.0)
    kw.setdefault("associated_delete", False)
    return FakeDo(
        droplets=[{"id": DROPLET_ID, "name": VOLUME}],
        volumes=[{"id": "vol-e", "name": VOLUME, "droplet_ids": [DROPLET_ID]}],
        **kw,
    )


def _install(monkeypatch: pytest.MonkeyPatch, do: FakeDo) -> FakeDo:
    monkeypatch.setattr(m, "_get_do_client", lambda: do)
    monkeypatch.setattr(m.time, "sleep", do.sleep)
    return do


def _never_listed(do: FakeDo, vid: str = "vol-e") -> None:
    """The volume exists but no listing will ever show it — the worst case of eventual
    consistency, and the one only a remembered id can survive."""
    do.hidden_until[vid] = FOREVER


class TestJobE:
    def test_a_volume_the_listing_misses_is_not_a_false_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The incident: sky.down returns, the listing right after it does not show the volume,
        DO's background delete fails. Recording `succeeded` there is the false all-clear."""
        do = _install(monkeypatch, _job_e())

        out = m.robust_teardown(FakeSky(do), CLUSTER, cloud="do", backoffs=(0,))

        if out["status"] == "succeeded":
            assert "vol-e" not in do.vols, "teardown said succeeded while the volume still bills"
        assert "vol-e" not in do.vols, "the volume was ours, known, and deletable: remove it"
        assert out["status"] == "succeeded"

    def test_a_remembered_volume_is_verified_though_no_listing_ever_shows_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _install(monkeypatch, _job_e(detach_s=0.0))
        _never_listed(do)

        out = m.robust_teardown(
            FakeSky(do), CLUSTER, cloud="do", backoffs=(0,), known_volume_ids=["vol-e"]
        )

        assert out["status"] == "succeeded", out["error"]
        assert "vol-e" not in do.vols
        assert "volumes.get:vol-e" in do.calls, "gone must be read from DO by id"


class TestUnverifiedIsNeverSucceeded:
    def test_still_there_and_refused_is_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        do = _install(monkeypatch, _job_e(detach_s=0.0))
        _never_listed(do)
        do.refuse["vol-e"] = http_error(403, "You do not have access", error_id="forbidden")

        out = m.robust_teardown(
            FakeSky(do), CLUSTER, cloud="do", backoffs=(0,), known_volume_ids=["vol-e"]
        )

        assert out["status"] == "failed"
        assert "vol-e" in (out["error"] or "") and "access" in (out["error"] or "")

    def test_cannot_read_it_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DO will not answer about a volume we know existed: that is not an all-clear."""
        do = _install(monkeypatch, _job_e(detach_s=0.0))
        _never_listed(do)
        do.get_error = RuntimeError("read timeout")
        do.refuse["vol-e"] = RuntimeError("connection reset by peer")

        out = m.robust_teardown(
            FakeSky(do), CLUSTER, cloud="do", backoffs=(0,), known_volume_ids=["vol-e"]
        )

        assert out["status"] == "unknown", out
        assert "vol-e" in (out["error"] or "")

    def test_a_delete_accepted_but_never_confirmed_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DO said 204 but the volume still reads as present for the whole ladder. We cannot
        claim it is gone, and we have no refusal to alarm on either."""
        do = _install(monkeypatch, _job_e(detach_s=0.0))
        _never_listed(do)
        do.volumes.delete = lambda volume_id, **kw: None  # accepted, and nothing happens

        out = m.robust_teardown(
            FakeSky(do), CLUSTER, cloud="do", backoffs=(0,), known_volume_ids=["vol-e"]
        )

        assert out["status"] == "unknown", out
        assert "vol-e" in do.vols

    def test_no_listing_and_nothing_remembered_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing to verify by id, and discovery never answered: no evidence either way."""
        do = _install(monkeypatch, _job_e())
        do.list_error = RuntimeError("503 Service Unavailable")

        out = m.robust_teardown(FakeSky(do), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "unknown"


class TestTheCostIsBounded:
    def test_the_healthy_path_never_sleeps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DO already deleted it (or the first delete lands): a few reads, no ladder."""
        do = _install(monkeypatch, _job_e(detach_s=0.0, hide_s=0.0, associated_delete=True))

        out = m.robust_teardown(FakeSky(do), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "succeeded"
        assert do.slept == []
        assert len([c for c in do.calls if c.startswith("volumes.")]) <= 6, do.calls

    def test_a_stuck_volume_costs_at_most_the_ladder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        do = _install(monkeypatch, _job_e(detach_s=None))

        out = m.robust_teardown(FakeSky(do), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "failed"
        assert sum(do.slept) == sum(m._VOLUME_DETACH_BACKOFFS)

    def test_no_do_call_is_made_for_other_clouds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom() -> Any:
            raise AssertionError("DO client built for a vast teardown")

        monkeypatch.setattr(m, "_get_do_client", _boom)

        class _OkSky:
            def get(self, x: Any) -> Any:
                return x

            def down(self, cluster: str) -> None:
                return None

        assert m.robust_teardown(_OkSky(), CLUSTER, cloud="vast", backoffs=(0,))["status"] == (
            "succeeded"
        )


class TestTheFallbackVerifiesToo:
    def test_a_remembered_volume_is_removed_by_the_do_direct_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _install(monkeypatch, _job_e(detach_s=0.0))
        _never_listed(do)
        sky = FakeSky(do, down_error=RuntimeError("sky.down: API server 503"))

        out = m.robust_teardown(
            sky, CLUSTER, cloud="do", backoffs=(), known_volume_ids=["vol-e"]
        )

        assert out["status"] == "succeeded", out["error"]
        assert not do.live_droplets and "vol-e" not in do.vols


class TestReconcileVerifiesToo:
    def test_a_delete_accepted_but_still_present_is_not_reported_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _install(monkeypatch, _job_e(detach_s=0.0))
        do.droplets.destroy(droplet_id=DROPLET_ID)
        do.volumes.delete = lambda volume_id, **kw: None  # accepted, and nothing happens

        gone, failures = m.delete_do_volumes(["vol-e"], client=do)

        assert gone == [] and "vol-e" in failures

    def test_a_volume_already_gone_counts_as_gone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reconcile listed it, DO deleted it since (the listing lags): 404 by id is gone."""
        do = _install(monkeypatch, _job_e(detach_s=0.0))
        del do.vols["vol-e"]

        gone, failures = m.delete_do_volumes(["vol-e"], client=do)

        assert gone == ["vol-e"] and failures == {}


class TestRememberedWhileRunning:
    def _store(self, tmp_path: Path, job_id: str) -> JobStore:
        store = JobStore(tmp_path / "runs")
        store.create(
            make_manifest(job_id, "python x.py", resources=ResourceRequest(cloud="do")).model_copy(
                update={"status": JobState.running, "started_at": datetime.now(UTC)}
            )
        )
        return store

    def test_the_supervisor_records_the_clusters_volume_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lab import sky_runner

        do = _job_e()
        do.vols["theirs"] = {"id": "theirs", "name": "lab-other-20261119-000000-aaaaaa-x-head"}
        monkeypatch.setattr(m, "_get_do_client", lambda: do)
        store = self._store(tmp_path, "20261119-111134-88cf91")

        found = sky_runner.remember_do_volumes(store, "20261119-111134-88cf91", CLUSTER)

        assert found == ["vol-e"]
        assert store.read_runtime("20261119-111134-88cf91")["do_volume_ids"] == ["vol-e"]

    def test_remembering_never_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from lab import sky_runner

        do = _job_e()
        do.list_error = RuntimeError("DO down")
        monkeypatch.setattr(m, "_get_do_client", lambda: do)
        store = self._store(tmp_path, "j1")

        assert sky_runner.remember_do_volumes(store, "j1", CLUSTER) == []
        assert "do_volume_ids" not in store.read_runtime("j1")

    def test_run_job_remembers_the_volume_once_the_box_is_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wiring, end to end through the supervisor: by the time the final teardown runs,
        `_runtime.json` names the volume the job's droplet was using."""
        import sys
        import types

        import lab.placement as P
        from lab import sky_runner as runner_mod
        from lab.models import BackendInfo

        do = _job_e()
        monkeypatch.setattr(m, "_get_do_client", lambda: do)
        store = JobStore(tmp_path / "runs")
        store.create(
            make_manifest("e2e", "python x.py", resources=ResourceRequest(cloud="do")).model_copy(
                update={"backend": BackendInfo(provisioner="skypilot")}
            )
        )
        store.write_runtime("e2e", cluster=CLUSTER)

        class _Sky(types.ModuleType):
            def launch(self, task: Any, cluster_name: str, **kw: Any) -> str:
                return "req-1"

            def api_cancel(self, request_id: Any) -> None:
                pass

            def stream_and_get(self, request_id: Any) -> tuple[int, Any]:
                return 1, types.SimpleNamespace(launched_resources=None)

            def tail_logs(self, *a: Any, **k: Any) -> None:
                return None

            def get(self, x: Any) -> Any:
                return x

            def queue(self, cluster: str, skip_finished: bool = False) -> Any:
                return [{"job_id": 1, "status": types.SimpleNamespace(name="SUCCEEDED")}]

            def status(self, *a: Any, **k: Any) -> Any:
                return []

        remembered_at_teardown: list[Any] = []

        def _teardown(sky_mod: Any, cluster: str, st: Any, jid: str, cloud: str = "vast", **k: Any):
            remembered_at_teardown.append(st.read_runtime(jid).get("do_volume_ids"))
            st.update_manifest(jid, teardown_status="succeeded")
            return True

        monkeypatch.setitem(sys.modules, "sky", _Sky("sky"))
        monkeypatch.setattr(runner_mod, "build_task", lambda *a, **k: "task")
        monkeypatch.setattr(
            runner_mod, "provision_with_watchdog", lambda s, r, *, timeout_s: s.stream_and_get(r)
        )
        monkeypatch.setattr(runner_mod, "tear_down_and_record", _teardown)
        monkeypatch.setattr(runner_mod, "resolve_cost", lambda *a, **k: runner_mod.CostInfo())
        monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
        monkeypatch.setattr(runner_mod, "r2_enabled", lambda: False)
        monkeypatch.setattr(runner_mod, "confirm_success", lambda final, out: final)
        monkeypatch.setattr(P, "affordable_under_cap", lambda r, a: True)

        assert runner_mod.run_job(store.job_dir("e2e")) == 0
        assert remembered_at_teardown == [["vol-e"]]

    def test_teardown_verifies_what_the_supervisor_remembered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The handoff: an id recorded mid-run reaches the teardown through `_runtime.json`,
        including `lab cancel`'s and the dead-supervisor recovery's, which run elsewhere."""
        do = _install(monkeypatch, _job_e(detach_s=0.0))
        _never_listed(do)
        store = self._store(tmp_path, "j2")
        store.write_runtime("j2", cluster=CLUSTER, do_volume_ids=["vol-e"])

        ok = m.tear_down_and_record(FakeSky(do), CLUSTER, store, "j2", "do", backoffs=(0,))

        assert ok is True
        assert "vol-e" not in do.vols
        assert store.read_manifest("j2").teardown_status == "succeeded"
