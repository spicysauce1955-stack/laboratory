"""Dead-host recording and bounded accelerator rotation in the supervisor (lab.sky_runner).

Hermetic: ``sky`` is a fake module, the Vast SDK is never imported, and no test touches a cloud,
an API server or a paid resource.

The failure being fixed, in the lab's own ledger: 28 ``provisioning exceeded`` closes over five
days (an undercount — scheduler-launched attempts are missing), each ~4-8 minutes of a
provisioning slot and **$0**, because a dead offer never bills and so nothing that watches money
ever notices. One real close, from ``~/.lab/events/2026-09-04.jsonl``:

    "error": {"message": "provisioning exceeded 240s (host never reached UP — likely a dead Vast
              offer; resubmit for a fresh host)"},
    "trace": [{"k": "provision.attempt", "d": {"cloud": "vast", "zone": null, "region": null,
               "instance": "RTX3090:1"}},
              {"k": "provision.timeout", "d": {"after_s": 240.0}}]

Note what the ledger knew about the host it had just spent 240s on: its cloud, and what we asked
for. That is the whole reason ``vast_machine_id`` has to go and ask Vast.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab import placement as P
from lab.backends.skypilot import ProvisionTimeout
from lab.models import JobState, ResourceRequest
from lab.store import JobStore


# --------------------------------------------------------------------------------------------
# vast_machine_id — the one sharp identity in reach
# --------------------------------------------------------------------------------------------
# Shaped like vastai's own `Instance` dataclass (vastai/data/instance.py): `id` is the
# rental/contract, `machine_id` the physical host, `host_id` its operator, `label` what SkyPilot
# tags it with. Only the fields this code reads are filled in.
CLUSTER = "lab-tempotr-5171-20260903-200140-f01cf6"


def _rental(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": 26214421,
        "machine_id": 41234,
        "host_id": 9911,
        "label": CLUSTER,
        "actual_status": "loading",
        "dph_total": 0.3585,
    }
    base.update(over)
    return base


def _fake_vast(monkeypatch, instances: list[dict[str, Any]] | Exception) -> list[int]:
    """Patch the rental listing; returns a call counter list the test can assert on."""
    import lab.backends.skypilot as sky_backend

    calls: list[int] = []

    def _list(client: Any = None) -> list[dict[str, Any]]:
        calls.append(1)
        if isinstance(instances, Exception):
            raise instances
        return instances

    monkeypatch.setattr(sky_backend, "list_vast_instances", _list)
    return calls


class TestVastMachineId:
    def test_it_reads_the_machine_id_of_our_own_rental(self, monkeypatch):
        _fake_vast(monkeypatch, [_rental(label="someone-else", machine_id=1), _rental()])
        assert runner_mod.vast_machine_id(CLUSTER) == "41234"

    def test_the_rental_id_is_not_the_machine_id(self, monkeypatch):
        """`id` is the contract and changes every time the same box is rented; only `machine_id`
        can recur, and recurring is the entire phenomenon being recorded."""
        _fake_vast(monkeypatch, [_rental()])
        assert runner_mod.vast_machine_id(CLUSTER) == "41234"
        assert runner_mod.vast_machine_id(CLUSTER) != "26214421"

    def test_no_matching_rental_is_not_an_answer(self, monkeypatch):
        _fake_vast(monkeypatch, [_rental(label="lab-other-job")])
        assert runner_mod.vast_machine_id(CLUSTER) is None

    def test_a_rental_without_the_field_is_not_an_answer(self, monkeypatch):
        _fake_vast(monkeypatch, [_rental(machine_id=None)])
        assert runner_mod.vast_machine_id(CLUSTER) is None

    def test_no_sdk_or_no_credentials_degrades_to_none(self, monkeypatch):
        _fake_vast(monkeypatch, RuntimeError("No API key found. Pass api_key=, set VAST_API_KEY"))
        assert runner_mod.vast_machine_id(CLUSTER) is None


# --------------------------------------------------------------------------------------------
# The no-CUDA guard, from real observed lines
# --------------------------------------------------------------------------------------------


class TestNoCudaMarker:
    @pytest.mark.parametrize(
        "line",
        [
            # Both verbatim from ~/.superset/projects/tempotron-capacity/runs/*/logs.txt: 20 runs
            # exited on the first, 2 printed the second.
            "(lab-2026, pid=1979) no CUDA device. Exiting non-zero (no CPU-smoke at GPU price).",
            "torch.cuda.is_available() is False",
        ],
    )
    def test_real_observed_lines_are_recognised(self, line: str):
        assert runner_mod.has_no_cuda_marker(line) is True

    def test_an_ordinary_log_is_not(self):
        assert runner_mod.has_no_cuda_marker("epoch 4 loss 0.2\ncuda:0 ready\n") is False


# --------------------------------------------------------------------------------------------
# record_dead_host
# --------------------------------------------------------------------------------------------

REAL_LOG_TAIL = (
    "2026-09-03T20:01:42.246Z  Vast (Romania, RO, EU)   1x-RTX_3090-32-65536   32      64   "
    "     RTX3090:1   0.25          ✔\n"
    "2026-09-03T20:01:42.548Z ⚙︎ Launching on Vast Romania, RO, EU.\n"
)


class TestRecordDeadHost:
    def test_a_failed_provision_records_the_machine_the_accelerator_and_the_placement(
        self, tmp_path, monkeypatch
    ):
        _fake_vast(monkeypatch, [_rental()])
        m = make_manifest("j1", "python x.py", accelerators="RTX3090:1")

        keys = runner_mod.record_dead_host(
            tmp_path, m, "vast", CLUSTER, why="provision_timeout", log_text=REAL_LOG_TAIL
        )

        assert keys == [
            "vast|machine:41234",
            "vast|accel:RTX3090:1",
            "vast|placement:1x-RTX_3090-32-65536@Romania, RO, EU",
        ]
        memo = P.DeadHostMemo.for_home(tmp_path)
        assert all(memo.strikes(k) == 1 for k in keys)

    def test_a_second_failure_strikes_the_same_machine_out(self, tmp_path, monkeypatch):
        _fake_vast(monkeypatch, [_rental()])
        m = make_manifest("j1", "python x.py", accelerators="RTX3090:1")
        memo = P.DeadHostMemo.for_home(tmp_path)
        key = P.machine_key("vast", 41234)

        runner_mod.record_dead_host(tmp_path, m, "vast", CLUSTER, why="provision_timeout")
        assert memo.is_dead(key) is False
        runner_mod.record_dead_host(tmp_path, m, "vast", CLUSTER, why="provision_timeout")
        assert memo.is_dead(key) is True

    def test_an_unidentifiable_host_still_records_what_is_real(self, tmp_path, monkeypatch):
        """No machine id is the common case (no SDK, no key, a rental already gone). The coarse
        keys are still written — as evidence — and nothing is invented to fill the gap."""
        _fake_vast(monkeypatch, RuntimeError("no api key"))
        m = make_manifest("j1", "python x.py", accelerators="RTX3090:1")

        keys = runner_mod.record_dead_host(
            tmp_path, m, "vast", CLUSTER, why="provision_timeout", log_text=REAL_LOG_TAIL
        )
        assert not any(k.startswith("vast|machine:") for k in keys)
        assert "vast|accel:RTX3090:1" in keys

    def test_a_non_vast_cloud_is_never_asked_for_a_machine_id(self, tmp_path, monkeypatch):
        calls = _fake_vast(monkeypatch, [_rental()])
        m = make_manifest("j1", "python x.py", accelerators="T4:1")
        keys = runner_mod.record_dead_host(tmp_path, m, "gcp", "lab-x", why="provision_timeout")
        assert calls == []  # Vast's API has nothing to say about a GCP host
        assert keys == ["gcp|accel:T4:1"]

    def test_recording_never_raises(self, tmp_path, monkeypatch):
        """It runs on the error path of a job that has already failed; nothing here may make that
        worse."""
        monkeypatch.setattr(
            P.DeadHostMemo, "record", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        )
        m = make_manifest("j1", "python x.py", accelerators="RTX3090:1")
        runner_mod.record_dead_host(tmp_path, m, "vast", CLUSTER, why="provision_timeout")


# --------------------------------------------------------------------------------------------
# drawn_dead_host — the early check that turns an 8-minute wait into seconds
# --------------------------------------------------------------------------------------------


class TestDrawnDeadHost:
    def _memo(self, tmp_path, *, dead: bool) -> P.DeadHostMemo:
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=1e9, strikes=2)
        if dead:
            memo.record(P.machine_key("vast", 41234))
            memo.record(P.machine_key("vast", 41234))
        return memo

    def test_a_known_dead_machine_drawn_again_is_named(self, tmp_path, monkeypatch):
        _fake_vast(monkeypatch, [_rental()])
        memo = self._memo(tmp_path, dead=True)
        assert runner_mod.drawn_dead_host(memo, CLUSTER, "vast") == "vast|machine:41234"

    def test_a_host_with_one_strike_is_not_dead_yet(self, tmp_path, monkeypatch):
        _fake_vast(monkeypatch, [_rental()])
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=1e9, strikes=2)
        memo.record(P.machine_key("vast", 41234))
        assert runner_mod.drawn_dead_host(memo, CLUSTER, "vast") is None

    def test_an_empty_memo_costs_not_one_api_call(self, tmp_path, monkeypatch):
        """With nothing recorded the launch path must be exactly what it was before this
        existed — including making no extra calls."""
        calls = _fake_vast(monkeypatch, [_rental()])
        memo = self._memo(tmp_path, dead=False)
        assert runner_mod.drawn_dead_host(memo, CLUSTER, "vast") is None
        assert calls == []

    def test_a_different_machine_is_left_alone(self, tmp_path, monkeypatch):
        _fake_vast(monkeypatch, [_rental(machine_id=99999)])
        memo = self._memo(tmp_path, dead=True)
        assert runner_mod.drawn_dead_host(memo, CLUSTER, "vast") is None

    def test_it_waits_for_the_rental_to_register_then_gives_up(self, tmp_path, monkeypatch):
        """A rental that never shows up inside the window is not evidence of anything."""
        _fake_vast(monkeypatch, [])
        memo = self._memo(tmp_path, dead=True)
        clock = iter([0.0, 10.0, 20.0, 30.0, 100.0])
        sleeps: list[float] = []

        got = runner_mod.drawn_dead_host(
            memo,
            CLUSTER,
            "vast",
            window_s=30.0,
            poll_s=5.0,
            sleep=sleeps.append,
            monotonic=lambda: next(clock),
        )
        assert got is None
        assert sleeps and all(s == 5.0 for s in sleeps)

    def test_a_broken_memo_never_blocks_a_launch(self, tmp_path):
        path = tmp_path / P.DeadHostMemo.FILENAME
        path.write_text("{corrupt")
        assert runner_mod.drawn_dead_host(P.DeadHostMemo(path), CLUSTER, "vast") is None

    def test_a_raising_memo_never_blocks_a_launch(self, tmp_path):
        class _Exploding:
            def has_any(self, *a: Any, **k: Any) -> bool:
                raise RuntimeError("memo on fire")

        assert runner_mod.drawn_dead_host(_Exploding(), CLUSTER, "vast") is None

    def test_only_vast_has_a_machine_id_to_check(self, tmp_path, monkeypatch):
        calls = _fake_vast(monkeypatch, [_rental()])
        memo = self._memo(tmp_path, dead=True)
        assert runner_mod.drawn_dead_host(memo, "lab-x", "gcp") is None
        assert calls == []


# --------------------------------------------------------------------------------------------
# LaunchRotation — the bound
# --------------------------------------------------------------------------------------------


def _res(**kw: Any) -> ResourceRequest:
    return ResourceRequest(cloud="vast", accelerators="RTX4090:1", **kw)


def _manifest(res: ResourceRequest, job_id: str = "r1"):
    return make_manifest(job_id, "python x.py", resources=res, timeout="10m")


class TestLaunchRotationBound:
    def test_no_pool_means_exactly_one_attempt(self):
        rot = runner_mod.LaunchRotation.for_manifest(_manifest(_res()))
        assert rot.max_attempts == 1
        assert rot.can_retry is False
        assert rot.advance(_res()) is None

    def test_a_pool_defaults_to_one_attempt_per_entry(self):
        res = _res(accelerator_pool="RTX3090:1,L40S:1")
        rot = runner_mod.LaunchRotation.for_manifest(_manifest(res))
        assert rot.pool == ["RTX4090:1", "RTX3090:1", "L40S:1"]  # the manifest's own goes first
        assert rot.max_attempts == 3

    def test_an_explicit_bound_wins_over_the_pool_size(self):
        res = _res(accelerator_pool="RTX3090:1,L40S:1", max_launch_attempts=2)
        rot = runner_mod.LaunchRotation.for_manifest(_manifest(res))
        assert rot.max_attempts == 2

    def test_the_bound_is_capped_because_every_attempt_costs_a_provisioning_slot(self):
        res = _res(accelerator_pool="a,b,c", max_launch_attempts=99)
        rot = runner_mod.LaunchRotation.for_manifest(_manifest(res))
        assert rot.max_attempts == runner_mod.MAX_LAUNCH_ATTEMPTS

    def test_a_bound_of_zero_still_launches_once(self):
        res = _res(max_launch_attempts=0)
        assert runner_mod.LaunchRotation.for_manifest(_manifest(res)).max_attempts == 1

    def test_advance_stops_at_the_bound_not_at_the_pool_end(self, monkeypatch):
        monkeypatch.setattr(P, "affordable_under_cap", lambda res, a: True)
        res = _res(accelerator_pool="RTX3090:1,L40S:1", max_launch_attempts=2)
        rot = runner_mod.LaunchRotation.for_manifest(_manifest(res))
        assert rot.advance(res) is not None  # attempt 2
        assert rot.advance(res) is None  # the bound, though L40S is still untried

    def test_rotation_never_repeats_an_accelerator(self, monkeypatch):
        monkeypatch.setattr(P, "affordable_under_cap", lambda res, a: True)
        res = _res(accelerator_pool="RTX4090:1,RTX3090:1")
        rot = runner_mod.LaunchRotation.for_manifest(_manifest(res))
        nxt = rot.advance(res)
        assert nxt is not None and nxt.accelerators == "RTX3090:1"
        assert rot.advance(nxt) is None

    def test_rotation_never_raises_the_price_cap(self, monkeypatch):
        monkeypatch.setattr(P, "affordable_under_cap", lambda res, a: True)
        res = _res(accelerator_pool="RTX3090:1", max_hourly_usd=0.40)
        rot = runner_mod.LaunchRotation.for_manifest(_manifest(res))
        nxt = rot.advance(res)
        assert nxt is not None
        assert nxt.max_hourly_usd == 0.40 and nxt.price_cap_strict is res.price_cap_strict

    def test_rotation_skips_what_the_cap_cannot_buy(self, monkeypatch):
        """An accelerator the catalog prices above the cap in every region can never launch;
        rotating onto it would spend an attempt to buy a slower failure. The cap does not move."""
        monkeypatch.setattr(P, "affordable_under_cap", lambda res, a: a != "L40S:1")
        res = _res(accelerator_pool="L40S:1,RTX3090:1", max_hourly_usd=0.40)
        rot = runner_mod.LaunchRotation.for_manifest(_manifest(res))
        nxt = rot.advance(res)
        assert nxt is not None and nxt.accelerators == "RTX3090:1"
        assert nxt.max_hourly_usd == 0.40


# --------------------------------------------------------------------------------------------
# run_job end to end
# --------------------------------------------------------------------------------------------


class _FakeSky(types.ModuleType):
    """The slice of ``sky`` the launch path uses. Provisioning either times out or comes up."""

    def __init__(self, *, status_name: str = "SUCCEEDED") -> None:
        super().__init__("sky")
        self.status_name = status_name
        self.launches: list[str] = []
        self.cancels: list[Any] = []

    # -- launch path -------------------------------------------------------------------
    def launch(self, task: Any, cluster_name: str, **kw: Any) -> str:
        self.launches.append(cluster_name)
        return f"req-{len(self.launches)}"

    def api_cancel(self, request_id: Any) -> None:
        self.cancels.append(request_id)

    def stream_and_get(self, request_id: Any) -> tuple[int, Any]:
        return 1, types.SimpleNamespace(
            launched_resources=types.SimpleNamespace(
                use_spot=False, instance_type="1x-RTX_4090-32-65536", region="Maryland, US, NA",
                zone=None,
            )
        )

    # -- run path ----------------------------------------------------------------------
    def tail_logs(self, *a: Any, **k: Any) -> None:
        return None

    def get(self, x: Any) -> Any:
        return x

    def queue(self, cluster: str, skip_finished: bool = False) -> Any:
        return [{"job_id": 1, "status": types.SimpleNamespace(name=self.status_name)}]

    def status(self, *a: Any, **k: Any) -> Any:
        return []


def _install(monkeypatch, sky: _FakeSky, *, timeouts: int) -> dict[str, int]:
    """Wire a fake sky + the seams run_job would otherwise take to a real cloud."""
    counters = {"provision": 0, "teardown": 0, "budget": 0}

    def _provision(sky_mod: Any, request_id: Any, *, timeout_s: float) -> tuple[Any, Any]:
        counters["provision"] += 1
        counters["budget"] = int(timeout_s)
        if counters["provision"] <= timeouts:
            raise ProvisionTimeout(f"provisioning did not complete within {timeout_s:.0f}s")
        return sky_mod.stream_and_get(request_id)

    def _teardown(sky_mod: Any, cluster: str, st: Any, jid: str, cloud: str = "vast", **kw: Any):
        counters["teardown"] += 1
        st.update_manifest(jid, teardown_status="succeeded")
        return True

    monkeypatch.setitem(sys.modules, "sky", sky)
    monkeypatch.setattr(runner_mod, "build_task", lambda *a, **k: "task")
    monkeypatch.setattr(runner_mod, "provision_with_watchdog", _provision)
    monkeypatch.setattr(runner_mod, "tear_down_and_record", _teardown)
    monkeypatch.setattr(runner_mod, "resolve_cost", lambda *a, **k: runner_mod.CostInfo())
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "r2_enabled", lambda: False)
    monkeypatch.setattr(runner_mod, "confirm_success", lambda final, out: final)
    monkeypatch.setattr(runner_mod, "vast_machine_id", lambda cluster, client=None: "41234")
    monkeypatch.setattr(P, "affordable_under_cap", lambda res, a: True)
    return counters


def _store(tmp_path: Path, job_id: str, res: ResourceRequest) -> JobStore:
    store = JobStore(tmp_path / "runs")
    store.create(_manifest(res, job_id))
    return store


class TestRunJobRotation:
    def test_without_a_pool_a_dead_offer_fails_exactly_as_before(self, tmp_path, monkeypatch):
        sky = _FakeSky()
        counters = _install(monkeypatch, sky, timeouts=1)
        store = _store(tmp_path, "d1", _res())

        rc = runner_mod.run_job(store.job_dir("d1"))

        assert rc == 1
        m = store.read_manifest("d1")
        assert m.status is JobState.failed
        assert (m.end_reason or "").startswith("provisioning exceeded")
        assert len(sky.launches) == 1  # one attempt, unchanged
        assert counters["teardown"] == 1

    def test_a_dead_offer_is_recorded_even_with_no_rotation(self, tmp_path, monkeypatch):
        sky = _FakeSky()
        _install(monkeypatch, sky, timeouts=1)
        store = _store(tmp_path, "d2", _res())

        runner_mod.run_job(store.job_dir("d2"))

        memo = P.DeadHostMemo.for_home(store.home)
        assert memo.strikes(P.machine_key("vast", "41234")) == 1
        assert memo.strikes(P.accelerator_key("vast", "RTX4090:1")) == 1

    def test_a_pool_rotates_past_a_dead_offer_and_lands(self, tmp_path, monkeypatch):
        sky = _FakeSky()
        counters = _install(monkeypatch, sky, timeouts=1)
        store = _store(tmp_path, "d3", _res(accelerator_pool="RTX3090:1"))

        rc = runner_mod.run_job(store.job_dir("d3"))

        assert rc == 0
        m = store.read_manifest("d3")
        assert m.status is JobState.succeeded
        assert len(sky.launches) == 2
        # The manifest records what actually ran, not what was first asked for.
        assert m.resources.accelerators == "RTX3090:1"
        assert counters["teardown"] == 2  # the abandoned attempt, then the real one

    def test_rotation_stops_at_its_bound(self, tmp_path, monkeypatch):
        sky = _FakeSky()
        _install(monkeypatch, sky, timeouts=99)  # every attempt dies
        store = _store(
            tmp_path, "d4", _res(accelerator_pool="RTX3090:1,L40S:1,A10:1", max_launch_attempts=3)
        )

        rc = runner_mod.run_job(store.job_dir("d4"))

        assert rc == 1
        assert len(sky.launches) == 3  # not 4, whatever the pool holds
        assert (store.read_manifest("d4").end_reason or "").startswith("provisioning exceeded")

    def test_rotation_refuses_to_relaunch_over_an_unconfirmed_teardown(
        self, tmp_path, monkeypatch
    ):
        """A second launch under a cluster name whose machine may still be billing would stack a
        leak on a failure. Fail toward the alarm instead (FR-C2)."""
        sky = _FakeSky()
        _install(monkeypatch, sky, timeouts=99)

        def _leaky(sky_mod: Any, cluster: str, st: Any, jid: str, cloud: str = "vast", **kw: Any):
            st.update_manifest(jid, teardown_status="failed")
            return False

        monkeypatch.setattr(runner_mod, "tear_down_and_record", _leaky)
        store = _store(tmp_path, "d5", _res(accelerator_pool="RTX3090:1,L40S:1"))

        rc = runner_mod.run_job(store.job_dir("d5"))

        assert rc == 1
        assert len(sky.launches) == 1  # no relaunch
        assert store.read_manifest("d5").teardown_status == "failed"

    def test_rotation_never_edits_the_price_cap(self, tmp_path, monkeypatch):
        sky = _FakeSky()
        _install(monkeypatch, sky, timeouts=1)
        store = _store(
            tmp_path, "d6", _res(accelerator_pool="RTX3090:1", max_hourly_usd=0.40)
        )

        runner_mod.run_job(store.job_dir("d6"))

        m = store.read_manifest("d6")
        assert m.resources.accelerators == "RTX3090:1"
        assert m.resources.max_hourly_usd == 0.40

    def test_a_known_dead_redraw_is_abandoned_early_instead_of_waited_out(
        self, tmp_path, monkeypatch
    ):
        """The point of the memo: the second draw of a struck-out machine costs one API call, not
        another whole provisioning budget."""
        sky = _FakeSky()
        counters = _install(monkeypatch, sky, timeouts=0)  # provisioning would have succeeded
        store = _store(tmp_path, "d7", _res(accelerator_pool="RTX3090:1"))
        memo = P.DeadHostMemo.for_home(store.home, strikes=2)
        memo.record(P.machine_key("vast", "41234"))
        memo.record(P.machine_key("vast", "41234"))

        rc = runner_mod.run_job(store.job_dir("d7"))

        assert rc == 0
        assert len(sky.launches) == 2  # first draw abandoned, second one kept
        assert sky.cancels == ["req-1"]  # the abandoned request was cancelled, not left running
        assert counters["provision"] == 1  # never waited on the known-dead host
        # The check's own time comes out of the provisioning budget, never on top of it.
        assert counters["budget"] <= 8 * 60
        assert store.read_manifest("d7").resources.accelerators == "RTX3090:1"

    def test_a_memo_that_would_exclude_everything_still_launches(self, tmp_path, monkeypatch):
        """Every entry in the pool struck out, and the drawn machine struck out too: the launch
        goes ahead anyway. A memo may spend an attempt; it may never refuse one."""
        sky = _FakeSky()
        counters = _install(monkeypatch, sky, timeouts=0)
        monkeypatch.setenv("LAB_DEAD_HOST_STRIKES", "1")  # one failure is enough to be "dead"
        store = _store(tmp_path, "d8", _res(accelerator_pool="RTX3090:1"))
        memo = P.DeadHostMemo.for_home(store.home)
        for key in (
            P.machine_key("vast", "41234"),
            P.accelerator_key("vast", "RTX4090:1"),
            P.accelerator_key("vast", "RTX3090:1"),
        ):
            memo.record(key)

        rc = runner_mod.run_job(store.job_dir("d8"))

        assert rc == 0
        assert store.read_manifest("d8").status is JobState.succeeded
        assert counters["provision"] >= 1  # it did wait on a machine, rather than give up

    def test_a_small_provision_timeout_is_never_inflated(self, tmp_path, monkeypatch):
        """The first cut of the budget deduction floored at a flat 60s — which turned a 0.05s
        provision timeout into a 60s one and hung
        ``test_events_notes.py::test_supervisor_provision_timeout_also_notes``. The floor may
        never exceed the budget it is protecting."""
        sky = _FakeSky()
        counters = _install(monkeypatch, sky, timeouts=0)
        store = _store(tmp_path, "db", _res(provision_timeout="5s"))

        runner_mod.run_job(store.job_dir("db"))

        assert counters["budget"] == 5

    def test_a_host_that_boots_without_the_gpu_it_advertises_is_recorded(
        self, tmp_path, monkeypatch
    ):
        """The other kind of dead host — it provisions fine, so the watchdog never sees it. The
        line below is verbatim from a real run on this machine."""
        sky = _FakeSky(status_name="FAILED")
        _install(monkeypatch, sky, timeouts=0)
        store = _store(tmp_path, "dc", _res())
        store.logs_path("dc").write_text(
            "(lab-dc, pid=2044) no CUDA device. Exiting non-zero (no CPU-smoke at GPU price).\n"
        )

        runner_mod.run_job(store.job_dir("dc"))

        assert store.read_manifest("dc").status is JobState.failed
        memo = P.DeadHostMemo.for_home(store.home)
        assert memo.strikes(P.machine_key("vast", "41234")) == 1

    def test_an_ordinary_failure_is_not_blamed_on_the_host(self, tmp_path, monkeypatch):
        """A job whose code raises is not evidence about the machine it ran on."""
        sky = _FakeSky(status_name="FAILED")
        _install(monkeypatch, sky, timeouts=0)
        store = _store(tmp_path, "df", _res())
        store.logs_path("df").write_text("Traceback ...\nValueError: bad alpha\n")

        runner_mod.run_job(store.job_dir("df"))

        memo = P.DeadHostMemo.for_home(store.home)
        assert memo.strikes(P.machine_key("vast", "41234")) == 0

    def test_an_unwritable_memo_does_not_stop_a_job_from_running(self, tmp_path, monkeypatch):
        sky = _FakeSky()
        _install(monkeypatch, sky, timeouts=0)
        store = _store(tmp_path, "d9", _res())
        monkeypatch.setattr(
            P.DeadHostMemo, "record", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only"))
        )

        assert runner_mod.run_job(store.job_dir("d9")) == 0
        assert store.read_manifest("d9").status is JobState.succeeded
