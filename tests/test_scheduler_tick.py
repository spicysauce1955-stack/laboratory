"""Tick brain — table-driven with fake clock; LocalBackend does real (tiny) launches."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import PYTHON, make_manifest, wait_terminal
from lab._util import now as utc_now
from lab.backends.local import LocalBackend
from lab.models import BackendInfo, CostInfo, JobSpec, JobState, ResourceRequest
from lab.scheduler.bundle import create_bundle
from lab.scheduler.models import ControlConfig, Guardrails, Registration, RegState, Triggers
from lab.scheduler.queue import LocalQueueStore
from lab.scheduler.tick import Scheduler

T0 = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self, t: datetime = T0) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def make_sched(tmp_path: Path, clock: FakeClock | None = None, **kw) -> tuple[Scheduler, LocalQueueStore]:
    q = LocalQueueStore(tmp_path / "queue")
    sched = Scheduler(q, home=tmp_path / "runs", now_fn=clock or FakeClock(), **kw)
    return sched, q


def put_reg(
    q: LocalQueueStore,
    tmp_path: Path,
    reg_id: str,
    *,
    command: str | None = None,
    triggers: Triggers | None = None,
    expires: datetime | None = None,
    state: RegState = RegState.pending,
    job_id: str | None = None,
    max_cost: float | None = None,
) -> Registration:
    """A registration whose bundle is a real (tiny) git repo snapshot."""
    import subprocess

    repo = tmp_path / f"src-{reg_id}"
    if not repo.exists():
        repo.mkdir()
        for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
        (repo / "uv.lock").write_text("lock")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-qm", "x"], check=True, capture_output=True
        )
    tar, code = create_bundle(repo, tmp_path / "tars")
    q.put_bundle(reg_id, tar)
    reg = Registration(
        reg_id=reg_id,
        created_at=T0,
        spec=JobSpec(command=command or f"{PYTHON} -c 'print(42)'"),
        triggers=triggers or Triggers(),
        guardrails=Guardrails(expires_at=expires or T0 + timedelta(days=1), max_cost_usd=max_cost),
        bundle_key=f"bundles/{reg_id}.tar.gz",
        code=code,
        state=state,
        job_id=job_id,
    )
    q.put_entry(reg)
    return reg


def test_heartbeat_written_even_when_paused(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    q.write_control(ControlConfig(paused=True))
    put_reg(q, tmp_path, "reg-a")
    rep = sched.tick()
    assert rep.launched == []
    hb = q.read_heartbeat()
    assert hb is not None and hb["tick_count"] == 1
    assert q.get_entry("reg-a").state is RegState.pending  # untouched while paused


def test_tick_count_increments(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    sched.tick()
    sched.tick()
    hb = q.read_heartbeat()
    assert hb is not None and hb["tick_count"] == 2


def test_expiry(tmp_path: Path):
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    put_reg(q, tmp_path, "reg-a", expires=T0 + timedelta(hours=1))
    clock.t = T0 + timedelta(hours=2)
    rep = sched.tick()
    assert rep.expired == ["reg-a"]
    e = q.get_entry("reg-a")
    assert e.state is RegState.expired and e.last_skip_reason is not None


def test_held_entries_still_expire_but_never_launch(tmp_path: Path):
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    put_reg(q, tmp_path, "reg-a", expires=T0 + timedelta(hours=1))
    q.hold("reg-a")
    rep = sched.tick()
    assert rep.launched == [] and rep.skipped["reg-a"] == "held"
    clock.t = T0 + timedelta(hours=2)
    sched.tick()
    assert q.get_entry("reg-a").state is RegState.expired


def test_no_triggers_launches_immediately(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a")
    rep = sched.tick()
    assert rep.launched == ["reg-a"]
    e = q.get_entry("reg-a")
    assert e.state is RegState.launched and e.job_id is not None
    # The job actually ran (LocalBackend, real subprocess).
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    assert wait_terminal(backend, e.job_id).value == "succeeded"


def test_not_before_gates(tmp_path: Path):
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    put_reg(q, tmp_path, "reg-a", triggers=Triggers(not_before=T0 + timedelta(hours=5)))
    assert sched.tick().launched == []
    assert "not_before" in q.get_entry("reg-a").last_skip_reason
    clock.t = T0 + timedelta(hours=6)
    assert sched.tick().launched == ["reg-a"]


def test_window_gates(tmp_path: Path):
    from datetime import time as dtime

    from lab.scheduler.models import DailyWindow

    clock = FakeClock()  # T0 = 12:00 UTC
    sched, q = make_sched(tmp_path, clock)
    w = DailyWindow(start=dtime(23, 0), end=dtime(7, 0), tz="UTC")
    put_reg(q, tmp_path, "reg-a", triggers=Triggers(window=w))
    assert sched.tick().launched == []
    clock.t = T0.replace(hour=23, minute=30)
    assert sched.tick().launched == ["reg-a"]


def test_dependency_waits_then_launches(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a")
    put_reg(q, tmp_path, "reg-b", triggers=Triggers(after=["reg-a"]))
    rep = sched.tick()
    assert rep.launched == ["reg-a"]  # b waits
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    wait_terminal(backend, q.get_entry("reg-a").job_id)
    sched.tick()  # syncs a -> succeeded (Task 7) ... then:
    rep3 = sched.tick()
    assert "reg-b" in rep3.launched or q.get_entry("reg-b").state is RegState.launched


def test_dependency_failure_cancels_dependent(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", state=RegState.failed)
    put_reg(q, tmp_path, "reg-b", triggers=Triggers(after=["reg-a"]))
    rep = sched.tick()
    assert rep.cancelled == ["reg-b"]
    e = q.get_entry("reg-b")
    assert e.state is RegState.cancelled and "reg-a" in (e.last_skip_reason or "")


def test_dead_dep_behind_waiting_dep_cancels_immediately(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a")  # pending (waiting)
    put_reg(q, tmp_path, "reg-b", state=RegState.failed)
    put_reg(q, tmp_path, "reg-c", triggers=Triggers(after=["reg-a", "reg-b"]))
    rep = sched.tick()
    assert "reg-c" in rep.cancelled
    assert q.get_entry("reg-c").state is RegState.cancelled


def test_cancel_marker_blocks_launch(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a")
    q.request_cancel("reg-a")
    rep = sched.tick()
    assert rep.launched == [] and q.get_entry("reg-a").state is RegState.cancelled


def test_orphaned_launching_reverts_to_pending(tmp_path: Path):
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    reg = put_reg(q, tmp_path, "reg-a", state=RegState.launching)
    q.put_entry(reg.model_copy(update={"state_changed_at": T0 - timedelta(minutes=20)}))
    rep = sched.tick()
    # repair runs and reverts to pending; launch happens on a LATER tick, not this one
    assert q.get_entry("reg-a").state is RegState.pending
    assert rep.launched == []


def test_orphaned_launching_with_existing_job_repairs_to_launched(tmp_path: Path):
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    put_reg(q, tmp_path, "reg-a")
    sched.tick()
    job_id = q.get_entry("reg-a").job_id
    assert job_id is not None
    # Simulate the crash window: entry says launching/stale, but the job exists.
    broken = q.get_entry("reg-a").model_copy(
        update={
            "state": RegState.launching,
            "job_id": None,
            "state_changed_at": T0 - timedelta(minutes=20),
        }
    )
    q.put_entry(broken)
    sched.tick()
    repaired = q.get_entry("reg-a")
    assert repaired.state is RegState.launched and repaired.job_id == job_id


def test_sync_mirrors_terminal_state_and_manifest(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a")
    sched.tick()
    e = q.get_entry("reg-a")
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    wait_terminal(backend, e.job_id)
    rep = sched.tick()
    assert rep.synced.get("reg-a") == "succeeded"
    assert q.get_entry("reg-a").state is RegState.succeeded
    mirrored = q.read_mirrored(e.job_id)
    assert mirrored is not None and mirrored.status.value == "succeeded"


def test_sync_maps_failed_job_to_failed_reg(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command=f"{PYTHON} -c 'raise SystemExit(3)'")
    sched.tick()
    e = q.get_entry("reg-a")
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    wait_terminal(backend, e.job_id)
    sched.tick()
    assert q.get_entry("reg-a").state is RegState.failed


def test_cancel_marker_on_launched_cancels_the_job(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command=f"{PYTHON} -c 'import time; time.sleep(60)'")
    sched.tick()
    e = q.get_entry("reg-a")
    q.request_cancel("reg-a")
    sched.tick()
    assert q.get_entry("reg-a").state is RegState.cancelled
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    assert backend.status(e.job_id).value == "cancelled"


class FakePrices:
    def __init__(self, hourly: float | None) -> None:
        self.hourly = hourly
        self.calls = 0

    def best_hourly(self, accelerators, extra_query=None):
        self.calls += 1
        return self.hourly


def _gpu_reg(q, tmp_path, reg_id, max_hourly: float, **kw):
    r = put_reg(q, tmp_path, reg_id, triggers=Triggers(max_hourly_usd=max_hourly), **kw)
    q.put_entry(
        r.model_copy(
            update={"spec": r.spec.model_copy(update={
                "resources": ResourceRequest(accelerators="RTX_4090:1", timeout="1h")})}
        )
    )
    return r


def test_price_above_threshold_skips(tmp_path: Path):
    feed = FakePrices(0.40)
    sched, q = make_sched(tmp_path, price_feed=feed)
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25)
    rep = sched.tick()
    assert rep.launched == [] and "price" in rep.skipped["reg-a"]


def test_price_below_threshold_launches(tmp_path: Path):
    sched, q = make_sched(tmp_path, price_feed=FakePrices(0.20))
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25)
    assert sched.tick().launched == ["reg-a"]


def test_no_offers_skips(tmp_path: Path):
    sched, q = make_sched(tmp_path, price_feed=FakePrices(None))
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25)
    assert "no matching offer" in sched.tick().skipped["reg-a"]


def test_price_feed_deduped_per_tick(tmp_path: Path):
    feed = FakePrices(0.20)
    sched, q = make_sched(tmp_path, price_feed=feed)
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25)
    _gpu_reg(q, tmp_path, "reg-b", max_hourly=0.25)
    sched.tick()
    assert feed.calls == 1  # same accelerator spec -> one search_offers call (spec §4.5)


def test_price_trigger_without_feed_skips(tmp_path: Path):
    sched, q = make_sched(tmp_path)  # no price_feed
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25)
    rep = sched.tick()
    assert rep.launched == [] and "no price feed" in rep.skipped["reg-a"]


def test_price_feed_error_skips_and_reports(tmp_path: Path):
    class BoomFeed:
        def best_hourly(self, accelerators, extra_query=None):
            raise RuntimeError("api down")

    sched, q = make_sched(tmp_path, price_feed=BoomFeed())
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25)
    rep = sched.tick()
    assert rep.launched == []
    assert any("api down" in e for e in rep.errors)


def test_per_job_cost_cap(tmp_path: Path):
    sched, q = make_sched(tmp_path, price_feed=FakePrices(0.50))
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=1.0)
    q.put_entry(q.get_entry("reg-a").model_copy(
        update={"guardrails": Guardrails(expires_at=T0 + timedelta(days=1), max_cost_usd=0.25)}))
    rep = sched.tick()  # 0.50/h x 1h = 0.50 > 0.25
    assert "max_cost" in rep.skipped["reg-a"]


def test_daily_budget_skips_not_cancels(tmp_path: Path):
    sched, q = make_sched(tmp_path, price_feed=FakePrices(2.0))
    q.write_control(ControlConfig(budget_usd_per_day=3.0))
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=3.0)
    _gpu_reg(q, tmp_path, "reg-b", max_hourly=3.0)
    rep = sched.tick()  # each estimated 2.0 -> only one fits the $3/day budget
    assert len(rep.launched) == 1
    other = ({"reg-a", "reg-b"} - set(rep.launched)).pop()
    assert "budget" in rep.skipped[other]
    assert q.get_entry(other).state is RegState.pending  # retried next tick


def test_max_concurrent(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    q.write_control(ControlConfig(max_concurrent=1))
    slow = f"{PYTHON} -c 'import time; time.sleep(60)'"
    put_reg(q, tmp_path, "reg-a", command=slow)
    put_reg(q, tmp_path, "reg-b", command=slow)
    rep = sched.tick()
    assert len(rep.launched) == 1
    other = ({"reg-a", "reg-b"} - set(rep.launched)).pop()
    assert "concurren" in rep.skipped[other]


def test_post_launch_price_verify_reverts_to_pending(tmp_path: Path, monkeypatch):
    from lab.models import CostInfo

    sched, q = make_sched(tmp_path, price_feed=FakePrices(0.20))
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25,
             command=f"{PYTHON} -c 'import time; time.sleep(60)'")
    sched.tick()
    e = q.get_entry("reg-a")
    # Supervisor later records the real rental price: way above the threshold (offer raced away).
    sched.store.update_manifest(e.job_id, cost=CostInfo(hourly_usd=0.90))
    sched.tick()
    again = q.get_entry("reg-a")
    assert again.state is RegState.pending and again.job_id is None
    assert "price" in (again.last_skip_reason or "")


# ------------------------------------------------------------------ Task 15: watchdog


def _watchdog_sched(tmp_path: Path, **kw) -> tuple[Scheduler, LocalQueueStore]:
    q = LocalQueueStore(tmp_path / "queue")
    return Scheduler(q, home=tmp_path / "runs", now_fn=utc_now, **kw), q


def _skypilot_launched_reg(
    tmp_path: Path, q: LocalQueueStore, sched: Scheduler,
    *, started_ago_s: float, timeout: str = "1h",
) -> None:
    """A launched skypilot-backed reg whose supervisor pid is dead (impossible pid)."""
    put_reg(q, tmp_path, "reg-a", command="python x.py")
    m = make_manifest("j-sky", "python x.py", timeout=timeout).model_copy(
        update={
            "status": JobState.running,
            "started_at": utc_now() - timedelta(seconds=started_ago_s),
            "backend": BackendInfo(provisioner="skypilot"),
            "registration_id": "reg-a",
        }
    )
    sched.store.create(m)
    sched.store.write_runtime("j-sky", runner_pid=99999999, cluster="lab-j-sky")
    q.put_entry(
        q.get_entry("reg-a").model_copy(
            update={
                "state": RegState.launched,
                "job_id": "j-sky",
                "launched_at": utc_now() - timedelta(seconds=started_ago_s),
            }
        )
    )


def test_watchdog_respawns_when_cluster_alive_within_timeout(tmp_path: Path):
    sched, q = _watchdog_sched(tmp_path)
    _skypilot_launched_reg(tmp_path, q, sched, started_ago_s=60)
    events: list[str] = []
    sched._cluster_alive = lambda cluster: True  # type: ignore[method-assign]
    sched._respawn_supervisor = lambda job_id: events.append(f"respawn:{job_id}")  # type: ignore[method-assign]
    sched.tick()
    assert events == ["respawn:j-sky"]
    assert q.get_entry("reg-a").state is RegState.launched


def test_watchdog_times_out_overdue_job(tmp_path: Path):
    sched, q = _watchdog_sched(tmp_path)
    _skypilot_launched_reg(tmp_path, q, sched, started_ago_s=2 * 3600, timeout="1h")
    events: list[str] = []
    sched._cluster_alive = lambda cluster: True  # type: ignore[method-assign]
    sched._teardown = lambda cluster, job_id: events.append(f"down:{cluster}") or True  # type: ignore[method-assign]
    sched.tick()
    assert events == ["down:lab-j-sky"]
    assert sched.store.read_manifest("j-sky").status.value == "timed_out"
    assert q.get_entry("reg-a").state is RegState.failed


def test_watchdog_marks_failed_when_cluster_gone(tmp_path: Path):
    sched, q = _watchdog_sched(tmp_path)
    _skypilot_launched_reg(tmp_path, q, sched, started_ago_s=60)
    sched._cluster_alive = lambda cluster: False  # type: ignore[method-assign]
    sched.tick()
    m = sched.store.read_manifest("j-sky")
    assert m.status.value == "failed" and "supervisor died" in (m.end_reason or "")


def test_reconcile_sweep_every_n_ticks(tmp_path: Path):
    sched, q = _watchdog_sched(tmp_path, reconcile_every=2)
    calls: list[bool] = []
    sched._reconcile = lambda apply: calls.append(apply) or {"orphans": []}  # type: ignore[method-assign]
    sched.tick()
    sched.tick()  # tick_count 2 -> sweep
    sched.tick()
    sched.tick()  # tick_count 4 -> sweep
    assert calls == [False, False]
    q.write_control(ControlConfig(auto_reconcile=True))
    sched.tick()
    sched.tick()  # tick_count 6 -> sweep with apply=True
    assert calls[-1] is True


def test_local_watchdog_fails_dead_runner(tmp_path: Path):
    """A dead local runner must not leave the reg `launched` forever (slot starvation)."""
    sched, q = _watchdog_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command="python x.py")
    m = make_manifest("j-loc", "python x.py", timeout="1h").model_copy(
        update={
            "status": JobState.running,
            "started_at": utc_now(),
            "registration_id": "reg-a",
        }
    )  # make_manifest defaults to provisioner="local"
    sched.store.create(m)
    sched.store.write_runtime("j-loc", runner_pid=99999999, command_pgid=99999998)
    q.put_entry(
        q.get_entry("reg-a").model_copy(
            update={"state": RegState.launched, "job_id": "j-loc", "launched_at": utc_now()}
        )
    )
    rep = sched.tick()
    final = sched.store.read_manifest("j-loc")
    assert final.status is JobState.failed
    assert "runner died" in (final.end_reason or "")
    assert q.get_entry("reg-a").state is RegState.failed
    assert rep.synced["reg-a"] == "failed"


def test_sync_remirrors_recently_terminal_manifests(tmp_path: Path):
    """teardown_status lands after the terminal status write — the mirror must catch up."""
    sched, q = _watchdog_sched(tmp_path)
    # This test runs on the real clock (utc_now), so anchor the expiry to real-now —
    # the default T0-relative expiry would be in the past and the reg would expire pre-launch.
    put_reg(q, tmp_path, "reg-a", expires=utc_now() + timedelta(days=1))
    sched.tick()
    e = q.get_entry("reg-a")
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    wait_terminal(backend, e.job_id)
    sched.tick()  # reg goes terminal; first mirror has no teardown_status yet
    mirrored = q.read_mirrored(e.job_id)
    assert mirrored is not None and mirrored.teardown_status is None
    sched.store.update_manifest(e.job_id, teardown_status="succeeded")  # late supervisor write
    sched.tick()
    refreshed = q.read_mirrored(e.job_id)
    assert refreshed is not None and refreshed.teardown_status == "succeeded"


def test_preempted_registered_job_is_resubmitted_under_budget(tmp_path):
    sched, q = _watchdog_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command="python x.py",
            expires=utc_now() + timedelta(days=1), max_cost=100.0)
    m = make_manifest("j-spot", "python x.py", timeout="1h").model_copy(update={
        "status": JobState.preempted, "ended_at": utc_now(), "registration_id": "reg-a",
        "teardown_status": "succeeded",
        "cost": CostInfo(actual_usd=0.4, hourly_usd=0.4, estimated_usd=0.4)})
    sched.store.create(m)
    q.put_entry(q.get_entry("reg-a").model_copy(update={
        "state": RegState.launched, "job_id": "j-spot", "launched_at": utc_now()}))
    relaunched: list[str] = []
    sched._relaunch_preempted = lambda reg: relaunched.append(reg.reg_id)  # type: ignore[method-assign]
    sched.tick()
    e = q.get_entry("reg-a")
    assert relaunched == ["reg-a"]
    assert e.preempt_count == 1
    assert e.cumulative_usd == 0.4


def test_preempted_stops_at_retry_cap(tmp_path):
    sched, q = _watchdog_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command="python x.py",
            expires=utc_now() + timedelta(days=1), max_cost=100.0)
    m = make_manifest("j-spot", "python x.py", timeout="1h").model_copy(update={
        "status": JobState.preempted, "ended_at": utc_now(), "registration_id": "reg-a",
        "teardown_status": "succeeded", "cost": CostInfo(actual_usd=0.4)})
    sched.store.create(m)
    q.put_entry(q.get_entry("reg-a").model_copy(update={
        "state": RegState.launched, "job_id": "j-spot", "launched_at": utc_now(),
        "preempt_count": 2}))  # cap reached (default max_preempt_retries=2)
    sched._relaunch_preempted = lambda reg: (_ for _ in ()).throw(AssertionError("must not relaunch"))  # type: ignore[method-assign]
    rep = sched.tick()
    assert q.get_entry("reg-a").state is RegState.failed
    assert rep.synced["reg-a"] == "preempted (retry cap reached)"


def test_preempted_stops_when_budget_exhausted(tmp_path):
    sched, q = _watchdog_sched(tmp_path)
    # cap 0.5; already spent 0.4; next attempt est would push over -> stop
    put_reg(q, tmp_path, "reg-a", command="python x.py",
            expires=utc_now() + timedelta(days=1), max_cost=0.5)
    m = make_manifest("j-spot", "python x.py", timeout="1h").model_copy(update={
        "status": JobState.preempted, "registration_id": "reg-a", "teardown_status": "succeeded",
        "cost": CostInfo(actual_usd=0.4)})
    sched.store.create(m)
    q.put_entry(q.get_entry("reg-a").model_copy(update={
        "state": RegState.launched, "job_id": "j-spot", "cumulative_usd": 0.0}))
    # force a non-trivial next-attempt estimate so spent(0.4)+est > cap(0.5)
    sched._estimate_cost = lambda reg: 0.4  # type: ignore[method-assign]
    sched._relaunch_preempted = lambda reg: (_ for _ in ()).throw(AssertionError("must not relaunch"))  # type: ignore[method-assign]
    rep = sched.tick()
    assert q.get_entry("reg-a").state is RegState.failed
    assert "budget" in rep.synced["reg-a"]


def test_preempted_with_failed_teardown_is_not_resubmitted(tmp_path):
    sched, q = _watchdog_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command="python x.py", expires=utc_now() + timedelta(days=1))
    m = make_manifest("j-spot", "python x.py").model_copy(update={
        "status": JobState.preempted, "registration_id": "reg-a",
        "teardown_status": "failed"})  # ambiguous billing — never relaunch
    sched.store.create(m)
    q.put_entry(q.get_entry("reg-a").model_copy(update={
        "state": RegState.launched, "job_id": "j-spot"}))
    sched._relaunch_preempted = lambda reg: (_ for _ in ()).throw(AssertionError("must not relaunch"))  # type: ignore[method-assign]
    rep = sched.tick()
    assert q.get_entry("reg-a").state is RegState.failed
    assert "teardown" in rep.synced["reg-a"]
