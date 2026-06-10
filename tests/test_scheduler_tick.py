"""Tick brain — table-driven with fake clock; LocalBackend does real (tiny) launches."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import PYTHON, wait_terminal
from lab.backends.local import LocalBackend
from lab.models import JobSpec
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
