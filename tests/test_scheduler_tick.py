"""Tick brain — table-driven with fake clock; LocalBackend does real (tiny) launches."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import PYTHON
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
