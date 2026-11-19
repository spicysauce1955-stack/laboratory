"""Scheduler tick against an R2-backed queue (dict-backed fake S3, no network).

The LocalQueueStore tests in ``test_scheduler_tick.py`` cannot see two whole classes of bug: an
object-store read that *fails* (rather than finding nothing), and the per-tick round-trip count
that is free on a local disk and ~200 sequential HTTPS calls against R2.
"""

from __future__ import annotations

import io
import tarfile
from datetime import timedelta
from pathlib import Path

from test_scheduler_r2queue import FakeS3, _reg

from lab._util import now as utc_now
from lab.scheduler.models import Registration, RegState
from lab.scheduler.r2queue import R2QueueStore
from lab.scheduler.tick import MarkerSnapshot, Scheduler
from lab.storage import R2Store


class CountingS3(FakeS3):
    """FakeS3 that counts calls per operation and can make HEADs on chosen keys fail."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: dict[str, int] = {}
        self.head_raises_for: set[str] = set()

    def _c(self, name: str) -> None:
        self.ops[name] = self.ops.get(name, 0) + 1

    def put_object(self, **kw):  # type: ignore[no-untyped-def]
        self._c("put")
        return super().put_object(**kw)

    def get_object(self, **kw):  # type: ignore[no-untyped-def]
        self._c("get")
        return super().get_object(**kw)

    def head_object(self, Bucket: str, Key: str) -> dict:
        self._c("head")
        if Key in self.head_raises_for:
            raise ConnectionError("Could not connect to the endpoint URL")
        return super().head_object(Bucket=Bucket, Key=Key)

    def list_objects_v2(self, **kw):  # type: ignore[no-untyped-def]
        self._c("list")
        return super().list_objects_v2(**kw)


def r2_sched(tmp_path: Path) -> tuple[Scheduler, R2QueueStore, CountingS3]:
    fake = CountingS3()
    q = R2QueueStore(R2Store("https://example.test", "bucket", client=fake), prefix="queue")
    return Scheduler(q, home=tmp_path / "runs", now_fn=utc_now), q, fake


def future_reg(reg_id: str, **upd: object) -> Registration:
    r = _reg(reg_id)
    return r.model_copy(
        update={
            "guardrails": r.guardrails.model_copy(
                update={"expires_at": utc_now() + timedelta(days=3)}
            ),
            **upd,
        }
    )


def _upload_tiny_bundle(q: R2QueueStore, tmp_path: Path, reg_id: str) -> None:
    tgz = tmp_path / f"{reg_id}.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        data = b"x"
        ti = tarfile.TarInfo("f.txt")
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    q.store.upload_file(tgz, f"queue/bundles/{reg_id}.tar.gz")


class _RecordingLab:
    def __init__(self) -> None:
        self.submitted: list[str] = []

    def submit(self, spec, **kw):  # type: ignore[no-untyped-def]
        self.submitted.append(kw["registration_id"])
        return "j-new"


def test_a_marker_recheck_that_cannot_be_read_does_not_launch(tmp_path: Path):
    """spec §5 cancel race, on R2: the cancel lands after the snapshot and the fresh pre-submit
    HEAD hits a network blip. "Could not ask" used to read as "no marker" and rent the machine;
    it must read as "don't launch yet" -- back to pending, retried next tick."""
    sched, q, fake = r2_sched(tmp_path)
    _upload_tiny_bundle(q, tmp_path, "reg-a")
    q.put_entry(future_reg("reg-a"))
    lab = _RecordingLab()
    sched.make_lab = lambda repo: lab  # type: ignore[method-assign,assignment,return-value]
    orig_sync = sched._sync

    def sync_then_user_cancels(entries, rep, markers: MarkerSnapshot) -> None:  # type: ignore[no-untyped-def]
        orig_sync(entries, rep, markers)
        q.request_cancel("reg-a")  # lands after the snapshot, exactly the race §5 names
        fake.head_raises_for.add("queue/cancelled/reg-a")  # ...and that HEAD hits a blip

    sched._sync = sync_then_user_cancels  # type: ignore[method-assign]
    rep = sched.tick()

    assert lab.submitted == []
    assert rep.launched == []
    e = q.get_entry("reg-a")
    assert e.state is RegState.pending  # retried next tick, never stranded in `launching`
    assert "marker" in (e.last_skip_reason or "")
    assert any("reg-a" in err for err in rep.errors)


def test_an_unreadable_hold_recheck_does_not_launch_either(tmp_path: Path):
    sched, q, fake = r2_sched(tmp_path)
    _upload_tiny_bundle(q, tmp_path, "reg-a")
    q.put_entry(future_reg("reg-a"))
    lab = _RecordingLab()
    sched.make_lab = lambda repo: lab  # type: ignore[method-assign,assignment,return-value]
    fake.head_raises_for.add("queue/held/reg-a")

    rep = sched.tick()

    assert lab.submitted == [] and rep.launched == []
    assert q.get_entry("reg-a").state is RegState.pending


def test_an_unreadable_dependency_is_waited_on_not_cancelled_as_missing(tmp_path: Path):
    """`list_entries` skips an entry it cannot validate (a version-skewed host's enum, a read
    racing a write). Its dependents used to see it as *missing* and were irreversibly cancelled
    ("dependency ended missing") over a one-tick blip."""
    from lab.scheduler.models import Triggers

    sched, q, fake = r2_sched(tmp_path)
    q.put_entry(future_reg("reg-dep", state=RegState.launched, job_id="j-x"))
    q.put_entry(future_reg("reg-child", triggers=Triggers(after=["reg-dep"])))
    good = fake.blobs["queue/entries/reg-dep.json"]
    fake.blobs["queue/entries/reg-dep.json"] = good.replace(b'"launched"', b'"launched_v2"')
    sched._launch = lambda reg, rep: None  # type: ignore[method-assign]

    rep = sched.tick()

    child = q.get_entry("reg-child")
    assert child.state is RegState.pending
    assert "reg-dep" in (child.last_skip_reason or "") and "unreadable" in child.last_skip_reason
    assert any("reg-dep" in e and "unreadable" in e for e in rep.errors)

    fake.blobs["queue/entries/reg-dep.json"] = good  # readable again: ordinary waiting resumes
    sched.tick()
    child = q.get_entry("reg-child")
    assert child.state is RegState.pending
    assert child.last_skip_reason == "waiting on dependency reg-dep (launched)"


def test_a_truly_missing_dependency_still_cancels(tmp_path: Path):
    from lab.scheduler.models import Triggers

    sched, q, _ = r2_sched(tmp_path)
    q.put_entry(future_reg("reg-child", triggers=Triggers(after=["reg-nope"])))
    sched.tick()
    child = q.get_entry("reg-child")
    assert child.state is RegState.cancelled
    assert child.last_skip_reason == "dependency reg-nope ended missing"


def test_a_dead_dependency_cancels_even_beside_an_unreadable_one(tmp_path: Path):
    from lab.scheduler.models import Triggers

    sched, q, fake = r2_sched(tmp_path)
    q.put_entry(future_reg("reg-dep", state=RegState.launched, job_id="j-x"))
    q.put_entry(future_reg("reg-dead", state=RegState.failed))
    q.put_entry(future_reg("reg-child", triggers=Triggers(after=["reg-dep", "reg-dead"])))
    fake.blobs["queue/entries/reg-dep.json"] = b'{"reg_id": "reg-dep"}'
    sched.tick()
    assert q.get_entry("reg-child").state is RegState.cancelled


def test_a_tick_over_unchanged_waiting_entries_writes_nothing_but_its_heartbeat(tmp_path: Path):
    """100 waiting entries used to cost 202 GETs + 101 PUTs *every* tick -- a fresh GET per
    entry and a PUT of an unchanged skip reason -- ~300 sequential R2 round trips per 60s tick
    on the always-on host. Unchanged entries need neither."""
    from lab.scheduler.models import Triggers

    sched, q, fake = r2_sched(tmp_path)
    nb = utc_now() + timedelta(hours=10)
    for i in range(100):
        q.put_entry(future_reg(f"reg-{i:03d}", triggers=Triggers(not_before=nb)))

    fake.ops.clear()
    sched.tick()  # first sight: each entry's skip reason is new, so each is written once
    assert fake.ops["put"] == 100 + 1

    fake.ops.clear()
    rep = sched.tick()  # nothing changed
    assert fake.ops.get("put") == 1  # the heartbeat
    assert fake.ops["get"] == 100 + 2  # the listing + heartbeat + control; no per-entry re-read
    assert len(rep.skipped) == 100  # still reported every tick, just not rewritten
    assert q.get_entry("reg-042").last_skip_reason == f"not_before {nb.isoformat()}"


def test_a_changed_skip_reason_is_still_written(tmp_path: Path):
    from lab.scheduler.models import Triggers

    sched, q, fake = r2_sched(tmp_path)
    q.put_entry(future_reg("reg-dep", state=RegState.launched, job_id="j-x"))
    q.put_entry(future_reg("reg-child", triggers=Triggers(after=["reg-dep"])))
    sched._sync = lambda entries, rep, markers: None  # type: ignore[method-assign]
    sched.tick()
    assert q.get_entry("reg-child").last_skip_reason == "waiting on dependency reg-dep (launched)"
    q.put_entry(q.get_entry("reg-dep").model_copy(update={"state": RegState.pending}))
    sched._launch = lambda reg, rep: None  # type: ignore[method-assign]
    sched.tick()
    assert q.get_entry("reg-child").last_skip_reason == "waiting on dependency reg-dep (pending)"


def test_an_entry_expired_this_tick_is_not_launched(tmp_path: Path):
    """Without the per-entry re-read, the in-memory copy must carry what this tick wrote."""
    sched, q, _ = r2_sched(tmp_path)
    reg = future_reg("reg-a")
    q.put_entry(reg.model_copy(update={"guardrails": reg.guardrails.model_copy(
        update={"expires_at": utc_now() - timedelta(minutes=1)})}))
    launched: list[str] = []
    sched._launch = lambda reg, rep: launched.append(reg.reg_id)  # type: ignore[method-assign]
    rep = sched.tick()
    assert launched == [] and rep.expired == ["reg-a"]
    assert q.get_entry("reg-a").state is RegState.expired


def test_a_readable_absent_marker_still_launches(tmp_path: Path):
    """The other half: a real 404 is "no marker", and the launch goes ahead."""
    sched, q, _ = r2_sched(tmp_path)
    _upload_tiny_bundle(q, tmp_path, "reg-a")
    q.put_entry(future_reg("reg-a"))
    lab = _RecordingLab()
    sched.make_lab = lambda repo: lab  # type: ignore[method-assign,assignment,return-value]

    rep = sched.tick()

    assert lab.submitted == ["reg-a"] and rep.launched == ["reg-a"]
