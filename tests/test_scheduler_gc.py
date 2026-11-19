"""Bundle GC: orphaned tarballs (only terminal / unreferenced) are reported and deleted; a shared
sweep bundle is kept until every point is terminal."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from lab.models import CodeRef, JobSpec
from lab.scheduler.gc import gc_bundles
from lab.scheduler.models import Guardrails, Registration, RegState
from lab.scheduler.queue import LocalQueueStore


OLD_ORPHAN = "reg-20200101-000000-abcdef"


def _entry(q: LocalQueueStore, reg_id: str, bundle_key: str, state: RegState) -> None:
    q.put_entry(
        Registration(
            reg_id=reg_id,
            created_at=datetime.now(timezone.utc),
            spec=JobSpec(command="python x.py"),
            guardrails=Guardrails(expires_at=datetime.now(timezone.utc) + timedelta(days=1)),
            bundle_key=bundle_key,
            code=CodeRef(git_commit="abc", git_dirty=False),
            state=state,
        )
    )


def _setup(tmp_path: Path) -> tuple[LocalQueueStore, dict[str, str]]:
    q = LocalQueueStore(tmp_path / "q")
    src = tmp_path / "code.tar.gz"
    src.write_bytes(b"x")
    # The unreferenced bundle is named like a real reg id from long ago: an unreferenced bundle is
    # only deletable once it is older than the GC minimum age (see the race tests below).
    names = ("live", "done", "sweep-x", OLD_ORPHAN)
    keys = {name: q.put_bundle(name, src) for name in names}
    keys["orphan"] = keys.pop(OLD_ORPHAN)
    _entry(q, "reg-live", keys["live"], RegState.pending)        # live -> bundle kept
    _entry(q, "reg-done", keys["done"], RegState.succeeded)      # terminal -> orphaned
    _entry(q, "p1", keys["sweep-x"], RegState.succeeded)         # sweep point 1 terminal...
    _entry(q, "p2", keys["sweep-x"], RegState.pending)           # ...point 2 still live -> kept
    # keys["orphan"] referenced by no entry -> orphaned
    return q, keys


def test_gc_dry_run_reports_orphans_without_deleting(tmp_path: Path):
    q, keys = _setup(tmp_path)
    rep = gc_bundles(q, apply=False)
    assert rep["orphaned"] == sorted([keys["done"], keys["orphan"]])
    assert rep["deleted"] == [] and rep["applied"] is False
    assert rep["total_bundles"] == 4 and rep["referenced"] == 2
    assert len(q.list_bundle_keys()) == 4  # nothing deleted on a dry run


def test_gc_apply_deletes_only_orphans(tmp_path: Path):
    q, keys = _setup(tmp_path)
    rep = gc_bundles(q, apply=True)
    assert rep["deleted"] == sorted([keys["done"], keys["orphan"]]) and rep["applied"] is True
    # the live bundle and the still-running sweep's shared bundle survive
    assert q.list_bundle_keys() == sorted([keys["live"], keys["sweep-x"]])


# --- races and unreadable entries: GC must never delete code a registration still needs ---------


def _r2_q():  # type: ignore[no-untyped-def]
    from test_scheduler_r2queue import make_q

    return make_q()


def _r2_reg(reg_id: str, bundle_key: str, state: RegState = RegState.pending) -> Registration:
    return Registration(
        reg_id=reg_id,
        created_at=datetime.now(timezone.utc),
        spec=JobSpec(command="python x.py"),
        guardrails=Guardrails(expires_at=datetime.now(timezone.utc) + timedelta(days=1)),
        bundle_key=bundle_key,
        code=CodeRef(git_commit="abc", git_dirty=False),
        state=state,
    )


def _reg_id(age: timedelta) -> str:
    return f"reg-{(datetime.now(timezone.utc) - age).strftime('%Y%m%d-%H%M%S')}-abc123"


def test_gc_never_deletes_the_bundle_of_an_entry_it_could_not_read() -> None:
    """A skewed client/host wrote an entry this model cannot validate. It is very likely still
    live -- deleting its code makes it unlaunchable -- so with any unreadable entry in the
    listing, GC deletes nothing and says why."""
    q, fake = _r2_q()
    rid = _reg_id(timedelta(days=30))
    q.put_entry(_r2_reg(rid, f"queue/bundles/{rid}.tar.gz"))
    fake.blobs[f"queue/bundles/{rid}.tar.gz"] = b"tgz"
    fake.blobs[f"queue/entries/{rid}.json"] = fake.blobs[f"queue/entries/{rid}.json"].replace(
        b'"pending"', b'"waiting"'
    )
    done = _reg_id(timedelta(days=40))
    q.put_entry(_r2_reg(done, f"queue/bundles/{done}.tar.gz", RegState.succeeded))
    fake.blobs[f"queue/bundles/{done}.tar.gz"] = b"tgz"

    rep = gc_bundles(q, apply=True)

    assert f"queue/bundles/{rid}.tar.gz" in fake.blobs
    assert f"queue/bundles/{done}.tar.gz" in fake.blobs  # nothing at all is deleted
    assert rep["deleted"] == [] and rep["unreadable_entries"] == [rid]
    assert rep["refused"]


def test_gc_keeps_a_fresh_bundle_whose_entry_is_not_committed_yet() -> None:
    """`register` uploads the bundle first and writes the entry last (the commit point). A
    `queue gc --apply` landing in between saw an unreferenced bundle and deleted it, leaving a
    committed registration with no code."""
    q, fake = _r2_q()
    rid = _reg_id(timedelta(minutes=1))
    fake.blobs[f"queue/bundles/{rid}.tar.gz"] = b"tgz"  # register(): bundle uploaded first

    rep = gc_bundles(q, apply=True)  # gc --apply runs now
    q.put_entry(_r2_reg(rid, f"queue/bundles/{rid}.tar.gz"))  # register(): commit point

    assert f"queue/bundles/{rid}.tar.gz" in fake.blobs
    assert rep["deleted"] == [] and rep["kept_too_young"] == [f"queue/bundles/{rid}.tar.gz"]


def test_gc_keeps_an_unreferenced_bundle_whose_age_it_cannot_tell() -> None:
    q, fake = _r2_q()
    fake.blobs["queue/bundles/hand-made.tar.gz"] = b"tgz"
    rep = gc_bundles(q, apply=True)
    assert "queue/bundles/hand-made.tar.gz" in fake.blobs
    assert rep["kept_unknown_age"] == ["queue/bundles/hand-made.tar.gz"]


def test_gc_still_deletes_an_old_unreferenced_bundle_and_a_terminal_ones() -> None:
    q, fake = _r2_q()
    old = _reg_id(timedelta(days=3))
    fake.blobs[f"queue/bundles/{old}.tar.gz"] = b"tgz"
    sweep = f"sweep-{(datetime.now(timezone.utc) - timedelta(days=3)).strftime('%Y%m%d-%H%M%S')}-abc123"
    fake.blobs[f"queue/bundles/{sweep}.tar.gz"] = b"tgz"
    done = _reg_id(timedelta(minutes=1))  # young, but its entry is terminal: no race possible
    q.put_entry(_r2_reg(done, f"queue/bundles/{done}.tar.gz", RegState.failed))
    fake.blobs[f"queue/bundles/{done}.tar.gz"] = b"tgz"

    rep = gc_bundles(q, apply=True)

    assert sorted(rep["deleted"]) == sorted(
        [f"queue/bundles/{old}.tar.gz", f"queue/bundles/{sweep}.tar.gz", f"queue/bundles/{done}.tar.gz"]
    )
