"""Bundle GC: orphaned tarballs (only terminal / unreferenced) are reported and deleted; a shared
sweep bundle is kept until every point is terminal."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from lab.models import CodeRef, JobSpec
from lab.scheduler.gc import gc_bundles
from lab.scheduler.models import Guardrails, Registration, RegState
from lab.scheduler.queue import LocalQueueStore


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
    keys = {name: q.put_bundle(name, src) for name in ("live", "done", "sweep-x", "orphan")}
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
