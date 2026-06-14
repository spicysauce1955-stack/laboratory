"""QueueStore contract via the local-dir implementation (R2 implements the same protocol)."""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from helpers import make_manifest
from lab.models import CodeRef, JobSpec
from lab.scheduler.models import ControlConfig, Guardrails, Registration, RegState
from lab.scheduler.queue import LocalQueueStore


def _reg(reg_id: str) -> Registration:
    return Registration(
        reg_id=reg_id,
        created_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
        spec=JobSpec(command="python x.py"),
        guardrails=Guardrails(expires_at=datetime(2026, 6, 11, tzinfo=timezone.utc)),
        bundle_key=f"bundles/{reg_id}.tar.gz",
        code=CodeRef(git_commit="0" * 40, git_dirty=False),
    )


def test_entry_crud_and_listing(tmp_path: Path):
    q = LocalQueueStore(tmp_path)
    q.put_entry(_reg("reg-b"))
    q.put_entry(_reg("reg-a"))
    assert [r.reg_id for r in q.list_entries()] == ["reg-a", "reg-b"]  # sorted
    r = q.get_entry("reg-a")
    r = r.model_copy(update={"state": RegState.launched, "job_id": "j1"})
    q.put_entry(r)
    assert q.get_entry("reg-a").job_id == "j1"


def test_control_default_and_roundtrip(tmp_path: Path):
    q = LocalQueueStore(tmp_path)
    assert q.read_control() == ControlConfig()  # missing file -> defaults
    q.write_control(ControlConfig(paused=True, budget_usd_per_day=5.0))
    assert q.read_control().paused is True


def test_heartbeat(tmp_path: Path):
    q = LocalQueueStore(tmp_path)
    assert q.read_heartbeat() is None
    q.write_heartbeat({"at": "2026-06-10T00:00:00Z", "tick_count": 3})
    hb = q.read_heartbeat()
    assert hb is not None and hb["tick_count"] == 3


def test_markers(tmp_path: Path):
    q = LocalQueueStore(tmp_path)
    assert not q.cancel_requested("reg-a")
    q.request_cancel("reg-a")
    assert q.cancel_requested("reg-a")
    q.hold("reg-a")
    assert q.held("reg-a")
    q.release("reg-a")
    assert not q.held("reg-a")


def test_bundle_roundtrip(tmp_path: Path):
    q = LocalQueueStore(tmp_path / "q")
    src = tmp_path / "code.tar.gz"
    src.write_bytes(b"tarball-bytes")
    key = q.put_bundle("reg-a", src)
    assert key.endswith("reg-a.tar.gz")
    out = q.fetch_bundle(key, tmp_path / "dl")
    assert out.read_bytes() == b"tarball-bytes"


def test_list_and_delete_bundle(tmp_path: Path):
    q = LocalQueueStore(tmp_path / "q")
    src = tmp_path / "code.tar.gz"
    src.write_bytes(b"x")
    k1 = q.put_bundle("reg-a", src)
    k2 = q.put_bundle("sweep-1", src)
    assert q.list_bundle_keys() == sorted([k1, k2])
    q.delete_bundle(k1)
    assert q.list_bundle_keys() == [k2]
    q.delete_bundle("bundles/nope.tar.gz")  # idempotent: deleting a missing bundle is a no-op
    assert q.list_bundle_keys() == [k2]


def test_manifest_mirror(tmp_path: Path):
    q = LocalQueueStore(tmp_path)
    assert q.read_mirrored("j1") is None
    m = make_manifest("j1", "python x.py")
    q.mirror_manifest(m)
    got = q.read_mirrored("j1")
    assert got is not None and got.job_id == "j1"
    assert [x.job_id for x in q.list_mirrored()] == ["j1"]


def test_get_entry_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LocalQueueStore(tmp_path).get_entry("reg-nope")
