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


def test_read_mirrored_partial_manifest_returns_none_instead_of_crashing(tmp_path: Path):
    """A stub/partial JSON blob in the mirror (e.g. a version-skewed scheduler host, or a read
    racing an in-progress write) must degrade to "not yet available", never raise an unhandled
    pydantic ValidationError (2026-09-04 `lab status` incident: 7 required fields missing)."""
    q = LocalQueueStore(tmp_path)
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "partial.json").write_text('{"job_id": "partial", "mirrored": true}')
    assert q.read_mirrored("partial") is None


def test_read_mirrored_corrupt_bytes_returns_none_instead_of_crashing(tmp_path: Path):
    """The same crash class as the partial-manifest fix above, one layer deeper: genuinely
    corrupted (non-UTF-8) bytes on disk raise UnicodeDecodeError out of `Path.read_text()`
    before `model_validate_json` ever sees them, so a guard that only catches
    `pydantic.ValidationError` still crashes the caller."""
    q = LocalQueueStore(tmp_path)
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "corrupt.json").write_bytes(b"\xff\xfe\x00bad-bytes")
    assert q.read_mirrored("corrupt") is None


def test_list_mirrored_skips_corrupt_bytes_instead_of_crashing(tmp_path: Path):
    """list_mirrored sibling of the corrupt-bytes fix above: one file with non-UTF-8 bytes must
    be skipped, not take down the whole listing."""
    q = LocalQueueStore(tmp_path)
    q.mirror_manifest(make_manifest("good", "python x.py"))
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "corrupt.json").write_bytes(b"\xff\xfe\x00bad-bytes")
    got = q.list_mirrored()
    assert [m.job_id for m in got] == ["good"]


def test_list_mirrored_skips_partial_manifest_instead_of_crashing(tmp_path: Path):
    """The sibling of the read_mirrored fix above: one corrupt/partial manifest anywhere in the
    mirror must not take down the whole listing — it should be skipped, with the rest of the
    jobs still returned (2026-09-06 review: list_mirrored had the same unguarded
    model_validate_json call read_mirrored was fixed for, but in a loop, so it was worse — one
    bad file failed every caller's listing, not just that one job's lookup)."""
    q = LocalQueueStore(tmp_path)
    q.mirror_manifest(make_manifest("good", "python x.py"))
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "partial.json").write_text('{"job_id": "partial", "mirrored": true}')
    got = q.list_mirrored()
    assert [m.job_id for m in got] == ["good"]


def test_read_mirrored_permission_error_propagates(tmp_path: Path, monkeypatch):
    """The `OSError` broadening (since narrowed) was too wide: `PermissionError` (bad
    permissions, disk-full, a stale mount) is a real, persistent I/O failure, not corruption, and
    must surface rather than silently degrade to "not yet available" forever — which would make
    every `lab status`/`fetch`/`metrics`/`logs` call quietly report "try again shortly" with zero
    signal that the real cause is an OS-level problem (2026-09-06 review)."""
    q = LocalQueueStore(tmp_path)
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "locked.json").write_text('{"job_id": "locked"}')

    real_read_text = Path.read_text

    def _flaky_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "locked.json":
            raise PermissionError("permission denied")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", _flaky_read_text)
    with pytest.raises(PermissionError):
        q.read_mirrored("locked")


def test_list_mirrored_permission_error_propagates(tmp_path: Path, monkeypatch):
    """list_mirrored sibling of the read_mirrored fix above: a real I/O failure on one manifest
    must propagate, not be swallowed as if it were just one skipped/corrupt file among many."""
    q = LocalQueueStore(tmp_path)
    q.mirror_manifest(make_manifest("good", "python x.py"))
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "locked.json").write_text('{"job_id": "locked"}')

    real_read_text = Path.read_text

    def _flaky_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "locked.json":
            raise PermissionError("permission denied")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", _flaky_read_text)
    with pytest.raises(PermissionError):
        q.list_mirrored()
