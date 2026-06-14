"""R2QueueStore satisfies the same contract as LocalQueueStore, over a dict-backed fake S3."""

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import make_manifest
from lab.models import CodeRef, JobSpec
from lab.scheduler.models import ControlConfig, Guardrails, Registration, RegState
from lab.scheduler.r2queue import R2QueueStore
from lab.storage import R2Store


class FakeS3:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:
        self.blobs[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self.blobs:
            raise self._missing()
        return {"Body": io.BytesIO(self.blobs[Key])}

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.blobs.pop(Key, None)

    def head_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self.blobs:
            raise self._missing()
        return {}

    def list_objects_v2(self, Bucket: str, Prefix: str, **kw) -> dict:
        keys = sorted(k for k in self.blobs if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        self.blobs[Key] = Path(Filename).read_bytes()

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        Path(Filename).write_bytes(self.blobs[Key])

    @staticmethod
    def _missing() -> Exception:
        import botocore.exceptions

        return botocore.exceptions.ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )


def make_q() -> tuple[R2QueueStore, FakeS3]:
    fake = FakeS3()
    store = R2Store("https://example.test", "bucket", client=fake)
    return R2QueueStore(store, prefix="queue"), fake


def _reg(reg_id: str) -> Registration:
    return Registration(
        reg_id=reg_id,
        created_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
        spec=JobSpec(command="python x.py"),
        guardrails=Guardrails(
            expires_at=datetime(2026, 6, 10, tzinfo=timezone.utc) + timedelta(days=1)
        ),
        bundle_key=f"queue/bundles/{reg_id}.tar.gz",
        code=CodeRef(git_commit="0" * 40, git_dirty=False),
    )


def test_entry_roundtrip_and_keys():
    q, fake = make_q()
    q.put_entry(_reg("reg-a"))
    assert "queue/entries/reg-a.json" in fake.blobs
    assert q.get_entry("reg-a").reg_id == "reg-a"
    assert [r.reg_id for r in q.list_entries()] == ["reg-a"]


def test_control_heartbeat_markers():
    q, _ = make_q()
    assert q.read_control() == ControlConfig()
    q.write_control(ControlConfig(paused=True))
    assert q.read_control().paused
    assert q.read_heartbeat() is None
    q.write_heartbeat({"tick_count": 1})
    assert q.read_heartbeat() == {"tick_count": 1}
    q.request_cancel("reg-a")
    assert q.cancel_requested("reg-a") and not q.held("reg-a")
    q.hold("reg-a")
    q.release("reg-a")
    assert not q.held("reg-a")


def test_bundle_and_manifest_mirror(tmp_path: Path):
    q, _ = make_q()
    src = tmp_path / "b.tar.gz"
    src.write_bytes(b"bytes")
    key = q.put_bundle("reg-a", src)
    assert key == "queue/bundles/reg-a.tar.gz"
    assert q.fetch_bundle(key, tmp_path / "dl").read_bytes() == b"bytes"
    q.mirror_manifest(make_manifest("j1", "python x.py"))
    got = q.read_mirrored("j1")
    assert got is not None and got.job_id == "j1"
    assert q.read_mirrored("nope") is None
    assert [m.job_id for m in q.list_mirrored()] == ["j1"]


def test_state_update_overwrites():
    q, _ = make_q()
    q.put_entry(_reg("reg-a"))
    q.put_entry(q.get_entry("reg-a").model_copy(update={"state": RegState.launched}))
    assert q.get_entry("reg-a").state is RegState.launched
