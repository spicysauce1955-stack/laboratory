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


def test_list_and_delete_bundle(tmp_path: Path):
    q, _ = make_q()
    src = tmp_path / "b.tar.gz"
    src.write_bytes(b"x")
    k1 = q.put_bundle("reg-a", src)
    k2 = q.put_bundle("sweep-1", src)
    assert q.list_bundle_keys() == sorted([k1, k2])
    q.delete_bundle(k1)
    assert q.list_bundle_keys() == [k2]


def test_state_update_overwrites():
    q, _ = make_q()
    q.put_entry(_reg("reg-a"))
    q.put_entry(q.get_entry("reg-a").model_copy(update={"state": RegState.launched}))
    assert q.get_entry("reg-a").state is RegState.launched


def test_read_mirrored_partial_manifest_returns_none_instead_of_crashing():
    """A stub/partial JSON blob in the mirror (e.g. a version-skewed scheduler host, or a read
    racing an in-progress write) must degrade to "not yet available", never raise an unhandled
    pydantic ValidationError (2026-09-04 `lab status` incident: 7 required fields missing)."""
    q, fake = make_q()
    fake.blobs["queue/jobs/partial.json"] = b'{"job_id": "partial", "mirrored": true}'
    assert q.read_mirrored("partial") is None


def test_read_mirrored_corrupt_bytes_returns_none_instead_of_crashing():
    """The same crash class as the partial-manifest fix above, one layer deeper: genuinely
    corrupted (non-UTF-8) blob bytes raise UnicodeDecodeError out of `R2Store.get_text`'s
    `.decode()` before `model_validate_json` ever sees the text, so a guard that only catches
    `pydantic.ValidationError` around the parse call still crashes the caller."""
    q, fake = make_q()
    fake.blobs["queue/jobs/corrupt.json"] = b"\xff\xfe\x00bad-bytes"
    assert q.read_mirrored("corrupt") is None


def test_list_mirrored_skips_corrupt_bytes_instead_of_crashing():
    """list_mirrored sibling of the corrupt-bytes fix above: one blob with non-UTF-8 bytes must
    be skipped, not take down the whole listing."""
    q, fake = make_q()
    q.mirror_manifest(make_manifest("good", "python x.py"))
    fake.blobs["queue/jobs/corrupt.json"] = b"\xff\xfe\x00bad-bytes"
    got = q.list_mirrored()
    assert [m.job_id for m in got] == ["good"]


def test_list_mirrored_skips_partial_manifest_instead_of_crashing():
    """The sibling of the read_mirrored fix above: one corrupt/partial manifest anywhere in the
    mirror must not take down the whole listing — it should be skipped, with the rest of the
    jobs still returned (2026-09-06 review: list_mirrored had the same unguarded
    model_validate_json call read_mirrored was fixed for, but in a loop, so it was worse — one
    bad file failed every caller's listing, not just that one job's lookup)."""
    q, fake = make_q()
    q.mirror_manifest(make_manifest("good", "python x.py"))
    fake.blobs["queue/jobs/partial.json"] = b'{"job_id": "partial", "mirrored": true}'
    got = q.list_mirrored()
    assert [m.job_id for m in got] == ["good"]
