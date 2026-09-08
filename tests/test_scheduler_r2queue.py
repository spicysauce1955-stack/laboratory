"""R2QueueStore satisfies the same contract as LocalQueueStore, over a dict-backed fake S3."""

import io
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from helpers import make_manifest
from lab import events
from lab.models import CodeRef, JobSpec
from lab.scheduler.models import ControlConfig, Guardrails, Registration, RegState
from lab.scheduler.r2queue import DEFAULT_LIST_CONCURRENCY, R2QueueStore, _list_concurrency
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


def test_read_mirrored_permission_error_propagates():
    """A real, persistent I/O failure from the backing store (bad permissions, a broken
    connection) is not corruption and must surface, not degrade to "not yet available" —
    confirms R2QueueStore's guard was never (and must never be) broadened past
    `(ValidationError, UnicodeDecodeError)` the way LocalQueueStore's briefly was
    (2026-09-06 review)."""
    q, fake = make_q()

    def _raise(Bucket: str, Key: str) -> dict:
        raise PermissionError("permission denied")

    fake.get_object = _raise  # type: ignore[method-assign]
    with pytest.raises(PermissionError):
        q.read_mirrored("whatever")


def test_list_mirrored_permission_error_propagates():
    """list_mirrored sibling of the read_mirrored case above: a real I/O failure on one manifest
    must propagate, not be swallowed as if it were just one skipped/corrupt blob among many."""
    q, fake = make_q()
    q.mirror_manifest(make_manifest("good", "python x.py"))
    fake.blobs["queue/jobs/locked.json"] = b"{}"

    real_get_object = fake.get_object

    def _flaky_get_object(Bucket: str, Key: str) -> dict:
        if Key.endswith("locked.json"):
            raise PermissionError("permission denied")
        return real_get_object(Bucket, Key)

    fake.get_object = _flaky_get_object  # type: ignore[method-assign]
    with pytest.raises(PermissionError):
        q.list_mirrored()


# -- concurrent listing ----------------------------------------------------------------------
# Measured over a 5-day campaign from the lab's own event ledger: 2,419 `lab queue list` calls,
# median 33.6s, p90 50.9s -- 23.3 hours of wall clock spent listing a queue, because 163
# registrations meant 163 *sequential* R2 GETs (and the scheduler host pays it every 60s tick).


class ProbingFakeS3(FakeS3):
    """FakeS3 whose reads block, so overlap is observable.

    Records the peak number of simultaneously in-flight ``get_object`` calls: with sequential
    reads that is 1 no matter how many keys there are, so it is the direct evidence that the
    listing is concurrent (a wall-clock assertion would only be indirect, and flaky on a loaded
    box). ``delays`` optionally maps a key suffix to its sleep, which lets a test make completion
    order disagree with key order.
    """

    def __init__(self, delay: float = 0.05, delays: dict[str, float] | None = None) -> None:
        super().__init__()
        self.delay = delay
        self.delays = delays or {}
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def get_object(self, Bucket: str, Key: str) -> dict:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(self.delays.get(Key.rsplit("/", 1)[-1], self.delay))
            return super().get_object(Bucket, Key)
        finally:
            with self._lock:
                self._in_flight -= 1


def make_probing_q(
    delay: float = 0.05, delays: dict[str, float] | None = None
) -> tuple[R2QueueStore, ProbingFakeS3]:
    fake = ProbingFakeS3(delay=delay, delays=delays)
    store = R2Store("https://example.test", "bucket", client=fake)
    return R2QueueStore(store, prefix="queue"), fake


def _seed_entries(q: R2QueueStore, n: int) -> list[str]:
    ids = [f"reg-{i:02d}" for i in range(n)]
    for reg_id in ids:
        q.put_entry(_reg(reg_id))
    return ids


def _seed_mirrored(q: R2QueueStore, n: int) -> list[str]:
    ids = [f"job-{i:02d}" for i in range(n)]
    for job_id in ids:
        q.mirror_manifest(make_manifest(job_id, "python x.py"))
    return ids


def test_list_entries_reads_concurrently():
    q, fake = make_probing_q()
    ids = _seed_entries(q, 8)

    got = q.list_entries()

    assert [r.reg_id for r in got] == ids
    assert fake.max_in_flight > 1, "list_entries issued its GETs one at a time"


def test_list_mirrored_reads_concurrently():
    q, fake = make_probing_q()
    ids = _seed_mirrored(q, 8)

    got = q.list_mirrored()

    assert [m.job_id for m in got] == ids
    assert fake.max_in_flight > 1, "list_mirrored issued its GETs one at a time"


def test_list_concurrency_env_var_of_one_reads_serially(monkeypatch):
    """The escape hatch has to actually work: a rate-limited account must be able to pin it."""
    monkeypatch.setenv("LAB_R2_LIST_CONCURRENCY", "1")
    q, fake = make_probing_q(delay=0.01)
    _seed_entries(q, 6)

    assert len(q.list_entries()) == 6
    assert fake.max_in_flight == 1


def test_list_concurrency_env_var_caps_the_in_flight_reads(monkeypatch):
    monkeypatch.setenv("LAB_R2_LIST_CONCURRENCY", "3")
    q, fake = make_probing_q()
    _seed_mirrored(q, 12)

    assert len(q.list_mirrored()) == 12
    assert 1 < fake.max_in_flight <= 3


def test_garbage_list_concurrency_env_var_falls_back_to_the_default(monkeypatch):
    """A typo'd or empty override must never turn a listing into a crash (and must not silently
    serialize it either) — same defensive parse as the LAB_EVENTS_* knobs."""
    for value in ("banana", "", "  ", "8.5", "0", "-4"):
        monkeypatch.setenv("LAB_R2_LIST_CONCURRENCY", value)
        assert _list_concurrency() >= 1
    monkeypatch.setenv("LAB_R2_LIST_CONCURRENCY", "banana")
    assert _list_concurrency() == DEFAULT_LIST_CONCURRENCY

    q, fake = make_probing_q()
    _seed_entries(q, 4)
    assert len(q.list_entries()) == 4
    assert fake.max_in_flight > 1


def test_list_concurrency_of_zero_does_not_crash_the_pool(monkeypatch):
    """ThreadPoolExecutor(max_workers=0) raises ValueError, so the clamp is load-bearing."""
    monkeypatch.setenv("LAB_R2_LIST_CONCURRENCY", "0")
    q, _ = make_probing_q(delay=0.0)
    ids = _seed_entries(q, 3)
    assert [r.reg_id for r in q.list_entries()] == ids


def test_list_entries_order_is_sorted_key_order_not_completion_order():
    """A caller diffing successive listings must not see spurious reordering, so the results are
    reassembled by key, never in the order the futures happen to finish."""
    ids = [f"reg-{i:02d}" for i in range(6)]
    # first key slowest, last key fastest: completion order is the exact reverse of key order
    delays = {f"{reg_id}.json": 0.06 - 0.01 * i for i, reg_id in enumerate(ids)}
    q, _ = make_probing_q(delays=delays)
    _seed_entries(q, 6)

    assert [r.reg_id for r in q.list_entries()] == ids


def test_list_mirrored_order_is_sorted_key_order_not_completion_order():
    ids = [f"job-{i:02d}" for i in range(6)]
    delays = {f"{job_id}.json": 0.06 - 0.01 * i for i, job_id in enumerate(ids)}
    q, _ = make_probing_q(delays=delays)
    _seed_mirrored(q, 6)

    assert [m.job_id for m in q.list_mirrored()] == ids


def test_list_entries_empty_queue():
    q, _ = make_q()
    assert q.list_entries() == []
    assert q.list_mirrored() == []


def test_list_entries_skips_partial_entry_instead_of_crashing(capsys):
    """The sibling of the list_mirrored degrade: one partial registration (a version-skewed
    scheduler host writing an entry an older/newer model can't validate, a read racing an
    in-progress write) must not take down the whole of `lab queue list` — every other job in the
    queue becomes invisible for the sake of one bad blob."""
    q, fake = make_q()
    q.put_entry(_reg("reg-good"))
    fake.blobs["queue/entries/reg-partial.json"] = b'{"reg_id": "reg-partial"}'

    with events.record("cli", "queue.list", {}):
        got = q.list_entries()
        notes = list(events.current().notes)  # type: ignore[union-attr]

    assert [r.reg_id for r in got] == ["reg-good"]
    err = capsys.readouterr().err
    assert "queue/entries/reg-partial.json" in err
    assert [n["k"] for n in notes] == ["queue.entry_corrupt"]
    # The key travels in the note's `key` field, exactly as list_mirrored's
    # `queue.manifest_corrupt` note does. `lab.events.sanitize`'s deny-list masks any field *named*
    # "key" (and entropy-masks a long path under any other name), so what the ledger stores today
    # is the mask — a pre-existing sanitizer collision that hits the existing manifest note
    # identically, and the reason the stderr line above is the surface that names the blob.
    assert "key" in notes[0]["d"]
    assert notes[0]["d"]["error"]


def test_list_entries_skips_corrupt_bytes_instead_of_crashing(capsys):
    """One layer deeper than the partial-entry case: genuinely corrupted (non-UTF-8) blob bytes
    raise UnicodeDecodeError out of `R2Store.get_text`'s `.decode()` before pydantic sees the
    text, so the guard has to cover the read too — and under concurrency that read happens in a
    worker thread, where an escaping exception would also discard every sibling result."""
    q, fake = make_q()
    q.put_entry(_reg("reg-good"))
    fake.blobs["queue/entries/reg-corrupt.json"] = b"\xff\xfe\x00bad-bytes"

    got = q.list_entries()

    assert [r.reg_id for r in got] == ["reg-good"]
    assert "queue/entries/reg-corrupt.json" in capsys.readouterr().err


def test_list_entries_degrades_per_entry_without_losing_concurrent_siblings(capsys):
    """The concurrency-specific half of the degrade: with many keys in flight, the one bad blob
    must cost exactly one entry, not the batch it shared a pool with."""
    q, fake = make_probing_q()
    ids = _seed_entries(q, 8)
    fake.blobs["queue/entries/reg-04-bad.json"] = b'{"reg_id": "nope"}'
    fake.blobs["queue/entries/reg-06-bad.json"] = b"\xff\xfe\x00bad-bytes"

    got = q.list_entries()

    assert [r.reg_id for r in got] == ids
    err = capsys.readouterr().err
    assert "queue/entries/reg-04-bad.json" in err
    assert "queue/entries/reg-06-bad.json" in err


def test_list_mirrored_degrades_per_manifest_without_losing_concurrent_siblings(capsys):
    q, fake = make_probing_q()
    ids = _seed_mirrored(q, 8)
    fake.blobs["queue/jobs/job-04-bad.json"] = b'{"job_id": "nope", "mirrored": true}'

    got = q.list_mirrored()

    assert [m.job_id for m in got] == ids
    assert "queue/jobs/job-04-bad.json" in capsys.readouterr().err


# -- bulk marker reads -----------------------------------------------------------------------


class CountingFakeS3(FakeS3):
    """FakeS3 that counts the calls that are network round trips on the real store."""

    def __init__(self) -> None:
        super().__init__()
        self.lists = 0
        self.heads = 0

    def list_objects_v2(self, Bucket: str, Prefix: str, **kw) -> dict:
        self.lists += 1
        return super().list_objects_v2(Bucket, Prefix, **kw)

    def head_object(self, Bucket: str, Key: str) -> dict:
        self.heads += 1
        return super().head_object(Bucket, Key)


def test_marker_id_sets_are_the_exact_inverse_of_hold_and_request_cancel():
    q, _ = make_q()
    assert q.held_ids() == set()
    assert q.cancel_requested_ids() == set()

    q.hold("reg-a")
    q.hold("reg-b")
    q.request_cancel("reg-b")

    assert q.held_ids() == {"reg-a", "reg-b"}
    assert q.cancel_requested_ids() == {"reg-b"}
    for reg_id in ("reg-a", "reg-b", "reg-missing"):
        assert q.held(reg_id) == (reg_id in q.held_ids())
        assert q.cancel_requested(reg_id) == (reg_id in q.cancel_requested_ids())

    q.release("reg-a")
    assert q.held_ids() == {"reg-b"}


def test_marker_id_sets_cost_one_list_call_each_and_no_heads():
    """The 2N sequential HEADs `lab queue list` paid per listing (one `held()` + one
    `cancel_requested()` per entry) collapse to two LISTs."""
    counting = CountingFakeS3()
    q = R2QueueStore(R2Store("https://example.test", "bucket", client=counting), prefix="queue")
    for i in range(40):
        q.hold(f"reg-{i:02d}")  # a put, neither a LIST nor a HEAD
        q.request_cancel(f"reg-{i:02d}")

    assert len(q.held_ids()) == 40
    assert len(q.cancel_requested_ids()) == 40
    assert (counting.lists, counting.heads) == (2, 0)


def test_marker_id_sets_ignore_a_prefix_placeholder_and_nested_keys():
    """A stray object under the marker prefix — a `queue/held/` directory placeholder some S3
    tools create, or a nested key — must be skipped, not read as a registration id (and must not
    crash the listing the way one bad blob used to crash `list_entries`)."""
    q, fake = make_q()
    q.hold("reg-real")
    fake.blobs["queue/held/"] = b""
    fake.blobs["queue/held/nested/reg-fake"] = b""
    fake.blobs["queue/heldish/reg-elsewhere"] = b""  # adjacent prefix, must not leak in

    assert q.held_ids() == {"reg-real"}


def test_list_entries_permission_error_propagates():
    """A real, persistent I/O failure is not corruption: it must surface even though it is now
    raised inside a worker thread, and must not be mistaken for one skippable bad blob."""
    q, fake = make_q()
    q.put_entry(_reg("reg-good"))
    fake.blobs["queue/entries/reg-locked.json"] = b"{}"

    real_get_object = fake.get_object

    def _flaky_get_object(Bucket: str, Key: str) -> dict:
        if Key.endswith("reg-locked.json"):
            raise PermissionError("permission denied")
        return real_get_object(Bucket, Key)

    fake.get_object = _flaky_get_object  # type: ignore[method-assign]
    with pytest.raises(PermissionError):
        q.list_entries()
