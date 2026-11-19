"""`atomic_write_text` and the shared memos under concurrent writers.

`atomic_write_text` wrote every caller through one fixed `<file>.tmp`, so two writers truncated
and interleaved into the same temp file and one renamed the other's half-written bytes into place
(a 4-process stress read 4,269 torn files; the loser of the rename race got FileNotFoundError).
And CapacityMemo / DeadHostMemo / the doctor cache each did read-modify-write with no lock, so
concurrent shards of one sweep lost each other's entries — exactly the case the memos exist for.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import stat
from pathlib import Path

from lab._util import atomic_write_text


def test_each_write_uses_its_own_temp_file_in_the_same_directory(tmp_path, monkeypatch):
    seen: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst):  # type: ignore[no-untyped-def]
        seen.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    target = tmp_path / "memo.json"
    atomic_write_text(target, "a")
    atomic_write_text(target, "b")

    assert len(seen) == 2
    (src1, _), (src2, _) = seen
    assert src1 != src2, "two writes shared one temp path"
    assert Path(src1).parent == tmp_path  # same filesystem, so the rename stays atomic
    assert target.read_text() == "b"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["memo.json"]  # no temp left behind


def test_a_failed_write_leaves_no_temp_file_and_the_old_content(tmp_path, monkeypatch):
    target = tmp_path / "memo.json"
    atomic_write_text(target, "old")

    def boom(src, dst):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    try:
        atomic_write_text(target, "new")
    except OSError:
        pass
    assert target.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["memo.json"]


def test_the_written_file_keeps_ordinary_permissions(tmp_path):
    """mkstemp creates 0600; a manifest or memo must not silently become owner-only."""
    target = tmp_path / "m.json"
    atomic_write_text(target, "{}")
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(target.stat().st_mode) == 0o666 & ~umask


def _hammer_writes(path: str, n: int, size: int) -> int:
    payload = json.dumps({"entries": {f"k{i}": i for i in range(size)}})
    errors = 0
    for _ in range(n):
        try:
            atomic_write_text(Path(path), payload)
        except FileNotFoundError:
            errors += 1
    return errors


def test_concurrent_writers_never_expose_a_torn_file(tmp_path):
    target = tmp_path / "capacity_memo.json"
    atomic_write_text(target, "{}")
    ctx = mp.get_context("fork")
    with ctx.Pool(4) as pool:
        jobs = [pool.apply_async(_hammer_writes, (str(target), 400, s)) for s in (5, 400, 20, 900)]
        torn = 0
        while not all(j.ready() for j in jobs):
            try:
                json.loads(target.read_text())
            except json.JSONDecodeError:
                torn += 1
        writer_errors = sum(j.get() for j in jobs)
    assert torn == 0
    assert writer_errors == 0
    json.loads(target.read_text())


def _record_zones(home: str, start: int, n: int) -> None:
    from lab.placement import CapacityMemo

    memo = CapacityMemo.for_home(Path(home), ttl_s=3600)
    for i in range(start, start + n):
        memo.record("gcp", "n1-standard-4", [f"zone-{i}"])


def _record_strikes(home: str, start: int, n: int) -> None:
    from lab.placement import DeadHostMemo

    memo = DeadHostMemo.for_home(Path(home), ttl_s=3600)
    for i in range(start, start + n):
        memo.record(f"vast|machine:{i}")


def _put_checks(home: str, start: int, n: int) -> None:
    from lab.doctor import CheckCache, _ok

    cache = CheckCache.for_home(Path(home))
    for i in range(start, start + n):
        cache.put(f"gcp|p|check{i}|-", _ok("apis", "on"))


def _run_concurrently(target, home: Path, procs: int = 6, each: int = 25) -> int:  # type: ignore[no-untyped-def]
    ctx = mp.get_context("fork")
    workers = [
        ctx.Process(target=target, args=(str(home), p * each, each)) for p in range(procs)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(60)
    return procs * each


def test_capacity_memo_loses_no_concurrent_record(tmp_path):
    from lab.placement import CapacityMemo

    total = _run_concurrently(_record_zones, tmp_path)
    zones = CapacityMemo.for_home(tmp_path, ttl_s=3600).exhausted_zones("gcp", "n1-standard-4")
    assert len(zones) == total


def test_dead_host_memo_loses_no_concurrent_strike(tmp_path):
    from lab.placement import DeadHostMemo

    total = _run_concurrently(_record_strikes, tmp_path)
    memo = DeadHostMemo.for_home(tmp_path, ttl_s=3600)
    assert sum(memo.strikes(f"vast|machine:{i}") for i in range(total)) == total


def test_doctor_cache_loses_no_concurrent_put(tmp_path):
    from lab.doctor import CheckCache

    total = _run_concurrently(_put_checks, tmp_path)
    cache = CheckCache.for_home(tmp_path)
    assert all(cache.get(f"gcp|p|check{i}|-") is not None for i in range(total))
