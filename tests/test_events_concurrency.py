"""A sharded sweep launches many `lab` processes against one ledger file. Torn or interleaved
lines would make the store untrustworthy exactly when it matters most.

Two scenarios are covered here, because they turned out to have different teeth:

* Pure concurrent ``append()`` (no ``compact()`` in the mix) is already safe on this platform
  without any application-level lock at all — Linux serializes a single ``write(2)`` syscall to
  a regular file at the VFS layer regardless of size, and ``append()`` performs exactly one such
  syscall per record (confirmed empirically up to 100 KB records via ``strace``, well past this
  test's 3 KB). Removing the lock does not make ``test_concurrent_writers_produce_only_whole_lines``
  fail on this filesystem. It is kept anyway as a regression pin on the real sharded-sweep usage
  pattern, and as a guard should this ever run somewhere without that OS guarantee (e.g. NFS).
* The lock's actual load-bearing job — per the Task 3 redesign note — is serializing ``append()``
  against ``compact()``. ``compact()`` replaces the day file's inode via ``os.replace``; an
  unlocked ``append()`` that opened the old inode just before the replace, and writes to it just
  after, has its record silently vanish into the now-unlinked file.
  ``test_concurrent_append_survives_racing_compaction`` below reproduces exactly that and *does*
  fail — reliably, with real data loss, not corruption — when the lock is neutralised.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WRITER = """
import os, sys
from datetime import datetime, timezone
from lab.events import store
when = datetime(2026, 8, 18, tzinfo=timezone.utc)
tag = sys.argv[1]
for i in range(200):
    store.append({"id": f"{tag}-{i}", "phase": "close", "pad": "x" * 3000}, when=when)
"""


def test_concurrent_writers_produce_only_whole_lines(tmp_path: Path) -> None:
    env = {**os.environ, "LAB_EVENTS_DIR": str(tmp_path / "events")}
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, f"w{n}"], env=env) for n in range(8)]
    for p in procs:
        assert p.wait() == 0
    path = tmp_path / "events" / "2026-08-18.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 8 * 200
    for line in lines:
        json.loads(line)  # every line whole and parseable


COMPACTOR = """
import sys
from datetime import datetime, timezone
from lab.events import store
now = datetime(2026, 8, 19, tzinfo=timezone.utc)
for _ in range(int(sys.argv[1])):
    store.compact(now=now, success_ttl_days=0)
"""


def test_concurrent_append_survives_racing_compaction(tmp_path: Path) -> None:
    """The per-day lock's real job: append() and compact() both act on the day file, and
    compact() swaps its inode with os.replace(). A writer racing that swap must not silently
    lose its record into the orphaned old inode.

    The day file is pre-seeded with one record compact() will actually drop (phase=close,
    outcome=ok), so compact() performs a genuine rewrite-and-replace rather than a no-op —
    without at least one real replace, this test would never exercise the race at all.
    """
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    day = events_dir / "2026-08-18.jsonl"
    day.write_text(json.dumps({"id": "seed", "phase": "close", "outcome": "ok"}) + "\n")

    env = {**os.environ, "LAB_EVENTS_DIR": str(events_dir)}
    writers = [
        subprocess.Popen([sys.executable, "-c", WRITER, f"w{n}"], env=env) for n in range(8)
    ]
    compactors = [
        subprocess.Popen([sys.executable, "-c", COMPACTOR, "100"], env=env) for _ in range(2)
    ]
    for p in writers + compactors:
        assert p.wait() == 0

    records = [json.loads(line) for line in day.read_text().splitlines()]  # every line whole
    writer_ids = [r["id"] for r in records if r.get("id", "").startswith("w")]
    assert len(writer_ids) == len(set(writer_ids)), "duplicate ids indicate a torn/re-read record"
    expected = 8 * 200
    assert len(writer_ids) == expected, (
        f"lost {expected - len(writer_ids)} of {expected} records to the compaction race"
    )
