"""Mirror a running job's result tables to durable storage *during* the run (§6c salvage).

Why
---
Artifacts reach the object store exactly once, near the end of :func:`lab.sky_runner.run_job`.
Anything that stops the supervisor before that line — a SIGKILL, a dead scheduler host, a box
that vanishes — returns an **empty** R2 prefix for a job that has been appending and fsyncing
``results.csv`` per row for hours. Three zero-row timeouts in the 2026-09 campaign cost $10.4
between them and returned nothing at all.

The rows were never unreachable. :class:`~lab.sky_runner.PartialsFetcher` has rsynced the remote
run dir down every 60 s since 2026-08-21, so they were sitting on the *supervisor's* disk — which
for a scheduler-launched job is the always-on droplet's, a machine nobody reads and the next
blue-green deploy replaces. Mirroring what has already been fetched is what turns "the rows exist
somewhere" into "the rows survive".

The three things it may not trade away
--------------------------------------
**A partial can never be read as a complete artifact.** Everything downstream —
``fetch_artifacts``, ``lab export``, ``sweep-aggregate`` — reads ``<job_id>/<name>``. A mid-run
copy therefore goes under ``<job_id>/_partial/`` and *never* under the job's own keys, and it
carries a :data:`PARTIAL_MARKER` sidecar inside that same prefix so the mark survives a plain
``download_dir`` or a hand copy. The marker records the producing job's status under
``lab.aggregate.STATUS_COLUMN`` — the same field, from the same source, that
:func:`lab.aggregate.merge_seed_rows` already uses to decide partiality for sweep shards. This is
not a second convention; it is the existing one, written down where a salvaged table can carry it.

**No half a row.** rsync copies a file that is being appended to, so the tail can be a fragment.
Only the bytes up to the last newline are mirrored (:func:`whole_lines`), which makes every
mirrored object a whole table. (``merge_seed_rows`` already tolerates a torn tail row from a
non-succeeded shard; this means it never has to.)

**A failed mirror is not a failed job.** Every entry point here is best-effort and returns
instead of raising: the caller counts it, the job carries on, and the end-of-run upload is
untouched. Nothing here can move ``teardown_status`` or an exit code.

Cost
----
An upload is paid for on a machine the user is renting, so it is gated twice: at most one mirror
per :data:`MIRROR_INTERVAL_S`, and none at all when the output dir's ``(name, size, mtime)``
fingerprint has not moved since the last one. At 300 s a 4 h run mirrors ~48 times instead of the
~240 it would at the rsync heartbeat's 60 s, and the extra exposure is at most five minutes of
rows out of four hours — against the alternative on the table, which is all of them.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from lab._util import now

PARTIAL_DIR = "_partial"
"""Key prefix under the job's own prefix. Nested rather than a sibling so a job's partials share
its lifecycle: delete ``<job_id>/`` and they go too."""

PARTIAL_MARKER = "_PARTIAL.json"
"""The sidecar that makes a directory of rows self-describing."""

MIRROR_PATTERNS: tuple[str, ...] = ("*.csv", "*.jsonl")
"""What is worth mirroring: the line-oriented tables an experiment appends to. Deliberately not
checkpoints or figures — those are the expensive half of an output dir and the useless half of a
salvage — and deliberately top-level only, so a ``checkpoints/`` tree can never be walked."""

MIRROR_MAX_BYTES = 32 * 1024 * 1024
"""Skip a table larger than this. A per-seed result table is kilobytes; something this size is
not the thing being salvaged, and re-uploading it every interval is a bill, not a safety net."""

MIRROR_INTERVAL_S = 300.0
"""Minimum seconds between mirrors. See the module docstring for the arithmetic."""


def whole_lines(data: bytes) -> bytes:
    """The leading whole-line part of ``data`` — everything up to and including the last newline.

    Pure and total. ``b""`` when nothing is complete yet, which the caller reads as "no rows to
    mirror" rather than as an empty table.
    """
    cut = data.rfind(b"\n")
    return b"" if cut < 0 else data[: cut + 1]


def eligible(out_dir: Path) -> list[Path]:
    """The files in ``out_dir`` worth mirroring, sorted by name. Never raises.

    A missing, unreadable or empty directory yields ``[]`` — this runs beside a live rsync and
    must treat "not there yet" as ordinary.
    """
    try:
        if not out_dir.is_dir():
            return []
        found: dict[str, Path] = {}
        for pattern in MIRROR_PATTERNS:
            for f in out_dir.glob(pattern):
                if not f.is_file() or f.name.startswith(".") or f.name == PARTIAL_MARKER:
                    continue
                if f.stat().st_size > MIRROR_MAX_BYTES:
                    continue
                found[f.name] = f
        return [found[name] for name in sorted(found)]
    except OSError:
        return []


def fingerprint(files: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
    """A cheap change key for a set of files: ``(name, size, mtime_ns)`` each. Never raises.

    ``rsync -a`` preserves the remote mtime, so both halves move when the box appends a row. A
    file that disappears mid-``stat`` is simply dropped from the key, which reads as a change and
    costs at most one extra mirror.
    """
    out: list[tuple[str, int, int]] = []
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        out.append((f.name, st.st_size, st.st_mtime_ns))
    return tuple(out)


def partial_key(job_id: str, name: str) -> str:
    """The object key a mid-run copy of ``name`` lands on. Never the job's own artifact key."""
    return f"{job_id}/{PARTIAL_DIR}/{name}"


def marker_text(job_id: str, status: str, files: list[dict[str, Any]]) -> str:
    """The :data:`PARTIAL_MARKER` sidecar body (pure).

    ``_shard_status`` is spelled with :data:`lab.aggregate.STATUS_COLUMN`'s name on purpose: a
    consumer that already knows how to read a partial sweep shard needs no new vocabulary for a
    salvaged job, and the value is the producing job's status — the same source
    ``sweep-aggregate`` uses.
    """
    from lab.aggregate import STATUS_COLUMN

    return json.dumps(
        {
            "job_id": job_id,
            "complete": False,
            STATUS_COLUMN: status,
            "mirrored_at": now().isoformat(),
            "files": files,
            "note": (
                "Written while the job was still running: these tables are truncated at the last "
                "complete line and are NOT the job's final artifacts. The job's own keys live "
                f"under {job_id}/."
            ),
        },
        indent=2,
    )


def default_r2() -> Any | None:
    """The configured object store, or ``None`` when R2 is not set up. Never raises."""
    try:
        from lab.storage import R2Store, r2_enabled

        if not r2_enabled():
            return None
        return R2Store.from_env()
    except Exception:  # noqa: BLE001 — an unconfigured/uninstalled store is a no-op, not a fault
        return None


class PartialMirror:
    """Mid-run mirroring of one job's result tables. Best-effort by contract; never raises.

    Owned by :class:`~lab.sky_runner.PartialsFetcher`, which calls :meth:`maybe_mirror` after
    each successful rsync — so this only ever uploads bytes that are already on local disk, and
    adds no traffic to the rented box at all.
    """

    def __init__(
        self,
        job_id: str,
        out_dir: Path,
        *,
        r2_factory: Callable[[], Any | None] | None = None,
        interval_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._job_id = job_id
        self._out_dir = Path(out_dir)
        self._r2_factory = r2_factory
        self._interval_s = interval_s
        self._clock = clock
        self._last_at: float | None = None
        self._fingerprint: tuple[tuple[str, int, int], ...] | None = None
        self.state: dict[str, Any] = {
            "attempts": 0,
            "ok": 0,
            "failed": 0,
            "skipped_unchanged": 0,
            "objects": 0,
            "bytes": 0,
            "last_at": None,
            "last_error": None,
            "uri": None,
        }

    def _interval(self) -> float:
        # Read the module global at call time so a test (or a future setting) can move it.
        return MIRROR_INTERVAL_S if self._interval_s is None else self._interval_s

    def maybe_mirror(self, status: str) -> int:
        """Mirror the run's tables if it is time and anything changed. Returns objects written.

        Zero is the ordinary answer and covers every "nothing to do" case — too soon, unchanged,
        no store configured, no complete rows yet — as well as every failure. The caller must not
        distinguish them: none of them is a reason to touch the job.
        """
        try:
            return self._mirror(status)
        except Exception as e:  # noqa: BLE001 — including this module's own bugs
            self.state["failed"] = int(self.state["failed"]) + 1
            self.state["last_error"] = f"{type(e).__name__}: {e}"[:300]
            self._last_at = self._clock()
            return 0

    def _mirror(self, status: str) -> int:
        t = self._clock()
        if self._last_at is not None and (t - self._last_at) < self._interval():
            return 0
        files = eligible(self._out_dir)
        if not files:
            return 0
        # The gate is claimed here, before any work that can fail: a store that is down must not
        # be retried on every heartbeat, and an unchanged dir must not be re-fingerprinted at the
        # rsync cadence for the rest of the rental.
        self._last_at = t
        fp = fingerprint(files)
        if fp == self._fingerprint:
            self.state["skipped_unchanged"] = int(self.state["skipped_unchanged"]) + 1
            return 0
        factory = self._r2_factory or default_r2
        r2 = factory()
        if r2 is None:
            return 0  # R2 not configured: a no-op, not a failure
        self.state["attempts"] = int(self.state["attempts"]) + 1
        written: list[dict[str, Any]] = []
        count = 0
        total = 0
        for f in files:
            body = whole_lines(f.read_bytes())
            if not body:
                continue  # nothing complete yet — a header still being written
            r2.put_bytes(partial_key(self._job_id, f.name), body)
            written.append({"name": f.name, "bytes": len(body), "lines": body.count(b"\n")})
            count += 1
            total += len(body)
        if not written:
            return 0
        # The marker goes last, so a mirror interrupted part-way leaves rows without a *stale*
        # marker rather than a marker that overstates what arrived.
        r2.put_text(
            partial_key(self._job_id, PARTIAL_MARKER),
            marker_text(self._job_id, status, written),
        )
        count += 1
        self._fingerprint = fp
        self.state["ok"] = int(self.state["ok"]) + 1
        self.state["objects"] = int(self.state["objects"]) + count
        self.state["bytes"] = int(self.state["bytes"]) + total
        self.state["last_at"] = now().isoformat()
        self.state["last_error"] = None
        try:
            self.state["uri"] = r2.uri(f"{self._job_id}/{PARTIAL_DIR}")
        except Exception:  # noqa: BLE001 — a label, not a requirement
            pass
        return count
