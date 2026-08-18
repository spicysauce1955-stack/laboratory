"""On-disk ledger: one JSONL file per UTC day under ``~/.lab/events``.

User-global rather than project-local on purpose — the lab installs into other projects
(v0.5.0+), so a project-local store would scatter the history across repos exactly when the
cross-project pattern is the thing worth seeing. Every event carries its project, so
per-project filtering is a read-side concern.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_DIR = "~/.lab/events"


def lock_path(path: Path) -> Path:
    """Return the stable-inode lock file for a day file.

    Uses string concatenation (not with_suffix) to avoid replacing .jsonl,
    ensuring lock files remain outside the ????-??-??.jsonl glob.
    """
    return Path(str(path) + ".lock")


def enabled() -> bool:
    return (os.environ.get("LAB_EVENTS") or "").strip() != "0"


def debug(message: str) -> None:
    """Surface a swallowed logging error, but only when asked (``LAB_EVENTS_DEBUG=1``)."""
    if (os.environ.get("LAB_EVENTS_DEBUG") or "").strip() == "1":
        print(f"[lab.events] {message}", file=sys.stderr)


def events_dir() -> Path:
    override = (os.environ.get("LAB_EVENTS_DIR") or "").strip()
    return Path(override).expanduser() if override else Path(DEFAULT_DIR).expanduser()


def day_file(when: datetime) -> Path:
    return events_dir() / f"{when.strftime('%Y-%m-%d')}.jsonl"


def day_files() -> list[Path]:
    try:
        return sorted(events_dir().glob("????-??-??.jsonl"))
    except OSError as e:
        debug(f"listing failed: {e}")
        return []


def append(record: dict[str, Any], *, when: datetime) -> None:
    """Append one record as a single locked ``O_APPEND`` write.

    A sharded sweep launches many ``lab`` processes against this one file; the lock is what
    rules out the torn or interleaved lines that would make the store untrustworthy in exactly
    the situation where it matters most. Best-effort throughout: a ledger failure must never
    fail a command.
    """
    if not enabled():
        return
    try:
        path = day_file(when)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
        lock = lock_path(path)
        with lock.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                with path.open("a", encoding="utf-8") as day_f:
                    day_f.write(line)
                    day_f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:  # noqa: BLE001 — logging must never fail a command
        debug(f"append failed: {e}")


def iter_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield every parseable record. A ledger unreadable because one line is bad would fail at
    its only job, so malformed lines are skipped, never raised on."""
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        yield record
        except OSError as e:
            debug(f"read failed for {path}: {e}")


STAMP = ".pruned"


def _int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _rewrite(path: Path, records: list[dict[str, Any]]) -> None:
    """Replace a day file. Caller must hold the lock via lock_path(path).

    The caller takes the lock, reads the file, computes kept records, writes this temp file
    under the lock, then replaces atomically. This ensures concurrent appends cannot be lost.
    """
    tmp = path.with_suffix(".jsonl.tmp")
    text = "".join(json.dumps(r, default=str, separators=(",", ":")) + "\n" for r in records)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def compact(*, now: datetime, success_ttl_days: int) -> None:
    """Drop successful calls older than the TTL. Failures — and dangling opens, which are
    themselves a finding — stay until the age cap takes them."""
    cutoff = now - timedelta(days=success_ttl_days)
    for path in day_files():
        try:
            if datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc) >= cutoff:
                continue
            lock = lock_path(path)
            with lock.open("a", encoding="utf-8") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    records = list(iter_records([path]))
                    succeeded = {r["id"] for r in records
                                 if r.get("phase") == "close" and r.get("outcome") == "ok"}
                    kept = [r for r in records if r.get("id") not in succeeded]
                    if len(kept) != len(records):
                        _rewrite(path, kept)
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        except Exception as e:  # noqa: BLE001
            debug(f"compaction failed for {path}: {e}")


def enforce_caps(*, now: datetime, max_age_days: int, max_mb: float) -> None:
    """Delete whole day files past the age cap, then oldest-first until under the byte cap."""
    cutoff = now - timedelta(days=max_age_days)
    remaining: list[Path] = []
    for path in day_files():
        try:
            stamped = datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if stamped < cutoff:
            path.unlink(missing_ok=True)
        else:
            remaining.append(path)
    budget = max_mb * 1024 * 1024
    total = sum(p.stat().st_size for p in remaining if p.exists())
    for path in remaining:  # oldest first
        if total <= budget:
            break
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total -= size
        path.unlink(missing_ok=True)


def maybe_prune(*, now: datetime) -> None:
    """Run retention at most once per UTC day per machine. Lazy, stamp-gated, best-effort."""
    if not enabled():
        return
    try:
        stamp = events_dir() / STAMP
        today = now.strftime("%Y-%m-%d")
        if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == today:
            return
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(today, encoding="utf-8")
        compact(now=now, success_ttl_days=_int_env("LAB_EVENTS_SUCCESS_TTL_DAYS", 14))
        enforce_caps(
            now=now,
            max_age_days=_int_env("LAB_EVENTS_MAX_AGE_DAYS", 90),
            max_mb=_float_env("LAB_EVENTS_MAX_MB", 50),
        )
    except Exception as e:  # noqa: BLE001
        debug(f"pruning failed: {e}")
