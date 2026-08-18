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
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DIR = "~/.lab/events"


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
        with path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
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
