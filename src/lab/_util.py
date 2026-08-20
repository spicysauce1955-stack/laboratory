"""Small internal helpers (no external deps)."""

from __future__ import annotations

import os
import shlex
from datetime import datetime, timezone
from pathlib import Path

_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}

_ARTIFACT_EXT = {
    "png": "figure", "pdf": "figure", "svg": "figure", "jpg": "figure", "jpeg": "figure",
    "csv": "table", "tsv": "table", "json": "table", "parquet": "table",
    "ckpt": "checkpoint", "pt": "checkpoint", "pth": "checkpoint", "safetensors": "checkpoint",
    "log": "log", "txt": "log",
}


def now() -> datetime:
    """Timezone-aware current time (UTC)."""
    return datetime.now(timezone.utc)


def duration_seconds(started: datetime | None, ended: datetime | None) -> float | None:
    """Wall-clock seconds between two timestamps, or None if either is missing (FR-I2)."""
    if started is None or ended is None:
        return None
    return (ended - started).total_seconds()


def actual_cost(hourly_usd: float | None, seconds: float | None) -> float | None:
    """Actual USD = hourly rate prorated over the run's wall-clock (FR-I2)."""
    if hourly_usd is None or seconds is None:
        return None
    return round(hourly_usd * seconds / 3600.0, 6)


def wrap_with_extras(command: str, extras: list[str] | None) -> str:
    """Layer extra runtime packages on top of an entrypoint via ``uv run --with``.

    Lets a single job declare deps the lean remote env (numpy/pydantic/hydra) doesn't include
    (e.g. ``scipy``) without modifying the project. If the command already starts with ``uv run``,
    the ``--with`` flags are injected after it (no double prefix).
    """
    if not extras:
        return command
    flags = " ".join(f"--with {shlex.quote(e)}" for e in extras)
    if command.startswith("uv run "):
        return command.replace("uv run ", f"uv run {flags} ", 1)
    return f"uv run {flags} {command}"


def parse_duration(value: str | float | None) -> float | None:
    """Parse a wall-clock limit (FR-I1). ``'2h'``/``'30m'``/``'45s'``/``'1d'`` or plain seconds
    (string or number).

    Returns seconds, or ``None`` for no limit.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s[-1] in _UNITS:
        return float(s[:-1]) * _UNITS[s[-1]]
    return float(s)


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a same-directory temp file + ``os.replace`` so readers never see a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def tail_last_line(path: Path, max_bytes: int = 1024) -> tuple[str | None, datetime | None]:
    """Last non-empty line of ``path`` (reading at most the final ``max_bytes``) + its mtime,
    or ``(None, None)`` if the file is missing/empty. Cheap enough for a status poll."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - max_bytes))
            chunk = f.read().decode(errors="replace")
    except OSError:
        return None, None
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    if not lines:
        return None, None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return lines[-1], mtime


def timeout_reason(seconds: int) -> str:
    """The manifest ``end_reason`` for a wall-clock timeout — one wording shared by the local
    and skypilot runners so they can't drift (FR-I1)."""
    return f"timed out after {seconds}s wall-clock cap"


def infer_artifact_type(name: str) -> str:
    """Map a filename to an ArtifactType (FR-E3); defaults to ``"other"``."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _ARTIFACT_EXT.get(ext, "other")


def process_start_time(pid: int | None) -> int | None:
    """The kernel's start-time for ``pid`` (``/proc/<pid>/stat`` field 22), or ``None``.

    Field 22 is clock ticks since boot, and the kernel guarantees it differs between a process and
    any later process that reuses its PID — which is exactly the identity :func:`pid_alive` needs
    and a bare PID cannot supply. Linux-only by construction; everywhere else this returns ``None``
    and liveness degrades to the PID-only answer.

    Never raises. This feeds leak detection, where a probe that throws is worse than one that
    shrugs: the caller's fallback is "assume alive", which is the safe direction.
    """
    if not pid or pid < 1:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, ValueError):
        return None
    # comm (field 2) is parenthesised and may itself contain spaces and parens, so split on the
    # LAST ')' rather than tokenising the whole line.
    try:
        tail = stat[stat.rindex(")") + 1 :].split()
        return int(tail[19])  # field 22 overall = index 19 after pid and comm
    except (ValueError, IndexError):
        return None


def pid_alive(pid: int | None, *, start_time: int | None = None) -> bool:
    """Is this pid still running — and, when ``start_time`` is given, still the *same* process?

    One implementation for the local runner, the skypilot runner and the scheduler tick, which
    each carried a byte-identical private copy — the same reason ``timeout_reason`` lives here.
    ``PermissionError`` means the process exists but belongs to another user, which is still
    alive for our purposes (supervisor liveness, leak detection).

    ``start_time`` closes the PID-reuse blind spot (F4). A supervisor's PID is freed quickly — the
    short-lived ``lab submit`` parent exits at once, so the child reparents to init and is reaped
    there — and on a busy machine the number gets recycled. Without an identity check every
    liveness probe for that job then answers "alive" permanently, silently disabling the
    dead-supervisor teardown in :meth:`SkyPilotBackend.status`, reconcile's ``unsupervised`` pass
    and the scheduler watchdog, while the machine bills.

    Absent or unreadable identity means **alive**, deliberately. Runtime files written before this
    existed carry no start-time, and reporting a live supervisor dead would let reconcile destroy
    the machine out from under a running job — the 2026-08-20 failure pointing the other way.
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    if start_time is None:
        return True
    current = process_start_time(pid)
    if current is None:
        return True  # cannot tell -> assume alive, never destroy on a guess
    return current == start_time
