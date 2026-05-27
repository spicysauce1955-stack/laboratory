"""Small internal helpers (no external deps)."""

from __future__ import annotations

from datetime import datetime, timezone

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


def parse_duration(value: str | None) -> float | None:
    """Parse a wall-clock limit (FR-I1). ``'2h'``/``'30m'``/``'45s'``/``'1d'`` or plain seconds.

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


def infer_artifact_type(name: str) -> str:
    """Map a filename to an ArtifactType (FR-E3); defaults to ``"other"``."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _ARTIFACT_EXT.get(ext, "other")
