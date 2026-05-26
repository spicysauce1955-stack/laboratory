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
