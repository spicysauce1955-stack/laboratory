"""What may be written. Recording argv means recording whatever was typed, and
:func:`lab.redact.redact` only knows patterns that appear in *subprocess output* — it will not
catch a key passed as a flag value. Everything entering the ledger passes through here first
(FR-J1, AC-7)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from lab.redact import redact

MASK = "…REDACTED…"
MAX_STR = 512
MAX_ITEMS = 32

_SECRET_KEY = re.compile(r"key|token|secret|password|credential|auth", re.IGNORECASE)
_SECRET_VALUE = (
    re.compile(r"^ya29\."),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$"),  # bare base64 blobs
)
_HEXISH = re.compile(r"^[0-9a-f-]+$", re.IGNORECASE)  # commits, cell ids, job ids — not secrets


def _entropy(s: str) -> float:
    counts = {c: s.count(c) for c in set(s)}
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in counts.values())


def _looks_secret(value: str) -> bool:
    # Hex-only strings (commit hashes, job IDs) are never secrets, so check first
    # to prevent the base64 pattern from matching 40-char hex strings.
    if _HEXISH.match(value):
        return False
    if any(p.search(value) for p in _SECRET_VALUE):
        return True
    if " " in value or len(value) < 32:
        return False
    return _entropy(value) > 3.5


def _scalar(value: Any) -> Any:
    if isinstance(value, str):
        if _looks_secret(value):
            return MASK
        value = redact(value)
        return value[:MAX_STR] + "…" if len(value) > MAX_STR else value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return f"<{type(value).__name__}>"


def _walk(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _SECRET_KEY.search(key):
        return MASK
    if isinstance(value, Mapping):
        return {str(k): _walk(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_walk(v) for v in list(value)[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"…{len(value) - MAX_ITEMS} more")
        return items
    return _scalar(value)


def sanitize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize a parameter mapping for the ledger. Never raises — a sanitizer that
    failed would take the whole record with it."""
    try:
        return {str(k): _walk(v, key=str(k)) for k, v in params.items()}
    except Exception:  # noqa: BLE001 — logging must never fail a command
        return {"_unsanitizable": True}


def sanitize_argv(argv: Sequence[str]) -> list[str]:
    """Sanitize a raw command line: mask ``--flag=<secret>`` and the token after ``--flag``."""
    out: list[str] = []
    mask_next = False
    for token in argv:
        if mask_next:
            out.append(MASK)
            mask_next = False
            continue
        if token.startswith("-") and "=" in token:
            flag, _, value = token.partition("=")
            out.append(f"{flag}={MASK}" if _SECRET_KEY.search(flag) else f"{flag}={_scalar(value)}")
            continue
        if token.startswith("-") and _SECRET_KEY.search(token):
            mask_next = True
        out.append(_scalar(token))
    return out
