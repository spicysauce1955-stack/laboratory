"""What may be written. Recording argv means recording whatever was typed, and
:func:`lab.redact.redact` only knows patterns that appear in *subprocess output* — it will not
catch a key passed as a flag value. Everything entering the ledger passes through here first
(FR-J1, AC-7)."""

from __future__ import annotations

import math
import re
import shlex
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
    # Hex-only strings ≤40 chars (commit SHAs, cell ids, job IDs) are never secrets, so check
    # first to prevent the base64 pattern from matching 40-char hex strings. Longer hex strings
    # (e.g. 64-char hex API tokens) fall through to entropy check and are masked.
    if _HEXISH.match(value) and len(value) <= 40:
        return False
    if any(p.search(value) for p in _SECRET_VALUE):
        return True
    if " " in value or len(value) < 32:
        return False
    return _entropy(value) > 3.5


def _mask_tokens(tokens: Sequence[str]) -> list[str]:
    """Flag-aware masking, token by token: the value after a secret-looking flag, and the value
    half of an inline ``--flag=value``. Shared by ``sanitize_argv`` (real argv, already split by
    the shell) and ``_mask_command_line`` (a single string parameter that turns out to *be* a
    whole quoted command line, e.g. ``lab submit -c "python train.py --hf-token=..."``)."""
    out: list[str] = []
    mask_next = False
    for token in tokens:
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
        out.append(str(_scalar(token)))
    return out


def _mask_command_line(text: str) -> str:
    """A single string parameter that is actually a whole command line (the common lab
    invocation: ``lab submit -c "python train.py --hf-token=..."`` puts it in one argv token, and
    MCP's ``command`` argument is the same shape). The flag-aware masking above only ever saw
    separate argv tokens, so a secret hiding as a flag *value* inside one string token — where it
    has spaces around it, so ``_looks_secret`` bails, and ``redact()`` doesn't know the flag name
    — sailed through unmasked. Tolerant splitting: try ``shlex.split`` (handles quoting), fall
    back to ``.split()`` on a malformed quote — this must never raise."""
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    if len(tokens) < 2:
        return text  # nothing shaped like a multi-token command; leave it as-is
    return " ".join(_mask_tokens(tokens))


def _scalar(value: Any) -> Any:
    if isinstance(value, str):
        if _looks_secret(value):
            return MASK
        if re.search(r"\s", value):
            value = _mask_command_line(value)
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
    """Sanitize a raw command line: mask ``--flag=<secret>`` and the token after ``--flag``
    across separate argv tokens, and (via ``_scalar`` -> ``_mask_command_line``) the same
    flag/value masking *inside* a single token that is itself a whole quoted command string.
    Never raises — a sanitizer that failed would take the whole record with it."""
    try:
        return _mask_tokens(argv)
    except Exception:  # noqa: BLE001 — logging must never fail a command
        return [MASK]
