"""Scrub secrets from captured subprocess output before it reaches disk (FR-J1).

SkyPilot/Vast log the Vast API key inside request URLs (``…?api_key=<key>``); that output is
streamed into ``logs.txt`` (and would go to R2). :func:`redact` masks the value at capture time
so the secret never lands on disk; :func:`install_log_redaction` wires it onto fds 1/2 in the
supervisor so even subprocess output is filtered.
"""

from __future__ import annotations

import re

_REDACTED = "…REDACTED…"
# Secret value = run of non-delimiter chars after the key marker. Delimiters: & whitespace quotes.
_PATTERNS = (
    re.compile(r"(api_key=)[^&\s\"']+", re.IGNORECASE),
    re.compile(r"([?&][\w-]*?_key=)[^&\s\"']+", re.IGNORECASE),
    re.compile(r"(Authorization:\s*)\S+", re.IGNORECASE),
)


def redact(text: str) -> str:
    """Mask ``api_key=…`` / ``?…_key=…`` query params and ``Authorization:`` headers in ``text``.

    Idempotent: re-redacting already-masked text is a no-op-equivalent (the masked value carries
    no delimiters, so it just re-masks to the same string).
    """
    for pattern in _PATTERNS:
        text = pattern.sub(rf"\1{_REDACTED}", text)
    return text
