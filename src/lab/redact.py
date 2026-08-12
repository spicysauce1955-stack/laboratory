"""Scrub secrets from captured subprocess output before it reaches disk (FR-J1).

SkyPilot/Vast log the Vast API key inside request URLs (``…?api_key=<key>``); gcloud/SkyPilot
on GCP can log OAuth material (``"access_token": "ya29.…"``, refresh tokens, service-account
``private_key`` JSON fields) and GCS **signed-URL** credentials (``X-Goog-Signature=…``), where
the signature itself grants read access until it expires. That output is streamed into
``logs.txt`` (and would go to R2).
:func:`redact` masks the value at capture time so the secret never lands on disk;
:func:`install_log_redaction` wires it onto fds 1/2 in the supervisor so even subprocess output
is filtered.
"""

from __future__ import annotations

import atexit
import os
import re
import sys
import threading
from pathlib import Path

_REDACTED = "…REDACTED…"
# Secret value = run of non-delimiter chars after the key marker. Delimiters: & whitespace quotes.
_PATTERNS = (
    re.compile(r"(api_key=)[^&\s\"']+", re.IGNORECASE),
    re.compile(r"([?&][\w-]*?_key=)[^&\s\"']+", re.IGNORECASE),
    # Mask the WHOLE header value (scheme + token), not just the first token: a real
    # `Authorization: Bearer <jwt>` has a space, so `\S+` would leave the token exposed.
    re.compile(r"(Authorization:[ \t]*).+", re.IGNORECASE),
    # GCP JSON credential fields (OAuth responses, ADC files, service-account keys). The value
    # class excludes `"` so the match stops at the closing quote and re-masks idempotently.
    re.compile(
        r'("(?:access_token|refresh_token|client_secret|id_token|private_key)"\s*:\s*")[^"]+',
        re.IGNORECASE,
    ),
    # Bare OAuth2 access tokens (gcloud logs them outside JSON too, e.g. in curl commands).
    re.compile(r"(ya29\.)[0-9A-Za-z_\-.]+"),
    # GCS V4 signed-URL credentials, which SkyPilot's bucket staging can emit into logs. The
    # signature *is* the credential — it grants whoever holds the URL read access to the object
    # until it expires — and X-Goog-Credential names the signing service account. The bucket and
    # object path are deliberately left intact so the line stays diagnosable (GCP-CREDS-5).
    re.compile(r"(X-Goog-(?:Signature|Credential)=)[^&\s\"']+", re.IGNORECASE),
)


def redact(text: str) -> str:
    """Mask ``api_key=…`` / ``?…_key=…`` query params and ``Authorization:`` headers in ``text``.

    Idempotent: re-redacting already-masked text is a no-op-equivalent (the masked value carries
    no delimiters, so it just re-masks to the same string).
    """
    for pattern in _PATTERNS:
        text = pattern.sub(rf"\1{_REDACTED}", text)
    return text


def install_log_redaction(log_path: str | Path) -> None:
    """Route this process's stdout+stderr (fds 1 & 2) through :func:`redact` into ``log_path``.

    Opens ``log_path`` (append), replaces fds 1/2 with the write end of a pipe, and drains the
    read end on a daemon thread that redacts each line before writing. Because child processes
    inherit fds 1/2, this also scrubs SkyPilot's subprocess output — the secret is filtered
    before it ever reaches disk. Call once, before any output that may carry a secret.
    """
    sink = open(log_path, "a", buffering=1, errors="replace")  # noqa: SIM115 — lives for process
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _drain() -> None:
        with os.fdopen(read_fd, "r", errors="replace") as pipe:
            for line in pipe:
                sink.write(redact(line))
                sink.flush()

    thread = threading.Thread(target=_drain, name="lab-log-redactor", daemon=True)
    thread.start()

    def _flush_on_exit() -> None:
        # Daemon threads are killed mid-flight at interpreter shutdown, which would drop the last
        # buffered lines (e.g. a teardown-failure annotation printed just before exit). Repoint
        # fds 1/2 at /dev/null so the only references to the pipe's write end are gone -> the
        # drain thread sees EOF, finishes writing every pending line, and we join it.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001 — best-effort flush during shutdown
                pass
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        thread.join(timeout=5)
        sink.close()

    atexit.register(_flush_on_exit)
