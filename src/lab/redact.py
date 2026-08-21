"""Scrub secrets from captured subprocess output before it reaches disk (FR-J1), and stamp it.

SkyPilot/Vast log the Vast API key inside request URLs (``…?api_key=<key>``); gcloud/SkyPilot
on GCP can log OAuth material (``"access_token": "ya29.…"``, refresh tokens, service-account
``private_key`` JSON fields) and GCS **signed-URL** credentials (``X-Goog-Signature=…``), where
the signature itself grants read access until it expires. That output is streamed into
``logs.txt`` (and would go to R2).
:func:`redact` masks the value at capture time so the secret never lands on disk;
:func:`install_log_redaction` wires it onto fds 1/2 in the supervisor so even subprocess output
is filtered.

That same pipe is where each job-log line gets its **UTC timestamp** (:class:`TimestampingWriter`).
Forensics on the 2026-08-20/21 supervisor incident hit a wall for want of one: a single job's log
had 1,608 lines — provisioning output, experiment stdout, 326 ssh failures, 278 ``[lab] queue poll
error``, 46 skipped heartbeat rsyncs — and not one time on any of them. When the network outage
began, whether the 7h wall-clock cap had already been passed, and how the log ordered against the
event ledger all had to be *inferred* by counting failure lines against the 60s heartbeat interval;
the first field report drew several wrong conclusions from that arithmetic and cost about an hour.
A time per line turns those into five-minute questions.
"""

from __future__ import annotations

import atexit
import codecs
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

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


# --- timestamping ------------------------------------------------------------------------------

_TERMINATOR = re.compile(r"[\r\n]")

# CSI/two-byte escape sequences, used only to ask "does this progress frame contain anything a
# person would read?" — SkyPilot erases the line before redrawing it (`\x1b[2K\r…`), and a
# stamped log line holding nothing but the erase code is pure noise.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-_]")

# How often a `\r` progress redraw may leave a line in the log. SkyPilot's spinner and rsync's
# progress rewrite one line many times a second; the old universal-newline decoding turned each
# frame into its own log line, so stamping per line would have meant thousands of near-identical
# timestamped fragments. Collapsing to at most one frame per interval keeps the log readable while
# still proving the job was alive at that moment (which is exactly what the incident needed).
_REDRAW_MIN_INTERVAL = 2.0

# A line with no terminator is held (see `TimestampingWriter.feed`), so a process that emits
# megabytes without a newline would otherwise grow the supervisor's memory without bound. Break
# it instead — at a whitespace boundary where possible, so a credential is not sliced in half and
# smuggled past `redact` in two pieces.
_MAX_LINE_CHARS = 1 << 20

_TS_ENV = "LAB_LOG_TIMESTAMPS"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(when: datetime) -> str:
    """``2026-08-21T04:12:33.123Z`` — ISO-8601, UTC, milliseconds.

    Milliseconds because the questions this answers are "how long between these two lines" and
    "did this happen before the cap": microseconds would only widen every line. Fixed width so a
    reader can strip the prefix with `cut -c1-25` or `^\\S+Z ` and diff two logs.
    """
    utc = when.astimezone(timezone.utc)
    return f"{utc.strftime('%Y-%m-%dT%H:%M:%S')}.{utc.microsecond // 1000:03d}Z"


def _timestamps_enabled() -> bool:
    """`LAB_LOG_TIMESTAMPS=0` restores the unstamped log format without a code change (blank
    means unset, per the repo's env convention)."""
    return os.environ.get(_TS_ENV, "").strip().lower() not in {"0", "false", "no", "off"}


class TimestampingWriter:
    """Line-oriented sink that prefixes each job-log line with a UTC timestamp, after redaction.

    This timestamps a **byte stream**, not logging calls: most of what arrives is other processes'
    output inherited through fds 1/2 (SkyPilot, ssh, rsync, the experiment on the remote box). So
    the hard cases are structural, and each is handled deliberately:

    * **Partial lines.** Text with no terminator yet is *held*, never flushed as a fragment. That
      is not only about not stamping mid-line: :func:`redact` matches ``api_key=`` and the key in
      one string, so emitting ``…?api_key=`` before its value arrived would push the value through
      a second, non-matching `redact` call and leak it. Holding until the terminator keeps the
      pattern whole; :meth:`close` flushes whatever is left, so nothing is lost.
    * **Chunked lines.** A line split over many ``write()``s is stamped exactly once, at its start,
      with the time its *first* bytes arrived — the event time, not the flush time.
    * **ANSI at line start.** The stamp goes before the whole line, so it can never land inside an
      escape sequence — including one straddling a chunk boundary (``\\x1b[`` | ``32m…``), which
      per-chunk stamping would have cut in half.
    * **`\\r` redraws.** A carriage return means "overwrite this line", not "end it": the frame it
      supersedes is dropped, and at most one frame per ``redraw_min_interval`` is kept so a live
      tail still shows progress. ``\\r\\n`` is a line ending, not a redraw, even when the two bytes
      arrive in different chunks. The trade-off is deliberate and it is the only lossy thing
      here: a producer using a bare ``\\r`` as its *only* line ending would keep one line per
      interval. Nothing in the supervisor's stream does (ssh, rsync and SkyPilot all end lines
      with ``\\n``), and ``LAB_LOG_TIMESTAMPS=0`` restores per-frame lines if one ever does.
    * **Invalid UTF-8.** Decoding is incremental with ``errors="replace"``: a multi-byte character
      split across chunks survives, and binary junk can never raise on the drain thread (an
      exception there would silently stop capturing the log for the rest of the job).

    All of this runs on the drain thread, never in the writer's callers — which matters because
    ``sky_runner._emit`` writes to fd 2 from a *signal handler* precisely to avoid taking a stream
    lock. It still just writes bytes into the pipe; the locking work happens over here.
    """

    def __init__(
        self,
        sink: TextIO,
        *,
        timestamps: bool = True,
        clock: Callable[[], datetime] = _utc_now,
        redraw_min_interval: float = _REDRAW_MIN_INTERVAL,
        max_line_chars: int = _MAX_LINE_CHARS,
    ) -> None:
        self._sink = sink
        self._timestamps = timestamps
        self._clock = clock
        # With stamping off the writer is a passthrough: every redraw frame becomes its own line,
        # which is exactly what the old universal-newline decoding produced.
        self._redraw_min_interval = redraw_min_interval if timestamps else 0.0
        self._max_line_chars = max_line_chars
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self._started_at: datetime | None = None
        self._last_frame_at = time.monotonic()
        self._closed = False

    def feed(self, chunk: bytes) -> None:
        """Absorb one read from the pipe and write out every line it completes."""
        text = self._decoder.decode(chunk)
        if not text:
            return
        if not self._pending:
            self._started_at = self._clock()
        self._pending += text
        self._consume()
        self._enforce_max_line()

    def close(self) -> None:
        """Flush the unterminated tail — a crashing process's last words have no newline."""
        if self._closed:
            return
        self._closed = True
        tail = self._decoder.decode(b"", final=True)
        if tail:
            if not self._pending:
                self._started_at = self._clock()
            self._pending += tail
        self._consume()
        # A lone trailing CR was held back in case it was the first half of a CRLF. At EOF it is
        # just the end of the last frame.
        if self._pending.endswith("\r"):
            self._pending = self._pending[:-1]
        if self._pending:
            self._emit(self._pending)
            self._pending = ""
        self._sink.flush()

    # -- internals --

    def _consume(self) -> None:
        while True:
            match = _TERMINATOR.search(self._pending)
            if match is None:
                return
            i = match.start()
            if self._pending[i] == "\n":
                self._emit(self._pending[:i])
                self._advance(i + 1)
                continue
            nxt = self._pending[i + 1 : i + 2]
            if nxt == "":
                return  # trailing CR: might be a CRLF split across chunks, so wait for the rest
            if nxt == "\n":
                self._emit(self._pending[:i])
                self._advance(i + 2)
                continue
            self._redraw(self._pending[:i])
            self._advance(i + 1)

    def _redraw(self, frame: str) -> None:
        """A frame the next one is about to overwrite: keep it only if it is this interval's."""
        now = time.monotonic()
        if _ANSI.sub("", frame).strip() and now - self._last_frame_at >= self._redraw_min_interval:
            self._emit(frame)
            self._last_frame_at = now

    def _advance(self, n: int) -> None:
        self._pending = self._pending[n:]
        # Whatever is left arrived in this same read, so "now" is its start time.
        self._started_at = self._clock() if self._pending else None

    def _enforce_max_line(self) -> None:
        while len(self._pending) > self._max_line_chars:
            window = self._pending[: self._max_line_chars]
            cut = max(window.rfind(" "), window.rfind("\t"))
            if cut > self._max_line_chars // 2:
                self._emit(self._pending[:cut])
                self._advance(cut + 1)  # the break replaces the whitespace it broke at
            else:
                self._emit(window)
                self._advance(self._max_line_chars)

    def _emit(self, text: str) -> None:
        out = redact(text)
        if self._timestamps and out:  # a blank separator line carries no event; leave it blank
            out = f"{_stamp(self._started_at or self._clock())} {out}"
        self._sink.write(out + "\n")
        self._sink.flush()


def install_log_redaction(log_path: str | Path, *, timestamps: bool | None = None) -> None:
    """Route this process's stdout+stderr (fds 1 & 2) through :func:`redact` into ``log_path``.

    Opens ``log_path`` (append), replaces fds 1/2 with the write end of a pipe, and drains the
    read end on a daemon thread that redacts and timestamps each line before writing. Because
    child processes inherit fds 1/2, this also scrubs SkyPilot's subprocess output — the secret is
    filtered before it ever reaches disk. Call once, before any output that may carry a secret.

    ``timestamps`` defaults to the ``LAB_LOG_TIMESTAMPS`` env switch (on unless set to ``0``), so
    the stamped format can be turned off in the field without redeploying a supervisor.
    """
    sink = open(log_path, "a", buffering=1, errors="replace")  # noqa: SIM115 — lives for process
    stamped = _timestamps_enabled() if timestamps is None else timestamps
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _drain() -> None:
        # Raw byte reads, not `os.fdopen(...)` line iteration: text-mode universal newlines turn
        # every `\r` progress frame into its own line, and a decode error on binary junk would
        # kill this thread and with it the rest of the job's log.
        writer = TimestampingWriter(sink, timestamps=stamped)
        try:
            while True:
                try:
                    chunk = os.read(read_fd, 65536)
                except OSError:
                    break  # the write ends are gone / the fd was closed under us
                if not chunk:
                    break
                writer.feed(chunk)
        finally:
            writer.close()
            os.close(read_fd)

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
