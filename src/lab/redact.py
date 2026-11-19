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
        r'("(?:access_token|refresh_token|client_secret|id_token|private_key'
        r'|password|passwd|secret|token|api_?key)"\s*:\s*")[^"]+',
        re.IGNORECASE,
    ),
    # Credential-bearing query params the `_key=` pattern above does not cover: bearer/session
    # tokens, passwords, and Azure-style `sig=` signatures.
    re.compile(
        r"([?&](?:[\w-]*?[_-])?(?:token|secret|password|passwd|signature|sig)=)[^&\s\"']+",
        re.IGNORECASE,
    ),
    # S3/R2 presigned-URL credentials, the twin of the GCS pattern below: the signature grants
    # access until expiry, the credential names the access key id, the session token is a key.
    re.compile(r"(X-Amz-(?:Signature|Credential|Security-Token)=)[^&\s\"']+", re.IGNORECASE),
    # Bare OAuth2 access tokens (gcloud logs them outside JSON too, e.g. in curl commands).
    re.compile(r"(ya29\.)[0-9A-Za-z_\-.]+"),
    # GCS V4 signed-URL credentials, which SkyPilot's bucket staging can emit into logs. The
    # signature *is* the credential — it grants whoever holds the URL read access to the object
    # until it expires — and X-Goog-Credential names the signing service account. The bucket and
    # object path are deliberately left intact so the line stays diagnosable (GCP-CREDS-5).
    re.compile(r"(X-Goog-(?:Signature|Credential)=)[^&\s\"']+", re.IGNORECASE),
)


#: Credentials recognisable by shape alone, with no name in front of them: ``(prefix)(body)``.
#: The prefix is kept so a reader can tell *what* leaked and rotate it; the body is masked.
#: Every prefix is one a vendor issues so that leaked tokens can be found, so none of them occurs
#: in ordinary text — and the left boundary keeps ``task-…``/``disk-…`` from reading as ``sk-``.
#: Length floors sit below the real formats (GitHub 36, HF 34, DO 64…) so a truncated paste is
#: still caught, and above anything a prose word could reach.
_TOKEN_SHAPES = (
    r"(gh[pousr]_)[A-Za-z0-9]{30,}",  # GitHub classic PAT / OAuth / user / server / refresh
    r"(github_pat_)[A-Za-z0-9_]{40,}",  # GitHub fine-grained PAT
    # OpenAI / Anthropic-style secret keys. `sk-` alone is too short a prefix to trust on its own,
    # so the body must be either a long unbroken run (legacy keys, and the all-letter fakes the
    # stress run used) or contain a digit — `sk-learn-is-a-python-library` is neither.
    r"(sk-(?:proj|svcacct|admin|ant-(?:api|admin)\d\d)-)[A-Za-z0-9_-]{20,}",
    r"(sk-)(?:[A-Za-z0-9]{20,}[A-Za-z0-9_-]*|(?=[A-Za-z0-9_-]{20,})[A-Za-z_-]*[0-9][A-Za-z0-9_-]*)",
    r"((?:sk|rk)_(?:live|test)_)[A-Za-z0-9]{16,}",  # Stripe secret / restricted keys
    r"(hf_)[A-Za-z0-9]{30,}",  # Hugging Face
    r"(do[por]_v1_)[A-Za-z0-9]{40,}",  # DigitalOcean PAT / OAuth / refresh token
    r"(xox[abposre]-)[A-Za-z0-9-]{10,}",  # Slack
    r"(AIza)[0-9A-Za-z_-]{35}",  # Google API key
    r"(eyJ)[A-Za-z0-9_-]{2,}\.eyJ[A-Za-z0-9_-]{2,}\.[A-Za-z0-9_-]*",  # JWT: {"…}.{"…}.signature
)
_TOKEN = re.compile(r"(?<![A-Za-z0-9_-])(?:" + "|".join(_TOKEN_SHAPES) + ")")

# A private key is many lines, and no single line of its body says what it is — so the block is
# masked from BEGIN to END as a unit, with the markers kept (the reader must be able to see that
# a key was printed, and rotate it). A certificate is public and is left alone.
_KEY_KIND = r"[A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?"
_KEY_BEGIN = re.compile(rf"-----BEGIN {_KEY_KIND}-----")
_KEY_END = re.compile(rf"-----END {_KEY_KIND}-----")
# What may sit between the markers: base64 lines, `Proc-Type:`-style armour headers, blank lines
# — each optionally behind the log writer's own timestamp, since an older logs.txt is stamped
# line by line. The first line that is none of these ends the block even with no END marker:
# masking the rest of a log after a stray BEGIN would destroy the only copy of the run's output.
_STAMP_PREFIX = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z ")
_KEY_LINE = re.compile(
    r"\s*(?:[A-Za-z0-9+/=]+"
    r"|(?:Proc-Type|DEK-Info|Version|Comment|Hash|Charset|MessageID):[^\n]*)?\s*"
)


def _masked_span(text: str) -> str:
    """``text`` (one line's worth of key material) masked, keeping a leading log timestamp."""
    stamp = _STAMP_PREFIX.match(text)
    body = text[stamp.end():] if stamp else text
    return (stamp.group(0) if stamp else "") + (_REDACTED if body.strip() else body)


def _mask_key_line(line: str, in_block: bool) -> tuple[str, bool]:
    """One line through the private-key state machine: ``(masked line, still inside a block)``."""
    if in_block:
        end = _KEY_END.search(line)
        if end is not None:
            rest, still = _mask_key_line(line[end.end():], False)
            return _masked_span(line[:end.start()]) + line[end.start():end.end()] + rest, still
        stamp = _STAMP_PREFIX.match(line)
        if not _KEY_LINE.fullmatch(line[stamp.end():] if stamp else line):
            return _mask_key_line(line, False)  # not key material: the block is over
        return _masked_span(line), True
    begin = _KEY_BEGIN.search(line)
    if begin is None:
        return line, False
    head, tail = line[:begin.end()], line[begin.end():]
    if _KEY_END.search(tail) is None:
        return head + (_REDACTED if tail.strip() else tail), True
    rest, still = _mask_key_line(tail, True)
    return head + rest, still


def _mask_private_keys(text: str) -> str:
    if "PRIVATE KEY" not in text:
        return text
    out: list[str] = []
    in_block = False
    for line in text.split("\n"):
        masked, in_block = _mask_key_line(line, in_block)
        out.append(masked)
    return "\n".join(out)


# --- names that say "the value next to me is a credential" -------------------------------------
#
# One name test for every surface — this module's log redaction and `lab.events.sanitize`'s
# ledger/note masking both use `sensitive_name`, so a log line and a ledger record can never
# disagree about what counts as a credential's name.
#
# The name segment is matched against a name lower-cased with camelCase split into `_` segments,
# with *alphanumeric-only* boundaries: `_`, `-` and `.` separate segments, so `--wandb_key`,
# `--api_key`, `DB_PASSWORD` and `apiKey` match, while `keyboard`, `monkey`, `tokenizer`,
# `keyword` and `max_tokens` (a count, not a credential — hence singular `token` only) do not.
# `\b` is the bug this replaced: `_` is a word character, so `\bkey\b` never matched `--wandb_key`.
# It must also be the name's LAST segment (or be followed only by `_id`, as in
# `aws_access_key_id`): a credential is what the name *is*, while `key_dim`, `token_budget` or
# `model.key_dim=64` merely describe something else, and masking a hyperparameter protects nothing.
_NAME_WORD = re.compile(
    r"(?<![a-z0-9])"
    r"(?:(?:api|access|secret|private|auth|client|session|refresh|bearer)?(?:key|token)"
    r"|secret|passw(?:or)?d|passphrase|credentials?|signature|authorization)"
    r"(?:[_.-]+id)?$"
)
# `pass` alone is a count as often as a password (`--pass 3`, `pass_rate`), so it only counts as
# the *last* segment of a compound name: `DB_PASS`, `SMTP_PASS`, `--db-pass`.
_TRAILING_PASS = re.compile(r"[a-z0-9][_.-]+pass$")
# Env-var convention glues a prefix straight onto the word with no separator — `PGPASSWORD`,
# `MYSQLPASSWORD` — so the segment boundary above never fires. Only for the password words: no
# ordinary name ends in `password`, while `monkey`, `turkey` and `hockey` end in `key`.
_GLUED_PASSWORD = re.compile(r"[a-z0-9](?:passw(?:or)?d|passphrase)$")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
#: Names that are *only* the generic word. In a ledger argv (`--token X`) they are a credential;
#: in prose and in a run's log they are a dict key, a sort key, a tokenizer's token — far more
#: often than not — so the prose-shaped forms below require a qualified name instead.
_BARE_NAMES = frozenset({"key", "token"})


def sensitive_name(name: str, *, bare: bool = True) -> bool:
    """Does a flag / env-var / field / query-param *name* say its value is a credential?

    ``bare=False`` refuses a name that is nothing but ``key`` or ``token`` (``key=``,
    ``--token``, ``Token:``) while still accepting every qualified one (``api_key``,
    ``HF_TOKEN``) — the setting for text where the bare word is usually not a credential.
    """
    norm = _CAMEL_BOUNDARY.sub("_", name).lower()
    if not bare and norm.strip("-_.$ ") in _BARE_NAMES:
        return False
    return bool(
        _NAME_WORD.search(norm) or _TRAILING_PASS.search(norm) or _GLUED_PASSWORD.search(norm)
    )


# A value: a bounded quoted span as one unit (so `password='a b'` and `password="x"` mask whole),
# else a bare run. `&` `,` `;` end a bare run only where they start the *next* `name=` pair (or
# end the text) — `?password=x&user=y` keeps `&user=y`, while `password=Tr0ub4dor&3` no longer
# leaves `&3` behind.
_VALUE = (
    r"""("[^"\n]{0,200}"|'[^'\n]{0,200}'"""
    r"""|(?:[^\s"'&,;]|[&,;](?![A-Za-z_][\w.-]*=|[\s"']|$))+)"""
)
#   - `NAME=value` (glued) and `NAME = value` (INI/TOML): an env prefix (`DB_PASSWORD=x python
#     t.py`), a query parameter, a config override, a dataclass repr. Not glued to a preceding
#     word or dash, so `--flag=value` stays the flag rule's. `==` is a comparison, not this.
_ASSIGNMENT = re.compile(rf"(?<![\w-])([A-Za-z_][A-Za-z0-9_.-]{{0,63}})([ \t]*=(?!=)[ \t]*){_VALUE}")
#   - `name: value`: YAML, a Hydra config dump, `password: hunter2` in a pasted error.
_COLON = re.compile(
    rf"(?<![\w.$-])([A-Za-z_][A-Za-z0-9_.-]{{0,63}})([ \t]*:[ \t]*)(?!//){_VALUE}"
)
#   - `"name": "value"` / `'name': 'value'`: JSON or a Python dict repr inside a line.
_FIELD = re.compile(
    r"""(["'])([^"'\\\n]{1,64})\1(\s*:\s*)(["'])((?:(?!\4)[^\\\n]|\\.){0,512})\4"""
)
#   - URL userinfo: `scheme://user:password@host`.
_USERINFO = re.compile(r"(://[^/\s:@]{1,128}:)([^/\s@]{1,256})(@)")
#   - A bearer token with no `Authorization:` in front of it.
_BEARER = re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._~+/=-]+)")
#   - `--flag=value` / `--flag value` in a log line (a command echo, a usage string).
_FLAG = re.compile(r"""(?<![\w-])(--?[A-Za-z][\w-]*)(=|[ \t]+)("[^"\n]{0,200}"|(?!-)[^\s"']+)""")


def _masked_value(value: str) -> str:
    """The mask, keeping a quoted value's quotes so the line still parses as what it was."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return f"{value[0]}{_REDACTED}{value[0]}"
    return _REDACTED


def mask_named_values(text: str, *, bare: bool = True) -> str:
    """Mask values whose *name* marks them as credentials, wherever they sit in ``text``:
    ``NAME=value`` / ``NAME = value`` assignments, ``name: value``, JSON / dict-repr fields, URL
    userinfo and bare bearer tokens. Needs no whitespace, so it covers a compact JSON body or a
    URL that is one "word". Leaves every other character alone, and is idempotent.

    ``bare`` is :func:`sensitive_name`'s: whether a name that is just ``key``/``token`` counts.
    The prose-shaped forms (``NAME = value`` with spaces, ``name: value``) never accept a bare
    name, whatever ``bare`` says — "the key: it was the accelerator" is a sentence.
    """

    def _assignment(m: re.Match[str]) -> str:
        name, sep, value = m.group(1), m.group(2), m.group(3)
        if not sensitive_name(name, bare=bare and sep == "="):
            return m.group(0)
        return f"{name}{sep}{_masked_value(value)}"

    def _colon(m: re.Match[str]) -> str:
        name, sep, value = m.group(1), m.group(2), m.group(3)
        if not sensitive_name(name, bare=False):
            return m.group(0)
        return f"{name}{sep}{_masked_value(value)}"

    def _field(m: re.Match[str]) -> str:
        q1, name, sep, q2 = m.group(1), m.group(2), m.group(3), m.group(4)
        if not sensitive_name(name, bare=bare):
            return m.group(0)
        return f"{q1}{name}{q1}{sep}{q2}{_REDACTED}{q2}"

    text = _FIELD.sub(_field, text)
    text = _ASSIGNMENT.sub(_assignment, text)
    text = _COLON.sub(_colon, text)
    text = _USERINFO.sub(rf"\g<1>{_REDACTED}\g<3>", text)
    return _BEARER.sub(rf"\g<1>{_REDACTED}", text)


def _flag(m: re.Match[str]) -> str:
    flag, sep, value = m.group(1), m.group(2), m.group(3)
    if not sensitive_name(flag, bare=False):
        return m.group(0)
    return f"{flag}{sep}{_masked_value(value)}"


def redact(text: str) -> str:
    """Mask credentials in ``text``: private-key blocks, vendor token shapes (``ghp_…``,
    ``sk-…``, ``hf_…``, JWTs, …), ``api_key=…`` / ``?…_key=…`` query params, ``Authorization:``
    headers, and any value whose name says it is a credential (``AWS_SECRET_ACCESS_KEY=…``,
    ``password: …``, ``--db-password …``) — :func:`sensitive_name` with ``bare=False``, since a
    lone ``key``/``token`` in a run's output is a dict key far more often than a secret, and
    this is the only copy of that output.

    Idempotent: re-redacting already-masked text is a no-op-equivalent (the masked value carries
    no delimiters, so it just re-masks to the same string).
    """
    text = _mask_private_keys(text)
    text = _TOKEN.sub(lambda m: next(g for g in m.groups() if g is not None) + _REDACTED, text)
    for pattern in _PATTERNS:
        text = pattern.sub(rf"\1{_REDACTED}", text)
    text = mask_named_values(text, bare=False)
    return _FLAG.sub(_flag, text)


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
        # Inside a private-key block: no single line of a key's body says what it is, so the
        # line-at-a-time `redact` cannot see one — this carries the BEGIN across lines.
        self._in_private_key = False

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
        text, self._in_private_key = _mask_key_line(text, self._in_private_key)
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
