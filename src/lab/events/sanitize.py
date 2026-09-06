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
# A `--flag=value` or `--flag value` pair anywhere inside a larger string. Matched directly
# against the raw text (see `_mask_command_line`) rather than by tokenizing it — free text is not
# shell-quoted, and treating it as if it were is its own bug (below). Two things a naive
# `(--?[A-Za-z][\w-]*)(=|\s+)(\S+)` gets wrong, both fixed here:
#   - no boundary before the dash, so it fires *inside* an ordinary hyphenated word ("pass-key",
#     "well-authenticated" — the `-key`/`-auth...` tail reads as a flag). `(?<!\w)` requires the
#     character right before the dash(es) be a non-word character (or start of string), which an
#     ordinary compound word never has.
#   - the value is `\S+`, so a quoted multi-word value (`--api-key "abc def ghi"`) only captures
#     up to the first internal space, leaking the rest. The value alternation tries a balanced
#     quoted span first (double or single — matched by a plain paired-quote regex, never a
#     quote-interpreting tokenizer like `shlex`, which is what corrupted prose apostrophes
#     before) and only falls back to a bare `\S+` token when the value isn't quoted.
_INLINE_FLAG = re.compile(
    r"""
    (?<!\w)                     # boundary: not glued onto a preceding word character
    (--?[A-Za-z][\w-]*)         # the flag itself, e.g. --api-key, -k
    (=|\s+)                     # separator: inline '=' or one-or-more whitespace
    ("[^"]*"|'[^']*'|\S+)       # value: a quoted span as one unit, else a bare token
    """,
    re.VERBOSE,
)
# Flag names treated as "this looks like it holds a secret" by `_mask_flag` below — deliberately
# narrower than `_SECRET_KEY` (which stays as-is for dict keys in `_walk` and real argv tokens in
# `_mask_tokens`, both different, less ambiguous contexts). `_SECRET_KEY` includes "auth", which
# also matches plenty of ordinary, non-secret-bearing flag names that just *mention* auth
# (`--basic-auth`, `--no-auth`) with nothing secret-shaped following — "auth" alone is too weak a
# signal once it's being matched against arbitrary prose rather than a known argv key. Word-
# boundary anchored so "keyword" (one glued-together word, no separator) does not match "key" the
# way "--api-key" (a real hyphen-separated segment) does.
_SENSITIVE_FLAG_NAME = re.compile(r"\b(?:key|token|secret|password|credential)\b", re.IGNORECASE)
# A free-standing word, for the pass that catches a secret with no flag in front of it at all
# (e.g. "failed with key AKIA... during connect"). Applied only to text `_INLINE_FLAG` did not
# already consume/mask, via a second, independent sweep — see `_mask_command_line`.
_WORD = re.compile(r"\S+")


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
    MCP's ``command`` argument is the same shape), or free-form prose (a ``lab note``) that
    happens to contain a secret. Two independent passes, because neither alone is a complete
    masking strategy:

    1. Flag-aware (``_INLINE_FLAG``): masks a ``--flag=value``/``--flag value`` pair — quoted
       value included, as one unit — when the flag name looks secret-shaped. The flag-aware
       masking above (``_mask_tokens``) only ever saw separate argv tokens, so a secret hiding as
       a flag *value* inside one string token — where it has spaces around it, so
       ``_looks_secret`` bails, and ``redact()`` doesn't know the flag name — sailed through
       unmasked.
    2. Free-standing-secret (``_WORD`` + ``_looks_secret``): catches a bare secret-shaped string
       with no flag in front of it at all — the common shape of pasted error text/tracebacks,
       which is exactly what ``lab note`` exists to hold. Only pass 1 knows about flags; this
       pass reuses the same shape/entropy check ``_looks_secret`` applies elsewhere, run
       word-by-word over whatever pass 1 left behind (a word pass 1 already replaced with
       ``MASK`` trivially fails this check and is left alone).

    Both passes are matched directly against ``text`` with a regex, not by ``shlex.split``-ing it
    into tokens and rejoining: this used to tokenize with ``shlex``, which treats ``'`` and ``"``
    as *shell* quoting. Free text is not shell-quoted — a contraction (``it's``, ``wasn't``) or a
    quoted word is an apostrophe or a quotation mark, not the start of a span to swallow — and an
    even number of them across a note's text (or one that happens to balance against a later
    ``'``) made ``shlex.split`` succeed *silently*, stripping the quote characters and gluing
    everything between them into one token, mangling the text with no error or warning. A regex
    substitution touches only an actual ``flag=value``/``flag value`` pair (pass 1) or a single
    secret-shaped word (pass 2) and leaves every other character — punctuation, quotes,
    whitespace, ordinary hyphenated words — exactly as written. Never raises.
    """

    def _mask_flag(m: re.Match[str]) -> str:
        flag, sep, value = m.group(1), m.group(2), m.group(3)
        # Strip a value's surrounding quotes before judging its shape — `_looks_secret` should
        # see "abc def ghi", not '"abc def ghi"'. Only mask when there's actual reason to
        # believe a secret is here: either the value itself is secret-shaped (catches a real
        # secret behind an unremarkable flag name, e.g. `--seed <token>`), or the flag name is
        # one of the established sensitive names (catches a short, low-entropy secret —
        # `--password hunter2` — that `_looks_secret`'s length/entropy bar alone would miss).
        # Masking on flag name alone unconditionally (the prior behavior) is what over-masked
        # ordinary prose that merely *mentions* a flag (`--basic-auth flag`, `--keyword search`).
        quoted = len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]
        stripped = value[1:-1] if quoted else value
        if _looks_secret(stripped) or _SENSITIVE_FLAG_NAME.search(flag):
            return f"{flag}{sep}{MASK}"
        return m.group(0)

    def _mask_word(m: re.Match[str]) -> str:
        word = m.group(0)
        return MASK if _looks_secret(word) else word

    text = _INLINE_FLAG.sub(_mask_flag, text)
    return _WORD.sub(_mask_word, text)


def _mask_secrets(value: str) -> str:
    """Secret-masking only, no length cap: the piece of ``_scalar`` a caller whose whole point
    is holding free text in full (``lab.notes``) needs on its own — the 512-char cap right after
    this belongs to the ledger's argv/param records, not to a note body."""
    if _looks_secret(value):
        return MASK
    if re.search(r"\s", value):
        value = _mask_command_line(value)
    return redact(value)


def _scalar(value: Any) -> Any:
    if isinstance(value, str):
        value = _mask_secrets(value)
        return value[:MAX_STR] + "…" if len(value) > MAX_STR else value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return f"<{type(value).__name__}>"


def mask_text(value: str) -> str:
    """Mask likely secrets in free text, with no length cap (FR-J1).

    ``lab note``'s whole purpose is holding a detailed write-up in full — a couple hundred words
    is the *expected* shape, not an edge case — so the ledger's 512-char cap (meant for a CLI
    argv value or a params digest, where a diagnostic summary is all that's wanted) must not
    apply here. A note's text used to go through :func:`sanitize_argv`, which silently truncated
    it at 512 characters with no warning: 15 of 27 real notes on file were cut this way, several
    mid-sentence, with nothing in the record to say so (2026-08/09). Never raises: a masking
    failure must not take the note down with it.
    """
    try:
        return _mask_secrets(value)
    except Exception:  # noqa: BLE001 — masking must never fail whatever calls this
        return value


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
