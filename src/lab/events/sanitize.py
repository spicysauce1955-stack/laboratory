"""What may be written. Recording argv means recording whatever was typed, and
:func:`lab.redact.redact` only knows patterns that appear in *subprocess output* — it will not
catch a key passed as a flag value. Everything entering the ledger passes through here first
(FR-J1, AC-7)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from lab.redact import mask_named_values, redact, sensitive_name

MASK = "…REDACTED…"
MAX_STR = 512
MAX_ITEMS = 32

_SECRET_KEY = re.compile(r"key|token|secret|password|credential|auth", re.IGNORECASE)
_BASE64_BLOB = re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$")  # bare base64 blobs
_SECRET_VALUE = (
    re.compile(r"^ya29\."),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    _BASE64_BLOB,
    # Provider key shapes too short (or too low-entropy) for the generic bar below. Anchored at
    # the start and closed with a no-more-key-characters guard, so they fire on one *word*
    # (`_mask_word`) and never mask a whole sentence that merely contains one.
    re.compile(r"^(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Za-z])"),  # AWS access key id
    re.compile(r"^AIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])"),  # Google API key
    re.compile(r"^dop_v1_[0-9a-f]{64}(?![0-9a-f])"),  # DigitalOcean personal access token
)
# Does a flag / env-var / field / query-param *name* say its value is a credential? The one name
# test, shared with the log redaction in `lab.redact` (where it lives, since this module imports
# that one): a log line and a ledger record must never disagree about a credential's name.
_sensitive_name = sensitive_name


def _sensitive_key(key: str) -> bool:
    """The dict-key / argv-flag test: the historical broad substring match, plus the bounded name
    test (which adds what the substring missed — `DB_PASS`, `passwd`, `signature`)."""
    return bool(_SECRET_KEY.search(key)) or _sensitive_name(key)
_HEXISH = re.compile(r"^[0-9a-f-]+$", re.IGNORECASE)  # commits, cell ids, job ids — not secrets
# A `--flag=value` or `--flag value` pair anywhere inside a larger string. Matched directly
# against the raw text (see `_mask_command_line`) rather than by tokenizing it — free text is not
# shell-quoted, and treating it as if it were is its own bug (below). Naive versions of this get
# several things wrong; each is fixed here:
#   - no boundary before the dash, so it fires *inside* an ordinary hyphenated word ("pass-key",
#     "well-authenticated" — the `-key`/`-auth...` tail reads as a flag). `(?<!\w)` requires the
#     character right before the dash(es) be a non-word character (or start of string), which an
#     ordinary compound word never has.
#   - a bare `\S+` value alternative with no restriction on its own shape will happily match a
#     *following* `--flag` token as if it were the current flag's value (`--dry-run --api-key ...`
#     matches flag=`--dry-run`, value=`--api-key`, consuming both in one span). That leaves the
#     real flag+secret pair never independently tried — the secret sails through unmasked. Fixed
#     by `(?!-)\S+`: a bare value may not itself start with `-`, so a boolean flag immediately
#     followed by another flag (nothing real between them) simply fails to match here at all —
#     correctly, since it isn't a `flag=value` pair — and the scan resumes at the next flag,
#     giving it its own independent match.
#   - an *unbounded* quoted-value alternative (`'[^']*'`) has no cap on distance: if the opening
#     quote has no genuine closing partner nearby, backtracking happily extends the match to the
#     next literal quote character anywhere later in the string — including an ordinary
#     apostrophe inside an unrelated contraction many words away, silently deleting real prose in
#     between. Double quotes carry far less of this risk in ordinary English prose (contractions
#     never use `"`), so only `"..."` is treated as a genuine quoted value; a `'`-containing value
#     (an unterminated quote, a contraction) falls through to the bare-token alternative above,
#     which is bounded by the next whitespace/flag — at worst one word gets swept up, never an
#     unbounded span. The double-quoted alternative is itself still explicitly bounded in length
#     (`{0,200}`, no newline) so even a genuinely unterminated `"` can't reach an unrelated `"`
#     much later in a long note.
_INLINE_FLAG = re.compile(
    r"""
    (?<!\w)                     # boundary: not glued onto a preceding word character
    (--?[A-Za-z][\w-]*)         # the flag itself, e.g. --api-key, -k
    (=|\s+)                     # separator: inline '=' or one-or-more whitespace
    ("[^"\n]{0,200}"|(?!-)\S+)  # value: a bounded double-quoted span as one unit, else a bare
                                 # token that doesn't itself look like a flag
    """,
    re.VERBOSE,
)
# Flag names treated as "this looks like it holds a secret" by `_mask_flag` below are judged by
# `_sensitive_name` — deliberately narrower than `_SECRET_KEY` (which stays as-is for dict keys in
# `_walk` and real argv tokens in `_mask_tokens`, both different, less ambiguous contexts).
# `_SECRET_KEY` includes "auth", which also matches plenty of ordinary, non-secret-bearing flag
# names that just *mention* auth (`--basic-auth`, `--no-auth`) with nothing secret-shaped
# following — "auth" alone is too weak a signal once it's being matched against arbitrary prose
# rather than a known argv key. Segment-bounded so "keyword" (one glued-together word, no
# separator) does not match "key" the way "--api-key" and "--api_key" do.
# A free-standing word, for the pass that catches a secret with no flag in front of it at all
# (e.g. "failed with key AKIA... during connect"). Applied only to text `_INLINE_FLAG` did not
# already consume/mask, via a second, independent sweep — see `_mask_command_line`.
_WORD = re.compile(r"\S+")


#: An object-store key, and nothing else: ``[A-Za-z0-9._-]`` segments joined by ``/``, at least
#: two of them, first character alphanumeric. Note what the character class leaves *out* — every
#: delimiter a credential travels with (``=`` base64 padding, ``+``, ``?``/``&`` query strings,
#: ``:``/``@`` URL userinfo, whitespace) — so a secret in its usual company cannot even reach the
#: rest of the test. Matched with ``fullmatch``.
_OBJECT_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9._-]+)+")
#: An unbroken alphanumeric run long enough to *be* a credential, anywhere inside a key. 32 is the
#: same bar :func:`_looks_secret` uses ("< 32 chars is not a secret"), applied here to runs rather
#: than to the whole string, because a path's structure is exactly what a pasted blob lacks: real
#: keys are punctuated (``j-20260908-1a2b3c4d-shard-07-of-32.json`` — longest run 8), while an API
#: key hiding as a path segment is one long run.
_LONG_ALNUM_RUN = re.compile(r"[A-Za-z0-9]{32,}")
#: The exemption is bounded. R2 allows a 1024-byte key; nothing lab writes comes near this, and an
#: unbounded pass-through would be a channel for putting arbitrary text into the ledger.
MAX_OBJECT_KEY = 256


class ObjectKey(str):
    """An object-store key an internal call site has vetted as non-secret, so that a diagnostic
    can name the object it is about. ``str`` subclass: it serialises, formats and compares
    exactly like the key it wraps.

    **Why it exists.** ``r2queue.list_mirrored``/``list_entries`` skip an unparseable blob and
    record ``events.note("queue.manifest_corrupt", key=..., ...)``. That note has never once
    named the offending object: ``key`` matches ``_SECRET_KEY``, and renaming the field doesn't
    rescue it either, because a real key is long and high-entropy enough that ``_looks_secret``
    masks it on sight (``queue/jobs/j-20260908-1a2b3c4d-shard-07-of-32.json`` measures 4.45
    bits). The single identifier the diagnostic exists to carry was the one thing destroyed, and
    only the stderr line — gone the moment the terminal scrolls — ever named it.

    **Why it cannot pass a credential.** Three independent reasons, in order of strength:

    1. *Provenance.* This is a Python type, and nothing crossing a trust boundary can ever be
       one. argv arrives as ``str`` from the shell; MCP tool arguments, ledger records and
       queue blobs arrive as ``str`` from ``json.loads``; config values arrive as ``str`` from
       YAML/pydantic; a note body arrives as ``str`` from ``-m``. Deserialisation cannot
       construct a subclass it has never heard of. A value therefore becomes an ``ObjectKey``
       only where lab's own source says so — an enumerable, reviewable set of call sites (grep
       the constructor), never as a consequence of anything a user typed.
       Note the limit of this argument, which is why it is not the only one: the first call
       sites (``r2queue``'s corruption notes) wrap a key that came back from an R2 *listing*,
       i.e. whatever object names exist in the bucket — lab chose the field, not the bytes. For
       anything remote-influenced it is (2), not (1), that carries the weight.
    2. *Shape.* Even at those call sites the exemption is conditional, so it survives an author
       wrapping the wrong thing: the value must fullmatch :data:`_OBJECT_KEY`, stay within
       :data:`MAX_OBJECT_KEY`, contain no :data:`_LONG_ALNUM_RUN`, and match none of the
       credential patterns ``_SECRET_VALUE`` already knows (``ya29.`` OAuth tokens, PEM blocks,
       bare base64 blobs — which is where an AWS-style secret access key lands, 40 base64 chars
       that genuinely do contain ``/`` and would otherwise look path-shaped). Anything failing
       any of those falls back to :func:`_scalar`, i.e. the unmodified deny-list. A mistaken
       wrap therefore degrades to today's behaviour; it can never *unmask* something.
    3. *Blast radius.* Nothing outside this one branch of :func:`_walk` changes. ``_looks_secret``,
       ``_SECRET_KEY``, ``_INLINE_FLAG`` and every default-pass path behave exactly as before for
       every plain ``str``, and the two paths that handle user input — :func:`sanitize_argv` and
       :func:`mask_text` — are given no bypass at all: they take ``str``, and a ``str`` is
       masked. ``redact()`` still runs on what is let through. (``_looks_secret`` has since
       learned to spare *anchored* structured words — a lab id, a ``lab-`` resource name, a
       digest, a rooted path or a scheme URL; see :func:`_structured`. A relative key like the
       one above carries none of those anchors and is still masked as a plain ``str``.)

    The residual risk is "a lab author wraps a secret in source", which is a code-review-visible
    act that (2) then has to be talked past as well. That is a much smaller hole than the
    alternative on offer — recognising object-key *shape* for every value, which would exempt
    any credential that happens to look like a path, from anyone, with no author in the loop.
    """

    __slots__ = ()


def _entropy(s: str) -> float:
    counts = {c: s.count(c) for c in set(s)}
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in counts.values())


# --- structured words the entropy bar must not take ---------------------------------------------
#
# The entropy bar is right for an *unknown* credential — one long run from a token alphabet — and
# wrong for anything structured: a run path, an object URL, a digest, a cluster name, the lab's
# own hint text all measure > 3.5 bits at ≥ 32 chars, and were masked on sight. That cost the
# ledger exactly the identifiers forensics needs: which output file, which bucket object, which
# cluster a teardown note is about (stress run 2026-11-19). `_structured` exempts such a word, and
# only one that (a) carries its own provenance and (b) has no credential-shaped run inside it.

#: A lab job id (``lab.core._new_job_id``), alone or inside a sweep/reg id, a run path, an object
#: key or a cluster name. Nothing a vendor issues is shaped like a UTC timestamp plus 6 hex.
_LAB_ID = re.compile(r"(?<![0-9])[0-9]{8}-[0-9]{6}-[0-9a-f]{6}(?![0-9a-f])")
#: A name lab gives its own cloud resources (``lab-<slug>-<job_id>``, and the SkyPilot nodes built
#: from one — GCP truncates those until no job id is left): lowercase kebab-case.
_LAB_NAME = re.compile(r"lab(?:-[a-z0-9]+)+")
#: A content digest says what it is: ``sha256:<hex>``, optionally behind an image reference.
_DIGEST = re.compile(
    r"(?:[\w./:-]*@)?(?:sha1|sha224|sha256|sha384|sha512|md5|blake2[bs]):[0-9a-f]{32,128}"
)
_URL_SCHEME = re.compile(r"[a-z][a-z0-9+.-]*://")
_URL_USERINFO = re.compile(r"://[^/\s]*:[^/\s]*@")
_ROOTED = re.compile(r"(?:~|\.{1,2})?/")
#: A character no standard-base64 credential contains. A word with ``/`` that also has one of
#: these cannot be standard base64 (and base64url and hex have no ``/``), which is what separates
#: a path from an AWS-style secret access key that merely contains slashes.
_NOT_BASE64 = re.compile(r"[^A-Za-z0-9+/=]")
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]{16,}")
#: Quoting and punctuation a word picks up from the sentence around it.
_EDGES = "\"'`()[]{}<>,;:."


def _credential_run(text: str) -> bool:
    """Does ``text`` contain a run that could be a credential on its own? 32+ alphanumerics, or
    16+ that mix letters and digits — the shape of every random token, and of nothing lab names:
    its ids, slugs, uuid8s and file names are punctuated into short runs."""
    for m in _ALNUM_RUN.finditer(text):
        run = m.group(0)
        if len(run) >= 32 or (any(c.isdigit() for c in run) and any(c.isalpha() for c in run)):
            return True
    return False


def _structured(word: str) -> bool:
    """Is ``word`` recognisably *not* a credential, for all its length and entropy?

    Needs provenance in the word itself — a digest prefix, a lab id or ``lab-`` resource name, a
    backtick (markdown, and the lab's own hint text), or a path/URL anchored at a filesystem root
    or a scheme — and nothing credential-shaped inside it. A relative object key with none of
    these (``queue/jobs/….json``) is *not* exempt: that is :class:`ObjectKey`'s job, vetted at the
    call site, by design.
    """
    core = word.strip(_EDGES)
    if _DIGEST.fullmatch(core):
        return True
    if _credential_run(core) or _URL_USERINFO.search(core):
        return False
    if _LAB_ID.search(core) or _LAB_NAME.fullmatch(core) or "`" in word:
        return True
    return (
        "/" in core
        and _NOT_BASE64.search(core) is not None
        and (_URL_SCHEME.match(core) is not None or _ROOTED.match(core) is not None)
    )


def _looks_secret(value: str) -> bool:
    # Hex-only strings ≤40 chars (commit SHAs, cell ids, job IDs) are never secrets, so check
    # first to prevent the base64 pattern from matching 40-char hex strings. Longer hex strings
    # (e.g. 64-char hex API tokens) fall through to entropy check and are masked — a bare 64-hex
    # value is the Vast API key / R2 secret shape; only a digest *prefix* (`_structured`) exempts.
    if _HEXISH.match(value) and len(value) <= 40:
        return False
    if any(p.search(value) for p in _SECRET_VALUE):
        return True
    if " " in value or len(value) < 32:
        return False
    if _structured(value):
        return False
    return _entropy(value) > 3.5


def _is_object_key(text: str) -> bool:
    """Is ``text`` unambiguously an object-store key? See :class:`ObjectKey` for the argument."""
    if not text or len(text) > MAX_OBJECT_KEY:
        return False
    if not _OBJECT_KEY.fullmatch(text):
        return False
    if _LONG_ALNUM_RUN.search(text):
        return False
    return not any(p.search(text) for p in _SECRET_VALUE)


def _vetted_object_key(value: ObjectKey) -> Any:
    """Let a vetted key through verbatim, or fall back to the ordinary deny-list if it does not
    look like one. ``str(value)`` first, so what lands in the record is a plain ``str`` and not a
    subclass smuggling the exemption onwards."""
    text = str(value)
    return redact(text) if _is_object_key(text) else _scalar(text)


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
            out.append(f"{flag}={MASK}" if _sensitive_key(flag) else f"{flag}={_scalar(value)}")
            continue
        if token.startswith("-") and _sensitive_key(token):
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
        # see "abc def ghi", not '"abc def ghi"'. Only double quotes are ever treated as a
        # genuine delimiter here (see `_INLINE_FLAG` above) — a single quote in `value` is just
        # part of a bare token (an unterminated quote, a contraction), never something to strip.
        # Only mask when there's actual reason to believe a secret is here: either the value
        # itself is secret-shaped (catches a real secret behind an unremarkable flag name, e.g.
        # `--seed <token>`), or the flag name is one of the established sensitive names (catches
        # a short, low-entropy secret — `--password hunter2` — that `_looks_secret`'s
        # length/entropy bar alone would miss). Masking on flag name alone unconditionally (the
        # prior behavior) is what over-masked ordinary prose that merely *mentions* a flag
        # (`--basic-auth flag`, `--keyword search`).
        quoted = len(value) >= 2 and value[0] == '"' and value[-1] == '"'
        stripped = value[1:-1] if quoted else value
        if _looks_secret(stripped) or _sensitive_name(flag):
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
    this belongs to the ledger's argv/param records, not to a note body.

    Named values first (``lab.redact.mask_named_values``: ``NAME=value``, ``name: value``, JSON
    fields, URL userinfo, bearer tokens — ``bare=True``, so a ledger ``key=…`` still masks), so
    the word pass below never sees ``WANDB_API_KEY=<40 hex>`` whole and masks the name with it.
    """
    if _looks_secret(value):
        return MASK
    value = mask_named_values(value, bare=True)
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
    # Before the field-name rule, deliberately: `ObjectKey` is a statement about the *value*,
    # made in lab's own source, and `key=` is the natural (and pre-existing) field name for an
    # object key — a diagnostic that has to misname its subject to keep it is no better off.
    if isinstance(value, ObjectKey):
        return _vetted_object_key(value)
    if key is not None and _sensitive_key(key):
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
