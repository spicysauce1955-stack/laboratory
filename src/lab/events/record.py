"""The writer: one open/close pair per call, plus the ring buffer that explains failures.

Outcome, duration and ``error`` are derived here from how the block exits — a caller never sets
them, so a call cannot misreport its own success. ``ref()`` and ``result()`` are the only things
a caller adds.
"""

from __future__ import annotations

import os
import secrets
import uuid
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
from typing import Any

from lab import __version__
from lab._util import now
from lab.events import store
from lab.events.sanitize import mask_text, sanitize_params

RING = 200

#: Cheap, read-only commands whose *successes* are rate-limited on the write path (see
#: ``store.claim_read_slot``). Nothing here provisions, destroys or spends: a poll that says the
#: same thing the last poll said one second ago is not a record worth keeping, and 98,654 of them
#: in five days is how the 2026-09 campaign filled the byte cap and cost itself a day of
#: forensics. Failures are never limited, whatever the action.
#:
#: Two spellings of the same read are both real and both handled: the CLI records a group leaf as
#: ``"queue list"`` (``cli._group_action``) and the MCP tool of the same name as ``"queue_list"``.
#: ``read_only_key`` normalises them onto the one entry here, so the window is shared across
#: surfaces rather than each surface getting its own.
# The reads that get polled in loops. `history`/`report` are deliberately absent: they are the
# ledger's own forensic surface, and over the 5 days that motivated this limiter they were 33 and
# 5 calls against 98,654 `status` — no volume to win, and a burst of investigative reads should
# leave its own trail rather than silently record nothing.
_READ_ONLY_ACTIONS = frozenset({"status", "list", "logs", "metrics", "queue list", "queue show"})
_current: ContextVar["Call | None"] = ContextVar("lab_events_current", default=None)
_seq = 0
_session: str | None = None
_pruned = False


def _session_id() -> str:
    """``LAB_SESSION_ID`` when an agent harness sets it — exact grouping. Otherwise a
    per-process value; the session *view* does not depend on this being perfect."""
    global _session
    override = (os.environ.get("LAB_SESSION_ID") or "").strip()
    if override:
        return override
    if _session is None:
        _session = f"sess_{uuid.uuid4().hex[:8]}"
    return _session


def session_id() -> str:
    """Public accessor for this process's effective session id (see ``_session_id``).

    Lets a child process (e.g. the SkyPilot supervisor, spawned detached) inherit the exact
    session its submitter resolved — including a per-process *generated* id, which a plain
    environment inheritance would miss since it never existed as a real env var."""
    return _session_id()


def _new_id() -> str:
    """Time-ordered id: millisecond timestamp + randomness, sortable, needs no coordination."""
    return f"{int(now().timestamp() * 1000):013d}{secrets.token_hex(4)}"


def _project() -> dict[str, Any]:
    from lab.manifest import current_commit, is_dirty, repo_root

    try:
        root = repo_root()
        return {"name": root.name, "commit": current_commit(root), "dirty": is_dirty(root)}
    except Exception as e:  # noqa: BLE001 — not every cwd is a repo
        store.debug(f"project probe failed: {e}")
        return {}


def error_dict(exc: BaseException) -> dict[str, Any]:
    """Type, message, and the innermost frame inside the lab — the ``where`` that matters."""
    tb = exc.__traceback__
    where = None
    while tb is not None:
        where = f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}"
        tb = tb.tb_next
    return {"type": type(exc).__name__, "message": str(exc)[:2048], "where": where}


def _sanitize_error(error: dict[str, Any] | None) -> dict[str, Any] | None:
    """Mask secret-shaped text in a captured error's message before it reaches the ledger — the
    same FR-J1 rule ``begin()`` already applies to ``params`` via ``sanitize_params``. ``message``
    is free text (an exception's ``str()``, or a usage error's ``format_message()`` captured by
    the CLI's ``_capturing_usage_errors``), never argv, so ``mask_text`` — the prose-shaped
    masker ``lab.notes`` also uses — is the right tool here, not ``sanitize_argv``. Before this,
    ``error`` was the one field written to the ledger with no sanitization at all, so a
    secret-shaped value that failed a flag's own validation (e.g. ``--price-cap AKIA...``,
    rejected by click's float parser) sailed straight into ``error.message`` verbatim even
    though the sibling ``params.argv`` field masked it correctly. Never raises: a masking
    failure must not take the record with it."""
    if error is None:
        return None
    message = error.get("message")
    if not isinstance(message, str):
        return error
    try:
        masked = mask_text(message)
    except Exception:  # noqa: BLE001 — masking must never fail a command
        return error
    if masked == message:
        return error
    return {**error, "message": masked}


class Call:
    """A call in flight. Annotate it with :meth:`ref` and :meth:`result`; nothing else."""

    def __init__(self, id: str, started: datetime, seq: int) -> None:
        self.id = id
        self.started = started
        self.seq = seq
        self.notes: deque[dict[str, Any]] = deque(maxlen=RING)
        self._refs: dict[str, Any] = {}
        self._result: dict[str, Any] = {}
        # Set by `begin()` right after `_current.set(call)`; `finish()` resets the ContextVar to
        # this token rather than blindly clearing it, so a call that genuinely nests inside
        # another (the MCP middleware's `record()` nested inside a hypothetical outer one, or
        # two back-to-back `record()` blocks sharing a context) leaves the outer call current
        # again once the inner one closes, instead of wiping it to `None`.
        self._token: "Token[Call | None] | None" = None
        # A read-only call's `open` record, held here until `finish()` knows the outcome — the
        # rate-limit decision depends on it. `None` for every mutating action, whose `open` is
        # written at call start exactly as before.
        self._deferred_open: dict[str, Any] | None = None
        self._read_key: tuple[str, str] | None = None

    def ref(self, **ids: Any) -> None:
        self._refs.update({k: v for k, v in ids.items() if v is not None})

    def result(self, **digest: Any) -> None:
        self._result.update({k: v for k, v in digest.items() if v is not None})


def current() -> Call | None:
    return _current.get()


def read_only_key(action: str) -> str | None:
    """The rate-limit key for a cheap read-only action, or ``None`` if it isn't one.

    ``queue_list`` (MCP) and ``queue list`` (CLI) name the same read, so they normalise onto one
    key. No non-read-only action collides under that normalisation — the allowlist is a closed
    set (``sweep_status``, say, normalises to ``"sweep status"``, which is not in it).
    """
    normalized = action.replace("_", " ")
    return normalized if normalized in _READ_ONLY_ACTIONS else None


def begin(surface: str, action: str, params: Mapping[str, Any]) -> Call:
    """Open a call: write the open line immediately, so a call that never closes is visible.

    "Immediately" still holds for every action that *does* something. A cheap read-only action
    (``_READ_ONLY_ACTIONS``) buffers its open line in memory instead and writes it in
    ``finish()``, because whether it is written at all depends on the outcome, which does not
    exist yet. The trade is deliberate and one-sided: a SIGKILLed ``lab status`` now leaves no
    trace, while a SIGKILLed ``lab submit`` still leaves the dangling ``open`` that CLAUDE.md and
    ``compact()`` both treat as a finding in its own right.
    """
    global _seq, _pruned
    call = Call(_new_id(), now(), _seq)
    _seq += 1
    call._token = _current.set(call)
    if not _pruned:
        _pruned = True
        store.maybe_prune(now=now())
    project = _project()
    opened: dict[str, Any] = {
        "id": call.id,
        "ts": call.started.isoformat(),
        "phase": "open",
        "session": _session_id(),
        "seq": call.seq,
        "surface": surface,
        "action": action,
        "params": sanitize_params(params),
        "project": project,
        "lab_version": __version__,
    }
    key = read_only_key(action)
    if key is None:
        store.append(opened, when=call.started)
    else:
        call._deferred_open = opened
        name = project.get("name")
        call._read_key = (key, name if isinstance(name, str) else "-")
    return call


def _restore_context(call: Call) -> None:
    if call._token is not None:
        try:
            _current.reset(call._token)
        except (RuntimeError, ValueError) as e:  # noqa: BLE001 — never fail a command
            store.debug(f"context reset failed: {e}")
            _current.set(None)
    else:
        _current.set(None)


def _skip_as_a_repeat_poll(call: Call, ended: datetime) -> bool:
    """True when this successful read-only call is a repeat inside the window, so neither of its
    records is written. Only ever consulted for ``outcome == "ok"``: a failure is never
    rate-limited, and never stamps the window either — a failure that consumed the window would
    hide the very next success behind a window that success never opened."""
    if call._read_key is None:
        return False
    action, project = call._read_key
    try:
        claimed = store.claim_read_slot(
            action, project, now=ended, min_interval_s=store.read_min_interval_s()
        )
    except Exception as e:  # noqa: BLE001 — a broken limiter must degrade to recording
        store.debug(f"read-rate check failed for {action}: {e}")
        return False
    return not claimed


def finish(
    call: Call,
    *,
    outcome: str,
    exit_code: int | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    ended = now()
    deferred = call._deferred_open
    call._deferred_open = None
    if deferred is not None and outcome == "ok" and _skip_as_a_repeat_poll(call, ended):
        _restore_context(call)
        return
    record_: dict[str, Any] = {
        "id": call.id,
        "ts": ended.isoformat(),
        "phase": "close",
        "outcome": outcome,
        "exit_code": exit_code,
        "duration_ms": int((ended - call.started).total_seconds() * 1000),
        "refs": call._refs,
        "result": call._result,
        "error": _sanitize_error(error),
    }
    if outcome != "ok" and call.notes:
        record_["trace"] = list(call.notes)
    if deferred is not None:
        # Written under the *start* day, so a call straddling UTC midnight still lands its open
        # in day N and its close in day N+1 — the pairing `compact()` is careful to honour.
        store.append(deferred, when=call.started)
    store.append(record_, when=ended)
    _restore_context(call)


def finish_current(
    *, outcome: str, exit_code: int | None = None, error: dict[str, Any] | None = None
) -> None:
    """Close whatever call is open. The CLI opens in the group callback and closes in ``main``."""
    call = _current.get()
    if call is not None:
        finish(call, outcome=outcome, exit_code=exit_code, error=error)


@contextmanager
def record(
    surface: str,
    action: str,
    params: Mapping[str, Any],
    *,
    error_types: tuple[type[BaseException], ...] = (),
) -> Iterator[Call]:
    """Open a call, derive its outcome from how the block exits, close it. Re-raises unchanged.

    ``error_types`` lets a caller designate an exception type that represents a *handled*
    failure — e.g. the MCP middleware passing FastMCP's ``ToolError`` — so it's recorded as
    ``outcome="error"`` instead of ``"crash"``, mirroring the CLI's own distinction between a
    known ``_fail`` site and an unhandled traceback. This does not let a caller declare success
    or pick an outcome for an arbitrary exception: only a pre-named type is ever reclassified,
    everything else still derives to ``"crash"`` exactly as before — ``record()``'s
    derive-don't-declare property holds either way.
    """
    call = begin(surface, action, params)
    try:
        yield call
    except KeyboardInterrupt:
        finish(call, outcome="interrupted")
        raise
    except error_types as e:
        finish(call, outcome="error", error=error_dict(e))
        raise
    except BaseException as e:  # noqa: BLE001 — every exit path must be recorded
        finish(call, outcome="crash", error=error_dict(e))
        raise
    else:
        finish(call, outcome="ok")


def note(kind: str, **fields: Any) -> None:
    """Buffer an internal step. Discarded on success, flushed into ``trace`` on failure.

    Additive to the stderr diagnostic at the same site, never a replacement: the printed line is
    the live UX, this is the durable record.
    """
    call = _current.get()
    if call is None:
        return
    try:
        call.notes.append(
            {
                "t": int((now() - call.started).total_seconds() * 1000),
                "k": kind,
                "d": sanitize_params(fields),
            }
        )
    except Exception as e:  # noqa: BLE001
        store.debug(f"note failed: {e}")
