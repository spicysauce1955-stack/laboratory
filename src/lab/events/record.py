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
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from lab import __version__
from lab._util import now
from lab.events import store
from lab.events.sanitize import sanitize_params

RING = 200
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


class Call:
    """A call in flight. Annotate it with :meth:`ref` and :meth:`result`; nothing else."""

    def __init__(self, id: str, started: datetime, seq: int) -> None:
        self.id = id
        self.started = started
        self.seq = seq
        self.notes: deque[dict[str, Any]] = deque(maxlen=RING)
        self._refs: dict[str, Any] = {}
        self._result: dict[str, Any] = {}

    def ref(self, **ids: Any) -> None:
        self._refs.update({k: v for k, v in ids.items() if v is not None})

    def result(self, **digest: Any) -> None:
        self._result.update({k: v for k, v in digest.items() if v is not None})


def current() -> Call | None:
    return _current.get()


def begin(surface: str, action: str, params: Mapping[str, Any]) -> Call:
    """Open a call: write the open line immediately, so a call that never closes is visible."""
    global _seq, _pruned
    call = Call(_new_id(), now(), _seq)
    _seq += 1
    _current.set(call)
    if not _pruned:
        _pruned = True
        store.maybe_prune(now=now())
    store.append(
        {
            "id": call.id,
            "ts": call.started.isoformat(),
            "phase": "open",
            "session": _session_id(),
            "seq": call.seq,
            "surface": surface,
            "action": action,
            "params": sanitize_params(params),
            "project": _project(),
            "lab_version": __version__,
        },
        when=call.started,
    )
    return call


def finish(
    call: Call,
    *,
    outcome: str,
    exit_code: int | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    ended = now()
    record_: dict[str, Any] = {
        "id": call.id,
        "ts": ended.isoformat(),
        "phase": "close",
        "outcome": outcome,
        "exit_code": exit_code,
        "duration_ms": int((ended - call.started).total_seconds() * 1000),
        "refs": call._refs,
        "result": call._result,
        "error": error,
    }
    if outcome != "ok" and call.notes:
        record_["trace"] = list(call.notes)
    store.append(record_, when=ended)
    _current.set(None)


def finish_current(
    *, outcome: str, exit_code: int | None = None, error: dict[str, Any] | None = None
) -> None:
    """Close whatever call is open. The CLI opens in the group callback and closes in ``main``."""
    call = _current.get()
    if call is not None:
        finish(call, outcome=outcome, exit_code=exit_code, error=error)


@contextmanager
def record(surface: str, action: str, params: Mapping[str, Any]) -> Iterator[Call]:
    """Open a call, derive its outcome from how the block exits, close it. Re-raises unchanged."""
    call = begin(surface, action, params)
    try:
        yield call
    except KeyboardInterrupt:
        finish(call, outcome="interrupted")
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
