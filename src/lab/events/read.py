"""Fold the ledger's open/close lines into rows, and filter them. Pure functions over JSONL."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from lab._util import now as _now
from lab._util import parse_duration
from lab.events import store
from lab.events.models import Event, Note


def _dt(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _as_int(value: object, *, default: int, field: str, id_: str) -> int:
    """Coerce a field that should be an int. A parseable-but-wrong-shaped record must degrade,
    not take the whole read down with it — ``store.debug`` surfaces the drop under
    ``LAB_EVENTS_DEBUG=1`` so a genuine schema bug doesn't hide silently forever."""
    if value is None:
        return default
    if isinstance(value, bool):  # bool is an int subclass; keep it explicit rather than magic
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    store.debug(f"{id_}: {field}={value!r} is not an int; defaulted to {default}")
    return default


def _as_dict(value: object, *, field: str, id_: str) -> dict[str, Any]:
    """Coerce a field that should be a mapping. Same tolerance rule as ``_as_int``."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    store.debug(f"{id_}: {field}={value!r} is not a mapping; dropped")
    return {}


def _as_error(value: object, *, id_: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    store.debug(f"{id_}: error={value!r} is not a mapping; dropped")
    return None


def _as_trace(value: object, *, id_: str) -> list[Note]:
    """Coerce ``trace``. If it isn't a list at all, the whole thing is dropped; if it is a list,
    only the individual entries that aren't mapping-shaped are skipped — one bad entry costs only
    itself, not the rest of the trace."""
    if value is None:
        return []
    if not isinstance(value, list):
        store.debug(f"{id_}: trace={value!r} is not a list; dropped")
        return []
    notes: list[Note] = []
    for entry in value:
        if not isinstance(entry, dict):
            store.debug(f"{id_}: trace entry {entry!r} is not a mapping; skipped")
            continue
        notes.append(Note(
            t=_as_int(entry.get("t"), default=0, field="trace.t", id_=id_),
            k=str(entry.get("k", "")),
            d=_as_dict(entry.get("d"), field="trace.d", id_=id_),
        ))
    return notes


def fold(records: Iterable[dict[str, Any]]) -> list[Event]:
    """Pair opens with closes. A close with no open is dropped (its open aged out); an open with
    no close becomes a ``running-or-died`` row, which is itself the finding.

    ``store.iter_records`` only guarantees each yielded record is parseable JSON shaped as a
    dict — a field inside it can still be the wrong type (e.g. ``trace`` written as a string).
    Every field read here is coerced per-field rather than trusted, so one malformed record
    degrades instead of raising and taking the rest of the batch down with it.
    """
    events: dict[str, Event] = {}
    closes: dict[str, dict[str, Any]] = {}
    for record in records:
        id_ = record.get("id")
        if not isinstance(id_, str):
            continue
        if record.get("phase") == "open":
            ts = _dt(record.get("ts"))
            if ts is None:
                continue
            events[id_] = Event(
                id=id_, ts=ts, session=str(record.get("session", "")),
                seq=_as_int(record.get("seq"), default=0, field="seq", id_=id_),
                surface=str(record.get("surface", "")),
                action=str(record.get("action", "")),
                params=_as_dict(record.get("params"), field="params", id_=id_),
                project=_as_dict(record.get("project"), field="project", id_=id_),
                lab_version=str(record.get("lab_version", "")),
            )
        elif record.get("phase") == "close":
            closes[id_] = record
    for id_, close in closes.items():
        event = events.get(id_)
        if event is None:
            continue
        event.outcome = close.get("outcome")
        event.exit_code = close.get("exit_code")
        event.duration_ms = close.get("duration_ms")
        event.refs = _as_dict(close.get("refs"), field="refs", id_=id_)
        event.result = _as_dict(close.get("result"), field="result", id_=id_)
        event.error = _as_error(close.get("error"), id_=id_)
        event.trace = _as_trace(close.get("trace"), id_=id_)
    return sorted(events.values(), key=lambda e: e.ts, reverse=True)


def read(
    *,
    since: str | None = None,
    project: str | None = None,
    action: str | None = None,
    session: str | None = None,
    job: str | None = None,
    failures_only: bool = False,
    limit: int | None = None,
    now_: datetime | None = None,
) -> list[Event]:
    """Folded rows, newest first. ``since`` takes a duration string (``2d``, ``30m``)."""
    events = fold(store.iter_records(store.day_files()))
    if since:
        seconds = parse_duration(since)
        if seconds is not None:
            cutoff = (now_ or _now()) - timedelta(seconds=seconds)
            events = [e for e in events if e.ts >= cutoff]
    if project:
        events = [e for e in events if e.project.get("name") == project]
    if action:
        events = [e for e in events if e.action == action]
    if session:
        events = [e for e in events if e.session == session]
    if job:
        events = [e for e in events
                  if job == e.refs.get("job_id") or job in (e.refs.get("job_ids") or [])]
    if failures_only:
        events = [e for e in events if e.failed]
    return events[:limit] if limit else events


def row(event: Event, *, full: bool = False) -> dict[str, Any]:
    """The display shape. Lives here so the CLI and the MCP server emit identical rows without
    either shell importing the other."""
    error = event.error if isinstance(event.error, dict) else {}
    out: dict[str, Any] = {
        "id": event.id, "ts": event.ts.isoformat(), "action": event.action,
        "surface": event.surface, "status": event.status, "duration_ms": event.duration_ms,
        "project": event.project.get("name"), "refs": event.refs, "result": event.result,
        "error": error.get("message"),
    }
    if full:
        out |= {
            "params": event.params, "session": event.session, "exit_code": event.exit_code,
            "lab_version": event.lab_version, "error_detail": event.error,
            "trace": [vars(n) for n in event.trace],
        }
    return out
