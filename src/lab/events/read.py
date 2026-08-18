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


def fold(records: Iterable[dict[str, Any]]) -> list[Event]:
    """Pair opens with closes. A close with no open is dropped (its open aged out); an open with
    no close becomes a ``running-or-died`` row, which is itself the finding."""
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
                seq=int(record.get("seq") or 0), surface=str(record.get("surface", "")),
                action=str(record.get("action", "")), params=dict(record.get("params") or {}),
                project=dict(record.get("project") or {}),
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
        event.refs = dict(close.get("refs") or {})
        event.result = dict(close.get("result") or {})
        event.error = close.get("error")
        event.trace = [
            Note(t=int(n.get("t") or 0), k=str(n.get("k", "")), d=dict(n.get("d") or {}))
            for n in (close.get("trace") or [])
        ]
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
    out: dict[str, Any] = {
        "id": event.id, "ts": event.ts.isoformat(), "action": event.action,
        "surface": event.surface, "status": event.status, "duration_ms": event.duration_ms,
        "project": event.project.get("name"), "refs": event.refs, "result": event.result,
        "error": (event.error or {}).get("message"),
    }
    if full:
        out |= {
            "params": event.params, "session": event.session, "exit_code": event.exit_code,
            "lab_version": event.lab_version, "error_detail": event.error,
            "trace": [vars(n) for n in event.trace],
        }
    return out
