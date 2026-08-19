"""Fold the ledger's open/close lines into rows, and filter them. Pure functions over JSONL."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any, Protocol

from lab._util import now as _now
from lab._util import parse_duration
from lab.events import store
from lab.events.models import Event, Note


class _ManifestStore(Protocol):
    """Structural shape ``crossref`` needs — matches ``lab.store.JobStore`` without ``lab.events``
    importing it. Cross-referencing depends on an *injected* store rather than a hard import so
    ``lab.events`` stays free of a ``lab.core``/``lab.store`` dependency (module boundary,
    spec §"Module boundaries"); only the CLI and the MCP server, which already depend on
    ``JobStore``, ever pass one in."""

    def read_manifest(self, job_id: str) -> Any: ...
    def logs_path(self, job_id: str) -> Any: ...


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
        try:
            return int(value)
        except (ValueError, OverflowError):
            # json.loads accepts NaN/Infinity/-Infinity by default, so a non-finite float is a
            # genuinely reachable field value here, not a hypothetical.
            pass
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


def crossref(event: Event, job_store: _ManifestStore) -> dict[str, Any]:
    """Best-effort cross-reference to the job's manifest and ``logs.txt`` path, for the forensic
    view (``lab history --full``, MCP ``history(full=True)``): "Forensic adds the full ``trace``
    and cross-references the manifest and the ``logs.txt`` path for each ``refs.job_id``, so the
    ledger is a jumping-off point rather than a silo."

    A missing or unreadable manifest — the job never got that far, or its ``runs/`` dir has
    since been cleaned up — degrades to an empty dict rather than raising, the same posture as
    everything else here.
    """
    job_id = event.refs.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return {}
    try:
        manifest = job_store.read_manifest(job_id)
        logs_path = job_store.logs_path(job_id)
    except Exception as e:  # noqa: BLE001 — a missing/corrupt manifest must degrade, not raise
        store.debug(f"crossref failed for {job_id}: {e}")
        return {}
    status = getattr(manifest, "status", None)
    return {
        "manifest_state": getattr(status, "value", status),
        "manifest_end_reason": getattr(manifest, "end_reason", None),
        "logs_path": str(logs_path),
    }


def row(event: Event, *, full: bool = False, job_store: _ManifestStore | None = None) -> dict[str, Any]:
    """The display shape. Lives here so the CLI and the MCP server emit identical rows without
    either shell importing the other. ``job_store``, when supplied, adds the forensic
    cross-reference (``crossref``) — both shells pass the same store they already build for
    other commands, so `--full`/`full=True` looks identical from either surface."""
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
        if job_store is not None:
            out |= crossref(event, job_store)
    return out
