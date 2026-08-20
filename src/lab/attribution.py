"""Machine-wide job attribution: *which project* owns a ``lab-*`` cluster.

``lab reconcile`` compares two sets that are not drawn at the same scope. The "does this cost
money?" side reads user-global, machine-wide state — ``~/.sky/state.db`` via ``sky.status``, the
Vast/DO/GCP APIs — while the "is it ours?" side reads ``JobStore(self.home)``, a single git
repo's ``runs/``. Since v0.5.0 the lab installs *into* other projects, so several projects on one
box share those global backends and each one can only see its own quarter of the picture. A
cluster launched from project B is simply absent from project A's ``runs/``, which made it look
exactly like a leak. On 2026-08-20 that mismatch destroyed seven running jobs belonging to
another project.

This module supplies the missing third scope: a machine-wide answer to "who owns job X?", drawn
from two sources in priority order.

1. **The job registry** (``~/.lab/jobs/index.jsonl``, override ``LAB_JOBS_INDEX_DIR``) — an
   append-only index written by ``JobStore.create`` via :func:`record_job`. Authoritative, and
   the only source that also knows the owner's ``runs_dir``, so a caller can go read the other
   project's manifest for itself.
2. **The event ledger** (``~/.lab/events/*.jsonl``) — already user-global since v0.6, and
   already records the project on every ``open`` line and the job id on the matching ``close``.
   It covers every job created before the registry existed, which on any real machine is all of
   them for the first 90 days.

Two rules govern everything here, both of them consequences of what a wrong answer costs.

**Never raise into a caller.** This feeds leak detection: a command that aborts because one JSONL
line is truncated leaves the boxes running and billing. Missing files, unreadable directories,
malformed lines and wrong-typed fields all degrade to "unknown" — the same advisory-never-fatal
posture as the placement zone memo.

**Never guess.** ``project is None`` means "not known to be owned by anyone", and the integration
reads that as *do not destroy*. A false unknown costs a warning line and a manual
``lab reconcile`` pass; a false attribution costs a running experiment. So an id claimed by two
different project names resolves to unknown rather than to whichever claim was seen first, and a
record whose project field is missing, blank or wrong-typed is never filled in from context.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lab._util import now
from lab.events import store

DEFAULT_DIR = "~/.lab/jobs"
INDEX_NAME = "index.jsonl"
SCHEMA = 1
DEFAULT_MAX_RECORDS = 20_000


@dataclass(frozen=True)
class Attribution:
    """What is known about one job id.

    ``project is None`` **iff** ``source == "unknown"``; both spellings of "we do not know" are
    kept in sync deliberately so a caller cannot pick the wrong one. Check :attr:`known`.

    ``runs_dir`` is a bonus, not a promise: only the registry records one, so a ledger-sourced
    attribution normally carries ``None``. When it is present it is the absolute path of the
    owning project's job store, which is what lets a caller read that project's manifest and ask
    whether the job is still non-terminal.
    """

    job_id: str
    project: str | None
    runs_dir: Path | None
    source: str  # "registry" | "ledger" | "unknown"

    @property
    def known(self) -> bool:
        """True only when some source positively names an owning project."""
        return self.project is not None


def _debug(message: str) -> None:
    """Same switch as the ledger's (``LAB_EVENTS_DEBUG=1``), different prefix.

    Everything in this module swallows its errors, so without a way to surface them a genuine
    schema or permissions bug would hide forever behind a stream of benign "unknown"s.
    """
    if (os.environ.get("LAB_EVENTS_DEBUG") or "").strip() == "1":
        print(f"[lab.attribution] {message}", file=sys.stderr)


def index_dir() -> Path:
    """Directory holding the registry. Mirrors ``store.events_dir()``: env override wins, blank
    means unset, ``~`` expanded."""
    override = (os.environ.get("LAB_JOBS_INDEX_DIR") or "").strip()
    return Path(override).expanduser() if override else Path(DEFAULT_DIR).expanduser()


def index_path() -> Path:
    return index_dir() / INDEX_NAME


def _max_records() -> int:
    try:
        value = int((os.environ.get("LAB_JOBS_INDEX_MAX_RECORDS") or "").strip()
                    or DEFAULT_MAX_RECORDS)
    except ValueError:
        return DEFAULT_MAX_RECORDS
    return value if value > 0 else DEFAULT_MAX_RECORDS


def local_project() -> str | None:
    """This checkout's project name, derived exactly as the event ledger derives it.

    Offered so the registry and the ledger agree on what a project is called — if the call site
    invented its own name (say, ``home.parent.name``), a job recorded in the registry and the
    same job seen through the ledger could disagree, which this module would then report as a
    conflict and refuse to attribute. Returns ``None`` when the cwd is not a repo.
    """
    try:
        from lab.manifest import repo_root

        name = repo_root().name.strip()
        return name or None
    except Exception as e:  # noqa: BLE001 — not every cwd is a repo
        _debug(f"project probe failed: {e}")
        return None


# ------------------------------------------------------------------ writing


def _record(job_id: str, project: str | None, runs_dir: Path, created_at: datetime) -> dict[str, Any]:
    return {
        "v": SCHEMA,
        "job_id": job_id,
        "project": project,
        "runs_dir": str(runs_dir),
        "created_at": created_at.isoformat(),
    }


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Idempotence key: everything except the timestamp, which a re-record legitimately moves."""
    return all(a.get(k) == b.get(k) for k in ("job_id", "project", "runs_dir"))


def record_job(
    job_id: str,
    *,
    project: str | None,
    runs_dir: Path,
    created_at: datetime | None = None,
) -> None:
    """Index one job as belonging to ``project`` at ``runs_dir``. Best-effort; never raises.

    Idempotent: re-recording a job whose newest entry already says the same thing writes nothing,
    so a retried ``JobStore.create`` cannot inflate the file. Recording it with a *different*
    project or runs_dir appends a new line, and the read side takes the newest — that is a job
    that genuinely moved, not a conflict to arbitrate.

    ``project=None`` is accepted and recorded as such. That still buys something: the id becomes
    :func:`known_job_ids`-visible and its ``runs_dir`` is available, while ownership stays
    honestly unproven rather than being back-filled from whichever process happened to write it.

    Writes go through the same per-file ``flock`` the ledger uses (``store.lock_path``), because
    the concurrency that matters here is identical: a sharded sweep creates many manifests from
    many processes at once.
    """
    project = project.strip() if isinstance(project, str) else None
    try:
        path = index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = _record(job_id, project or None, Path(runs_dir).expanduser().resolve(),
                         created_at or now())
        lock = store.lock_path(path)
        with lock.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                _locked_append(path, record)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:  # noqa: BLE001 — indexing must never fail a submit
        _debug(f"record_job({job_id}) failed: {e}")


def _locked_append(path: Path, record: dict[str, Any]) -> None:
    """Append ``record`` unless it is already the newest entry for its id. Caller holds the lock.

    Reads the file as raw bytes first and only parses it when it has to: the overwhelmingly
    common case is a brand-new job id, whose absence a substring scan settles without decoding a
    single JSON object. That keeps the cost of indexing a job proportional to file *size*, not to
    the number of records in it, on the path that runs once per ``submit``.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    line = json.dumps(record, separators=(",", ":")) + "\n"
    needle = json.dumps({"job_id": record["job_id"]}, separators=(",", ":"))[1:-1]
    over_cap = text.count("\n") >= _max_records()
    if needle not in text and not over_cap:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        return

    records = [r for r in store.iter_records([path]) if isinstance(r.get("job_id"), str)]
    newest = next((r for r in reversed(records) if r["job_id"] == record["job_id"]), None)
    if newest is not None and _same(newest, record):
        return  # already indexed exactly this way
    if over_cap:
        # Forgetting the oldest jobs is the safe direction to fail: an id we no longer index
        # reads as "unknown", and unknown means "do not destroy". Growing without bound in a
        # file every reconcile reads is not.
        keep = records[-(_max_records() - 1):] if _max_records() > 1 else []
        _rewrite(path, keep + [record])
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def _rewrite(path: Path, records: list[dict[str, Any]]) -> None:
    """Replace the index atomically. Caller holds the lock, so a concurrent append cannot be
    lost into the orphaned inode (the failure ``lab.events.store`` documents for ``compact``)."""
    tmp = path.with_suffix(".jsonl.tmp")
    text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ reading


def _name_of(value: object) -> str | None:
    """A project name, or ``None`` for anything that is not a non-blank string."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _registry_index() -> dict[str, dict[str, Any]]:
    """Newest record per job id. Corrupt lines are skipped by ``store.iter_records``; a missing
    or unreadable index simply yields nothing."""
    index: dict[str, dict[str, Any]] = {}
    for record in store.iter_records([index_path()]):
        job_id = record.get("job_id")
        if isinstance(job_id, str) and job_id:
            index[job_id] = record
    return index


def _runs_dir_of(record: dict[str, Any]) -> Path | None:
    value = record.get("runs_dir")
    return Path(value) if isinstance(value, str) and value else None


def _job_ids_in(refs: object, wanted: set[str]) -> set[str]:
    """The wanted ids this ``close`` record's ``refs`` names — ``job_id`` plus the ``job_ids``
    list a sweep writes. Deliberately narrow: ``lab.events.annotate.refs_from`` already refuses
    to harvest job ids out of nested payloads (``lab list``'s ``jobs``), and widening it here
    would resurrect exactly that false-positive."""
    if not isinstance(refs, dict):
        return set()
    found: set[str] = set()
    single = refs.get("job_id")
    if isinstance(single, str) and single in wanted:
        found.add(single)
    many = refs.get("job_ids")
    if isinstance(many, list):
        found |= {v for v in many if isinstance(v, str) and v in wanted}
    return found


def _ledger_claims(wanted: set[str]) -> dict[str, set[str | None]]:
    """Every project name the ledger associates with each wanted id, newest files first.

    The project lives on the ``open`` line and the job id on its ``close`` line, so a claim needs
    both halves of a pair. Two orderings have to work:

    * within one day file the open precedes its close, so each file is indexed both ways before
      being resolved;
    * a pair may *straddle* midnight — any supervisor run of more than a few hours, which is most
      of the expensive ones — and since we walk newest-first the close is seen a whole file
      before its open. Unmatched closes are therefore carried backwards in ``pending``. The
      reverse never happens: a close is never older than its open.

    Stops at the first file by which every wanted id has a real (non-``None``) claim. That bounds
    the common case to one file instead of the ~90 the retention window allows. The trade-off is
    that a conflicting claim living only in an older file goes unseen; conflicts are a
    defence-in-depth guard against an id namespace collision, not an expected condition (job ids
    are millisecond-stamped plus random), so paying a full 90-day scan on every reconcile to
    close that gap is the wrong trade.
    """
    claims: dict[str, set[str | None]] = {}
    pending: dict[str, set[str]] = {}  # event id -> wanted job ids still awaiting their open

    def claim(job_ids: Iterable[str], name: str | None) -> None:
        for job_id in job_ids:
            claims.setdefault(job_id, set()).add(name)

    for path in sorted(store.day_files(), reverse=True):
        opens: dict[str, str | None] = {}
        closes: dict[str, set[str]] = {}
        for record in store.iter_records([path]):
            event_id = record.get("id")
            if not isinstance(event_id, str) or not event_id:
                continue
            phase = record.get("phase")
            if phase == "open":
                project = record.get("project")
                opens[event_id] = _name_of(project.get("name")) if isinstance(project, dict) \
                    else None
            elif phase == "close":
                hit = _job_ids_in(record.get("refs"), wanted)
                if hit:
                    closes[event_id] = closes.get(event_id, set()) | hit
        for event_id, job_ids in closes.items():
            if event_id in opens:
                claim(job_ids, opens[event_id])
            else:
                pending[event_id] = pending.get(event_id, set()) | job_ids
        for event_id, name in opens.items():
            carried = pending.pop(event_id, None)
            if carried:
                claim(carried, name)
        if wanted <= {job_id for job_id, names in claims.items() if names - {None}}:
            break
    return claims


def attribute_jobs(job_ids: Iterable[str]) -> dict[str, Attribution]:
    """Attribute every id in ``job_ids``. Never raises; never guesses.

    The result has an entry for **every** requested id, so callers can index it directly; ids
    nothing knows about come back as ``Attribution(job_id, None, None, "unknown")``.

    Registry beats ledger. A registry record that names no project does not end the search — the
    ledger is still consulted — but its ``runs_dir`` is kept and attached to whatever the ledger
    concludes, since the path is useful even when ownership is not proven.
    """
    wanted = {j for j in job_ids if isinstance(j, str) and j}
    if not wanted:
        return {}

    out: dict[str, Attribution] = {}
    runs_dirs: dict[str, Path] = {}
    try:
        registry = _registry_index()
    except Exception as e:  # noqa: BLE001 — a leak check must not die on a bad index
        _debug(f"registry read failed: {e}")
        registry = {}
    for job_id in wanted:
        record = registry.get(job_id)
        if record is None:
            continue
        runs_dir = _runs_dir_of(record)
        if runs_dir is not None:
            runs_dirs[job_id] = runs_dir
        name = _name_of(record.get("project"))
        if name is not None:
            out[job_id] = Attribution(job_id, name, runs_dir, "registry")

    remaining = wanted - set(out)
    if remaining:
        try:
            claims = _ledger_claims(remaining)
        except Exception as e:  # noqa: BLE001
            _debug(f"ledger scan failed: {e}")
            claims = {}
        for job_id in remaining:
            names = {n for n in claims.get(job_id, set()) if n is not None}
            if len(names) == 1:
                out[job_id] = Attribution(job_id, names.pop(), runs_dirs.get(job_id), "ledger")
            elif len(names) > 1:
                _debug(f"{job_id}: claimed by {sorted(names)}; refusing to attribute")

    for job_id in wanted:
        out.setdefault(job_id, Attribution(job_id, None, runs_dirs.get(job_id), "unknown"))
    return out


def known_job_ids() -> set[str]:
    """Every job id either source has *heard of*, whether or not it can be attributed.

    A different question from :func:`attribute_jobs`: an id here with no owner is one this
    machine really did create at some point, which is worth distinguishing from an id no lab on
    this box has ever seen. Unlike ``attribute_jobs`` this cannot stop early — "all of them" has
    no early exit — so it reads the whole retention window. Call it for a report, not per-run.
    """
    found: set[str] = set()
    try:
        found |= set(_registry_index())
    except Exception as e:  # noqa: BLE001
        _debug(f"registry read failed: {e}")
    try:
        for record in store.iter_records(store.day_files()):
            if record.get("phase") != "close":
                continue
            refs = record.get("refs")
            if not isinstance(refs, dict):
                continue
            single = refs.get("job_id")
            if isinstance(single, str) and single:
                found.add(single)
            many = refs.get("job_ids")
            if isinstance(many, list):
                found |= {v for v in many if isinstance(v, str) and v}
    except Exception as e:  # noqa: BLE001
        _debug(f"ledger scan failed: {e}")
    return found
