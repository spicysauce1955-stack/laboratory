"""What the person running the job wrote down, filed where the machine's own record already is.

The lab records what it did; this records what its user made of it. The 2026-08-26 ledger review
found the gap this closes: three of the most expensive findings in the consuming project's
history were written down as prose in *its* repo — a misleading error message, a price cap that
did not hold, an aggregator that crashed — and none of them ever reached the tool. The ledger
cannot hold them because it only knows what it called and what came back, never what a person
concluded.

Two readers, and they need different things from the same note:

*The maintainer*, retrospectively and across projects. Served by the user-global index — which is
why that index exists rather than a per-project file. The consuming project runs ``lab`` from one
checkout and does its thinking in another; a project-local store would be invisible from the repo
where the conclusion gets written.

*The next run*, at the moment it is about to repeat the mistake. That reader will never go
looking, so the note has to find them, which means it cannot be keyed by the job id it was
attached to — the next run has a different one. It is keyed by what *recurs*: the error signature
(highest precision, and :func:`lab.events.stats.signature` already computes it), or the
entrypoint the job ran. Never by cloud alone: a match that fires on every DO submit is noise, and
noise is how an alarm earns being ignored (R10).

Every note carries the ``lab_version`` it was written at, because the failure this feature must
not reproduce is its own worst case. The consuming project still runs a hand-written watchdog
against a cap that has been enforced on the box since v0.1.0, and still records a capability as
impossible eight days after it shipped. A channel that distributes advice without dating it
manufactures exactly that, at scale — so notes can also be *retired*, and a retired note is
history rather than guidance.

Best-effort throughout, like the ledger: a note that cannot be filed must never fail the command
that was trying to file it.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DIR = "~/.lab/notes"
SCHEMA = 1

#: Kinds lifted from the vocabulary the consuming project already writes by hand, rather than
#: invented here — its logs carry GOTCHA, BUDGET EVENT, ROOT CAUSE, LESSON and the rest already,
#: so adopting them costs nobody a new habit. Free text is always allowed; this is a hint, not a
#: schema, and an unrecognised kind is passed through untouched.
KINDS = (
    "NOTE",
    "GOTCHA",
    "BUDGET EVENT",
    "ROOT CAUSE",
    "INCIDENT",
    "LESSON",
    "DEVIATION",
    "FEATURE REQUEST",
)

#: A facet match must pin the *work*, not just where it ran. ``entrypoint`` is the discriminating
#: one; cloud and accelerators only ever narrow it further.
_REQUIRED_FACET = "entrypoint"


@dataclass(frozen=True)
class Note:
    """One thing a person concluded, and enough context for it to find its next reader."""

    id: str
    ts: str
    text: str
    kind: str = "NOTE"
    job_id: str | None = None
    sweep_id: str | None = None
    project: str | None = None
    session: str | None = None
    author: str = "human"
    lab_version: str | None = None
    usd: float | None = None
    signature: str | None = None
    facets: dict[str, Any] = field(default_factory=dict)
    retired: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"v": SCHEMA}
        record.update(
            {
                "id": self.id,
                "ts": self.ts,
                "text": self.text,
                "kind": self.kind,
                "job_id": self.job_id,
                "sweep_id": self.sweep_id,
                "project": self.project,
                "session": self.session,
                "author": self.author,
                "lab_version": self.lab_version,
                "usd": self.usd,
                "signature": self.signature,
                "facets": self.facets,
                "retired": self.retired,
            }
        )
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Note | None:
        """Rebuild a note, or ``None`` if the line is not one. A store made unreadable by a
        single bad line would fail at its only job."""
        rid, text = record.get("id"), record.get("text")
        if not isinstance(rid, str) or not isinstance(text, str):
            return None
        facets = record.get("facets")
        retired = record.get("retired")
        return cls(
            id=rid,
            ts=str(record.get("ts") or ""),
            text=text,
            kind=str(record.get("kind") or "NOTE"),
            job_id=_opt_str(record.get("job_id")),
            sweep_id=_opt_str(record.get("sweep_id")),
            project=_opt_str(record.get("project")),
            session=_opt_str(record.get("session")),
            author=str(record.get("author") or "human"),
            lab_version=_opt_str(record.get("lab_version")),
            usd=_opt_float(record.get("usd")),
            signature=_opt_str(record.get("signature")),
            facets=facets if isinstance(facets, dict) else {},
            retired=retired if isinstance(retired, dict) else None,
        )


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _opt_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def debug(message: str) -> None:
    if (os.environ.get("LAB_NOTES_DEBUG") or "").strip() == "1":
        import sys

        print(f"[lab.notes] {message}", file=sys.stderr)


def enabled() -> bool:
    return (os.environ.get("LAB_NOTES") or "").strip() != "0"


def notes_dir() -> Path:
    override = (os.environ.get("LAB_NOTES_DIR") or "").strip()
    return Path(override).expanduser() if override else Path(DEFAULT_DIR).expanduser()


def index_path() -> Path:
    return notes_dir() / "index.jsonl"


def current_version() -> str | None:
    try:
        from lab import __version__

        return str(__version__)
    except Exception as e:  # noqa: BLE001 — a version probe must never fail a note
        debug(f"version probe failed: {e}")
        return None


def _new_id() -> str:
    """Short, sortable-ish, and unique enough to name on a command line."""
    return f"n-{int(time.time() * 1000):x}-{uuid.uuid4().hex[:4]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(text: str) -> str:
    """Mask secrets in free text before it is durable anywhere (FR-J1).

    Reuses the ledger's deny-list so a note and an argv are masked by one set of rules. Free text
    is a far larger surface than a command line — a pasted traceback is the expected case — so
    this runs on the way in, not on the way out.
    """
    try:
        from lab.events.sanitize import sanitize_argv

        return sanitize_argv([text])[0]
    except Exception as e:  # noqa: BLE001 — never fail a note over masking
        debug(f"masking failed: {e}")
        return text


def _project_now() -> str | None:
    try:
        from lab.attribution import local_project

        return local_project()
    except Exception as e:  # noqa: BLE001 — not every cwd is a repo
        debug(f"project probe failed: {e}")
        return None


def _session_now() -> str | None:
    return (os.environ.get("LAB_SESSION_ID") or "").strip() or None


def local_project_name() -> str | None:
    """This checkout's project name — offered so the CLI does not import attribution itself."""
    return _project_now()


def facets_of(manifest: Any) -> dict[str, Any]:
    """The features of a run that a *later* run can be recognised by.

    Deliberately not the job id, the seed or the config: those identify one run, and a note keyed
    to them could never fire again. What recurs is the script, the cloud it ran on and the
    accelerator it asked for — the three things that made the RTX_4090 naming gotcha a gotcha
    twice. Best-effort: a manifest shape this cannot read yields no facets, never an exception.
    """
    out: dict[str, Any] = {}
    try:
        command = str(getattr(getattr(manifest, "run", None), "entrypoint_command", "") or "")
        for token in command.split():
            if token.endswith(".py"):
                out["entrypoint"] = Path(token).name
                break
        resources = getattr(manifest, "resources", None)
        if resources is not None:
            out["cloud"] = getattr(resources, "cloud", None) or None
            out["accelerators"] = getattr(resources, "accelerators", None) or None
        backend = getattr(getattr(manifest, "backend", None), "provisioner", None)
        if backend:
            out["backend"] = str(backend)
    except Exception as e:  # noqa: BLE001 — facets are a convenience, never a failure
        debug(f"facet probe failed: {e}")
    return {k: v for k, v in out.items() if v is not None}


def job_notes_path(job_id: str, *, home: Path | str) -> Path:
    return Path(home) / job_id / "notes.jsonl"


def _append(path: Path, record: dict[str, Any]) -> None:
    """One locked append. Same discipline as the ledger: a sharded sweep runs many processes
    against one index, and a torn line would break the store exactly when it matters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
    lock = Path(str(path) + ".lock")
    with lock.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as out:
                out.write(line)
                out.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read(path: Path) -> list[Note]:
    out: list[Note] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and (note := Note.from_record(record)) is not None:
            out.append(note)
    return out


def write(
    *,
    text: str,
    job_id: str | None = None,
    sweep_id: str | None = None,
    kind: str = "NOTE",
    author: str = "human",
    usd: float | None = None,
    signature: str | None = None,
    facets: dict[str, Any] | None = None,
    project: str | None = None,
    home: Path | str | None = None,
) -> Note | None:
    """File one note. Returns it, or ``None`` if nothing could be written.

    Two destinations, on purpose. The per-job file sits beside ``logs.txt`` and travels with the
    run into an export bundle, so the note reaches whoever reads the result later. The global
    index is what makes the note findable at all — by another project, and by a later run that
    knows nothing about this job id.
    """
    note = Note(
        id=_new_id(),
        ts=_now(),
        text=_clean(text),
        kind=kind or "NOTE",
        job_id=job_id,
        sweep_id=sweep_id,
        project=project if project is not None else _project_now(),
        session=_session_now(),
        author=author,
        lab_version=current_version(),
        usd=usd,
        signature=signature,
        facets=dict(facets or {}),
    )
    if not enabled():
        return None
    record = note.to_record()
    wrote = False
    if job_id is not None and home is not None:
        try:
            _append(job_notes_path(job_id, home=home), record)
            wrote = True
        except Exception as e:  # noqa: BLE001 — a note must never fail its command
            debug(f"job-dir append failed: {e}")
    try:
        _append(index_path(), record)
        wrote = True
    except Exception as e:  # noqa: BLE001
        debug(f"index append failed: {e}")
    return note if wrote else None


def for_job(job_id: str, *, home: Path | str) -> list[Note]:
    """This job's own notes, oldest first — the per-job file when it exists, else the index."""
    local = _read(job_notes_path(job_id, home=home))
    if local:
        return local
    return [n for n in _read(index_path()) if n.job_id == job_id]


def search(
    *,
    project: str | None = None,
    job_id: str | None = None,
    kind: str | None = None,
    include_retired: bool = False,
    limit: int | None = None,
) -> list[Note]:
    """Notes from the global index, oldest first, newest truncated last."""
    found = _read(index_path())
    found = _resolve_retirements(found)
    if not include_retired:
        found = [n for n in found if n.retired is None]
    if project is not None:
        found = [n for n in found if n.project == project]
    if job_id is not None:
        found = [n for n in found if n.job_id == job_id]
    if kind is not None:
        found = [n for n in found if n.kind == kind]
    return found[-limit:] if limit is not None else found


def match(
    *,
    signature: str | None = None,
    facets: dict[str, Any] | None = None,
    limit: int = 3,
) -> list[Note]:
    """Notes worth showing to a run that is about to repeat something, newest first.

    Precision is the whole design. A signature match means *this exact failure* has been seen
    before and is the only match worth firing unprompted. A facet match must pin the entrypoint —
    matching on cloud alone would fire on every submit, and a push that fires every time is one
    nobody reads. Retired notes never appear here; they are history, not guidance.
    """
    live = [n for n in search() if n.retired is None]
    hits: list[Note] = []
    if signature:
        hits = [n for n in live if n.signature and n.signature == signature]
    elif facets:
        wanted = {k: v for k, v in facets.items() if v is not None}
        if _REQUIRED_FACET not in wanted:
            return []
        hits = [n for n in live if _facets_agree(n.facets, wanted)]
    return list(reversed(hits))[:limit]


def _facets_agree(have: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """Every facet the caller pinned must be present and equal on the note."""
    if not have:
        return False
    return all(have.get(key) == value for key, value in wanted.items())


def _resolve_retirements(found: list[Note]) -> list[Note]:
    """Fold retirement records onto the notes they retire.

    Retirement is an append, never an edit: the index stays append-only, so a note and its
    retirement are two lines and the later one wins. That keeps concurrent writers honest and
    leaves the original wording legible.
    """
    retirements = {n.id: n.retired for n in found if n.retired is not None}
    if not retirements:
        return found
    out: list[Note] = []
    seen: set[str] = set()
    for note in found:
        if note.retired is not None:
            continue  # a retirement line is not itself a note
        if note.id in seen:
            continue
        seen.add(note.id)
        retired = retirements.get(note.id)
        out.append(note if retired is None else _replace(note, retired=retired))
    return out


def _replace(note: Note, **changes: Any) -> Note:
    from dataclasses import replace

    return replace(note, **changes)


def retire(note_id: str, *, reason: str) -> Note:
    """Mark a note as no longer current. Raises ``KeyError`` if it is not there.

    The one operation without which this feature becomes the problem it exists to solve: a
    channel that never retires anything is a machine for distributing folklore.
    """
    existing = {n.id: n for n in _resolve_retirements(_read(index_path()))}
    note = existing.get(note_id)
    if note is None:
        raise KeyError(note_id)
    retired = {"ts": _now(), "reason": _clean(reason), "lab_version": current_version()}
    stamped = _replace(note, retired=retired)
    _append(index_path(), stamped.to_record())
    return stamped


def render_push(found: list[Note], *, current: str | None = None) -> str | None:
    """The stderr block shown to a run that is about to repeat something, or ``None``.

    ``None`` rather than an empty string so the caller cannot print a blank alarm. A note written
    on a different version is dated inline: that one clause is the difference between advice and
    folklore, and it works with nobody curating anything.
    """
    if not found:
        return None
    running = current if current is not None else current_version()
    lines = ["[lab] a previous run left a note on this:"]
    for note in found:
        when = (note.ts or "")[:10]
        stale = ""
        if note.lab_version and running and note.lab_version != running:
            stale = f", lab {note.lab_version} — you are on {running}"
        who = note.author
        lines.append(f"  · {note.kind} ({when}, {who}{stale}) {note.text}")
    lines.append("  (`lab notes --retire <id>` when one of these stops being true)")
    return "\n".join(lines)


def last_failure_signature() -> str | None:
    """The signature of the most recent failed call in the ledger, or ``None``.

    Exists because a signature cannot be written by hand. The ledger masks an error message
    before signing it, so the signature of what the terminal printed is *not* the signature the
    digest computes — a user who typed one from the screen would produce a key that never
    matches. Reading it back off the ledger is the only way the push loop is closeable by a
    person, which makes this the difference between a wired feature and a used one.
    """
    try:
        from lab.events.read import read as read_events
        from lab.events.stats import signature

        for event in read_events(failures_only=True, limit=50):  # newest first
            if not event.error or not str(event.error.get("message") or "").strip():
                continue
            sig = signature(event.error)
            return None if sig == "unknown" else sig
    except Exception as e:  # noqa: BLE001 — a convenience lookup, never a failure
        debug(f"last-failure lookup failed: {e}")
    return None


def push_for_error(error: dict[str, Any] | None) -> str | None:
    """The block to print when a call fails, or ``None``.

    Keyed on :func:`lab.events.stats.signature`, the same normalisation the digest groups by — so
    a note fires exactly when the failure it was written about recurs, across differing job ids,
    zones and magnitudes. An errorless failure signs as the literal ``"unknown"``, which would
    match every unsigned note and turn a precise channel into a nag, so that case is refused
    outright. Wrapped whole: a note lookup must never convert a clean failure into a crash.
    """
    try:
        from lab.events.stats import signature

        if not error or not str(error.get("message") or "").strip():
            return None
        sig = signature(error)
        if sig == "unknown":
            return None
        return render_push(match(signature=sig))
    except Exception as e:  # noqa: BLE001 — never break the error path
        debug(f"push lookup failed: {e}")
        return None


def count_for_job(job_id: str, *, home: Path | str) -> int:
    """How many live notes this job carries. Cheap enough for `lab status`, which is polled."""
    try:
        return len([n for n in for_job(job_id, home=home) if n.retired is None])
    except Exception as e:  # noqa: BLE001 — a status read must never fail over notes
        debug(f"count failed: {e}")
        return 0


def as_markdown(found: list[Note]) -> str:
    """Render notes as the table the consuming project already keeps by hand.

    Its workflow mandates a per-workstream ``TEAM-LOG.md`` row at submit time
    (``when | actor | action | job id | cost``). Emitting that shape is what makes this feature
    subtraction rather than one more thing to remember.
    """
    header = "| when (UTC) | actor | kind | note | job id | cost |"
    rule = "|---|---|---|---|---|---|"
    rows = [
        "| {when} | {actor} | {kind} | {text} | {job} | {usd} |".format(
            when=(note.ts or "")[:19],
            actor=note.author,
            kind=note.kind,
            text=note.text.replace("|", r"\|").replace("\n", " "),
            job=note.job_id or "—",
            usd=f"${note.usd:.2f}" if note.usd is not None else "—",
        )
        for note in found
    ]
    return "\n".join([header, rule, *rows])
