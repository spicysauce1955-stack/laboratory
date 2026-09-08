"""On-disk ledger: one JSONL file per UTC day under ``~/.lab/events``.

User-global rather than project-local on purpose — the lab installs into other projects
(v0.5.0+), so a project-local store would scatter the history across repos exactly when the
cross-project pattern is the thing worth seeing. Every event carries its project, so
per-project filtering is a read-side concern.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_DIR = "~/.lab/events"


def lock_path(path: Path) -> Path:
    """Return the stable-inode lock file for a day file.

    Uses string concatenation (not with_suffix) to avoid replacing .jsonl,
    ensuring lock files remain outside the ????-??-??.jsonl glob.
    """
    return Path(str(path) + ".lock")


def enabled() -> bool:
    return (os.environ.get("LAB_EVENTS") or "").strip() != "0"


def debug(message: str) -> None:
    """Surface a swallowed logging error, but only when asked (``LAB_EVENTS_DEBUG=1``)."""
    if (os.environ.get("LAB_EVENTS_DEBUG") or "").strip() == "1":
        print(f"[lab.events] {message}", file=sys.stderr)


def events_dir() -> Path:
    override = (os.environ.get("LAB_EVENTS_DIR") or "").strip()
    return Path(override).expanduser() if override else Path(DEFAULT_DIR).expanduser()


def day_file(when: datetime) -> Path:
    return events_dir() / f"{when.strftime('%Y-%m-%d')}.jsonl"


def day_files() -> list[Path]:
    try:
        return sorted(events_dir().glob("????-??-??.jsonl"))
    except OSError as e:
        debug(f"listing failed: {e}")
        return []


def append(record: dict[str, Any], *, when: datetime) -> None:
    """Append one record as a single locked ``O_APPEND`` write.

    A sharded sweep launches many ``lab`` processes against this one file; the lock is what
    rules out the torn or interleaved lines that would make the store untrustworthy in exactly
    the situation where it matters most. Best-effort throughout: a ledger failure must never
    fail a command.
    """
    if not enabled():
        return
    try:
        path = day_file(when)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
        lock = lock_path(path)
        with lock.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                with path.open("a", encoding="utf-8") as day_f:
                    day_f.write(line)
                    day_f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:  # noqa: BLE001 — logging must never fail a command
        debug(f"append failed: {e}")


def iter_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield every parseable record. A ledger unreadable because one line is bad would fail at
    its only job, so malformed lines are skipped, never raised on."""
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        yield record
        except OSError as e:
            debug(f"read failed for {path}: {e}")


STAMP = ".pruned"
READ_STAMP_DIR = ".reads"
READ_MIN_INTERVAL_S = 60.0


def _int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def read_min_interval_s() -> float:
    """Minimum gap between two *recorded* successes of the same read-only action.

    ``LAB_EVENTS_READ_MIN_INTERVAL_S``, default 60s. Garbage falls back to the default (see
    ``_float_env``); ``0`` or a negative value disables the limit and records every poll.
    """
    return _float_env("LAB_EVENTS_READ_MIN_INTERVAL_S", READ_MIN_INTERVAL_S)


def read_stamp_path(action: str, project: str) -> Path:
    """The per-``(action, project)`` last-recorded-success stamp.

    Kept in a subdirectory so it can never be mistaken for a day file: ``day_files()`` globs
    ``????-??-??.jsonl`` at the top level, so nothing here enters the byte budget, the age cap or
    any read. The name is slugified and hash-suffixed — a project directory name is arbitrary
    text, and two different projects must not collide onto one window.
    """
    key = f"{action}\x00{project}"
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{action}-{project}")[:48]
    digest = hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:8]
    return events_dir() / READ_STAMP_DIR / f"{slug}-{digest}.stamp"


def claim_read_slot(action: str, project: str, *, now: datetime, min_interval_s: float) -> bool:
    """Claim the right to record one success for ``(action, project)``; ``False`` means skip it.

    The 2026-09 campaign made 98,654 *successful* ``lab status`` calls in five days — a flat
    1,050/hour of runaway shell loops — and every pair was written, filling the byte cap with
    records too fresh for the age-gated ``compact()`` to touch. The fix is to record at most one
    successful read of an action per project per window and drop the rest.

    Note what that does *not* say. The key is ``(action, project)``, with no target in it, so the
    dropped reads need not have named the same job as the recorded one: a successful
    ``lab status jobB`` inside ``lab status jobA``'s window is simply not written. That cost was
    measured against the real ledger and accepted — ``record._READ_ONLY_ACTIONS`` carries the
    numbers and the reason a target-keyed window does not work.

    Cost matters more than precision here: this runs in ~100k short-lived processes, so the
    last-recorded-success time lives in its own tiny stamp file rather than in the day file,
    which would otherwise have to be read (or rewritten) on every poll.

    Concurrency: the check and the stamp update happen under ``flock`` on the stamp itself, so
    two simultaneous polls cannot both decide they are the first. The lock is taken
    non-blocking — another process holding it *is* a concurrent poll of this same action, which
    is exactly the case to skip, and a poll must never wait on the ledger.

    Every failure mode resolves to ``True`` (record it): a missing, corrupt, or unreadable stamp
    means "no recent success", and a clock that moved backwards records rather than suppresses.
    An unrecorded call is invisible forever, so the fallback direction is to keep it.
    """
    if not enabled() or min_interval_s <= 0:
        return True
    path = read_stamp_path(action, project)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_RDWR (not "a+"): O_APPEND would pin every write to end-of-file, and this file is
        # rewritten in place, not appended to.
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, "r+", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            try:
                last = None
                try:
                    last = float(f.read(64).strip())
                except (ValueError, OSError) as e:
                    debug(f"unreadable read-rate stamp {path} (recording): {e}")
                current = now.timestamp()
                if last is not None and 0.0 <= current - last < min_interval_s:
                    return False
                f.seek(0)
                f.truncate()
                f.write(f"{current:.3f}")
                f.flush()
                return True
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        debug(f"read-rate stamp failed for {action}: {e}")
        return True


def _rewrite(path: Path, records: list[dict[str, Any]]) -> None:
    """Replace a day file. Caller must hold the lock via lock_path(path).

    The caller takes the lock, reads the file, computes kept records, writes this temp file
    under the lock, then replaces atomically. This ensures concurrent appends cannot be lost.
    """
    tmp = path.with_suffix(".jsonl.tmp")
    text = "".join(json.dumps(r, default=str, separators=(",", ":")) + "\n" for r in records)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def is_read_only_action(action: str) -> bool:
    """True for the cheap polling reads — the ``_READ_ONLY_ACTIONS`` allowlist in
    ``lab.events.record``, which is also what the write-path rate limit keys off.

    Imported lazily because ``record`` imports *this* module; a module-level import would be
    circular. Going through ``read_only_key`` (rather than testing the raw string) is what folds
    the two real spellings of a grouped read — the CLI's ``queue list`` and the MCP tool's
    ``queue_list`` — onto one entry.
    """
    from lab.events.record import read_only_key

    return read_only_key(action) is not None


def _compactable_ids(paths: list[Path], action_filter: Callable[[str], bool] | None) -> set[str]:
    """Ids of successfully-closed calls, optionally narrowed to actions ``action_filter`` accepts.

    Both passes read *every* day file — see ``compact``'s docstring for why the succeeded set has
    to be global. The narrowing needs it twice over: a call's outcome lives on its ``close`` and
    its action lives on its ``open``, so a call straddling UTC midnight has the two halves of the
    test in two different files. A success whose ``open`` cannot be found has no action to judge
    and is therefore *not* selected — the narrow lever only ever drops what it can positively
    identify, and the broader stage above it is what eventually takes the rest.
    """
    succeeded = {
        r["id"] for r in iter_records(paths)
        if r.get("phase") == "close" and r.get("outcome") == "ok"
        and isinstance(r.get("id"), str)
    }
    if action_filter is None:
        return succeeded
    selected: set[str] = set()
    for r in iter_records(paths):
        id_, action = r.get("id"), r.get("action")
        if r.get("phase") != "open" or not isinstance(id_, str) or not isinstance(action, str):
            continue
        if id_ in succeeded and action_filter(action):
            selected.add(id_)
    return selected


def compact(
    *,
    now: datetime,
    success_ttl_days: int,
    ignore_ttl: bool = False,
    exclude_today: bool = False,
    action_filter: Callable[[str], bool] | None = None,
) -> None:
    """Drop successful calls older than the TTL. Failures — and dangling opens, which are
    themselves a finding — stay until the age cap takes them.

    The succeeded-id set is built across *every* day file, not just the ones old enough to be
    rewritten here: a call whose ``open`` lands in day N and whose ``close`` lands in day N+1
    (any supervisor run of more than a few hours, or an overnight scheduled job — exactly the
    expensive jobs this ledger most needs to get right) would otherwise never be recognised as
    succeeded when day N is compacted on its own — day N has no close to prove it, and day N+1's
    close, read on its own turn, has no matching open in *that* file to pair it with. Read that
    way, a known success becomes a permanent "running-or-died" phantom after the TTL. Computing
    the set globally first, then filtering each eligible file against it, closes that gap — and
    it holds for both selections below, since only the *rewritten* set narrows, never the set the
    ids are read from.

    ``ignore_ttl`` widens the selection to successes of any age and ``exclude_today`` spares the
    current day file; ``action_filter`` *narrows* it to successes of the actions it accepts.
    Together they are what ``enforce_caps`` asks for when the ledger is over the byte cap, where
    the alternative lever is deleting a whole day of forensics — and the narrowing is what keeps
    the first, cheapest stage from spending a successful ``submit`` to relieve an overage made
    entirely of ``lab status`` polls.
    """
    cutoff = now - timedelta(days=success_ttl_days)
    today = day_file(now).name
    paths = day_files()
    old_paths = []
    for path in paths:
        try:
            stamped = datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as e:
            debug(f"compaction skipped unparseable day file {path}: {e}")
            continue
        if exclude_today and path.name == today:
            continue
        if ignore_ttl or stamped < cutoff:
            old_paths.append(path)
    if not old_paths:
        return
    try:
        droppable = _compactable_ids(paths, action_filter)
    except Exception as e:  # noqa: BLE001
        debug(f"compaction failed building the succeeded set: {e}")
        return
    for path in old_paths:
        try:
            lock = lock_path(path)
            with lock.open("a", encoding="utf-8") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    records = list(iter_records([path]))
                    kept = [r for r in records if r.get("id") not in droppable]
                    if len(kept) != len(records):
                        _rewrite(path, kept)
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        except Exception as e:  # noqa: BLE001
            debug(f"compaction failed for {path}: {e}")


def _total_size(paths: Iterable[Path]) -> int:
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def enforce_caps(*, now: datetime, max_age_days: int, max_mb: float) -> None:
    """Delete whole day files past the age cap; then, if over the byte cap, buy the overage back
    in escalating stages, cheapest loss first, stopping the moment the total is under budget:

    1. drop successful **read-only** calls (``is_read_only_action``: the polls) outside today;
    2. still over → drop **all** successes outside today;
    3. still over → delete whole day files, oldest first, today excluded.

    Compacting before deleting is the lesson of the 2026-09 campaign: 98,654 successful ``lab
    status`` calls filled the cap in days, all of them far too fresh for the TTL-gated
    ``compact()``, so deleting a whole day file was the only lever left — and it took day one of a
    live campaign, the day of the incident under investigation, while the investigation was
    running.

    Stage 1 exists because stage 2 is far broader than the problem: those 98,654 polls *were*
    essentially the whole overage, so relieving it should not also cost every prior day's
    successful ``submit``/``sweep``/``register``/``reconcile`` record — with its ``refs`` and
    ``result``, the rows ``lab history --job`` and ``lab report``'s costs are read from — for
    records still well inside the 14-day success TTL. In practice stage 1 ends it; 2 and 3 are the
    fallback for an overage the polls cannot explain. Under the cap nothing is compacted early, so
    a fresh success inside its TTL is still readable.

    Failures (and dangling opens) are never compacted at any stage; only stage 3 can take them,
    and only by the day file. Today's file is never the sacrifice, by any lever: it is excluded
    from both compactions (its records are what anyone is about to read) and from the deletion (a
    single day over the cap overshoots by at most that day, and tomorrow it is compactable like
    any other).
    """
    cutoff = now - timedelta(days=max_age_days)
    today = day_file(now).name
    remaining: list[Path] = []
    for path in day_files():
        try:
            stamped = datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if stamped < cutoff:
            path.unlink(missing_ok=True)
        else:
            remaining.append(path)
    budget = max_mb * 1024 * 1024
    total = _total_size(remaining)
    # Stage 1 (polls only) then stage 2 (every success), with the total rechecked in between:
    # whichever stage gets the ledger under budget is the last one that runs.
    for stage_filter in (is_read_only_action, None):
        if total <= budget:
            break
        compact(
            now=now,
            success_ttl_days=0,
            ignore_ttl=True,
            exclude_today=True,
            action_filter=stage_filter,
        )
        remaining = [p for p in remaining if p.exists()]
        total = _total_size(remaining)
    for path in remaining:  # stage 3: oldest first
        if total <= budget:
            break
        if path.name == today:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total -= size
        path.unlink(missing_ok=True)


def reap_stale_locks(*, now: datetime, min_age_days: float = 1.0) -> None:
    """Delete ``.jsonl.lock`` files whose day file is gone and which are themselves a day stale.

    Ten of these were sitting in the real ledger directory, one for the day the byte cap had
    already deleted: litter that outlives the file it guarded, and a misleading one at that.

    Three conditions make this safe against a process that might be holding a lock right now,
    and all three are needed:

    * **The day file must be absent.** A lock whose day file exists is live infrastructure — it
      is the mutual exclusion between ``append()`` and the rewrite in ``compact()``.
    * **The lock must be older than a day.** ``append()`` creates the lock before the day file,
      so a brand-new lock with no day file is a writer mid-flight, not litter.
    * **We must be able to take the lock ourselves**, non-blocking. Anything holding it makes us
      leave it alone.

    Even so, unlinking a lock is not free of theory: a holder keeps its ``flock`` on the now
    unnamed inode, while a later process creating the same name gets a *different* inode and a
    lock that excludes nobody. That is why the flock probe is taken before the unlink — it closes
    the window down to a writer that has opened the name but not yet flocked it, for a day whose
    file has been absent for over 24h, and whose worst case is one interleaved line rather than
    lost data. Deleting nothing at all is the other option, and it is what left the litter.
    """
    try:
        locks = sorted(events_dir().glob("????-??-??.jsonl.lock"))
    except OSError as e:
        debug(f"lock listing failed: {e}")
        return
    for lock in locks:
        day = Path(str(lock)[: -len(".lock")])
        if day.exists():
            continue
        try:
            mtime = datetime.fromtimestamp(lock.stat().st_mtime, tz=timezone.utc)
            if now - mtime < timedelta(days=min_age_days):
                continue
            with lock.open("a", encoding="utf-8") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as e:
                    debug(f"lock {lock} is held, leaving it: {e}")
                    continue
                try:
                    if not day.exists():  # re-checked under the lock
                        lock.unlink(missing_ok=True)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:  # noqa: BLE001 — retention is best-effort, always
            debug(f"reaping {lock} failed: {e}")


def maybe_prune(*, now: datetime) -> None:
    """Run retention at most once per UTC day per machine. Lazy, stamp-gated, best-effort."""
    if not enabled():
        return
    try:
        stamp = events_dir() / STAMP
        today = now.strftime("%Y-%m-%d")
        if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == today:
            return
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(today, encoding="utf-8")
        compact(now=now, success_ttl_days=_int_env("LAB_EVENTS_SUCCESS_TTL_DAYS", 14))
        enforce_caps(
            now=now,
            max_age_days=_int_env("LAB_EVENTS_MAX_AGE_DAYS", 90),
            max_mb=_float_env("LAB_EVENTS_MAX_MB", 50),
        )
        reap_stale_locks(now=now)
    except Exception as e:  # noqa: BLE001
        debug(f"pruning failed: {e}")
