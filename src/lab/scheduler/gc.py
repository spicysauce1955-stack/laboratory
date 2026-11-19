"""Bundle garbage collection — delete code tarballs no live registration still needs.

A bundle is safe to delete once every registration that references its ``bundle_key`` is terminal
(``TERMINAL_REG_STATES``): a terminal reg will never launch or relaunch, so its code snapshot is
dead weight. A *shared* sweep bundle is therefore kept until ALL of the sweep's points are terminal
(each point references the same key). Dry-run by default — deletions are explicit (cost-safety).

Two cases look like "no live reg references it" without being safe, and are kept:

* **An entry that could not be read.** ``list_entries`` skips a blob it cannot validate (a
  version-skewed client or host, a read racing a write). That registration is very likely live,
  and its bundle key is unknowable from the outside — so with any unreadable entry in the
  listing, nothing is deleted at all and the report says which entries blocked it.
* **A registration that is not committed yet.** ``register`` uploads the bundle first and writes
  the entry last (the commit point, spec §5); a GC landing in between sees a bundle nothing
  references. A bundle no entry references at all is only deleted once it is older than
  :data:`MIN_BUNDLE_AGE` — its age read from the timestamp lab embeds in every reg/sweep id
  (``reg-YYYYmmdd-HHMMSS-xxxxxx``); a name that carries none is kept and reported.
  A bundle referenced only by *terminal* entries needs no age: no register can be in flight
  for it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any

from lab.scheduler.models import TERMINAL_REG_STATES
from lab.scheduler.queue import QueueStore, unreadable_entry_ids

MIN_BUNDLE_AGE = timedelta(hours=24)
"""How old an unreferenced bundle must be before GC may delete it. Generous on purpose: what it
protects against is a register that is seconds-to-minutes from committing (or a laptop clock a
little off), and what it costs is a day of storage for a tarball."""

_ID_TIMESTAMP = re.compile(r"^(?:reg|sweep)-(\d{8})-(\d{6})-")


def bundle_created_at(bundle_key: str) -> datetime | None:
    """When the registration owning ``bundle_key`` was created, from its id; None if unknowable."""
    m = _ID_TIMESTAMP.match(PurePosixPath(bundle_key).name)
    if m is None:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def gc_bundles(
    queue: QueueStore,
    *,
    apply: bool = False,
    min_age: timedelta = MIN_BUNDLE_AGE,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Report (and, with ``apply``, delete) bundle tarballs no non-terminal reg references."""
    entries = queue.list_entries()
    unreadable = sorted(unreadable_entry_ids(queue))
    live = {reg.bundle_key for reg in entries if reg.state not in TERMINAL_REG_STATES}
    referenced_by_any = {reg.bundle_key for reg in entries}
    all_keys = queue.list_bundle_keys()
    now = now_fn()
    orphaned: list[str] = []
    too_young: list[str] = []
    unknown_age: list[str] = []
    for key in sorted(k for k in all_keys if k not in live):
        if key in referenced_by_any:  # only terminal entries reference it: no register in flight
            orphaned.append(key)
            continue
        created = bundle_created_at(key)
        if created is None:
            unknown_age.append(key)
        elif now - created < min_age:
            too_young.append(key)
        else:
            orphaned.append(key)
    refused = (
        f"{len(unreadable)} queue entr{'y' if len(unreadable) == 1 else 'ies'} could not be read; "
        "their bundles cannot be identified, so nothing is deleted until they can"
        if unreadable
        else None
    )
    deleted: list[str] = []
    if apply and refused is None:
        for key in orphaned:
            queue.delete_bundle(key)
            deleted.append(key)
    return {
        "total_bundles": len(all_keys),
        "referenced": len([k for k in all_keys if k in live]),
        "orphaned": orphaned,
        "deleted": deleted,
        "applied": apply,
        "kept_too_young": too_young,
        "kept_unknown_age": unknown_age,
        "min_age_s": int(min_age.total_seconds()),
        "unreadable_entries": unreadable,
        "refused": refused,
    }
