"""Bundle garbage collection — delete code tarballs no live registration still needs.

A bundle is safe to delete once every registration that references its ``bundle_key`` is terminal
(``TERMINAL_REG_STATES``): a terminal reg will never launch or relaunch, so its code snapshot is
dead weight. A *shared* sweep bundle is therefore kept until ALL of the sweep's points are terminal
(each point references the same key). Dry-run by default — deletions are explicit (cost-safety).
"""

from __future__ import annotations

from typing import Any

from lab.scheduler.models import TERMINAL_REG_STATES
from lab.scheduler.queue import QueueStore


def gc_bundles(queue: QueueStore, *, apply: bool = False) -> dict[str, Any]:
    """Report (and, with ``apply``, delete) bundle tarballs no non-terminal reg references."""
    live = {reg.bundle_key for reg in queue.list_entries() if reg.state not in TERMINAL_REG_STATES}
    all_keys = queue.list_bundle_keys()
    orphaned = sorted(k for k in all_keys if k not in live)
    deleted: list[str] = []
    if apply:
        for key in orphaned:
            queue.delete_bundle(key)
            deleted.append(key)
    return {
        "total_bundles": len(all_keys),
        "referenced": len(all_keys) - len(orphaned),
        "orphaned": orphaned,
        "deleted": deleted,
        "applied": apply,
    }
