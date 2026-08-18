"""Aggregate the ledger into a fix-this-first list.

Everything here hangs off the *error signature*: type plus the message with ids, numbers, paths
and zone names normalized out. Without it, twelve occurrences of one bug are twelve unique
strings and the aggregate view is voluminous rather than actionable.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from lab.events.models import ActionStat, Event, SignatureStat, StatsView

# Order matters: zones before bare numbers, paths before shas.
#
# The trailing `(?:ms|[a-z])?` on the number pattern absorbs a duration-shorthand unit glued
# directly onto the digits (`20m`, `3s`, `45ms`) — with no separating space, `\b` doesn't fire
# between a digit and the following letter, so a plain `\d+` never matches `20m` at all and two
# timeouts that differ only in how long they waited stay two different signatures.
_NORMALIZERS = (
    (re.compile(r"\b[a-z]+-[a-z]+\d+(?:-[a-z])?\b"), "<zone>"),
    (re.compile(r"(?:/[\w.\-]+){2,}"), "<path>"),
    (re.compile(r"\b[0-9a-f]{7,40}\b"), "<sha>"),
    (re.compile(r"\bj-[0-9a-z]+\b"), "<job>"),
    (re.compile(r"\b\d+(?:\.\d+)?(?:ms|[a-z])?\b"), "<n>"),
)


def signature(error: dict[str, Any] | None) -> str:
    """A stable key for 'the same bug', across differing ids, zones and magnitudes."""
    if not error:
        return "unknown"
    message = str(error.get("message", ""))
    for pattern, placeholder in _NORMALIZERS:
        message = pattern.sub(placeholder, message)
    return f"{error.get('type', 'Error')}: {message[:120]}".strip()


def _cost_usd(event: Event) -> float:
    """``result`` is whatever the entrypoint wrote to JSON — ``cost_usd`` can be any JSON value,
    not just a number. A non-numeric value is treated as no cost rather than raised, so one
    malformed record degrades instead of crashing the whole aggregate view."""
    cost = event.result.get("cost_usd")
    return float(cost) if isinstance(cost, (int, float)) else 0.0


def stats(events: Sequence[Event], *, since: datetime | None = None) -> StatsView:
    """Per-action failure rates, ranked error signatures, and the money spent on failures."""
    by_action: dict[str, list[Event]] = defaultdict(list)
    by_sig: dict[str, list[Event]] = defaultdict(list)
    usd_burned = 0.0
    dangling = 0
    for event in events:
        by_action[event.action].append(event)
        if event.outcome is None:
            dangling += 1
        if event.failed:
            by_sig[signature(event.error)].append(event)
            usd_burned += _cost_usd(event)

    actions: list[ActionStat] = []
    for action, group in by_action.items():
        failures = sum(1 for e in group if e.failed)
        durations = [e.duration_ms for e in group if isinstance(e.duration_ms, int)]
        actions.append(
            ActionStat(
                action=action, calls=len(group), failures=failures,
                failure_rate=round(failures / len(group), 3),
                median_ms=int(statistics.median(durations)) if durations else 0,
            )
        )
    actions.sort(key=lambda a: (-a.failures, -a.calls, a.action))

    signatures: list[SignatureStat] = [
        SignatureStat(
            signature=sig,
            count=len(group),
            first_seen=min(e.ts for e in group),
            last_seen=max(e.ts for e in group),
            actions=sorted({e.action for e in group}),
            usd=round(sum(_cost_usd(e) for e in group), 4),
        )
        for sig, group in by_sig.items()
    ]
    signatures.sort(key=lambda s: (-s.count, -s.usd, s.signature))

    return StatsView(
        since=since,
        total=len(events),
        failures=sum(1 for e in events if e.failed),
        dangling=dangling,
        usd_burned=round(usd_burned, 4),
        actions=actions,
        signatures=signatures,
    )


def stats_dict(view: StatsView) -> dict[str, Any]:
    """The JSON shape ``lab history --stats`` emits."""
    return {
        "since": view.since.isoformat() if view.since else None,
        "total": view.total,
        "failures": view.failures,
        "dangling": view.dangling,
        "usd_burned": view.usd_burned,
        "actions": [vars(a) for a in view.actions],
        "signatures": [
            {**vars(s), "first_seen": s.first_seen.isoformat(), "last_seen": s.last_seen.isoformat()}
            for s in view.signatures
        ],
    }
