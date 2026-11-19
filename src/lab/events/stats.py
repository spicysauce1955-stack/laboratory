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

#: A lab job id, exactly as ``lab.core._new_job_id`` mints it: ``%Y%m%d-%H%M%S`` + 6 hex. Sweep
#: and registration ids are the same thing behind a ``sweep-``/``reg-`` prefix, and a cluster is
#: ``lab-[<project-slug>-]<job_id>`` (``backends.skypilot.cluster_name_for``).
_JOB_ID = r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}"
#: A node SkyPilot built for one of our clusters: ``lab-<…>-<head|worker>-<uuid8>-<node type>``.
#: The middle is left loose on purpose — GCP truncates the cluster part past 35 characters, so a
#: live node can carry no job id at all (``lab-laboratory-20260820-ef-<userhash>-head-…``).
_NODE = r"\blab-[a-z0-9-]*?-(?:head|worker)-[a-z0-9]{8}-(?:compute|tpu|mig)\b"

# Order matters: zones before bare numbers, paths before shas, and lab ids before shas — the sha
# rule would otherwise eat a job id's 8-digit date and leave the rest of it behind.
#
# The trailing `(?:ms|[a-z])?` on the number pattern absorbs a duration-shorthand unit glued
# directly onto the digits (`20m`, `3s`, `45ms`) — with no separating space, `\b` doesn't fire
# between a digit and the following letter, so a plain `\d+` never matches `20m` at all and two
# timeouts that differ only in how long they waited stay two different signatures.
_NORMALIZERS = (
    (re.compile(r"\b[a-z]+-[a-z]+\d+(?:-[a-z])?\b"), "<zone>"),
    (re.compile(r"(?:/[\w.\-]+){2,}"), "<path>"),
    (re.compile(_NODE), "<cluster>"),
    (re.compile(rf"\blab-(?:[a-z0-9<>-]+?-)?{_JOB_ID}\b"), "<cluster>"),
    (re.compile(rf"\bsweep-{_JOB_ID}\b"), "<sweep>"),
    (re.compile(rf"\breg-{_JOB_ID}\b"), "<reg>"),
    (re.compile(rf"\b{_JOB_ID}\b"), "<job>"),
    (re.compile(r"\b[0-9a-f]{7,40}\b"), "<sha>"),
    (re.compile(r"\bj-[0-9a-z]+\b"), "<job>"),
    (re.compile(r"\b\d+(?:\.\d+)?(?:ms|[a-z])?\b"), "<n>"),
)
#: How much of the normalised message a signature keeps.
MAX_MESSAGE = 120

# What a lab id looked like *after* the normaliser that predates the lab-id rules above: the
# 8-digit date fell to the sha rule, the time to the number rule, and the 6-hex suffix survived
# (or became `<n>` when the number rule could take it, e.g. all digits or `12345a`). Notes store
# the signature they were filed under, so these rewrite such a stored key onto today's form.
_LEGACY_JOB = r"<sha>-<n>-(?:[0-9a-f]{6}|<n>)"
_LEGACY_UPGRADES = (
    (
        re.compile(
            r"\blab-[a-z0-9<>-]*?-(?:head|worker)-(?:[a-z0-9]{8}|<sha>)-(?:compute|tpu|mig)\b"
        ),
        "<cluster>",
    ),
    (re.compile(rf"\blab-(?:[a-z0-9<>-]+?-)?{_LEGACY_JOB}(?![\w-])"), "<cluster>"),
    (re.compile(rf"\bsweep-{_LEGACY_JOB}(?![\w-])"), "<sweep>"),
    (re.compile(rf"\breg-{_LEGACY_JOB}(?![\w-])"), "<reg>"),
    (re.compile(rf"(?<![\w-]){_LEGACY_JOB}(?![\w-])"), "<job>"),
)


def signature(error: dict[str, Any] | None) -> str:
    """A stable key for 'the same bug', across differing ids, zones and magnitudes."""
    if not error:
        return "unknown"
    message = str(error.get("message", ""))
    for pattern, placeholder in _NORMALIZERS:
        message = pattern.sub(placeholder, message)
    return f"{error.get('type', 'Error')}: {message[:MAX_MESSAGE]}".strip()


def upgrade_signature(sig: str) -> str:
    """Rewrite a signature stored by an older normaliser onto today's placeholders.

    Before lab ids were normalised, ``unknown job id '20261119-102456-7486e1'`` signed as
    ``…'<sha>-<n>-7486e1'`` — a key unique to that one job, so a note filed under it could never
    fire for the next one. Upgrading the stored key on read (``lab.notes.match``) is what turns
    those notes back on, with no migration of the append-only index. Idempotent, and the
    identity on a signature with no legacy-shaped id in it.
    """
    for pattern, placeholder in _LEGACY_UPGRADES:
        sig = pattern.sub(placeholder, sig)
    return sig


def same_signature(stored: str, current: str) -> bool:
    """Does a note filed under ``stored`` speak to a failure signed ``current`` today?

    Exact match, or a match after :func:`upgrade_signature`. One more case: the old normaliser
    left ~13 characters more per id, so a long message reached the :data:`MAX_MESSAGE` cap
    *earlier* — its upgraded key is then a strict prefix of today's. That prefix is only
    accepted when the stored message sits exactly at the cap (it was cut) and the upgrade
    actually rewrote an id (it is a legacy key at all), so an ordinary short signature can
    never match a longer one by prefix.
    """
    if stored == current:
        return True
    upgraded = upgrade_signature(stored)
    if upgraded == current:
        return True
    _type, sep, message = stored.partition(": ")
    return (
        bool(sep)
        and len(message) == MAX_MESSAGE
        and upgraded != stored
        and current.startswith(upgraded)
    )


_ACTION_SEPARATORS = re.compile(r"[\s_-]+")


def action_key(action: str) -> str:
    """One key per command, whichever surface recorded it.

    The CLI records a top-level command by its name (``sweep-aggregate``) and a group leaf as
    ``"<group> <leaf>"`` (``queue list``, ``cli._group_action``); the MCP tool of the same
    command records ``sweep_aggregate`` / ``queue_list``. Folding ``-``, ``_`` and whitespace
    together is what makes them one row. Only the *grouping* uses this: every record keeps the
    action and the ``surface`` it was written with.
    """
    return _ACTION_SEPARATORS.sub(" ", action.strip().lower())


def action_labels(events: Sequence[Event]) -> dict[str, str]:
    """The display name for each :func:`action_key` among ``events``.

    The CLI's spelling when one is present — it is what a person types to run the command again
    — otherwise whatever was recorded. ``min`` keeps the choice independent of event order.
    """
    spellings: dict[str, dict[bool, set[str]]] = defaultdict(lambda: defaultdict(set))
    for event in events:
        spellings[action_key(event.action)][event.surface == "cli"].add(event.action)
    return {key: min(by_cli[True] or by_cli[False]) for key, by_cli in spellings.items()}


def cost_usd(event: Event) -> float:
    """``result`` is whatever the entrypoint wrote to JSON — ``cost_usd`` can be any JSON value,
    not just a number. A non-numeric value is treated as no cost rather than raised, so one
    malformed record degrades instead of crashing the whole aggregate view."""
    cost = event.result.get("cost_usd")
    return float(cost) if isinstance(cost, (int, float)) else 0.0


def stats(events: Sequence[Event], *, since: datetime | None = None) -> StatsView:
    """Per-action failure rates, ranked error signatures, and the money spent on failures."""
    labels = action_labels(events)
    by_action: dict[str, list[Event]] = defaultdict(list)
    by_sig: dict[str, list[Event]] = defaultdict(list)
    usd_burned = 0.0
    dangling = 0
    for event in events:
        by_action[labels[action_key(event.action)]].append(event)
        if event.outcome is None:
            dangling += 1
        if event.failed:
            by_sig[signature(event.error)].append(event)
            usd_burned += cost_usd(event)

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
            actions=sorted({labels[action_key(e.action)] for e in group}),
            usd=round(sum(cost_usd(e) for e in group), 4),
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
