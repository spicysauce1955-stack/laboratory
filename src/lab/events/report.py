"""The markdown digest — a field report generated instead of hand-written.

Shaped after ``FIELD-REPORT-2026-08-12-capability-campaign.md``: a triage table first, then
per-finding *attempted / observed / cost*, with the job ids that reach the manifest and logs.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from lab.events.models import Event
from lab.events.stats import cost_usd, signature, stats

_DANGLING_KEY = "never closed (running-or-died)"
_WS_RUN = re.compile(r"\s+")
_BACKTICK_RUN = re.compile(r"`+")


def _severity(count: int, usd: float) -> float:
    """Frequency, weighted by what it burned. A cheap bug seen five times outranks an expensive
    one seen once only when the money is small — which is the call the reader wants made."""
    return count * (1.0 + usd)


def _inline(text: str) -> str:
    """Collapse embedded newlines/carriage-returns and runs of whitespace so free text stays on
    one line wherever it's rendered outside a fenced code block. Error messages come from
    ``str(exc)`` verbatim (``error_dict``) and routinely span multiple lines — a subprocess
    traceback, a pydantic validation error — which would otherwise truncate a markdown heading
    or split a ``**Observed:**`` line onto an orphan line with no label."""
    return _WS_RUN.sub(" ", text).strip()


def _escape_cell(text: str) -> str:
    """A literal pipe inside a table cell would split it into extra columns. Whitespace
    collapsing is a separate job (``_inline``) — compose the two for table cells rather than
    merging them, since prose sections need inlining without ever needing pipe-escaping."""
    return text.replace("|", "\\|")


def _fence_for(lines: list[str]) -> str:
    """A fenced code block's fence must be at least as long as any run of backticks inside it,
    or a literal ``` ` `` `` in the content (a quoted shell pipeline, a markdown snippet in an
    error) closes the fence early and spills the rest of the trace into the document as prose."""
    longest = max((len(m.group()) for line in lines for m in _BACKTICK_RUN.finditer(line)),
                  default=0)
    return "`" * max(3, longest + 1)


def _params_line(event: Event) -> str:
    items = ", ".join(f"{k}={_inline(str(v))}" for k, v in list(event.params.items())[:6])
    return f"`lab {event.action}` ({items})" if items else f"`lab {event.action}`"


def _group_key(event: Event) -> str:
    """A dangling open has no error to sign, so it gets its own bucket rather than whatever
    ``signature(None)`` would otherwise collapse every errorless failure into."""
    return _DANGLING_KEY if event.outcome is None else signature(event.error)


def report(events: Sequence[Event], *, since: datetime | None = None) -> str:
    """Render failures in the window as a pasteable markdown report."""
    view = stats(events, since=since)
    window = f"since {since.isoformat()}" if since else "all recorded history"
    lines: list[str] = [
        f"# Lab event report — {window}",
        "",
        f"{view.total} calls, {view.failures} failed, {view.dangling} never closed, "
        f"${view.usd_burned:.4f} burned in failed calls.",
        "",
    ]
    failures = [e for e in events if e.failed]
    if not failures:
        lines += ["**No failures in this window.**", ""]
        return "\n".join(lines)

    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in failures:
        grouped[_group_key(event)].append(event)
    ranked = sorted(
        grouped.items(),
        key=lambda kv: -_severity(len(kv[1]), sum(cost_usd(e) for e in kv[1])),
    )

    lines += [
        "## Triage",
        "",
        "| # | Finding | Seen | $ burned | Actions |",
        "|---|---|---|---|---|",
    ]
    for i, (key, group) in enumerate(ranked, 1):
        usd = sum(cost_usd(e) for e in group)
        actions = ", ".join(sorted({e.action for e in group}))
        lines.append(
            f"| F{i} | {_escape_cell(_inline(key))} | {len(group)} | ${usd:.4f} | "
            f"{_escape_cell(_inline(actions))} |"
        )
    lines += ["", "---", ""]

    for i, (key, group) in enumerate(ranked, 1):
        newest = max(group, key=lambda e: e.ts)
        usd = sum(cost_usd(e) for e in group)
        job_ids = sorted({str(e.refs.get("job_id")) for e in group if e.refs.get("job_id")})
        message = _inline(str(newest.error.get("message"))) if newest.error else ""
        where = (
            _inline(str(newest.error.get("where")))
            if newest.error and "where" in newest.error else ""
        )
        lines += [
            f"## F{i} — {_inline(key)}",
            "",
            f"**Attempted:** {_params_line(newest)}  ",
            f"**Observed:** {newest.status}"
            + (f" — {message}" if newest.error else "")
            + (f" (at `{where}`)" if where else "") + "  ",
            f"**Seen:** {len(group)}× between {min(e.ts for e in group).isoformat()} "
            f"and {max(e.ts for e in group).isoformat()}  ",
            f"**Cost:** ${usd:.4f}  ",
        ]
        if job_ids:
            lines.append(f"**Jobs:** {', '.join(job_ids)} (`runs/<job_id>/logs.txt`)  ")
        if newest.trace:
            trace_lines = [f"+{n.t:>7}ms  {n.k}  {n.d}" for n in newest.trace]
            fence = _fence_for(trace_lines)
            lines += ["", "Trace of the most recent occurrence:", "", fence]
            lines += trace_lines
            lines += [fence]
        lines.append("")
    return "\n".join(lines)


def report_dict(events: Sequence[Event], *, since: datetime | None = None) -> dict[str, Any]:
    """``{markdown}`` — the MCP tool's return shape."""
    return {"markdown": report(events, since=since)}
