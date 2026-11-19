"""The payload `lab list` and the MCP `list` tool both return — one implementation, two shells.

Bounded by default: at 10k jobs the unbounded listing measured 1.24 MB over MCP, which an agent
cannot use and pays for in context. A page keeps the newest ``limit`` jobs (by ``created_at``)
in the order the listing has always had, and says how many it left out. The ``spend`` block is
computed over **every** job: it is the project's money, not a page of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lab.core import Lab

#: Jobs per `list` page when the caller does not say (``limit=0`` returns every job).
DEFAULT_LIST_LIMIT = 100


def list_view(
    lab: Lab, *, limit: int = DEFAULT_LIST_LIMIT, spend_alert: float | None = None
) -> dict[str, Any]:
    """``{jobs, jobs_total, jobs_omitted, limit, spend}``; ``created_at`` is ISO-8601."""
    if limit < 0:
        raise ValueError(f"limit must be >= 0 (0 = every job), got {limit}")
    jobs = lab.list_jobs()
    shown = jobs
    if limit and len(jobs) > limit:
        newest = sorted(jobs, key=lambda m: (m.created_at.timestamp(), m.job_id))[-limit:]
        keep = {m.job_id for m in newest}
        shown = [m for m in jobs if m.job_id in keep]
    return {
        "jobs": [
            {
                "job_id": m.job_id,
                "sweep_id": m.sweep_id,
                "status": m.status.value,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in shown
        ],
        "jobs_total": len(jobs),
        "jobs_omitted": len(jobs) - len(shown),
        "limit": limit,
        "spend": lab.spend(jobs, alert_usd=spend_alert),
    }
