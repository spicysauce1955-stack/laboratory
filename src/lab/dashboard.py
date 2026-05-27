"""Human dashboard (FR-D3) — a live terminal view of job status + cost + latest metrics.

`dashboard_rows` is a pure builder (unit-tested); `run_dashboard` is the thin rich.Live loop.
The spec allows "a human dashboard (or integrate one)"; this is the built-in terminal one.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.table import Table

    from lab.core import Lab


def _fmt_value(v: Any) -> str:
    """Format a metric value defensively — values may be non-numeric (e.g. a logged label)."""
    return f"{v:.4g}" if isinstance(v, (int, float)) else str(v)


def dashboard_rows(lab: Lab, job_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """One row per job: status, duration, cost (actual, else the estimate), latest metric value."""
    manifests = [lab.manifest(j) for j in job_ids] if job_ids else lab.list_jobs()
    rows: list[dict[str, Any]] = []
    for m in manifests:
        latest = []
        try:
            for name, points in lab.metrics(m.job_id).items():
                if points:
                    latest.append(f"{name}={_fmt_value(points[-1]['value'])}@{points[-1]['step']}")
        except Exception:  # noqa: BLE001 - a dashboard must never crash on one bad job
            pass
        cost = m.cost
        cost_usd = None
        if cost is not None:  # show actual once known, else the up-front estimate (FR-I2)
            cost_usd = cost.actual_usd if cost.actual_usd is not None else cost.estimated_usd
        rows.append(
            {
                "job_id": m.job_id,
                "sweep_id": m.sweep_id or "",
                "state": m.status.value,
                "duration_s": round(cost.duration_seconds, 1)
                if cost and cost.duration_seconds is not None
                else None,
                "cost_usd": cost_usd,
                "latest_metric": ", ".join(latest[:3]),
            }
        )
    return rows


def render_table(rows: list[dict[str, Any]]) -> Table:
    from rich.table import Table

    table = Table(title="lab — jobs", expand=True)
    for col in ("job_id", "sweep", "state", "dur(s)", "cost($)", "latest metric"):
        table.add_column(col, overflow="fold")
    for r in rows:
        table.add_row(
            r["job_id"],
            r["sweep_id"] or "-",
            r["state"],
            "-" if r["duration_s"] is None else str(r["duration_s"]),
            "-" if r["cost_usd"] is None else f"{r['cost_usd']:.4f}",
            r["latest_metric"] or "-",
        )
    return table


def run_dashboard(lab: Lab, job_ids: list[str] | None = None, interval: float = 2.0) -> None:
    """Refresh a live job table every ``interval`` seconds until Ctrl-C (FR-D3)."""
    from rich.live import Live

    with Live(render_table(dashboard_rows(lab, job_ids)), refresh_per_second=4) as live:
        try:
            while True:
                time.sleep(max(0.2, interval))
                live.update(render_table(dashboard_rows(lab, job_ids)))
        except KeyboardInterrupt:
            pass
