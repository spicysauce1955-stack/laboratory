"""Human dashboard (FR-D3) — a live terminal view of job status + cost + latest metrics.

`dashboard_rows` is a pure builder (unit-tested); `run_dashboard` is the thin rich.Live loop.
The spec allows "a human dashboard (or integrate one)"; this is the built-in terminal one.

Built for a project with thousands of jobs: the 2026-11 stress run measured 4.2 s per frame at
10k jobs, because every frame read every manifest, re-read every job's whole ``metrics.jsonl``
and rendered 10k rows into a terminal that shows ~50. A frame now reads only the most recent
``limit`` manifests, and :class:`LatestMetrics` reads a metrics file once, then only what was
appended since the last frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lab.metrics import METRICS_FILE, last_line_boundary, parse_point, series_view

if TYPE_CHECKING:
    from rich.table import Table

    from lab.core import Lab

#: Rows the table shows when the caller does not say (``limit=0`` shows every job).
DEFAULT_DASHBOARD_ROWS = 40


def _fmt_value(v: Any) -> str:
    """Format a metric value defensively — values may be non-numeric (e.g. a logged label)."""
    return f"{v:.4g}" if isinstance(v, (int, float)) else str(v)


@dataclass
class _Tail:
    ino: int
    size: int
    mtime_ns: int
    offset: int  # a line boundary: everything before it has been folded into `latest`
    latest: dict[str, tuple[Any, Any]] = field(default_factory=dict)


class LatestMetrics:
    """The newest ``(value, step)`` of each series in a ``metrics.jsonl``, across frames.

    Cached by the file's (inode, size, mtime): an unchanged file is not opened at all, a file that
    grew is read from the last line boundary already folded in, and one that shrank or was
    replaced is read afresh — from the end, via :func:`lab.metrics.series_view`, so even the first
    frame does not parse a long run line by line.
    """

    def __init__(self) -> None:
        self._files: dict[Path, _Tail] = {}

    def latest(self, path: Path) -> dict[str, tuple[Any, Any]]:
        try:
            st = path.stat()
        except OSError:
            self._files.pop(path, None)
            return {}
        tail = self._files.get(path)
        if tail is not None and (tail.ino, tail.size, tail.mtime_ns) == (
            st.st_ino, st.st_size, st.st_mtime_ns
        ):
            return tail.latest
        if tail is None or tail.ino != st.st_ino or st.st_size < tail.offset:
            tail = self._fresh(path, st.st_size)
        else:
            self._grow(path, tail, st.st_size)
        tail.ino, tail.size, tail.mtime_ns = st.st_ino, st.st_size, st.st_mtime_ns
        self._files[path] = tail
        return tail.latest

    @staticmethod
    def _fresh(path: Path, size: int) -> _Tail:
        """First sight: the newest point per series, read from the end; the offset is the last
        line boundary at or before ``size``, so anything after it is re-read next frame (folding
        a line in twice is harmless: the newest value wins either way)."""
        view = series_view(path, max_points=1)
        latest = {name: (pts[-1]["value"], pts[-1]["step"]) for name, pts in view["series"].items()}
        offset = last_line_boundary(path, size)
        return _Tail(ino=0, size=0, mtime_ns=0, offset=offset, latest=latest)

    @staticmethod
    def _grow(path: Path, tail: _Tail, size: int) -> None:
        """Fold in the whole lines appended since ``tail.offset``; a half-written last line waits
        for the next frame."""
        try:
            with path.open("rb") as fh:
                fh.seek(tail.offset)
                data = fh.read(max(0, size - tail.offset))
        except OSError:
            return
        cut = data.rfind(b"\n")
        if cut < 0:
            return
        for line in data[: cut + 1].split(b"\n"):
            pt = parse_point(line)
            if pt is not None:
                tail.latest[pt["name"]] = (pt["value"], pt["step"])
        tail.offset += cut + 1


def _recent_manifests(
    lab: Lab, job_ids: list[str] | None, limit: int
) -> tuple[list[Any], int]:
    """The most recent ``limit`` jobs' manifests and how many jobs there are in all. Job ids
    begin with their creation time, so the sorted id list is chronological and only the shown
    manifests are read; unreadable ones are skipped."""
    ids = sorted(job_ids) if job_ids is not None else lab.store.list_job_ids()
    total = len(ids)
    if limit:
        ids = ids[-limit:]
    out = []
    for job_id in ids:
        try:
            out.append(lab.store.read_manifest(job_id))
        except (OSError, ValueError):  # a dashboard must never crash on one bad job
            continue
    return out, total


def dashboard_rows(
    lab: Lab,
    job_ids: list[str] | None = None,
    *,
    limit: int = DEFAULT_DASHBOARD_ROWS,
    cache: LatestMetrics | None = None,
) -> list[dict[str, Any]]:
    """One row per job — the most recent ``limit`` (0 = all) — with status, duration, cost
    (actual, else the estimate) and latest metric values. Pass the same ``cache`` every frame."""
    return _frame_rows(lab, job_ids, limit=limit, cache=cache)[0]


def _frame_rows(
    lab: Lab,
    job_ids: list[str] | None,
    *,
    limit: int,
    cache: LatestMetrics | None,
) -> tuple[list[dict[str, Any]], int]:
    if limit < 0:
        raise ValueError(f"limit must be >= 0 (0 = every job), got {limit}")
    metrics_cache = cache if cache is not None else LatestMetrics()
    manifests, total = _recent_manifests(lab, job_ids, limit)
    rows: list[dict[str, Any]] = []
    for m in manifests:
        latest = []
        try:
            path = lab.store.output_dir(m.job_id) / METRICS_FILE
            for name, (value, step) in metrics_cache.latest(path).items():
                latest.append(f"{name}={_fmt_value(value)}@{step}")
        except Exception:  # noqa: BLE001 - a dashboard must never crash on one bad job
            pass
        cost = m.cost
        cost_usd = None
        if cost is not None:  # show actual once known, else the up-front estimate (FR-I2)
            cost_usd = cost.actual_usd if cost.actual_usd is not None else cost.estimated_usd
        # teardown column: surfaced loudly because a "LEAK" here costs money (FR-C2 leak detection)
        td = m.teardown_status
        # Three states, not two (R10). Anything unrecognised shows as "?" rather than blank:
        # a value this version does not know about must not read as "nothing to see here".
        if td == "failed":
            teardown = "LEAK"
        elif td == "succeeded":
            teardown = "ok"
        elif td is None:
            teardown = ""
        else:
            teardown = "UNKNOWN?"
        rows.append(
            {
                "job_id": m.job_id,
                "sweep_id": m.sweep_id or "",
                "state": m.status.value,
                "duration_s": round(cost.duration_seconds, 1)
                if cost and cost.duration_seconds is not None
                else None,
                "cost_usd": cost_usd,
                "teardown": teardown,
                "latest_metric": ", ".join(latest[:3]),
            }
        )
    return rows, total


def render_table(rows: list[dict[str, Any]], caption: str | None = None) -> Table:
    from rich.table import Table

    table = Table(title="lab — jobs", caption=caption, expand=True)
    for col in ("job_id", "sweep", "state", "dur(s)", "cost($)", "teardown", "latest metric"):
        table.add_column(col, overflow="fold")
    for r in rows:
        table.add_row(
            r["job_id"],
            r["sweep_id"] or "-",
            r["state"],
            "-" if r["duration_s"] is None else str(r["duration_s"]),
            "-" if r["cost_usd"] is None else f"{r['cost_usd']:.4f}",
            r["teardown"] or "-",
            r["latest_metric"] or "-",
        )
    return table


def run_dashboard(
    lab: Lab,
    job_ids: list[str] | None = None,
    interval: float = 2.0,
    limit: int = DEFAULT_DASHBOARD_ROWS,
) -> None:
    """Refresh a live job table every ``interval`` seconds until Ctrl-C (FR-D3)."""
    from rich.live import Live

    cache = LatestMetrics()

    def frame() -> Table:
        rows, total = _frame_rows(lab, job_ids, limit=limit, cache=cache)
        caption = (
            f"most recent {len(rows)} of {total} jobs — --limit N to change, 0 for all"
            if limit and total > len(rows)
            else None
        )
        return render_table(rows, caption)

    with Live(frame(), refresh_per_second=4) as live:
        try:
            while True:
                time.sleep(max(0.2, interval))
                live.update(frame())
        except KeyboardInterrupt:
            pass
