"""Local metric series — the credential-free P0 tracker (FR-D2).

Convention: an experiment writes incremental metric points as JSON lines to
``$LAB_RUN_DIR/metrics.jsonl`` (one object per line: ``name``/``value``/``step``[/``wall_time``]).
The lab reads them **live** — tolerating a half-written trailing line — which powers the
"are we on track? cancel early" loop. ``log_metric`` is an optional helper experiments MAY use;
writing the JSONL directly also works (no lab import required, keeping the Experiment Contract
coupling-free). MLflow can slot in behind this same read interface later.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

METRICS_FILE = "metrics.jsonl"


def log_metric(name: str, value: float, step: int, *, run_dir: str | Path | None = None) -> None:
    """Append one metric point (optional convenience for experiments)."""
    base = Path(run_dir) if run_dir is not None else Path(os.environ.get("LAB_RUN_DIR", "."))
    base.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"name": name, "value": float(value), "step": int(step), "wall_time": time.time()}
    )
    with (base / METRICS_FILE).open("a") as f:
        f.write(line + "\n")


def read_points(
    metrics_file: str | Path,
    names: Iterable[str] | None = None,
    since_step: int | None = None,
) -> list[dict[str, Any]]:
    """Parse metric points, filtered by ``names`` and ``since_step`` (exclusive).

    Tolerant of a malformed final line so it can be read while the file is being written.
    """
    p = Path(metrics_file)
    if not p.exists():
        return []
    name_set = set(names) if names is not None else None
    points: list[dict[str, Any]] = []
    for raw in p.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue  # half-written trailing line during a live read
        name, step = obj.get("name"), obj.get("step")
        if name is None or step is None:
            continue
        if name_set is not None and name not in name_set:
            continue
        if since_step is not None and step <= since_step:
            continue
        points.append(
            {"name": name, "value": obj.get("value"), "step": step, "wall_time": obj.get("wall_time")}
        )
    return points


def final_values(points: list[dict[str, Any]]) -> dict[str, float]:
    """Reduce flat points to the last value per metric series (by max ``step``).

    The durable per-job baseline a reproducibility re-run is judged against (FR-B4): the manifest
    snapshots this at terminal time so an old run stays confirmable even after its run dir / object
    store copy is pruned.
    """
    best: dict[str, tuple[int, float]] = {}
    for pt in points:
        name, step, value = pt["name"], pt["step"], pt["value"]
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue  # a non-numeric point can't be a baseline; skip it, never crash finalize
        if name not in best or step >= best[name][0]:
            best[name] = (step, numeric)
    return {name: value for name, (_, value) in best.items()}


def snapshot_final_metrics(run_dir: str | Path) -> dict[str, float]:
    """Read ``<run_dir>/metrics.jsonl`` and reduce to the final value per series (empty if none)."""
    return final_values(read_points(Path(run_dir) / METRICS_FILE))


def group_series(points: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group flat points into ``{name: [{step, value, wall_time}, ...]}`` (spec §9)."""
    series: dict[str, list[dict[str, Any]]] = {}
    for pt in points:
        series.setdefault(pt["name"], []).append(
            {"step": pt["step"], "value": pt["value"], "wall_time": pt["wall_time"]}
        )
    return series
