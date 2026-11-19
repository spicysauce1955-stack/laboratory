"""Contract fixture for ``lab.search``: which (gain, tau) make a leaky integrator spike, per seed.

Not science — a smoke test for the search helper inside a real lab job, shaped the way a sharded
sweep uses it: the shard's seeds are a grid axis and ``summary_axis="seed"`` writes one
``seed,tested,hits`` row per seed, so ``sweep-aggregate`` counts zero-hit seeds as present.

Overrides (all optional):
    seeds=0,1   the shard's seeds (a sharded sweep injects this; default $LAB_SEED)
    n=40        grid points per axis
    batch=0     0 = one combination per call, otherwise grid_search's batch_size
    sleep=0     seconds per evaluated combination, to make a job long enough to watch or kill

Outputs under $LAB_RUN_DIR: hits.jsonl (passing combinations with their peak), results.csv
(per-seed counts), metrics.jsonl (search progress) and boundary.json (find_boundary per seed).

Run standalone:
    LAB_RUN_DIR=runs/local-dev LAB_SEED=0 uv run python experiments/example_search.py n=10
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from lab.experiment import get_overrides
from lab.search import crosses_threshold, find_boundary, grid_search


def trace(seed: int, gain: float, tau: float, steps: int = 200) -> np.ndarray:
    """Membrane potential of a leaky integrator driven by noisy input."""
    rng = np.random.default_rng(seed)
    drive = gain * (0.5 + 0.1 * rng.standard_normal(steps))
    v = np.zeros(steps)
    for t in range(1, steps):
        v[t] = v[t - 1] + (-v[t - 1] + drive[t]) / tau
    return v


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    o = get_overrides(known={"seeds", "n", "batch", "sleep"}, run_dir=run_dir)
    seeds = [int(s) for s in str(o.get("seeds", os.environ.get("LAB_SEED", "0"))).split(",")]
    n = int(o.get("n", "40"))
    batch = int(o.get("batch", "0")) or None
    pause = float(o.get("sleep", "0"))

    def one(seed: int, gain: float, tau: float) -> dict[str, Any]:
        if pause:
            time.sleep(pause)
        v = trace(int(seed), float(gain), float(tau))
        return {"hit": crosses_threshold(v, 1.0), "peak": float(v.max())}

    def evaluate(seed: Any, gain: Any, tau: Any) -> dict[str, Any]:
        if batch is None:
            return one(seed, gain, tau)
        answers = [one(s, g, t) for s, g, t in zip(seed, gain, tau, strict=True)]
        return {"hit": [a["hit"] for a in answers], "peak": [a["peak"] for a in answers]}

    grid = {
        "seed": seeds,
        "gain": [round(float(x), 6) for x in np.linspace(0.5, 4.0, n)],
        "tau": [round(float(x), 6) for x in np.linspace(5.0, 40.0, n)],
    }
    result = grid_search(evaluate, grid, batch_size=batch, run_dir=run_dir, summary_axis="seed")

    bounds: dict[str, float] = {}
    for s in seeds:

        def passes(gain: float, s: int = s) -> bool:
            return crosses_threshold(trace(s, gain, 20.0), 1.0)

        bounds[str(s)] = find_boundary(passes, "gain", lo=0.5, hi=4.0, tol=1e-3)
    (run_dir / "boundary.json").write_text(json.dumps(bounds))
    print(json.dumps({"tested": result.tested, "hits": result.hits, "boundary": bounds}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
