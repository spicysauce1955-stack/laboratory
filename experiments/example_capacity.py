"""Minimal Experiment-Contract-compliant example (spec §7, EC-1..6).

Not a real tempotron experiment — a smoke test for the lab's run loop. It:
  - reads the seed + output dir the lab injects via env (EC-2, FR-C4),
  - writes ALL outputs under ``$LAB_RUN_DIR`` (EC-3),
  - logs an incremental metric series (EC-4) — here to a JSONL file; the lab will
    swap in MLflow's ``log_metric`` once the tracker is wired,
  - exits non-zero on failure (EC-5).

Run standalone:
    LAB_RUN_DIR=runs/local-dev LAB_SEED=0 uv run python experiments/example_capacity.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get("LAB_SEED", "0"))

    import numpy as np

    rng = np.random.default_rng(seed)

    # Incremental metric series (EC-4) — observable live by tailing this file.
    with (run_dir / "metrics.jsonl").open("w") as f:
        for step in range(10):
            point = {
                "name": "demo_metric",
                "value": float(rng.random()),
                "step": step,
                "wall_time": time.time(),
            }
            f.write(json.dumps(point) + "\n")

    # An artifact (EC-3).
    (run_dir / "result.json").write_text(json.dumps({"seed": seed, "ok": True}))
    print(f"wrote artifacts to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
