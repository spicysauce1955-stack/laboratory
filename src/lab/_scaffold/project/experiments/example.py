"""Example entrypoint honouring the lab's Experiment Contract.

Run it through the lab, not directly:

    uv run lab submit -c "python experiments/example.py" --seed 0 -- steps=5

The contract, in four lines of code below:
  - read the seed and output dir the lab injects (`$LAB_SEED`, `$LAB_RUN_DIR`),
  - declare which config keys you consume, so a typo fails the job instead of silently
    running a different experiment (`get_overrides`),
  - log an incremental metric series so you can watch and kill early (`log_metric`),
  - write every output under `$LAB_RUN_DIR`, and exit non-zero on failure.
"""

from __future__ import annotations

import os
from pathlib import Path

from lab.experiment import get_overrides
from lab.metrics import log_metric


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get("LAB_SEED", "0"))

    overrides = get_overrides(known={"steps"})
    steps = int(overrides.get("steps", "10"))

    for step in range(steps):
        log_metric("score", value=(step + seed) / max(steps, 1), step=step)

    (run_dir / "result.txt").write_text(f"seed={seed} steps={steps}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
