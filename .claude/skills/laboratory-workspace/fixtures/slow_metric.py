"""Eval fixture: a slow, observable run that DIVERGES so an agent has a reason to kill it early.

Contract-compliant (spec §7): reads $LAB_RUN_DIR / $LAB_SEED, writes a live metric series to
metrics.jsonl (flushed every step so the lab can read it incrementally), exits non-zero never on
its own — it's meant to be cancelled mid-flight. Steps/sleep are tunable via env for fast evals.

    LAB_RUN_DIR=runs/x uv run python slow_metric.py
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
    steps = int(os.environ.get("EVAL_STEPS", "24"))
    sleep_s = float(os.environ.get("EVAL_SLEEP", "1.0"))

    with (run_dir / "metrics.jsonl").open("w", buffering=1) as f:  # line-buffered -> live reads
        for step in range(steps):
            # loss falls for the first few steps, then clearly diverges upward — a kill signal.
            loss = 0.5 - 0.04 * step if step < 5 else 0.30 + 0.08 * (step - 5)
            f.write(json.dumps({"name": "loss", "value": float(loss), "step": step}) + "\n")
            f.flush()
            os.fsync(f.fileno())
            time.sleep(sleep_s)

    (run_dir / "result.json").write_text(json.dumps({"completed": True, "steps": steps}))
    print(f"completed all {steps} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
