"""Eval fixture: logs a couple of metric points, then crashes with a non-zero exit (EC-5).

Used to test how the lab surfaces a failed run: status=failed, a non-zero exit_code, an end_reason,
and the partial metrics it managed to log before dying.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.jsonl").open("w", buffering=1) as f:
        for step in range(3):
            f.write(json.dumps({"name": "loss", "value": 0.5 - 0.1 * step, "step": step}) + "\n")
    print("logged 3 steps, now failing on purpose", file=sys.stderr)
    raise SystemExit(3)  # non-zero -> the lab must mark this failed (EC-5)


if __name__ == "__main__":
    main()
