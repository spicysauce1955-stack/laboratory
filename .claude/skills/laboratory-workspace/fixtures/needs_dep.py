"""Eval fixture: imports a package that is NOT in the base locked env (humanize).

Run bare, this raises ModuleNotFoundError and the job fails. Run with the dep layered in
(lab `with_pkg=["humanize"]` / CLI `--with humanize`), the import resolves and the job succeeds.
That makes it a clean test of the per-job extra-dependency mechanism.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    import humanize  # not in the base env — only present if layered in per-job

    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    pretty = humanize.intcomma(1234567)
    (run_dir / "result.json").write_text(json.dumps({"humanized": pretty, "ok": True}))
    with (run_dir / "metrics.jsonl").open("w", buffering=1) as f:
        f.write(json.dumps({"name": "ok", "value": 1.0, "step": 0}) + "\n")
    print(f"humanize works: {pretty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
