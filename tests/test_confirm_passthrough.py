"""`lab confirm` re-runs with the original's own opt-outs, and doesn't idle on a local re-run."""

from __future__ import annotations

import time
from pathlib import Path

from core_fixtures import local_lab, make_repo
from helpers import PYTHON, wait_terminal

from lab.models import JobSpec, JobState

# Reports only `a` as consumed, so an `extra=` override is "unconsumed" — the shape a job
# submitted with --allow-unknown-config is allowed to have.
_EXP = """\
import json, os, sys
from pathlib import Path
out = Path(os.environ["LAB_RUN_DIR"])
ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
(out / "effective_config.json").write_text(json.dumps({"a": ov.get("a")}))
with (out / "metrics.jsonl").open("w") as f:
    f.write(json.dumps({"name": "m", "value": float(ov.get("a", 0)), "step": 0}) + "\\n")
"""


def test_confirm_inherits_allow_unknown_config_and_polls_fast(tmp_path: Path, monkeypatch):
    repo = make_repo(tmp_path, {"exp.py": _EXP})
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    orig = lab.submit(
        JobSpec(
            command=f"{PYTHON} exp.py a=3 extra=1",
            config={"a": "3", "extra": "1"},
            allow_unknown_config=True,
        )
    )
    assert wait_terminal(lab.backend, orig) is JobState.succeeded
    assert lab.manifest(orig).unconsumed_config == ["extra"]

    t0 = time.monotonic()
    result = lab.confirm(orig, timeout=60)
    elapsed = time.monotonic() - t0

    assert result["verdict"] == "match", result
    rerun = lab.manifest(result["confirm_id"])
    assert rerun.run.allow_unknown_config is True
    # A local re-run lands in well under a second; the old fixed 10s poll made every local
    # confirm sit idle for the whole first interval.
    assert elapsed < 6.0, elapsed
