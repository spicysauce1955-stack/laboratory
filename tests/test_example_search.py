"""experiments/example_search.py: lab.search inside a real entrypoint, shaped like a sweep shard."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lab.aggregate import merge_seed_rows

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "example_search.py"


def _run(run_dir: Path, *args: str) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env={"PATH": "/usr/bin:/bin", "LAB_RUN_DIR": str(run_dir), "LAB_SEED": "0"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("batch", ["0", "5"])
def test_every_seed_gets_a_summary_row_and_hits_are_logged(tmp_path: Path, batch: str):
    out = _run(tmp_path, "seeds=0,1,2", "n=6", f"batch={batch}")

    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert [r["seed"] for r in rows] == ["0", "1", "2"]
    assert all(int(r["tested"]) == 36 for r in rows)
    assert sum(int(r["hits"]) for r in rows) == out["hits"]

    hits = [json.loads(line) for line in (tmp_path / "hits.jsonl").read_text().splitlines()]
    assert len(hits) == out["hits"] and all({"seed", "gain", "tau", "peak"} <= set(h) for h in hits)
    assert set(json.loads((tmp_path / "boundary.json").read_text())) == {"0", "1", "2"}
    assert json.loads((tmp_path / "effective_config.json").read_text())


def test_batched_and_unbatched_runs_agree(tmp_path: Path):
    _run(tmp_path / "one", "seeds=3", "n=7")
    _run(tmp_path / "many", "seeds=3", "n=7", "batch=4")
    assert (tmp_path / "one" / "hits.jsonl").read_text() == (
        tmp_path / "many" / "hits.jsonl"
    ).read_text()


def test_its_summary_is_what_sweep_aggregate_counts_as_present(tmp_path: Path):
    _run(tmp_path, "seeds=4,5", "n=4")
    _, present, partial = merge_seed_rows(
        [((tmp_path / "results.csv").read_text(), "succeeded")], "seed"
    )
    assert present == [4, 5] and partial == []
