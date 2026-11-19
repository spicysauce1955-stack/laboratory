"""`lab dashboard` at scale: bounded rows, and metrics files read once rather than every frame.

The stress run measured **4.2 s per frame** at 10k jobs: every frame read every manifest, built a
row for each, re-read each job's whole `metrics.jsonl` to find its latest values, and rendered all
10k rows into a terminal that shows ~50. The table now holds the most recent `limit` jobs (40 by
default, `--limit` to change, `0` for all) and latest-metric values come from a cache keyed by the
file's (inode, size, mtime): unchanged files are not read at all, and a growing file is read only
from where the last frame stopped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from helpers import PYTHON, make_manifest
from typer.testing import CliRunner

from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab
from lab.dashboard import DEFAULT_DASHBOARD_ROWS, LatestMetrics, dashboard_rows
from lab.models import JobState
from lab.store import JobStore


def _lab(tmp_path: Path, n: int) -> Lab:
    home = tmp_path / "runs"
    store = JobStore(home)
    for i in range(n):
        m = make_manifest(f"20261119-1000{i:02d}-abcdef", f"{PYTHON} -c 'print(1)'")
        store.write_manifest(m.model_copy(update={"status": JobState.succeeded}))
    return Lab(backend=LocalBackend(home=home, repo=tmp_path), repo=tmp_path, home=home)


def _log(path: Path, *points: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for p in points:
            f.write(json.dumps(p) + "\n")


def test_rows_are_capped_to_the_most_recent_jobs(tmp_path: Path):
    lab = _lab(tmp_path, 12)

    rows = dashboard_rows(lab, limit=5)

    ids = [r["job_id"] for r in rows]
    assert ids == sorted(lab.store.list_job_ids())[-5:]


def test_the_default_cap(tmp_path: Path):
    lab = _lab(tmp_path, DEFAULT_DASHBOARD_ROWS + 3)

    assert len(dashboard_rows(lab)) == DEFAULT_DASHBOARD_ROWS
    assert len(dashboard_rows(lab, limit=0)) == DEFAULT_DASHBOARD_ROWS + 3


def test_a_sweep_filter_is_capped_too(tmp_path: Path):
    lab = _lab(tmp_path, 6)
    ids = lab.store.list_job_ids()

    rows = dashboard_rows(lab, ids[:4], limit=2)

    assert [r["job_id"] for r in rows] == ids[2:4]


def test_an_unchanged_metrics_file_is_not_read_again(tmp_path: Path, monkeypatch):
    f = tmp_path / "metrics.jsonl"
    _log(f, {"name": "loss", "value": 1.0, "step": 0}, {"name": "acc", "value": 0.1, "step": 0})
    cache = LatestMetrics()
    assert cache.latest(f) == {"loss": (1.0, 0), "acc": (0.1, 0)}

    opened: list[Path] = []
    real_open = Path.open
    monkeypatch.setattr(
        Path, "open", lambda self, *a, **kw: opened.append(self) or real_open(self, *a, **kw)
    )
    for _ in range(3):
        assert cache.latest(f) == {"loss": (1.0, 0), "acc": (0.1, 0)}
    assert opened == []


def test_a_growing_file_is_read_from_where_the_last_frame_stopped(tmp_path: Path, monkeypatch):
    f = tmp_path / "metrics.jsonl"
    _log(f, *({"name": "loss", "value": float(i), "step": i} for i in range(5000)))
    cache = LatestMetrics()
    assert cache.latest(f) == {"loss": (4999.0, 4999)}

    read_bytes: list[int] = []

    class _Spy:
        def __init__(self, fh: Any) -> None:
            self._fh = fh

        def __getattr__(self, name: str) -> Any:
            return getattr(self._fh, name)

        def __enter__(self) -> _Spy:
            return self

        def __exit__(self, *exc: Any) -> None:
            self._fh.close()

        def read(self, size: int = -1) -> Any:
            data = self._fh.read(size)
            read_bytes.append(len(data))
            return data

    real_open = Path.open
    monkeypatch.setattr(Path, "open", lambda self, *a, **kw: _Spy(real_open(self, *a, **kw)))
    _log(f, {"name": "loss", "value": -1.0, "step": 5000}, {"name": "lr", "value": 3, "step": 1})

    assert cache.latest(f) == {"loss": (-1.0, 5000), "lr": (3, 1)}
    assert sum(read_bytes) < 4096 < f.stat().st_size


def test_a_half_written_line_is_picked_up_once_it_is_complete(tmp_path: Path):
    f = tmp_path / "metrics.jsonl"
    _log(f, {"name": "loss", "value": 1.0, "step": 0})
    with f.open("a") as fh:
        fh.write('{"name": "loss", "value": 2.0, "st')
    cache = LatestMetrics()
    assert cache.latest(f) == {"loss": (1.0, 0)}

    with f.open("a") as fh:
        fh.write('ep": 1}\n')

    assert cache.latest(f) == {"loss": (2.0, 1)}


def test_a_replaced_or_truncated_file_is_reread(tmp_path: Path):
    f = tmp_path / "metrics.jsonl"
    _log(f, *({"name": "loss", "value": 1.0, "step": i} for i in range(100)))
    cache = LatestMetrics()
    cache.latest(f)

    f.write_text(json.dumps({"name": "acc", "value": 0.5, "step": 0}) + "\n")

    assert cache.latest(f) == {"acc": (0.5, 0)}


def test_a_missing_file_has_no_metrics(tmp_path: Path):
    assert LatestMetrics().latest(tmp_path / "nope.jsonl") == {}


def test_dashboard_rows_show_the_latest_values_from_the_cache(tmp_path: Path):
    lab = _lab(tmp_path, 1)
    (jid,) = lab.store.list_job_ids()
    f = lab.store.output_dir(jid) / "metrics.jsonl"
    _log(f, {"name": "loss", "value": 0.5, "step": 3}, {"name": "loss", "value": 0.25, "step": 4})
    cache = LatestMetrics()

    (row,) = dashboard_rows(lab, cache=cache)

    assert row["latest_metric"] == "loss=0.25@4"


def test_cli_dashboard_has_a_limit_flag() -> None:
    result = CliRunner().invoke(app, ["dashboard", "--help"])
    assert result.exit_code == 0
    assert "--limit" in result.output


@pytest.mark.parametrize("limit", [-1])
def test_a_negative_limit_is_refused(tmp_path: Path, limit: int):
    with pytest.raises(ValueError, match="limit"):
        dashboard_rows(_lab(tmp_path, 1), limit=limit)
