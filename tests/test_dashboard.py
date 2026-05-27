from pathlib import Path

from helpers import PYTHON, wait_terminal

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.dashboard import _fmt_value, dashboard_rows, render_table
from lab.manifest import repo_root
from lab.models import JobSpec, JobState


def test_dashboard_rows(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    jid = lab.submit(JobSpec(command=f"{PYTHON} experiments/example_capacity.py", seed=2))
    assert wait_terminal(backend, jid) == JobState.succeeded

    rows = dashboard_rows(lab)
    assert len(rows) == 1
    r = rows[0]
    assert r["job_id"] == jid and r["state"] == "succeeded"
    assert r["cost_usd"] is not None  # cost surfaced
    assert "demo_metric" in r["latest_metric"]  # live metric surfaced


def test_fmt_value_handles_non_numeric():
    # must not crash on non-numeric/None metric values (would otherwise blank the whole column)
    assert _fmt_value(0.5) == "0.5"
    assert _fmt_value(10) == "10"
    assert _fmt_value("label") == "label"
    assert _fmt_value(None) == "None"


def test_render_table_smoke():
    rows = [
        {
            "job_id": "j1",
            "sweep_id": "",
            "state": "running",
            "duration_s": 1.2,
            "cost_usd": 0.01,
            "latest_metric": "loss=0.5@3",
        }
    ]
    assert render_table(rows) is not None  # builds a rich Table without error
