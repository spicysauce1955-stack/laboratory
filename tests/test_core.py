import json
from pathlib import Path

import pytest
from helpers import PYTHON, wait_terminal

from lab.backends.local import LocalBackend
from lab.core import Lab, LabError, expand_grid
from lab.manifest import repo_root
from lab.models import JobSpec, JobState


def test_end_to_end_submit_and_fetch(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)

    job_id = lab.submit(
        JobSpec(code_ref="HEAD", command=f"{PYTHON} experiments/example_capacity.py", seed=11)
    )
    assert wait_terminal(backend, job_id) == JobState.succeeded

    m = lab.manifest(job_id)
    assert len(m.code.git_commit) == 40  # commit pinned (FR-B1)
    assert m.env.uv_lock_sha256 and m.env.python_version  # env recorded (FR-B2)
    assert m.run.seed == 11  # seed recorded (FR-B4)

    arts = lab.fetch_artifacts(job_id)
    assert "result.json" in {a.name for a in arts}
    result = json.loads((tmp_path / job_id / "output" / "result.json").read_text())
    assert result["seed"] == 11

    assert [j.job_id for j in lab.list_jobs()] == [job_id]


def test_metrics_query_incremental(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    job_id = lab.submit(
        JobSpec(code_ref="HEAD", command=f"{PYTHON} experiments/example_capacity.py", seed=1)
    )
    assert wait_terminal(backend, job_id) == JobState.succeeded

    series = lab.metrics(job_id)
    assert set(series) == {"demo_metric"}
    assert [p["step"] for p in series["demo_metric"]] == list(range(10))

    incremental = lab.metrics(job_id, since_step=4)  # the early-kill "what's new?" query
    assert [p["step"] for p in incremental["demo_metric"]] == [5, 6, 7, 8, 9]


def test_expand_grid():
    assert expand_grid({}) == [{}]
    assert expand_grid({"a": [1, 2], "b": [9]}) == [{"a": 1, "b": 9}, {"a": 2, "b": 9}]
    assert len(expand_grid({"a": [1, 2], "b": [3, 4, 5]})) == 6  # cartesian product


def test_sweep_local(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)

    sweep_id, job_ids = lab.sweep(
        f"{PYTHON} experiments/example_capacity.py", {"K": [1, 2], "alpha": [0.5]}
    )
    assert sweep_id.startswith("sweep-")
    assert len(job_ids) == 2  # 2 x 1 grid

    for jid in job_ids:
        assert wait_terminal(backend, jid) == JobState.succeeded
        m = lab.manifest(jid)
        assert m.sweep_id == sweep_id  # shared sweep id
        assert m.run.resolved_config["alpha"] == 0.5
        assert "K=" in m.run.entrypoint_command  # override appended to the command

    ks = sorted(lab.manifest(j).run.resolved_config["K"] for j in job_ids)
    assert ks == [1, 2]  # the grid actually varied K across jobs


def test_sweep_quotes_values(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    # a value with a space + shell metachars must be quoted into one safe token (no injection)
    _, job_ids = lab.sweep(f"{PYTHON} experiments/example_capacity.py", {"x": ["a b; echo hi"]})
    cmd = lab.manifest(job_ids[0]).run.entrypoint_command
    assert "'x=a b; echo hi'" in cmd
    assert wait_terminal(backend, job_ids[0]) == JobState.succeeded


def test_sweep_seed_from_grid(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    _, job_ids = lab.sweep(f"{PYTHON} experiments/example_capacity.py", {"seed": [1, 2]})
    assert sorted(lab.manifest(j).run.seed for j in job_ids) == [1, 2]  # seed varies per point


def test_sweep_job_cap(tmp_path: Path):
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    with pytest.raises(LabError):
        lab.sweep("python x.py", {"a": list(range(20))}, max_jobs=5)
