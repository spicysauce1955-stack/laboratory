import json
from pathlib import Path

from helpers import PYTHON, wait_terminal

from lab.backends.local import LocalBackend
from lab.core import Lab
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
