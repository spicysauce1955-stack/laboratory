import json
from pathlib import Path

from helpers import PYTHON, make_manifest

from lab.models import JobState
from lab.runner import run_job
from lab.store import JobStore


def _prep(tmp_path: Path, job_id: str, command: str, **kw) -> JobStore:
    store = JobStore(tmp_path)
    store.create(make_manifest(job_id, command, **kw))
    return store


def test_success_injects_env_and_records_state(tmp_path: Path):
    # The example experiment reads $LAB_SEED and writes into $LAB_RUN_DIR (Experiment Contract).
    store = _prep(tmp_path, "s1", f"{PYTHON} experiments/example_capacity.py", seed=7)
    rc = run_job(store.job_dir("s1"))
    assert rc == 0
    m = store.read_manifest("s1")
    assert m.status == JobState.succeeded
    assert m.exit_code == 0
    assert m.started_at is not None and m.ended_at is not None
    result = json.loads((store.output_dir("s1") / "result.json").read_text())
    assert result["seed"] == 7  # LAB_SEED reached the experiment
    assert (store.output_dir("s1") / "metrics.jsonl").exists()


def test_failure_records_exit_code(tmp_path: Path):
    store = _prep(tmp_path, "f1", f'{PYTHON} -c "import sys; sys.exit(3)"')
    rc = run_job(store.job_dir("f1"))
    assert rc == 3
    m = store.read_manifest("f1")
    assert m.status == JobState.failed
    assert m.exit_code == 3


def test_timeout_terminates(tmp_path: Path):
    store = _prep(tmp_path, "t1", f'{PYTHON} -c "import time; time.sleep(30)"', timeout="1s")
    run_job(store.job_dir("t1"))
    m = store.read_manifest("t1")
    assert m.status == JobState.timed_out
    assert m.end_reason == "wall-clock timeout"
