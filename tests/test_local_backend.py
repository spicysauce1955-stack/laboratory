import time
from pathlib import Path

from helpers import PYTHON, make_manifest, wait_terminal

from lab.backends.local import LocalBackend
from lab.models import JobState
from lab.store import JobStore


def _submit(tmp_path: Path, job_id: str, command: str, **kw) -> tuple[LocalBackend, JobStore]:
    store = JobStore(tmp_path)
    store.create(make_manifest(job_id, command, **kw))
    backend = LocalBackend(home=tmp_path)
    backend.submit(store.read_manifest(job_id))
    return backend, store


def test_submit_runs_to_success_with_artifacts(tmp_path: Path):
    backend, store = _submit(tmp_path, "j1", f"{PYTHON} experiments/example_capacity.py", seed=3)
    assert wait_terminal(backend, "j1") == JobState.succeeded
    arts = backend.collect_artifacts("j1", str(store.job_dir("j1")))
    names = {a.name for a in arts}
    assert {"result.json", "metrics.jsonl"} <= names
    assert all(a.sha256 and a.bytes > 0 for a in arts)
    assert list(backend.tail_logs("j1")) is not None  # logs file exists


def test_cancel_running_job(tmp_path: Path):
    backend, _ = _submit(tmp_path, "j2", f'{PYTHON} -c "import time; time.sleep(30)"')
    for _ in range(50):  # wait until actually running
        if backend.status("j2") == JobState.running:
            break
        time.sleep(0.1)
    assert backend.cancel("j2") == JobState.cancelled
    assert backend.status("j2") == JobState.cancelled
