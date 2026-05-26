from pathlib import Path

from helpers import make_manifest

from lab.models import JobState
from lab.store import JobStore


def test_create_and_roundtrip(tmp_path: Path):
    store = JobStore(tmp_path)
    d = store.create(make_manifest("job-1", "echo hi"))
    assert (d / "output").is_dir()
    assert store.logs_path("job-1").exists()
    got = store.read_manifest("job-1")
    assert got.job_id == "job-1"
    assert got.status == JobState.queued
    assert store.list_job_ids() == ["job-1"]


def test_update_manifest(tmp_path: Path):
    store = JobStore(tmp_path)
    store.create(make_manifest("job-2", "echo hi"))
    store.update_manifest("job-2", status=JobState.running)
    assert store.read_manifest("job-2").status == JobState.running


def test_runtime_merge(tmp_path: Path):
    store = JobStore(tmp_path)
    store.create(make_manifest("job-3", "echo hi"))
    store.write_runtime("job-3", runner_pid=111)
    store.write_runtime("job-3", command_pgid=222)
    assert store.read_runtime("job-3") == {"runner_pid": 111, "command_pgid": 222}
