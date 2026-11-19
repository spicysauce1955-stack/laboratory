"""Manifest read-modify-write is serialised per job, never regresses a terminal job, and the
dead-runner finalization only fires on a job that is still non-terminal under the lock."""

from __future__ import annotations

import threading
from pathlib import Path

from helpers import make_manifest

import lab.backends.local as local_mod
from lab.backends.local import LocalBackend
from lab.models import ArtifactRecord, JobState
from lab.store import JobStore


def _running(store: JobStore, job_id: str) -> None:
    store.create(make_manifest(job_id, "python x.py").model_copy(
        update={"status": JobState.running}
    ))


def test_a_concurrent_writer_cannot_be_overwritten_by_a_stale_read(tmp_path: Path, monkeypatch):
    """`lab fetch` (collect_artifacts) reads a running manifest; the supervisor then records
    terminal + teardown_status before fetch writes. Fetch's write must not erase them."""
    store = JobStore(tmp_path)
    _running(store, "j")
    real_read = JobStore.read_manifest
    main = threading.current_thread()
    supervisor: dict[str, threading.Thread] = {}

    def racing_read(self, job_id):
        m = real_read(self, job_id)
        if threading.current_thread() is main and "t" not in supervisor:
            supervisor["t"] = t = threading.Thread(
                target=lambda: JobStore(tmp_path).update_manifest(
                    job_id, status=JobState.succeeded, exit_code=0, teardown_status="failed"
                )
            )
            t.start()
            t.join(timeout=0.5)  # unlocked: it lands now, inside our read-modify-write
        return m

    monkeypatch.setattr(JobStore, "read_manifest", racing_read)
    store.update_manifest("j", artifacts=[
        ArtifactRecord(name="a.csv", path="/x/a.csv", sha256="0" * 64, bytes=1)
    ])
    supervisor["t"].join(timeout=10)
    # Restore only what this test patched: `monkeypatch.undo()` also reverts conftest's isolation
    # (ledger/index dirs, the cloud-client guards) for the rest of the test.
    monkeypatch.setattr(JobStore, "read_manifest", real_read)

    final = store.read_manifest("j")
    assert final.status is JobState.succeeded
    assert final.teardown_status == "failed"  # the FR-C2 alarm survives
    assert [a.name for a in final.artifacts] == ["a.csv"]


def test_a_terminal_manifest_never_moves_back_to_non_terminal(tmp_path: Path):
    store = JobStore(tmp_path)
    store.create(make_manifest("j", "python x.py").model_copy(
        update={"status": JobState.cancelled, "end_reason": "cancelled by user"}
    ))
    m = store.update_manifest("j", status=JobState.running, exit_code=0)
    assert m.status is JobState.cancelled
    assert m.exit_code == 0  # the rest of the write still lands
    assert store.read_manifest("j").status is JobState.cancelled


def test_only_if_active_is_a_no_op_on_a_terminal_job(tmp_path: Path):
    store = JobStore(tmp_path)
    store.create(make_manifest("j", "python x.py").model_copy(
        update={"status": JobState.succeeded, "end_reason": "completed"}
    ))
    m = store.update_manifest(
        "j", only_if_active=True, status=JobState.failed, end_reason="runner exited"
    )
    assert m.status is JobState.succeeded and m.end_reason == "completed"


def test_dead_runner_finalization_cannot_overwrite_a_real_success(tmp_path: Path, monkeypatch):
    """status() reads `running`; the runner records `succeeded` and exits; status()'s pid check
    then sees it dead. The success stands."""
    store = JobStore(tmp_path)
    _running(store, "j")
    store.write_runtime("j", runner_pid=999999, runner_start_time=None)

    def runner_finished_just_now(pid, start_time=None):
        JobStore(tmp_path).update_manifest(
            "j", status=JobState.succeeded, exit_code=0, end_reason="completed"
        )
        return False

    monkeypatch.setattr(local_mod, "pid_alive", runner_finished_just_now)
    assert LocalBackend(tmp_path).status("j") is JobState.succeeded
    m = store.read_manifest("j")
    assert m.status is JobState.succeeded and m.end_reason == "completed"


def test_the_lock_file_is_hidden_and_outside_output(tmp_path: Path):
    store = JobStore(tmp_path)
    _running(store, "j")
    store.update_manifest("j", end_reason="x")
    lock = store.manifest_lock_path("j")
    assert lock.exists() and lock.name.startswith(".")  # hidden, like the other sentinels
    assert lock.parent == store.job_dir("j")  # never inside output/ (collected and exported)
    assert not any(p.name.endswith(".lock") for p in store.output_dir("j").rglob("*"))
