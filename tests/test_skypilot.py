from pathlib import Path

import pytest
from helpers import make_manifest

import lab.backends.skypilot as skypilot_mod
from lab.backends.skypilot import (
    TIMEOUT_SENTINEL,
    build_run_script,
    build_setup_script,
    build_task,
    cluster_name_for,
    map_job_status,
    promote_timeout,
    robust_teardown,
    tear_down_and_record,
)
from lab.models import JobState
from lab.store import JobStore


def test_map_job_status():
    assert map_job_status("SUCCEEDED") == JobState.succeeded
    assert map_job_status("RUNNING") == JobState.running
    assert map_job_status("PENDING") == JobState.queued
    assert map_job_status("SETTING_UP") == JobState.queued
    assert map_job_status("FAILED_SETUP") == JobState.failed
    assert map_job_status("CANCELLED") == JobState.cancelled
    assert map_job_status("WAT") == JobState.failed  # unknown -> fail-loud


def test_cluster_name_for():
    assert cluster_name_for("20260527-114219-c8e4e5") == "lab-20260527-114219-c8e4e5"
    assert cluster_name_for("Weird_Name!!") == "lab-weird-name"
    long = cluster_name_for("x" * 100)
    assert long.startswith("lab-") and len(long) <= 60


def test_build_scripts_and_timeout():
    setup = build_setup_script()
    assert "uv sync --frozen" in setup and "astral.sh/uv/install" in setup
    assert "--no-default-groups" in setup  # remote installs runtime deps only, not the CLI/MCP

    m = make_manifest("j1", "python experiments/example_capacity.py", timeout="30m")
    run = build_run_script(m)
    assert "timeout 1800 " in run  # 30m -> 1800s wall-clock guard (FR-I1)
    assert "python experiments/example_capacity.py" in run
    assert "source .venv/bin/activate" in run

    assert "timeout " not in build_run_script(make_manifest("j2", "python x.py"))  # no limit


def test_run_script_timeout_sentinel():
    run = build_run_script(make_manifest("j", "python x.py", timeout="30m"))
    assert "timeout 1800 python x.py" in run
    assert TIMEOUT_SENTINEL in run and "rc=$?" in run and "exit $rc" in run


def test_promote_timeout(tmp_path):
    assert promote_timeout(JobState.failed, tmp_path) == JobState.failed
    (tmp_path / TIMEOUT_SENTINEL).touch()
    assert promote_timeout(JobState.failed, tmp_path) == JobState.timed_out
    # only a failed run is promoted
    assert promote_timeout(JobState.succeeded, tmp_path) == JobState.succeeded


def test_build_task_fields(tmp_path: Path):
    pytest.importorskip("sky")  # build_task needs the optional `skypilot` extra
    m = make_manifest("j1", "python run.py", seed=5)
    task = build_task(m, workdir=tmp_path)
    assert task.envs["LAB_RUN_ID"] == "j1"
    assert task.envs["LAB_SEED"] == "5"
    assert task.envs["LAB_RUN_DIR"]
    assert "run.py" in task.run
    assert "uv sync" in task.setup

    # accelerators flow through to the Resources (required for Vast)
    gpu_task = build_task(make_manifest("j2", "python r.py", accelerators="T4:1"), workdir=tmp_path)
    res = next(iter(gpu_task.resources))
    assert "T4" in str(res.accelerators)


# ----------------------------------------------------------------------------
# robust_teardown — FR-C2 leak prevention (sky.down retries + vast-sdk fallback)
# ----------------------------------------------------------------------------


class _FakeSky:
    """Minimal stand-in for the `sky` module. ``down(cluster)`` records the call and returns a
    sentinel "request id"; ``get(req)`` either succeeds or raises depending on ``fail_times``.
    """

    def __init__(self, fail_times: int = 0, exc: type[Exception] = ConnectionError) -> None:
        self.fail_times = fail_times
        self.exc = exc
        self.down_calls: list[str] = []
        self.get_calls: int = 0

    def down(self, cluster: str) -> str:
        self.down_calls.append(cluster)
        return f"req-{len(self.down_calls)}"

    def get(self, req: str) -> None:
        self.get_calls += 1
        if self.get_calls <= self.fail_times:
            raise self.exc(f"transient #{self.get_calls}")


class _FakeVast:
    def __init__(self, instances: list[dict] | None = None) -> None:
        self.instances = instances or []
        self.destroyed: list[int] = []

    def show_instances(self) -> list[dict]:
        return list(self.instances)

    def destroy_instance(self, id: int) -> dict:  # noqa: A002 - mirror real vastai-sdk arg name
        self.destroyed.append(int(id))
        return {"id": id, "destroyed": True}


@pytest.fixture
def _no_sleep(monkeypatch: pytest.MonkeyPatch):
    """Make retries instantaneous in tests."""
    monkeypatch.setattr(skypilot_mod.time, "sleep", lambda _s: None)


def test_robust_teardown_first_attempt_succeeds(_no_sleep):
    sky = _FakeSky(fail_times=0)
    out = robust_teardown(sky, "lab-abc")
    assert out["status"] == "succeeded"
    assert out["attempts"] == 1
    assert out["vast_fallback_used"] is False
    assert sky.down_calls == ["lab-abc"]


def test_robust_teardown_retries_then_succeeds(_no_sleep):
    sky = _FakeSky(fail_times=3)  # 3 transient errors, then succeeds on the 4th attempt
    out = robust_teardown(sky, "lab-abc")
    assert out["status"] == "succeeded"
    assert out["attempts"] == 4
    assert out["vast_fallback_used"] is False
    assert out["error"] is None  # no error retained on eventual sky.down success


def test_robust_teardown_falls_back_to_vast_when_sky_exhausted(
    _no_sleep, monkeypatch: pytest.MonkeyPatch
):
    sky = _FakeSky(fail_times=99)  # always fails
    vast = _FakeVast(
        instances=[
            {"id": 7, "label": "sky-lab-abc-xyz"},  # matches cluster -> destroyed
            {"id": 8, "label": "someone-elses"},  # not ours -> left alone
        ]
    )
    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: vast)
    out = robust_teardown(sky, "lab-abc")
    assert out["status"] == "succeeded"  # fallback succeeded
    assert out["vast_fallback_used"] is True
    assert out["vast_destroyed"] == [7]
    assert out["attempts"] == 6  # 1 + 5 retries (cf. TEARDOWN_BACKOFFS)
    assert vast.destroyed == [7]  # only the matching instance was destroyed


def test_robust_teardown_failed_when_both_sky_and_vast_error(
    _no_sleep, monkeypatch: pytest.MonkeyPatch
):
    sky = _FakeSky(fail_times=99)

    class _BrokenVast:
        def show_instances(self) -> list[dict]:
            raise RuntimeError("vast api down")

    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: _BrokenVast())
    out = robust_teardown(sky, "lab-abc")
    assert out["status"] == "failed"
    assert out["vast_fallback_used"] is True
    assert "sky.down" in out["error"] and "vast-direct" in out["error"]


def test_tear_down_and_record_records_failure_loudly(
    _no_sleep, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A persistent teardown leak must show up in the manifest as teardown_status='failed' and
    an actionable end_reason annotation so `lab status`/`lab wait` can flag it."""
    store = JobStore(tmp_path)
    m = make_manifest("job-1", "python x.py")
    store.create(m)
    store.update_manifest("job-1", end_reason="some prior reason")  # confirm we append, not clobber
    sky = _FakeSky(fail_times=99)

    class _BrokenVast:
        def show_instances(self) -> list[dict]:
            raise RuntimeError("vast api down")

    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: _BrokenVast())
    ok = tear_down_and_record(sky, "lab-job-1", store, "job-1")
    assert ok is False
    persisted = store.read_manifest("job-1")
    assert persisted.teardown_status == "failed"
    assert persisted.end_reason is not None
    assert "TEARDOWN FAILED" in persisted.end_reason
    assert "some prior reason" in persisted.end_reason  # annotation appended, not overwritten


def test_tear_down_and_record_marks_success_when_sky_down_succeeds(_no_sleep, tmp_path: Path):
    store = JobStore(tmp_path)
    store.create(make_manifest("job-2", "python x.py"))
    sky = _FakeSky(fail_times=0)
    ok = tear_down_and_record(sky, "lab-job-2", store, "job-2")
    assert ok is True
    assert store.read_manifest("job-2").teardown_status == "succeeded"
