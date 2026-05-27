from pathlib import Path

from helpers import make_manifest

from lab.backends.skypilot import (
    build_run_script,
    build_setup_script,
    build_task,
    cluster_name_for,
    map_job_status,
)
from lab.models import JobState


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

    m = make_manifest("j1", "python experiments/example_capacity.py", timeout="30m")
    run = build_run_script(m)
    assert "timeout 1800 " in run  # 30m -> 1800s wall-clock guard (FR-I1)
    assert "python experiments/example_capacity.py" in run
    assert "source .venv/bin/activate" in run

    assert "timeout " not in build_run_script(make_manifest("j2", "python x.py"))  # no limit


def test_build_task_fields(tmp_path: Path):
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
