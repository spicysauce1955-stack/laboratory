"""`lab ps` — the cross-project answer to "what's running right now."

Found live 2026-08-27: `lab reconcile` and `lab queue list` were each checked in isolation, and
both missed real running jobs in a *different* project on the same machine — because neither one,
nor anything else in the tool, ever looks at the machine-wide job registry to ask that question
directly. This is the fix: one command that walks every project this machine has ever heard of
(via `lab.attribution`'s registry) and reports which of their jobs are still non-terminal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from helpers import make_manifest
from typer.testing import CliRunner

from lab import attribution
from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab
from lab.models import BackendInfo, JobState

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_JOBS_INDEX_DIR", str(tmp_path / "lab-jobs"))


def _lab_at(home: Path) -> Lab:
    return Lab(backend=LocalBackend(home=home, repo=home), repo=home, home=home)


def _seed(
    lab: Lab, job_id: str, *, status: JobState, provisioner: str = "local", pid: int | None = None
) -> None:
    """Seed a job under ``lab``'s store.

    ``JobStore.create`` records attribution via ``local_project()``, which resolves from the
    *test process's* real cwd (this repo) rather than the fake per-test ``home`` — real usage
    never has that ambiguity, since each project is a separate ``lab`` invocation from its own
    cwd. Re-recording under ``lab.home.name`` afterwards (last write wins) simulates that.
    """
    m = make_manifest(job_id, "python x.py").model_copy(
        update={"status": status, "backend": BackendInfo(provisioner=provisioner)}
    )
    lab.store.create(m)
    attribution.record_job(job_id, project=lab.home.name, runs_dir=lab.store.home)
    if pid is not None:
        lab.store.write_runtime(job_id, runner_pid=pid)


def test_ps_finds_a_running_job_in_a_different_project(tmp_path: Path) -> None:
    """The exact gap this closes: a job this project's own store has never heard of."""
    other = _lab_at(tmp_path / "other-project")
    _seed(other, "job-elsewhere", status=JobState.running, provisioner="skypilot", pid=os.getpid())

    here = _lab_at(tmp_path / "here")
    report = here.ps()

    assert [j["job_id"] for j in report["jobs"]] == ["job-elsewhere"]
    assert report["jobs"][0]["project"] == "other-project"
    assert report["count"] == 1


def test_ps_excludes_terminal_jobs(tmp_path: Path) -> None:
    lab = _lab_at(tmp_path / "proj")
    _seed(lab, "job-done", status=JobState.succeeded)
    _seed(lab, "job-running", status=JobState.running)

    report = lab.ps()

    assert [j["job_id"] for j in report["jobs"]] == ["job-running"]


def test_ps_flags_a_dead_supervisor_as_unsupervised(tmp_path: Path) -> None:
    """A running job whose local supervisor pid is gone — the money-alarm case, not just noise."""
    lab = _lab_at(tmp_path / "proj")
    _seed(lab, "job-orphaned", status=JobState.running, provisioner="skypilot", pid=999999999)

    report = lab.ps()

    assert report["jobs"][0]["supervised"] == "unsupervised"


def test_ps_flags_a_live_supervisor_as_local(tmp_path: Path) -> None:
    lab = _lab_at(tmp_path / "proj")
    _seed(lab, "job-alive", status=JobState.running, provisioner="skypilot", pid=os.getpid())

    report = lab.ps()

    assert report["jobs"][0]["supervised"] == "local"


def test_ps_on_an_empty_registry_is_empty(tmp_path: Path) -> None:
    lab = _lab_at(tmp_path / "proj")
    assert lab.ps() == {"jobs": [], "count": 0}


def test_ps_skips_a_job_whose_manifest_is_gone(tmp_path: Path) -> None:
    """The registry can outlive the manifest (a wiped runs/ dir) — must not crash, must not lie."""
    attribution.record_job(
        "ghost-id", project="vanished-project", runs_dir=tmp_path / "gone" / "runs"
    )
    lab = _lab_at(tmp_path / "proj")
    assert lab.ps() == {"jobs": [], "count": 0}


def test_ps_cli_command_reports_a_running_job(tmp_path: Path) -> None:
    lab = _lab_at(tmp_path / "proj")
    _seed(lab, "job-cli", status=JobState.running)

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [j["job_id"] for j in data["jobs"]] == ["job-cli"]


def test_ps_spans_multiple_projects_at_once(tmp_path: Path) -> None:
    a = _lab_at(tmp_path / "a")
    b = _lab_at(tmp_path / "b")
    _seed(a, "job-a", status=JobState.running)
    _seed(b, "job-b", status=JobState.queued)

    report = _lab_at(tmp_path / "c").ps()

    assert {j["job_id"] for j in report["jobs"]} == {"job-a", "job-b"}
    assert {j["project"] for j in report["jobs"]} == {"a", "b"}
