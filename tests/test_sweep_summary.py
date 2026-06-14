"""Tests for Lab.sweep_summary — preemption / fallback / per-point spend (Task 13)."""

from __future__ import annotations

from pathlib import Path

from helpers import make_manifest

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import BackendInfo, CostInfo, JobState, ResourceRequest


def _seed(
    lab: Lab,
    jid: str,
    *,
    status: JobState,
    use_spot: bool,
    launched_spot: bool | None,
    actual: float,
) -> None:
    m = make_manifest(jid, "x").model_copy(
        update={
            "sweep_id": "sw",
            "status": status,
            "cost": CostInfo(actual_usd=actual),
            "resources": ResourceRequest(use_spot=use_spot),
            "backend": BackendInfo(provisioner="skypilot", launched_spot=launched_spot),
        }
    )
    lab.store.create(m)


def test_sweep_summary_counts(tmp_path: Path) -> None:
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)

    # job a: succeeded on spot, normal
    _seed(lab, "a", status=JobState.succeeded, use_spot=True, launched_spot=True, actual=1.0)
    # job b: preempted on spot
    _seed(lab, "b", status=JobState.preempted, use_spot=True, launched_spot=True, actual=0.2)
    # job c: succeeded, requested spot but fell back to on-demand
    _seed(lab, "c", status=JobState.succeeded, use_spot=True, launched_spot=False, actual=2.0)

    s = lab.sweep_summary("sw")

    assert s["total"] == 3
    assert s["succeeded"] == 2
    assert s["preempted"] == 1
    assert s["fell_back_to_on_demand"] == 1
    assert round(s["total_usd"], 2) == 3.20
    assert s["per_point"]["b"]["state"] == "preempted"
    assert s["per_point"]["c"]["launched_spot"] is False


def test_sweep_summary_empty(tmp_path: Path) -> None:
    """A sweep_id with no jobs returns zeroed totals and an empty per_point."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    s = lab.sweep_summary("no-such-sweep")
    assert s["total"] == 0
    assert s["preempted"] == 0
    assert s["fell_back_to_on_demand"] == 0
    assert s["total_usd"] == 0.0
    assert s["per_point"] == {}


def test_sweep_summary_no_fallback_counted_for_non_spot(tmp_path: Path) -> None:
    """A job that never requested spot (use_spot=False) is NOT counted as a fallback
    even if launched_spot is False."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed(lab, "d", status=JobState.succeeded, use_spot=False, launched_spot=False, actual=1.5)
    s = lab.sweep_summary("sw")
    assert s["fell_back_to_on_demand"] == 0


def test_sweep_summary_spend_aggregation(tmp_path: Path) -> None:
    """per_point spend and total_usd are correctly rounded."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed(lab, "e", status=JobState.succeeded, use_spot=True, launched_spot=True, actual=0.123456)
    _seed(lab, "f", status=JobState.failed, use_spot=True, launched_spot=True, actual=0.0)
    s = lab.sweep_summary("sw")
    assert s["per_point"]["e"]["usd"] == round(0.123456, 6)
    assert s["per_point"]["f"]["usd"] == 0.0
    assert s["failed"] == 1
