"""Tests for the reproducibility gate: final-metric snapshots, verdict comparison, and
``lab confirm`` (relaunch a pinned run fresh, then compare against the original)."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import PYTHON, make_manifest, wait_terminal

from lab.backends.local import LocalBackend
from lab.core import Lab, LabError, build_backend, compare_final_metrics
from lab.manifest import current_commit, repo_root
from lab.metrics import final_values
from lab.models import CodeRef, JobSpec, JobState
from lab.scheduler.bundle import create_bundle, extract_bundle

# ---------------------------------------------------------------------------
# final_values — last value per metric series, by max step
# ---------------------------------------------------------------------------


def test_final_values_takes_last_value_per_name_by_step():
    points = [
        {"name": "loss", "value": 1.0, "step": 0, "wall_time": 0.0},
        {"name": "loss", "value": 0.5, "step": 2, "wall_time": 0.0},
        {"name": "loss", "value": 0.7, "step": 1, "wall_time": 0.0},  # out of order
        {"name": "acc", "value": 0.9, "step": 5, "wall_time": 0.0},
    ]
    assert final_values(points) == {"loss": 0.5, "acc": 0.9}


def test_final_values_empty():
    assert final_values([]) == {}


def test_final_values_skips_non_numeric_values(tmp_path: Path):
    """A non-coercible metric value (e.g. a stray string) is skipped, not raised — so one bad
    line can't crash the finalize snapshot and leave the job stuck non-terminal (#3)."""
    points = [
        {"name": "loss", "value": "not-a-number", "step": 1, "wall_time": 0.0},
        {"name": "acc", "value": 0.9, "step": 1, "wall_time": 0.0},
    ]
    assert final_values(points) == {"acc": 0.9}


def test_final_values_ignores_none_and_breaks_ties_by_last_seen():
    points = [
        {"name": "loss", "value": 0.4, "step": 3, "wall_time": 0.0},
        {"name": "loss", "value": None, "step": 9, "wall_time": 0.0},  # skipped (no value)
        {"name": "loss", "value": 0.42, "step": 3, "wall_time": 0.0},  # tie on step -> wins
    ]
    assert final_values(points) == {"loss": 0.42}


# ---------------------------------------------------------------------------
# compare_final_metrics — match / drift verdicts within tolerance
# ---------------------------------------------------------------------------


def test_compare_match_within_tolerance():
    verdict, deltas = compare_final_metrics(
        {"loss": 0.5000}, {"loss": 0.5001}, names=None, rtol=1e-2, atol=0.0
    )
    assert verdict == "match"
    assert deltas["loss"]["orig"] == 0.5 and deltas["loss"]["new"] == 0.5001


def test_compare_drift_outside_tolerance():
    verdict, deltas = compare_final_metrics(
        {"loss": 0.5}, {"loss": 0.9}, names=None, rtol=1e-3, atol=0.0
    )
    assert verdict == "drift"
    assert deltas["loss"]["within_tol"] is False


def test_compare_restricts_to_named_metrics():
    # acc drifts but is not selected -> only loss (match) is judged
    verdict, deltas = compare_final_metrics(
        {"loss": 0.5, "acc": 0.1}, {"loss": 0.5, "acc": 0.99}, names=["loss"], rtol=1e-3, atol=0.0
    )
    assert verdict == "match"
    assert set(deltas) == {"loss"}


def test_compare_absolute_tolerance_allows_small_diff():
    # rtol=0 so only atol matters: 0.1 abs diff is within atol=0.2 -> match
    verdict, _ = compare_final_metrics(
        {"x": 1.0}, {"x": 1.1}, names=None, rtol=0.0, atol=0.2
    )
    assert verdict == "match"


def test_compare_zero_valued_metric_does_not_false_drift():
    """A baseline of exactly 0.0 vs a tiny non-zero re-run value must not report drift under the
    default tolerances — math.isclose's relative tolerance collapses to 0 at zero (#2)."""
    # default atol (no atol passed) tolerates float noise around zero
    assert compare_final_metrics({"x": 0.0}, {"x": 1e-13}, names=None)[0] == "match"
    # a real, meaningful difference from zero still drifts
    assert compare_final_metrics({"x": 0.0}, {"x": 0.5}, names=None)[0] == "drift"


def test_compare_zero_tolerance_requires_exact_match():
    assert compare_final_metrics({"x": 1.0}, {"x": 1.0}, names=None, rtol=0.0, atol=0.0)[0] == "match"
    assert compare_final_metrics({"x": 1.0}, {"x": 1.000001}, names=None, rtol=0.0, atol=0.0)[0] == "drift"


def test_compare_empty_baseline_is_vacuous_match():
    # no metrics selected -> nothing to disprove (callers gate on an empty baseline separately)
    verdict, deltas = compare_final_metrics({}, {"x": 1.0}, names=None, rtol=1e-3, atol=0.0)
    assert verdict == "match"
    assert deltas == {}


def test_compare_missing_metric_is_drift():
    # a baseline metric absent from the re-run can't be confirmed -> drift
    verdict, deltas = compare_final_metrics(
        {"loss": 0.5}, {}, names=None, rtol=1e-3, atol=0.0
    )
    assert verdict == "drift"
    assert deltas["loss"]["new"] is None


# ---------------------------------------------------------------------------
# final_metrics snapshot — populated at finalize on success
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# create_bundle — snapshot an arbitrary pinned commit, committed tree only
# ---------------------------------------------------------------------------


def test_create_bundle_rejects_dirty_with_explicit_commit(tmp_path: Path):
    """include_dirty captures the working tree (a diff vs HEAD), so it's incoherent with an
    explicit historical commit — guard the contradictory combo rather than corrupt the bundle (#4)."""
    repo = repo_root(Path.cwd())
    with pytest.raises(ValueError, match="include_dirty"):
        create_bundle(repo, tmp_path, commit=current_commit(repo), include_dirty=True)


def test_create_bundle_pins_given_commit_without_dirty(tmp_path: Path):
    repo = repo_root(Path.cwd())
    commit = current_commit(repo)
    tar, code = create_bundle(repo, tmp_path, commit=commit, include_dirty=False)
    # name reflects the pinned commit and is NOT marked dirty even if the dev tree is dirty
    assert tar.name == f"{commit[:12]}.tar.gz"
    assert code == CodeRef(git_commit=commit, git_dirty=False)
    extracted = extract_bundle(tar, tmp_path / "tree")
    # the committed experiment is present in the snapshot (re-run can execute it)
    assert (extracted / "experiments" / "example_capacity.py").exists()


# ---------------------------------------------------------------------------
# submit confirms= kwarg — durable audit link to the run being confirmed
# ---------------------------------------------------------------------------


def test_submit_records_confirms_link(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "uv.lock").write_text("lock")
    home = tmp_path / "runs"
    lab = Lab(backend=LocalBackend(home=home, repo=bundle), repo=bundle, home=home)
    jid = lab.submit(
        JobSpec(command=f"{PYTHON} -c 'print(1)'"),
        code=CodeRef(git_commit="a" * 40),
        confirms="orig-123",
    )
    assert lab.manifest(jid).confirms == "orig-123"


def test_build_backend_maps_names(tmp_path: Path):
    """The single name->backend factory both Lab construction paths and the scheduler share (#7)."""
    repo = repo_root(Path.cwd())
    assert type(build_backend("local", home=tmp_path, repo=repo)).__name__ == "LocalBackend"
    assert type(build_backend("skypilot", home=tmp_path, repo=repo)).__name__ == "SkyPilotBackend"
    # unknown name falls back to local (preserves prior default_lab behavior)
    assert type(build_backend("bogus", home=tmp_path, repo=repo)).__name__ == "LocalBackend"


def test_update_manifest_snapshots_final_metrics_centrally(tmp_path: Path):
    """The FR-B4 baseline invariant lives in store.update_manifest: any transition to succeeded
    snapshots final_metrics, so no backend's finalize path can forget it (#8)."""
    import json

    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)

    def _seed_with_metrics(job_id: str) -> None:
        lab.store.create(make_manifest(job_id, f"{PYTHON} {EXAMPLE}", seed=1))
        with (lab.store.output_dir(job_id) / "metrics.jsonl").open("w") as f:
            f.write(json.dumps({"name": "loss", "value": 0.25, "step": 0, "wall_time": 0.0}) + "\n")

    # transition -> succeeded auto-snapshots from the output dir
    _seed_with_metrics("j-ok")
    assert lab.store.update_manifest("j-ok", status=JobState.succeeded).final_metrics == {"loss": 0.25}
    # an explicitly-supplied snapshot is respected, not overwritten
    _seed_with_metrics("j-explicit")
    got = lab.store.update_manifest("j-explicit", status=JobState.succeeded, final_metrics={"loss": 9.0})
    assert got.final_metrics == {"loss": 9.0}
    # a non-succeeded transition never snapshots
    _seed_with_metrics("j-fail")
    assert lab.store.update_manifest("j-fail", status=JobState.failed).final_metrics == {}


def test_succeeded_run_snapshots_final_metrics(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    jid = lab.submit(
        JobSpec(code_ref="HEAD", command=f"{PYTHON} experiments/example_capacity.py", seed=7)
    )
    assert wait_terminal(backend, jid) == JobState.succeeded
    m = lab.manifest(jid)
    # example_capacity logs demo_metric over 10 steps; the snapshot keeps the last value
    assert "demo_metric" in m.final_metrics
    series = lab.metrics(jid)["demo_metric"]
    assert m.final_metrics["demo_metric"] == series[-1]["value"]


# ---------------------------------------------------------------------------
# Lab.confirm — the reproducibility gate
# ---------------------------------------------------------------------------

EXAMPLE = "experiments/example_capacity.py"


def _lab(tmp_path: Path) -> tuple[Lab, LocalBackend]:
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    return Lab(backend=backend, repo=repo, home=tmp_path), backend


def _baseline_for(lab: Lab, backend: LocalBackend, seed: int) -> dict[str, float]:
    """Run example_capacity once to learn its (deterministic) final metric values."""
    jid = lab.submit(JobSpec(code_ref="HEAD", command=f"{PYTHON} {EXAMPLE}", seed=seed))
    assert wait_terminal(backend, jid) == JobState.succeeded
    return dict(lab.manifest(jid).final_metrics)


def _seed_original(
    lab: Lab,
    *,
    command: str,
    seed: int,
    status: JobState = JobState.succeeded,
    final_metrics: dict[str, float] | None = None,
    git_dirty: bool = False,
) -> str:
    """Drop a clean, succeeded 'original' manifest (real HEAD commit) on disk to confirm against."""
    repo = repo_root(Path.cwd())
    job_id = f"orig-{seed}-{status.value}"
    m = make_manifest(job_id, command, seed=seed).model_copy(
        update={
            "code": CodeRef(
                git_commit=current_commit(repo),
                git_dirty=git_dirty,
                diff_ref="test" if git_dirty else None,
            ),
            "status": status,
            "final_metrics": final_metrics or {},
        }
    )
    lab.store.create(m)
    lab.store.write_manifest(m)
    return job_id


@pytest.mark.parametrize(
    "bad", [JobState.failed, JobState.timed_out, JobState.preempted, JobState.cancelled]
)
def test_confirm_refuses_non_succeeded_producer(tmp_path: Path, bad: JobState):
    """The highest-value gate: a number whose producing run did not succeed is never confirmed,
    and no re-run is launched (catches the figure-asserted control: j3/j3b were failed, 0/16)."""
    lab, _ = _lab(tmp_path)
    orig = _seed_original(lab, command=f"{PYTHON} {EXAMPLE}", seed=1, status=bad,
                          final_metrics={"demo_metric": 0.5})
    before = len(lab.list_jobs())
    with pytest.raises(LabError, match=bad.value):
        lab.confirm(orig)
    assert len(lab.list_jobs()) == before  # nothing launched


def test_confirm_refuses_dirty_producer(tmp_path: Path):
    lab, _ = _lab(tmp_path)
    orig = _seed_original(lab, command=f"{PYTHON} {EXAMPLE}", seed=1, git_dirty=True,
                          final_metrics={"demo_metric": 0.5})
    with pytest.raises(LabError, match="dirty"):
        lab.confirm(orig)


def test_confirm_match_on_deterministic_rerun(tmp_path: Path):
    lab, backend = _lab(tmp_path)
    baseline = _baseline_for(lab, backend, seed=7)
    orig = _seed_original(lab, command=f"{PYTHON} {EXAMPLE}", seed=7, final_metrics=baseline)
    result = lab.confirm(orig, timeout=60)
    assert result["verdict"] == "match"
    assert result["deltas"]["demo_metric"]["within_tol"] is True
    # the re-run is a real, tracked job that records its provenance link back to the original
    assert lab.manifest(result["confirm_id"]).confirms == orig
    assert isinstance(result["env_drift"], bool)


def test_confirm_drift_when_baseline_differs(tmp_path: Path):
    lab, backend = _lab(tmp_path)
    baseline = _baseline_for(lab, backend, seed=7)
    tampered = {k: v + 1.0 for k, v in baseline.items()}  # a number that won't re-derive
    orig = _seed_original(lab, command=f"{PYTHON} {EXAMPLE}", seed=7, final_metrics=tampered)
    result = lab.confirm(orig, timeout=60)
    assert result["verdict"] == "drift"
    assert result["deltas"]["demo_metric"]["within_tol"] is False


def test_confirm_timeout_while_rerun_still_running(tmp_path: Path):
    """If the re-run hasn't reached a terminal state before the wait timeout, the verdict is
    'timed_out_waiting' (NOT 'rerun_failed') — the job is still alive, not failed (#1)."""
    lab, backend = _lab(tmp_path)
    orig = _seed_original(
        lab, command=f'{PYTHON} -c "import time; time.sleep(30)"', seed=1,
        final_metrics={"demo_metric": 0.5},
    )
    result = lab.confirm(orig, timeout=0.5)
    assert result["verdict"] == "timed_out_waiting"
    assert result["rerun_status"] in {"running", "queued"}
    assert "deltas" not in result
    backend.cancel(result["confirm_id"])  # clean up the sleeper


def test_confirm_rerun_failed(tmp_path: Path):
    lab, _ = _lab(tmp_path)
    orig = _seed_original(
        lab, command=f'{PYTHON} -c "import sys; sys.exit(1)"', seed=1,
        final_metrics={"demo_metric": 0.5},
    )
    result = lab.confirm(orig, timeout=60)
    assert result["verdict"] == "rerun_failed"
    assert result["rerun_status"] == "failed"


def test_confirm_falls_back_to_metrics_file(tmp_path: Path):
    """An old run with no snapshot but a surviving metrics.jsonl is still confirmable."""
    lab, backend = _lab(tmp_path)
    baseline = _baseline_for(lab, backend, seed=7)
    orig = _seed_original(lab, command=f"{PYTHON} {EXAMPLE}", seed=7, final_metrics={})
    # write the original's metrics file so the fallback path can read the baseline
    out = lab.store.output_dir(orig)
    out.mkdir(parents=True, exist_ok=True)
    import json
    with (out / "metrics.jsonl").open("w") as f:
        for step, (name, val) in enumerate((n, v) for n, v in baseline.items()):
            f.write(json.dumps({"name": name, "value": val, "step": 0, "wall_time": 0.0}) + "\n")
    result = lab.confirm(orig, timeout=60)
    assert result["verdict"] == "match"


def test_confirm_fails_loud_without_baseline(tmp_path: Path):
    lab, _ = _lab(tmp_path)
    orig = _seed_original(lab, command=f"{PYTHON} {EXAMPLE}", seed=7, final_metrics={})
    before = len(lab.list_jobs())
    with pytest.raises(LabError, match="baseline"):
        lab.confirm(orig)
    assert len(lab.list_jobs()) == before  # nothing launched


def test_confirm_unknown_run_id_is_fail_loud(tmp_path: Path):
    """An unknown run id is a structured LabError (FR-F3), not a raw FileNotFoundError."""
    lab, _ = _lab(tmp_path)
    with pytest.raises(LabError, match="not found"):
        lab.confirm("nope-does-not-exist")


def test_confirm_missing_commit_tells_user_to_fetch(tmp_path: Path):
    """A pinned commit absent from the local repo fails loud with actionable guidance, not a raw
    git CalledProcessError, and launches nothing."""
    lab, _ = _lab(tmp_path)
    job_id = "orig-missing-commit"
    m = make_manifest(job_id, f"{PYTHON} {EXAMPLE}", seed=7).model_copy(
        update={
            "code": CodeRef(git_commit="deadbeef" * 5, git_dirty=False),  # not in the repo
            "status": JobState.succeeded,
            "final_metrics": {"demo_metric": 0.5},
        }
    )
    lab.store.create(m)
    lab.store.write_manifest(m)
    before = len(lab.list_jobs())
    with pytest.raises(LabError, match="commit"):
        lab.confirm(job_id)
    assert len(lab.list_jobs()) == before


def test_confirm_no_wait_launches_without_comparing(tmp_path: Path):
    lab, backend = _lab(tmp_path)
    baseline = _baseline_for(lab, backend, seed=7)
    orig = _seed_original(lab, command=f"{PYTHON} {EXAMPLE}", seed=7, final_metrics=baseline)
    result = lab.confirm(orig, wait=False)
    assert result["verdict"] == "pending"
    assert "deltas" not in result
    # the re-run is a real, tracked job linked back to the original
    rerun = lab.manifest(result["confirm_id"])
    assert rerun.confirms == orig
    assert wait_terminal(backend, result["confirm_id"]) == JobState.succeeded


def test_confirm_env_drift_flagged_on_lockfile_mismatch(tmp_path: Path):
    """When the original's recorded uv.lock hash differs from the re-run's, env_drift is True even
    though the metric still matches."""
    lab, backend = _lab(tmp_path)
    baseline = _baseline_for(lab, backend, seed=7)
    orig = _seed_original(lab, command=f"{PYTHON} {EXAMPLE}", seed=7, final_metrics=baseline)
    # rewrite the original's env to a bogus lock hash to simulate a since-changed environment
    m = lab.manifest(orig)
    m = m.model_copy(update={"env": m.env.model_copy(update={"uv_lock_sha256": "stale-hash"})})
    lab.store.write_manifest(m)
    result = lab.confirm(orig, timeout=60)
    assert result["verdict"] == "match"
    assert result["env_drift"] is True


def test_confirm_partial_drift_overall_drift(tmp_path: Path):
    """Multiple metrics: one matches, one drifts -> overall drift, deltas reported per metric."""
    verdict, deltas = compare_final_metrics(
        {"acc": 0.9, "loss": 0.1}, {"acc": 0.9, "loss": 0.5}, names=None, rtol=1e-3, atol=0.0
    )
    assert verdict == "drift"
    assert deltas["acc"]["within_tol"] is True
    assert deltas["loss"]["within_tol"] is False
