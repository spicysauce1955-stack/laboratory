"""`fetch`/`metrics`/`logs` on a scheduler-launched job (confirmed gap, 2026-09-06 field report):
every single `fetch` call against a mirrored-only job failed in three days of production logs
with the generic "not in local runs/" message, even though the project documents
`lab status` -> `lab fetch` as the next-morning workflow for exactly this case ("artifacts come
from R2"). `job_status_view` (behind `lab status`) already falls back to the scheduler's
mirrored manifest (spec §4.3); these three read-only commands now do the same.

`cancel` is deliberately NOT given the same fallback: it reaches into a live backend from this
machine, and cross-machine SkyPilot client/server skew (`lab._skycompat`) makes acting on a job
this machine never locally supervised a real correctness risk. It gets a specific redirect to
`lab queue cancel <reg_id>` instead — the documented way to cancel a scheduler-launched job.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from helpers import PYTHON, make_manifest
from typer.testing import CliRunner

from lab.cli import app
from lab.metrics import log_metric
from lab.models import CodeRef, JobSpec, JobState
from lab.scheduler.models import Guardrails, Registration, RegState, Triggers
from lab.scheduler.queue import LocalQueueStore
from lab.store import JobStore

runner = CliRunner()


def _isolate(tmp_path: Path, monkeypatch) -> Path:
    """Point LAB_REPO_DIR/LAB_QUEUE_DIR at a scratch tree so the test never touches this repo's
    own runs/ or queue/, and return the `runs/` home the CLI will resolve to.

    Also chdir's into `tmp_path` (not a git work tree): otherwise cwd stays this real checkout,
    which differs from the `LAB_REPO_DIR` override and trips `_warn_if_repo_override_shadows_cwd`
    -- a stderr warning that `CliRunner` merges into `result.output` ahead of the JSON payload.
    """
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_REPO_DIR", str(repo))
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    return repo / "runs"


def _mirror_only(tmp_path: Path, job_id: str, **overrides) -> None:
    """Mirror a manifest for `job_id` without ever creating it in the local job store — the
    exact shape of a job the scheduler launched on a different machine."""
    m = make_manifest(job_id, f"{PYTHON} -c 'print(1)'").model_copy(
        update={"status": JobState.succeeded, **overrides}
    )
    LocalQueueStore(tmp_path / "queue").mirror_manifest(m)


def _reg(reg_id: str, job_id: str | None) -> Registration:
    return Registration(
        reg_id=reg_id,
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        spec=JobSpec(command=f"{PYTHON} -c 'print(1)'"),
        triggers=Triggers(),
        guardrails=Guardrails(expires_at=datetime(2026, 9, 2, tzinfo=timezone.utc)),
        bundle_key=f"bundles/{reg_id}.tar.gz",
        code=CodeRef(git_commit="0" * 40),
        state=RegState.launched,
        job_id=job_id,
    )


class _RaisingQueue:
    """A QueueStore whose `read_mirrored` blows up -- standing in for the separate, in-flight
    bug (partial/stub mirrored manifests crashing `read_mirrored`) this task must not assume is
    fixed yet. Every other method is unused by the code paths under test."""

    def read_mirrored(self, job_id: str):  # noqa: ANN001, ANN201 - test double
        raise ValueError("stub manifest, missing required field 'run'")


# ---------------------------------------------------------------------------
# logs / metrics / fetch: fall back to the mirror
# ---------------------------------------------------------------------------


class TestReadOnlyCommandsFallBackToMirror:
    def test_logs_reads_a_mirror_only_job(self, tmp_path, monkeypatch):
        home = _isolate(tmp_path, monkeypatch)
        _mirror_only(tmp_path, "jmir-logs")
        logs_path = JobStore(home).logs_path("jmir-logs")
        logs_path.parent.mkdir(parents=True)
        logs_path.write_text("line one\nline two\n")

        result = runner.invoke(app, ["logs", "jmir-logs"])

        assert result.exit_code == 0, result.output
        assert "line one" in result.output
        assert "line two" in result.output

    def test_metrics_reads_a_mirror_only_job(self, tmp_path, monkeypatch):
        home = _isolate(tmp_path, monkeypatch)
        _mirror_only(tmp_path, "jmir-metrics")
        out_dir = JobStore(home).output_dir("jmir-metrics")
        out_dir.mkdir(parents=True)
        log_metric("accuracy", 0.9, 1, run_dir=out_dir)

        result = runner.invoke(app, ["metrics", "jmir-metrics"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["series"]["accuracy"][0]["value"] == 0.9

    def test_fetch_reads_a_mirror_only_job(self, tmp_path, monkeypatch):
        home = _isolate(tmp_path, monkeypatch)
        _mirror_only(tmp_path, "jmir-fetch")
        out_dir = JobStore(home).output_dir("jmir-fetch")
        out_dir.mkdir(parents=True)
        (out_dir / "result.txt").write_text("42")

        result = runner.invoke(app, ["fetch", "jmir-fetch"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert any(a["name"] == "result.txt" for a in data["artifacts"])
        # `collect_artifacts` calls `store.update_manifest`, which needs a local manifest to
        # exist -- confirms the mirrored manifest was seeded locally, not merely bypassed.
        assert JobStore(home).read_manifest("jmir-fetch").job_id == "jmir-fetch"

    @pytest.mark.parametrize("command", ["logs", "metrics", "fetch"])
    def test_unknown_everywhere_still_gives_the_original_message(
        self, tmp_path, monkeypatch, command
    ):
        _isolate(tmp_path, monkeypatch)

        result = runner.invoke(app, [command, "nope"])

        assert result.exit_code == 2
        assert "unknown job id" in result.output
        assert "'nope'" in result.output

    @pytest.mark.parametrize("command", ["logs", "metrics", "fetch"])
    def test_a_crashing_mirror_read_is_a_clear_error_not_a_crash(
        self, tmp_path, monkeypatch, command
    ):
        """`read_mirrored` can currently raise on a partial/stub manifest (a separate, in-flight
        fix elsewhere) -- that must surface as a clear message here, not an unhandled traceback."""
        _isolate(tmp_path, monkeypatch)
        monkeypatch.setattr("lab.scheduler.queue.default_queue", lambda: _RaisingQueue())

        result = runner.invoke(app, [command, "flaky-job"])

        assert result.exit_code == 2
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "not yet available" in result.output


# ---------------------------------------------------------------------------
# cancel: redirect to `lab queue cancel`, never act on a mirror-only job
# ---------------------------------------------------------------------------


class TestCancelRedirectsInsteadOfActing:
    def test_redirects_to_queue_cancel_naming_the_reg_id(self, tmp_path, monkeypatch):
        home = _isolate(tmp_path, monkeypatch)
        _mirror_only(tmp_path, "jmir-cancel")
        LocalQueueStore(tmp_path / "queue").put_entry(_reg("reg-abc", "jmir-cancel"))

        result = runner.invoke(app, ["cancel", "jmir-cancel"])

        assert result.exit_code == 2
        assert "lab queue cancel reg-abc" in result.output
        assert "not in local runs/" not in result.output
        # never built a Lab / touched the local store for this job
        assert not JobStore(home).manifest_path("jmir-cancel").exists()

    def test_redirects_generically_when_no_registration_maps_to_the_job(
        self, tmp_path, monkeypatch
    ):
        _isolate(tmp_path, monkeypatch)
        _mirror_only(tmp_path, "jmir-cancel-noreg")
        # a registration exists, but for a different job_id -- must not be mistaken for a match
        LocalQueueStore(tmp_path / "queue").put_entry(_reg("reg-other", "some-other-job"))

        result = runner.invoke(app, ["cancel", "jmir-cancel-noreg"])

        assert result.exit_code == 2
        assert "lab queue cancel" in result.output
        assert "reg-other" not in result.output

    def test_unknown_everywhere_still_gives_the_original_message(self, tmp_path, monkeypatch):
        _isolate(tmp_path, monkeypatch)

        result = runner.invoke(app, ["cancel", "nope"])

        assert result.exit_code == 2
        assert "unknown job id 'nope'" in result.output
        assert "lab queue cancel" not in result.output

    def test_a_crashing_mirror_read_does_not_crash_cancel(self, tmp_path, monkeypatch):
        _isolate(tmp_path, monkeypatch)
        monkeypatch.setattr("lab.scheduler.queue.default_queue", lambda: _RaisingQueue())

        result = runner.invoke(app, ["cancel", "flaky-job"])

        assert result.exit_code == 2
        assert "could not be read" in result.output
