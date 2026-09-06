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
from typing import Any

import pytest
from helpers import PYTHON, make_manifest
from typer.testing import CliRunner

from lab.cli import app
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


class _RaisingListEntriesQueue:
    """A QueueStore whose `read_mirrored` works fine (delegates to a real `LocalQueueStore`)
    but whose `list_entries` blows up -- the second, previously-unguarded call inside
    `_cancel_redirect_message` (finding the registration that maps to a mirror-only job, for a
    friendlier redirect message). A transient queue-store error here (network timeout, corrupt
    registration entry) must not crash `cancel`'s own error path, exactly like a `read_mirrored`
    crash already doesn't."""

    def __init__(self, root: Path) -> None:
        self._inner = LocalQueueStore(root)

    def read_mirrored(self, job_id: str):  # noqa: ANN001, ANN201 - test double
        return self._inner.read_mirrored(job_id)

    def list_entries(self):  # noqa: ANN201 - test double
        raise ValueError("queue store timeout")


# ---------------------------------------------------------------------------
# logs / metrics / fetch: fall back to the mirror
# ---------------------------------------------------------------------------


class TestReadOnlyCommandsFallBackToMirror:
    def test_logs_does_not_error_on_a_mirror_only_job(self, tmp_path, monkeypatch):
        """A job this machine never supervised has no local `logs.txt` anywhere it could read
        one from -- the scheduler only ever uploads `output/` to R2 (`sky_runner.py`'s
        `r2.upload_dir(store.output_dir(job_id), ...)`), never `logs.txt`, which sits beside it.
        The fix here is that this no longer errors "unknown job id" the way it did before the
        mirror fallback existed -- it resolves the job and returns (empty) logs cleanly, and
        (see TestMirrorReadNeverTouchesTheRealLocalStore) never creates a local manifest."""
        home = _isolate(tmp_path, monkeypatch)
        _mirror_only(tmp_path, "jmir-logs")

        result = runner.invoke(app, ["logs", "jmir-logs"])

        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""
        assert not JobStore(home).manifest_path("jmir-logs").exists()

    def test_metrics_does_not_error_on_a_mirror_only_job(self, tmp_path, monkeypatch):
        """Same reasoning as the `logs` case above: no local `metrics.jsonl` exists anywhere
        this machine can see for a job it never ran, so the correct behaviour is an empty,
        successful result -- not an error, and not a seeded local manifest."""
        home = _isolate(tmp_path, monkeypatch)
        _mirror_only(tmp_path, "jmir-metrics")

        result = runner.invoke(app, ["metrics", "jmir-metrics"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["series"] == {}
        assert not JobStore(home).manifest_path("jmir-metrics").exists()

    def test_fetch_recovers_artifacts_from_r2_for_a_mirror_only_job(self, tmp_path, monkeypatch):
        """Unlike logs/metrics, artifacts genuinely do have a cross-machine recovery path: R2
        (module docstring; `Lab.fetch_artifacts`'s own fallback). Simulate that path instead of
        (as before the fix) placing the file directly under this machine's own `runs/` -- which
        would only be true if this machine were the one that actually ran the job, not what
        "mirror-only" means."""
        home = _isolate(tmp_path, monkeypatch)
        _mirror_only(tmp_path, "jmir-fetch", artifacts_uri="r2://lab-artifacts/jmir-fetch")

        class _FakeR2Store:
            @staticmethod
            def from_env() -> "_FakeR2Store":
                return _FakeR2Store()

            def download_dir(self, prefix: str, local_dir: Path) -> int:
                local_dir = Path(local_dir)
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "result.txt").write_text("42")
                return 1

        monkeypatch.setattr("lab.core.r2_enabled", lambda: True)
        monkeypatch.setattr("lab.core.R2Store", _FakeR2Store)

        result = runner.invoke(app, ["fetch", "jmir-fetch"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert any(a["name"] == "result.txt" for a in data["artifacts"])
        # `collect_artifacts` calls `store.update_manifest`, which needs a local manifest to
        # exist -- it gets one from an ephemeral, throwaway JobStore (see `_lab_for_mirrored`),
        # never the real local store. A mirror-only job must stay invisible to `runs/`, or
        # `cancel`/`reconcile` could no longer tell it apart from one this machine actually
        # supervises (see TestMirrorReadNeverTouchesTheRealLocalStore below).
        assert not JobStore(home).manifest_path("jmir-fetch").exists()

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

    def test_a_crashing_list_entries_does_not_crash_cancel(self, tmp_path, monkeypatch):
        """`read_mirrored` succeeds (the job really is mirror-only), but the later
        `list_entries()` call -- made to find the matching registration for a friendlier
        message -- raises. That must still degrade to the generic "scheduler-launched, no
        matching registration" redirect, not an unhandled exception out of `cancel`."""
        _isolate(tmp_path, monkeypatch)
        _mirror_only(tmp_path, "jmir-list-entries-crash")
        monkeypatch.setattr(
            "lab.scheduler.queue.default_queue",
            lambda: _RaisingListEntriesQueue(tmp_path / "queue"),
        )

        result = runner.invoke(app, ["cancel", "jmir-list-entries-crash"])

        assert result.exit_code == 2
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "lab queue cancel" in result.output


# ---------------------------------------------------------------------------
# safety regression: a read-only mirror-fallback must never create anything
# that looks like a locally-supervised job -- the bug where `fetch`/`logs`/
# `metrics` seeded the mirrored manifest into the REAL local store, letting a
# later `cancel` on the same job bypass its own mirror-redirect safety check
# and letting `reconcile`'s `unsupervised` pass see (and, under `--apply`,
# destroy) a job that is actually running fine under the scheduler.
# ---------------------------------------------------------------------------


class TestMirrorReadNeverTouchesTheRealLocalStore:
    def test_cancel_still_redirects_after_the_job_was_fetched_logged_and_metriced(
        self, tmp_path, monkeypatch
    ):
        home = _isolate(tmp_path, monkeypatch)
        _mirror_only(tmp_path, "jmir-safety")
        LocalQueueStore(tmp_path / "queue").put_entry(_reg("reg-safety", "jmir-safety"))

        for command in ("fetch", "logs", "metrics"):
            touched = runner.invoke(app, [command, "jmir-safety"])
            assert touched.exit_code == 0, touched.output
            # none of the three read-only commands may seed the real local store
            assert not JobStore(home).manifest_path("jmir-safety").exists()

        cancel_result = runner.invoke(app, ["cancel", "jmir-safety"])

        assert cancel_result.exit_code == 2
        assert "lab queue cancel reg-safety" in cancel_result.output
        # cancel never proceeded to a real teardown attempt (which would have needed -- and
        # left behind -- a local manifest to act on)
        assert not JobStore(home).manifest_path("jmir-safety").exists()

    def test_reconcile_unsupervised_pass_ignores_a_job_only_touched_via_fetch(
        self, tmp_path, monkeypatch
    ):
        from datetime import timedelta

        from lab._util import now
        from lab.backends.local import LocalBackend
        from lab.backends.skypilot import GcpNotConfigured
        from lab.core import Lab
        from lab.models import BackendInfo
        from test_leak_blindspots import _patch_empty_sky

        home = _isolate(tmp_path, monkeypatch)
        _mirror_only(
            tmp_path,
            "jmir-recon",
            status=JobState.running,
            started_at=now() - timedelta(hours=1),
            backend=BackendInfo(provisioner="skypilot"),
        )

        for command in ("fetch", "logs", "metrics"):
            touched = runner.invoke(app, [command, "jmir-recon"])
            assert touched.exit_code == 0, touched.output
        assert not JobStore(home).manifest_path("jmir-recon").exists()

        lab = Lab(backend=LocalBackend(home=home, repo=home.parent), repo=home.parent, home=home)
        monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
        _patch_empty_sky(monkeypatch)
        monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])

        def _not_configured(*a: object, **k: object) -> list[dict[str, Any]]:
            raise GcpNotConfigured("gcp extra not installed in this test environment")

        monkeypatch.setattr("lab.backends.skypilot.list_gcp_instances", _not_configured)
        monkeypatch.setattr("lab.backends.skypilot.list_gcp_disks", _not_configured)

        report = lab.reconcile()

        assert report["unsupervised"] == []

    def test_confirm_error_message_names_what_actually_works(self, tmp_path, monkeypatch):
        """Bug 3: the stale message used to say only `lab status` reads the mirrored manifest --
        no longer true now that `fetch`/`metrics`/`logs` do too (`confirm` itself still doesn't,
        deliberately, same as `cancel`)."""
        _isolate(tmp_path, monkeypatch)

        result = runner.invoke(app, ["confirm", "nope"])

        assert result.exit_code == 2
        assert "fetch" in result.output
        assert "metrics" in result.output
        assert "logs" in result.output
        assert "status" in result.output
