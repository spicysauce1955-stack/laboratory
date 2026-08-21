"""Regressions introduced by this incident's own fixes, found by code review (2026-08-21).

Both of the defects pinned here were written *while* fixing the 2026-08-20 incident, which is the
argument for reviewing a fix as carefully as the bug it replaces.

**F1 — a finished job recorded as `failed`.** `remaining_wall_budget` anchors the local cap to
`manifest.started_at`, which is stamped before `sky.launch`, so it charges provisioning *and* the
remote `uv sync` — while the box-side cap (`timeout --kill-after`) wraps only the entrypoint. When
the budget is already spent, `_wait_terminal`'s `while time.time() < deadline:` never runs even
once, `name` stays `None`, and the function returns `map_job_status("FAILED")`. `promote_timeout`
and `confirm_success` only ever *downgrade*, so a clean success is persisted as a failure. On spot
it can be relabelled `preempted` and resubmitted — paying twice for a job that already succeeded.

**F4 — machines destroyed outside the approval prompt.** `reconcile --apply` documents "lists what
it will destroy and asks... Only the approved set is destroyed". The `unsupervised` remediation
added this session called `SkyPilotBackend.status()` — which tears down and finalises — while
`unsupervised` was absent from `_ORPHAN_FIELDS`. So it never appeared in the preview, and with no
other orphans the prompt was skipped entirely: a bare `lab reconcile --apply` destroyed machines it
never mentioned. A false dead-supervisor reading then destroys a *live* job rather than reporting
it, which is precisely the failure this whole effort exists to prevent.
"""

import pytest
from typer.testing import CliRunner

import lab.cli as cli_mod
import lab.sky_runner as runner
from lab.cli import app
from lab.models import JobState

runner_cli = CliRunner()


class _SucceededSky:
    """A cluster whose job has already finished successfully."""

    def __init__(self):
        self.polls = 0

    def get(self, x):
        return x

    def queue(self, cluster, skip_finished=False):
        self.polls += 1
        return [{"job_id": 1, "status": type("S", (), {"name": "SUCCEEDED"})()}]


class TestASpentBudgetMustNotInventAFailure:
    def test_a_finished_job_is_still_read_as_succeeded(self):
        """The budget bounds how long we *wait*, not whether we bother to look."""
        sky = _SucceededSky()

        state, reached, lost = runner._wait_terminal(sky, "lab-x", None, max_wait=0.0)

        assert sky.polls >= 1, "must ask the cloud at least once before giving up"
        assert state is JobState.succeeded
        assert reached is True
        assert lost is None

    def test_a_genuinely_unfinished_job_still_gives_up(self):
        """No regression: an exhausted budget must not wait forever either."""

        class _RunningSky:
            def __init__(self):
                self.polls = 0

            def get(self, x):
                return x

            def queue(self, cluster, skip_finished=False):
                self.polls += 1
                return [{"job_id": 1, "status": type("S", (), {"name": "RUNNING"})()}]

        sky = _RunningSky()

        state, reached, _ = runner._wait_terminal(sky, "lab-x", None, max_wait=0.0)

        assert sky.polls == 1, "one look, then give up — not a loop"
        assert reached is False

    def test_a_vanished_cluster_is_still_detected_on_the_final_look(self):
        class ClusterDoesNotExist(Exception):
            """Matched by type name, exactly as `lab._skycompat` does."""

        class _GoneSky:
            def get(self, x):
                return x

            def queue(self, cluster, skip_finished=False):
                raise ClusterDoesNotExist("Cluster 'lab-x' does not exist.")

        state, reached, lost = runner._wait_terminal(_GoneSky(), "lab-x", None, max_wait=0.0)

        assert lost is not None and "does not exist" in lost
        assert state is JobState.failed


_EMPTY = {
    "vast_pass": "ran", "gcp_pass": "ran", "gcp_disk_pass": "ran",
    "gcp_project": "p", "gcp_unmatched": [], "instances_total": 0,
    "unsupervised": [], "orphans": [], "destroyed": [], "ghosts": [],
    "sky_orphans": [], "sky_destroyed": [], "do_volume_orphans": [], "do_volumes_destroyed": [],
    "gcp_orphans": [], "gcp_destroyed": [], "gcp_disk_orphans": [], "gcp_disks_destroyed": [],
    "other_projects": [], "unattributed": [], "destroy_outcomes": [],
    "applied": False,
}

UNSUP = {"job_id": "20260820-124800-05befa", "cluster": "lab-p-20260820-124800-05befa"}


class _FakeLab:
    def __init__(self, found):
        self.found = found
        self.calls: list[bool] = []
        self.only = None

    def reconcile(self, apply=False, only=None):
        self.calls.append(apply)
        self.only = only
        return {**_EMPTY, **self.found, "applied": apply}


@pytest.fixture(autouse=True)
def _interactive(monkeypatch):
    monkeypatch.setattr(cli_mod, "_stdin_is_a_tty", lambda: True)


def _patch(monkeypatch, found):
    lab = _FakeLab(found)
    monkeypatch.setattr(cli_mod, "_lab", lambda backend="local": lab)
    return lab


class TestUnsupervisedRemediationIsApproved:
    def test_an_unsupervised_job_appears_in_the_destroy_preview(self, monkeypatch):
        """It is torn down, so it must be listed among what will be torn down."""
        _patch(monkeypatch, {"unsupervised": [UNSUP]})

        result = runner_cli.invoke(app, ["reconcile", "--apply"], input="y\n")

        preview = result.stderr.split("about to destroy")[1].split("proceed?")[0]
        assert UNSUP["job_id"] in preview or UNSUP["cluster"] in preview

    def test_declining_the_prompt_remediates_nothing(self, monkeypatch):
        """The whole point: `no` must reach no teardown, for this pass like every other."""
        lab = _patch(monkeypatch, {"unsupervised": [UNSUP]})

        result = runner_cli.invoke(app, ["reconcile", "--apply"], input="n\n")

        assert result.exit_code == 4
        assert lab.calls == [False], "the applying pass must never have run"

    def test_an_unsupervised_job_alone_still_prompts(self, monkeypatch):
        """With no other orphans the prompt used to be skipped entirely, and machines still died."""
        _patch(monkeypatch, {"unsupervised": [UNSUP]})

        result = runner_cli.invoke(app, ["reconcile", "--apply"], input="n\n")

        assert "about to destroy" in result.stderr

    def test_only_the_approved_unsupervised_entry_is_remediated(self, monkeypatch):
        other = {"job_id": "other", "cluster": "lab-p-other"}
        lab = _patch(monkeypatch, {"unsupervised": [UNSUP, other]})

        runner_cli.invoke(app, ["reconcile", "--apply"], input="y\n")

        from lab.core import orphan_key

        assert lab.only is not None
        assert orphan_key("unsupervised", UNSUP) in lab.only

    def test_a_dry_run_reports_it_as_action_required(self, monkeypatch):
        """An unsupervised job's cluster may be billing; a dry run must not exit 0 on it."""
        _patch(monkeypatch, {"unsupervised": [UNSUP]})

        assert runner_cli.invoke(app, ["reconcile"]).exit_code == 3


def test_orphan_key_identifies_an_unsupervised_entry(tmp_path):
    """`{job_id, cluster}` has neither `name` nor `id`, so the shared identity helper must know it.

    Without this the approval set contains `unsupervised:None` and every entry collides, so
    approving one would approve them all.
    """
    from lab.core import orphan_key

    a = orphan_key("unsupervised", {"job_id": "j1", "cluster": "lab-p-j1"})
    b = orphan_key("unsupervised", {"job_id": "j2", "cluster": "lab-p-j2"})

    assert a != b
    assert "None" not in a


# ---------------------------------------------------------------------------
# Remaining review findings (2026-08-21).
# ---------------------------------------------------------------------------


class TestOwnershipFallsBackToTheLocalJobStore:
    """The most authoritative source for a single project is its own `runs/` directory.

    Attribution consulted only the machine-wide index (written from this release onward) and the
    event ledger (retention-bounded, and disabled entirely by `LAB_EVENTS=0`). So a cluster leaked
    by a job submitted *before* this release became permanently unattributable, and `--apply`
    silently refused to clean it up — reported under `unattributed`, which does not exit 3 either.
    A job in this project's own store is ours by definition.
    """

    def test_a_job_in_our_own_runs_dir_is_attributable(self, tmp_path, monkeypatch):
        from helpers import make_manifest
        from lab.backends.local import LocalBackend
        from lab.core import Lab
        from lab.models import BackendInfo

        lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
        m = make_manifest("20260819-101010-abcdef", "python x.py", timeout="1h").model_copy(
            update={"status": JobState.succeeded, "backend": BackendInfo(provisioner="skypilot")}
        )
        lab.store.create(m)

        monkeypatch.setattr("lab.core.attribute_jobs", lambda ids: {})   # index + ledger blind
        monkeypatch.setattr("lab.core.local_project", lambda: "laboratory")
        monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
        _patch_sky(monkeypatch, ["lab-20260819-101010-abcdef"])

        report = lab.reconcile(apply=False)

        assert report["sky_orphans"] == ["lab-20260819-101010-abcdef"]
        assert report["unattributed"] == []


class TestGcpAttributionUsesLabels:
    """GCP truncates the cluster name to 35 chars, shearing off the job id.

    The `lab-project`/`lab-job-id` instance labels were added precisely because they survive that
    truncation — but nothing read them, so every GCP orphan landed in `unattributed` and could
    never be destroyed.
    """

    def test_a_truncated_gcp_name_is_attributed_from_its_labels(self, tmp_path, monkeypatch):
        from lab.backends.local import LocalBackend
        from lab.core import Lab

        lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
        truncated = "lab-laboratory-20260820-ef-3dd12990-head-4oq7tgke-compute"
        inst = {
            "name": truncated,
            "zone": "us-central1-a",
            "status": "RUNNING",
            "labels": {"lab-project": "laboratory", "lab-job-id": "20260820-071905-771110"},
        }
        monkeypatch.setattr("lab.core.attribute_jobs", lambda ids: {})
        monkeypatch.setattr("lab.core.local_project", lambda: "laboratory")
        monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.list_gcp_instances", lambda *a, **k: [inst])
        monkeypatch.setattr("lab.backends.skypilot.list_gcp_disks", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.gcp_project", lambda: "p")
        _patch_sky(monkeypatch, [])

        report = lab.reconcile(apply=False)

        assert [o["name"] for o in report["gcp_orphans"]] == [truncated]
        assert report["unattributed"] == []

    def test_another_projects_label_is_protected(self, tmp_path, monkeypatch):
        from lab.backends.local import LocalBackend
        from lab.core import Lab

        lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
        inst = {
            "name": "lab-tempotron-cap-20260820-ef-3dd12990-head-4oq7tgke-compute",
            "zone": "us-central1-a",
            "status": "RUNNING",
            "labels": {"lab-project": "tempotron-capacity", "lab-job-id": "20260820-124800-05befa"},
        }
        monkeypatch.setattr("lab.core.attribute_jobs", lambda ids: {})
        monkeypatch.setattr("lab.core.local_project", lambda: "laboratory")
        monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.list_gcp_instances", lambda *a, **k: [inst])
        monkeypatch.setattr("lab.backends.skypilot.list_gcp_disks", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.gcp_project", lambda: "p")
        _patch_sky(monkeypatch, [])

        report = lab.reconcile(apply=False)

        assert report["gcp_orphans"] == []
        assert [r["project"] for r in report["other_projects"]] == ["tempotron-capacity"]


def test_a_respawned_supervisor_clears_the_stale_exit_record():
    """`runner_exit` is permanent once written, and `status()` reads it as proof of death.

    So a job whose first supervisor died, then was respawned with `--adopt` by the watchdog, gets
    torn down on the very next status poll — destroying a live cluster and failing a running job.
    """
    import ast
    from pathlib import Path

    src = Path("src/lab/scheduler/tick.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "write_runtime":
            kwargs = {kw.arg for kw in node.keywords}
            if "runner_pid" in kwargs:
                assert "runner_exit" in kwargs, (
                    "a respawn must clear the previous supervisor's exit record, or status() "
                    "reads the new supervisor as already dead and destroys its cluster"
                )
                return
    raise AssertionError("no respawn write_runtime call found")


def _patch_sky(monkeypatch, clusters):
    import sys
    import types

    fake = types.ModuleType("sky")
    fake.get = lambda x: x
    fake.status = lambda refresh=None: [{"name": c} for c in clusters]
    fake.down = lambda c: None
    fake.StatusRefreshMode = types.SimpleNamespace(AUTO="AUTO", FORCE="FORCE", NONE="NONE")
    monkeypatch.setitem(sys.modules, "sky", fake)


class TestDoTeardownDoesNotManufactureAlarms:
    """Two ways the DO paths raised alarms they could not support.

    A `sky.down` that *succeeded* got downgraded to `unknown` whenever the DO client could not be
    built at all — pydo absent, no token in the supervisor's environment. That is a false alarm on
    the healthy path, and "an alarm that is usually wrong stops being an alarm" is the very thing
    the three-state work set out to fix. Not being *configured* for DO is a different fact from a
    listing that failed.
    """

    def test_an_unconfigured_do_client_leaves_a_clean_teardown_alone(self, monkeypatch):
        from lab.backends import skypilot as m

        def _not_configured(*a, **k):
            raise ImportError("No module named 'pydo'")

        monkeypatch.setattr(m, "_get_do_client", _not_configured)

        class _OkSky:
            def get(self, x):
                return x

            def down(self, cluster):
                return None

        out = m.robust_teardown(_OkSky(), "lab-x", cloud="do", backoffs=(0,))

        assert out["status"] == "succeeded"

    def test_a_failed_listing_still_yields_unknown(self, monkeypatch):
        """Configured but unable to answer is genuinely unknown — that distinction is the point."""
        from lab.backends import skypilot as m

        class _Client:
            class volumes:
                @staticmethod
                def list(**kw):
                    raise RuntimeError("503 Service Unavailable")

        monkeypatch.setattr(m, "_get_do_client", lambda: _Client())

        class _OkSky:
            def get(self, x):
                return x

            def down(self, cluster):
                return None

        out = m.robust_teardown(_OkSky(), "lab-x", cloud="do", backoffs=(0,))

        assert out["status"] == "unknown"


class TestVersionGateOnlyBlocksTheSkyPasses:
    """A skewed sky client says nothing about the Vast, DO or GCP provider-direct passes.

    Blocking the whole sweep meant the emergency leak-stopping command refused to destroy anything
    at all — including passes that talk straight to the provider — precisely when the local
    SkyPilot API server is unhealthy, which is when leaks are most likely.
    """

    def test_apply_still_runs_the_provider_passes_under_skew(self, tmp_path, monkeypatch):
        from lab.backends.local import LocalBackend
        from lab.core import Lab

        lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
        from lab._skycompat import SkyVersions

        monkeypatch.setattr(
            "lab._skycompat.sky_versions",
            lambda **kw: SkyVersions(
                client="0.12.3", server="0.13.0", compatible=False, detail="upgrade"
            ),
        )
        monkeypatch.setattr("lab.core.attribute_jobs", lambda ids: {})
        monkeypatch.setattr("lab.core.local_project", lambda: "laboratory")
        monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
        _patch_sky(monkeypatch, [])

        report = lab.reconcile(apply=True)

        assert report["applied"] is True
        assert report["sky_pass"] == "skipped (client/server version skew)"

    def test_the_sky_destroy_pass_is_still_refused(self, tmp_path, monkeypatch):
        """What must NOT happen: destroying via a client that cannot read the result."""
        from lab.backends.local import LocalBackend
        from lab.core import Lab
        from lab._skycompat import SkyVersions

        lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
        monkeypatch.setattr(
            "lab._skycompat.sky_versions",
            lambda **kw: SkyVersions(
                client="0.12.3", server="0.13.0", compatible=False, detail="upgrade"
            ),
        )
        monkeypatch.setattr("lab.core.attribute_jobs", lambda ids: {})
        monkeypatch.setattr("lab.core.local_project", lambda: "laboratory")
        monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])

        def _must_not_run(cluster):
            raise AssertionError("destroyed through a client that cannot read the result")

        _patch_sky(monkeypatch, ["lab-laboratory-20260820-071905-771110"])
        import sys

        sys.modules["sky"].down = _must_not_run

        report = lab.reconcile(apply=True)

        assert report["sky_destroyed"] == []
