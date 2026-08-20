"""`lab reconcile` must never report a destroy it did not confirm (incident 2026-08-20).

Two failures compounded on 2026-08-20. Reconcile classified seven *running* jobs belonging to a
different project on the same machine as orphans, and destroyed them. Then it reported
``sky_destroyed: []`` and **exited 0** -- because the 0.12.3 sky client could not deserialise the
0.13.0 server's success response, so every `sky.down` surfaced as an exception and was swallowed
by a bare ``print``. The operator was told nothing had happened while seven machines were gone.

The exit-code contract these tests pin:

* ``0``  -- nothing to do, or everything `--apply` attempted was *confirmed* destroyed.
* ``3``  -- orphans found in a dry run (unchanged; re-run with ``--apply``).
* ``4``  -- the confirmation was declined, or there was no tty to ask at (unchanged).
* ``5``  -- ``--apply`` ran and at least one destroy did **not** confirm success. Either it
  genuinely failed, or -- the case that made this incident invisible -- its outcome is *unknown*
  and must be verified against the cloud provider rather than believed either way.

``5`` is deliberately distinct from ``2`` (the command itself errored) and from ``3``: the sweep
worked, the destroy did not, and the cloud is now in a state the tool cannot vouch for.
"""

import json
import os

import pytest
from typer.testing import CliRunner

import lab.cli as cli_mod
from lab.cli import app

runner = CliRunner()

_EMPTY = {
    "vast_pass": "ran", "gcp_pass": "ran", "gcp_disk_pass": "ran",
    "gcp_project": "myproject-505213", "gcp_unmatched": [], "instances_total": 0,
    "unsupervised": [], "orphans": [], "destroyed": [], "ghosts": [],
    "sky_orphans": [], "sky_destroyed": [], "do_volume_orphans": [], "do_volumes_destroyed": [],
    "gcp_orphans": [], "gcp_destroyed": [], "gcp_disk_orphans": [], "gcp_disks_destroyed": [],
    "unattributed": [], "destroy_outcomes": [],
    "applied": False,
}

# The exact shape the live incident produced: the cluster WAS destroyed, the client could not tell.
UNDECODABLE = {
    "pass": "sky_orphans",
    "resource": "lab-20260820-071905-771110",
    "outcome": "unknown",
    "error": (
        "Can't get attribute 'user_initiated_down' on <module 'sky.core' from "
        "'/home/user/.superset/projects/laboratory/.venv/lib/python3.12/site-packages/sky/core.py'>"
    ),
}
GENUINE_FAILURE = {
    "pass": "do_volume_orphans",
    "resource": "lab-20260820-071905-771110-3dd12990-f5bf-head",
    "outcome": "failed",
    "error": "failed to delete volume: attached volume cannot be deleted",
}


def _report(**over):
    return {**_EMPTY, **over}


class _FakeLab:
    def __init__(self, found):
        self.found = found
        self.calls: list[bool] = []

    def reconcile(self, apply=False, only=None):
        self.calls.append(apply)
        self.only = only
        return _report(**self.found, applied=apply)


@pytest.fixture(autouse=True)
def _interactive(monkeypatch):
    monkeypatch.setattr(cli_mod, "_stdin_is_a_tty", lambda: True)


def _patch(monkeypatch, found):
    lab = _FakeLab(found)
    monkeypatch.setattr(cli_mod, "_lab", lambda backend="local": lab)
    return lab


class TestUnconfirmedDestroyExitCode:
    def test_an_unknown_destroy_outcome_exits_5(self, monkeypatch):
        """The incident, exactly: the destroy happened, the client could not confirm it."""
        _patch(monkeypatch, {"destroy_outcomes": [UNDECODABLE]})

        result = runner.invoke(app, ["reconcile", "--apply", "--yes"])

        assert result.exit_code == 5, result.output

    def test_a_genuine_destroy_failure_exits_5(self, monkeypatch):
        _patch(monkeypatch, {"destroy_outcomes": [GENUINE_FAILURE]})

        result = runner.invoke(app, ["reconcile", "--apply", "--yes"])

        assert result.exit_code == 5, result.output

    def test_a_fully_confirmed_apply_exits_0(self, monkeypatch):
        """No regression: a clean cleanup must stay exit 0, or every wrapper breaks."""
        _patch(
            monkeypatch,
            {"sky_orphans": ["lab-x"], "sky_destroyed": ["lab-x"], "destroy_outcomes": []},
        )

        result = runner.invoke(app, ["reconcile", "--apply", "--yes"])

        assert result.exit_code == 0, result.output

    def test_unconfirmed_outcome_outranks_the_dry_run_signal(self, monkeypatch):
        """`--apply` with leftovers must not exit 3 ('re-run with --apply') -- it just did."""
        _patch(
            monkeypatch,
            {"sky_orphans": ["lab-x"], "sky_destroyed": [], "destroy_outcomes": [UNDECODABLE]},
        )

        result = runner.invoke(app, ["reconcile", "--apply", "--yes"])

        assert result.exit_code == 5, result.output

    def test_dry_run_is_unaffected(self, monkeypatch):
        """A dry run destroys nothing, so it can never have an unconfirmed outcome: still 3."""
        _patch(monkeypatch, {"sky_orphans": ["lab-x"]})

        result = runner.invoke(app, ["reconcile"])

        assert result.exit_code == 3, result.output


class TestUnconfirmedDestroyIsVisible:
    def test_stdout_stays_parseable_json(self, monkeypatch):
        """Callers parse stdout (CLAUDE.md conventions); diagnostics belong on stderr."""
        _patch(monkeypatch, {"destroy_outcomes": [UNDECODABLE]})

        result = runner.invoke(app, ["reconcile", "--apply", "--yes"])

        payload = json.loads(result.stdout)
        assert payload["destroy_outcomes"][0]["outcome"] == "unknown"

    def test_the_unknown_case_says_to_verify_against_the_provider(self, monkeypatch):
        """An 'unknown' outcome is useless unless it tells the reader what to actually do.

        The operator's mistake in the incident was believing `sky_destroyed: []`. The message has
        to state that the resource may well be gone *and* may well be running.
        """
        _patch(monkeypatch, {"destroy_outcomes": [UNDECODABLE]})

        result = runner.invoke(app, ["reconcile", "--apply", "--yes"])

        stderr = result.stderr.lower()
        assert "unknown" in stderr
        assert "verify" in stderr
        assert "lab-20260820-071905-771110" in result.stderr

    def test_every_unconfirmed_resource_is_named(self, monkeypatch):
        _patch(monkeypatch, {"destroy_outcomes": [UNDECODABLE, GENUINE_FAILURE]})

        result = runner.invoke(app, ["reconcile", "--apply", "--yes"])

        assert UNDECODABLE["resource"] in result.stderr
        assert GENUINE_FAILURE["resource"] in result.stderr


class TestUnattributedIsNeverDestroyed:
    """A `lab-*` resource no known job store claims is reported, warned about -- never destroyed.

    This is the inversion the incident argues for: the old predicate was 'not in *this project's*
    job list => orphan => destroy', which treats absence of evidence as evidence of orphanhood on
    a destructive path. `gcp_unmatched` already had exactly this treatment; the sky pass did not.
    """

    def test_unattributed_resources_do_not_trigger_the_dry_run_orphan_exit(self, monkeypatch):
        """They are not actionable via `--apply`, so exit 3's 'now re-run with --apply' is wrong
        advice -- the same reasoning that keeps `gcp_unmatched` out of ``_ORPHAN_FIELDS``."""
        _patch(monkeypatch, {"unattributed": ["lab-20260820-071905-771110"]})

        result = runner.invoke(app, ["reconcile"])

        assert result.exit_code == 0, result.output

    def test_unattributed_resources_warn_loudly_on_stderr(self, monkeypatch):
        _patch(monkeypatch, {"unattributed": ["lab-20260820-071905-771110"]})

        result = runner.invoke(app, ["reconcile"])

        assert "lab-20260820-071905-771110" in result.stderr
        assert "not" in result.stderr.lower()

    def test_unattributed_resources_are_not_in_the_destroy_preview(self, monkeypatch):
        """The preview is the human's last line of defence; it must list only what will die."""
        lab = _patch(
            monkeypatch,
            {"unattributed": ["lab-unknown-owner"], "sky_orphans": ["lab-mine"]},
        )

        result = runner.invoke(app, ["reconcile", "--apply"], input="y\n")

        # Only the block between the preview header and the prompt counts: the unattributed
        # warning is printed later, after the report, and naming them there is the point.
        preview = result.stderr.split("about to destroy")[1].split("proceed?")[0]
        assert "lab-mine" in preview
        assert "lab-unknown-owner" not in preview
        assert lab.calls == [False, True]


# ---------------------------------------------------------------------------
# Library level: `Lab.reconcile` must report the outcome of every destroy it attempts.
# ---------------------------------------------------------------------------

from helpers import make_manifest  # noqa: E402
from lab.backends.local import LocalBackend  # noqa: E402
from lab.core import Lab  # noqa: E402


OURS_ID = "20260820-999999-abcdef"
OURS = f"lab-{OURS_ID}"


def _patch_sky(monkeypatch, *, clusters, down):
    """A fake `sky` exposing `clusters` and routing `down` through the supplied callable."""
    import sys
    import types

    fake = types.ModuleType("sky")
    fake.get = lambda x: x
    fake.status = lambda refresh=None: [{"name": c} for c in clusters]
    fake.down = down
    fake.StatusRefreshMode = types.SimpleNamespace(AUTO="AUTO", FORCE="FORCE", NONE="NONE")
    monkeypatch.setitem(sys.modules, "sky", fake)


def _compatible_sky(monkeypatch):
    """Satisfy the version-skew gate. The fake `sky` module a test installs has no `sky.server`,
    so the real probe honestly reports "cannot determine" -- which is (correctly) not compatible."""
    from lab._skycompat import SkyVersions

    monkeypatch.setattr(
        "lab._skycompat.sky_versions",
        lambda **kw: SkyVersions(client="0.12.3", server="0.12.3", compatible=True, detail="ok"),
    )


def _lab_with_no_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
    _compatible_sky(monkeypatch)
    monkeypatch.setattr("lab.core.attribute_jobs", lambda ids: {})
    monkeypatch.setattr("lab.core.local_project", lambda: "laboratory")
    return Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)


class TestDestroyOutcomesAreReported:
    def test_a_raising_destroy_is_reported_not_swallowed(self, tmp_path, monkeypatch):
        """The incident's silence: the failure went to `print` and nothing reached the report."""
        lab = _lab_with_no_jobs(tmp_path, monkeypatch)
        _patch_attribution(monkeypatch, {OURS_ID: "laboratory"})

        def _boom(cluster):
            raise RuntimeError("could not reach the API server")

        _patch_sky(monkeypatch, clusters=[OURS], down=_boom)

        report = lab.reconcile(apply=True)

        assert report["sky_destroyed"] == []
        outcomes = report["destroy_outcomes"]
        assert len(outcomes) == 1
        assert outcomes[0]["resource"] == OURS
        assert outcomes[0]["pass"] == "sky_orphans"
        assert "could not reach the API server" in outcomes[0]["error"]

    def test_a_successful_destroy_reports_no_outcome_entry(self, tmp_path, monkeypatch):
        lab = _lab_with_no_jobs(tmp_path, monkeypatch)
        _patch_attribution(monkeypatch, {OURS_ID: "laboratory"})
        _patch_sky(monkeypatch, clusters=[OURS], down=lambda cluster: None)

        report = lab.reconcile(apply=True)

        assert report["sky_destroyed"] == [OURS]
        assert report["destroy_outcomes"] == []

    def test_a_dry_run_attempts_nothing_so_reports_no_outcomes(self, tmp_path, monkeypatch):
        lab = _lab_with_no_jobs(tmp_path, monkeypatch)

        def _boom(cluster):
            raise AssertionError("a dry run must never call down()")

        _patch_sky(monkeypatch, clusters=[OURS], down=_boom)

        report = lab.reconcile(apply=False)

        assert report["destroy_outcomes"] == []

    def test_a_running_local_job_still_protects_its_cluster(self, tmp_path, monkeypatch):
        """No regression on the one attribution the old code did get right."""
        from lab.backends.skypilot import cluster_name_for
        from lab.models import BackendInfo, JobState

        lab = _lab_with_no_jobs(tmp_path, monkeypatch)
        m = make_manifest("jlive", "python x.py", timeout="1h").model_copy(
            update={"status": JobState.running, "backend": BackendInfo(provisioner="skypilot")}
        )
        lab.store.create(m)
        lab.store.write_runtime("jlive", runner_pid=os.getpid(), cluster=cluster_name_for("jlive"))

        def _boom(cluster):
            raise AssertionError(f"must not destroy a live job's cluster: {cluster}")

        _patch_sky(monkeypatch, clusters=[cluster_name_for("jlive")], down=_boom)

        report = lab.reconcile(apply=True)

        assert report["sky_orphans"] == []
        assert report["destroy_outcomes"] == []


# ---------------------------------------------------------------------------
# Attribution: a resource this project cannot prove it owns is never destroyed.
# ---------------------------------------------------------------------------

from lab.attribution import Attribution  # noqa: E402


def _patch_attribution(monkeypatch, mapping, *, me="laboratory"):
    """Route attribution through a fake so tests never touch the real `~/.lab`."""
    def _fake(job_ids):
        out = {}
        for jid in job_ids:
            project = mapping.get(jid)
            out[jid] = Attribution(
                job_id=jid,
                project=project,
                runs_dir=None,
                source="ledger" if project else "unknown",
            )
        return out

    monkeypatch.setattr("lab.core.attribute_jobs", _fake)
    monkeypatch.setattr("lab.core.local_project", lambda: me)


class TestForeignResourcesAreNeverDestroyed:
    """The incident in one class: another project's live cluster must survive `--apply --yes`."""

    def test_a_cluster_owned_by_another_project_is_not_destroyed(self, tmp_path, monkeypatch):
        lab = _lab_with_no_jobs(tmp_path, monkeypatch)
        _patch_attribution(monkeypatch, {"20260820-071905-771110": "tempotron-capacity"})

        def _boom(cluster):
            raise AssertionError(f"destroyed another project's cluster: {cluster}")

        _patch_sky(monkeypatch, clusters=["lab-20260820-071905-771110"], down=_boom)

        report = lab.reconcile(apply=True)

        assert report["sky_orphans"] == []
        assert report["sky_destroyed"] == []
        assert report["other_projects"] == [
            {
                "pass": "sky_orphans",
                "resource": "lab-20260820-071905-771110",
                "project": "tempotron-capacity",
            }
        ]

    def test_a_volume_named_after_another_projects_job_is_not_deleted(self, tmp_path, monkeypatch):
        """The DO volume pass is how the incident's volumes actually died -- it matches on a
        name with SkyPilot's own suffixes appended, so attribution must see through those."""
        lab = _lab_with_no_jobs(tmp_path, monkeypatch)
        _patch_attribution(monkeypatch, {"20260820-071905-771110": "tempotron-capacity"})
        volume = {
            "id": "71d631a8-9c67-11f1-9b51-1622bf496a70",
            "name": "lab-20260820-071905-771110-3dd12990-f5bf-head",
            "droplet_ids": [],
        }
        monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [volume])

        def _client():
            raise AssertionError("must not reach the DO client for a foreign volume")

        monkeypatch.setattr("lab.backends.skypilot._get_do_client", _client)
        _patch_sky(monkeypatch, clusters=[], down=lambda c: None)

        report = lab.reconcile(apply=True)

        assert report["do_volume_orphans"] == []
        assert report["do_volumes_destroyed"] == []
        assert report["other_projects"][0]["project"] == "tempotron-capacity"

    def test_an_unattributable_cluster_is_reported_not_destroyed(self, tmp_path, monkeypatch):
        lab = _lab_with_no_jobs(tmp_path, monkeypatch)
        _patch_attribution(monkeypatch, {})  # nothing resolves

        def _boom(cluster):
            raise AssertionError(f"destroyed an unattributable cluster: {cluster}")

        _patch_sky(monkeypatch, clusters=["lab-whoknows"], down=_boom)

        report = lab.reconcile(apply=True)

        assert report["unattributed"] == ["lab-whoknows"]
        assert report["sky_orphans"] == []

    def test_our_own_orphan_is_still_destroyed(self, tmp_path, monkeypatch):
        """No regression: the whole point of reconcile still has to work."""
        lab = _lab_with_no_jobs(tmp_path, monkeypatch)
        _patch_attribution(monkeypatch, {"20260820-999999-abcdef": "laboratory"})
        destroyed = []
        _patch_sky(
            monkeypatch,
            clusters=["lab-20260820-999999-abcdef"],
            down=lambda c: destroyed.append(c),
        )

        report = lab.reconcile(apply=True)

        assert report["sky_orphans"] == ["lab-20260820-999999-abcdef"]
        assert destroyed == ["lab-20260820-999999-abcdef"]
        assert report["other_projects"] == []


class TestVersionSkewBlocksDestruction:
    """Under skew the client cannot read the result of a destroy, so it must not destroy.

    Dry-run stays available on purpose: reading state is still useful, and a leak detector that
    refuses to *look* is worse than one that refuses to act.
    """

    def _skewed(self, monkeypatch):
        from lab._skycompat import SkyVersions

        monkeypatch.setattr(
            "lab._skycompat.sky_versions",
            lambda **kw: SkyVersions(
                client="0.12.3", server="0.13.0", compatible=False, detail="upgrade the client"
            ),
        )

    def test_apply_refuses_under_skew(self, tmp_path, monkeypatch):
        from lab._skycompat import SkyVersionSkewError

        lab = _lab_with_no_jobs(tmp_path, monkeypatch)
        self._skewed(monkeypatch)

        def _boom(cluster):
            raise AssertionError("must not destroy anything under version skew")

        _patch_sky(monkeypatch, clusters=["lab-x"], down=_boom)

        with pytest.raises(SkyVersionSkewError):
            lab.reconcile(apply=True)

    def test_dry_run_still_works_under_skew(self, tmp_path, monkeypatch):
        lab = _lab_with_no_jobs(tmp_path, monkeypatch)
        self._skewed(monkeypatch)
        monkeypatch.setattr("lab.core.attribute_jobs", lambda ids: {})
        _patch_sky(monkeypatch, clusters=["lab-x"], down=lambda c: None)

        report = lab.reconcile(apply=False)

        assert report["unattributed"] == ["lab-x"]


class TestForeignResourcesAreVisible:
    def test_other_projects_are_noted_on_stderr_with_their_owner(self, monkeypatch):
        """Silence about another project's resources is what made destroying them look fine."""
        _patch(
            monkeypatch,
            {"other_projects": [
                {"pass": "sky_orphans", "resource": "lab-x", "project": "tempotron-capacity"}
            ]},
        )

        result = runner.invoke(app, ["reconcile"])

        assert result.exit_code == 0, result.output
        assert "tempotron-capacity" in result.stderr

    def test_other_projects_do_not_trigger_the_orphan_exit(self, monkeypatch):
        _patch(
            monkeypatch,
            {"other_projects": [
                {"pass": "sky_orphans", "resource": "lab-x", "project": "other"}
            ]},
        )

        assert runner.invoke(app, ["reconcile"]).exit_code == 0


class TestSkewRefusalIsStructured:
    """A refusal must be machine-readable: callers parse stdout, and a traceback is not an answer."""

    def _skew_lab(self, monkeypatch):
        from lab._skycompat import SkyVersions, SkyVersionSkewError

        versions = SkyVersions(
            client="0.12.3", server="0.13.0", compatible=False, detail="upgrade the client"
        )

        class _Skewed:
            def reconcile(self, apply=False, only=None):
                if apply:
                    raise SkyVersionSkewError(versions)
                return _report()

        monkeypatch.setattr(cli_mod, "_lab", lambda backend="local": _Skewed())

    def test_apply_under_skew_exits_4_with_json(self, monkeypatch):
        self._skew_lab(monkeypatch)

        result = runner.invoke(app, ["reconcile", "--apply", "--yes"])

        assert result.exit_code == 4, result.output
        payload = json.loads(result.stdout)
        assert payload["aborted"] is True
        assert payload["reason"] == "sky version skew"
        assert payload["versions"]["server"] == "0.13.0"

    def test_the_refusal_names_the_remedy_on_stderr(self, monkeypatch):
        self._skew_lab(monkeypatch)

        result = runner.invoke(app, ["reconcile", "--apply", "--yes"])

        assert "upgrade the client" in result.stderr
