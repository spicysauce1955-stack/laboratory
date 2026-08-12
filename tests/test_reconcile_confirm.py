"""`lab reconcile --apply` must ask before destroying (GCP-LEAK-7, second half).

The narrowed node predicate makes a destructive false positive far less likely; it does not make
one *recoverable*. A confirmation is the difference between a bug in `is_lab_cluster_node` costing
someone a re-launch and costing them a VM they cared about.

Deliberately CLI-only: the scheduler's unattended sweep calls ``Lab.reconcile(apply=...)`` through
the library, which is untouched, so ``auto_reconcile`` keeps working with no prompt to hang on.
"""

import json

import pytest
from typer.testing import CliRunner

import lab.cli as cli_mod
from lab.cli import app

runner = CliRunner()

ORPHAN_INSTANCE = {
    "name": "lab-20260811-144501-c5b340-3dd12990-head-c0h9pkx0-compute",
    "zone": "us-central1-a",
    "status": "RUNNING",
}
_EMPTY = {
    "vast_pass": "ran", "gcp_pass": "ran", "gcp_disk_pass": "ran",
    "gcp_project": "myproject-505213", "gcp_unmatched": [], "instances_total": 0,
    "unsupervised": [], "orphans": [], "destroyed": [], "ghosts": [],
    "sky_orphans": [], "sky_destroyed": [], "do_volume_orphans": [], "do_volumes_destroyed": [],
    "gcp_orphans": [], "gcp_destroyed": [], "gcp_disk_orphans": [], "gcp_disks_destroyed": [],
    "applied": False,
}


def _report(**over):
    return {**_EMPTY, **over}


class _FakeLab:
    """Records every reconcile call so a test can assert nothing was destroyed."""

    def __init__(self, found):
        self.found = found
        self.calls: list[bool] = []

    def reconcile(self, apply=False, only=None):
        self.calls.append(apply)
        self.only = only
        return _report(**self.found, applied=apply)


@pytest.fixture(autouse=True)
def _interactive(monkeypatch):
    """Default every test to 'a human is watching'; the non-interactive cases override it."""
    monkeypatch.setattr(cli_mod, "_stdin_is_a_tty", lambda: True)


def _patch(monkeypatch, found):
    lab = _FakeLab(found)
    monkeypatch.setattr(cli_mod, "_lab", lambda backend="local": lab)
    return lab


def test_declining_the_prompt_destroys_nothing(monkeypatch):
    """The point of the whole exercise: answering no must not reach a destroy."""
    lab = _patch(monkeypatch, {"gcp_orphans": [ORPHAN_INSTANCE]})

    result = runner.invoke(app, ["reconcile", "--apply"], input="n\n")

    assert True not in lab.calls, "a declined confirmation still ran a destructive pass"
    assert result.exit_code != 0


def test_accepting_the_prompt_applies(monkeypatch):
    lab = _patch(monkeypatch, {"gcp_orphans": [ORPHAN_INSTANCE]})

    result = runner.invoke(app, ["reconcile", "--apply"], input="y\n")

    assert True in lab.calls
    assert result.exit_code == 0


def test_yes_skips_the_prompt(monkeypatch):
    """Scriptable: the scheduler host and any cron wrapper must not hang on a tty read."""
    lab = _patch(monkeypatch, {"gcp_orphans": [ORPHAN_INSTANCE]})

    result = runner.invoke(app, ["reconcile", "--apply", "--yes"], input="")

    assert True in lab.calls
    assert result.exit_code == 0


def test_nothing_to_destroy_does_not_prompt(monkeypatch):
    """No needless friction — and no prompt to hang on when a clean sweep runs unattended."""
    lab = _patch(monkeypatch, {})

    result = runner.invoke(app, ["reconcile", "--apply"], input="")

    assert True in lab.calls
    assert result.exit_code == 0


def test_the_prompt_names_what_it_will_destroy(monkeypatch):
    """A confirmation that doesn't say what it is destroying trains people to hit y.

    Asserted on **stderr** specifically: that is where the preview belongs, and asserting on
    stdout would pass on the JSON report alone without any prompt existing.
    """
    _patch(monkeypatch, {"gcp_orphans": [ORPHAN_INSTANCE], "orphans": [{"id": 12345}]})

    result = runner.invoke(app, ["reconcile", "--apply"], input="n\n")

    assert "lab-20260811-144501-c5b340-3dd12990-head-c0h9pkx0-compute" in result.stderr
    assert "12345" in result.stderr
    assert "myproject-505213" in result.stderr  # which project, per GCP-LEAK-7


def test_stdout_stays_parseable_json(monkeypatch):
    """Diagnostics go to stderr; stdout carries only JSON, which callers parse. A prompt written
    to stdout would corrupt the report for every scripted caller."""
    _patch(monkeypatch, {"gcp_orphans": [ORPHAN_INSTANCE]})

    result = runner.invoke(
        app, ["reconcile", "--apply", "--yes"], input="", catch_exceptions=False
    )

    json.loads(result.stdout)  # raises if the prompt or preview leaked into stdout


# --- review findings ---------------------------------------------------------------------------


def test_no_tty_refuses_with_json_and_exit_4(monkeypatch):
    """Review finding: `typer.confirm` raises `click.Abort` on EOF, so every non-interactive
    caller (cron, CI, an agent shell — and every documented recovery recipe, none of which pass
    `--yes`) exited 1 with **no JSON at all**, breaking the stdout-is-JSON contract and reporting
    a generic error rather than the documented 'declined' code."""
    monkeypatch.setattr(cli_mod, "_stdin_is_a_tty", lambda: False)
    lab = _patch(monkeypatch, {"gcp_orphans": [ORPHAN_INSTANCE]})

    result = runner.invoke(app, ["reconcile", "--apply"], input="")

    assert True not in lab.calls
    assert result.exit_code == 4
    body = json.loads(result.stdout)
    assert body["aborted"] is True and body["reason"] == "no tty"
    assert "--yes" in result.stderr  # tell the operator how to proceed


def test_no_tty_with_yes_still_applies(monkeypatch):
    """--yes is the documented unattended path and must not need a terminal."""
    monkeypatch.setattr(cli_mod, "_stdin_is_a_tty", lambda: False)
    lab = _patch(monkeypatch, {"gcp_orphans": [ORPHAN_INSTANCE]})

    result = runner.invoke(app, ["reconcile", "--apply", "--yes"], input="")

    assert True in lab.calls
    assert result.exit_code == 0


def test_only_the_approved_set_is_destroyed(monkeypatch):
    """Review finding: the confirmed pass is a second, independent sweep, so a resource that
    becomes an orphan *between* the preview and the destroy was destroyed without ever being
    shown — e.g. a running job whose supervisor pid dies in that window, dropping its live
    cluster out of `running_clusters`. The approval is now binding."""
    seen: list[set[str] | None] = []

    class _DriftingLab:
        """Second sweep finds an extra orphan the operator never saw."""

        def reconcile(self, apply=False, only=None):
            seen.append(only)
            found = [ORPHAN_INSTANCE] if not apply else [ORPHAN_INSTANCE, {
                "name": "lab-20260811-999999-ffffff-3dd12990-head-zzzzzzzz-compute",
                "zone": "us-central1-b", "status": "RUNNING",
            }]
            return _report(gcp_orphans=found, applied=apply)

    monkeypatch.setattr(cli_mod, "_lab", lambda backend="local": _DriftingLab())

    runner.invoke(app, ["reconcile", "--apply"], input="y\n")

    approved = seen[-1]
    assert approved is not None, "the confirmed pass ran unfiltered"
    assert approved == {f"gcp_orphans:{ORPHAN_INSTANCE['name']}"}
    assert not any("999999" in k for k in approved)  # the drifted-in resource is not approved


def test_yes_does_not_filter(monkeypatch):
    """--yes solicits no approval, so there is no approved set to enforce — it must not
    accidentally become an empty allowlist that silently destroys nothing."""
    seen: list[set[str] | None] = []

    class _Lab:
        def reconcile(self, apply=False, only=None):
            seen.append(only)
            return _report(gcp_orphans=[ORPHAN_INSTANCE], applied=apply)

    monkeypatch.setattr(cli_mod, "_lab", lambda backend="local": _Lab())

    runner.invoke(app, ["reconcile", "--apply", "--yes"], input="")

    assert seen == [None]


def test_the_preview_shows_a_vast_label(monkeypatch):
    """Review finding: `_describe_orphan` read a `volume_id` key no pass emits and dropped
    `label` — the only human-readable field the Vast pass produces, and the thing that lets an
    operator recognise which job a rental belonged to."""
    _patch(monkeypatch, {"orphans": [{"id": 12345, "label": "lab-20260811-144501-c5b340"}]})

    result = runner.invoke(app, ["reconcile", "--apply"], input="n\n")

    assert "lab-20260811-144501-c5b340" in result.stderr
    assert "12345" in result.stderr


def test_unmatched_lab_resources_warn_loudly(monkeypatch):
    """Review finding: narrowing the predicate demoted non-node `lab-*` resources to an advisory
    field excluded from the exit-3 alarm, so a genuine leak under an unrecognised name shape made
    reconcile exit 0 in silence. It still must not claim `--apply` fixes it — it can't — so the
    signal is a loud warning, not an exit code."""
    _patch(monkeypatch, {"gcp_unmatched": ["lab-notebook"]})

    result = runner.invoke(app, ["reconcile"])

    assert result.exit_code == 0
    assert "lab-notebook" in result.stderr
    assert "did not match" in result.stderr
