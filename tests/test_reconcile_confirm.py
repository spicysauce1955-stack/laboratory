"""`lab reconcile --apply` must ask before destroying (GCP-LEAK-7, second half).

The narrowed node predicate makes a destructive false positive far less likely; it does not make
one *recoverable*. A confirmation is the difference between a bug in `is_lab_cluster_node` costing
someone a re-launch and costing them a VM they cared about.

Deliberately CLI-only: the scheduler's unattended sweep calls ``Lab.reconcile(apply=...)`` through
the library, which is untouched, so ``auto_reconcile`` keeps working with no prompt to hang on.
"""

import json

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

    def reconcile(self, apply=False):
        self.calls.append(apply)
        return _report(**self.found, applied=apply)


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
