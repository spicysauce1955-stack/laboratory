"""``reconcile --apply`` always returns its report, even when one pass is blind (review finding).

The destroy passes run in sequence and the report was only built at the end, so a listing that
failed after earlier passes had destroyed things (a GCP 5xx, a sky.status failure) raised
``LabError``: the CLI printed ``{"error": ...}`` and exited 2. What *was* destroyed, and which
destroys had an unknown outcome, was never reported — and the exit 5 those unknowns demand was
lost with it.

Now every pass's failure is collected. Under ``--apply`` the report always comes back, carries
``pass_errors`` (additive), and lists each blind pass in ``destroy_outcomes`` as ``unknown`` — so
the CLI's documented exit 5 ("a destroy did not confirm success", docs/COMPATIBILITY.md) fires:
whatever a pass that could not list covers may still be billing. A dry run destroys nothing, so
it keeps the documented exit 2 for a pass that could not answer: reporting clean while blind is
the one thing a leak check must never do.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import lab.backends.skypilot as sp
import lab.cli as cli_mod
import lab.core as core_mod
from helpers import make_manifest
from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab, LabError
from lab.models import JobState

PROJECT = "proj"
JOB = "20261119-101010-abcdef"


def _boom(msg: str) -> Any:
    def _raise(*a: Any, **k: Any) -> Any:
        raise RuntimeError(msg)

    return _raise


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lab:
    home = tmp_path / "runs"
    lab = Lab(backend=LocalBackend(home=home, repo=tmp_path), repo=tmp_path, home=home)
    lab.store.create(make_manifest(JOB, "python x.py").model_copy(update={"status": JobState.failed}))
    monkeypatch.setattr(core_mod, "local_project", lambda: PROJECT)
    monkeypatch.setattr(sp, "list_vast_instances", lambda *a, **k: [])
    monkeypatch.setattr(sp, "list_do_volumes", lambda *a, **k: [])
    monkeypatch.setattr(sp, "list_gcp_instances", lambda *a, **k: [])
    monkeypatch.setattr(sp, "list_gcp_disks", lambda *a, **k: [])
    monkeypatch.setattr(sp, "gcp_project", lambda: "p")
    return lab


def _sky(monkeypatch: pytest.MonkeyPatch, clusters: list[str], down: Any) -> None:
    fake = types.ModuleType("sky")
    fake.get = lambda x: x  # type: ignore[attr-defined]
    fake.status = lambda refresh=None: [{"name": c} for c in clusters]  # type: ignore[attr-defined]
    fake.down = down  # type: ignore[attr-defined]
    fake.StatusRefreshMode = types.SimpleNamespace(AUTO="AUTO")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky", fake)


def test_unknown_destroy_then_a_gcp_hiccup_still_reports_and_exits_5(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's reproduction, end to end through the CLI."""
    cluster = sp.cluster_name_for(JOB, project=PROJECT)

    def _down(c: str) -> None:
        raise AttributeError(
            "Can't get attribute 'user_initiated_down' on <module 'sky.core' from '/x'>"
        )

    _sky(monkeypatch, [cluster], _down)
    monkeypatch.setattr(sp, "list_gcp_instances", _boom("503 backendError"))
    monkeypatch.setattr(cli_mod, "_lab", lambda backend="local": lab)

    res = CliRunner().invoke(app, ["reconcile", "--apply", "--yes"])

    assert res.exit_code == 5, res.output
    report = json.loads(res.stdout)
    assert report["sky_orphans"] == [cluster]
    passes = {(o["pass"], o["outcome"]) for o in report["destroy_outcomes"]}
    assert ("sky_orphans", "undecodable_response") in passes  # the unread destroy is not lost
    assert ("gcp_orphans", "unknown") in passes  # and neither is the blind pass
    assert "503 backendError" in report["pass_errors"]["gcp_orphans"]
    assert report["gcp_pass"].startswith("failed")


def test_a_failed_vast_listing_does_not_stop_the_other_passes_under_apply(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One provider's outage must not disarm the emergency cleanup on the others."""
    cluster = sp.cluster_name_for(JOB, project=PROJECT)
    downs: list[str] = []
    _sky(monkeypatch, [cluster], downs.append)
    monkeypatch.setattr(sp, "list_vast_instances", _boom("vast api down"))

    report = lab.reconcile(apply=True)

    assert downs == [cluster] and report["sky_destroyed"] == [cluster]
    assert report["vast_pass"].startswith("failed")
    assert [o["pass"] for o in report["destroy_outcomes"]] == ["orphans"]


def test_a_failed_sky_status_is_reported_not_raised_under_apply(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sky(monkeypatch, [], lambda c: None)
    sys.modules["sky"].status = _boom("API server unreachable")  # type: ignore[attr-defined]

    report = lab.reconcile(apply=True)

    assert report["sky_pass"].startswith("failed")
    assert "API server unreachable" in report["pass_errors"]["sky_orphans"]
    assert report["ghosts"] == []  # unverifiable is never a ghost


def test_a_client_that_cannot_be_built_is_an_unconfirmed_destroy_not_a_crash(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Destroys already happened in earlier passes; a DO client failure must not discard them."""
    cluster = sp.cluster_name_for(JOB, project=PROJECT)
    downs: list[str] = []
    _sky(monkeypatch, [cluster], downs.append)
    monkeypatch.setattr(
        sp, "list_do_volumes", lambda *a, **k: [{"id": "v1", "name": f"{cluster}-ab12-head"}]
    )
    monkeypatch.setattr(sp, "_get_do_client", _boom("doctl token revoked"))

    report = lab.reconcile(apply=True)

    assert report["sky_destroyed"] == [cluster]
    assert [(o["pass"], o["resource"]) for o in report["destroy_outcomes"]] == [
        ("do_volume_orphans", f"{cluster}-ab12-head")
    ]


def test_a_dry_run_that_cannot_list_still_errors(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit 2, as documented: a dry run that reported clean while blind would be a lie."""
    _sky(monkeypatch, [], lambda c: None)
    monkeypatch.setattr(sp, "list_gcp_disks", _boom("500 backendError"))

    with pytest.raises(LabError, match="GCP"):
        lab.reconcile(apply=False)


def test_a_clean_apply_has_no_pass_errors(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    _sky(monkeypatch, [], lambda c: None)

    report = lab.reconcile(apply=True)

    assert report["pass_errors"] == {} and report["destroy_outcomes"] == []
