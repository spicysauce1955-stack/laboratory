"""`timeout`/`provision_timeout` are parsed when a job is *asked for*, not when it is launched.

`lab submit` never parsed `--timeout` up front: `3h30` (documented as refused) was accepted into
the spec, and a negative timeout paid for a provision that died the moment it came up. The check
lives on every create path — `resolve_backend_profile` (CLI + MCP submit/sweep), `Lab.submit`
(library callers, the scheduler) and registration — and deliberately NOT as a model validator:
legacy manifests carrying such a value must still load (the FR-B1 "enforce on create" rule).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lab.core import LabError, resolve_backend_profile
from lab.models import JobManifest, ResourceRequest

BAD = ["3h30", "-5m", "inf", "nan", "0"]


@pytest.mark.parametrize("field", ["timeout", "provision_timeout"])
@pytest.mark.parametrize("value", BAD)
def test_resolve_backend_profile_refuses_a_bad_duration(field: str, value: str) -> None:
    with pytest.raises(LabError, match=field):
        resolve_backend_profile("skypilot", ResourceRequest(**{field: value}))


@pytest.mark.parametrize("backend", ["local", "skypilot", "cpu"])
def test_good_durations_pass(backend: str) -> None:
    resolve_backend_profile(backend, ResourceRequest(timeout="3h30m", provision_timeout="10m"))


def test_cli_submit_refuses_before_creating_anything(tmp_path: Path, monkeypatch) -> None:
    from lab.cli import app

    monkeypatch.setenv("LAB_REPO_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["submit", "-c", "python x.py", "--backend", "local", "--timeout", "3h30"]
    )
    assert result.exit_code != 0
    assert "timeout" in result.output
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_lab_submit_refuses_a_bad_duration(tmp_path: Path) -> None:
    from lab.backends.local import LocalBackend
    from lab.core import Lab
    from lab.models import JobSpec

    lab = Lab(backend=LocalBackend(home=tmp_path / "lab", repo=tmp_path), repo=tmp_path,
              home=tmp_path / "lab")
    spec = JobSpec(command="python x.py", resources=ResourceRequest(timeout="-1h"))
    with pytest.raises(LabError, match="timeout"):
        lab.submit(spec, preflight=False)


def test_registration_refuses_a_bad_duration() -> None:
    from lab.scheduler.models import Triggers
    from lab.scheduler.register import _check_cloud_triggers

    with pytest.raises(LabError, match="provision_timeout"):
        _check_cloud_triggers(ResourceRequest(provision_timeout="inf"), Triggers())


def test_a_legacy_manifest_with_a_bad_timeout_still_loads() -> None:
    """Refusing on create must not make an already-written job unreadable: a manifest's
    `resources` is a ResourceRequest, so this is a check, not a model validator."""
    assert JobManifest.model_fields["resources"].annotation is ResourceRequest
    loaded = ResourceRequest.model_validate_json(json.dumps({"timeout": "3h30"}))
    assert loaded.timeout == "3h30"
