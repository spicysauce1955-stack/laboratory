"""An unknown backend name fails before anything runs — on every surface.

Stress run 2026-11-19: `--backend skypilto`, `--backend Local` and MCP `backend="nonsense"` all
exited 0 and ran the job *on this machine*: `build_backend` mapped any name it did not know to
the local backend. A typo meant for a remote GPU box then ran (or tried to run) a GPU workload on
the laptop with no remote timeout or teardown in sight, and reported success at it. Cloud names
were already validated (`validate_cloud`); backend names now are too.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from core_fixtures import make_repo
from fastmcp import Client
from fastmcp.exceptions import ToolError
from typer.testing import CliRunner

from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import (
    SUPPORTED_BACKENDS,
    Lab,
    LabError,
    build_backend,
    default_lab,
    resolve_backend_profile,
    validate_backend,
)
from lab.mcp_server import build_server
from lab.models import ResourceRequest
from lab.scheduler.queue import LocalQueueStore
from lab.scheduler.tick import Scheduler

_BAD = ["skypilto", "Local", "nonsense", "", " local", "vast"]


@pytest.mark.parametrize("name", _BAD)
def test_validate_backend_refuses_an_unknown_name(name: str) -> None:
    with pytest.raises(LabError, match="unknown backend"):
        validate_backend(name)


@pytest.mark.parametrize("name", ["local", "skypilot", "cpu"])
def test_the_supported_names_pass(name: str) -> None:
    assert name in SUPPORTED_BACKENDS
    assert validate_backend(name) == name


@pytest.mark.parametrize("name", _BAD)
def test_resolve_backend_profile_refuses_before_resolving(name: str) -> None:
    with pytest.raises(LabError, match="unknown backend"):
        resolve_backend_profile(name, ResourceRequest())


@pytest.mark.parametrize("name", _BAD)
def test_build_backend_no_longer_falls_back_to_local(tmp_path: Path, name: str) -> None:
    with pytest.raises(LabError, match="unknown backend"):
        build_backend(name, home=tmp_path, repo=tmp_path)


def test_default_lab_refuses_an_unknown_backend(tmp_path: Path, monkeypatch) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    with pytest.raises(LabError, match="unknown backend"):
        default_lab(backend="skypilto")


@pytest.mark.parametrize("command", ["submit", "sweep"])
@pytest.mark.parametrize("name", ["skypilto", "Local"])
def test_cli_refuses_and_runs_nothing(tmp_path: Path, monkeypatch, command: str, name: str) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    args = [command, "-c", "true", "--backend", name]
    if command == "sweep":
        args += ["--seeds", "0-1"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0, result.output
    assert "unknown backend" in result.output
    runs = repo / "runs"
    assert not runs.exists() or not any(runs.iterdir()), "a job was created anyway"


@pytest.mark.parametrize("tool", ["submit", "sweep"])
def test_mcp_refuses_and_runs_nothing(tmp_path: Path, monkeypatch, tool: str) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    home = repo / "runs"
    server = build_server(Lab(backend=LocalBackend(home=home, repo=repo), repo=repo, home=home))
    args: dict[str, object] = {"command": "true", "backend": "nonsense"}
    if tool == "sweep":
        args["seeds"] = "0-1"

    async def go() -> None:
        async with Client(server) as c:
            await c.call_tool(tool, args)

    with pytest.raises(ToolError, match="unknown backend"):
        asyncio.run(go())
    assert not home.exists() or not any(home.iterdir()), "a job was created anyway"


def test_the_scheduler_refuses_an_unknown_launch_backend(tmp_path: Path) -> None:
    """`lab scheduler tick --backend skypilto` would otherwise launch every due registration on
    the scheduler host itself."""
    with pytest.raises(LabError, match="unknown backend"):
        Scheduler(LocalQueueStore(tmp_path / "q"), home=tmp_path / "runs", backend="skypilto")
