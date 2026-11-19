"""The MCP deferred-path tools store what `lab register`/`lab register-sweep` store.

`tests/test_cli_mcp_parity.py` proves the parameters *exist* on both surfaces; this proves they
mean the same thing, by registering the same job through each shell and comparing the persisted
`ResourceRequest` — the record the unattended scheduler will launch from.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from typer.testing import CliRunner

from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab
from lab.mcp_server import build_server
from lab.scheduler.queue import LocalQueueStore
from test_scheduler_bundle import _make_repo

runner = CliRunner()

_CLI_FLAGS = [
    "--accelerators", "RTX4090:1", "--gpus", "1", "--timeout", "1h",
    "--accelerator-pool", "RTX4090:1,RTX3090:1", "--max-launch-attempts", "2",
    "--region", "us-east", "--zone", "us-east-1a",
    "--price-cap", "0.5", "--price-cap-strict", "--spot", "--no-fallback",
]
_MCP_ARGS: dict[str, Any] = {
    "accelerators": "RTX4090:1", "gpus": 1, "timeout": "1h",
    "accelerator_pool": "RTX4090:1,RTX3090:1", "max_launch_attempts": 2,
    "region": "us-east", "zone": "us-east-1a",
    "price_cap": 0.5, "price_cap_strict": True, "use_spot": True, "spot_fallback": False,
}


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, LocalQueueStore]:
    repo = _make_repo(tmp_path)
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("LAB_REPO_DIR", str(repo))
    return repo, LocalQueueStore(tmp_path / "queue")


def _call(repo: Path, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    lab = Lab(backend=LocalBackend(home=repo / "runs", repo=repo), repo=repo, home=repo / "runs")

    async def go() -> dict[str, Any]:
        async with Client(build_server(lab)) as c:
            data: dict[str, Any] = (await c.call_tool(tool, args)).structured_content
            return data

    return asyncio.run(go())


def _resources(q: LocalQueueStore, reg_id: str) -> dict[str, Any]:
    return q.get_entry(reg_id).spec.resources.model_dump()


def test_register_persists_the_same_resources_as_the_cli(env) -> None:
    repo, q = env
    cli = runner.invoke(
        app, ["register", "--command", "python exp.py", "--expires", "+1d", *_CLI_FLAGS]
    )
    assert cli.exit_code == 0, cli.output
    cli_reg = json.loads(cli.stdout)["reg_id"]

    mcp_reg = _call(
        repo, "register", {"command": "python exp.py", "expires": "+1d", **_MCP_ARGS}
    )["reg_id"]

    mcp_res = _resources(q, mcp_reg)
    assert mcp_res == _resources(q, cli_reg)
    assert mcp_res["max_hourly_usd"] == 0.5 and mcp_res["price_cap_strict"] is True


def test_register_hold_registers_it_held(env) -> None:
    repo, q = env
    out = _call(
        repo, "register",
        {"command": "python exp.py", "expires": "+1d", "timeout": "1h", "hold": True},
    )
    assert out["state"] == "held"
    assert q.held(out["reg_id"])


def test_register_sweep_persists_the_same_points_as_the_cli(env) -> None:
    repo, q = env
    cli = runner.invoke(
        app,
        ["register-sweep", "--command", "python exp.py", "--grid", "K=1,2", "--expires", "+1d",
         "--with", "rich", *_CLI_FLAGS],
    )
    assert cli.exit_code == 0, cli.output
    cli_regs = json.loads(cli.stdout)["reg_ids"]

    mcp_regs = _call(
        repo, "register_sweep",
        {"command": "python exp.py", "grid": {"K": ["1", "2"]}, "expires": "+1d",
         "with_pkg": ["rich"], **_MCP_ARGS},
    )["reg_ids"]

    for cli_id, mcp_id in zip(cli_regs, mcp_regs, strict=True):
        assert _resources(q, mcp_id) == _resources(q, cli_id)
        assert q.get_entry(mcp_id).spec.command == q.get_entry(cli_id).spec.command
    assert "rich" in q.get_entry(mcp_regs[0]).spec.command
