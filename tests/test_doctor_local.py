"""`doctor(cloud="local")` / `lab doctor --cloud local` (stress run L5).

`local` is the default backend, yet the MCP doctor tool rejected it as an unknown cloud. There is
nothing to check for it — no credentials, quota or billing — so both surfaces now return a
trivial ok report that says so, instead of an error an agent has to work around.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from typer.testing import CliRunner

from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab
from lab.mcp_server import build_server

runner = CliRunner()


def _lab(home: Path) -> Lab:
    return Lab(backend=LocalBackend(home=home, repo=home), repo=home, home=home)


def _call(lab: Lab, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        async with Client(build_server(lab)) as c:
            res = await c.call_tool(tool, args)
            assert res.structured_content is not None
            return res.structured_content

    return asyncio.run(go())


# --- doctor(cloud="local") -----------------------------------------------------------------


def test_doctor_local_is_a_trivial_ok_report(tmp_path: Path) -> None:
    out = _call(_lab(tmp_path), "doctor", {"cloud": "local"})

    assert out["cloud"] == "local" and out["ok"] is True and out["blocking"] == []
    assert [c["status"] for c in out["checks"]] == ["ok"]
    assert "no credentials" in out["checks"][0]["detail"]


def test_cli_doctor_local_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_REPO_DIR", str(tmp_path))

    result = runner.invoke(app, ["doctor", "--cloud", "local", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == _call(_lab(tmp_path / "runs"), "doctor", {"cloud": "local"})


def test_an_unknown_cloud_is_still_refused(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="unknown cloud"):
        _call(_lab(tmp_path), "doctor", {"cloud": "azure"})
