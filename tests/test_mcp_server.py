import asyncio
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from helpers import PYTHON, wait_terminal

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.mcp_server import build_server
from lab.models import JobState


def _make(tmp_path: Path):
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    return lab, build_server(lab)


def test_tools_registered(tmp_path: Path):
    _, server = _make(tmp_path)

    async def go():
        async with Client(server) as c:
            return sorted(t.name for t in await c.list_tools())

    assert asyncio.run(go()) == [
        "cancel",
        "fetch_artifacts",
        "list",
        "logs",
        "metrics",
        "status",
        "submit",
    ]


def test_submit_status_logs_fetch(tmp_path: Path):
    lab, server = _make(tmp_path)

    async def do_submit() -> str:
        async with Client(server) as c:
            r = await c.call_tool(
                "submit",
                {"command": f"{PYTHON} experiments/example_capacity.py", "seed": 9},
            )
            return r.data["job_id"]

    job_id = asyncio.run(do_submit())
    assert wait_terminal(lab.backend, job_id) == JobState.succeeded

    async def query():
        async with Client(server) as c:
            st = (await c.call_tool("status", {"job_id": job_id})).data
            lg = (await c.call_tool("logs", {"job_id": job_id})).data
            mt = (await c.call_tool("metrics", {"job_id": job_id})).data
            ft = (await c.call_tool("fetch_artifacts", {"job_id": job_id})).data
            ls = (await c.call_tool("list", {})).data
            return st, lg, mt, ft, ls

    st, lg, mt, ft, ls = asyncio.run(query())
    assert st["state"] == "succeeded" and st["exit_code"] == 0
    assert isinstance(lg["lines"], list)
    assert "demo_metric" in mt["series"]
    assert any(a["name"] == "result.json" for a in ft["artifacts"])
    assert ls["jobs"][0]["job_id"] == job_id


def test_unknown_job_is_fail_loud(tmp_path: Path):
    _, server = _make(tmp_path)

    async def go():
        async with Client(server) as c:
            await c.call_tool("status", {"job_id": "does-not-exist"})

    with pytest.raises(ToolError):
        asyncio.run(go())
