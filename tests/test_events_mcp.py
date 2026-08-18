"""Every MCP tool call must land in the ledger with surface "mcp", and the middleware must
never change what a tool returns or raises."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from helpers import PYTHON, wait_terminal

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.events import store
from lab.manifest import repo_root
from lab.mcp_server import build_server
from lab.models import JobState


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


def _make(tmp_path: Path):
    repo = repo_root(Path.cwd())
    home = tmp_path / "home"
    lab = Lab(backend=LocalBackend(home=home, repo=repo), repo=repo, home=home)
    return lab, build_server(lab)


def _records() -> list[dict]:
    return list(store.iter_records(store.day_files()))


def test_a_tool_call_records_an_open_close_pair(tmp_path: Path) -> None:
    _, server = _make(tmp_path)

    async def go() -> None:
        async with Client(server) as client:
            await client.call_tool("list", {})

    asyncio.run(go())
    opened, closed = _records()
    assert opened["surface"] == "mcp" and opened["action"] == "list"
    assert closed["outcome"] == "ok"


def test_a_tool_error_records_a_crash_with_the_offending_job_id(tmp_path: Path) -> None:
    _, server = _make(tmp_path)

    async def go() -> None:
        async with Client(server) as client:
            with pytest.raises(ToolError):
                await client.call_tool("status", {"job_id": "j-nope"})

    asyncio.run(go())
    opened, closed = _records()
    assert opened["surface"] == "mcp" and opened["action"] == "status"
    assert opened["params"] == {"job_id": "j-nope"}
    # A ToolError raised by the tool propagates through record() unrelabeled: the middleware
    # does not choose "error" for a raised exception, it reports what actually happened.
    assert closed["outcome"] == "crash"
    assert closed["error"]["type"] == "ToolError"
    assert "j-nope" in closed["error"]["message"]


def test_a_successful_call_records_real_refs_and_a_result_digest(tmp_path: Path) -> None:
    """Drives a real job through submit -> status so refs_from/digest_of see an actual
    payload (job_id, state), not None — the silent failure Trap 1 warns about would show up
    here as empty dicts rather than as a crash."""
    lab, server = _make(tmp_path)

    async def do_submit() -> str:
        async with Client(server) as client:
            r = await client.call_tool(
                "submit", {"command": f"{PYTHON} experiments/example_capacity.py", "seed": 3}
            )
            job_id = r.data["job_id"]
            assert isinstance(job_id, str)
            return job_id

    job_id = asyncio.run(do_submit())
    assert wait_terminal(lab.backend, job_id) == JobState.succeeded

    async def do_status() -> None:
        async with Client(server) as client:
            await client.call_tool("status", {"job_id": job_id})

    asyncio.run(do_status())
    records = _records()
    closed = records[-1]  # status's close record (submit opened/closed a pair before it)
    assert closed["outcome"] == "ok"
    assert closed["refs"] == {"job_id": job_id}
    assert closed["result"]["state"] == "succeeded"
