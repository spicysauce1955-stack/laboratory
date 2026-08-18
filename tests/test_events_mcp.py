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


def test_history_tool_returns_recent_calls(tmp_path: Path) -> None:
    _, server = _make(tmp_path)

    async def go() -> dict:
        async with Client(server) as client:
            await client.call_tool("list", {})
            result = await client.call_tool("history", {"limit": 10, "all_projects": True})
            return result.data

    data = asyncio.run(go())
    actions = [e["action"] for e in data["events"]]
    assert "list" in actions


def test_history_tool_excludes_its_own_in_flight_call(tmp_path: Path) -> None:
    """The middleware opens this very `history` call's ledger entry before the tool body runs,
    and only closes it after the tool returns. Without self-exclusion the tool would see its
    own still-open call as a phantom 'running-or-died' row, freshest-first, ahead of the real
    'list' call recorded just before it."""
    _, server = _make(tmp_path)

    async def go() -> dict:
        async with Client(server) as client:
            await client.call_tool("list", {})
            result = await client.call_tool("history", {"limit": 10, "all_projects": True})
            return result.data

    data = asyncio.run(go())
    actions = [e["action"] for e in data["events"]]
    assert actions == ["list"]
    assert data["events"][0]["status"] == "ok"


def test_history_stats_reflects_the_since_window(tmp_path: Path) -> None:
    """`stats=True` must report the cutoff it actually applied, not null (which reads as
    'unfiltered' even though a --since window was genuinely in effect)."""
    _, server = _make(tmp_path)

    async def go() -> dict:
        async with Client(server) as client:
            await client.call_tool("list", {})
            result = await client.call_tool(
                "history", {"since": "1h", "all_projects": True, "stats": True}
            )
            return result.data

    data = asyncio.run(go())
    assert data["since"] is not None
    assert data["total"] == 1


def test_history_tool_rejects_unparseable_since_with_a_clean_tool_error(tmp_path: Path) -> None:
    """FastMCP wraps *any* uncaught exception as a ToolError, so merely asserting
    `pytest.raises(ToolError)` would pass even with no explicit guard (its generic wrapper
    message is `Error calling tool 'history': ...`). Assert on our own message instead, which
    only appears if `since` is validated before `events.read` ever sees it."""
    _, server = _make(tmp_path)

    async def go() -> str:
        async with Client(server) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("history", {"since": "not-a-duration"})
            return str(exc_info.value)

    message = asyncio.run(go())
    assert "bad since" in message
    assert "not-a-duration" in message


def test_report_tool_returns_markdown(tmp_path: Path) -> None:
    _, server = _make(tmp_path)

    async def go() -> dict:
        async with Client(server) as client:
            result = await client.call_tool("report", {"since": "7d", "all_projects": True})
            return result.data

    data = asyncio.run(go())
    assert data["markdown"].startswith("# Lab event report")


def test_report_tool_reflects_the_since_window_in_its_header(tmp_path: Path) -> None:
    """`report`'s markdown header must say the window it actually filtered by, not fall back
    to 'all recorded history' while genuinely restricting to --since."""
    _, server = _make(tmp_path)

    async def go() -> dict:
        async with Client(server) as client:
            result = await client.call_tool("report", {"since": "7d", "all_projects": True})
            return result.data

    data = asyncio.run(go())
    assert data["markdown"].startswith("# Lab event report — since ")
    assert "all recorded history" not in data["markdown"]


def test_report_tool_rejects_unparseable_since_with_a_clean_tool_error(tmp_path: Path) -> None:
    """See the matching `history` test: assert on our own message so this fails without an
    explicit guard, rather than passing on FastMCP's generic wrapper message."""
    _, server = _make(tmp_path)

    async def go() -> str:
        async with Client(server) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("report", {"since": "not-a-duration"})
            return str(exc_info.value)

    message = asyncio.run(go())
    assert "bad since" in message
    assert "not-a-duration" in message
