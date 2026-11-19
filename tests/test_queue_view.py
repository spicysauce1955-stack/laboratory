"""`lab queue list` and the MCP `queue_list` tool are one view, not two copies of it.

They were two hand-maintained copies, and the MCP one drifted: it never gained `scheduler_skew`
(the diagnostic that exists because an older scheduler host silently dropped `--price-cap` from
deferred jobs) nor the per-entry `cancel_requested`, so an agent watching the queue could see
neither. Both shells now call `lab.queue_view.queue_list_view`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, LocalQueueStore]:
    repo = _make_repo(tmp_path)
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("LAB_REPO_DIR", str(repo))
    return repo, LocalQueueStore(tmp_path / "queue")


def _register(repo: Path) -> str:
    res = runner.invoke(
        app, ["register", "--command", "python exp.py", "--timeout", "1h", "--expires", "+3d"]
    )
    assert res.exit_code == 0, res.output
    reg_id: str = json.loads(res.stdout)["reg_id"]
    return reg_id


def _mcp_queue_list(repo: Path, tmp_path: Path) -> dict:
    lab = Lab(backend=LocalBackend(home=tmp_path / "runs", repo=repo), repo=repo,
              home=tmp_path / "runs")

    async def go() -> dict:
        async with Client(build_server(lab)) as c:
            data: dict = (await c.call_tool("queue_list", {})).structured_content
            return data

    return asyncio.run(go())


def test_the_mcp_tool_reports_skew_and_cancel_requests(tmp_path, monkeypatch):
    repo, q = _setup(tmp_path, monkeypatch)
    reg_id = _register(repo)
    q.request_cancel(reg_id)
    q.write_heartbeat(
        {"at": "2026-09-08T00:00:00+00:00", "host": "old", "tick_count": 1, "lab_version": "0.4.0"}
    )

    data = _mcp_queue_list(repo, tmp_path)

    assert data["scheduler_skew"]["verdict"] == "host_older"
    assert data["entries"][0]["cancel_requested"] is True


def test_both_shells_return_the_same_payload(tmp_path, monkeypatch):
    repo, q = _setup(tmp_path, monkeypatch)
    held = _register(repo)
    cancelled = _register(repo)
    _register(repo)
    q.hold(held)
    q.request_cancel(cancelled)
    # No heartbeat: `heartbeat_age_s` is then null on both sides instead of a clock reading.

    cli = json.loads(runner.invoke(app, ["queue", "list"]).stdout)
    mcp = _mcp_queue_list(repo, tmp_path)

    assert cli == mcp
