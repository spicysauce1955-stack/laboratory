"""`lab list` / `mcp__lab__list` return a bounded page of jobs, and say how many they left out.

At 10k jobs the MCP `list` tool returned 1.24 MB (CLI 1.67 MB) with no way to ask for less. Both
surfaces now return the newest `limit` (default 100) jobs, keep the rows in the order they always
had (oldest to newest), and add `jobs_total`/`jobs_omitted`/`limit`. `limit=0` is the explicit
opt-out. The `spend` block still covers every job: it is the project's money, not a page of it.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from helpers import PYTHON, make_manifest
from typer.testing import CliRunner

from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab
from lab.list_view import DEFAULT_LIST_LIMIT
from lab.mcp_server import build_server
from lab.store import JobStore

runner = CliRunner()
T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int) -> tuple[Lab, list[str]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_REPO_DIR", str(repo))
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    home = repo / "runs"
    store = JobStore(home)
    ids = []
    for i in range(n):
        # ids sort in the reverse of creation order, so "newest" must come from created_at
        job_id = f"job-{n - i:05d}"
        m = make_manifest(job_id, f"{PYTHON} -c 'print(1)'")
        store.write_manifest(m.model_copy(update={"created_at": T0 + timedelta(minutes=i)}))
        ids.append(job_id)
    return Lab(backend=LocalBackend(home=home, repo=repo), repo=repo, home=home), ids


def _cli(*args: str) -> dict[str, Any]:
    result = runner.invoke(app, ["list", *args])
    assert result.exit_code == 0, result.output
    loaded: dict[str, Any] = json.loads(result.stdout)
    return loaded


def _mcp(lab: Lab, args: dict[str, Any]) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        async with Client(build_server(lab)) as c:
            res = await c.call_tool("list", args)
            assert res.structured_content is not None
            return res.structured_content

    return asyncio.run(go())


def test_limit_keeps_the_newest_jobs_in_the_usual_order(tmp_path, monkeypatch):
    _, ids = _project(tmp_path, monkeypatch, 7)

    out = _cli("--limit", "3")

    newest = set(ids[-3:])  # created last
    assert {r["job_id"] for r in out["jobs"]} == newest
    assert [r["job_id"] for r in out["jobs"]] == sorted(newest)  # the listing's own order
    assert (out["jobs_total"], out["jobs_omitted"], out["limit"]) == (7, 4, 3)


def test_the_default_is_bounded(tmp_path, monkeypatch):
    _, _ = _project(tmp_path, monkeypatch, DEFAULT_LIST_LIMIT + 5)

    out = _cli()

    assert len(out["jobs"]) == DEFAULT_LIST_LIMIT
    assert out["jobs_omitted"] == 5 and out["limit"] == DEFAULT_LIST_LIMIT


def test_limit_zero_returns_every_job(tmp_path, monkeypatch):
    _, ids = _project(tmp_path, monkeypatch, 12)

    out = _cli("--limit", "0")

    assert len(out["jobs"]) == 12 and out["jobs_omitted"] == 0


def test_spend_still_covers_every_job(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, 9)

    assert _cli("--limit", "2")["spend"] == _cli("--limit", "0")["spend"]


def test_cli_and_mcp_return_the_same_page(tmp_path, monkeypatch):
    lab, _ = _project(tmp_path, monkeypatch, 6)

    for cli_args, mcp_args in (((), {}), (("--limit", "2"), {"limit": 2})):
        assert _cli(*cli_args) == _mcp(lab, mcp_args)


def test_a_negative_limit_is_refused(tmp_path, monkeypatch):
    lab, _ = _project(tmp_path, monkeypatch, 1)

    assert runner.invoke(app, ["list", "--limit", "-1"]).exit_code == 2
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="limit"):
        _mcp(lab, {"limit": -1})
