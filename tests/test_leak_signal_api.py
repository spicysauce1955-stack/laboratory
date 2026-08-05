"""The leak-signal chain across the API surfaces (review roadmap item 1): teardown state must
be visible to agents — MCP status carries teardown_status, mirrored manifests are readable from
both shells, and reconcile/wait exist as MCP tools."""

import asyncio
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from helpers import PYTHON, make_manifest, wait_terminal

from lab.backends.local import LocalBackend
from lab.core import Lab, job_status_view
from lab.manifest import repo_root
from lab.mcp_server import build_server
from lab.models import JobState
from lab.scheduler.queue import LocalQueueStore


def _make(tmp_path: Path):
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    return lab, build_server(lab)


def _mirrored_queue(tmp_path: Path, monkeypatch, manifest):
    qdir = tmp_path / "queue"
    q = LocalQueueStore(qdir)
    q.mirror_manifest(manifest)
    monkeypatch.setenv("LAB_QUEUE_DIR", str(qdir))
    return q


# ---------------------------------------------------------------------------
# core: job_status_view — one shape, mirrored fallback
# ---------------------------------------------------------------------------


def test_job_status_view_includes_teardown_and_provenance(tmp_path):
    lab, _ = _make(tmp_path)
    m = make_manifest("jv1", "python x.py").model_copy(
        update={"status": JobState.failed, "teardown_status": "failed", "exit_code": 1}
    )
    lab.store.create(m)
    view = job_status_view(tmp_path, lab.repo, "jv1")
    assert view["teardown_status"] == "failed"
    assert view["state"] == "failed"
    assert view["code"]["git_commit"] == "0" * 40
    assert view["mirrored"] is False


def test_job_status_view_falls_back_to_mirror(tmp_path, monkeypatch):
    lab, _ = _make(tmp_path)
    m = make_manifest("jmir", "python x.py").model_copy(
        update={"status": JobState.succeeded, "teardown_status": "succeeded"}
    )
    _mirrored_queue(tmp_path, monkeypatch, m)
    view = job_status_view(tmp_path, lab.repo, "jmir")  # not in local runs/
    assert view["state"] == "succeeded"
    assert view["teardown_status"] == "succeeded"
    assert view["mirrored"] is True


def test_job_status_view_unknown_raises(tmp_path, monkeypatch):
    lab, _ = _make(tmp_path)
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "empty-queue"))
    with pytest.raises(FileNotFoundError):
        job_status_view(tmp_path, lab.repo, "nope")


# ---------------------------------------------------------------------------
# core: Lab.wait_summary — the FR-C2 verdict as data
# ---------------------------------------------------------------------------


def test_wait_summary_reports_leaks(tmp_path):
    lab, _ = _make(tmp_path)
    from lab.models import BackendInfo

    leaked = make_manifest("jl1", "python x.py").model_copy(
        update={
            "status": JobState.failed,
            "teardown_status": "failed",
            "backend": BackendInfo(provisioner="skypilot"),
        }
    )
    clean = make_manifest("jl2", "python x.py").model_copy(
        update={"status": JobState.succeeded, "teardown_status": "succeeded",
                "backend": BackendInfo(provisioner="skypilot")}
    )
    lab.store.create(leaked)
    lab.store.create(clean)
    summary = lab.wait_summary(["jl1", "jl2"], interval=0.1, timeout=5)
    assert summary["all_terminal"] is True
    assert summary["teardown_leaks"] == ["jl1"]
    assert summary["teardown_unconfirmed"] == []
    assert {j["job_id"]: j["teardown_status"] for j in summary["jobs"]} == {
        "jl1": "failed", "jl2": "succeeded",
    }


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_mcp_status_returns_teardown_status(tmp_path):
    lab, server = _make(tmp_path)
    m = make_manifest("jm1", "python x.py").model_copy(
        update={"status": JobState.failed, "teardown_status": "failed"}
    )
    lab.store.create(m)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool("status", {"job_id": "jm1"})).data

    out = asyncio.run(go())
    assert out["teardown_status"] == "failed"
    assert out["state"] == "failed"


def test_mcp_status_reads_mirrored_manifest(tmp_path, monkeypatch):
    """A deferred/scheduler-launched job must be observable via MCP, not write-only."""
    lab, server = _make(tmp_path)
    m = make_manifest("jm2", "python x.py").model_copy(
        update={"status": JobState.running}
    )
    _mirrored_queue(tmp_path, monkeypatch, m)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool("status", {"job_id": "jm2"})).data

    out = asyncio.run(go())
    assert out["state"] == "running" and out["mirrored"] is True


def test_mcp_reconcile_tool_is_dry_run(tmp_path, monkeypatch):
    lab, server = _make(tmp_path)
    seen: list = []

    def _fake_reconcile(self, *, apply=False):
        seen.append(apply)
        return {"orphans": [], "sky_orphans": [], "unsupervised": [], "applied": apply}

    monkeypatch.setattr(Lab, "reconcile", _fake_reconcile)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool("reconcile", {})).data

    out = asyncio.run(go())
    assert seen == [False]  # read-only: never applies
    assert out["applied"] is False


def test_mcp_wait_tool_returns_leak_verdict(tmp_path):
    lab, server = _make(tmp_path)

    async def go():
        async with Client(server) as c:
            r = await c.call_tool(
                "submit", {"command": f"{PYTHON} experiments/example_capacity.py"}
            )
            job_id = r.data["job_id"]
            wait_terminal(lab.backend, job_id)
            return (await c.call_tool("wait", {"job_ids": [job_id], "timeout": 30})).data

    out = asyncio.run(go())
    assert out["all_terminal"] is True
    assert out["teardown_leaks"] == []


def test_mcp_wait_requires_ids(tmp_path):
    _, server = _make(tmp_path)

    async def go():
        async with Client(server) as c:
            await c.call_tool("wait", {})

    with pytest.raises(ToolError, match="job id"):
        asyncio.run(go())
