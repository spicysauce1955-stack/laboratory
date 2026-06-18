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
        "confirm",
        "fetch_artifacts",
        "list",
        "logs",
        "metrics",
        "queue_cancel",
        "queue_list",
        "queue_pause",
        "queue_show",
        "register",
        "register_sweep",
        "status",
        "submit",
        "sweep",
        "sweep_aggregate",
        "sweep_retry",
        "sweep_status",
    ]


def test_sweep_tool(tmp_path: Path):
    lab, server = _make(tmp_path)

    async def go():
        async with Client(server) as c:
            r = await c.call_tool(
                "sweep",
                {"command": f"{PYTHON} experiments/example_capacity.py", "grid": {"K": [1, 2]}},
            )
            return r.data

    data = asyncio.run(go())
    assert data["sweep_id"].startswith("sweep-")
    assert len(data["job_ids"]) == 2
    for jid in data["job_ids"]:
        assert wait_terminal(lab.backend, jid) == JobState.succeeded


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


def test_submit_accepts_backend_param(tmp_path: Path):
    lab, server = _make(tmp_path)

    async def do_submit() -> str:
        async with Client(server) as c:
            r = await c.call_tool(
                "submit",
                {"command": f"{PYTHON} experiments/example_capacity.py", "backend": "local", "seed": 2},
            )
            return r.data["job_id"]

    job_id = asyncio.run(do_submit())
    assert wait_terminal(lab.backend, job_id) == JobState.succeeded
    assert lab.manifest(job_id).backend.provisioner == "local"


def test_submit_disk_size_recorded(tmp_path: Path):
    """The MCP submit tool accepts disk_size and records it on the spec's resources."""
    lab, server = _make(tmp_path)

    async def do_submit() -> str:
        async with Client(server) as c:
            r = await c.call_tool(
                "submit",
                {
                    "command": f"{PYTHON} experiments/example_capacity.py",
                    "backend": "local",
                    "disk_size": 120,
                },
            )
            return r.data["job_id"]

    job_id = asyncio.run(do_submit())
    assert lab.manifest(job_id).resources.disk_size == 120


def test_unknown_job_is_fail_loud(tmp_path: Path):
    _, server = _make(tmp_path)

    async def go():
        async with Client(server) as c:
            await c.call_tool("status", {"job_id": "does-not-exist"})

    with pytest.raises(ToolError):
        asyncio.run(go())


def _make_with_repo(tmp_path: Path):
    from test_scheduler_bundle import _make_repo

    repo = _make_repo(tmp_path)
    lab = Lab(backend=LocalBackend(home=tmp_path / "runs", repo=repo), repo=repo, home=tmp_path / "runs")
    return lab, build_server(lab)


def test_register_and_queue_tools(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    _, server = _make_with_repo(tmp_path)

    async def go():
        async with Client(server) as c:
            out = (await c.call_tool(
                "register",
                {"command": "python exp.py", "expires": "+1d",
                 "max_hourly": 0.25, "timeout": "1h"},
            )).data
            listed = (await c.call_tool("queue_list", {})).data
            shown = (await c.call_tool("queue_show", {"reg_id": out["reg_id"]})).data
            cancelled = (await c.call_tool("queue_cancel", {"reg_id": out["reg_id"]})).data
            paused = (await c.call_tool("queue_pause", {"paused": True})).data
            return out, listed, shown, cancelled, paused

    out, listed, shown, cancelled, paused = asyncio.run(go())
    assert out["reg_id"].startswith("reg-")
    assert out["worst_case_cost_usd"] == 0.25
    assert listed["entries"][0]["reg_id"] == out["reg_id"]
    assert shown["state"] == "pending"
    assert cancelled["cancel_requested"] is True
    assert paused["paused"] is True


def test_register_sweep_tool(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    _, server = _make_with_repo(tmp_path)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool(
                "register_sweep",
                {"command": "python exp.py", "grid": {"K": ["1", "2"]},
                 "expires": "+1d", "sweep_max_cost": 5.0},
            )).data

    out = asyncio.run(go())
    assert out["count"] == 2
    assert out["sweep_id"].startswith("sweep-")
    assert len(out["reg_ids"]) == 2
    assert all(r.startswith("reg-") for r in out["reg_ids"])


def test_register_unknown_queue_ops_fail_loud(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    _, server = _make_with_repo(tmp_path)

    async def go():
        async with Client(server) as c:
            with pytest.raises(ToolError):
                await c.call_tool("queue_show", {"reg_id": "reg-nope"})

    asyncio.run(go())


def test_sweep_sharded_returns_cells(tmp_path: Path):
    """sweep tool with seeds + shard_size returns structured cells view (not flat job_ids)."""
    _, server = _make(tmp_path)

    async def go():
        async with Client(server) as c:
            r = await c.call_tool(
                "sweep",
                {
                    "command": "true",
                    "grid": {"N": [1000]},
                    "seeds": "0-3",
                    "shard_size": 2,
                },
            )
            return r.data

    data = asyncio.run(go())
    assert "cells" in data, f"expected 'cells' key, got: {list(data.keys())}"
    assert len(data["cells"]) == 1
    cell = data["cells"][0]
    assert cell["seeds_expected"] == 4
    assert len(cell["shard_job_ids"]) == 2
    assert cell["status"] == "pending"


def test_sweep_aggregate_tool(tmp_path: Path):
    """sweep_aggregate tool row-concatenates succeeded shards and returns updated cell view."""
    lab, server = _make(tmp_path)

    # arrange: sweep with shards
    async def do_sweep():
        async with Client(server) as c:
            r = await c.call_tool(
                "sweep",
                {
                    "command": "true",
                    "grid": {"N": [1000]},
                    "seeds": "0-3",
                    "shard_size": 2,
                },
            )
            return r.data

    sweep_data = asyncio.run(do_sweep())
    sweep_id = sweep_data["sweep_id"]
    cell = sweep_data["cells"][0]

    # write shard results so aggregate can find them
    def _write_shard(job_id: str, seeds: list[int]) -> None:
        out = lab.store.output_dir(job_id)
        out.mkdir(parents=True, exist_ok=True)
        lines = ["seed,acc"] + [f"{s},0.{s}" for s in seeds]
        (out / "results.csv").write_text("\n".join(lines) + "\n")
        lab.store.update_manifest(job_id, status=JobState.succeeded)

    _write_shard(cell["shard_job_ids"][0], [0, 1])
    _write_shard(cell["shard_job_ids"][1], [2, 3])

    async def do_aggregate():
        async with Client(server) as c:
            r = await c.call_tool("sweep_aggregate", {"sweep_id": sweep_id})
            return r.data

    agg_data = asyncio.run(do_aggregate())
    assert "cells" in agg_data
    assert len(agg_data["cells"]) == 1
    result_cell = agg_data["cells"][0]
    assert result_cell["status"] == "complete"
    assert result_cell["seeds_present"] == 4
