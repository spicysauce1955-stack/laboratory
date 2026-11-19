"""`lab metrics` and `mcp__lab__metrics` return the same bounded payload for the same arguments.

The stress run measured 27.7 MB over MCP and 199 MB on the CLI for one unbounded read. Both
surfaces now default to the newest `max_points` (200) points per series and report what they cut
under `truncated`; `max_points=0` / `--max-points 0` is the explicit opt-out and `downsample` /
`--downsample` spreads the cap over the whole run. The parity test pins the argument names and
defaults; this pins that the two shells return the same thing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from helpers import PYTHON, make_manifest
from typer.testing import CliRunner

from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab
from lab.mcp_server import build_server
from lab.metrics import DEFAULT_MAX_POINTS
from lab.models import JobState
from lab.store import JobStore

runner = CliRunner()
N = 450


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Lab, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_REPO_DIR", str(repo))
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    home = repo / "runs"
    store = JobStore(home)
    m = make_manifest("job-curve", f"{PYTHON} -c 'print(1)'")
    store.write_manifest(m.model_copy(update={"status": JobState.succeeded}))
    out = store.output_dir("job-curve")
    out.mkdir(parents=True)
    with (out / "metrics.jsonl").open("w") as f:
        for i in range(N):
            for name in ("loss", "acc"):
                f.write(json.dumps({"name": name, "value": 1 / (i + 1), "step": i, "wall_time": 1.0}))
                f.write("\n")
    lab = Lab(backend=LocalBackend(home=home, repo=repo), repo=repo, home=home)
    return lab, "job-curve"


def _mcp(lab: Lab, args: dict[str, Any]) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        async with Client(build_server(lab)) as c:
            res = await c.call_tool("metrics", args)
            assert res.structured_content is not None
            return res.structured_content

    return asyncio.run(go())


def _cli(*args: str) -> dict[str, Any]:
    result = runner.invoke(app, ["metrics", *args])
    assert result.exit_code == 0, result.output
    loaded: dict[str, Any] = json.loads(result.stdout)
    return loaded


@pytest.mark.parametrize(
    ("cli_args", "mcp_args"),
    [
        ((), {}),
        (("--max-points", "5"), {"max_points": 5}),
        (("--max-points", "0"), {"max_points": 0}),
        (("--max-points", "7", "--downsample"), {"max_points": 7, "downsample": True}),
        (("--since-step", "440", "-n", "loss"), {"since_step": 440, "names": ["loss"]}),
    ],
)
def test_cli_and_mcp_return_the_same_bounded_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_args: tuple[str, ...], mcp_args: dict
) -> None:
    lab, job = _project(tmp_path, monkeypatch)

    assert _cli(job, *cli_args) == _mcp(lab, {"job_id": job, **mcp_args})


def test_the_default_is_bounded_and_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, job = _project(tmp_path, monkeypatch)

    out = _cli(job)

    assert out["max_points"] == DEFAULT_MAX_POINTS
    assert [p["step"] for p in out["series"]["loss"]] == list(range(N - DEFAULT_MAX_POINTS, N))
    assert out["truncated"]["loss"] == {"returned": DEFAULT_MAX_POINTS, "omitted": N - 200}


def test_max_points_zero_returns_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, job = _project(tmp_path, monkeypatch)

    out = _cli(job, "--max-points", "0")

    assert len(out["series"]["loss"]) == N and out["truncated"] == {}


def test_a_negative_cap_is_a_usage_error_on_both_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab, job = _project(tmp_path, monkeypatch)

    result = runner.invoke(app, ["metrics", job, "--max-points", "-1"])
    assert result.exit_code == 2
    with pytest.raises(ToolError, match="max_points"):
        _mcp(lab, {"job_id": job, "max_points": -1})
