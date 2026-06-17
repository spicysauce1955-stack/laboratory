"""Thin-shell wiring for the reproducibility gate: `lab confirm` CLI exit codes and the MCP
`confirm` tool. The verdict logic itself is covered end-to-end in test_confirm.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from helpers import PYTHON, make_manifest, wait_terminal
from typer.testing import CliRunner

import lab.cli as cli_mod
from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab, LabError
from lab.manifest import current_commit, repo_root
from lab.mcp_server import build_server
from lab.models import CodeRef, JobSpec, JobState

EXAMPLE = "experiments/example_capacity.py"


class _FakeLab:
    def __init__(self, result: dict | None = None, error: str | None = None) -> None:
        self._result = result
        self._error = error

    def confirm(self, run_id, **kwargs):
        if self._error is not None:
            raise LabError(self._error)
        return self._result


def _invoke(monkeypatch, fake: _FakeLab, *args: str):
    monkeypatch.setattr(cli_mod, "_lab_for_or_fail", lambda job_id: fake)
    return CliRunner().invoke(app, ["confirm", *args])


def test_cli_confirm_match_exits_zero(monkeypatch):
    fake = _FakeLab(result={"orig_id": "j1", "confirm_id": "j2", "verdict": "match", "deltas": {}})
    res = _invoke(monkeypatch, fake, "j1")
    assert res.exit_code == 0
    assert json.loads(res.stdout)["verdict"] == "match"


def test_cli_confirm_drift_exits_nonzero(monkeypatch):
    fake = _FakeLab(result={"orig_id": "j1", "confirm_id": "j2", "verdict": "drift", "deltas": {}})
    res = _invoke(monkeypatch, fake, "j1")
    assert res.exit_code == 1
    assert json.loads(res.stdout)["verdict"] == "drift"


def test_cli_confirm_rerun_failed_exits_nonzero(monkeypatch):
    fake = _FakeLab(result={"orig_id": "j1", "confirm_id": "j2", "verdict": "rerun_failed"})
    res = _invoke(monkeypatch, fake, "j1")
    assert res.exit_code == 1


def test_cli_confirm_gate_error_exits_nonzero(monkeypatch):
    fake = _FakeLab(error="cannot confirm j1: its producing run is 'failed'")
    res = _invoke(monkeypatch, fake, "j1")
    assert res.exit_code == 1
    assert "failed" in json.loads(res.stdout)["error"]


def test_cli_confirm_pending_exits_zero(monkeypatch):
    fake = _FakeLab(result={"orig_id": "j1", "confirm_id": "j2", "verdict": "pending"})
    res = _invoke(monkeypatch, fake, "j1", "--no-wait")
    assert res.exit_code == 0  # --no-wait is not a failure


class _CapturingLab:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def confirm(self, run_id, **kwargs):
        self.kwargs = {"run_id": run_id, **kwargs}
        return {"orig_id": run_id, "confirm_id": "j2", "verdict": "match", "deltas": {}}


def test_cli_confirm_flags_reach_core(monkeypatch):
    """--metric (repeatable), --rtol, --atol, --no-wait, --timeout are plumbed into Lab.confirm."""
    cap = _CapturingLab()
    monkeypatch.setattr(cli_mod, "_lab_for_or_fail", lambda job_id: cap)
    res = CliRunner().invoke(
        app,
        ["confirm", "j1", "--metric", "acc", "--metric", "loss",
         "--rtol", "0.05", "--atol", "0.01", "--no-wait", "--timeout", "30"],
    )
    assert res.exit_code == 0
    assert cap.kwargs["run_id"] == "j1"
    assert cap.kwargs["metrics"] == ["acc", "loss"]
    assert cap.kwargs["rtol"] == 0.05
    assert cap.kwargs["atol"] == 0.01
    assert cap.kwargs["wait"] is False
    assert cap.kwargs["timeout"] == 30.0


def test_cli_confirm_default_metric_is_none(monkeypatch):
    """No --metric -> metrics=None (judge every baseline metric), not an empty list."""
    cap = _CapturingLab()
    monkeypatch.setattr(cli_mod, "_lab_for_or_fail", lambda job_id: cap)
    CliRunner().invoke(app, ["confirm", "j1"])
    assert cap.kwargs["metrics"] is None
    assert cap.kwargs["wait"] is True


# --- MCP tool: real lab, real run -----------------------------------------


def _seed_original(lab: Lab, *, status: JobState, final_metrics: dict[str, float]) -> str:
    job_id = f"orig-{status.value}"
    m = make_manifest(job_id, f"{PYTHON} {EXAMPLE}", seed=7).model_copy(
        update={
            "code": CodeRef(git_commit=current_commit(repo_root(Path.cwd())), git_dirty=False),
            "status": status,
            "final_metrics": final_metrics,
        }
    )
    lab.store.create(m)
    lab.store.write_manifest(m)
    return job_id


def test_mcp_confirm_returns_verdict(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    # learn the deterministic baseline, then seed a clean succeeded original with it
    jid = lab.submit(JobSpec(code_ref="HEAD", command=f"{PYTHON} {EXAMPLE}", seed=7))
    assert wait_terminal(backend, jid) == JobState.succeeded
    orig = _seed_original(lab, status=JobState.succeeded,
                          final_metrics=dict(lab.manifest(jid).final_metrics))
    server = build_server(lab)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool("confirm", {"run_id": orig, "timeout": 60})).data

    data = asyncio.run(go())
    assert data["verdict"] == "match"


def test_mcp_confirm_gate_raises_tool_error(tmp_path: Path):
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    orig = _seed_original(lab, status=JobState.failed, final_metrics={"demo_metric": 0.5})
    server = build_server(lab)

    async def go():
        async with Client(server) as c:
            await c.call_tool("confirm", {"run_id": orig})

    with pytest.raises(ToolError, match="failed"):
        asyncio.run(go())
