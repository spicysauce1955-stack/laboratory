"""MCP server exposing the lab as structured tools (FR-F1, spec §9).

Run (stdio):  uv run python -m lab.mcp_server

Tools return JSON-serializable dicts so FastMCP emits ``structuredContent``. Unknown jobs and
submission errors raise ``ToolError`` (fail-loud, FR-F3). ``build_server(lab)`` lets tests inject
a Lab pointed at a temp dir; ``__main__`` runs the default repo-rooted lab.
"""

from __future__ import annotations

from datetime import datetime

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from lab.core import Lab, LabError, default_lab
from lab.models import JobSpec, ResourceRequest


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def build_server(lab: Lab) -> FastMCP:
    mcp: FastMCP = FastMCP("laboratory")

    def _require(job_id: str) -> None:
        try:
            lab.manifest(job_id)
        except FileNotFoundError as e:
            raise ToolError(f"job '{job_id}' not found") from e

    @mcp.tool
    def submit(
        command: str,
        seed: int | None = None,
        code_ref: str = "HEAD",
        cpus: int | None = None,
        timeout: str | None = None,
    ) -> dict:
        """Submit a job without blocking; returns {job_id, status} (FR-A1)."""
        try:
            job_id = lab.submit(
                JobSpec(
                    code_ref=code_ref,
                    command=command,
                    seed=seed,
                    resources=ResourceRequest(cpus=cpus, timeout=timeout),
                    submitted_by="agent",
                )
            )
        except LabError as e:
            raise ToolError(str(e)) from e
        return {"job_id": job_id, "status": lab.status(job_id).value}

    @mcp.tool
    def status(job_id: str) -> dict:
        """Return a job's state + timing (FR-A2); cheap to poll (FR-G2)."""
        _require(job_id)
        state = lab.status(job_id)
        m = lab.manifest(job_id)
        return {
            "job_id": job_id,
            "state": state.value,
            "started_at": _iso(m.started_at),
            "ended_at": _iso(m.ended_at),
            "exit_code": m.exit_code,
            "end_reason": m.end_reason,
        }

    @mcp.tool
    def logs(job_id: str, tail: int | None = 100) -> dict:
        """Tail a job's logs; returns {lines: [...]} (FR-D1)."""
        _require(job_id)
        return {"lines": lab.logs(job_id, tail=tail)}

    @mcp.tool
    def metrics(
        job_id: str, names: list[str] | None = None, since_step: int | None = None
    ) -> dict:
        """Query incremental metric series; returns {series:{name:[{step,value,wall_time}]}} (FR-D2)."""
        _require(job_id)
        return {"series": lab.metrics(job_id, names=names, since_step=since_step)}

    @mcp.tool
    def fetch_artifacts(job_id: str) -> dict:
        """Collect artifacts into runs/<job_id>/; returns {local_paths, artifacts} (FR-E2)."""
        _require(job_id)
        arts = lab.fetch_artifacts(job_id)
        return {
            "local_paths": [a.path for a in arts],
            "artifacts": [a.model_dump() for a in arts],
        }

    @mcp.tool
    def cancel(job_id: str) -> dict:
        """Cancel a job and tear down its machine; returns {state} (FR-A3, FR-C2)."""
        _require(job_id)
        return {"job_id": job_id, "state": lab.cancel(job_id).value}

    @mcp.tool(name="list")
    def list_jobs() -> dict:
        """List jobs; returns {jobs: [...]} (FR-H1)."""
        return {
            "jobs": [
                {"job_id": j.job_id, "status": j.status.value, "created_at": _iso(j.created_at)}
                for j in lab.list_jobs()
            ]
        }

    return mcp


if __name__ == "__main__":
    build_server(default_lab()).run()
