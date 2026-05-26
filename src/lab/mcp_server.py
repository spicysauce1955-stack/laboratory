"""MCP server exposing the lab as structured tools (FR-F1, spec §9).

Run:  uv run python -m lab.mcp_server   (stdio transport; register in your MCP client config)

Tools return typed Pydantic models so FastMCP emits ``structuredContent`` / ``outputSchema``
(machine-readable JSON, not free text). Stubbed until the core + backends land.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp: FastMCP = FastMCP("laboratory")

_NOT_IMPL = "Not implemented yet — see the P0 build order in research/16-decisions.md."


@mcp.tool
def submit(code_ref: str, command: str, seed: int | None = None) -> dict:
    """Submit a job without blocking; returns {job_id, status} (FR-A1)."""
    raise NotImplementedError(_NOT_IMPL)


@mcp.tool
def status(job_id: str) -> dict:
    """Return {state, started_at?, eta?, ...} for a job (FR-A2, cheap to poll FR-G2)."""
    raise NotImplementedError(_NOT_IMPL)


@mcp.tool
def logs(job_id: str, tail: int | None = None) -> dict:
    """Return {lines: [...]} from the job's logs (FR-D1)."""
    raise NotImplementedError(_NOT_IMPL)


@mcp.tool
def fetch_artifacts(job_id: str, dest: str | None = None) -> dict:
    """Fetch artifacts into runs/<job_id>/; returns {local_paths: [...]} (FR-E2)."""
    raise NotImplementedError(_NOT_IMPL)


@mcp.tool
def cancel(job_id: str) -> dict:
    """Cancel a job and tear down its machine; returns {state} (FR-A3, FR-C2)."""
    raise NotImplementedError(_NOT_IMPL)


if __name__ == "__main__":
    mcp.run()
