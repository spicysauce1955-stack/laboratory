"""MCP server exposing the lab as structured tools (FR-F1, spec §9).

Run (stdio):  uv run python -m lab.mcp_server

Tools return JSON-serializable dicts so FastMCP emits ``structuredContent``. Unknown jobs and
submission errors raise ``ToolError`` (fail-loud, FR-F3). ``submit`` chooses the backend
(``local``/``skypilot``); job-scoped tools pick the backend the job actually ran on (from its
manifest), mirroring the CLI. ``build_server(lab)`` lets tests inject a Lab at a temp home;
``__main__`` runs the default repo-rooted lab.
"""

from __future__ import annotations

from datetime import datetime

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from lab.core import Lab, LabError, default_lab
from lab.models import JobManifest, JobSpec, ResourceRequest
from lab.store import JobStore


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def build_server(lab: Lab) -> FastMCP:
    mcp: FastMCP = FastMCP("laboratory")
    home = lab.home
    store = JobStore(home)

    def _require(job_id: str) -> JobManifest:
        try:
            return store.read_manifest(job_id)
        except FileNotFoundError as e:
            raise ToolError(f"job '{job_id}' not found") from e

    def _lab(backend: str = "local") -> Lab:
        return default_lab(home=home, backend=backend)

    def _lab_for(job_id: str) -> Lab:
        return default_lab(home=home, backend=_require(job_id).backend.provisioner)

    @mcp.tool
    def submit(
        command: str,
        backend: str = "local",
        cache: bool = False,
        seed: int | None = None,
        code_ref: str = "HEAD",
        cpus: int | None = None,
        memory: str | None = None,
        gpus: int | None = None,
        accelerators: str | None = None,
        timeout: str | None = None,
    ) -> dict:
        """Submit a job without blocking (backend local|skypilot); returns {job_id, cached, status} (FR-A1). cache=True reuses a prior identical succeeded job (FR-B5)."""
        the_lab = _lab(backend)
        spec = JobSpec(
            code_ref=code_ref,
            command=command,
            seed=seed,
            resources=ResourceRequest(
                cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout
            ),
            submitted_by="agent",
        )
        if cache and (cached_id := the_lab.find_cached(spec)) is not None:
            return {"job_id": cached_id, "cached": True, "status": the_lab.status(cached_id).value}
        try:
            job_id = the_lab.submit(spec)
        except LabError as e:
            raise ToolError(str(e)) from e
        return {"job_id": job_id, "cached": False, "status": the_lab.status(job_id).value}

    @mcp.tool
    def sweep(
        command: str,
        grid: dict[str, list],
        backend: str = "local",
        seed: int | None = None,
        cpus: int | None = None,
        memory: str | None = None,
        gpus: int | None = None,
        accelerators: str | None = None,
        timeout: str | None = None,
    ) -> dict:
        """Submit a parameter-grid sweep (one job per point under a sweep_id); {sweep_id, job_ids} (FR-A5)."""
        the_lab = _lab(backend)
        try:
            sweep_id, job_ids = the_lab.sweep(
                command,
                grid,
                seed=seed,
                resources=ResourceRequest(
                    cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout
                ),
            )
        except LabError as e:
            raise ToolError(str(e)) from e
        return {"sweep_id": sweep_id, "job_ids": job_ids}

    @mcp.tool
    def status(job_id: str) -> dict:
        """Return a job's state + timing (FR-A2); cheap to poll (FR-G2)."""
        m = _require(job_id)
        state = _lab_for(job_id).status(job_id)
        return {
            "job_id": job_id,
            "state": state.value,
            "started_at": _iso(m.started_at),
            "ended_at": _iso(m.ended_at),
            "exit_code": m.exit_code,
            "end_reason": m.end_reason,
            "cost": m.cost.model_dump() if m.cost else None,
        }

    @mcp.tool
    def metrics(
        job_id: str, names: list[str] | None = None, since_step: int | None = None
    ) -> dict:
        """Query incremental metric series; returns {series:{name:[{step,value,wall_time}]}} (FR-D2)."""
        _require(job_id)
        return {"series": _lab_for(job_id).metrics(job_id, names=names, since_step=since_step)}

    @mcp.tool
    def logs(job_id: str, tail: int | None = 100) -> dict:
        """Tail a job's logs; returns {lines: [...]} (FR-D1)."""
        _require(job_id)
        return {"lines": _lab_for(job_id).logs(job_id, tail=tail)}

    @mcp.tool
    def fetch_artifacts(job_id: str) -> dict:
        """Collect artifacts into runs/<job_id>/; returns {local_paths, artifacts} (FR-E2)."""
        _require(job_id)
        arts = _lab_for(job_id).fetch_artifacts(job_id)
        return {
            "local_paths": [a.path for a in arts],
            "artifacts": [a.model_dump() for a in arts],
        }

    @mcp.tool
    def cancel(job_id: str) -> dict:
        """Cancel a job and tear down its machine; returns {state} (FR-A3, FR-C2)."""
        _require(job_id)
        return {"job_id": job_id, "state": _lab_for(job_id).cancel(job_id).value}

    @mcp.tool(name="list")
    def list_jobs() -> dict:
        """List jobs; returns {jobs: [...]} (FR-H1)."""
        return {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "sweep_id": j.sweep_id,
                    "status": j.status.value,
                    "created_at": _iso(j.created_at),
                }
                for j in _lab().list_jobs()
            ]
        }

    return mcp


if __name__ == "__main__":
    build_server(default_lab()).run()
