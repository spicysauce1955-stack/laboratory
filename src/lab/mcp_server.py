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
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from lab._util import wrap_with_extras
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
        provision_timeout: str | None = None,
        with_pkg: list[str] | None = None,
    ) -> dict:
        """Submit a job without blocking (backend local|skypilot); returns {job_id, cached, status} (FR-A1). cache=True reuses a prior identical succeeded job (FR-B5); with_pkg layers extra runtime packages via uv run --with. provision_timeout (skypilot, e.g. '10m', default 8m) aborts a host that never reaches UP."""
        the_lab = _lab(backend)
        spec = JobSpec(
            code_ref=code_ref,
            command=wrap_with_extras(command, with_pkg),
            seed=seed,
            resources=ResourceRequest(
                cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout,
                provision_timeout=provision_timeout,
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
        provision_timeout: str | None = None,
        with_pkg: list[str] | None = None,
    ) -> dict:
        """Submit a parameter-grid sweep (one job per point under a sweep_id); {sweep_id, job_ids} (FR-A5). with_pkg layers extra runtime packages via uv run --with. provision_timeout (skypilot, e.g. '10m', default 8m) aborts a host that never reaches UP."""
        the_lab = _lab(backend)
        try:
            sweep_id, job_ids = the_lab.sweep(
                wrap_with_extras(command, with_pkg),
                grid,
                seed=seed,
                resources=ResourceRequest(
                    cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout,
                    provision_timeout=provision_timeout,
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

    from lab.scheduler.models import DailyWindow, Guardrails, RegState, Triggers
    from lab.scheduler.queue import default_queue
    from lab.scheduler.register import register as sched_register
    from lab.scheduler.register import worst_case_cost

    @mcp.tool
    def register(
        command: str,
        expires: str,
        seed: int | None = None,
        cpus: int | None = None,
        memory: str | None = None,
        accelerators: str | None = None,
        timeout: str | None = None,
        window: str | None = None,
        tz: str = "UTC",
        not_before: str | None = None,
        max_hourly: float | None = None,
        offer_query: str | None = None,
        max_cost: float | None = None,
        after: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register a deferred job: launched by the scheduler when all triggers hold (time window HH:MM-HH:MM, Vast price <= max_hourly $/h, after=reg_ids succeeded). expires (+3d / ISO) is the required run-by guardrail; worst-case cost = max_hourly x timeout."""
        from datetime import datetime, timedelta
        from datetime import time as dt_time

        from lab._util import now, parse_duration

        if accelerators and timeout is None:
            raise ToolError("timeout is required for GPU registrations (it is the cost bound)")
        if expires.startswith("+"):
            secs = parse_duration(expires[1:])
            if secs is None:
                raise ToolError(f"bad relative expiry {expires!r}")
            expires_at = now() + timedelta(seconds=secs)
        else:
            expires_at = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        win = None
        if window:
            try:
                s, e = window.split("-", 1)
                win = DailyWindow(
                    start=dt_time.fromisoformat(s.strip()),
                    end=dt_time.fromisoformat(e.strip()),
                    tz=tz,
                )
            except ValueError as exc:
                raise ToolError(f"window expects HH:MM-HH:MM (got {window!r})") from exc
        triggers = Triggers(
            not_before=(
                datetime.fromisoformat(not_before.replace("Z", "+00:00")) if not_before else None
            ),
            window=win,
            max_hourly_usd=max_hourly,
            offer_query=offer_query,
            after=list(after or []),
        )
        spec = JobSpec(
            command=command,
            seed=seed,
            resources=ResourceRequest(
                cpus=cpus, memory=memory, accelerators=accelerators, timeout=timeout
            ),
            submitted_by="agent",
        )
        try:
            reg = sched_register(
                lab.repo, default_queue(), spec, triggers,
                Guardrails(expires_at=expires_at, max_cost_usd=max_cost),
            )
        except Exception as e:  # noqa: BLE001
            raise ToolError(str(e)) from e
        return {
            "reg_id": reg.reg_id,
            "state": reg.state.value,
            "expires_at": _iso(reg.guardrails.expires_at),
            "worst_case_cost_usd": worst_case_cost(triggers, spec.resources),
        }

    @mcp.tool
    def queue_list() -> dict[str, Any]:
        """List deferred registrations: state, job_id, last skip reason, scheduler heartbeat age."""
        queue = default_queue()
        hb = queue.read_heartbeat()
        age = None
        if hb and "at" in hb:
            from lab._util import now

            age = max(0.0, (now() - datetime.fromisoformat(str(hb["at"]))).total_seconds())
        return {
            "heartbeat_age_s": age,
            "control": queue.read_control().model_dump(),
            "entries": [
                {
                    "reg_id": r.reg_id,
                    "state": "held" if (r.state is RegState.pending and queue.held(r.reg_id))
                    else r.state.value,
                    "job_id": r.job_id,
                    "last_skip_reason": r.last_skip_reason,
                    "expires_at": _iso(r.guardrails.expires_at),
                }
                for r in queue.list_entries()
            ],
        }

    @mcp.tool
    def queue_show(reg_id: str) -> dict[str, Any]:
        """Full registration record (triggers, guardrails, provenance, state history fields)."""
        import json as _json

        try:
            reg = default_queue().get_entry(reg_id)
        except FileNotFoundError as e:
            raise ToolError(f"registration '{reg_id}' not found") from e
        loaded: dict[str, Any] = _json.loads(reg.model_dump_json())
        return loaded

    @mcp.tool
    def queue_cancel(reg_id: str) -> dict[str, Any]:
        """Request cancellation; the scheduler applies it on its next tick (also cancels a launched job)."""
        queue = default_queue()
        try:
            queue.get_entry(reg_id)
        except FileNotFoundError as e:
            raise ToolError(f"registration '{reg_id}' not found") from e
        queue.request_cancel(reg_id)
        return {"reg_id": reg_id, "cancel_requested": True}

    @mcp.tool
    def queue_pause(paused: bool = True) -> dict[str, Any]:
        """Pause/resume all scheduler launches (global switch; heartbeat keeps beating)."""
        queue = default_queue()
        queue.write_control(queue.read_control().model_copy(update={"paused": paused}))
        return {"paused": paused}

    return mcp


if __name__ == "__main__":
    build_server(default_lab()).run()
