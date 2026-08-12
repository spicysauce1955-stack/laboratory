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
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from lab._util import parse_duration, wrap_with_extras
from lab.core import Lab, LabError, default_lab, job_status_view, resolve_backend_profile
from lab.env import load_lab_env
from lab.manifest import repo_root
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
        disk_size: int | None = None,
        accelerators: str | None = None,
        cloud: str | None = None,
        region: str | None = None,
        zone: str | None = None,
        price_cap: float | None = None,
        timeout: str | None = None,
        provision_timeout: str | None = None,
        with_pkg: list[str] | None = None,
        use_spot: bool = False,
        spot_fallback: bool = True,
        allow_dirty: bool = True,
        allow_unknown_config: bool = False,
        preflight: bool = True,
    ) -> dict[str, Any]:
        """Submit a job without blocking (backend local|skypilot); returns {job_id, cached, status} (FR-A1). cache=True reuses a prior identical succeeded job (FR-B5); with_pkg layers extra runtime packages via uv run --with. provision_timeout (skypilot, e.g. '10m', default 8m) aborts a host that never reaches UP. use_spot uses spot instances (skypilot); spot_fallback=False makes it spot-only. allow_dirty=False refuses a dirty working tree (default snapshots the diff, FR-B1). disk_size sizes the boot/attached volume in GB (skypilot; DO volume size). cloud picks the skypilot cloud: vast (default) | do | gcp — GCP runs both CPU and GPU jobs (accelerators e.g. 'T4:1'/'L4:1', spot allowed). backend="cpu" provisions a cheap CPU box (DigitalOcean by default, cloud="gcp" to override; default 4 vCPU + 50GB volume, up to 48; --accelerators rejected). region/zone pin where it lands (default: SkyPilot's optimizer picks the cheapest available). price_cap is a hard $/hr ceiling on compute enforced by the optimizer. preflight=False skips the pre-launch credential/quota checks (see the doctor tool)."""
        resources = ResourceRequest(
            cpus=cpus, memory=memory, gpus=gpus, disk_size=disk_size, accelerators=accelerators,
            cloud=cloud, region=region, zone=zone, max_hourly_usd=price_cap,
            timeout=timeout, provision_timeout=provision_timeout, use_spot=use_spot,
            spot_fallback=spot_fallback,
        )
        try:
            provisioner, resources = resolve_backend_profile(backend, resources)
        except LabError as e:
            raise ToolError(str(e)) from e
        the_lab = _lab(provisioner)
        spec = JobSpec(
            code_ref=code_ref,
            command=wrap_with_extras(command, with_pkg),
            seed=seed,
            resources=resources,
            submitted_by="agent",
            allow_unknown_config=allow_unknown_config,
        )
        if cache and (cached_id := the_lab.find_cached(spec)) is not None:
            return {"job_id": cached_id, "cached": True, "status": the_lab.status(cached_id).value}
        try:
            job_id = the_lab.submit(spec, allow_dirty=allow_dirty, preflight=preflight)
        except LabError as e:
            raise ToolError(str(e)) from e
        return {"job_id": job_id, "cached": False, "status": the_lab.status(job_id).value}

    @mcp.tool
    def confirm(
        run_id: str,
        metric: list[str] | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-12,
        wait: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Reproducibility gate (FR-B): re-derive a prior result from its pinned provenance and check it still holds. Relaunches run_id fresh (no cache) from its committed commit, then compares the re-run's final metric(s) against the original's snapshot within tolerance -> verdict 'match'|'drift'|'rerun_failed' with per-metric deltas. Raises ToolError for a non-succeeded or dirty producer (no honest result to re-derive) or a missing baseline. metric restricts which metrics are judged (default: all). wait=False submits the re-run and returns {confirm_id, verdict:'pending'}."""
        _require(run_id)
        try:
            return _lab_for(run_id).confirm(
                run_id, metrics=metric or None, rtol=rtol, atol=atol, wait=wait, timeout=timeout
            )
        except LabError as e:
            raise ToolError(str(e)) from e

    @mcp.tool
    def sweep(
        command: str,
        grid: dict[str, list[Any]] | None = None,
        backend: str = "local",
        seed: int | None = None,
        cpus: int | None = None,
        memory: str | None = None,
        gpus: int | None = None,
        disk_size: int | None = None,
        accelerators: str | None = None,
        cloud: str | None = None,
        region: str | None = None,
        zone: str | None = None,
        price_cap: float | None = None,
        timeout: str | None = None,
        provision_timeout: str | None = None,
        with_pkg: list[str] | None = None,
        use_spot: bool = False,
        spot_fallback: bool = True,
        sweep_max_cost: float | None = None,
        seeds: str | list[int] | None = None,
        shard_size: int | None = None,
        results_file: str = "results.csv",
        seed_column: str = "seed",
        row_key: str | list[str] | None = None,
        allow_unknown_config: bool = False,
    ) -> dict[str, Any]:
        """Submit a parameter-grid sweep (one job per point under a sweep_id); {sweep_id, job_ids} (FR-A5). with_pkg layers extra runtime packages via uv run --with. provision_timeout (skypilot, e.g. '10m', default 8m) aborts a host that never reaches UP. use_spot uses spot instances (skypilot); spot_fallback=False makes it spot-only. sweep_max_cost is an up-front admission cap: the sweep is refused if its total would exceed the daily budget (during-run enforcement is on register_sweep). disk_size sizes the boot/attached volume in GB (skypilot; DO volume size). cloud picks the skypilot cloud: vast (default) | do | gcp. backend="cpu" provisions a cheap CPU box (DigitalOcean by default, cloud="gcp" to override; default 4 vCPU + 50GB volume, up to 48; --accelerators rejected). With seeds + shard_size each cell's seeds are split into shards of at most shard_size, run as independent jobs (own timeout + teardown) and aggregated per cell; returns {sweep_id, cells:[{coords, shard_job_ids, aggregate_ref, seeds_expected, seeds_present, status}]}. results_file/seed_column name the per-run result table and its seed column; row_key (e.g. 'seed,alpha') names the columns identifying a row when an inner-loop axis writes multiple rows per seed."""
        from lab.scheduler.queue import default_queue

        resources = ResourceRequest(
            cpus=cpus, memory=memory, gpus=gpus, disk_size=disk_size, accelerators=accelerators,
            cloud=cloud, region=region, zone=zone, max_hourly_usd=price_cap,
            timeout=timeout, provision_timeout=provision_timeout, use_spot=use_spot,
            spot_fallback=spot_fallback,
        )
        try:
            provisioner, resources = resolve_backend_profile(backend, resources)
        except LabError as e:
            raise ToolError(str(e)) from e
        the_lab = _lab(provisioner)
        try:
            sweep_id, job_ids = the_lab.sweep(
                wrap_with_extras(command, with_pkg),
                grid or {},
                seed=seed,
                resources=resources,
                sweep_max_cost=sweep_max_cost,
                daily_budget=(
                    default_queue().read_control().budget_usd_per_day
                    if sweep_max_cost is not None
                    else None
                ),
                seeds=seeds,
                shard_size=shard_size,
                results_file=results_file,
                seed_column=seed_column,
                row_key=row_key,
                allow_unknown_config=allow_unknown_config,
            )
        except LabError as e:
            raise ToolError(str(e)) from e
        if the_lab.store.has_sweep_plan(sweep_id):
            return the_lab.sweep_plan(sweep_id).view()
        return {"sweep_id": sweep_id, "job_ids": job_ids}

    @mcp.tool
    def status(job_id: str) -> dict[str, Any]:
        """Return a job's state + timing + cost + teardown_status (FR-A2/FR-I2/FR-C2); cheap to poll (FR-G2). teardown_status "failed" is a money alarm: a paid machine may still be running — call reconcile. Scheduler-launched (deferred) jobs are read from the mirrored manifest (mirrored=true; may be a tick stale)."""
        try:
            view = job_status_view(home, lab.repo, job_id)
        except FileNotFoundError as e:
            raise ToolError(f"job '{job_id}' not found (locally or in the scheduler mirror)") from e
        view["started_at"] = _iso(view["started_at"])
        view["ended_at"] = _iso(view["ended_at"])
        view["last_log_at"] = _iso(view["last_log_at"])
        return view

    @mcp.tool
    def wait(
        job_ids: list[str] | None = None,
        sweep: str | None = None,
        timeout: float | str = 600,
        interval: float = 10,
        fail_fast: bool = False,
    ) -> dict[str, Any]:
        """Block until the job(s) (or every job in a sweep) reach a terminal state, up to timeout (seconds or '10m'/'2h'), then return the FR-C2 verdict: {all_terminal, failed_fast, pending, teardown_leaks, teardown_unconfirmed, jobs}. fail_fast=True returns as soon as any job is failed/timed_out (failed_fast=true, offender listed first; pending names still-running — and still-billing — jobs; nothing is cancelled). Non-empty teardown_leaks = a paid machine may still be billing — call reconcile. teardown_unconfirmed = teardown never recorded either way; treat as suspect, not clean."""
        try:
            timeout_s = parse_duration(timeout)
        except ValueError as e:
            raise ToolError(f"bad timeout {timeout!r}: {e}") from e
        ids = _lab().jobs_in_sweep(sweep) if sweep else list(job_ids or [])
        if not ids:
            raise ToolError("pass job id(s) or sweep=<sweep_id>")
        for j in ids:
            _require(j)
        return _lab_for(ids[0]).wait_summary(
            ids, interval=interval, timeout=timeout_s, fail_fast=fail_fast
        )

    @mcp.tool
    def export(job_or_sweep_id: str, to: str, include_logs: bool = False) -> dict[str, Any]:
        """Export a committable provenance bundle for a job or sweep to a directory: manifests + result tables (results.csv etc.) + resolved config + code diffs (+ sweep plan/cell aggregates), indexed by index.json (files tied to commit/seed/state/spend). Blobs (.npz/checkpoints) and oversized files are excluded and listed under `skipped`. Use this to move the durable subset of runs/ into the analysis repo (field-report #5)."""
        try:
            return lab.export(job_or_sweep_id, Path(to), include_logs=include_logs)
        except LabError as e:
            raise ToolError(str(e)) from e

    @mcp.tool
    def reconcile() -> dict[str, Any]:
        """Dry-run cloud leak sweep (FR-C2): cross-checks provider instances (Vast rentals, SkyPilot-tracked clusters covering DO/GCP, GCE instances + unattached disks, DO volumes) against local jobs. Read-only — it never destroys anything; run `lab reconcile --apply --yes` at the CLI to clean up (`--apply` alone prompts, and refuses with exit 4 when there is no terminal). Non-empty orphans/sky_orphans/gcp_orphans/gcp_disk_orphans/do_volume_orphans/unsupervised means something may still be billing. `gcp_unmatched` lists `lab-*` GCE resources that did NOT match our node shape: they are never destroyed, but if they look like ours the leak passes have gone blind. `gcp_project` names the project actually swept — check it is the one SkyPilot launches into."""
        try:
            return _lab("skypilot").reconcile(apply=False)
        except LabError as e:
            raise ToolError(str(e)) from e

    @mcp.tool
    def doctor(
        cloud: str | None = None,
        accelerators: str | None = None,
        cpus: int | None = None,
        disk_size: int | None = None,
        region: str | None = None,
        zone: str | None = None,
        use_spot: bool = False,
    ) -> dict[str, Any]:
        """Check whether a launch on this cloud would work, BEFORE it costs a provision — run this when a submit fails to provision, or before the first job on a new cloud/shape. Returns {cloud, ok, checks:[{name,status,detail,fix}], blocking:[...]} where status is ok|warn|fail|skip. Verifies credentials (including SkyPilot's API-server daemon, which does not inherit .env), project, billing, enabled APIs, IAM permissions, and quota for the requested shape, then reports the catalog's instance type and price band. GCP GPU quota is checked at both levels Google enforces: a project can hold regional NVIDIA_*_GPUS quota and still be blocked by a global GPUS_ALL_REGIONS of 0. status="skip" means the check could not answer and is never a blocker; only "fail" is a definitive negative."""
        from lab.core import default_disk_gb, validate_cloud
        from lab.doctor import doctor_view, run_checks

        try:
            validate_cloud(cloud)
        except LabError as e:
            raise ToolError(str(e)) from e
        resources = ResourceRequest(
            cpus=cpus, disk_size=disk_size, accelerators=accelerators,
            cloud=cloud, region=region, zone=zone, use_spot=use_spot,
        )
        resources = resources.model_copy(update={"disk_size": default_disk_gb(resources)})
        return doctor_view(cloud, run_checks(cloud, resources, home=home))

    @mcp.tool
    def metrics(
        job_id: str, names: list[str] | None = None, since_step: int | None = None
    ) -> dict[str, Any]:
        """Query incremental metric series; returns {series:{name:[{step,value,wall_time}]}} (FR-D2)."""
        _require(job_id)
        return {"series": _lab_for(job_id).metrics(job_id, names=names, since_step=since_step)}

    @mcp.tool
    def logs(job_id: str, tail: int | None = 100) -> dict[str, Any]:
        """Tail a job's logs; returns {lines: [...]} (FR-D1)."""
        _require(job_id)
        return {"lines": _lab_for(job_id).logs(job_id, tail=tail)}

    @mcp.tool
    def fetch_artifacts(job_id: str) -> dict[str, Any]:
        """Collect artifacts into runs/<job_id>/; returns {local_paths, artifacts} (FR-E2)."""
        _require(job_id)
        arts = _lab_for(job_id).fetch_artifacts(job_id)
        return {
            "local_paths": [a.path for a in arts],
            "artifacts": [a.model_dump() for a in arts],
        }

    @mcp.tool
    def cancel(job_id: str) -> dict[str, Any]:
        """Cancel a job and tear down its machine; returns {state} (FR-A3, FR-C2)."""
        _require(job_id)
        return {"job_id": job_id, "state": _lab_for(job_id).cancel(job_id).value}

    @mcp.tool
    def sweep_status(sweep_id: str) -> dict[str, Any]:
        """Summarize a sweep's outcomes: preemptions, on-demand fallback, per-point spend."""
        return _lab().sweep_summary(sweep_id)

    @mcp.tool
    def sweep_aggregate(
        sweep_id: str, strict: bool = False, row_key: str | list[str] | None = None
    ) -> dict[str, Any]:
        """Row-concatenate each cell's shard results into one per-cell result; returns the cell view (P1-2). By default rows recovered from non-succeeded shards (timed-out etc.) are included — stamped with a _shard_status column and listed per-cell in seeds_partial. strict=True aggregates succeeded shards only. row_key (e.g. 'seed,alpha') declares the columns identifying a row for sweeps created before row_key existed; it persists onto the plan."""
        try:
            return _lab().aggregate_sweep(
                sweep_id, include_partial=not strict, row_key=row_key
            ).view()
        except LabError as e:
            raise ToolError(str(e)) from e

    @mcp.tool
    def sweep_retry(sweep_id: str) -> dict[str, Any]:
        """Resubmit only the missing shards of incomplete cells, then re-aggregate (P1-2)."""
        try:
            return _lab().retry_sweep(sweep_id).view()
        except LabError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(name="list")
    def list_jobs() -> dict[str, Any]:
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

    from lab.scheduler.models import Guardrails, RegState, Triggers
    from lab.scheduler.queue import default_queue
    from lab.scheduler.register import parse_expires, parse_window
    from lab.scheduler.register import register as sched_register
    from lab.scheduler.register import register_sweep as sched_register_sweep
    from lab.scheduler.register import worst_case_cost

    @mcp.tool
    def register(
        command: str,
        expires: str,
        seed: int | None = None,
        cpus: int | None = None,
        memory: str | None = None,
        accelerators: str | None = None,
        cloud: str | None = None,
        timeout: str | None = None,
        window: str | None = None,
        tz: str = "UTC",
        not_before: str | None = None,
        max_hourly: float | None = None,
        offer_query: str | None = None,
        max_cost: float | None = None,
        after: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register a deferred job: launched by the scheduler when all triggers hold (time window HH:MM-HH:MM, Vast price <= max_hourly $/h, after=reg_ids succeeded). expires (+3d / ISO) is the required run-by guardrail; worst-case cost = max_hourly x timeout. cloud picks the skypilot cloud: vast (default) | do | gcp — max_hourly/offer_query gate on Vast offers and are rejected for other clouds."""
        from datetime import datetime

        if accelerators and timeout is None:
            raise ToolError("timeout is required for GPU registrations (it is the cost bound)")
        try:
            expires_at = parse_expires(expires)
            win = parse_window(window, tz) if window else None
            not_before_dt = (
                datetime.fromisoformat(not_before.replace("Z", "+00:00")) if not_before else None
            )
        except ValueError as e:
            raise ToolError(str(e)) from e
        triggers = Triggers(
            not_before=not_before_dt,
            window=win,
            max_hourly_usd=max_hourly,
            offer_query=offer_query,
            after=list(after or []),
        )
        spec = JobSpec(
            command=command,
            seed=seed,
            resources=ResourceRequest(
                cpus=cpus, memory=memory, accelerators=accelerators, cloud=cloud, timeout=timeout
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
    def register_sweep(
        command: str,
        grid: dict[str, list[Any]],
        expires: str,
        seed: int | None = None,
        cpus: int | None = None,
        memory: str | None = None,
        accelerators: str | None = None,
        cloud: str | None = None,
        timeout: str | None = None,
        window: str | None = None,
        tz: str = "UTC",
        not_before: str | None = None,
        max_hourly: float | None = None,
        offer_query: str | None = None,
        max_cost: float | None = None,
        sweep_max_cost: float | None = None,
    ) -> dict[str, Any]:
        """Register a parameter grid as N deferred points sharing one sweep_id + ceiling + bundle;
        the always-on scheduler paces them (subject to triggers and the queue-wide max_concurrent
        control) and stops launching once finished-spend hits sweep_max_cost (it never kills a
        running point). grid is {key: [values]}. expires (+3d / ISO) is the required run-by
        guardrail. cloud: vast (default) | do | gcp — max_hourly/offer_query are Vast-only.
        Returns {sweep_id, count, reg_ids}."""
        from datetime import datetime

        if accelerators and timeout is None:
            raise ToolError("timeout is required for GPU registrations (it is the cost bound)")
        try:
            expires_at = parse_expires(expires)
            win = parse_window(window, tz) if window else None
            not_before_dt = (
                datetime.fromisoformat(not_before.replace("Z", "+00:00")) if not_before else None
            )
        except ValueError as e:
            raise ToolError(str(e)) from e
        triggers = Triggers(
            not_before=not_before_dt,
            window=win,
            max_hourly_usd=max_hourly,
            offer_query=offer_query,
        )
        queue = default_queue()
        try:
            sweep_id, regs = sched_register_sweep(
                lab.repo, queue, command, grid,
                resources=ResourceRequest(
                    cpus=cpus, memory=memory, accelerators=accelerators, cloud=cloud,
                    timeout=timeout
                ),
                triggers=triggers,
                guardrails=Guardrails(expires_at=expires_at, max_cost_usd=max_cost),
                seed=seed,
                sweep_max_cost=sweep_max_cost,
                daily_budget=queue.read_control().budget_usd_per_day,
                submitted_by="agent",
            )
        except Exception as e:  # noqa: BLE001
            raise ToolError(str(e)) from e
        return {"sweep_id": sweep_id, "count": len(regs), "reg_ids": [r.reg_id for r in regs]}

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
    # Before default_lab(): the .env carries the cloud project/credentials the backends read.
    load_lab_env(repo_root())
    build_server(default_lab()).run()
