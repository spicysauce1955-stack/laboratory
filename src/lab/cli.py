"""Human-facing CLI — a thin mirror of the MCP tools (FR-F2). Entry point: ``lab``.

Wired to the local backend by default; structured JSON output mirrors the MCP §9 returns.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from lab._util import now, wrap_with_extras
from lab.core import Lab, LabError, default_lab
from lab.manifest import repo_root
from lab.models import JobSpec, JobState, ResourceRequest
from lab.scheduler.models import Guardrails, RegState, Triggers
from lab.scheduler.price import PriceFeed
from lab.scheduler.queue import QueueStore, default_queue
from lab.scheduler.register import parse_expires, parse_window
from lab.scheduler.register import register as sched_register
from lab.scheduler.register import register_sweep as sched_register_sweep
from lab.scheduler.register import worst_case_cost
from lab.scheduler.tick import Scheduler
from lab.store import JobStore

_TERMINAL = {
    JobState.succeeded, JobState.failed, JobState.cancelled, JobState.timed_out, JobState.preempted
}

app = typer.Typer(
    help="Laboratory — remote experiment runner (CLI mirror of the MCP tools, spec §9).",
    no_args_is_help=True,
)


def _lab(backend: str = "local") -> Lab:
    return default_lab(backend=backend)


def _lab_for(job_id: str) -> Lab:
    """Build a Lab over whichever backend actually ran the job (from its manifest)."""
    home = repo_root() / "runs"
    provisioner = JobStore(home).read_manifest(job_id).backend.provisioner
    return default_lab(home=home, backend=provisioner)


def _lab_for_or_fail(job_id: str) -> Lab:
    """`_lab_for`, but a job missing from the local store is a structured error (FR-F3) —
    scheduler-launched jobs mirror only their manifest, so logs/metrics/fetch live on the
    scheduler host; `lab status` is the command that reads the mirror."""
    try:
        return _lab_for(job_id)
    except FileNotFoundError:
        _emit(
            {
                "error": (
                    f"unknown job id {job_id!r} — not in local runs/ "
                    "(for scheduler-launched jobs only `lab status` reads the mirrored manifest)"
                )
            }
        )
        raise typer.Exit(code=2) from None


def _emit(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


def _parse_grid(items: list[str]) -> dict[str, list[str]]:
    """Parse repeated `--grid key=v1,v2,...` options into {key: [values]}.

    Values stay strings — the experiment (e.g. Hydra) coerces types, so the lab doesn't guess.
    """
    grid: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"--grid expects key=v1,v2,... (got {item!r})")
        key, vals = item.split("=", 1)
        key = key.strip()
        values = [v.strip() for v in vals.split(",") if v.strip()]
        if not values:
            raise typer.BadParameter(f"--grid {key!r} has no values")
        if key in grid:
            raise typer.BadParameter(f"--grid {key!r} given more than once")
        grid[key] = values
    return grid


@app.command()
def submit(
    command: str = typer.Option(..., "--command", "-c", help="entrypoint, e.g. 'python experiments/x.py'"),
    backend: str = typer.Option("local", "--backend", help="local | skypilot"),
    cache: bool = typer.Option(False, "--cache", help="reuse a prior succeeded identical job (FR-B5)"),
    seed: int | None = typer.Option(None, help="explicit seed (recorded in the manifest)"),
    code_ref: str = typer.Option("HEAD", help="git ref to pin"),
    cpus: int | None = typer.Option(None),
    memory: str | None = typer.Option(None, help="e.g. 8 or 8+ (GB)"),
    gpus: int | None = typer.Option(None),
    accelerators: str | None = typer.Option(None, "--accelerators", help="e.g. RTX_3070:1 (required for Vast)"),
    timeout: str | None = typer.Option(
        None, help="hard wall-clock cap, e.g. 2h / 30m / 45s — on overrun the job is killed, the "
        "machine torn down, and the run marked timed_out (FR-I1)"
    ),
    provision_timeout: str | None = typer.Option(None, "--provision-timeout", help="abort if the host doesn't reach UP in time, e.g. 10m (skypilot; default 8m)"),
    with_pkg: list[str] = typer.Option(None, "--with", help="extra runtime package(s) for this job (repeatable; layered via uv run --with)"),
    spot: bool = typer.Option(False, "--spot", help="use spot/interruptible instances (skypilot)"),
    no_fallback: bool = typer.Option(
        False, "--no-fallback", "--spot-only",
        help="with --spot, do NOT fall back to on-demand if spot is scarce (wait/skip instead)",
    ),
    no_dirty: bool = typer.Option(
        False, "--no-dirty",
        help="refuse to launch from a dirty working tree (default: snapshot the diff, FR-B1)",
    ),
) -> None:
    """Submit a job without blocking; prints {job_id, cached, status} (FR-A1).

    Provenance is fail-closed (FR-B1): the manifest always pins a real commit, and a dirty tree is
    snapshotted into a reproducible diff (pass --no-dirty to refuse instead). On --timeout overrun
    the job is killed, the machine torn down, and the run marked timed_out with the wall in its
    end_reason.
    """
    lab = _lab(backend)
    spec = JobSpec(
        code_ref=code_ref,
        command=wrap_with_extras(command, with_pkg),
        seed=seed,
        resources=ResourceRequest(
            cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout,
            provision_timeout=provision_timeout, use_spot=spot, spot_fallback=not no_fallback,
        ),
        submitted_by="human",
    )
    if cache and (cached_id := lab.find_cached(spec)) is not None:
        _emit({"job_id": cached_id, "cached": True, "status": lab.status(cached_id).value})
        return
    try:
        job_id = lab.submit(spec, allow_dirty=not no_dirty)
    except LabError as e:  # fail-loud, actionable (FR-F3)
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit({"job_id": job_id, "cached": False, "status": lab.status(job_id).value})


@app.command()
def confirm(
    run_id: str = typer.Argument(..., help="the run to re-derive and verify"),
    metric: list[str] = typer.Option(
        None, "--metric", help="metric(s) to judge (repeatable; default: every baseline metric)"
    ),
    rtol: float = typer.Option(1e-3, "--rtol", help="relative tolerance for a match"),
    atol: float = typer.Option(1e-12, "--atol", help="absolute tolerance for a match (float noise floor)"),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="submit the fresh re-run and return its id without comparing"
    ),
    timeout: float | None = typer.Option(
        None, help="seconds to wait for the re-run (default: no limit)"
    ),
) -> None:
    """Re-derive a prior result from its pinned provenance and check it still holds (FR-B).

    Relaunches the run fresh (no cache) and compares its final metric(s) against the original within
    tolerance: match / drift / rerun_failed. Refuses a non-succeeded or dirty producer outright.
    Exits non-zero unless the verdict is 'match' (or '--no-wait'), so it can gate a writeup.
    """
    lab = _lab_for_or_fail(run_id)
    try:
        result = lab.confirm(
            run_id, metrics=metric or None, rtol=rtol, atol=atol, wait=not no_wait, timeout=timeout
        )
    except LabError as e:  # the gate (non-succeeded/dirty) and missing-baseline are fail-loud
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit(result)
    if result["verdict"] not in {"match", "pending"}:
        raise typer.Exit(code=1)


@app.command()
def sweep(
    command: str = typer.Option(..., "--command", "-c", help="entrypoint, e.g. 'python experiments/x.py'"),
    grid: list[str] = typer.Option(..., "--grid", "-g", help="key=v1,v2,... (repeatable)"),
    backend: str = typer.Option("local", "--backend", help="local | skypilot"),
    seed: int | None = typer.Option(None),
    cpus: int | None = typer.Option(None),
    memory: str | None = typer.Option(None),
    gpus: int | None = typer.Option(None),
    accelerators: str | None = typer.Option(None, "--accelerators"),
    timeout: str | None = typer.Option(None, help="wall-clock per job, e.g. 2h"),
    provision_timeout: str | None = typer.Option(None, "--provision-timeout", help="abort a host that doesn't reach UP in time, e.g. 10m (skypilot; default 8m)"),
    with_pkg: list[str] = typer.Option(None, "--with", help="extra runtime package(s) per job (repeatable; layered via uv run --with)"),
    spot: bool = typer.Option(False, "--spot", help="use spot/interruptible instances (skypilot)"),
    no_fallback: bool = typer.Option(
        False, "--no-fallback", "--spot-only",
        help="with --spot, do NOT fall back to on-demand if spot is scarce (wait/skip instead)",
    ),
    sweep_max_cost: float | None = typer.Option(None, "--sweep-max-cost", help="up-front admission cap in USD: refuse the sweep if its total would exceed your daily budget (cost-safety); during-run enforcement is on register-sweep"),
) -> None:
    """Submit a parameter-grid sweep: one job per point under a sweep_id (FR-A5)."""
    lab = _lab(backend)
    try:
        sweep_id, job_ids = lab.sweep(
            wrap_with_extras(command, with_pkg),
            _parse_grid(grid),
            seed=seed,
            resources=ResourceRequest(
                cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout,
                provision_timeout=provision_timeout, use_spot=spot, spot_fallback=not no_fallback,
            ),
            sweep_max_cost=sweep_max_cost,
            # only consult the control budget when there's a cap to admit against (avoids an
            # unnecessary queue read on every plain sweep)
            daily_budget=(
                default_queue().read_control().budget_usd_per_day
                if sweep_max_cost is not None
                else None
            ),
        )
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit({"sweep_id": sweep_id, "count": len(job_ids), "job_ids": job_ids})


@app.command()
def status(job_id: str) -> None:
    """Show a job's state + cost + teardown_status (FR-A2, FR-I2, FR-C2)."""
    try:
        lab = _lab_for(job_id)
    except FileNotFoundError:
        mirrored = default_queue().read_mirrored(job_id)  # scheduler-launched job (spec §4.3)
        if mirrored is None:
            _emit({"error": f"unknown job id {job_id!r}"})
            raise typer.Exit(code=2) from None
        _emit(
            {
                "job_id": job_id,
                "state": mirrored.status.value,
                "exit_code": mirrored.exit_code,
                "cost": mirrored.cost.model_dump() if mirrored.cost else None,
                "teardown_status": mirrored.teardown_status,
                "end_reason": mirrored.end_reason,
                "mirrored": True,  # may be up to one tick stale
            }
        )
        return
    state = lab.status(job_id)
    m = lab.manifest(job_id)
    _emit(
        {
            "job_id": job_id,
            "state": state.value,
            "exit_code": m.exit_code,
            "cost": m.cost.model_dump() if m.cost else None,
            "teardown_status": m.teardown_status,
            "end_reason": m.end_reason,
        }
    )


@app.command()
def logs(job_id: str, tail: int = typer.Option(100)) -> None:
    """Tail a job's logs (FR-D1)."""
    for line in _lab_for_or_fail(job_id).logs(job_id, tail=tail):
        typer.echo(line)


@app.command()
def metrics(
    job_id: str,
    name: list[str] = typer.Option(None, "--name", "-n", help="filter to these metric names"),
    since_step: int | None = typer.Option(None, help="only points with step > since_step"),
) -> None:
    """Query a job's incremental metric series (FR-D2 — the early-kill loop)."""
    _emit({"series": _lab_for_or_fail(job_id).metrics(job_id, names=name or None, since_step=since_step)})


@app.command()
def fetch(job_id: str) -> None:
    """Collect artifacts into runs/<job_id>/; prints local paths (FR-E2)."""
    arts = _lab_for_or_fail(job_id).fetch_artifacts(job_id)
    _emit({"local_paths": [a.path for a in arts], "artifacts": [a.model_dump() for a in arts]})


@app.command()
def cancel(job_id: str) -> None:
    """Cancel a job and tear down its machine (FR-A3, FR-C2)."""
    _emit({"job_id": job_id, "state": _lab_for_or_fail(job_id).cancel(job_id).value})


@app.command(name="sweep-status")
def sweep_status(sweep_id: str) -> None:
    """Summarize a sweep's outcomes: preemptions, on-demand fallback, per-point spend."""
    _emit(_lab().sweep_summary(sweep_id))


@app.command(name="list")
def list_jobs() -> None:
    """List jobs (FR-H1)."""
    jobs = _lab().list_jobs()
    _emit(
        {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "sweep_id": j.sweep_id,
                    "status": j.status.value,
                    "created_at": j.created_at,
                }
                for j in jobs
            ]
        }
    )


@app.command()
def wait(
    job_ids: list[str] = typer.Argument(None, help="job id(s) to wait for"),
    sweep: str | None = typer.Option(None, "--sweep", help="wait for all jobs in this sweep_id"),
    interval: float = typer.Option(10.0, help="seconds between cheap status polls (FR-G2)"),
    timeout: float | None = typer.Option(None, help="give up after N seconds"),
    done_file: Path | None = typer.Option(
        None, "--done-file", help="write the final summary here on completion (a sentinel a hook can watch)"
    ),
) -> None:
    """Block until the job(s) reach a terminal state, then exit (FR-G1).

    Run as a Claude Code background task — its completion is the push signal the session acts on,
    so the agent need not poll. Exits non-zero if it gave up on a timeout.
    """
    ids = _lab().jobs_in_sweep(sweep) if sweep else list(job_ids or [])
    if not ids:
        msg = f"sweep {sweep!r} matched no jobs" if sweep else "pass job id(s) or --sweep <sweep_id>"
        _emit({"error": msg})
        raise typer.Exit(code=2)
    store = JobStore(repo_root() / "runs")
    missing = [j for j in ids if not store.manifest_path(j).exists()]
    if missing:  # fail-loud (FR-F3), not a raw traceback
        _emit({"error": f"unknown job id(s): {missing}"})
        raise typer.Exit(code=2)
    manifests = _lab_for(ids[0]).wait(ids, interval=interval, timeout=timeout)
    all_terminal = all(m.status in _TERMINAL for m in manifests)
    teardown_leaks = [m.job_id for m in manifests if m.teardown_status == "failed"]
    summary = {
        "all_terminal": all_terminal,
        "teardown_leaks": teardown_leaks,  # FR-C2 — non-empty == a paid rental may still be running
        "jobs": [
            {
                "job_id": m.job_id,
                "state": m.status.value,
                "exit_code": m.exit_code,
                "teardown_status": m.teardown_status,
            }
            for m in manifests
        ],
    }
    _emit(summary)
    if done_file is not None:
        done_file.write_text(json.dumps(summary, indent=2, default=str))
    if not all_terminal:
        raise typer.Exit(code=1)
    if teardown_leaks:
        raise typer.Exit(code=3)  # all terminal but at least one cluster may still be billing


@app.command()
def dashboard(
    sweep: str | None = typer.Option(None, "--sweep", help="only jobs in this sweep_id"),
    interval: float = typer.Option(2.0, help="refresh seconds"),
) -> None:
    """Live terminal dashboard of job status + cost + latest metrics (FR-D3). Ctrl-C to exit."""
    from lab.dashboard import run_dashboard

    lab = _lab()
    ids = lab.jobs_in_sweep(sweep) if sweep else None
    run_dashboard(lab, ids, interval=interval)


@app.command()
def reconcile(
    apply: bool = typer.Option(
        False, "--apply", help="destroy orphaned rentals (default: dry-run report only)"
    ),
) -> None:
    """Cross-check Vast.ai rentals against local jobs to find leaks (FR-C2).

    Dry-run by default. ``--apply`` destroys every Vast.ai rental whose label matches the lab
    cluster pattern but has no live local job — use this to clean up after a teardown failure
    (look for ``teardown_status: "failed"`` in ``lab status``). Exits 3 if orphans are found in
    dry-run mode — re-run with --apply, or destroy by hand via ``vastai destroy_instance <id>``.
    """
    try:
        report = _lab(backend="skypilot").reconcile(apply=apply)
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=2) from e
    _emit(report)
    if report["orphans"] and not apply:
        raise typer.Exit(code=3)  # action required: re-run with --apply


def _repo() -> Path:
    import os

    env = os.environ.get("LAB_REPO_DIR")
    return Path(env) if env else repo_root()


@app.command()
def register(
    command: str = typer.Option(
        ..., "--command", "-c", help="entrypoint, e.g. 'uv run experiments/x.py'"
    ),
    expires: str = typer.Option(
        ...,
        "--expires",
        help="run-by deadline: +3d / +12h / ISO timestamp (required guardrail)",
    ),
    seed: int | None = typer.Option(None),
    cpus: int | None = typer.Option(None),
    memory: str | None = typer.Option(None),
    gpus: int | None = typer.Option(None),
    accelerators: str | None = typer.Option(
        None, "--gpu", "--accelerators", help="e.g. RTX_4090:1"
    ),
    timeout: str | None = typer.Option(
        None, help="wall-clock limit per job, e.g. 2h (cost bound, FR-I1)"
    ),
    window: str | None = typer.Option(
        None, "--window", help="daily launch window, e.g. 23:00-07:00"
    ),
    tz: str = typer.Option("UTC", "--tz", help="IANA timezone for --window"),
    not_before: str | None = typer.Option(
        None, "--not-before", help="absolute earliest start (ISO)"
    ),
    max_hourly: float | None = typer.Option(
        None, "--max-hourly", help="launch only if a matching Vast offer is at/below this $/h"
    ),
    offer_query: str | None = typer.Option(
        None, "--offer-query", help="extra vastai search filter"
    ),
    max_cost: float | None = typer.Option(None, "--max-cost", help="per-job worst-case $ cap"),
    after: list[str] = typer.Option(
        None, "--after", help="reg_id(s) that must succeed first (repeatable)"
    ),
    hold: bool = typer.Option(False, "--hold", help="register held; release with `lab queue release`"),
    spot: bool = typer.Option(False, "--spot", help="use spot/interruptible instances (skypilot)"),
    no_fallback: bool = typer.Option(
        False, "--no-fallback", "--spot-only",
        help="with --spot, do NOT fall back to on-demand if spot is scarce (wait/skip instead)",
    ),
) -> None:
    """Register a deferred job; the scheduler launches it when all triggers hold (spec §6)."""
    if accelerators and timeout is None:
        _emit({"error": "--timeout is required for GPU registrations (it is the cost bound)"})
        raise typer.Exit(code=1)
    queue = default_queue()
    try:
        expires_at = parse_expires(expires)
        win = parse_window(window, tz) if window else None
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    triggers = Triggers(
        not_before=(
            datetime.fromisoformat(not_before.replace("Z", "+00:00")) if not_before else None
        ),
        window=win,
        max_hourly_usd=max_hourly,
        offer_query=offer_query,
        after=list(after or []),
    )
    guardrails = Guardrails(expires_at=expires_at, max_cost_usd=max_cost)
    spec = JobSpec(
        command=command,
        seed=seed,
        resources=ResourceRequest(
            cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout,
            use_spot=spot, spot_fallback=not no_fallback,
        ),
        submitted_by="human",
    )
    try:
        reg = sched_register(_repo(), queue, spec, triggers, guardrails)
    except LabError as e:  # fail-loud, actionable (FR-F3)
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    if hold:
        queue.hold(reg.reg_id)
    _emit(
        {
            "reg_id": reg.reg_id,
            "state": "held" if hold else reg.state.value,
            "bundle_key": reg.bundle_key,
            "expires_at": reg.guardrails.expires_at,
            "worst_case_cost_usd": worst_case_cost(triggers, spec.resources),
        }
    )


@app.command(name="register-sweep")
def register_sweep(
    command: str = typer.Option(
        ..., "--command", "-c", help="entrypoint, e.g. 'uv run experiments/x.py'"
    ),
    grid: list[str] = typer.Option(..., "--grid", "-g", help="key=v1,v2,... (repeatable)"),
    expires: str = typer.Option(
        ..., "--expires", help="run-by deadline: +3d / +12h / ISO timestamp (required guardrail)"
    ),
    seed: int | None = typer.Option(None),
    cpus: int | None = typer.Option(None),
    memory: str | None = typer.Option(None),
    gpus: int | None = typer.Option(None),
    accelerators: str | None = typer.Option(
        None, "--gpu", "--accelerators", help="e.g. RTX_4090:1"
    ),
    timeout: str | None = typer.Option(
        None, help="wall-clock limit per job, e.g. 2h (cost bound, FR-I1)"
    ),
    with_pkg: list[str] = typer.Option(
        None, "--with", help="extra runtime package(s) per job (repeatable; uv run --with)"
    ),
    window: str | None = typer.Option(
        None, "--window", help="daily launch window, e.g. 23:00-07:00"
    ),
    tz: str = typer.Option("UTC", "--tz", help="IANA timezone for --window"),
    not_before: str | None = typer.Option(
        None, "--not-before", help="absolute earliest start (ISO)"
    ),
    max_hourly: float | None = typer.Option(
        None, "--max-hourly", help="launch only if a matching Vast offer is at/below this $/h"
    ),
    offer_query: str | None = typer.Option(
        None, "--offer-query", help="extra vastai search filter"
    ),
    max_cost: float | None = typer.Option(None, "--max-cost", help="per-point worst-case $ cap"),
    sweep_max_cost: float | None = typer.Option(
        None, "--sweep-max-cost",
        help="cap total sweep spend in USD; refused if worst case exceeds the daily budget",
    ),
    spot: bool = typer.Option(False, "--spot", help="use spot/interruptible instances (skypilot)"),
    no_fallback: bool = typer.Option(
        False, "--no-fallback", "--spot-only",
        help="with --spot, do NOT fall back to on-demand if spot is scarce (wait/skip instead)",
    ),
) -> None:
    """Register a grid as N deferred points sharing one sweep_id + ceiling; the scheduler paces them."""
    if accelerators and timeout is None:
        _emit({"error": "--timeout is required for GPU registrations (it is the cost bound)"})
        raise typer.Exit(code=1)
    queue = default_queue()
    try:
        expires_at = parse_expires(expires)
        win = parse_window(window, tz) if window else None
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    triggers = Triggers(
        not_before=(
            datetime.fromisoformat(not_before.replace("Z", "+00:00")) if not_before else None
        ),
        window=win,
        max_hourly_usd=max_hourly,
        offer_query=offer_query,
    )
    guardrails = Guardrails(expires_at=expires_at, max_cost_usd=max_cost)
    resources = ResourceRequest(
        cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout,
        use_spot=spot, spot_fallback=not no_fallback,
    )
    try:
        sweep_id, regs = sched_register_sweep(
            _repo(), queue, wrap_with_extras(command, with_pkg), _parse_grid(grid),
            resources=resources, triggers=triggers, guardrails=guardrails, seed=seed,
            sweep_max_cost=sweep_max_cost,
            daily_budget=queue.read_control().budget_usd_per_day,
            submitted_by="human",
        )
    except LabError as e:  # fail-loud, actionable (FR-F3)
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit({"sweep_id": sweep_id, "count": len(regs), "reg_ids": [r.reg_id for r in regs]})


queue_app = typer.Typer(help="Manage deferred registrations (spec §6).", no_args_is_help=True)
app.add_typer(queue_app, name="queue")


def _heartbeat_age_s(queue: QueueStore) -> float | None:
    hb = queue.read_heartbeat()
    if not hb or "at" not in hb:
        return None
    at = datetime.fromisoformat(str(hb["at"]))
    return max(0.0, (now() - at).total_seconds())


def _require_entry(queue: QueueStore, reg_id: str) -> None:
    try:
        queue.get_entry(reg_id)
    except FileNotFoundError:
        _emit({"error": f"unknown registration {reg_id!r}"})
        raise typer.Exit(code=2) from None


@queue_app.command(name="list")
def queue_list() -> None:
    """Entries + state + skip reason, plus scheduler heartbeat age (spec §6)."""
    queue = default_queue()
    entries = queue.list_entries()
    _emit(
        {
            "heartbeat_age_s": _heartbeat_age_s(queue),
            "control": queue.read_control().model_dump(),
            "entries": [
                {
                    "reg_id": r.reg_id,
                    "state": "held"
                    if (r.state is RegState.pending and queue.held(r.reg_id))
                    else r.state.value,
                    "cancel_requested": queue.cancel_requested(r.reg_id),
                    "job_id": r.job_id,
                    "last_skip_reason": r.last_skip_reason,
                    "expires_at": r.guardrails.expires_at,
                }
                for r in entries
            ],
        }
    )


@queue_app.command(name="show")
def queue_show(reg_id: str) -> None:
    """Full registration record."""
    queue = default_queue()
    _require_entry(queue, reg_id)
    _emit(json.loads(queue.get_entry(reg_id).model_dump_json()))


@queue_app.command(name="cancel")
def queue_cancel(reg_id: str) -> None:
    """Write the cancel marker; the scheduler applies it on its next tick (spec §5)."""
    queue = default_queue()
    _require_entry(queue, reg_id)
    queue.request_cancel(reg_id)
    _emit({"reg_id": reg_id, "cancel_requested": True})


@queue_app.command(name="gc")
def queue_gc(
    apply: bool = typer.Option(
        False, "--apply", help="actually delete orphaned bundles (default: dry-run report)"
    ),
) -> None:
    """Delete code bundles no live registration references (dry-run unless --apply).

    A shared sweep bundle is kept until all of its points are terminal.
    """
    from lab.scheduler.gc import gc_bundles

    _emit(gc_bundles(default_queue(), apply=apply))


@queue_app.command(name="hold")
def queue_hold(reg_id: str) -> None:
    """Hold a pending entry (skipped until released)."""
    queue = default_queue()
    _require_entry(queue, reg_id)
    queue.hold(reg_id)
    _emit({"reg_id": reg_id, "held": True})


@queue_app.command(name="release")
def queue_release(reg_id: str) -> None:
    """Release a held entry."""
    default_queue().release(reg_id)
    _emit({"reg_id": reg_id, "held": False})


@queue_app.command(name="pause")
def queue_pause() -> None:
    """Globally stop the scheduler from launching (heartbeat keeps beating)."""
    queue = default_queue()
    queue.write_control(queue.read_control().model_copy(update={"paused": True}))
    _emit({"paused": True})


@queue_app.command(name="resume")
def queue_resume() -> None:
    queue = default_queue()
    queue.write_control(queue.read_control().model_copy(update={"paused": False}))
    _emit({"paused": False})


@queue_app.command(name="budget")
def queue_budget(
    per_day: float | None = typer.Option(
        None, "--per-day", min=0.0, help="trailing-24h estimated-spend cap, USD (>= 0)"
    ),
    clear_budget: bool = typer.Option(
        False, "--clear-budget", help="remove the daily cap (budget_usd_per_day -> null)"
    ),
    max_concurrent: int | None = typer.Option(
        None, "--max-concurrent", min=1, help="max scheduler-launched jobs running at once (>= 1)"
    ),
    auto_reconcile: bool | None = typer.Option(
        None, "--auto-reconcile/--no-auto-reconcile"
    ),
) -> None:
    """Edit control.json guardrails."""
    if clear_budget and per_day is not None:
        raise typer.BadParameter("--clear-budget conflicts with --per-day")
    queue = default_queue()
    control = queue.read_control()
    updates: dict[str, object] = {}
    if clear_budget:
        updates["budget_usd_per_day"] = None
    if per_day is not None:
        updates["budget_usd_per_day"] = per_day
    if max_concurrent is not None:
        updates["max_concurrent"] = max_concurrent
    if auto_reconcile is not None:
        updates["auto_reconcile"] = auto_reconcile
    control = control.model_copy(update=updates)
    queue.write_control(control)
    _emit(control.model_dump())


scheduler_app = typer.Typer(help="Scheduler host commands (spec §4).", no_args_is_help=True)
app.add_typer(scheduler_app, name="scheduler")


@scheduler_app.command(name="tick")
def scheduler_tick(
    backend: str = typer.Option(
        "local", "--backend", help="local | skypilot (droplet uses skypilot)"
    ),
) -> None:
    """One idempotent scheduling pass — what the systemd timer runs every ~60s."""
    price_feed: PriceFeed | None = None
    if backend == "skypilot":
        from lab.scheduler.price import VastPriceFeed

        price_feed = VastPriceFeed()
    sched = Scheduler(
        default_queue(), home=_repo() / "runs", backend=backend, price_feed=price_feed
    )
    _emit(json.loads(sched.tick().model_dump_json()))


if __name__ == "__main__":
    app()
