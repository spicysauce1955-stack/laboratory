"""Human-facing CLI — a thin mirror of the MCP tools (FR-F2). Entry point: ``lab``.

Wired to the local backend by default; structured JSON output mirrors the MCP §9 returns.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from lab._util import atomic_write_text, now, parse_duration, wrap_with_extras
from lab.core import (
    Lab,
    LabError,
    default_lab,
    default_disk_gb,
    job_status_view,
    orphan_key,
    resolve_backend_profile,
    validate_cloud,
)
from lab.env import load_lab_env
from lab.manifest import git_work_tree, repo_root
from lab.models import JobSpec, ResourceRequest
from lab.scheduler.models import Guardrails, RegState, Triggers
from lab.scheduler.price import PriceFeed
from lab.scheduler.queue import QueueStore, default_queue
from lab.scheduler.register import parse_expires, parse_window
from lab.scheduler.register import register as sched_register
from lab.scheduler.register import register_sweep as sched_register_sweep
from lab.scheduler.register import worst_case_cost
from lab.scheduler.tick import Scheduler
from lab.store import JobStore

app = typer.Typer(
    help="Laboratory — remote experiment runner (CLI mirror of the MCP tools, spec §9).",
    no_args_is_help=True,
)


@app.callback()
def _load_env() -> None:
    """Apply the git-ignored ``.env`` before any command (cloud creds/project; real env wins).

    ``repo_root()`` honours ``LAB_REPO_DIR``, which matters most on the scheduler host: its
    systemd unit's ``WorkingDirectory`` need not be the repo, and it is the host that most needs a
    service-account key. A cwd-derived lookup found no ``.env`` there and the failure surfaced one
    layer down as an opaque auth error (GCP-CREDS-2).
    """
    load_lab_env(repo_root())
    _warn_if_repo_override_shadows_cwd()


def _warn_if_repo_override_shadows_cwd() -> None:
    """Warn when ``LAB_REPO_DIR`` points somewhere other than the work tree you are standing in.

    The override governs ``Lab.repo`` — the tree whose commit is pinned into the manifest and
    whose contents are rsynced as the SkyPilot workdir. That is right on the scheduler host, whose
    cwd is not a repo at all. But a laptop that has it set in ``.env`` and then runs `lab submit`
    from a second checkout or a git worktree would silently launch the *other* tree's code and
    record a commit that never contained the change — FR-B1 provenance wrong with no error
    anywhere. Only fires when cwd really is inside a different work tree, so the scheduler's
    intended use stays quiet.
    """
    override = (os.environ.get("LAB_REPO_DIR") or "").strip()
    if not override:
        return
    here = git_work_tree(Path.cwd())
    if here is not None and here != repo_root():  # cwd IS a work tree, and not the one we'd use
        typer.echo(
            f"warning: LAB_REPO_DIR={override} overrides the repo used for provenance and the "
            f"workdir upload, but you are inside {here}. Jobs will pin and upload "
            f"{repo_root()}, not the tree you are standing in.",
            err=True,
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
    backend: str = typer.Option("local", "--backend", help="local | skypilot | cpu"),
    cloud: str | None = typer.Option(None, "--cloud", help="vast | do | gcp (default vast; --backend cpu defaults to do)"),
    cache: bool = typer.Option(False, "--cache", help="reuse a prior succeeded identical job (FR-B5)"),
    seed: int | None = typer.Option(None, help="explicit seed (recorded in the manifest)"),
    code_ref: str = typer.Option("HEAD", help="git ref to pin"),
    cpus: int | None = typer.Option(None),
    memory: str | None = typer.Option(None, help="e.g. 8 or 8+ (GB)"),
    gpus: int | None = typer.Option(None),
    disk_size: int | None = typer.Option(None, "--disk-size", help="boot/attached volume size in GB (skypilot; DO volume size). cpu backend defaults to 50"),
    accelerators: str | None = typer.Option(None, "--accelerators", help="e.g. RTX_3070:1 (required for Vast)"),
    timeout: str | None = typer.Option(
        None, help="hard wall-clock cap, e.g. 2h / 30m / 45s — on overrun the job is killed, the "
        "machine torn down, and the run marked timed_out (FR-I1)"
    ),
    provision_timeout: str | None = typer.Option(None, "--provision-timeout", help="abort if the host doesn't reach UP in time, e.g. 10m (skypilot; default per-cloud: vast 8m, do 12m, gcp 20m)"),
    region: str | None = typer.Option(None, "--region", help="pin the cloud region, e.g. europe-west1 (skypilot; default: the optimizer picks)"),
    zone: str | None = typer.Option(None, "--zone", help="pin the zone, e.g. europe-west1-b (skypilot)"),
    price_cap: float | None = typer.Option(None, "--price-cap", help="refuse any instance above this compute $/hr — enforced by SkyPilot's optimizer, so the worst case is a ceiling not an estimate"),
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
    no_preflight: bool = typer.Option(
        False, "--no-preflight",
        help="skip the pre-launch credential/quota checks (see `lab doctor`)",
    ),
    allow_unknown_config: bool = typer.Option(
        False, "--allow-unknown-config",
        help="don't fail the job when the entrypoint reports it ignored some config keys",
    ),
) -> None:
    """Submit a job without blocking; prints {job_id, cached, status} (FR-A1).

    Provenance is fail-closed (FR-B1): the manifest always pins a real commit, and a dirty tree is
    snapshotted into a reproducible diff (pass --no-dirty to refuse instead). On --timeout overrun
    the job is killed, the machine torn down, and the run marked timed_out with the wall in its
    end_reason.
    """
    resources = ResourceRequest(
        cpus=cpus, memory=memory, gpus=gpus, disk_size=disk_size, accelerators=accelerators,
        cloud=cloud, region=region, zone=zone, max_hourly_usd=price_cap,
        timeout=timeout, provision_timeout=provision_timeout, use_spot=spot,
        spot_fallback=not no_fallback,
    )
    try:
        provisioner, resources = resolve_backend_profile(backend, resources)
    except LabError as e:  # e.g. --backend cpu with --accelerators, unknown --cloud (FR-F3)
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    lab = _lab(provisioner)
    spec = JobSpec(
        code_ref=code_ref,
        command=wrap_with_extras(command, with_pkg),
        seed=seed,
        resources=resources,
        submitted_by="human",
        allow_unknown_config=allow_unknown_config,
    )
    if cache and (cached_id := lab.find_cached(spec)) is not None:
        _emit({"job_id": cached_id, "cached": True, "status": lab.status(cached_id).value})
        return
    try:
        job_id = lab.submit(spec, allow_dirty=not no_dirty, preflight=not no_preflight)
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
    grid: list[str] = typer.Option([], "--grid", "-g", help="key=v1,v2,... (repeatable; optional when --seeds is given)"),
    backend: str = typer.Option("local", "--backend", help="local | skypilot | cpu"),
    cloud: str | None = typer.Option(None, "--cloud", help="vast | do | gcp (default vast; --backend cpu defaults to do)"),
    seed: int | None = typer.Option(None),
    cpus: int | None = typer.Option(None),
    memory: str | None = typer.Option(None),
    gpus: int | None = typer.Option(None),
    disk_size: int | None = typer.Option(None, "--disk-size", help="boot/attached volume size in GB per job (skypilot; DO volume size). cpu backend defaults to 50"),
    accelerators: str | None = typer.Option(None, "--accelerators"),
    timeout: str | None = typer.Option(None, help="wall-clock per job, e.g. 2h"),
    provision_timeout: str | None = typer.Option(None, "--provision-timeout", help="abort a host that doesn't reach UP in time, e.g. 10m (skypilot; default per-cloud: vast 8m, do 12m, gcp 20m)"),
    region: str | None = typer.Option(None, "--region", help="pin the cloud region for every job, e.g. europe-west1 (skypilot)"),
    zone: str | None = typer.Option(None, "--zone", help="pin the zone for every job, e.g. europe-west1-b (skypilot)"),
    price_cap: float | None = typer.Option(None, "--price-cap", help="refuse any instance above this compute $/hr, per job (enforced by SkyPilot's optimizer)"),
    with_pkg: list[str] = typer.Option(None, "--with", help="extra runtime package(s) per job (repeatable; layered via uv run --with)"),
    spot: bool = typer.Option(False, "--spot", help="use spot/interruptible instances (skypilot)"),
    no_fallback: bool = typer.Option(
        False, "--no-fallback", "--spot-only",
        help="with --spot, do NOT fall back to on-demand if spot is scarce (wait/skip instead)",
    ),
    sweep_max_cost: float | None = typer.Option(None, "--sweep-max-cost", help="up-front admission cap in USD: refuse the sweep if its total would exceed your daily budget (cost-safety); during-run enforcement is on register-sweep"),
    seeds: str | None = typer.Option(None, "--seeds", help="seed set as a range '0-31' or comma list '0,1,2'; declares seeds as a sharded axis (P1-2)"),
    shard_size: int | None = typer.Option(None, "--shard-size", help="max seeds per sub-job; each cell's seeds are split into shards of this size"),
    results_file: str = typer.Option("results.csv", "--results-file", help="per-run row-structured result file to aggregate per cell"),
    seed_column: str = typer.Option("seed", "--seed-column", help="column in --results-file identifying each row's seed"),
    row_key: str | None = typer.Option(
        None, "--row-key",
        help="comma-separated columns identifying a result row, e.g. 'seed,alpha' when an "
        "inner-loop axis writes multiple rows per seed (default: the seed column alone)",
    ),
    allow_unknown_config: bool = typer.Option(
        False, "--allow-unknown-config",
        help="don't fail jobs when the entrypoint reports it ignored some config keys",
    ),
) -> None:
    """Submit a parameter-grid sweep: one job per point under a sweep_id (FR-A5). A seeds-only
    sweep (no --grid) is one cell sharded over --seeds."""
    resources = ResourceRequest(
        cpus=cpus, memory=memory, gpus=gpus, disk_size=disk_size, accelerators=accelerators,
        cloud=cloud, region=region, zone=zone, max_hourly_usd=price_cap,
        timeout=timeout, provision_timeout=provision_timeout, use_spot=spot,
        spot_fallback=not no_fallback,
    )
    try:
        provisioner, resources = resolve_backend_profile(backend, resources)
    except LabError as e:  # e.g. --backend cpu with --accelerators (FR-F3)
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    lab = _lab(provisioner)
    try:
        sweep_id, job_ids = lab.sweep(
            wrap_with_extras(command, with_pkg),
            _parse_grid(grid),
            seed=seed,
            resources=resources,
            sweep_max_cost=sweep_max_cost,
            # only consult the control budget when there's a cap to admit against (avoids an
            # unnecessary queue read on every plain sweep)
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
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    if lab.store.has_sweep_plan(sweep_id):
        plan = lab.sweep_plan(sweep_id)
        _emit(plan.view())
    else:
        _emit({"sweep_id": sweep_id, "count": len(job_ids), "job_ids": job_ids})


@app.command()
def export(
    target_id: str = typer.Argument(..., help="a job id or a sweep-... id"),
    to: Path = typer.Option(..., "--to", help="destination directory for the bundle"),
    logs: bool = typer.Option(False, "--logs", help="also include per-job logs.txt (redacted)"),
) -> None:
    """Export a committable provenance bundle: manifests + result tables + resolved config +
    code diffs (+ sweep plan/aggregates), with an index.json tying files to commits and spend.

    The part of git-ignored runs/ that belongs in version control next to the paper — excluded
    blobs are listed in the index under `skipped`, never silently dropped (field-report #5).
    """
    try:
        _emit(_lab().export(target_id, to, include_logs=logs))
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e


@app.command()
def lint(
    command: str = typer.Option(..., "--command", "-c", help="entrypoint, e.g. 'python experiments/x.py'"),
    grid: list[str] = typer.Option([], "--grid", "-g", help="key=v1,v2,... (repeatable)"),
    key: list[str] = typer.Option([], "--key", help="extra override key(s) to check (repeatable)"),
) -> None:
    """Pre-submit check: warn about override keys the entrypoint source never references.

    A heuristic grep for legacy entrypoints that don't write effective_config.json — catches the
    silent-dropped-override trap (field-report #1) before money is spent. Exits 1 on findings.
    """
    from lab.experiment import unreferenced_keys

    script = next((tok for tok in command.split() if tok.endswith(".py")), None)
    if script is None or not Path(script).exists():
        _emit({"error": f"could not find a .py entrypoint in {command!r} to lint"})
        raise typer.Exit(code=2)
    keys = list(_parse_grid(grid)) + list(key)
    missing = unreferenced_keys(Path(script).read_text(), keys)
    _emit({"script": script, "checked_keys": sorted(keys), "missing_keys": missing})
    if missing:
        raise typer.Exit(code=1)


@app.command()
def status(job_id: str) -> None:
    """Show a job's state + cost + teardown_status (FR-A2, FR-I2, FR-C2); scheduler-launched
    jobs fall back to the mirrored manifest (spec §4.3). Same shape as the MCP status tool."""
    try:
        _emit(job_status_view(repo_root() / "runs", repo_root(), job_id))
    except FileNotFoundError:
        _emit({"error": f"unknown job id {job_id!r}"})
        raise typer.Exit(code=2) from None


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


@app.command(name="sweep-aggregate")
def sweep_aggregate(
    sweep_id: str,
    strict: bool = typer.Option(
        False, "--strict",
        help="only aggregate succeeded shards (exclude recovered rows from timed-out/failed "
        "shards; those rows carry a _shard_status column and show up in seeds_partial)",
    ),
    row_key: str | None = typer.Option(
        None, "--row-key",
        help="declare the columns identifying a result row (e.g. 'seed,alpha') for sweeps "
        "created before --row-key existed; persisted onto the plan for future aggregates",
    ),
) -> None:
    """Row-concatenate each cell's shard results into one per-cell result (P1-2). By default
    partial rows from non-succeeded shards are included (stamped + listed in seeds_partial)."""
    try:
        plan = _lab().aggregate_sweep(sweep_id, include_partial=not strict, row_key=row_key)
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit(plan.view())


@app.command(name="sweep-retry")
def sweep_retry(sweep_id: str) -> None:
    """Resubmit only the missing shards of incomplete cells, then re-aggregate (P1-2)."""
    try:
        plan = _lab().retry_sweep(sweep_id)
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit(plan.view())


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
    timeout: str | None = typer.Option(
        None, help="give up after this long, e.g. 600 / '10m' / '2h' (bare numbers = seconds)"
    ),
    done_file: Path | None = typer.Option(
        None, "--done-file",
        help="summary JSON a hook can watch — atomically rewritten after each job finishes "
        "(carries `pending`), and finally with the complete verdict",
    ),
    fail_fast: bool = typer.Option(
        False, "--fail-fast",
        help="exit 4 as soon as any job is failed/timed_out (preempted/cancelled don't trip it); "
        "surviving jobs are NOT cancelled — `pending` in the summary names them",
    ),
) -> None:
    """Block until the job(s) reach a terminal state, then exit (FR-G1).

    Run as a Claude Code background task — its completion is the push signal the session acts on,
    so the agent need not poll. Exit codes: 0 clean; 1 gave up on --timeout; 2 bad args;
    3 teardown leaked (a paid machine may still bill); 4 --fail-fast tripped.
    """
    try:
        timeout_s = parse_duration(timeout)
    except ValueError as e:
        raise typer.BadParameter(f"bad --timeout {timeout!r}: {e}") from e
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
    the_lab = _lab_for(ids[0])
    on_update = (
        (lambda s: atomic_write_text(done_file, json.dumps(s, indent=2, default=str)))
        if done_file is not None
        else None
    )
    summary = the_lab.wait_summary(
        ids, interval=interval, timeout=timeout_s, fail_fast=fail_fast, on_update=on_update
    )
    all_terminal = summary["all_terminal"]
    teardown_leaks = summary["teardown_leaks"]
    teardown_unconfirmed = summary["teardown_unconfirmed"]
    _emit(summary)
    if teardown_unconfirmed and not teardown_leaks:
        typer.echo(
            f"[lab] warning: teardown not confirmed for {teardown_unconfirmed} "
            "(status is null, not 'failed') — run `lab reconcile` to be sure no machine or "
            "block volume is still billing.",
            err=True,
        )
    if summary["failed_fast"]:
        # A confirmed leak outranks the fail-fast signal: exit 3 is the documented URGENT
        # "a paid machine may still be billing — run `lab reconcile` now" alarm (FR-C2).
        raise typer.Exit(code=3 if teardown_leaks else 4)
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


# Every orphan list `reconcile` can report. The dry-run leak alarm reads the UNION: each pass
# covers a blind spot of the others (the GCP compute pass exists precisely for clusters SkyPilot's
# registry has lost, where `sky_orphans` is empty by definition), so reading a subset silences the
# alarm exactly where the missing pass was needed. Add a pass -> add it here (FR-C2).
_ORPHAN_FIELDS = (
    "orphans",  # Vast-direct rentals
    "sky_orphans",  # SkyPilot-tracked clusters (DO/GCP/Vast)
    "gcp_orphans",  # GCE instances via the compute API
    "gcp_disk_orphans",  # unattached GCE persistent disks
    "do_volume_orphans",  # detached DO block volumes
)
# Deliberately NOT here: `gcp_unmatched` — `lab-*` GCE names that do not match our cluster-node
# shape. It is advisory (something in this project is named like us but is not ours to destroy),
# so it must not exit 3 and send the reader to `--apply` (GCP-LEAK-7).


def _describe_orphan(item: Any) -> str:
    """One line naming a single doomed resource, however each pass shapes its entries.

    Vast emits ``{id, label}`` — the label is the only human-readable thing it has, and it is what
    lets an operator recognise which job a rental belonged to, so it must not be dropped.
    """
    if not isinstance(item, dict):
        return str(item)
    name = item.get("name") or item.get("label") or item.get("id")
    extra = [str(v) for v in (item.get("zone") or item.get("region"), item.get("status")) if v]
    if item.get("label") and item.get("id") is not None:
        extra.insert(0, f"id={item['id']}")
    return f"{name}{f'  ({", ".join(extra)})' if extra else ''}"


def _doomed(report: dict[str, Any]) -> list[tuple[str, Any]]:
    """Every ``(pass, resource)`` `--apply` would destroy. Empty = nothing to confirm."""
    return [(field, item) for field in _ORPHAN_FIELDS for item in report.get(field) or []]


def _lines(doomed: list[tuple[str, Any]]) -> list[str]:
    return [f"{field}: {_describe_orphan(item)}" for field, item in doomed]


def _stdin_is_a_tty() -> bool:
    """Whether there is a human to prompt. Test seam — a CliRunner swaps ``sys.stdin`` during
    invoke, so patching the stream's own ``isatty`` from a test cannot reach it."""
    return sys.stdin.isatty()


@app.command()
def reconcile(
    apply: bool = typer.Option(
        False, "--apply", help="destroy orphaned rentals (default: dry-run report only)"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="skip the confirmation prompt (for unattended use)"
    ),
) -> None:
    """Cross-check cloud instances against local jobs to find leaks (FR-C2).

    Dry-run by default. ``--apply`` destroys every Vast.ai rental and SkyPilot-tracked ``lab-*``
    cluster (the ``sky.status`` pass covers DO/GCP) with no live local job — use this to clean up
    after a teardown failure (look for ``teardown_status: "failed"`` in ``lab status``). The
    Vast-direct pass is skipped when vastai-sdk isn't installed. Exits 3 if orphans are found in
    dry-run mode — re-run with --apply, or destroy by hand via ``vastai destroy_instance <id>``.

    ``--apply`` lists what it is about to destroy and asks first; ``--yes`` skips the prompt for
    unattended use. The narrowed GCP predicate makes a destructive false positive unlikely, not
    *recoverable* — the prompt is what keeps a bug in it from costing someone a VM they cared
    about. Declining exits 4 and destroys nothing.

    The GCP passes only claim resources matching SkyPilot's real node shape
    (``lab-…-<head|worker>-<uuid8>-<compute|tpu|mig>``), and report the project they swept as
    ``gcp_project`` — check it matches the project SkyPilot launches into. Anything else named
    ``lab-*`` is listed under ``gcp_unmatched`` and never destroyed.
    """
    lab = _lab(backend="skypilot")
    approved: set[str] | None = None
    try:
        if apply and not yes:
            # Price the blast radius, then make the approval *binding*: the confirmed pass is a
            # second, independent sweep, so a resource that becomes an orphan in between (a job
            # whose supervisor dies, dropping its live cluster out of running_clusters) would
            # otherwise be destroyed without ever having been shown. Diagnostics go to stderr —
            # stdout carries only JSON, which callers parse (§9).
            dry = lab.reconcile(apply=False)
            doomed = _doomed(dry)
            if doomed:
                if not _stdin_is_a_tty():
                    # Nobody to ask. Fail closed and say exactly how to proceed — a bare
                    # click.Abort here would exit 1 with no JSON at all.
                    typer.echo(
                        f"refusing to destroy {len(doomed)} resource(s) with no terminal to "
                        "confirm at — re-run with --yes to proceed unattended.",
                        err=True,
                    )
                    _emit({"aborted": True, "reason": "no tty", "would_destroy": _lines(doomed)})
                    raise typer.Exit(code=4)
                project = dry.get("gcp_project")
                typer.echo(
                    f"about to destroy {len(doomed)} resource(s)"
                    f"{f' in project {project}' if project else ''}:",
                    err=True,
                )
                for line in _lines(doomed):
                    typer.echo(f"  {line}", err=True)
                if not typer.confirm("proceed?", err=True):
                    _emit({"aborted": True, "would_destroy": _lines(doomed)})
                    raise typer.Exit(code=4)
                approved = {orphan_key(field, item) for field, item in doomed}
        report = lab.reconcile(apply=apply, only=approved)
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=2) from e
    _emit(report)
    if unmatched := report.get("gcp_unmatched"):
        # Not an orphan field (see above) — it is not `--apply`-able, so exit 3's "re-run with
        # --apply" would be wrong advice. But it must not be silent either: if these are in fact
        # our own nodes under a name shape we stopped recognising, the leak passes have gone
        # blind and would report clean while the VMs bill (GCP-LEAK-7).
        typer.echo(
            f"warning: {len(unmatched)} `lab-*` GCP resource(s) did not match our node shape and "
            "were NOT considered for cleanup — check whether they are ours:",
            err=True,
        )
        for name in unmatched:
            typer.echo(f"  {name}", err=True)
    if any(report.get(k) for k in _ORPHAN_FIELDS) and not apply:
        raise typer.Exit(code=3)  # action required: re-run with --apply


@app.command()
def doctor(
    cloud: str | None = typer.Option(None, "--cloud", help="vast | do | gcp (default vast)"),
    accelerators: str | None = typer.Option(
        None, "--gpu", "--accelerators", help="check quota for this accelerator, e.g. T4:1"
    ),
    cpus: int | None = typer.Option(None, help="check vCPU quota for this size"),
    disk_size: int | None = typer.Option(None, "--disk-size", help="check disk quota for this size"),
    region: str | None = typer.Option(None, "--region", help="check quota in this region"),
    zone: str | None = typer.Option(None, "--zone", help="check quota in this zone's region"),
    spot: bool = typer.Option(False, "--spot", help="price the spec as spot"),
    as_json: bool = typer.Option(False, "--json", help="emit the structured findings"),
    no_cache: bool = typer.Option(False, "--no-cache", help="re-run every check, ignoring the cache"),
) -> None:
    """Check whether a launch on this cloud would work, before it costs a provision.

    Verifies credentials (including SkyPilot's daemon, which does not inherit ``.env``), project
    and billing, enabled APIs, IAM permissions, and quota for the shape you ask about — then
    reports what the catalog says it will cost. Exits 1 if any check fails.

    Quota is checked at both levels GCP enforces: a fresh project can hold regional GPU quota and
    still be blocked by a global ``GPUS_ALL_REGIONS`` of 0.
    """
    from lab.doctor import doctor_view, format_report, run_checks

    resources = ResourceRequest(
        cpus=cpus, disk_size=disk_size, accelerators=accelerators,
        cloud=cloud, region=region, zone=zone, use_spot=spot,
    )
    try:
        validate_cloud(cloud)
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=2) from e
    # Apply the same disk defaults a real submit would, so the quota check asks about the size
    # that would actually be provisioned rather than the one the user happened to type.
    resources = resources.model_copy(update={"disk_size": default_disk_gb(resources)})
    results = run_checks(cloud, resources, home=repo_root() / "runs", use_cache=not no_cache)
    if as_json:
        _emit(doctor_view(cloud, results))
    else:
        typer.echo(f"lab doctor — {cloud or 'vast'}")
        typer.echo(format_report(results))
    if any(r.status == "fail" for r in results):
        raise typer.Exit(code=1)


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
    cloud: str | None = typer.Option(
        None, "--cloud", help="vast | do | gcp (default vast; price/offer triggers are Vast-only)"
    ),
    region: str | None = typer.Option(
        None, "--region", help="pin the cloud region, e.g. europe-west1 (skypilot)"
    ),
    zone: str | None = typer.Option(None, "--zone", help="pin the zone, e.g. europe-west1-b"),
    price_cap: float | None = typer.Option(
        None, "--price-cap",
        help="refuse any instance above this compute $/hr (enforced by SkyPilot's optimizer; "
             "unlike --max-hourly this is a ceiling, not a wait-until trigger)",
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
            cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, cloud=cloud,
            region=region, zone=zone, max_hourly_usd=price_cap,
            timeout=timeout, use_spot=spot, spot_fallback=not no_fallback,
        ),
        submitted_by="human",
    )
    try:
        reg = sched_register(repo_root(), queue, spec, triggers, guardrails)
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
    cloud: str | None = typer.Option(
        None, "--cloud", help="vast | do | gcp (default vast; price/offer triggers are Vast-only)"
    ),
    region: str | None = typer.Option(
        None, "--region", help="pin the cloud region, e.g. europe-west1 (skypilot)"
    ),
    zone: str | None = typer.Option(None, "--zone", help="pin the zone, e.g. europe-west1-b"),
    price_cap: float | None = typer.Option(
        None, "--price-cap",
        help="refuse any instance above this compute $/hr (enforced by SkyPilot's optimizer; "
             "unlike --max-hourly this is a ceiling, not a wait-until trigger)",
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
        cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, cloud=cloud,
            region=region, zone=zone, max_hourly_usd=price_cap,
        timeout=timeout, use_spot=spot, spot_fallback=not no_fallback,
    )
    try:
        sweep_id, regs = sched_register_sweep(
            repo_root(), queue, wrap_with_extras(command, with_pkg), _parse_grid(grid),
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
        default_queue(), home=repo_root() / "runs", backend=backend, price_feed=price_feed
    )
    _emit(json.loads(sched.tick().model_dump_json()))


if __name__ == "__main__":
    app()
