"""Human-facing CLI — a thin mirror of the MCP tools (FR-F2). Entry point: ``lab``.

Wired to the local backend by default; structured JSON output mirrors the MCP §9 returns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from lab.core import Lab, LabError, default_lab
from lab.manifest import repo_root
from lab.models import JobSpec, JobState, ResourceRequest
from lab.store import JobStore

_TERMINAL = {JobState.succeeded, JobState.failed, JobState.cancelled, JobState.timed_out}

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


def _emit(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


def _parse_grid(items: list[str]) -> dict[str, list]:
    """Parse repeated `--grid key=v1,v2,...` options into {key: [values]}.

    Values stay strings — the experiment (e.g. Hydra) coerces types, so the lab doesn't guess.
    """
    grid: dict[str, list] = {}
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
    timeout: str | None = typer.Option(None, help="wall-clock limit, e.g. 2h / 30m / 45s"),
) -> None:
    """Submit a job without blocking; prints {job_id, cached, status} (FR-A1)."""
    lab = _lab(backend)
    spec = JobSpec(
        code_ref=code_ref,
        command=command,
        seed=seed,
        resources=ResourceRequest(
            cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout
        ),
        submitted_by="human",
    )
    if cache and (cached_id := lab.find_cached(spec)) is not None:
        _emit({"job_id": cached_id, "cached": True, "status": lab.status(cached_id).value})
        return
    try:
        job_id = lab.submit(spec)
    except LabError as e:  # fail-loud, actionable (FR-F3)
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit({"job_id": job_id, "cached": False, "status": lab.status(job_id).value})


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
) -> None:
    """Submit a parameter-grid sweep: one job per point under a sweep_id (FR-A5)."""
    lab = _lab(backend)
    try:
        sweep_id, job_ids = lab.sweep(
            command,
            _parse_grid(grid),
            seed=seed,
            resources=ResourceRequest(
                cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout
            ),
        )
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit({"sweep_id": sweep_id, "count": len(job_ids), "job_ids": job_ids})


@app.command()
def status(job_id: str) -> None:
    """Show a job's state (FR-A2)."""
    _emit({"job_id": job_id, "state": _lab_for(job_id).status(job_id).value})


@app.command()
def logs(job_id: str, tail: int = typer.Option(100)) -> None:
    """Tail a job's logs (FR-D1)."""
    for line in _lab_for(job_id).logs(job_id, tail=tail):
        typer.echo(line)


@app.command()
def metrics(
    job_id: str,
    name: list[str] = typer.Option(None, "--name", "-n", help="filter to these metric names"),
    since_step: int | None = typer.Option(None, help="only points with step > since_step"),
) -> None:
    """Query a job's incremental metric series (FR-D2 — the early-kill loop)."""
    _emit({"series": _lab_for(job_id).metrics(job_id, names=name or None, since_step=since_step)})


@app.command()
def fetch(job_id: str) -> None:
    """Collect artifacts into runs/<job_id>/; prints local paths (FR-E2)."""
    arts = _lab_for(job_id).fetch_artifacts(job_id)
    _emit({"local_paths": [a.path for a in arts], "artifacts": [a.model_dump() for a in arts]})


@app.command()
def cancel(job_id: str) -> None:
    """Cancel a job and tear down its machine (FR-A3, FR-C2)."""
    _emit({"job_id": job_id, "state": _lab_for(job_id).cancel(job_id).value})


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
    summary = {
        "all_terminal": all_terminal,
        "jobs": [
            {"job_id": m.job_id, "state": m.status.value, "exit_code": m.exit_code}
            for m in manifests
        ],
    }
    _emit(summary)
    if done_file is not None:
        done_file.write_text(json.dumps(summary, indent=2, default=str))
    if not all_terminal:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
