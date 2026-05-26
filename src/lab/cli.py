"""Human-facing CLI — a thin mirror of the MCP tools (FR-F2). Entry point: ``lab``.

Wired to the local backend by default; structured JSON output mirrors the MCP §9 returns.
"""

from __future__ import annotations

import json
from typing import Any

import typer

from lab.core import Lab, LabError, default_lab
from lab.models import JobSpec, ResourceRequest

app = typer.Typer(
    help="Laboratory — remote experiment runner (CLI mirror of the MCP tools, spec §9).",
    no_args_is_help=True,
)


def _lab() -> Lab:
    return default_lab()


def _emit(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.command()
def submit(
    command: str = typer.Option(..., "--command", "-c", help="entrypoint, e.g. 'python experiments/x.py'"),
    seed: int | None = typer.Option(None, help="explicit seed (recorded in the manifest)"),
    code_ref: str = typer.Option("HEAD", help="git ref to pin"),
    cpus: int | None = typer.Option(None),
    timeout: str | None = typer.Option(None, help="wall-clock limit, e.g. 2h / 30m / 45s"),
) -> None:
    """Submit a job without blocking; prints {job_id, status} (FR-A1)."""
    lab = _lab()
    try:
        job_id = lab.submit(
            JobSpec(
                code_ref=code_ref,
                command=command,
                seed=seed,
                resources=ResourceRequest(cpus=cpus, timeout=timeout),
                submitted_by="human",
            )
        )
    except LabError as e:  # fail-loud, actionable (FR-F3)
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit({"job_id": job_id, "status": lab.status(job_id).value})


@app.command()
def status(job_id: str) -> None:
    """Show a job's state (FR-A2)."""
    _emit({"job_id": job_id, "state": _lab().status(job_id).value})


@app.command()
def logs(job_id: str, tail: int = typer.Option(100)) -> None:
    """Tail a job's logs (FR-D1)."""
    for line in _lab().logs(job_id, tail=tail):
        typer.echo(line)


@app.command()
def fetch(job_id: str) -> None:
    """Collect artifacts into runs/<job_id>/; prints local paths (FR-E2)."""
    arts = _lab().fetch_artifacts(job_id)
    _emit({"local_paths": [a.path for a in arts], "artifacts": [a.model_dump() for a in arts]})


@app.command()
def cancel(job_id: str) -> None:
    """Cancel a job and tear down its machine (FR-A3, FR-C2)."""
    _emit({"job_id": job_id, "state": _lab().cancel(job_id).value})


@app.command(name="list")
def list_jobs() -> None:
    """List jobs (FR-H1)."""
    jobs = _lab().list_jobs()
    _emit(
        {
            "jobs": [
                {"job_id": j.job_id, "status": j.status.value, "created_at": j.created_at}
                for j in jobs
            ]
        }
    )


if __name__ == "__main__":
    app()
