"""Human-facing CLI — a thin mirror of the MCP tools (FR-F2). Entry point: ``lab``.

Commands are stubs until the core + backends land (P0 build order). They exist so the
entry point and ``lab --help`` work, and to fix the command surface early.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    help="Laboratory — remote experiment runner (CLI mirror of the MCP tools, spec §9).",
    no_args_is_help=True,
)


def _todo(name: str) -> None:
    typer.echo(f"`lab {name}` is not implemented yet (see research/16-decisions.md build order).")
    raise typer.Exit(code=1)


@app.command()
def submit(
    code_ref: str = typer.Option("HEAD", help="git ref to pin"),
    command: str = typer.Option(..., help="entrypoint command, e.g. 'uv run python experiments/x.py'"),
    seed: int | None = typer.Option(None),
) -> None:
    """Submit a job (non-blocking) and print its job_id (FR-A1)."""
    _todo("submit")


@app.command()
def status(job_id: str) -> None:
    """Show a job's state (FR-A2)."""
    _todo("status")


@app.command()
def logs(job_id: str, tail: int = typer.Option(100)) -> None:
    """Tail a job's logs (FR-D1)."""
    _todo("logs")


@app.command()
def fetch(job_id: str) -> None:
    """Fetch artifacts into runs/<job_id>/ (FR-E2)."""
    _todo("fetch")


@app.command()
def cancel(job_id: str) -> None:
    """Cancel a job and tear down its machine (FR-A3, FR-C2)."""
    _todo("cancel")


@app.command(name="list")
def list_jobs() -> None:
    """List jobs (FR-H1)."""
    _todo("list")


if __name__ == "__main__":
    app()
