"""CLI surface added for packaged releases: --version, mcp, and init (Tasks 2-4)."""

from __future__ import annotations

from typer.testing import CliRunner

from lab import __version__
from lab.cli import app

runner = CliRunner()


def test_version_flag_prints_the_installed_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_mcp_command_exists() -> None:
    """The scaffolded .mcp.json shells `lab mcp`; it must not depend on a module path."""
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
