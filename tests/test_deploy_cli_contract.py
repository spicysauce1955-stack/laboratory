"""deploy.sh drives the real `lab` CLI, but its harness (tests/test_deploy_script.py) replaces
`uv run lab` with a simulation, so nothing there notices a lab change that breaks a command line
the deploy depends on. On 2026-11-19 exactly that happened: `parse_duration` began refusing 0,
and step 2c's `lab queue wait-drain --timeout 0s` race check would have aborted every real
deploy. This parses every literal `uv run lab ...` invocation in deploy.sh against the real
command tree, and runs the zero-timeout drain check for real against a local queue.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import click
import pytest
import typer
from typer.testing import CliRunner

from lab._util import parse_duration
from lab.cli import app

DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "scheduler" / "deploy.sh"
_TEXT = DEPLOY.read_text()
# `NAME="${LAB_DEPLOY_X:-30m}"` -> {"NAME": "30m"}: the values a variable has by default.
_DEFAULTS = dict(re.findall(r'^([A-Z_]+)="\$\{[A-Z_]+:-([^}]*)\}"', _TEXT, re.M))
_STOP = {";", "|", "||", "&", "&&", ")", "(", "\n"}


def _invocations() -> list[list[str]]:
    """Every `uv run lab ...` argv in deploy.sh, with variables replaced by their defaults (or a
    placeholder). Lines that are not shell (inside a multi-line echo) don't tokenize and are
    skipped; text inside a quoted echo is a single token and never matches."""
    found: list[list[str]] = []
    lines: list[str] = []
    for line in _TEXT.replace("\\\n", " ").splitlines():
        if line.lstrip().startswith("#") or "uv run lab " not in line:
            continue
        # `x="$(uv run lab queue list)"` is one quoted token to a tokenizer: scan the bodies of
        # command substitutions as lines of their own.
        lines += [line, *re.findall(r"\$\(([^()]*)\)", line)]
    for line in lines:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            continue  # unbalanced quotes: part of a multi-line message, not a command
        for i in range(len(tokens) - 2):
            if tokens[i : i + 3] != ["uv", "run", "lab"]:
                continue
            argv: list[str] = []
            for tok in tokens[i + 3 :]:
                if tok in _STOP or tok.startswith("#"):
                    break
                if tok.startswith("$"):
                    tok = _DEFAULTS.get(tok.strip("${}"), "placeholder")
                argv.append(tok)
            if argv not in found:
                found.append(argv)
    return found


def _parse(argv: list[str]) -> tuple[str, dict[str, Any]]:
    """Resolve `argv` through the real click tree without invoking anything."""
    cmd: click.Command = typer.main.get_command(app)
    ctx = cmd.make_context("lab", list(argv))
    path = []
    while isinstance(cmd, click.Group):
        name, sub, rest = cmd.resolve_command(ctx, [*ctx.protected_args, *ctx.args])
        assert sub is not None and name is not None
        path.append(name)
        cmd, ctx = sub, sub.make_context(name, rest, parent=ctx)
    return " ".join(path), dict(ctx.params)


INVOCATIONS = _invocations()


def test_the_scan_sees_the_commands_the_deploy_relies_on():
    seen = {" ".join(a[:2]) for a in INVOCATIONS}
    assert {"queue wait-drain", "queue pause", "queue resume", "queue list"} <= seen
    assert any(a[:1] == ["register"] for a in INVOCATIONS)


@pytest.mark.parametrize("argv", INVOCATIONS, ids=[" ".join(a) for a in INVOCATIONS])
def test_every_lab_command_line_in_deploy_sh_parses(argv: list[str]):
    command, params = _parse(argv)
    if command == "queue wait-drain":
        parse_duration(params["timeout"], allow_zero=True)  # what the command itself does
    if command == "register":
        parse_duration(params["timeout"])


def test_wait_drain_with_a_zero_timeout_checks_once_and_reports_drained(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    result = CliRunner().invoke(app, ["queue", "wait-drain", "--timeout", "0s"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"drained": True, "blocking": []}


def test_a_zero_duration_is_still_refused_as_a_job_limit():
    with pytest.raises(ValueError, match="positive"):
        parse_duration("0s")
    assert parse_duration("0s", allow_zero=True) == 0.0
