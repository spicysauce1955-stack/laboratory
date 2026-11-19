"""The CLI and the MCP server are two shells over one library (FR-F2); their arguments must agree.

They drifted silently on the deferred path: the MCP ``register``/``register_sweep`` tools never
gained ``price_cap``, ``price_cap_strict``, ``region``/``zone``, spot, the accelerator pool, ``gpus``
(nor ``hold`` / ``with_pkg``), so an agent could not price-cap an unattended deferred job — the
exact case where an unnoticed overrun bills longest — while ``lab register --price-cap`` could.

This test diffs every command present on both surfaces: parameter names, which are required, and
defaults. A difference fails CI unless it is listed below **with the reason it is intentional**.
Adding a CLI command or an MCP tool also fails until it is paired or listed as one-surface-only.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import typer
from fastmcp import Client

from lab import cli
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.mcp_server import build_server

# MCP tool -> the CLI command it mirrors (a space separates a group from its subcommand).
PAIRS: dict[str, str] = {
    "submit": "submit",
    "confirm": "confirm",
    "sweep": "sweep",
    "status": "status",
    "wait": "wait",
    "export": "export",
    "reconcile": "reconcile",
    "doctor": "doctor",
    "metrics": "metrics",
    "logs": "logs",
    "fetch_artifacts": "fetch",
    "cancel": "cancel",
    "sweep_status": "sweep-status",
    "sweep_aggregate": "sweep-aggregate",
    "sweep_retry": "sweep-retry",
    "list": "list",
    "ps": "ps",
    "register": "register",
    "register_sweep": "register-sweep",
    "queue_list": "queue list",
    "queue_show": "queue show",
    "queue_cancel": "queue cancel",
    "queue_pause": "queue pause",
    "history": "history",
    "report": "report",
}

# CLI commands with no MCP tool, and why.
CLI_ONLY_COMMANDS: dict[str, str] = {
    "mcp": "starts the MCP server itself",
    "init": "scaffolds a project once, from a human's shell; writes files an agent then reads",
    "dashboard": "interactive full-screen terminal UI",
    "lint": "pre-submit heuristic for humans editing entrypoints; agents get the same check "
    "for free, since a run that ignores an override fails (effective_config.json)",
    "note": "the human-to-next-run channel; agents write notes through the CLI (`--agent=`)",
    "notes": "reads/retires that channel; same reason as `note`",
    "queue resume": "the MCP `queue_pause` tool takes `paused`, so resume is `paused=False`",
    "queue hold": "operator control of the deferred queue; not exposed to agents",
    "queue release": "operator control of the deferred queue; not exposed to agents",
    "queue gc": "deletes code bundles from the shared queue store; operator-only",
    "queue budget": "the daily-spend guardrail is a human's cap on agents; an agent that could "
    "raise its own budget would defeat it",
    "queue wait-drain": "the redeploy safety gate run by deploy/scheduler/deploy.sh",
    "scheduler tick": "run by the always-on host's systemd timer, never by a client",
}

# One setting spelled differently on the two surfaces: (tool or "*", MCP name) -> (CLI name,
# inverted). Inverted means the CLI exposes the negation as an opt-out flag (`--no-dirty`) while
# the MCP argument is the positive switch (`allow_dirty=True`); their defaults must be opposite.
RENAMES: dict[tuple[str, str], tuple[str, bool]] = {
    ("*", "use_spot"): ("spot", False),  # MCP's long-standing name for --spot
    ("*", "spot_fallback"): ("no_fallback", True),
    ("submit", "allow_dirty"): ("no_dirty", True),
    ("submit", "preflight"): ("no_preflight", True),
    ("doctor", "use_cache"): ("no_cache", True),
    ("confirm", "wait"): ("no_wait", True),
    ("export", "job_or_sweep_id"): ("target_id", False),
    ("export", "include_logs"): ("logs", False),
    ("metrics", "names"): ("name", False),
}

# Parameters only one surface has, and why.
CLI_ONLY_PARAMS: dict[str, dict[str, str]] = {
    "wait": {
        "done_file": "a file a shell hook watches while `lab wait` runs in the background; an "
        "MCP caller receives the same summary as the tool's return value",
    },
    "reconcile": {
        "apply": "the MCP tool is deliberately read-only: destroying cloud resources needs a "
        "human at a terminal, or an explicit --yes typed by one",
        "yes": "see `apply`",
    },
    "report": {"out": "writes a file; the MCP tool returns the markdown in its payload"},
    "doctor": {"as_json": "presentation only: the MCP tool always returns the structured form"},
}
MCP_ONLY_PARAMS: dict[str, dict[str, str]] = {
    "queue_pause": {"paused": "one tool with a switch; the CLI spells it `queue pause`/`resume`"},
}

# (tool, MCP name) -> why the defaults differ.
DEFAULT_DIFFERENCES: dict[tuple[str, str], str] = {
    ("wait", "timeout"): "an MCP call blocks the agent's turn and cannot be backgrounded the way "
    "`lab wait` can, so the tool is bounded (600s) where the CLI waits until done",
}

_EMPTY: tuple[Any, ...] = (None, (), [])


def _cli_root() -> Any:
    return typer.main.get_command(cli.app)


def _cli_commands() -> dict[str, Any]:
    """Every leaf CLI command, keyed by its path (`queue list`)."""
    out: dict[str, Any] = {}
    for name, cmd in _cli_root().commands.items():
        sub = getattr(cmd, "commands", None)
        if sub:
            out.update({f"{name} {leaf}": c for leaf, c in sub.items()})
        else:
            out[name] = cmd
    return out


def _mcp_tools() -> dict[str, dict[str, Any]]:
    lab = Lab(backend=LocalBackend(home=Path("/nonexistent"), repo=Path(".")), repo=Path("."),
              home=Path("/nonexistent"))

    async def go() -> dict[str, dict[str, Any]]:
        async with Client(build_server(lab)) as c:
            return {t.name: t.inputSchema for t in await c.list_tools()}

    return asyncio.run(go())


@pytest.fixture(scope="module")
def tools() -> dict[str, dict[str, Any]]:
    return _mcp_tools()


def _cli_name(tool: str, mcp_name: str) -> tuple[str, bool]:
    return RENAMES.get((tool, mcp_name)) or RENAMES.get(("*", mcp_name)) or (mcp_name, False)


def _same_default(cli_default: Any, mcp_default: Any, inverted: bool) -> bool:
    if inverted:
        return bool(cli_default) is not bool(mcp_default)
    if cli_default in _EMPTY and mcp_default in _EMPTY:
        return True
    if isinstance(cli_default, int | float) and isinstance(mcp_default, int | float):
        return float(cli_default) == float(mcp_default)
    return bool(cli_default == mcp_default)


def test_every_tool_and_command_is_accounted_for(tools: dict[str, dict[str, Any]]) -> None:
    assert sorted(tools) == sorted(PAIRS), "pair (or explain) every MCP tool in PAIRS"
    commands = set(_cli_commands())
    assert set(PAIRS.values()) <= commands, "PAIRS names a CLI command that does not exist"
    unaccounted = commands - set(PAIRS.values()) - set(CLI_ONLY_COMMANDS)
    assert not unaccounted, f"new CLI command(s) {unaccounted}: add an MCP tool or a reason"
    stale = set(CLI_ONLY_COMMANDS) - commands
    assert not stale, f"CLI_ONLY_COMMANDS lists commands that no longer exist: {stale}"


@pytest.mark.parametrize("tool", sorted(PAIRS))
def test_parameters_match(tool: str, tools: dict[str, dict[str, Any]]) -> None:
    schema = tools[tool]
    props: dict[str, Any] = schema.get("properties") or {}
    mcp_required = set(schema.get("required") or [])
    cli_params = {p.name: p for p in _cli_commands()[PAIRS[tool]].params}

    mapped = {_cli_name(tool, name)[0]: name for name in props}
    cli_only = set(cli_params) - set(mapped) - set(CLI_ONLY_PARAMS.get(tool, {}))
    mcp_only = {n for c, n in mapped.items() if c not in cli_params}
    mcp_only -= set(MCP_ONLY_PARAMS.get(tool, {}))
    assert not cli_only, f"{tool}: CLI has {sorted(cli_only)} but the MCP tool does not"
    assert not mcp_only, f"{tool}: MCP has {sorted(mcp_only)} but the CLI does not"
    for listed in CLI_ONLY_PARAMS.get(tool, {}):
        assert listed in cli_params and listed not in mapped, f"{tool}: stale CLI_ONLY {listed}"
    for listed in MCP_ONLY_PARAMS.get(tool, {}):
        assert listed in props, f"{tool}: stale MCP_ONLY {listed}"

    for mcp_name, spec in props.items():
        cli_name, inverted = _cli_name(tool, mcp_name)
        if cli_name not in cli_params:
            continue
        param = cli_params[cli_name]
        assert bool(param.required) == (mcp_name in mcp_required), (
            f"{tool}.{mcp_name}: required on one surface only"
        )
        if param.required or (tool, mcp_name) in DEFAULT_DIFFERENCES:
            continue
        assert _same_default(param.default, spec.get("default"), inverted), (
            f"{tool}.{mcp_name}: default {spec.get('default')!r} vs CLI --{cli_name} "
            f"{param.default!r}{' (inverted)' if inverted else ''}"
        )


def test_allow_lists_name_real_tools() -> None:
    for table in (CLI_ONLY_PARAMS, MCP_ONLY_PARAMS):
        assert set(table) <= set(PAIRS)
    assert {t for t, _ in DEFAULT_DIFFERENCES} <= set(PAIRS)
    assert {t for t, _ in RENAMES} <= set(PAIRS) | {"*"}
