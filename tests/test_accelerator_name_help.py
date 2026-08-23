"""The CLI's own ``--accelerators`` help advertised a GPU name that cannot provision.

Field report 2026-08-23. Three Vast jobs (``20260823-105413-9b5435``, ``-105422-e50c00``,
``-105605-3d8127``) died at launch with::

    launch error: Catalog does not contain any instances satisfying the request:
    1x Vast({'RTX_4090': 1}, max_cost=$0.66/hr).

The user had not guessed the name. ``lab register --help`` and ``lab register-sweep --help``
both said ``e.g. RTX_4090:1``, and ``lab submit --help`` said ``e.g. RTX_3070:1``. Neither form
exists: sky's v8 vast catalog carries 17 accelerator names and **not one contains an
underscore** -- the only 4090 spelling is ``RTX4090``, and ``RTX_3070`` names no GPU at all.

The underscore is not imaginary, which is what makes this trap durable: Vast's *own* API wants
``RTX_4090``, and ``lab.scheduler.price.vast_gpu_name`` translates into it on purpose. Sky's
launcher wants the catalog form. The scaffolded skill already documents the distinction under
"Live-learned gotchas (these cost real money to discover)" -- the CLI's help contradicted it.

So this guards the *example*, not the parser: `lab` deliberately does not validate accelerator
strings itself (sky owns that catalog and it moves), which is exactly why the one name we print
has to be a real one.
"""

from __future__ import annotations

import re

import pytest
import typer

from lab.cli import app

# Vast's API spelling, which sky's launcher rejects. `lab.scheduler.price` converts *into* this
# form for the price feed; nothing user-facing should ever suggest typing it.
_VAST_API_FORM = re.compile(r"RTX_\d")


def _accelerator_helps() -> list[tuple[str, str]]:
    """Every registered command's ``--accelerators``/``--gpu`` help text, by command name."""
    found: list[tuple[str, str]] = []
    for cmd in app.registered_commands:
        name = cmd.name or (cmd.callback.__name__ if cmd.callback else "?")
        params = getattr(cmd.callback, "__defaults__", None) or ()
        for param in params:
            if not isinstance(param, typer.models.OptionInfo):
                continue
            decls = [d for d in (param.param_decls or ()) if isinstance(d, str)]
            if "--accelerators" not in decls and "--gpu" not in decls:
                continue
            found.append((name, param.help or ""))
    return found


def test_some_command_documents_accelerators() -> None:
    """Guard the guard: if introspection stops finding anything, the test below is vacuous."""
    assert _accelerator_helps(), "no --accelerators help text found to check"


@pytest.mark.parametrize("command,help_text", _accelerator_helps())
def test_accelerator_help_uses_sky_catalog_spelling(command: str, help_text: str) -> None:
    """No user-facing example may use Vast's underscored API spelling."""
    assert not _VAST_API_FORM.search(help_text), (
        f"`lab {command} --help` advertises an accelerator name sky's launcher rejects: "
        f"{help_text!r}. Use the sky-catalog form (e.g. RTX4090:1); see the 'GPU names' gotcha "
        f"in src/lab/_scaffold/project/skills/laboratory/SKILL.md."
    )
