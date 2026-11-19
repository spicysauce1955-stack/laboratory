"""`lab lint` checks the keys a sweep will pass against what the entrypoint DECLARES (L7).

The stress run: `lab lint -c "python experiments/example_search.py" --grid gain=2` passed, and
the job then failed with "unknown config override(s): ['gain']". Lint grepped the source for the
literal key, and `gain` is all over that file — as a `lab.search` grid axis, not as an override.
An entrypoint that calls `get_overrides(known={...})` states exactly which keys it accepts and
refuses the rest at runtime, so lint now reads that declaration and checks against it; the grep
stays as the fallback for legacy entrypoints that declare nothing. `--seeds` adds the `seeds=`
key a sharded sweep appends to every shard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lab.cli import app
from lab.lint import declared_keys, lint_keys

runner = CliRunner()
SEARCH = "experiments/example_search.py"


def _lint(*args: str) -> tuple[int, dict]:
    result = runner.invoke(app, ["lint", *args])
    return result.exit_code, json.loads(result.stdout)


def test_the_stress_run_false_negative_is_caught() -> None:
    code, out = _lint("-c", f"python {SEARCH}", "--grid", "gain=2")

    assert code == 1
    assert out["missing_keys"] == ["gain"]
    assert out["check"] == "declared"
    assert out["declared_keys"] == ["batch", "n", "seeds", "sleep"]


def test_a_declared_key_passes() -> None:
    code, out = _lint("-c", f"python {SEARCH}", "--grid", "n=10,20")

    assert code == 0 and out["missing_keys"] == []


def test_seeds_checks_the_key_a_sharded_sweep_appends(tmp_path: Path) -> None:
    script = tmp_path / "exp.py"
    script.write_text(
        "from lab.experiment import get_overrides\nov = get_overrides(known={'n'})\n# seeds\n"
    )

    code, out = _lint("-c", f"python {script}", "--grid", "n=1", "--seeds", "0-3")

    assert code == 1 and out["missing_keys"] == ["seeds"]
    assert _lint("-c", f"python {SEARCH}", "--seeds", "0-7")[0] == 0


def test_a_bad_seed_spec_is_a_usage_error() -> None:
    result = runner.invoke(app, ["lint", "-c", f"python {SEARCH}", "--seeds", "7-1"])
    assert result.exit_code == 2


def test_a_grid_seed_key_is_checked_against_the_declaration() -> None:
    """`get_overrides(known=...)` refuses `seed=` like any undeclared key, so a `--grid seed=…`
    sweep of such an entrypoint fails at runtime — lint must say so."""
    code, out = _lint("-c", f"python {SEARCH}", "--grid", "seed=1,2")

    assert code == 1 and out["missing_keys"] == ["seed"]


def test_a_legacy_entrypoint_falls_back_to_the_grep(tmp_path: Path) -> None:
    script = tmp_path / "legacy.py"
    script.write_text('import sys\nargs = dict(a.split("=") for a in sys.argv[1:])\nargs["lr"]\n')

    code, out = _lint("-c", f"python {script}", "--grid", "lr=1", "--grid", "momentum=0.9")

    assert code == 1
    assert out["check"] == "heuristic" and out["declared_keys"] is None
    assert out["missing_keys"] == ["momentum"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("get_overrides(known={'a', 'b'})", {"a", "b"}),
        ("get_overrides({'a'})", {"a"}),
        ("get_overrides(known=['a', 'b'], run_dir='x')", {"a", "b"}),
        ("get_overrides(known=frozenset({'a'}))", {"a"}),
        ("KNOWN = {'a', 'c'}\nget_overrides(known=KNOWN)", {"a", "c"}),
        ("DEFAULTS = {'a': 1, 'b': 2}\nget_overrides(known=DEFAULTS)", {"a", "b"}),
        ("DEFAULTS = {'a': 1}\nget_overrides(known=DEFAULTS.keys())", {"a"}),
        ("get_overrides(known={'a'} | {'b'})", {"a", "b"}),
        ("import lab.experiment as e\ne.get_overrides(known={'a'})", {"a"}),
        ("from lab.experiment import get_overrides as go\ngo(known={'z'})", {"z"}),
        ("get_overrides(known={'a'})\nget_overrides(known={'b'})", {"a", "b"}),
        # Nothing static to read: the grep decides.
        ("get_overrides()", None),
        ("get_overrides(known=None)", None),
        ("get_overrides(known=load_schema())", None),
        ("print('no handshake at all')", None),
        ("this is not python (", None),
    ],
)
def test_declared_keys(source: str, expected: set[str] | None) -> None:
    assert declared_keys(source) == expected


def test_lint_keys_heuristic_leaves_a_grid_seed_to_the_lab() -> None:
    """Without a declaration nothing refuses `seed=` at runtime, and the store's config audit
    skips it (the lab consumes it as `$LAB_SEED`), so flagging it would be a false alarm."""
    assert lint_keys("print('x')", ["seed"])["missing_keys"] == []
