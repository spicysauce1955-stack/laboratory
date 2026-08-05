"""Experiment-side contract helpers (spec §7) — the config-consumption handshake.

The lab passes sweep overrides to entrypoints as loose ``key=value`` argv tokens. Historically
entrypoints parsed those ad hoc and silently ignored unknown keys — a typo or stale script could
run a *different experiment than requested* while reporting success (field-report #1). The
handshake that closes this loop: the entrypoint writes ``$LAB_RUN_DIR/effective_config.json``
with the config it actually consumed, and the store compares it against the submitted config at
the succeeded transition, failing the job on unconsumed keys (see ``JobStore.update_manifest``).

Adopting the convention is one call::

    from lab.experiment import get_overrides
    ov = get_overrides(known={"steps", "seeds"})       # parses sys.argv, writes the file,
    steps = int(ov.get("steps", "100"))                # and exits non-zero on unknown keys

Entrypoints that avoid importing ``lab`` may simply write the JSON file themselves — the store
only looks at the file (same zero-coupling stance as ``metrics.log_metric``).

This module deliberately imports nothing from ``lab.store``/``lab.core``.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

EFFECTIVE_CONFIG_FILE = "effective_config.json"


def parse_overrides(argv: Sequence[str]) -> dict[str, str]:
    """Parse loose ``key=value`` tokens (the lab's sweep-override convention). Tokens without
    ``=`` are ignored — matching the ad-hoc parsers this replaces. Pure."""
    out: dict[str, str] = {}
    for token in argv:
        if "=" not in token or token.startswith("-"):
            continue
        key, _, value = token.partition("=")
        out[key] = value
    return out


def _resolve_run_dir(run_dir: str | Path | None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    return Path(os.environ.get("LAB_RUN_DIR", "."))


def write_effective_config(config: Mapping[str, Any], run_dir: str | Path | None = None) -> Path:
    """Write the config the entrypoint actually consumed to ``$LAB_RUN_DIR`` (or ``run_dir``)."""
    path = _resolve_run_dir(run_dir) / EFFECTIVE_CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(config), indent=2, sort_keys=True, default=str))
    return path


def read_effective_config(run_dir: str | Path) -> dict[str, Any] | None:
    """The entrypoint's reported config, ``None`` if never reported. Raises :class:`ValueError`
    on corrupt/non-dict JSON — corrupt evidence must not pass as absent."""
    path = Path(run_dir) / EFFECTIVE_CONFIG_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{EFFECTIVE_CONFIG_FILE} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{EFFECTIVE_CONFIG_FILE} must contain a JSON object")
    return data


def get_overrides(
    known: Iterable[str] | None = None,
    *,
    argv: Sequence[str] | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, str]:
    """Parse overrides, write ``effective_config.json``, and (with a ``known`` schema) refuse
    unknown keys by exiting non-zero — failing at runtime instead of post-hoc.

    The effective file is written *before* the unknown-key check so the lab can still diagnose
    what was consumed even when the job dies here.
    """
    overrides = parse_overrides(sys.argv[1:] if argv is None else argv)
    write_effective_config(overrides, run_dir=run_dir)
    if known is not None:
        unknown = sorted(set(overrides) - set(known))
        if unknown:
            raise SystemExit(
                f"unknown config override(s): {unknown} — known keys: {sorted(set(known))}"
            )
    return overrides


def unreferenced_keys(source: str, keys: Iterable[str]) -> list[str]:
    """Keys whose literal string never appears in ``source`` — the cheap pre-submit lint for
    legacy entrypoints that don't write ``effective_config.json``. Heuristic by design."""
    return sorted(k for k in keys if k not in source)
