"""Experiment-side parameter search: which parameter combinations pass your test?

The lab knows nothing about the model. The entrypoint supplies ``evaluate``, which runs whatever
it likes (any simulator, neuron, network or learning rule) and answers True or False. This module
walks the combinations and records the ones that passed::

    from lab.search import crosses_threshold, grid_search

    def evaluate(gain: float, tau: float) -> bool:
        trace = simulate(gain=gain, tau=tau)
        return crosses_threshold(trace, threshold=1.0)

    grid_search(evaluate, {"gain": [0.5, 1.0, 2.0], "tau": [10.0, 20.0]})

To log more than the parameters, return a mapping: ``{"hit": passed, "latency": t}``. An evaluator
that handles many combinations at once passes ``batch_size``; it then receives equal-length lists
and returns one answer per combination.

Hits go to ``$LAB_RUN_DIR/hits.jsonl``, one JSON object per line, written as they are found so a
crashed job keeps them. Being a top-level ``*.jsonl``, the file is also picked up by the mid-run
mirror (:mod:`lab.partials`). It is deliberately not ``results.csv``: ``sweep-aggregate`` reads any
row there as "this seed is present", so a seed with no hits would look missing.

An empty file means the search ran and nothing passed. Like :mod:`lab.experiment`, this module
imports nothing from ``lab.store`` or ``lab.core``.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lab.experiment import _resolve_run_dir

HITS_FILE = "hits.jsonl"
HIT_KEY = "hit"

Outcome = tuple[bool, dict[str, Any]]


@dataclass(frozen=True)
class SearchResult:
    tested: int
    hits: int
    hits_path: Path


def crosses_threshold(trace: Any, threshold: float) -> bool:
    """True if ``trace`` rises through ``threshold``, with time on the first axis.

    An upward crossing is required, so a trace that merely starts above threshold is False. With
    more than one channel, any channel crossing counts. NumPy, PyTorch and JAX arrays are compared
    where they live, so a GPU trace stays on the GPU.
    """
    values = np.asarray(trace) if isinstance(trace, list | tuple) else trace
    return bool(((values[:-1] < threshold) & (values[1:] >= threshold)).any())


def _plain(value: Any) -> Any:
    """A NumPy or tensor scalar as the Python value it holds; anything else unchanged."""
    return value.item() if hasattr(value, "item") else value


def _as_bool(value: Any) -> bool:
    """Refuse a score passed off as an answer: ``bool(0.3)`` would log every combination."""
    plain = _plain(value)
    if not isinstance(plain, bool):
        raise TypeError(f"evaluate must answer True or False, got {plain!r}")
    return plain


def _split(answer: Any) -> tuple[Any, dict[str, Any]]:
    """Separate the True/False answer from any extra columns returned with it."""
    if not isinstance(answer, Mapping):
        return answer, {}
    if HIT_KEY not in answer:
        raise ValueError(f"evaluate returned a mapping without the {HIT_KEY!r} key")
    return answer[HIT_KEY], {name: value for name, value in answer.items() if name != HIT_KEY}


def _outcomes(
    evaluate: Callable[..., Any], params: Sequence[Mapping[str, Any]], *, batched: bool
) -> list[Outcome]:
    if not batched:
        hit, extras = _split(evaluate(**params[0]))
        return [(_as_bool(hit), extras)]

    columns = {name: [p[name] for p in params] for name in params[0]}
    hit_column, extra_columns = _split(evaluate(**columns))
    hits = list(hit_column)
    extras_by_column = {name: list(column) for name, column in extra_columns.items()}
    for name, values in [(HIT_KEY, hits), *extras_by_column.items()]:
        if len(values) != len(params):
            raise ValueError(
                f"evaluate returned {len(values)} values for {len(params)} combinations "
                f"in column {name!r}"
            )
    return [
        (_as_bool(hit), {name: values[i] for name, values in extras_by_column.items()})
        for i, hit in enumerate(hits)
    ]


def _row(params: Mapping[str, Any], extras: Mapping[str, Any]) -> dict[str, Any]:
    clash = sorted(set(params) & set(extras))
    if clash:
        raise ValueError(f"extra column(s) would overwrite parameter(s): {clash}")
    return {name: _plain(value) for name, value in {**params, **extras}.items()}


def grid_search(
    evaluate: Callable[..., Any],
    grid: Mapping[str, Iterable[Any]],
    *,
    batch_size: int | None = None,
    run_dir: str | Path | None = None,
) -> SearchResult:
    """Run ``evaluate`` on every combination in ``grid`` and log the ones that pass.

    ``grid`` maps ``evaluate`` keyword names to their values, which need not be numeric. Use a
    closure or :func:`functools.partial` for anything held fixed. A log left by an earlier search
    in the same directory is replaced.
    """
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    axes = {name: list(values) for name, values in grid.items()}
    if not axes or not all(axes.values()):
        raise ValueError("grid has nothing to test: every parameter needs at least one value")

    path = _resolve_run_dir(run_dir) / HITS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")

    names = list(axes)
    combinations = itertools.product(*axes.values())
    tested = hits = 0
    while chunk := list(itertools.islice(combinations, batch_size or 1)):
        params = [dict(zip(names, combination, strict=True)) for combination in chunk]
        outcomes = _outcomes(evaluate, params, batched=batch_size is not None)
        rows = [
            _row(p, extras) for p, (hit, extras) in zip(params, outcomes, strict=True) if hit
        ]
        if rows:
            with path.open("a") as f:
                f.writelines(json.dumps(row, default=str) + "\n" for row in rows)
        tested += len(params)
        hits += len(rows)
    return SearchResult(tested=tested, hits=hits, hits_path=path)


def find_boundary(
    evaluate: Callable[..., Any],
    param: str,
    *,
    lo: float,
    hi: float,
    tol: float = 1e-3,
    batched: bool = False,
) -> float:
    """The smallest value of ``param`` in ``[lo, hi]`` that passes, to within ``tol``.

    A True/False answer has no gradient, so this bisects. It assumes a single False-to-True
    transition inside the range, and it refuses a range whose ends do not show one. Pass
    ``batched=True`` if ``evaluate`` takes lists, as it does with ``grid_search(batch_size=...)``.
    """
    if not lo < hi:
        raise ValueError(f"need lo < hi, got lo={lo}, hi={hi}")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")

    def passes(value: float) -> bool:
        return _outcomes(evaluate, [{param: value}], batched=batched)[0][0]

    if passes(lo) or not passes(hi):
        raise ValueError(f"[{lo}, {hi}] does not bracket a False-to-True transition in {param!r}")
    while hi - lo > tol:
        middle = (lo + hi) / 2
        if not lo < middle < hi:  # tol is below what floats can resolve here
            break
        if passes(middle):
            hi = middle
        else:
            lo = middle
    return hi
