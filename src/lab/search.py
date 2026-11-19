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
    """A NumPy or tensor value as plain Python: one element as a scalar, more as a (nested) list.

    Anything without ``.item`` is returned unchanged. ``.item()`` is tried first because it is the
    one call NumPy, PyTorch and JAX share; each raises its own error type for a larger array.
    """
    if not hasattr(value, "item"):
        return value
    try:
        return value.item()
    except (ValueError, RuntimeError, TypeError):
        if not hasattr(value, "tolist"):
            raise
        return value.tolist()


def _jsonable(value: Any) -> Any:
    """``json.dumps`` hook for what it cannot encode itself, at any depth: NumPy or tensor values
    become plain Python; anything else is refused rather than logged as an unreadable string."""
    plain = _plain(value)
    if plain is value:
        raise TypeError(f"{type(value).__name__} is not JSON-serializable")
    return plain


def _check_loggable(name: str, value: Any, what: str) -> None:
    try:
        json.dumps(value, default=_jsonable)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"{what} {name!r} cannot be written to {HITS_FILE} ({e}): use numbers, strings, "
            "booleans, or lists or arrays of them (look anything else up inside evaluate)"
        ) from e


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


_ONE_AT_A_TIME = "no batch_size in grid_search, batched=False in find_boundary"


def _single_answer(answer: Any, params: Mapping[str, Any]) -> bool:
    """The answer of an evaluator called with one combination, refusing a batch's worth."""
    plain = _plain(answer)
    if isinstance(plain, list | tuple):
        raise TypeError(
            f"evaluate returned {len(plain)} answers for one combination ({dict(params)}). If it "
            "evaluates many combinations per call, pass batch_size to grid_search (batched=True "
            "to find_boundary); otherwise return a single True or False"
        )
    return _as_bool(plain)


def _per_combination(column: Any, name: str, count: int) -> list[Any]:
    """A batched evaluator's column as a list, refusing a single value where ``count`` belong."""
    if isinstance(column, str | bytes | Mapping) or not hasattr(column, "__iter__"):
        single = True
    else:
        try:
            values = list(column)
        except TypeError:  # a 0-d array or tensor is iterable in type only
            single = True
        else:
            single = False
    if single:
        what = "its answer" if name == HIT_KEY else f"extra column {name!r}"
        noun = "answer" if name == HIT_KEY else "value"
        raise TypeError(
            f"evaluate is called in batch mode with {count} combination(s) per call, so {what} "
            f"must hold one {noun} per combination (a list or array), but it returned the single "
            f"value {column!r}. Return one {noun} per combination, or evaluate one combination "
            f"at a time ({_ONE_AT_A_TIME})"
        )
    if len(values) != count:
        raise ValueError(
            f"evaluate returned {len(values)} values for {count} combinations in column {name!r}"
        )
    return values


def _outcomes(
    evaluate: Callable[..., Any], params: Sequence[Mapping[str, Any]], *, batched: bool
) -> list[Outcome]:
    if not batched:
        hit, extras = _split(evaluate(**params[0]))
        return [(_single_answer(hit, params[0]), extras)]

    columns = {name: [p[name] for p in params] for name in params[0]}
    hit_column, extra_columns = _split(evaluate(**columns))
    hits = _per_combination(hit_column, HIT_KEY, len(params))
    extras_by_column = {
        name: _per_combination(column, name, len(params)) for name, column in extra_columns.items()
    }
    return [
        (_as_bool(hit), {name: values[i] for name, values in extras_by_column.items()})
        for i, hit in enumerate(hits)
    ]


def _encode_row(params: Mapping[str, Any], extras: Mapping[str, Any]) -> str:
    """One ``hits.jsonl`` line, naming the column at fault if it cannot be written."""
    clash = sorted(set(params) & set(extras))
    if clash:
        raise ValueError(f"extra column(s) would overwrite parameter(s): {clash}")
    row = {**params, **extras}
    try:
        return json.dumps(row, default=_jsonable) + "\n"
    except (TypeError, ValueError):
        for name, value in row.items():  # the slow path only runs to name the culprit
            _check_loggable(name, value, "extra column" if name in extras else "grid parameter")
        raise


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
    for name, values in axes.items():
        for value in values:
            _check_loggable(name, value, "grid parameter")

    path = _resolve_run_dir(run_dir) / HITS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")

    names = list(axes)
    combinations = itertools.product(*axes.values())
    tested = hits = 0
    while chunk := list(itertools.islice(combinations, batch_size or 1)):
        params = [dict(zip(names, combination, strict=True)) for combination in chunk]
        outcomes = _outcomes(evaluate, params, batched=batch_size is not None)
        if not tested:
            # A bad extra column is refused at the first answer rather than the first pass,
            # which in a search with few hits can be hours in. Checked once, not per row.
            _encode_row(params[0], outcomes[0][1])
        lines = [
            _encode_row(p, extras)
            for p, (hit, extras) in zip(params, outcomes, strict=True)
            if hit
        ]
        if lines:
            with path.open("a") as f:
                f.writelines(lines)
        tested += len(params)
        hits += len(lines)
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
