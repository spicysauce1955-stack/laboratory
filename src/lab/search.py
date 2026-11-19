"""Experiment-side parameter search: which parameter combinations pass your test?

The lab knows nothing about the model. The entrypoint supplies ``evaluate``, which runs whatever
it likes (any simulator, neuron, network or learning rule) and answers True or False. This module
walks the combinations and records the ones that passed::

    from lab.search import crosses_threshold, grid_search

    def evaluate(gain: float, tau: float) -> bool:
        trace = simulate(gain=gain, tau=tau)
        return crosses_threshold(trace, threshold=1.0)

    grid_search(evaluate, {"gain": [0.5, 1.0, 2.0], "tau": [10.0, 20.0]})

To log more than the parameters, return a mapping: ``{"hit": passed, "latency": t}``. Extra values
may be numbers, strings, booleans, lists, or NumPy/tensor values (a one-element array is logged as
a number, a larger one as a list); anything else is refused at the first answer, naming the
column. An evaluator that handles many combinations at once passes ``batch_size``; it then
receives equal-length lists and returns one answer per combination.

What it writes, all in ``$LAB_RUN_DIR``:

- ``hits.jsonl`` -- one JSON object per passing combination, appended as they are found so a
  crashed job keeps them, and picked up by the mid-run mirror (:mod:`lab.partials`) like any
  top-level ``*.jsonl``. A new search replaces it unless ``append=True``.
- ``metrics.jsonl`` -- ``search/tested`` and ``search/hits`` progress points (step = combinations
  tested so far), at most one pair every :data:`PROGRESS_INTERVAL_S` plus the final count, so
  ``lab metrics`` shows a search with few hits moving.
- ``results.csv`` -- only with ``summary_axis``: one ``<axis>,tested,hits`` row per value of that
  grid key, zero-hit values included, replaced like ``hits.jsonl``. ``sweep-aggregate`` judges a
  seed present only by its rows in ``results.csv`` (and reads the seed as an integer), so an
  entrypoint writing hits alone leaves every seed "missing" and ``sweep-retry`` buys them again.

**An empty** ``hits.jsonl`` **means "nothing passed" only if the job succeeded.** A job that
crashed, timed out or was cancelled leaves the same empty (or short) file; check the job's status
before reading absence as a result.

Seeds: make the seed a grid key and summarise on it, rather than calling ``grid_search`` once per
seed. In a sharded sweep each shard is told its seeds as ``seeds=0,1,2,3``::

    seeds = [int(s) for s in get_overrides()["seeds"].split(",")]  # from lab.experiment
    grid_search(evaluate, {"seed": seeds, "gain": gains}, summary_axis="seed")

Put the seed first, as here: values of the first grid key finish one after another, so each
seed's row is written the moment it is done and a job killed at its wall-clock cap keeps the seeds
it finished. A later key's rows can only be written when the whole search ends. If you must loop
over seeds yourself, pass ``append=True``, or each call erases the previous seed's hits.

:func:`find_boundary` bisects one numeric parameter for where the answer flips. **It assumes a
single False-to-True transition in the range and cannot detect more**: given several, it returns
one of them without warning. Map the range with a coarse ``grid_search`` first if unsure. **It
writes nothing to** ``hits.jsonl`` (or anywhere): its return value is its only output.

Like :mod:`lab.experiment`, this module imports nothing from ``lab.store`` or ``lab.core``.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from lab.experiment import _resolve_run_dir
from lab.metrics import log_metric

HITS_FILE = "hits.jsonl"
HIT_KEY = "hit"
SUMMARY_FILE = "results.csv"
"""Written only with ``summary_axis``; the default ``--results-file`` that sweep-aggregate reads."""

TESTED_METRIC = "search/tested"
HITS_METRIC = "search/hits"
PROGRESS_INTERVAL_S = 10.0
"""Least wall-clock time between two progress records in ``metrics.jsonl``."""

_clock: Callable[[], float] = time.monotonic
"""Read once per evaluate call; tests replace it to move time without sleeping."""

Outcome = tuple[bool, dict[str, Any]]


class AxisCount(NamedTuple):
    """One value of the ``summary_axis``: combinations tested with it, and how many passed."""

    value: Any
    tested: int
    hits: int


@dataclass(frozen=True)
class SearchResult:
    tested: int
    hits: int
    hits_path: Path
    summary_path: Path | None = None
    """``results.csv``, when ``summary_axis`` was given."""
    summary: tuple[AxisCount, ...] = ()
    """One entry per ``summary_axis`` value, in grid order, zero-hit values included."""


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
        needs = (
            "it must return one answer"
            if name == HIT_KEY
            else f"its extra column {name!r} must hold one value"
        )
        raise TypeError(
            f"evaluate is called in batch mode with {count} combination(s) per call, so {needs} "
            f"per combination (a list or array), but it returned the single value {column!r}. "
            "Return one per combination, or evaluate one combination at a time (no batch_size "
            "in grid_search, batched=False in find_boundary)"
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


class _Summary:
    """Per-value counts for one grid axis, kept as ``results.csv`` rows: ``<axis>,tested,hits``.

    Every value of an axis is tested by the same number of combinations, so only hits are counted,
    and only when they happen. The first axis is the outermost loop of :func:`itertools.product`:
    its values finish one after another, so each row is appended the moment its value is done and
    a killed job keeps them. Any other axis's values all finish together, at the very end.
    """

    def __init__(
        self, path: Path, axis: str, axes: Mapping[str, Sequence[Any]], *, append: bool
    ) -> None:
        names = list(axes)
        if axis not in axes:
            raise ValueError(f"summary_axis {axis!r} is not a grid key; the grid has {names}")
        self.header = [axis, "tested", "hits"]
        if axis in self.header[1:]:
            raise ValueError(f"summary_axis {axis!r} would clash with a results.csv column")
        self.cells = [str(_plain(value)) for value in axes[axis]]
        repeated = sorted(cell for cell, n in Counter(self.cells).items() if n > 1)
        if repeated:
            raise ValueError(
                f"summary_axis {axis!r} repeats {repeated}: each value gets exactly one row "
                "(sweep-aggregate refuses a seed listed twice)"
            )
        position = names.index(axis)
        self.path = path
        self.values = list(axes[axis])
        self.stride = math.prod(len(axes[name]) for name in names[position + 1 :])
        self.per_value = math.prod(len(values) for values in axes.values()) // len(self.values)
        self.streams = position == 0
        self.hits = [0] * len(self.values)
        self.written = 0
        self.fresh = True
        if append and path.exists():
            with path.open(newline="") as f:
                existing = next(csv.reader(f), [])
            if existing and existing != self.header:
                raise ValueError(
                    f"{path} has the header {existing}, not {self.header}: append=True only "
                    "adds rows to a summary of the same axis"
                )
            self.fresh = not existing

    def start(self) -> None:
        """Write the header, unless appending to a summary that already has one."""
        if self.fresh:
            with self.path.open("w", newline="") as f:
                csv.writer(f, lineterminator="\n").writerow(self.header)

    def count_hit(self, combination: int) -> None:
        """Count a pass at ``combination``, its index in grid order."""
        self.hits[(combination // self.stride) % len(self.values)] += 1

    def advance(self, tested: int) -> None:
        """Append the rows of every first-axis value whose combinations are all done."""
        if self.streams:
            self._write_through(tested // self.per_value)

    def finish(self) -> tuple[AxisCount, ...]:
        self._write_through(len(self.values))
        return tuple(
            AxisCount(value, self.per_value, hits)
            for value, hits in zip(self.values, self.hits, strict=True)
        )

    def _write_through(self, done: int) -> None:
        if done <= self.written:
            return
        with self.path.open("a", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(
                [self.cells[i], self.per_value, self.hits[i]] for i in range(self.written, done)
            )
        self.written = done


class _Progress:
    """Combinations tested and hits so far, as ``search/tested`` and ``search/hits`` points.

    ``hits.jsonl`` only grows on a pass, so a long search with few hits would otherwise look idle
    in ``lab metrics``. The first answer is logged at once (the search is alive), then at most one
    pair every :data:`PROGRESS_INTERVAL_S`, then the final count. The step is the number of
    combinations tested so far in this call.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.last_time = -math.inf
        self.last_step = 0

    def update(self, tested: int, hits: int) -> None:
        now = _clock()
        if now - self.last_time >= PROGRESS_INTERVAL_S:
            self._log(tested, hits)
            self.last_time = now

    def finish(self, tested: int, hits: int) -> None:
        if tested != self.last_step:
            self._log(tested, hits)

    def _log(self, tested: int, hits: int) -> None:
        log_metric(TESTED_METRIC, tested, tested, run_dir=self.run_dir)
        log_metric(HITS_METRIC, hits, tested, run_dir=self.run_dir)
        self.last_step = tested


def grid_search(
    evaluate: Callable[..., Any],
    grid: Mapping[str, Iterable[Any]],
    *,
    batch_size: int | None = None,
    run_dir: str | Path | None = None,
    summary_axis: str | None = None,
    append: bool = False,
) -> SearchResult:
    """Run ``evaluate`` on every combination in ``grid`` and log the ones that pass.

    ``grid`` maps ``evaluate`` keyword names to their values, which need not be numeric. Use a
    closure or :func:`functools.partial` for anything held fixed. A log left by an earlier search
    in the same directory is replaced, unless ``append=True``: then this search's hits (and
    summary rows) go after the ones already there.

    ``summary_axis`` names a grid key whose every value gets a ``results.csv`` row,
    ``<axis>,tested,hits``, zero-hit values included: pass ``"seed"`` in a sharded sweep, so that
    ``sweep-aggregate`` sees every seed the job finished (it reads only ``results.csv``). Put that
    key first in ``grid`` and each row is written as soon as its value is done; otherwise the rows
    are written when the search ends. Make the seed a grid key rather than calling this once per
    seed; if you do loop, pass ``append=True`` or each call erases the previous seed's results.
    """
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    axes = {name: list(values) for name, values in grid.items()}
    if not axes or not all(axes.values()):
        raise ValueError("grid has nothing to test: every parameter needs at least one value")
    for name, values in axes.items():
        for value in values:
            _check_loggable(name, value, "grid parameter")
    base = _resolve_run_dir(run_dir)
    summary = (
        None
        if summary_axis is None
        else _Summary(base / SUMMARY_FILE, summary_axis, axes, append=append)
    )

    path = base / HITS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        path.touch()
    else:
        path.write_text("")
    if summary is not None:
        summary.start()

    progress = _Progress(base)
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
        lines = []
        for offset, (p, (hit, extras)) in enumerate(zip(params, outcomes, strict=True)):
            if hit:
                lines.append(_encode_row(p, extras))
                if summary is not None:
                    summary.count_hit(tested + offset)
        if lines:
            with path.open("a") as f:
                f.writelines(lines)
        tested += len(params)
        hits += len(lines)
        if summary is not None:
            summary.advance(tested)
        progress.update(tested, hits)
    progress.finish(tested, hits)
    if summary is None:
        return SearchResult(tested=tested, hits=hits, hits_path=path)
    return SearchResult(
        tested=tested,
        hits=hits,
        hits_path=path,
        summary_path=summary.path,
        summary=summary.finish(),
    )


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

    A True/False answer has no gradient, so this bisects. **It assumes a single False-to-True
    transition inside the range and cannot detect more**: it only ever looks at a handful of
    points, so with several transitions (False, True, False, True) it returns one of them, not
    necessarily the smallest, and says nothing. It refuses a range whose ends are not False then
    True, but cannot check what lies between. If unsure, map the range with a coarse
    :func:`grid_search` first.

    **It writes nothing** -- no ``hits.jsonl`` row, no metric: the returned value is its only
    output, so log it yourself if you need it. Pass ``batched=True`` if ``evaluate`` takes lists,
    as it does with ``grid_search(batch_size=...)``.
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
