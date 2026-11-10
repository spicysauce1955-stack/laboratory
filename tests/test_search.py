"""lab.search: which parameter combinations pass an experiment-supplied test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lab.search import HITS_FILE, crosses_threshold, find_boundary, grid_search


def read_hits(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / HITS_FILE).read_text().splitlines()]


# --- crosses_threshold ---------------------------------------------------------------------------


def test_an_upward_crossing_is_true():
    assert crosses_threshold(np.array([0.0, 0.5, 2.0, 0.1]), 1.0) is True


def test_never_reaching_the_threshold_is_false():
    assert crosses_threshold(np.array([0.0, 0.5, 0.9]), 1.0) is False


def test_starting_above_the_threshold_is_not_a_crossing():
    assert crosses_threshold(np.array([3.0, 2.0, 1.5]), 1.0) is False


def test_a_plain_list_is_accepted():
    assert crosses_threshold([0.0, 2.0], 1.0) is True


def test_any_channel_crossing_counts_with_time_on_the_first_axis():
    quiet = [0.0, 0.1, 0.2]
    active = [0.0, 0.1, 5.0]
    assert crosses_threshold(np.array([quiet, active]).T, 1.0) is True
    assert crosses_threshold(np.array([quiet, quiet]).T, 1.0) is False


# --- grid_search, one combination at a time ------------------------------------------------------


def test_every_combination_is_evaluated_once(tmp_path):
    seen: list[tuple[float, str]] = []

    def evaluate(a: float, b: str) -> bool:
        seen.append((a, b))
        return False

    result = grid_search(evaluate, {"a": [1.0, 2.0], "b": ["x", "y", "z"]}, run_dir=tmp_path)

    assert sorted(seen) == [
        (1.0, "x"),
        (1.0, "y"),
        (1.0, "z"),
        (2.0, "x"),
        (2.0, "y"),
        (2.0, "z"),
    ]
    assert result.tested == 6


def test_only_combinations_that_pass_are_logged(tmp_path):
    result = grid_search(lambda a, b: a + b > 3, {"a": [1, 2], "b": [1, 2]}, run_dir=tmp_path)

    assert read_hits(tmp_path) == [{"a": 2, "b": 2}]
    assert (result.tested, result.hits) == (4, 1)
    assert result.hits_path == tmp_path / HITS_FILE


def test_parameters_need_not_be_numeric(tmp_path):
    grid_search(lambda rule, n: rule == "stdp", {"rule": ["stdp", "hebb"], "n": [10]}, run_dir=tmp_path)
    assert read_hits(tmp_path) == [{"rule": "stdp", "n": 10}]


def test_extra_columns_are_logged_beside_the_parameters(tmp_path):
    grid_search(lambda a: {"hit": a > 1, "score": a * 10}, {"a": [1, 2]}, run_dir=tmp_path)
    assert read_hits(tmp_path) == [{"a": 2, "score": 20}]


def test_numpy_values_are_logged_as_plain_numbers(tmp_path):
    grid_search(
        lambda a: {"hit": np.bool_(True), "score": np.float32(0.5)},
        {"a": np.array([1.5])},
        run_dir=tmp_path,
    )
    assert read_hits(tmp_path) == [{"a": 1.5, "score": 0.5}]


def test_the_log_defaults_to_the_run_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_RUN_DIR", str(tmp_path))
    grid_search(lambda a: True, {"a": [1]})
    assert read_hits(tmp_path) == [{"a": 1}]


def test_a_search_with_no_hits_leaves_an_empty_log(tmp_path):
    result = grid_search(lambda a: False, {"a": [1, 2]}, run_dir=tmp_path)
    assert result.hits == 0
    assert (tmp_path / HITS_FILE).read_text() == ""


def test_a_new_search_replaces_an_old_log(tmp_path):
    grid_search(lambda a: True, {"a": [1]}, run_dir=tmp_path)
    grid_search(lambda a: True, {"a": [2]}, run_dir=tmp_path)
    assert read_hits(tmp_path) == [{"a": 2}]


def test_hits_found_before_a_crash_are_kept(tmp_path):
    def evaluate(a: int) -> bool:
        if a == 3:
            raise RuntimeError("simulator crashed")
        return True

    with pytest.raises(RuntimeError, match="simulator crashed"):
        grid_search(evaluate, {"a": [1, 2, 3]}, run_dir=tmp_path)
    assert read_hits(tmp_path) == [{"a": 1}, {"a": 2}]


# --- grid_search, in batches ---------------------------------------------------------------------


def test_a_batched_evaluator_receives_equal_length_columns(tmp_path):
    calls: list[tuple[list[int], list[int]]] = []

    def evaluate(a: list[int], b: list[int]) -> list[bool]:
        calls.append((list(a), list(b)))
        return [False] * len(a)

    result = grid_search(evaluate, {"a": [1, 2], "b": [3, 4]}, batch_size=3, run_dir=tmp_path)

    assert calls == [([1, 1, 2], [3, 4, 3]), ([2], [4])]
    assert result.tested == 4


def test_a_batched_evaluator_can_return_an_array(tmp_path):
    grid_search(
        lambda a: np.asarray(a) > 1,
        {"a": [1, 2, 3]},
        batch_size=2,
        run_dir=tmp_path,
    )
    assert read_hits(tmp_path) == [{"a": 2}, {"a": 3}]


def test_a_batched_evaluator_can_return_extra_columns(tmp_path):
    def evaluate(a: list[int]) -> dict[str, Any]:
        values = np.asarray(a)
        return {"hit": values > 1, "score": values * 10}

    grid_search(evaluate, {"a": [1, 2]}, batch_size=2, run_dir=tmp_path)
    assert read_hits(tmp_path) == [{"a": 2, "score": 20}]


def test_batching_does_not_change_which_combinations_pass(tmp_path):
    grid = {"a": list(range(5)), "b": list(range(5))}
    one = tmp_path / "one"
    many = tmp_path / "many"

    grid_search(lambda a, b: a * b > 6, grid, run_dir=one)
    grid_search(
        lambda a, b: [x * y > 6 for x, y in zip(a, b)],
        grid,
        batch_size=7,
        run_dir=many,
    )

    assert read_hits(one) == read_hits(many)


def test_a_batch_returning_the_wrong_length_is_refused(tmp_path):
    with pytest.raises(ValueError, match="1 values for 2"):
        grid_search(lambda a: [True], {"a": [1, 2]}, batch_size=2, run_dir=tmp_path)


def test_an_extra_column_of_the_wrong_length_is_refused(tmp_path):
    with pytest.raises(ValueError, match="score"):
        grid_search(
            lambda a: {"hit": [True, True], "score": [1]},
            {"a": [1, 2]},
            batch_size=2,
            run_dir=tmp_path,
        )


# --- refusing results that would corrupt the log -------------------------------------------------


@pytest.mark.parametrize("bad", [0.3, 1, "yes", None])
def test_a_result_that_is_not_true_or_false_is_refused(tmp_path, bad):
    with pytest.raises(TypeError, match="True or False"):
        grid_search(lambda a: bad, {"a": [1]}, run_dir=tmp_path)


def test_a_batch_of_scores_instead_of_booleans_is_refused(tmp_path):
    with pytest.raises(TypeError, match="True or False"):
        grid_search(lambda a: np.array([0.3]), {"a": [1]}, batch_size=1, run_dir=tmp_path)


def test_a_mapping_without_the_hit_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match="'hit'"):
        grid_search(lambda a: {"score": 1}, {"a": [1]}, run_dir=tmp_path)


def test_an_extra_column_cannot_overwrite_a_parameter(tmp_path):
    with pytest.raises(ValueError, match="'a'"):
        grid_search(lambda a: {"hit": True, "a": 99}, {"a": [1]}, run_dir=tmp_path)


@pytest.mark.parametrize("grid", [{}, {"a": []}, {"a": [1], "b": []}])
def test_a_grid_with_nothing_to_test_is_refused(tmp_path, grid):
    with pytest.raises(ValueError, match="grid"):
        grid_search(lambda **kw: True, grid, run_dir=tmp_path)


@pytest.mark.parametrize("size", [0, -1])
def test_a_batch_size_below_one_is_refused(tmp_path, size):
    with pytest.raises(ValueError, match="batch_size"):
        grid_search(lambda a: [True], {"a": [1]}, batch_size=size, run_dir=tmp_path)


# --- find_boundary -------------------------------------------------------------------------------


def test_the_boundary_is_found_within_tolerance():
    found = find_boundary(lambda x: x >= 1.2345, "x", lo=0.0, hi=10.0, tol=1e-4)
    assert 1.2345 <= found <= 1.2345 + 1e-4


def test_a_batched_evaluator_works_for_the_boundary():
    found = find_boundary(
        lambda x: np.asarray(x) >= 2.5, "x", lo=0.0, hi=10.0, tol=1e-4, batched=True
    )
    assert found == pytest.approx(2.5, abs=1e-4)


def test_a_true_lower_end_is_not_a_bracket():
    with pytest.raises(ValueError, match="bracket"):
        find_boundary(lambda x: True, "x", lo=0.0, hi=1.0)


def test_a_false_upper_end_is_not_a_bracket():
    with pytest.raises(ValueError, match="bracket"):
        find_boundary(lambda x: False, "x", lo=0.0, hi=1.0)


def test_a_reversed_range_is_refused():
    with pytest.raises(ValueError, match="lo"):
        find_boundary(lambda x: x > 0, "x", lo=1.0, hi=0.0)


def test_a_tolerance_of_zero_is_refused():
    with pytest.raises(ValueError, match="tol"):
        find_boundary(lambda x: x > 0.5, "x", lo=0.0, hi=1.0, tol=0.0)


def test_a_tolerance_below_float_resolution_still_finishes():
    found = find_boundary(lambda x: x >= 1e9 + 0.5, "x", lo=1e9, hi=1e9 + 1.0, tol=1e-30)
    assert found == pytest.approx(1e9 + 0.5, abs=1e-5)
