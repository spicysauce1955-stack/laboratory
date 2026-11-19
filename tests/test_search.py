"""lab.search: which parameter combinations pass an experiment-supplied test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import lab.search as search
from lab.aggregate import merge_seed_rows
from lab.metrics import METRICS_FILE, log_metric, read_points
from lab.search import (
    HITS_FILE,
    SUMMARY_FILE,
    AxisCount,
    crosses_threshold,
    find_boundary,
    grid_search,
)


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


# --- summary_axis: one results.csv row per value, for sweep-aggregate ----------------------------


def hit_when_seed_is_one(seed: int, a: int) -> bool:
    return seed == 1 and a > 1


def test_there_is_no_summary_unless_asked_for(tmp_path):
    result = grid_search(hit_when_seed_is_one, {"seed": [0, 1], "a": [1, 2]}, run_dir=tmp_path)
    assert not (tmp_path / SUMMARY_FILE).exists()
    assert (result.summary_path, result.summary) == (None, ())


def test_the_summary_has_a_row_for_every_value_even_with_no_hits(tmp_path):
    grid_search(
        hit_when_seed_is_one,
        {"seed": [0, 1, 2], "a": [1, 2, 3]},
        summary_axis="seed",
        run_dir=tmp_path,
    )
    assert (tmp_path / SUMMARY_FILE).read_text() == "seed,tested,hits\n0,3,0\n1,3,2\n2,3,0\n"


def test_an_inner_axis_can_be_summarised(tmp_path):
    grid_search(
        hit_when_seed_is_one,
        {"a": [1, 2, 3], "seed": [0, 1, 2]},
        summary_axis="seed",
        run_dir=tmp_path,
    )
    assert (tmp_path / SUMMARY_FILE).read_text() == "seed,tested,hits\n0,3,0\n1,3,2\n2,3,0\n"


def test_batching_does_not_change_the_summary(tmp_path):
    def batched(seed: list[int], a: list[int]) -> list[bool]:
        return [hit_when_seed_is_one(s, x) for s, x in zip(seed, a)]

    for size in (2, 4, 9):
        grid_search(
            batched,
            {"a": [1, 2, 3], "seed": [0, 1, 2]},
            summary_axis="seed",
            batch_size=size,
            run_dir=tmp_path / str(size),
        )
        text = (tmp_path / str(size) / SUMMARY_FILE).read_text()
        assert text == "seed,tested,hits\n0,3,0\n1,3,2\n2,3,0\n"


def test_the_summary_is_returned(tmp_path):
    result = grid_search(
        hit_when_seed_is_one, {"seed": [0, 1], "a": [1, 2]}, summary_axis="seed", run_dir=tmp_path
    )
    assert result.summary_path == tmp_path / SUMMARY_FILE
    assert result.summary == (AxisCount(0, 2, 0), AxisCount(1, 2, 1))
    assert result.summary[1].hits == 1


def test_finished_values_of_the_first_axis_survive_a_crash(tmp_path):
    def evaluate(seed: int, a: int) -> bool:
        if seed == 2:
            raise RuntimeError("killed at the wall-clock cap")
        return a > 1

    with pytest.raises(RuntimeError):
        grid_search(
            evaluate, {"seed": [0, 1, 2], "a": [1, 2]}, summary_axis="seed", run_dir=tmp_path
        )
    assert (tmp_path / SUMMARY_FILE).read_text() == "seed,tested,hits\n0,2,1\n1,2,1\n"


def test_a_batch_spanning_two_values_writes_only_the_finished_one_before_a_crash(tmp_path):
    def evaluate(seed: list[int], a: list[int]) -> list[bool]:
        if 2 in seed:
            raise RuntimeError("killed")
        return [True] * len(seed)

    # batches: (0,1) (0,2) (1,1) | (1,2) (2,1) (2,2) -> only seed 0 is finished when it dies
    with pytest.raises(RuntimeError):
        grid_search(
            evaluate,
            {"seed": [0, 1, 2], "a": [1, 2]},
            summary_axis="seed",
            batch_size=3,
            run_dir=tmp_path,
        )
    assert (tmp_path / SUMMARY_FILE).read_text() == "seed,tested,hits\n0,2,2\n"


def test_an_inner_summary_axis_is_written_only_when_the_search_ends(tmp_path):
    def evaluate(a: int, seed: int) -> bool:
        if a == 2:
            raise RuntimeError("killed")
        return True

    with pytest.raises(RuntimeError):
        grid_search(
            evaluate, {"a": [1, 2], "seed": [0, 1]}, summary_axis="seed", run_dir=tmp_path
        )
    assert (tmp_path / SUMMARY_FILE).read_text() == "seed,tested,hits\n"


def test_a_new_search_replaces_an_old_summary(tmp_path):
    grid_search(lambda seed: True, {"seed": [0, 1]}, summary_axis="seed", run_dir=tmp_path)
    grid_search(lambda seed: True, {"seed": [5]}, summary_axis="seed", run_dir=tmp_path)
    assert (tmp_path / SUMMARY_FILE).read_text() == "seed,tested,hits\n5,1,1\n"


def test_sweep_aggregate_reads_every_seed_as_present(tmp_path):
    grid_search(
        hit_when_seed_is_one,
        {"seed": [0, 1, 2], "a": [1, 2]},
        summary_axis="seed",
        run_dir=tmp_path,
    )
    merged, present, partial = merge_seed_rows(
        [((tmp_path / SUMMARY_FILE).read_text(), "succeeded")], seed_column="seed"
    )
    assert (present, partial) == ([0, 1, 2], [])
    assert merged.splitlines()[0] == "seed,tested,hits,_shard_status"


def test_sweep_aggregate_keeps_the_seeds_a_crashed_shard_finished(tmp_path):
    def evaluate(seed: int, a: int) -> bool:
        if seed == 2:
            raise RuntimeError("killed")
        return False

    with pytest.raises(RuntimeError):
        grid_search(
            evaluate, {"seed": [0, 1, 2], "a": [1, 2]}, summary_axis="seed", run_dir=tmp_path
        )
    _, present, partial = merge_seed_rows(
        [((tmp_path / SUMMARY_FILE).read_text(), "timed_out")], seed_column="seed"
    )
    assert (present, partial) == ([0, 1], [0, 1])


def test_numpy_seeds_are_written_as_plain_integers(tmp_path):
    grid_search(lambda seed: False, {"seed": np.arange(2)}, summary_axis="seed", run_dir=tmp_path)
    assert (tmp_path / SUMMARY_FILE).read_text() == "seed,tested,hits\n0,1,0\n1,1,0\n"


def test_a_summary_axis_that_is_not_in_the_grid_is_refused_before_touching_anything(tmp_path):
    (tmp_path / HITS_FILE).write_text('{"a": 1}\n')
    calls: list[int] = []

    with pytest.raises(ValueError, match="summary_axis"):
        grid_search(
            lambda a: calls.append(a) or True, {"a": [1]}, summary_axis="seed", run_dir=tmp_path
        )
    assert calls == []
    assert (tmp_path / HITS_FILE).read_text() == '{"a": 1}\n'
    assert not (tmp_path / SUMMARY_FILE).exists()


def test_a_summary_axis_with_a_repeated_value_is_refused(tmp_path):
    with pytest.raises(ValueError, match="repeat"):
        grid_search(lambda seed: True, {"seed": [0, 1, 1]}, summary_axis="seed", run_dir=tmp_path)


def test_a_summary_axis_named_like_a_count_column_is_refused(tmp_path):
    with pytest.raises(ValueError, match="clash"):
        grid_search(lambda hits: True, {"hits": [0]}, summary_axis="hits", run_dir=tmp_path)


# --- append: one search per call, one log ---------------------------------------------------------


def test_append_keeps_the_hits_of_earlier_searches(tmp_path):
    for seed in (0, 1, 2):
        grid_search(
            lambda seed, a: a > 1, {"seed": [seed], "a": [1, 2]}, append=True, run_dir=tmp_path
        )
    assert read_hits(tmp_path) == [{"seed": 0, "a": 2}, {"seed": 1, "a": 2}, {"seed": 2, "a": 2}]


def test_append_starts_a_log_that_does_not_exist_yet(tmp_path):
    grid_search(lambda a: True, {"a": [1]}, append=True, run_dir=tmp_path / "fresh")
    assert read_hits(tmp_path / "fresh") == [{"a": 1}]


def test_append_adds_summary_rows_under_a_single_header(tmp_path):
    for seed in (0, 1, 2):
        grid_search(
            hit_when_seed_is_one,
            {"seed": [seed], "a": [1, 2, 3]},
            summary_axis="seed",
            append=True,
            run_dir=tmp_path,
        )
    text = (tmp_path / SUMMARY_FILE).read_text()
    assert text == "seed,tested,hits\n0,3,0\n1,3,2\n2,3,0\n"
    _, present, _ = merge_seed_rows([(text, "succeeded")], seed_column="seed")
    assert present == [0, 1, 2]


def test_append_refuses_a_summary_with_another_header_before_touching_anything(tmp_path):
    (tmp_path / SUMMARY_FILE).write_text("alpha,tested,hits\n0.5,3,1\n")
    (tmp_path / HITS_FILE).write_text('{"alpha": 0.5}\n')
    calls: list[int] = []

    with pytest.raises(ValueError, match="header"):
        grid_search(
            lambda seed: calls.append(seed) or True,
            {"seed": [0]},
            summary_axis="seed",
            append=True,
            run_dir=tmp_path,
        )
    assert calls == []
    assert (tmp_path / SUMMARY_FILE).read_text() == "alpha,tested,hits\n0.5,3,1\n"
    assert (tmp_path / HITS_FILE).read_text() == '{"alpha": 0.5}\n'


def test_append_writes_a_header_into_an_empty_summary(tmp_path):
    (tmp_path / SUMMARY_FILE).write_text("")
    grid_search(
        lambda seed: True, {"seed": [0]}, summary_axis="seed", append=True, run_dir=tmp_path
    )
    assert (tmp_path / SUMMARY_FILE).read_text() == "seed,tested,hits\n0,1,1\n"


# --- progress in metrics.jsonl -------------------------------------------------------------------


@pytest.fixture
def clock(monkeypatch) -> list[float]:
    """A wall clock the test moves by hand: ``clock[0] += 4.0`` is four seconds passing."""
    now = [0.0]
    monkeypatch.setattr(search, "_clock", lambda: now[0])
    return now


def progress(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    points = read_points(run_dir / METRICS_FILE)
    return {
        name: [(p["step"], p["value"]) for p in points if p["name"] == name]
        for name in ("search/tested", "search/hits")
    }


def test_progress_is_logged_at_most_once_per_interval_and_at_the_end(tmp_path, clock):
    assert search.PROGRESS_INTERVAL_S == 10.0

    def evaluate(a: int) -> bool:
        clock[0] += 4.0  # each answer takes four seconds
        return a % 3 == 0

    grid_search(evaluate, {"a": list(range(1, 11))}, run_dir=tmp_path)

    # t=4 (first answer), 16, 28 (each >= 10 s after the last), 40 (the end)
    assert progress(tmp_path) == {
        "search/tested": [(1, 1.0), (4, 4.0), (7, 7.0), (10, 10.0)],
        "search/hits": [(1, 0.0), (4, 1.0), (7, 2.0), (10, 3.0)],
    }


def test_a_fast_search_logs_its_first_answer_and_its_final_count(tmp_path, clock):
    grid_search(lambda a: a > 7, {"a": list(range(10))}, run_dir=tmp_path)
    assert progress(tmp_path) == {
        "search/tested": [(1, 1.0), (10, 10.0)],
        "search/hits": [(1, 0.0), (10, 2.0)],
    }


def test_the_final_count_is_not_logged_twice(tmp_path, clock):
    grid_search(lambda a: True, {"a": [1]}, run_dir=tmp_path)
    assert progress(tmp_path) == {"search/tested": [(1, 1.0)], "search/hits": [(1, 1.0)]}


def test_progress_steps_count_combinations_not_batches(tmp_path, clock):
    grid_search(
        lambda a: [x > 7 for x in a], {"a": list(range(10))}, batch_size=4, run_dir=tmp_path
    )
    assert progress(tmp_path)["search/tested"] == [(4, 4.0), (10, 10.0)]


def test_progress_defaults_to_the_run_directory(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("LAB_RUN_DIR", str(tmp_path))
    grid_search(lambda a: True, {"a": [1, 2]})
    assert progress(tmp_path)["search/hits"] == [(1, 1.0), (2, 2.0)]


def test_progress_is_added_to_the_metrics_already_there(tmp_path, clock):
    log_metric("loss", 0.5, 0, run_dir=tmp_path)
    grid_search(lambda a: True, {"a": [1]}, run_dir=tmp_path)
    assert [p["name"] for p in read_points(tmp_path / METRICS_FILE)] == [
        "loss",
        "search/tested",
        "search/hits",
    ]


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


def test_an_array_extra_is_logged_as_a_list(tmp_path):
    grid_search(
        lambda a: {"hit": True, "trace": np.array([1.0, 2.0])}, {"a": [1]}, run_dir=tmp_path
    )
    assert read_hits(tmp_path) == [{"a": 1, "trace": [1.0, 2.0]}]


@pytest.mark.parametrize("scalar", [np.array(0.5), np.array([0.5])])
def test_a_one_element_array_extra_is_logged_as_a_number(tmp_path, scalar):
    grid_search(lambda a: {"hit": True, "score": scalar}, {"a": [1]}, run_dir=tmp_path)
    assert read_hits(tmp_path) == [{"a": 1, "score": 0.5}]


def test_numpy_values_inside_a_list_extra_are_logged_as_plain_numbers(tmp_path):
    grid_search(
        lambda a: {"hit": True, "xs": [np.float32(1.5), np.int64(2)]}, {"a": [1]}, run_dir=tmp_path
    )
    assert read_hits(tmp_path) == [{"a": 1, "xs": [1.5, 2]}]


def test_a_batched_evaluator_can_return_an_array_per_combination(tmp_path):
    def evaluate(a: list[int]) -> dict[str, Any]:
        return {"hit": [True, True], "trace": np.array([[1.0, 2.0], [3.0, 4.0]])}

    grid_search(evaluate, {"a": [1, 2]}, batch_size=2, run_dir=tmp_path)
    assert read_hits(tmp_path) == [{"a": 1, "trace": [1.0, 2.0]}, {"a": 2, "trace": [3.0, 4.0]}]


def test_an_extra_that_cannot_be_logged_is_refused_naming_its_column(tmp_path):
    with pytest.raises(TypeError, match="'blob'"):
        grid_search(lambda a: {"hit": True, "blob": object()}, {"a": [1]}, run_dir=tmp_path)


def test_a_bad_extra_fails_at_the_first_answer_not_the_first_pass(tmp_path):
    calls: list[int] = []

    def evaluate(a: int) -> dict[str, Any]:
        calls.append(a)
        return {"hit": a == 3, "blob": object()}

    with pytest.raises(TypeError, match="'blob'"):
        grid_search(evaluate, {"a": [1, 2, 3]}, run_dir=tmp_path)
    assert calls == [1]


def test_an_extra_overwriting_a_parameter_fails_at_the_first_answer(tmp_path):
    calls: list[int] = []

    def evaluate(a: int) -> dict[str, Any]:
        calls.append(a)
        return {"hit": a == 3, "a": 99}

    with pytest.raises(ValueError, match="'a'"):
        grid_search(evaluate, {"a": [1, 2, 3]}, run_dir=tmp_path)
    assert calls == [1]


def test_a_grid_value_that_cannot_be_logged_is_refused_before_anything_runs(tmp_path):
    calls: list[Any] = []

    def evaluate(a: int, rule: Any) -> bool:
        calls.append(rule)
        return True

    with pytest.raises(TypeError, match="'rule'"):
        grid_search(evaluate, {"a": [1], "rule": ["stdp", object()]}, run_dir=tmp_path)
    assert calls == []
    assert not (tmp_path / HITS_FILE).exists()


# --- a batched and an unbatched evaluator mixed up -----------------------------------------------


@pytest.mark.parametrize("answers", [np.array([True, False]), [True, False]])
def test_many_answers_without_batch_size_says_to_pass_it(tmp_path, answers):
    with pytest.raises(TypeError, match="2 answers for one combination.*batch_size"):
        grid_search(lambda a: answers, {"a": [1, 2]}, run_dir=tmp_path)


@pytest.mark.parametrize("answer", [True, np.bool_(True), np.array(True)])
def test_one_answer_with_batch_size_says_to_return_one_per_combination(tmp_path, answer):
    with pytest.raises(TypeError, match="one answer per combination"):
        grid_search(lambda a: answer, {"a": [1, 2]}, batch_size=2, run_dir=tmp_path)


@pytest.mark.parametrize("column", [5, "abc"])
def test_one_extra_value_with_batch_size_is_refused_naming_its_column(tmp_path, column):
    with pytest.raises(TypeError, match="'label'.*one value per combination"):
        grid_search(
            lambda a: {"hit": [True] * 3, "label": column},
            {"a": [1, 2, 3]},
            batch_size=3,
            run_dir=tmp_path,
        )


def test_a_scalar_answer_to_a_batched_boundary_search_is_explained():
    with pytest.raises(TypeError, match="one answer per combination"):
        find_boundary(lambda x: True, "x", lo=0.0, hi=1.0, batched=True)


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
