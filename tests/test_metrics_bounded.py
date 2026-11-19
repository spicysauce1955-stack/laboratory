"""`lab metrics` / `mcp__lab__metrics` are bounded by default and read the file from the end.

The 2026-11 stress run (one 500k-point series per name, 2M lines, 154 MB) measured:

* MCP `metrics` with no `since_step`: **27.7 MB / 17 s** into an agent's context; CLI `lab metrics`
  printed **199 MB** at **2.0 GB RSS** in 7.9 s.
* the early-kill poll, `--since-step <last>`: 3.7 KB of answer for **2.4 s** of rescanning the
  whole file, every poll.

So the series are capped (`max_points`, the latest points of each series, with what was cut
reported under `truncated`), `downsample` spreads the cap over the whole run instead, and the
reader walks the file backwards in blocks, skipping whole blocks whose every line belongs to a
series it has finished with. `max_points=0` is the explicit opt-out: everything, as before.

The old forward reader (`group_series(read_points(...))`) is kept as the oracle: the bounded
reader must agree with it wherever their contracts overlap.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

import lab.metrics as metrics_mod
from lab.metrics import (
    DEFAULT_MAX_POINTS,
    METRICS_FILE,
    group_series,
    read_points,
    series_view,
)


def _write(path: Path, lines: list[Any], *, tail: str = "") -> Path:
    f = path / METRICS_FILE
    f.write_text("".join((x if isinstance(x, str) else json.dumps(x)) + "\n" for x in lines) + tail)
    return f


def _oracle(f: Path, names: list[str] | None = None, since_step: int | None = None):
    return group_series(read_points(f, names=names, since_step=since_step))


def _training_log(n_steps: int, names: tuple[str, ...] = ("loss", "acc", "lr", "grad_norm")):
    return [
        {"name": nm, "value": (i * 7 + j) % 1000 / 1000, "step": i, "wall_time": 1.0 + i}
        for i in range(n_steps)
        for j, nm in enumerate(names)
    ]


# --- the cap --------------------------------------------------------------------------------


def test_default_keeps_the_latest_points_of_each_series_and_says_what_it_cut(tmp_path: Path):
    f = _write(tmp_path, _training_log(1000))

    view = series_view(f)

    assert view["max_points"] == DEFAULT_MAX_POINTS
    full = _oracle(f)
    assert list(view["series"]) == list(full)  # first-appearance order, as before
    for name, points in view["series"].items():
        assert points == full[name][-DEFAULT_MAX_POINTS:]  # the newest, in file order
        assert view["truncated"][name] == {
            "returned": DEFAULT_MAX_POINTS,
            "omitted": 1000 - DEFAULT_MAX_POINTS,
        }


def test_nothing_is_reported_truncated_when_nothing_was_cut(tmp_path: Path):
    f = _write(tmp_path, _training_log(10))

    view = series_view(f)

    assert view["series"] == _oracle(f)
    assert view["truncated"] == {}


def test_max_points_zero_is_the_explicit_opt_out(tmp_path: Path):
    f = _write(tmp_path, _training_log(500))

    view = series_view(f, max_points=0)

    assert view["series"] == _oracle(f)
    assert view["truncated"] == {} and view["max_points"] == 0


def test_a_negative_cap_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="max_points"):
        series_view(_write(tmp_path, []), max_points=-1)


def test_downsample_spreads_the_cap_over_the_whole_run_and_keeps_the_newest(tmp_path: Path):
    f = _write(tmp_path, _training_log(10_000, names=("loss",)))

    view = series_view(f, max_points=50, downsample=True)

    steps = [p["step"] for p in view["series"]["loss"]]
    assert len(steps) == 50
    assert steps == sorted(steps)
    assert steps[-1] == 9_999  # the newest point always survives
    assert steps[0] < 500  # ...and the early run is represented, not just the tail
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    assert max(gaps) <= 2 * min(gaps) + 1  # evenly spread, not clumped
    assert view["truncated"]["loss"] == {"returned": 50, "omitted": 9_950}


def test_downsample_under_the_cap_returns_everything(tmp_path: Path):
    f = _write(tmp_path, _training_log(30, names=("loss",)))

    view = series_view(f, max_points=50, downsample=True)

    assert view["series"] == _oracle(f) and view["truncated"] == {}


# --- equivalence with the old reader --------------------------------------------------------


def _random_file(rng: random.Random, path: Path) -> Path:
    """Independent per-series step counters (an epoch series beside a global-step one, so the
    file is NOT globally monotone), junk lines of every shape the reader must skip, duplicate
    steps, and a half-written tail."""
    counters = {"loss": 0, "acc": 0, "epoch_acc": 0, "lr": 0}
    lines: list[Any] = []
    for _ in range(rng.randint(0, 400)):
        roll = rng.random()
        if roll < 0.05:
            lines.append(rng.choice(["7", "[1, 2]", '"x"', "null", "", "{not json", "   "]))
            continue
        if roll < 0.08:
            lines.append({"name": "loss", "value": 1, "step": rng.choice(["late", True, None])})
            continue
        if roll < 0.10:
            lines.append({"name": {"un": "hashable"}, "value": 1, "step": 3})
            continue
        name = rng.choice(list(counters))
        counters[name] += rng.choice([0, 1, 1, 2])  # non-decreasing, duplicates allowed
        value: Any = rng.choice([rng.random(), float("nan"), "label", None])
        lines.append({"name": name, "value": value, "step": counters[name], "wall_time": 1.0})
    return _write(path, lines, tail=rng.choice(["", '{"name": "loss", "value": 0.1, "st']))


@pytest.mark.parametrize("seed", range(40))
def test_agrees_with_the_forward_reader(tmp_path: Path, monkeypatch, seed: int):
    """Tiny blocks force lines to straddle block boundaries on every read."""
    monkeypatch.setattr(metrics_mod, "_BLOCK", random.Random(seed).choice([7, 64, 333, 4096]))
    rng = random.Random(seed)
    f = _random_file(rng, tmp_path)

    for names in (None, ["loss"], ["acc", "epoch_acc"], ["absent"]):
        for since in (None, 0, 3, rng.randint(0, 60)):
            full = _oracle(f, names, since)
            everything = series_view(f, names, since, max_points=0)["series"]
            assert _nan_safe(everything) == _nan_safe(full)
            assert list(everything) == list(full)
            k = rng.randint(1, 12)
            view = series_view(f, names, since, max_points=k)
            # Same series; their order can differ only via a malformed line the block-skip
            # counted without parsing (the deterministic tests above pin order on clean files).
            assert sorted(view["series"]) == sorted(full)
            for name, points in view["series"].items():
                assert _nan_safe({"_": points}) == _nan_safe({"_": full[name][-k:]})
                if len(full[name]) > k:
                    # Counted without parsing when skipped by the block reader, so a malformed
                    # line naming the series can count too — never fewer than were cut.
                    assert view["truncated"][name]["omitted"] >= len(full[name]) - k
                else:
                    assert name not in view["truncated"]


def _nan_safe(series: dict[str, list[dict[str, Any]]]) -> str:
    return json.dumps(series, sort_keys=True)  # NaN != NaN; compare the serialised form


# --- since_step: the early-kill poll --------------------------------------------------------


class _CountingJson:
    def __init__(self) -> None:
        self.calls = 0

    def loads(self, s: Any, **kw: Any) -> Any:
        self.calls += 1
        return json.loads(s, **kw)

    JSONDecodeError = json.JSONDecodeError


def test_a_since_step_poll_parses_only_the_tail(tmp_path: Path, monkeypatch):
    n = 50_000
    f = _write(tmp_path, _training_log(n))
    counting = _CountingJson()
    monkeypatch.setattr(metrics_mod, "json", counting)

    view = series_view(f, since_step=n - 4)

    assert {name: [p["step"] for p in pts] for name, pts in view["series"].items()} == {
        nm: [n - 3, n - 2, n - 1] for nm in ("loss", "acc", "lr", "grad_norm")
    }
    assert counting.calls < 2_000, f"parsed {counting.calls} of {4 * n} lines for a tail poll"


def test_a_named_since_step_poll_stops_reading_once_the_series_is_settled(tmp_path, monkeypatch):
    n = 50_000
    f = _write(tmp_path, _training_log(n))
    reads: list[int] = []
    real_open = Path.open

    class _Spy:
        def __init__(self, fh: Any) -> None:
            self._fh = fh

        def __getattr__(self, name: str) -> Any:
            return getattr(self._fh, name)

        def __enter__(self) -> _Spy:
            return self

        def __exit__(self, *exc: Any) -> None:
            self._fh.close()

        def read(self, size: int = -1) -> Any:
            data = self._fh.read(size)
            reads.append(len(data))
            return data

    monkeypatch.setattr(Path, "open", lambda self, *a, **kw: _Spy(real_open(self, *a, **kw)))

    view = series_view(f, names=["loss"], since_step=n - 3)

    assert [p["step"] for p in view["series"]["loss"]] == [n - 2, n - 1]
    assert sum(reads) <= 2 * metrics_mod._BLOCK < f.stat().st_size


def test_a_series_seen_going_backwards_is_read_in_full(tmp_path: Path):
    """The reverse reader stops a series at its first step <= since_step, which is only right if
    that series' steps never go backwards. Where it can see them go backwards, it must not stop."""
    lines = [{"name": "loss", "value": 0, "step": s} for s in (1, 2, 3, 10, 11, 5, 12)]
    f = _write(tmp_path, lines)

    assert series_view(f, since_step=4)["series"] == _oracle(f, since_step=4)
    assert [p["step"] for p in series_view(f, since_step=4)["series"]["loss"]] == [10, 11, 5, 12]


def test_default_view_skips_finished_series_without_parsing(tmp_path: Path, monkeypatch):
    n = 50_000
    f = _write(tmp_path, _training_log(n))
    counting = _CountingJson()
    monkeypatch.setattr(metrics_mod, "json", counting)

    view = series_view(f)

    assert all(len(p) == DEFAULT_MAX_POINTS for p in view["series"].values())
    assert view["truncated"]["loss"] == {"returned": DEFAULT_MAX_POINTS, "omitted": n - 200}
    assert counting.calls < 0.05 * 4 * n, f"parsed {counting.calls} of {4 * n} lines"


def test_memory_is_bounded_by_the_answer_not_the_file(tmp_path: Path, monkeypatch):
    """Peak memory is a few blocks plus the kept points, whatever the file's size — shrink the
    block so a test-sized file is many blocks long."""
    import tracemalloc

    monkeypatch.setattr(metrics_mod, "_BLOCK", 1 << 16)
    f = _write(tmp_path, _training_log(60_000))
    size = f.stat().st_size
    tracemalloc.start()
    try:
        series_view(f)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < size / 4, f"peak {peak} bytes for a {size}-byte file"


def test_a_missing_file_is_an_empty_view(tmp_path: Path):
    assert series_view(tmp_path / METRICS_FILE) == {
        "series": {},
        "max_points": DEFAULT_MAX_POINTS,
        "truncated": {},
    }
