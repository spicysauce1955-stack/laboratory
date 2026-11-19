"""A cell's hits.jsonl only carries hits for seeds its results.csv lists as present.

Stress run 2026-11-19: an incomplete cell's hits.jsonl held hits for seeds the CSV named as
missing — a partial shard had flushed some of a seed's hits and died before writing that seed's
result row. The two cell files then disagreed about which seeds the cell covers, and a reader of
hits.jsonl alone had no way to tell those hits came from a search that never finished.

Decision: a hit that names its seed (the cell's ``seed_column``) is kept only when that seed is
present. The shard's own hits.jsonl is untouched (and still exported under its job), and
``sweep-retry`` re-runs the missing seed, whose hits then land with its row. A hit that does not
name a seed, or names one that is not an integer, cannot be judged and is kept. The filter runs
*before* per-key ownership is decided, so a dropped hit can never displace a kept one.
"""

from __future__ import annotations

import json
from pathlib import Path

from lab.aggregate import merge_hits
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import JobState

from helpers import wait_terminal


def _jl(*objs: dict) -> str:
    return "".join(json.dumps(o) + "\n" for o in objs)


def _parse(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines()]


def test_hits_for_absent_seeds_are_dropped_and_the_rest_kept() -> None:
    text = _jl(
        {"seed": 0, "g": 1},
        {"seed": 1, "g": 1},
        {"seed": "0", "g": 2},  # a string seed still names seed 0
        {"seed": "1", "g": 2},
        {"g": 3},  # names no seed: cannot be judged, kept
        {"seed": "x", "g": 4},  # not an integer seed: cannot be judged, kept
    )
    merged, n = merge_hits([(text, "timed_out")], seed_column="seed", present_seeds={0})
    assert [h["g"] for h in _parse(merged)] == [1, 2, 3, 4]
    assert n == 4


def test_without_present_seeds_nothing_is_filtered() -> None:
    """The default is the old contract, unchanged."""
    text = _jl({"seed": 0}, {"seed": 1})
    assert merge_hits([(text, "failed")]) == merge_hits([(text, "failed")], seed_column="seed")
    assert merge_hits([(text, "failed")])[1] == 2


def test_the_filter_runs_before_ownership_so_a_dropped_hit_displaces_nothing() -> None:
    """Key ``g`` is owned per shard. Shard B (later, equal precedence) would own g=1 — but its
    only g=1 hit is for an absent seed. Filtering first lets shard A's present-seed hit stand,
    instead of B winning the key and then losing its hit, which would leave g=1 with nothing."""
    a = _jl({"g": 1, "seed": 0})
    b = _jl({"g": 1, "seed": 1})
    merged, n = merge_hits(
        [(a, "timed_out"), (b, "timed_out")], row_key=["g"], seed_column="seed",
        present_seeds={0},
    )
    assert _parse(merged) == [{"g": 1, "seed": 0, "_shard_status": "timed_out"}]
    assert n == 1


def test_an_incomplete_cells_hits_agree_with_its_csv(tmp_path: Path) -> None:
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1]}, seeds="0-1", shard_size=2)
    (jid,) = lab.sweep_plan(sweep_id).cells[0].shard_job_ids
    wait_terminal(lab.backend, jid)
    out = lab.store.output_dir(jid)
    # Seed 0 finished (row + hits); seed 1 flushed hits and died before its row.
    (out / "results.csv").write_text("seed,acc\n0,0.5\n")
    (out / "hits.jsonl").write_text(_jl({"seed": 0, "g": 1}, {"seed": 1, "g": 1}))
    lab.store.update_manifest(jid, status=JobState.timed_out)

    c = lab.aggregate_sweep(sweep_id).cells[0]

    assert c.seeds_present == [0] and c.missing_seeds == [1]
    assert [h["seed"] for h in _parse(Path(c.hits_ref).read_text())] == [0]
    assert c.hits == 1
    assert len(_parse((out / "hits.jsonl").read_text())) == 2  # the shard's own file is untouched
