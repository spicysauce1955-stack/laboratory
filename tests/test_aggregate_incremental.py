"""`sweep-aggregate` is cheap to repeat, and its hits output is compact.

Stress run 2026-11-19: one cell of 866k hits took 7.8 s and 1.44 GB on *every* call, and wrote
91.5 MB from 65 MB of input. Three changes: hits are written as compact JSON; the merge keeps one
serialized line per hit instead of a parsed object; and a cell whose included shards' inputs
have not changed since the last aggregate (a fingerprint of each shard's id, state and files'
size/mtime, plus everything else that shapes the output) is not re-read or re-written at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import lab.core as core_mod
from lab.aggregate import merge_hits
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import JobState

from helpers import wait_terminal


def test_merged_hits_are_compact_json() -> None:
    merged, n = merge_hits([('{"seed": 0, "g": 1.5}\n{"seed": 1, "g": [1, 2]}\n', "succeeded")])
    assert merged == (
        '{"seed":0,"g":1.5,"_shard_status":"succeeded"}\n'
        '{"seed":1,"g":[1,2],"_shard_status":"succeeded"}\n'
    )
    assert n == 2


def test_merge_hits_accepts_any_iterable_of_shards() -> None:
    """The core feeds shard texts lazily, so each file's text can be freed once parsed."""
    shards = [('{"g": 1}\n', "succeeded"), ('{"g": 2}\n', "failed")]
    assert merge_hits(iter(shards)) == merge_hits(shards)


def _lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    return Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)


def _aggregated(tmp_path: Path) -> tuple[Lab, str, list[str]]:
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1]}, seeds="0-3", shard_size=2)
    ids = list(lab.sweep_plan(sweep_id).cells[0].shard_job_ids)
    for jid, seeds in zip(ids, ([0, 1], [2, 3])):
        wait_terminal(lab.backend, jid)
        out = lab.store.output_dir(jid)
        (out / "results.csv").write_text("seed,acc\n" + "".join(f"{s},0.{s}\n" for s in seeds))
        (out / "hits.jsonl").write_text("".join(f'{{"seed": {s}, "g": 1}}\n' for s in seeds))
    lab.store.update_manifest(ids[1], status=JobState.timed_out)
    lab.aggregate_sweep(sweep_id)
    return lab, sweep_id, ids


def _aggregated_then_watch(tmp_path: Path, merges: list[str]) -> tuple[Lab, str, list[str]]:
    lab, sweep_id, ids = _aggregated(tmp_path)
    merges.clear()  # forget the first aggregate's merges: only what follows is under test
    return lab, sweep_id, ids


@pytest.fixture
def merges(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every merge the core runs (by name), still running the real ones."""
    calls: list[str] = []
    real_rows, real_hits = core_mod.merge_seed_rows, core_mod.merge_hits

    def rows(*a, **kw):  # type: ignore[no-untyped-def]
        calls.append("rows")
        return real_rows(*a, **kw)

    def hits(*a, **kw):  # type: ignore[no-untyped-def]
        calls.append("hits")
        return real_hits(*a, **kw)

    monkeypatch.setattr(core_mod, "merge_seed_rows", rows)
    monkeypatch.setattr(core_mod, "merge_hits", hits)
    return calls


def _cell_files(lab: Lab, sweep_id: str) -> dict[str, tuple[int, str]]:
    c = lab.sweep_plan(sweep_id).cells[0]
    paths = [Path(c.aggregate_ref)] + ([Path(c.hits_ref)] if c.hits_ref else [])
    return {p.name: (p.stat().st_mtime_ns, p.read_text()) for p in paths}


def test_an_unchanged_cell_is_neither_re_merged_nor_re_written(tmp_path: Path, merges) -> None:
    lab, sweep_id, _ = _aggregated_then_watch(tmp_path, merges)
    before_view = lab.sweep_plan(sweep_id).view()
    before_files = _cell_files(lab, sweep_id)

    again = lab.aggregate_sweep(sweep_id)

    assert merges == []
    assert again.view() == before_view
    assert _cell_files(lab, sweep_id) == before_files


def test_a_changed_shard_file_is_re_merged(tmp_path: Path, merges) -> None:
    lab, sweep_id, ids = _aggregated_then_watch(tmp_path, merges)
    hits = lab.store.output_dir(ids[1]) / "hits.jsonl"
    with hits.open("a") as f:
        f.write('{"seed": 3, "g": 2}\n')

    c = lab.aggregate_sweep(sweep_id).cells[0]

    assert merges == ["rows", "hits"]
    assert c.hits == 5


def test_a_shard_changing_state_is_re_merged(tmp_path: Path, merges) -> None:
    lab, sweep_id, ids = _aggregated_then_watch(tmp_path, merges)
    lab.store.update_manifest(ids[1], status=JobState.succeeded)  # terminal -> terminal: allowed
    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert merges == ["rows", "hits"]
    assert c.seeds_partial == []


def test_strict_and_row_key_are_part_of_the_fingerprint(tmp_path: Path, merges) -> None:
    lab, sweep_id, _ = _aggregated_then_watch(tmp_path, merges)
    assert lab.aggregate_sweep(sweep_id, include_partial=False).cells[0].seeds_present == [0, 1]
    assert merges == ["rows", "hits"]
    merges.clear()
    lab.aggregate_sweep(sweep_id, include_partial=False)
    assert merges == []
    lab.aggregate_sweep(sweep_id, row_key="seed,acc")
    assert merges == ["rows", "hits"]


def test_a_deleted_cell_output_is_rewritten(tmp_path: Path, merges) -> None:
    lab, sweep_id, _ = _aggregated_then_watch(tmp_path, merges)
    c = lab.sweep_plan(sweep_id).cells[0]
    os.unlink(c.hits_ref)
    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert merges == ["rows", "hits"]
    assert Path(c.hits_ref).exists()


def test_a_plan_from_before_fingerprints_is_merged_once_then_skipped(tmp_path: Path, merges) -> None:
    lab, sweep_id, _ = _aggregated_then_watch(tmp_path, merges)
    plan = lab.sweep_plan(sweep_id)
    plan.cells[0].inputs_fingerprint = None
    lab.store.write_sweep_plan(plan)
    lab.aggregate_sweep(sweep_id)
    assert merges == ["rows", "hits"]
    merges.clear()
    lab.aggregate_sweep(sweep_id)
    assert merges == []
