# tests/test_aggregate_sweep.py
from __future__ import annotations

from pathlib import Path

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import JobState

from helpers import pin_in_flight, wait_terminal


def _lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    return Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)


def _write_shard_result(lab: Lab, job_id: str, seeds: list[int]) -> None:
    """Mark a shard succeeded and drop a results.csv with one row per seed."""
    out = lab.store.output_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["seed,acc"] + [f"{s},0.{s}" for s in seeds]
    (out / "results.csv").write_text("\n".join(lines) + "\n")
    lab.store.update_manifest(job_id, status=JobState.succeeded)


def test_aggregate_complete_cell(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    plan = lab.sweep_plan(sweep_id)
    cell = plan.cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    _write_shard_result(lab, cell.shard_job_ids[1], [2, 3])

    updated = lab.aggregate_sweep(sweep_id)
    c = updated.cells[0]
    assert c.status == "complete"
    assert c.seeds_present == [0, 1, 2, 3]
    assert c.missing_seeds == []
    agg = Path(c.aggregate_ref).read_text()
    assert agg == (
        "seed,acc,_shard_status\n0,0.0,succeeded\n1,0.1,succeeded\n"
        "2,0.2,succeeded\n3,0.3,succeeded\n"
    )


def test_aggregate_partial_failure_is_honest(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    plan = lab.sweep_plan(sweep_id)
    cell = plan.cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.failed)  # shard 2 dies

    updated = lab.aggregate_sweep(sweep_id)
    c = updated.cells[0]
    assert c.status == "incomplete"
    assert c.seeds_present == [0, 1]
    assert c.missing_seeds == [2, 3]
    assert Path(c.aggregate_ref).read_text() == (
        "seed,acc,_shard_status\n0,0.0,succeeded\n1,0.1,succeeded\n"
    )


def test_aggregate_is_idempotent_and_resumable(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    first = lab.aggregate_sweep(sweep_id).cells[0]
    assert first.status == "incomplete" and first.missing_seeds == [2, 3]
    _write_shard_result(lab, cell.shard_job_ids[1], [2, 3])  # second shard finishes later
    second = lab.aggregate_sweep(sweep_id).cells[0]
    assert second.status == "complete" and second.seeds_present == [0, 1, 2, 3]


def _write_partial_result(lab: Lab, job_id: str, seeds: list[int], status: JobState) -> None:
    """A shard that reached a non-succeeded terminal state but left rows on disk
    (heartbeat-rsynced partial output)."""
    out = lab.store.output_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["seed,acc"] + [f"{s},0.{s}" for s in seeds]
    (out / "results.csv").write_text("\n".join(lines) + "\n")
    lab.store.update_manifest(job_id, status=status)


def test_aggregate_includes_timed_out_shard_rows_by_default(tmp_path: Path):
    """Field-report #2: rows the user already paid for must not be discarded."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    _write_partial_result(lab, cell.shard_job_ids[1], [2], JobState.timed_out)  # 1 of 2 seeds

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.seeds_present == [0, 1, 2]
    assert c.seeds_partial == [2]
    assert c.missing_seeds == [3]
    assert c.status == "incomplete"
    agg = Path(c.aggregate_ref).read_text()
    assert "2,0.2,timed_out" in agg and "0,0.0,succeeded" in agg


def test_aggregate_strict_excludes_partial_shards(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    _write_partial_result(lab, cell.shard_job_ids[1], [2], JobState.timed_out)

    c = lab.aggregate_sweep(sweep_id, include_partial=False).cells[0]
    assert c.seeds_present == [0, 1]
    assert c.seeds_partial == []
    assert c.missing_seeds == [2, 3]


def test_aggregate_excludes_running_shard(tmp_path: Path):
    """A running shard's file is still moving under the heartbeat — never read it."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-1", shard_size=1)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0])
    out = lab.store.output_dir(cell.shard_job_ids[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.csv").write_text("seed,acc\n1,0.1\n")
    wait_terminal(lab.backend, cell.shard_job_ids[1])
    pin_in_flight(lab.store, cell.shard_job_ids[1])

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.seeds_present == [0] and c.missing_seeds == [1]


def test_aggregate_excludes_unconsumed_config_shard(tmp_path: Path):
    """Cross-fix guard via the REAL path: a timed-out shard whose effective_config.json shows
    it never consumed a passed override has its rows excluded (wrong-config rows are what the
    fail-closed check exists to kill; the audit records on every terminal transition)."""
    import json as _json

    lab = _lab(tmp_path)
    # `N` is a grid key, i.e. an override the lab injects — the only kind the audit judges (a
    # `k=v` typed into the base command is indistinguishable from an env prefix or a
    # `--with pkg==x` spec, which is how real successes used to be flipped to failed).
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-1", shard_size=1)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0])
    # shard 2 wrote rows + an effective config that consumed only `seeds` — the injected
    # `N` went unconsumed. It then timed out; the terminal audit records unconsumed_config.
    out = lab.store.output_dir(cell.shard_job_ids[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.csv").write_text("seed,acc\n1,0.1\n")
    (out / "effective_config.json").write_text(_json.dumps({"seeds": "1"}))
    m = lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.timed_out)
    assert m.unconsumed_config == ["N"]  # recorded without flipping the timed_out verdict
    assert m.status is JobState.timed_out

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.seeds_present == [0] and c.missing_seeds == [1]


def test_old_plan_json_reads_with_default_seeds_partial(tmp_path: Path):
    """plan.json written before seeds_partial existed must still read (backward compat)."""
    import json as _json

    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-1", shard_size=1)
    p = lab.store.sweep_plan_path(sweep_id)
    raw = _json.loads(p.read_text())
    for c in raw["cells"]:
        c.pop("seeds_partial", None)
    p.write_text(_json.dumps(raw))
    plan = lab.sweep_plan(sweep_id)
    assert plan.cells[0].seeds_partial == []


def test_sweep_row_key_flows_to_aggregation(tmp_path: Path):
    """--row-key seed,alpha: multi-row-per-seed shard results aggregate instead of raising
    (verification report 2026-08-06 §2 — the layout behind the hand-stitched headline data)."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep(
        "true", {"N": [1000]}, seeds="0-1", shard_size=1, row_key="seed,alpha"
    )
    plan = lab.sweep_plan(sweep_id)
    cell = plan.cells[0]
    assert cell.row_key == ["seed", "alpha"]
    for i, jid in enumerate(cell.shard_job_ids):
        out = lab.store.output_dir(jid)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.csv").write_text(
            f"seed,alpha,acc\n{i},2.7,0.9\n{i},2.72,0.8\n"
        )
        lab.store.update_manifest(jid, status=JobState.succeeded)

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.status == "complete"
    assert c.seeds_present == [0, 1]
    agg = Path(c.aggregate_ref).read_text()
    assert agg.count("\n") == 5  # header + 4 rows (2 seeds x 2 alphas)


def test_sweep_row_key_must_contain_seed_column(tmp_path: Path):
    lab = _lab(tmp_path)
    import pytest as _pytest

    from lab.core import LabError

    with _pytest.raises(LabError, match="seed"):
        lab.sweep("true", {"N": [1]}, seeds="0-1", shard_size=1, row_key="alpha")


def test_aggregate_row_key_override_for_pre_existing_plans(tmp_path: Path):
    """Sweeps created before --row-key existed (plan has none) can declare it at AGGREGATE
    time — the real 2026-08-04 headline sweep predates v0.2.1 and must not need plan.json
    hand-edits. The override persists onto the plan for future aggregates/retries."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-1", shard_size=1)  # no row_key
    cell = lab.sweep_plan(sweep_id).cells[0]
    for i, jid in enumerate(cell.shard_job_ids):
        out = lab.store.output_dir(jid)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.csv").write_text(f"seed,alpha,acc\n{i},2.7,0.9\n{i},2.72,0.8\n")
        lab.store.update_manifest(jid, status=JobState.timed_out)

    import pytest as _pytest

    with _pytest.raises(ValueError, match="duplicate"):
        lab.aggregate_sweep(sweep_id)  # default seed-only key still rejects the layout

    c = lab.aggregate_sweep(sweep_id, row_key="seed,alpha").cells[0]
    assert c.seeds_present == [0, 1] and c.seeds_partial == [0, 1]
    assert Path(c.aggregate_ref).read_text().count("\n") == 5  # header + 4 rows

    # Persisted: a later aggregate WITHOUT the override keeps working.
    c2 = lab.aggregate_sweep(sweep_id).cells[0]
    assert c2.seeds_present == [0, 1]
    assert lab.sweep_plan(sweep_id).cells[0].row_key == ["seed", "alpha"]


def _write_hits(lab: Lab, job_id: str, text: str) -> None:
    out = lab.store.output_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hits.jsonl").write_text(text)


def _hits(path: Path) -> list[dict]:
    import json as _json

    return [_json.loads(line) for line in path.read_text().splitlines()]


def test_aggregate_concatenates_shard_hits_beside_the_csv(tmp_path: Path):
    """lab.search writes hits.jsonl per shard; the cell gets one concatenated hits.jsonl next
    to its results.csv aggregate, every hit stamped with the shard's terminal state."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_hits(lab, cell.shard_job_ids[0], '{"seed": 0, "g": 1}\n{"seed": 1, "g": 1}\n')
    _write_hits(lab, cell.shard_job_ids[1], '{"seed": 3, "g": 2}\n')
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    _write_partial_result(lab, cell.shard_job_ids[1], [2, 3], JobState.timed_out)

    c = lab.aggregate_sweep(sweep_id).cells[0]
    hits_path = Path(c.aggregate_ref).parent / "hits.jsonl"
    assert c.hits_ref == str(hits_path)
    assert _hits(hits_path) == [
        {"seed": 0, "g": 1, "_shard_status": "succeeded"},
        {"seed": 1, "g": 1, "_shard_status": "succeeded"},
        {"seed": 3, "g": 2, "_shard_status": "timed_out"},
    ]
    assert c.hits == 3
    view = lab.sweep_plan(sweep_id).view()["cells"][0]  # persisted, and in the shells' view
    assert view["hits"] == 3 and view["hits_ref"] == str(hits_path)


def test_aggregate_hits_follow_the_csv_inclusion_rules(tmp_path: Path):
    """Strict drops partial shards' hits; running shards and unconsumed-config shards are never
    read — the same three rules the CSV rows obey."""
    import json as _json

    lab = _lab(tmp_path)
    # `N` is a grid key, i.e. an override the lab injects: the only kind the config audit judges.
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=1)
    ok, partial, running, wrong_config = lab.sweep_plan(sweep_id).cells[0].shard_job_ids
    for jid, seed in zip((ok, partial, running, wrong_config), range(4)):
        _write_hits(lab, jid, f'{{"seed": {seed}}}\n')
        (lab.store.output_dir(jid) / "effective_config.json").write_text(
            _json.dumps({"seeds": str(seed), "N": 1000})
        )
    _write_shard_result(lab, ok, [0])
    _write_partial_result(lab, partial, [1], JobState.failed)
    lab.store.update_manifest(running, status=JobState.running)
    (lab.store.output_dir(wrong_config) / "effective_config.json").write_text(
        _json.dumps({"seeds": "3"})  # never consumed N
    )
    assert lab.store.update_manifest(wrong_config, status=JobState.timed_out).unconsumed_config

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert [h["seed"] for h in _hits(Path(c.hits_ref))] == [0, 1]

    c = lab.aggregate_sweep(sweep_id, include_partial=False).cells[0]
    assert _hits(Path(c.hits_ref)) == [{"seed": 0, "_shard_status": "succeeded"}]
    assert c.hits == 1


def test_aggregate_without_any_hits_file_writes_none_and_clears_a_stale_one(tmp_path: Path):
    """No included shard searched -> no cell hits.jsonl and hits=None. A hits aggregate left by
    an earlier, wider aggregate (here: non-strict) is removed rather than left to be exported."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-1", shard_size=1)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0])
    _write_hits(lab, cell.shard_job_ids[1], '{"seed": 1}\n')
    _write_partial_result(lab, cell.shard_job_ids[1], [1], JobState.timed_out)

    c = lab.aggregate_sweep(sweep_id).cells[0]
    stale = Path(c.hits_ref)
    assert stale.exists()

    c = lab.aggregate_sweep(sweep_id, include_partial=False).cells[0]
    assert c.hits is None and c.hits_ref is None
    assert not stale.exists()
    assert lab.sweep_plan(sweep_id).view()["cells"][0]["hits"] is None


def test_aggregate_empty_hits_file_means_searched_and_nothing_passed(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0", shard_size=1)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_hits(lab, cell.shard_job_ids[0], "")
    _write_shard_result(lab, cell.shard_job_ids[0], [0])

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.hits == 0
    assert Path(c.hits_ref).read_text() == ""


def test_aggregate_hits_never_count_a_seed_as_present(tmp_path: Path):
    """Seed presence is judged by results.csv alone: a shard that left hits but no result rows
    still has its seeds missing, so sweep-retry buys them back — and the missing seed's hits are
    left out of the cell file, which only covers the seeds results.csv lists as present."""
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-1", shard_size=1)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0])
    _write_hits(lab, cell.shard_job_ids[1], '{"seed": 1, "g": 1}\n')
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.timed_out)

    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.seeds_present == [0] and c.missing_seeds == [1] and c.status == "incomplete"
    assert c.hits == 0  # searched (the file exists), but seed 1's search never finished


def test_aggregate_succeeded_shard_with_torn_hits_fails_loud(tmp_path: Path):
    import pytest as _pytest

    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0", shard_size=1)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_hits(lab, cell.shard_job_ids[0], '{"seed": 0}\n{"se')
    _write_shard_result(lab, cell.shard_job_ids[0], [0])
    with _pytest.raises(ValueError, match="succeeded"):
        lab.aggregate_sweep(sweep_id)


def test_aggregate_mirrors_cell_hits_to_r2_best_effort(tmp_path: Path, monkeypatch):
    """Same best-effort mirror as the CSV aggregate, under the cell's own prefix; a failed
    upload keeps the local copy and the aggregate still completes."""
    import lab.core as core_mod

    uploads: list[str] = []
    fail = {"on": False}

    class _FakeR2:
        @classmethod
        def from_env(cls) -> "_FakeR2":
            return cls()

        def upload_file(self, local: Path, key: str) -> None:
            if fail["on"]:
                raise RuntimeError("r2 boom")
            assert Path(local).exists()
            uploads.append(key)

        def download_dir(self, prefix: str, dest: Path) -> None:
            pass

    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0", shard_size=1)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_hits(lab, cell.shard_job_ids[0], '{"seed": 0}\n')
    _write_shard_result(lab, cell.shard_job_ids[0], [0])
    monkeypatch.setattr(core_mod, "r2_enabled", lambda: True)
    monkeypatch.setattr(core_mod, "R2Store", _FakeR2)

    lab.aggregate_sweep(sweep_id)
    prefix = f"{sweep_id}/cells/{cell.cell_id}"
    assert f"{prefix}/hits.jsonl" in uploads and f"{prefix}/results.csv" in uploads

    fail["on"] = True
    c = lab.aggregate_sweep(sweep_id).cells[0]
    assert c.hits == 1 and Path(c.hits_ref).exists()


def test_old_plan_json_without_hits_fields_still_reads(tmp_path: Path):
    import json as _json

    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-1", shard_size=1)
    p = lab.store.sweep_plan_path(sweep_id)
    raw = _json.loads(p.read_text())
    for c in raw["cells"]:
        c.pop("hits", None)
        c.pop("hits_ref", None)
    p.write_text(_json.dumps(raw))
    cell = lab.sweep_plan(sweep_id).cells[0]
    assert cell.hits is None and cell.hits_ref is None


def test_aggregate_row_key_override_must_contain_seed_column(tmp_path: Path):
    import pytest as _pytest

    from lab.core import LabError

    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1]}, seeds="0-1", shard_size=1)
    with _pytest.raises(LabError, match="seed"):
        lab.aggregate_sweep(sweep_id, row_key="alpha")
