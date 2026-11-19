"""Every read-modify-write of a sweep's plan.json is serialized by a per-sweep lock.

Stress run 2026-11-19: eight concurrent `lab sweep-retry <sid>` on one sweep with missing seeds
each read the plan, each saw no in-flight retry, each submitted a shard, and each wrote its own
copy of the plan back — 8 retry jobs launched, 1 recorded in plan.json. The other 7 were real
(and on a remote backend, paid) shards that `sweep-aggregate` never reads and `sweep-retry` never
sees as in flight. The same lost update hits a `sweep-aggregate` whose read predates a retry's
write. The fix holds an exclusive per-sweep flock across read -> submit -> write, so the second
retry waits, then sees the first one's in-flight shard and buys nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from core_fixtures import local_lab, make_repo
from helpers import PYTHON, wait_terminal

import lab.core as core_mod
from lab.models import JobState

# Exits at once unless `<hold>` exists, then sleeps: original shards finish immediately, while the
# retries launched after the hold file appears stay genuinely in flight (a live runner and all).
_SHARD = """\
import os, sys, time
hold = sys.argv[1]
if os.path.exists(hold):
    time.sleep(60)
"""


def _sweep_with_a_failed_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = make_repo(tmp_path, {"shard.py": _SHARD})
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    hold = tmp_path / "hold"
    sweep_id, _ = lab.sweep(f"{PYTHON} shard.py {hold}", {}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    for jid in cell.shard_job_ids:
        wait_terminal(lab.backend, jid)
    # Shard [0,1] landed its rows; shard [2,3] failed having written nothing.
    out = lab.store.output_dir(cell.shard_job_ids[0])
    (out / "results.csv").write_text("seed,acc\n0,0.0\n1,0.1\n")
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.failed)
    hold.write_text("x")
    return repo, lab, sweep_id, list(cell.shard_job_ids)


def _sweep_job_ids(lab, sweep_id: str) -> set[str]:
    return {m.job_id for m in lab.list_jobs() if m.sweep_id == sweep_id}


def _cancel_all(lab, ids) -> None:
    for jid in ids:
        try:
            lab.cancel(jid)
        except Exception:  # noqa: BLE001 — cleanup only
            pass


def test_concurrent_retries_in_separate_processes_buy_one_shard_and_record_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, lab, sweep_id, originals = _sweep_with_a_failed_shard(tmp_path, monkeypatch)
    go = tmp_path / "go"
    script = textwrap.dedent(
        f"""
        import os, time
        from pathlib import Path
        from lab.backends.local import LocalBackend
        from lab.core import Lab
        repo = Path({str(repo)!r})
        home = repo / "runs"
        lab = Lab(backend=LocalBackend(home=home, repo=repo), repo=repo, home=home)
        while not os.path.exists({str(go)!r}):
            time.sleep(0.005)
        lab.retry_sweep({sweep_id!r})
        """
    )
    n = 6
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    procs = [
        subprocess.Popen([PYTHON, "-c", script], cwd=repo, env=env, stderr=subprocess.PIPE)
        for _ in range(n)
    ]
    try:
        time.sleep(1.5)  # let every interpreter import lab and park on the start line
        go.write_text("go")
        errs = [p.communicate(timeout=120)[1].decode() for p in procs]
        assert [p.returncode for p in procs] == [0] * n, errs

        plan_ids = set(lab.sweep_plan(sweep_id).cells[0].shard_job_ids)
        launched = _sweep_job_ids(lab, sweep_id) - set(originals)
        assert len(launched) == 1, f"{len(launched)} retry shards bought for one missing shard"
        assert launched <= plan_ids, "a launched shard is missing from plan.json"
        assert _sweep_job_ids(lab, sweep_id) == plan_ids
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        _cancel_all(lab, _sweep_job_ids(lab, sweep_id))


def test_an_aggregate_cannot_write_back_a_plan_that_predates_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lost update with the other writer: an aggregate reads the plan, a retry appends a
    shard and writes, then the aggregate writes its stale copy over it. Under the lock the retry
    waits for the aggregate to finish and then works from the plan it wrote."""
    _repo, lab, sweep_id, _ = _sweep_with_a_failed_shard(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    blocker = threading.current_thread()  # replaced below by the aggregating thread
    real_merge = core_mod.merge_seed_rows

    def slow_merge(*a, **kw):
        if threading.current_thread() is blocker:
            entered.set()
            assert release.wait(30)
        return real_merge(*a, **kw)

    monkeypatch.setattr(core_mod, "merge_seed_rows", slow_merge)
    results: dict[str, object] = {}

    def aggregate() -> None:
        results["agg"] = lab.aggregate_sweep(sweep_id)

    def retry() -> None:
        results["retry"] = lab.retry_sweep(sweep_id)

    agg_thread = threading.Thread(target=aggregate)
    blocker = agg_thread
    agg_thread.start()
    try:
        assert entered.wait(30)
        retry_thread = threading.Thread(target=retry)
        retry_thread.start()
        retry_thread.join(1.0)
        # The retry must not have run its read-submit-write while the aggregate held the plan.
        assert retry_thread.is_alive(), "retry ran concurrently with an in-progress aggregate"
        release.set()
        agg_thread.join(30)
        retry_thread.join(60)
        assert not retry_thread.is_alive()

        plan_ids = set(lab.sweep_plan(sweep_id).cells[0].shard_job_ids)
        assert _sweep_job_ids(lab, sweep_id) == plan_ids
        assert len(plan_ids) == 3  # two originals + the one retry, none lost
    finally:
        release.set()
        _cancel_all(lab, _sweep_job_ids(lab, sweep_id))


def test_retry_records_each_shard_in_the_plan_as_it_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry killed partway (SIGKILL: no `except` runs) must not leave a launched shard out of
    the plan — every launch is persisted before the next one starts."""
    _repo, lab, sweep_id, originals = _sweep_with_a_failed_shard(tmp_path, monkeypatch)
    seen: list[set[str]] = []
    real_submit = core_mod.Lab.submit

    def recording_submit(self, spec, **kw):
        seen.append(set(lab.sweep_plan(sweep_id).cells[0].shard_job_ids))
        return real_submit(self, spec, **kw)

    # Two missing shards, so the second launch can observe the first one already on disk.
    lab.store.update_manifest(originals[0], status=JobState.failed)
    (lab.store.output_dir(originals[0]) / "results.csv").unlink()
    monkeypatch.setattr(core_mod.Lab, "submit", recording_submit)
    try:
        plan = lab.retry_sweep(sweep_id)
        new = [j for j in plan.cells[0].shard_job_ids if j not in originals]
        assert len(new) == 2
        assert new[0] in seen[1], "the first retry shard was not in plan.json before the second"
    finally:
        _cancel_all(lab, _sweep_job_ids(lab, sweep_id))


def test_the_lock_file_lives_beside_the_plan_and_is_never_created_for_an_unknown_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    with pytest.raises(core_mod.LabError):
        lab.aggregate_sweep("sweep-nope")
    with pytest.raises(core_mod.LabError):
        lab.retry_sweep("sweep-nope")
    assert not (lab.home / "sweep-nope").exists()
