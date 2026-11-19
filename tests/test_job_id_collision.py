"""A job id is claimed exclusively; a collision gets a fresh id, never someone else's run dir.

Stress run 2026-11-19: job ids carry a 24-bit random suffix under a one-second timestamp, and
`JobStore.create` made the run dir with ``mkdir(exist_ok=True)`` — so two submits that drew the
same id in the same second shared one run dir and the second manifest silently replaced the
first (about 0.2% per 1000-shard sweep). The id is now claimed with an exclusive mkdir, retried
with a new id on collision, and `create` refuses to replace an existing manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core_fixtures import local_lab, make_repo
from helpers import make_manifest

import lab.core as core_mod
from lab.core import LabError
from lab.models import JobSpec
from lab.store import JobStore


def _ids(monkeypatch: pytest.MonkeyPatch, *ids: str) -> list[str]:
    queue = list(ids)
    drawn: list[str] = []

    def fake() -> str:
        drawn.append(queue.pop(0))
        return drawn[-1]

    monkeypatch.setattr(core_mod, "_new_job_id", fake)
    return drawn


def test_create_refuses_to_replace_an_existing_manifest(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "runs")
    store.create(make_manifest("j1", "first"))
    with pytest.raises(FileExistsError):
        store.create(make_manifest("j1", "second"))
    assert store.read_manifest("j1").run.entrypoint_command == "first"


def test_a_colliding_submit_draws_a_new_id(tmp_path: Path, monkeypatch) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    drawn = _ids(monkeypatch, "20260101-000000-aaaaaa", "20260101-000000-aaaaaa", "20260101-000000-bbbbbb")

    first = lab.submit(JobSpec(command="echo first"))
    second = lab.submit(JobSpec(command="echo second"))

    assert first == "20260101-000000-aaaaaa" and second == "20260101-000000-bbbbbb"
    assert drawn == ["20260101-000000-aaaaaa"] * 2 + ["20260101-000000-bbbbbb"]
    assert lab.manifest(first).run.entrypoint_command == "echo first"
    assert lab.manifest(second).run.entrypoint_command == "echo second"


def test_a_dirty_submit_never_writes_its_diff_into_another_jobs_dir(tmp_path: Path, monkeypatch) -> None:
    """The dirty path captured the diff into the job dir *before* creating the manifest, so it
    used to be the first writer into a colliding dir."""
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    (repo / "uv.lock").write_text("changed\n")  # dirty tree
    taken = lab.store.job_dir("20260101-000000-aaaaaa")
    taken.mkdir(parents=True)  # another submit has just claimed this id
    _ids(monkeypatch, "20260101-000000-aaaaaa", "20260101-000000-cccccc")

    jid = lab.submit(JobSpec(command="echo dirty"))

    assert jid == "20260101-000000-cccccc"
    assert list(taken.iterdir()) == []
    assert (lab.store.job_dir(jid) / "code_diff.tar.gz").exists()


def test_a_refused_submit_leaves_no_claimed_dir_behind(tmp_path: Path, monkeypatch) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    (repo / "uv.lock").unlink()  # submit refuses: no lockfile to hash (FR-B2)
    _ids(monkeypatch, "20260101-000000-dddddd")
    with pytest.raises(LabError, match="uv.lock"):
        lab.submit(JobSpec(command="echo x"))
    assert not lab.store.job_dir("20260101-000000-dddddd").exists()


def test_a_colliding_sweep_id_draws_a_new_one(tmp_path: Path, monkeypatch) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    first, _ = lab.sweep("true", {}, seeds="0", shard_size=1)
    stamp = first.removeprefix("sweep-")
    real = core_mod._new_job_id
    queue = [stamp]

    monkeypatch.setattr(core_mod, "_new_job_id", lambda: queue.pop(0) if queue else real())
    second, _ = lab.sweep("true", {}, seeds="0", shard_size=1)

    assert second != first
    assert len(lab.sweep_plan(first).cells[0].shard_job_ids) == 1
