from pathlib import Path

import pytest

from helpers import make_manifest
from lab.models import CodeRef, JobState
from lab.store import JobStore


def test_clean_coderef_passes():
    CodeRef(git_commit="a" * 40, git_dirty=False).assert_fail_closed()


def test_dirty_with_ref_passes():
    CodeRef(git_commit="a" * 40, git_dirty=True, diff_ref="r2://b/x").assert_fail_closed()


def test_empty_commit_rejected():
    with pytest.raises(ValueError, match="git_commit"):
        CodeRef(git_commit="", git_dirty=False).assert_fail_closed()


def test_dirty_without_ref_rejected():
    with pytest.raises(ValueError, match="diff_ref"):
        CodeRef(git_commit="a" * 40, git_dirty=True, diff_ref=None).assert_fail_closed()


def test_create_rejects_gapb(tmp_path: Path):
    # The guard is at create() — the single new-manifest chokepoint.
    store = JobStore(tmp_path)
    m = make_manifest("g1", "echo hi")
    m.code.git_dirty = True  # dirty but diff_ref is None -> Gap B
    with pytest.raises(ValueError, match="diff_ref"):
        store.create(m)


def test_read_manifest_tolerates_legacy_gapb(tmp_path: Path):
    # A legacy Gap-B manifest already on disk must still LOAD (no migration break).
    store = JobStore(tmp_path)
    m = make_manifest("g2", "echo hi")
    (tmp_path / "g2").mkdir()
    m.code.git_dirty = True
    (tmp_path / "g2" / "manifest.json").write_text(m.model_dump_json(indent=2))
    loaded = store.read_manifest("g2")  # must not raise
    assert loaded.code.git_dirty is True and loaded.code.diff_ref is None


def test_update_manifest_tolerates_legacy_gapb(tmp_path: Path):
    # A legacy Gap-B job that gets a lifecycle status update must NOT crash — the guard is on
    # create(), not on every write, so in-flight legacy runs can still reach a terminal state.
    store = JobStore(tmp_path)
    m = make_manifest("g3", "echo hi")
    (tmp_path / "g3").mkdir()
    m.code.git_dirty = True
    (tmp_path / "g3" / "manifest.json").write_text(m.model_dump_json(indent=2))
    updated = store.update_manifest("g3", status=JobState.failed)  # must not raise
    assert updated.status is JobState.failed and updated.code.diff_ref is None
