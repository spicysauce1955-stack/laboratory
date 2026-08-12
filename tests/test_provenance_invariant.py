import json
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


def test_create_stamps_the_running_lab_version(tmp_path: Path):
    """A manifest records which lab produced it — the tool now versions independently of the
    project, so 'which lab wrote this run' is provenance, not trivia."""
    from lab import __version__

    store = JobStore(tmp_path)
    m = make_manifest("j-ver-1", "true")
    assert m.lab_version is None
    store.create(m)
    assert store.read_manifest("j-ver-1").lab_version == __version__


def test_create_does_not_overwrite_an_explicit_lab_version(tmp_path: Path):
    """Adoption/import paths may replay a manifest produced elsewhere; their stamp wins."""
    store = JobStore(tmp_path)
    m = make_manifest("j-ver-2", "true")
    m.lab_version = "0.4.0"
    store.create(m)
    assert store.read_manifest("j-ver-2").lab_version == "0.4.0"


def test_legacy_manifest_without_lab_version_still_reads(tmp_path: Path):
    """Read-compatibility is part of the public surface: a v0.4.0 manifest has no such key."""
    store = JobStore(tmp_path)
    m = make_manifest("j-ver-3", "true")
    store.create(m)
    raw = json.loads(store.manifest_path("j-ver-3").read_text())
    del raw["lab_version"]
    store.manifest_path("j-ver-3").write_text(json.dumps(raw))
    assert store.read_manifest("j-ver-3").lab_version is None
