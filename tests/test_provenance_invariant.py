from pathlib import Path

import pytest

from helpers import make_manifest
from lab.models import CodeRef
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


def test_write_manifest_rejects_gapb(tmp_path: Path):
    store = JobStore(tmp_path)
    m = make_manifest("g1", "echo hi")
    m.code.git_dirty = True  # dirty but diff_ref is None -> Gap B
    with pytest.raises(ValueError, match="diff_ref"):
        store.write_manifest(m)


def test_read_manifest_tolerates_legacy_gapb(tmp_path: Path):
    # A legacy Gap-B manifest already on disk must still LOAD (no migration break).
    store = JobStore(tmp_path)
    m = make_manifest("g2", "echo hi")
    (tmp_path / "g2").mkdir()
    m.code.git_dirty = True
    (tmp_path / "g2" / "manifest.json").write_text(m.model_dump_json(indent=2))
    loaded = store.read_manifest("g2")  # must not raise
    assert loaded.code.git_dirty is True and loaded.code.diff_ref is None
