import pytest

from lab.models import CodeRef


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
