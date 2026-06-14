from pathlib import Path
from lab.models import JobState
from lab.backends.skypilot import confirm_success, SUCCESS_SENTINEL


def test_success_requires_sentinel(tmp_path: Path):
    assert confirm_success(JobState.succeeded, tmp_path) is JobState.failed
    (tmp_path / SUCCESS_SENTINEL).write_text("1")
    assert confirm_success(JobState.succeeded, tmp_path) is JobState.succeeded


def test_non_success_unchanged(tmp_path: Path):
    assert confirm_success(JobState.failed, tmp_path) is JobState.failed
    (tmp_path / SUCCESS_SENTINEL).write_text("1")
    assert confirm_success(JobState.preempted, tmp_path) is JobState.preempted
