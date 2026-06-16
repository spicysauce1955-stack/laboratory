from datetime import datetime, timedelta, timezone
from pathlib import Path

from lab.models import JobSpec
from lab.scheduler.models import Guardrails, Triggers
from lab.scheduler.queue import LocalQueueStore
from lab.scheduler.register import register
from test_scheduler_bundle import _make_repo


def test_dirty_registration_sets_diff_ref(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "wip.py").write_text("print('wip')\n")  # make the tree dirty
    q = LocalQueueStore(tmp_path / "q")
    reg = register(
        repo,
        q,
        JobSpec(command="python exp.py"),
        Triggers(),
        Guardrails(expires_at=datetime.now(timezone.utc) + timedelta(days=1)),
    )
    assert reg.code.git_dirty is True
    assert reg.code.diff_ref == reg.bundle_key  # the bundle IS the captured dirty state
    reg.code.assert_fail_closed()  # the eventual submit must pass the write-path guard
