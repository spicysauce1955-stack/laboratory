"""Regression: `lab wait` must exit non-zero when it gives up on a timeout (LAB-BUGS §1)."""

from typer.testing import CliRunner

import lab.cli as cli_mod
from helpers import make_manifest
from lab.cli import app
from lab.models import JobState


def test_wait_exits_1_on_timeout_without_completion(monkeypatch, tmp_path):
    running = make_manifest("j1", "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running}
    )

    class _FakeLab:
        def wait(self, ids, *, interval, timeout):
            return [running]  # never reached terminal -> the timeout path

    class _FakeStore:
        def __init__(self, home):
            pass

        def manifest_path(self, job_id):
            p = tmp_path / f"{job_id}.json"
            p.touch()  # exists -> passes the "unknown job id" guard
            return p

    monkeypatch.setattr(cli_mod, "_lab_for", lambda job_id: _FakeLab())
    monkeypatch.setattr(cli_mod, "JobStore", _FakeStore)

    result = CliRunner().invoke(app, ["wait", "j1", "--timeout", "0.5"])
    assert result.exit_code == 1
