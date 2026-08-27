"""`lab queue list` must surface which host wrote the heartbeat — needed to tell two droplets
ticking against the same queue apart during a scheduler cutover (see
docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from lab.cli import app
from lab.scheduler.queue import LocalQueueStore

runner = CliRunner()


def test_queue_list_reports_which_host_wrote_the_heartbeat(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    q = LocalQueueStore(tmp_path / "queue")
    q.write_heartbeat({"at": "2026-08-27T00:00:00+00:00", "host": "lab-scheduler-old", "tick_count": 1})

    result = runner.invoke(app, ["queue", "list"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["host"] == "lab-scheduler-old"


def test_queue_list_host_is_none_with_no_heartbeat_yet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    LocalQueueStore(tmp_path / "queue")  # never writes a heartbeat

    result = runner.invoke(app, ["queue", "list"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["host"] is None
