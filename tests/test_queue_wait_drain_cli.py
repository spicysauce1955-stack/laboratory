"""`lab queue wait-drain` — the CLI surface for the scheduler-cutover drain gate."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import PYTHON
from typer.testing import CliRunner

from lab.cli import app
from lab.models import CodeRef, JobSpec
from lab.scheduler.models import Guardrails, Registration, RegState, Triggers
from lab.scheduler.queue import LocalQueueStore

runner = CliRunner()
T0 = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)


def _reg(reg_id: str, state: RegState) -> Registration:
    return Registration(
        reg_id=reg_id,
        created_at=T0,
        spec=JobSpec(command=f"{PYTHON} -c 'print(1)'"),
        triggers=Triggers(),
        guardrails=Guardrails(expires_at=T0 + timedelta(days=1)),
        bundle_key=f"bundles/{reg_id}.tar.gz",
        code=CodeRef(git_commit="0" * 40),
        state=state,
    )


def test_exits_0_when_already_drained(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    LocalQueueStore(tmp_path / "queue")

    result = runner.invoke(app, ["queue", "wait-drain", "--interval", "0.01", "--timeout", "1"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data == {"drained": True, "blocking": []}


def test_exits_1_and_names_blockers_on_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("reg-stuck", RegState.launched))

    result = runner.invoke(app, ["queue", "wait-drain", "--interval", "0.01", "--timeout", "0.05"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["drained"] is False
    assert data["blocking"] == ["reg-stuck"]


def test_bad_timeout_string_is_usage_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    LocalQueueStore(tmp_path / "queue")

    result = runner.invoke(app, ["queue", "wait-drain", "--timeout", "2hr"])

    assert result.exit_code == 2
