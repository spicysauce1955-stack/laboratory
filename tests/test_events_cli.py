"""Every CLI exit path must land in the ledger, and none may change the exit code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lab.events import store


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


def _run(*args: str, env_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "from lab.cli import main; main()", *args],
        capture_output=True, text=True,
        env={**dict(__import__("os").environ), "LAB_EVENTS_DIR": str(env_dir)},
    )


def _folded(dir_: Path) -> list[dict]:
    return list(store.iter_records(sorted(dir_.glob("*.jsonl"))))


def test_a_successful_command_records_ok(_events_dir: Path) -> None:
    proc = _run("list", env_dir=_events_dir)
    assert proc.returncode == 0
    opened, closed = _folded(_events_dir)
    assert opened["action"] == "list" and opened["surface"] == "cli"
    assert closed["outcome"] == "ok" and closed["exit_code"] == 0


def test_an_unknown_job_records_an_error_and_keeps_exit_code_2(_events_dir: Path) -> None:
    proc = _run("status", "j-does-not-exist", env_dir=_events_dir)
    assert proc.returncode == 2
    closed = _folded(_events_dir)[1]
    assert closed["outcome"] == "error"
    assert closed["exit_code"] == 2


def test_a_bad_flag_records_a_usage_error(_events_dir: Path) -> None:
    proc = _run("list", "--nonexistent-flag", env_dir=_events_dir)
    assert proc.returncode == 2
    records = _folded(_events_dir)
    opened, closed = records[-2], records[-1]
    assert closed["outcome"] == "usage_error"
    assert closed["exit_code"] == 2
    # click never got the chance to hand us its "no such option" message (it prints and exits 2
    # entirely inside its own dispatch, standalone) — the sanitized argv already on the open
    # record is what tells a reader what was actually typed.
    assert opened["params"]["argv"] == ["list", "--nonexistent-flag"]


def test_an_unknown_command_records_a_usage_error_with_sanitized_argv(_events_dir: Path) -> None:
    proc = _run("nosuchcommand", "--token", "s" * 40, env_dir=_events_dir)
    assert proc.returncode == 2
    records = _folded(_events_dir)
    opened, closed = records[-2], records[-1]
    assert opened["action"] == "<unparsed>"
    assert closed["outcome"] == "usage_error"
    assert "…REDACTED…" in opened["params"]["argv"]


def test_the_cause_behind_typer_exit_becomes_the_recorded_error(
    monkeypatch: pytest.MonkeyPatch, _events_dir: Path
) -> None:
    from lab import cli

    @cli.app.command()
    def boom() -> None:  # a stand-in for the ~25 `_fail(N, e)` call sites
        try:
            raise RuntimeError("no capacity in europe-west1")
        except RuntimeError as e:
            cli._fail(1, e)

    with pytest.raises(SystemExit) as exc:
        cli.main(["boom"])
    assert exc.value.code == 1
    closed = _folded(_events_dir)[1]
    assert closed["outcome"] == "error"
    assert closed["error"]["type"] == "RuntimeError"
    assert closed["error"]["message"] == "no capacity in europe-west1"


def test_emit_annotates_the_call_with_refs_and_a_digest(
    monkeypatch: pytest.MonkeyPatch, _events_dir: Path
) -> None:
    from lab import cli, events

    with events.record("cli", "submit", {}):
        cli._emit({"job_id": "j-4f2a", "state": "succeeded", "actual_cost_usd": 1.25})
    closed = _folded(_events_dir)[1]
    assert closed["refs"] == {"job_id": "j-4f2a"}
    assert closed["result"] == {"state": "succeeded", "cost_usd": 1.25}
