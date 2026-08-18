"""Every CLI exit path must land in the ledger, and none may change the exit code."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lab.events import store


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


@pytest.fixture
def _sandbox_app():
    """Registers throwaway commands on the shared module-level ``cli.app`` for one test, then
    removes them. ``cli.app`` is a process-wide singleton every other test in the session also
    imports — a command left behind here (as the original version of this test did) pollutes
    every test that runs afterward, in this file and beyond."""
    from lab import cli

    before = list(cli.app.registered_commands)
    yield cli.app
    cli.app.registered_commands[:] = before


def _run(
    *args: str, env_dir: Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {**dict(__import__("os").environ), "LAB_EVENTS_DIR": str(env_dir)}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", "from lab.cli import main; main()", *args],
        capture_output=True, text=True,
        env=env,
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
    _sandbox_app, _events_dir: Path
) -> None:
    from lab import cli

    @_sandbox_app.command()
    def boom() -> None:  # a stand-in for the ~30 `_fail(N, e)` call sites
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


def test_emit_survives_a_broken_refs_from_without_turning_success_into_a_crash(
    monkeypatch: pytest.MonkeyPatch, _events_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`refs_from`/`digest_of` run inside the open `record()` block, so an unguarded raise there
    would finish the call as `crash` for a command that actually succeeded — the one thing the
    design says must never happen. Break `refs_from` deliberately and prove `_emit` still prints
    the real payload and the call still records `ok`."""
    from lab import cli, events

    def _boom(payload: object) -> dict[str, object]:
        raise RuntimeError("refs_from is broken")

    monkeypatch.setattr(cli, "refs_from", _boom)
    with events.record("cli", "submit", {}):
        cli._emit({"job_id": "j-4f2a", "state": "succeeded"})
    closed = _folded(_events_dir)[1]
    assert closed["outcome"] == "ok"
    assert '"job_id": "j-4f2a"' in capsys.readouterr().out  # the real payload still printed


def test_a_keyboard_interrupt_during_a_command_is_recorded_as_interrupted(
    _sandbox_app, _events_dir: Path
) -> None:
    """click's own dispatch catches an in-command Ctrl-C and re-raises it as `Exit(130)`, so
    this arrives at `main()`'s `except SystemExit` branch, not the dedicated
    `except KeyboardInterrupt` handler — that branch must classify code 130 as `interrupted`,
    not fabricate a generic `"error"`."""
    from lab import cli

    @_sandbox_app.command()
    def boominterrupt() -> None:
        raise KeyboardInterrupt()

    with pytest.raises(SystemExit) as exc:
        cli.main(["boominterrupt"])
    assert exc.value.code == 130
    closed = _folded(_events_dir)[1]
    assert closed["outcome"] == "interrupted"
    assert closed["exit_code"] == 130


def test_an_unhandled_exception_is_recorded_as_a_crash(
    _sandbox_app, _events_dir: Path
) -> None:
    from lab import cli

    @_sandbox_app.command()
    def boomcrash() -> None:
        raise ValueError("unexpected: disk full")

    with pytest.raises(SystemExit) as exc:
        cli.main(["boomcrash"])
    assert exc.value.code == 1
    closed = _folded(_events_dir)[1]
    assert closed["outcome"] == "crash"
    assert closed["exit_code"] == 1
    assert closed["error"]["type"] == "ValueError"
    assert closed["error"]["message"] == "unexpected: disk full"


def test_wait_style_exit_codes_3_and_4_propagate_through_main(
    _sandbox_app, _events_dir: Path
) -> None:
    """`lab wait`'s exit codes are contract (3 teardown leak, 4 fail-fast) — stand-ins for the
    exact `_fail(3, ...)`/`_fail(4, ...)` calls `wait` itself makes, driven through `main()`."""
    from lab import cli

    @_sandbox_app.command()
    def boom3() -> None:
        cli._fail(3, "simulated teardown leak")

    @_sandbox_app.command()
    def boom4() -> None:
        cli._fail(4, "simulated fail-fast")

    with pytest.raises(SystemExit) as exc3:
        cli.main(["boom3"])
    assert exc3.value.code == 3
    closed3 = _folded(_events_dir)[1]
    assert closed3["outcome"] == "error"
    assert closed3["exit_code"] == 3
    assert closed3["error"]["message"] == "simulated teardown leak"

    with pytest.raises(SystemExit) as exc4:
        cli.main(["boom4"])
    assert exc4.value.code == 4
    closed4 = _folded(_events_dir)[-1]
    assert closed4["outcome"] == "error"
    assert closed4["exit_code"] == 4
    assert closed4["error"]["message"] == "simulated fail-fast"


def test_load_env_does_not_open_a_ledger_call_for_the_mcp_command(_events_dir: Path) -> None:
    """`lab mcp` is a long-lived server; a client kills it with SIGTERM/SIGKILL, so `main()`'s
    post-dispatch close (see its docstring) never runs. Before this fix, `_load_env` opened a
    ledger call for every command including `mcp`, so a killed server left a permanent dangling
    `open` every session — dangling opens are exempt from compaction, so one accumulates per
    agent session until the 90-day cap, and each counts as a failure in `--stats`/`lab report`.
    `mcp` must never open a call in the first place, so there is nothing to leave dangling."""
    from lab import cli, events

    class _FakeCtx:
        invoked_subcommand = "mcp"

    cli._load_env(ctx=_FakeCtx())  # type: ignore[arg-type]
    assert events.current() is None
    assert _folded(_events_dir) == []


def test_a_killed_mcp_process_leaves_no_dangling_open(_events_dir: Path) -> None:
    """End-to-end version of the unit test above: actually launch `lab mcp` as a real process,
    SIGKILL it (the same way an MCP client tears down its server, and the one signal a
    SIGTERM handler could never catch), and check the ledger it left behind."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "from lab.cli import main; main()", "mcp"],
        env={**os.environ, "LAB_EVENTS_DIR": str(_events_dir)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Give the interpreter time to start, import, and reach `_load_env` (which is where the
        # bug's `events.begin` call lived) before it blocks forever on stdio.
        time.sleep(1.5)
        assert proc.poll() is None, "the mcp server exited before it could be killed"
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert _folded(_events_dir) == []


def test_queue_and_scheduler_subcommands_record_the_leaf_action(
    tmp_path: Path, _events_dir: Path
) -> None:
    """`ctx.invoked_subcommand` only ever resolves to the immediate top-level child, so without
    `_group_action` every `lab queue ...`/`lab scheduler ...` invocation would collapse to the
    same indistinguishable action name (`"queue"`/`"scheduler"`). The design doc names
    `action: "scheduler tick"` specifically as what the scheduler's systemd timer records."""
    (tmp_path / "repo").mkdir()
    (tmp_path / "queue").mkdir()
    extra_env = {"LAB_REPO_DIR": str(tmp_path / "repo"), "LAB_QUEUE_DIR": str(tmp_path / "queue")}

    proc = _run("queue", "list", env_dir=_events_dir, extra_env=extra_env)
    assert proc.returncode == 0, proc.stderr
    opened = _folded(_events_dir)[0]
    assert opened["action"] == "queue list"

    proc = _run("scheduler", "tick", env_dir=_events_dir, extra_env=extra_env)
    assert proc.returncode == 0, proc.stderr
    records = _folded(_events_dir)
    tick_opened = records[2]  # after queue-list's open+close pair
    assert tick_opened["action"] == "scheduler tick"
