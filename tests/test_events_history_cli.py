from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from lab import events
from lab.cli import app
from lab.events import store
from lab.manifest import repo_root

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
runner = CliRunner()


def _invoke(*args: str) -> Result:
    """``CliRunner.invoke`` drives click's own dispatch directly, not ``lab.cli.main`` — so the
    close half of the open/close pair ``_load_env`` starts never lands (that only happens in
    ``main``'s post-dispatch cleanup, see its docstring). Left open, a call from one invocation
    in a test would still be there — dangling, ``running-or-died`` — for the *next* invocation
    in the same test to read back, which no real invocation of a separate `lab` process would
    ever see (its own close always lands before the next process starts). This closes it the way
    ``main`` would, so multi-invocation tests see what a real terminal session would."""
    result = runner.invoke(app, list(args))
    events.finish_current(
        outcome="ok" if result.exit_code == 0 else "error", exit_code=result.exit_code
    )
    return result


@pytest.fixture(autouse=True)
def _ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    store.append({"id": "a", "ts": NOW.isoformat(), "phase": "open", "session": "s", "seq": 0,
                  "surface": "cli", "action": "submit", "params": {"backend": "cpu"},
                  "project": {"name": "capacity"}, "lab_version": "0.5.1"}, when=NOW)
    store.append({"id": "a", "ts": NOW.isoformat(), "phase": "close", "outcome": "error",
                  "exit_code": 1, "duration_ms": 2000, "refs": {"job_id": "j-1"},
                  "result": {"cost_usd": 0.29},
                  "error": {"type": "ProvisionTimeout", "message": "no capacity"},
                  "trace": [{"t": 5, "k": "provision.attempt", "d": {"zone": "europe-west1-b"}}]},
                 when=NOW)


def test_history_emits_json_rows_newest_first() -> None:
    result = _invoke("history", "--all-projects")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["events"][0]["id"] == "a"
    assert payload["events"][0]["status"] == "error"


def test_history_omits_the_trace_unless_full_is_given() -> None:
    brief = json.loads(_invoke("history", "--all-projects").stdout)
    assert "trace" not in brief["events"][0]


def test_history_full_includes_the_trace() -> None:
    """A separate invocation from the brief-view test above: `lab history` itself records its
    own call (Task 5), so a *second* `_invoke` in the same test would pick up the first one's
    now-closed, all-projects-visible ``history`` row alongside the fixture's — one call per test
    keeps each assertion about the fixture's one event, not the test's own side effects."""
    full = json.loads(_invoke("history", "--all-projects", "--full").stdout)
    assert full["events"][0]["trace"][0]["k"] == "provision.attempt"


def test_history_filters_by_job() -> None:
    """Renamed from the brief's `..._and_failures`: this test only ever exercised `--job` (the
    dedicated `--failures` proof lives in `test_history_failures_flag_excludes_successful_calls`
    below); the old name claimed coverage the test didn't have."""
    out = json.loads(_invoke("history", "--all-projects", "--job", "j-1").stdout)
    assert len(out["events"]) == 1
    empty = json.loads(_invoke("history", "--all-projects", "--job", "j-2").stdout)
    assert empty["events"] == []


def test_history_stats_emits_the_aggregate_view() -> None:
    out = json.loads(_invoke("history", "--all-projects", "--stats").stdout)
    assert out["failures"] == 1
    assert out["signatures"][0]["count"] == 1
    assert out["usd_burned"] == 0.29


def test_report_emits_markdown_on_stdout() -> None:
    result = _invoke("report", "--all-projects")
    assert result.exit_code == 0
    assert result.stdout.startswith("# Lab event report")


def test_report_out_writes_a_file(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    result = _invoke("report", "--all-projects", "--out", str(target))
    assert result.exit_code == 0
    assert target.read_text().startswith("# Lab event report")


def _add_current_project_event() -> None:
    """A second ledger event filed under this repo's own project name (distinct from the
    fixture's "capacity"), so a scoping test can prove inclusion and exclusion at once rather
    than relying on the fixture happening to match."""
    store.append({"id": "b", "ts": NOW.isoformat(), "phase": "open", "session": "s", "seq": 0,
                  "surface": "cli", "action": "list", "params": {},
                  "project": {"name": repo_root().name}, "lab_version": "0.5.1"}, when=NOW)
    store.append({"id": "b", "ts": NOW.isoformat(), "phase": "close", "outcome": "ok",
                  "exit_code": 0, "duration_ms": 10, "refs": {}, "result": {}},
                 when=NOW)


def test_history_defaults_to_the_current_project() -> None:
    """The fixture's event lives under project "capacity"; without --all-projects, default
    scoping is the current repo's directory name — the "capacity" event must be filtered out
    while the same-project one survives, proving the default actually scopes rather than just
    returning everything."""
    _add_current_project_event()
    scoped = json.loads(_invoke("history").stdout)
    assert {e["id"] for e in scoped["events"]} == {"b"}


def test_history_all_projects_widens_the_default_scope() -> None:
    """A one-invoke counterpart to the default-scoping test above (kept in its own test — see
    `test_history_full_includes_the_trace`'s docstring for why a second `_invoke` in the same
    test would pick up the first `_invoke`'s own now-closed ledger row): --all-projects must
    surface the "capacity" event that the default omits."""
    _add_current_project_event()
    unscoped = json.loads(_invoke("history", "--all-projects").stdout)
    assert {e["id"] for e in unscoped["events"]} == {"a", "b"}


def test_history_failures_flag_excludes_successful_calls() -> None:
    """Adds a successful event alongside the fixture's failed one. Without --failures both
    appear; with it, only the failed one survives — proving the flag did the excluding rather
    than the fixture never having a success to begin with."""
    store.append({"id": "c", "ts": NOW.isoformat(), "phase": "open", "session": "s", "seq": 0,
                  "surface": "cli", "action": "list", "params": {},
                  "project": {"name": "capacity"}, "lab_version": "0.5.1"}, when=NOW)
    store.append({"id": "c", "ts": NOW.isoformat(), "phase": "close", "outcome": "ok",
                  "exit_code": 0, "duration_ms": 10, "refs": {}, "result": {}},
                 when=NOW)

    both = json.loads(_invoke("history", "--all-projects").stdout)
    assert {e["id"] for e in both["events"]} == {"a", "c"}

    only_failed = json.loads(
        _invoke("history", "--all-projects", "--failures").stdout
    )
    assert {e["id"] for e in only_failed["events"]} == {"a"}


def test_history_stats_since_reflects_the_applied_window() -> None:
    """`--stats`'s `since` field used to always read `null`, even when `--since` genuinely
    filtered the rows (`events.stats()` was called with no `since=` kwarg) — the sibling bug to
    the one fixed in `report`'s markdown header. `NOW` is fixed at 2026-08-18T12:00Z; the fixture
    events are stamped there, so `--since 1000d` keeps them in range while giving a
    deterministic, checkable cutoff."""
    out = json.loads(_invoke("history", "--all-projects", "--since", "1000d", "--stats").stdout)
    assert out["since"] is not None
    assert out["failures"] == 1  # the window still holds the fixture's one failure


def test_history_since_garbage_is_a_clean_usage_error_not_a_crash() -> None:
    """`events.read` calls `parse_duration` with no guard — an unparsable `--since` used to
    propagate a raw `ValueError` out as an unhandled traceback (exit 1, ``crash`` in the ledger).
    `wait`'s `--timeout` already turns the same `parse_duration` `ValueError` into a
    `typer.BadParameter`; `--since` must do the same, landing as a normal exit-2 usage error."""
    result = _invoke("history", "--all-projects", "--since", "garbage")
    assert result.exit_code == 2
    assert "garbage" in result.output


def test_report_since_garbage_is_a_clean_usage_error_not_a_crash() -> None:
    result = _invoke("report", "--all-projects", "--since", "garbage")
    assert result.exit_code == 2
    assert "garbage" in result.output


def test_report_out_to_an_unwritable_path_fails_cleanly() -> None:
    """`Path(out).write_text(text)` was unguarded — a bad `--out` directory raised a raw
    `OSError` straight through as an unhandled traceback (exit 1, ``crash``). It must instead
    exit 1 through `_fail`, with the path and the reason in a message the user can read, and
    land in the ledger as a real, named cause rather than a generic crash."""
    result = _invoke("report", "--all-projects", "--out", "/nonexistent_dir_xyz/report.md")
    assert result.exit_code == 1
    assert "nonexistent_dir_xyz" in result.stdout


def test_history_through_main_excludes_its_own_call(tmp_path: Path) -> None:
    """The `_invoke` helper above proves `_exclude_self` filters the currently-open call when
    driven straight through click's dispatch — but that bypasses `lab.cli.main`, the real
    console-script entry point that actually performs the close (`_invoke`'s own docstring
    explains why). This drives `lab history` through a real subprocess running `main()`, the way
    a person or an agent actually invokes it, and checks the same thing the task's manual
    verification checked by hand: the command never lists itself as `running-or-died`."""
    env = {**os.environ, "LAB_EVENTS_DIR": str(tmp_path / "events")}
    proc = subprocess.run(
        [sys.executable, "-c", "from lab.cli import main; main()", "history", "--all-projects"],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert not any(
        e["action"] == "history" and e["status"] == "running-or-died"
        for e in payload["events"]
    )
