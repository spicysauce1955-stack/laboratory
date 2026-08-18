from __future__ import annotations

import json
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


def test_history_filters_by_job_and_failures() -> None:
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
