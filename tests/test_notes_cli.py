"""`lab note` / `lab notes`, and the push that puts an old note in front of a new failure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lab import notes
from lab.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_NOTES_DIR", str(tmp_path / "notes"))
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))


# --------------------------------------------------------------------------- lab note


def test_note_writes_and_prints_json(tmp_path: Path) -> None:
    result = runner.invoke(app, ["note", "--text", "billed over cap", "--kind", "BUDGET EVENT"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["kind"] == "BUDGET EVENT"
    assert payload["note_id"]
    assert [n.text for n in notes.search()] == ["billed over cap"]


def test_note_accepts_a_job_id_positionally(tmp_path: Path) -> None:
    result = runner.invoke(app, ["note", "j-1", "--text", "this one timed out"])

    assert result.exit_code == 0, result.output
    assert [n.job_id for n in notes.search()] == ["j-1"]


def test_note_without_text_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["note"])

    assert result.exit_code != 0


def test_note_records_a_dollar_amount(tmp_path: Path) -> None:
    runner.invoke(app, ["note", "--text", "over cap", "--usd", "11.88"])

    assert notes.search()[0].usd == pytest.approx(11.88)


def test_note_marks_an_agent_author(tmp_path: Path) -> None:
    runner.invoke(app, ["note", "--text", "surprised me", "--agent"])

    assert notes.search()[0].author == "agent"


# --------------------------------------------------------------------------- lab notes


def test_notes_lists_what_was_written(tmp_path: Path) -> None:
    notes.write(text="first thing")

    result = runner.invoke(app, ["notes"])

    assert result.exit_code == 0, result.output
    assert "first thing" in result.stdout


def test_notes_renders_the_team_log_table(tmp_path: Path) -> None:
    notes.write(text="billed over cap", job_id="j-1", kind="BUDGET EVENT", usd=11.88)

    result = runner.invoke(app, ["notes", "--format", "md"])

    assert result.exit_code == 0, result.output
    assert "| when (UTC) | actor | kind | note | job id | cost |" in result.stdout
    assert "billed over cap" in result.stdout


def test_notes_retire_stops_a_note_being_pushed(tmp_path: Path) -> None:
    notes.write(text="do not trust --timeout", signature="LabError: no offer")
    note_id = notes.search()[0].id

    result = runner.invoke(app, ["notes", "--retire", note_id, "--reason", "fixed in v0.1.0"])

    assert result.exit_code == 0, result.output
    assert notes.match(signature="LabError: no offer") == []


def test_notes_retire_of_an_unknown_id_fails_loudly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["notes", "--retire", "n-nope", "--reason", "x"])

    assert result.exit_code != 0
    assert "n-nope" in result.output


# --------------------------------------------------------------------------- the push


def test_push_renders_a_matching_note_for_a_failure(tmp_path: Path) -> None:
    """The highest-precision surface: this exact failure already has a note on it."""
    notes.write(text="it was the accelerator name, not the price", signature="LabError: no offer")

    rendered = notes.render_push(notes.match(signature="LabError: no offer"))

    assert rendered is not None
    assert "it was the accelerator name" in rendered


def test_push_is_silent_when_nothing_matches(tmp_path: Path) -> None:
    assert notes.render_push([]) is None


def test_push_dates_a_note_written_on_an_older_version(tmp_path: Path) -> None:
    """A version delta is a staleness hint even with nobody curating — the information the
    consuming project lacked while guarding a bug that had already been fixed."""
    notes.write(text="do not trust --timeout", signature="LabError: no offer")
    stale = [notes._replace(n, lab_version="0.1.0") for n in notes.search()]

    rendered = notes.render_push(stale, current="0.9.0")

    assert rendered is not None
    assert "0.1.0" in rendered


def test_push_does_not_date_a_note_from_the_running_version(tmp_path: Path) -> None:
    notes.write(text="fresh advice", signature="LabError: no offer")
    fresh = [notes._replace(n, lab_version="0.9.0") for n in notes.search()]

    rendered = notes.render_push(fresh, current="0.9.0")

    assert rendered is not None
    assert "0.9.0" not in rendered


# --------------------------------------------------------------------------- --last


def test_note_adopts_the_signature_of_the_last_failure(tmp_path: Path) -> None:
    """Nobody can hand-write a signature that matches.

    The ledger sanitizes an error message before signing it, so a signature computed from what
    the terminal printed does not equal the one the digest computes. Without `--last` the push
    can never fire in practice: the field would only ever be populated by someone who had read
    `events/stats.py` and guessed right.
    """
    from lab import events
    from lab.events.stats import signature

    events.begin("cli", "submit", {"argv": ["submit"]})
    events.note("cli.error", type="LabError", message="no offer under $0.66/hr")
    events.finish_current(
        outcome="error",
        exit_code=1,
        error={"type": "LabError", "message": "no offer under $0.66/hr", "where": None},
    )

    runner.invoke(app, ["note", "--last", "--text", "it was the accelerator name"])

    expected = signature({"type": "LabError", "message": "no offer under $0.66/hr"})
    assert notes.search()[0].signature == expected


def test_note_last_with_no_recorded_failure_still_records_the_note(tmp_path: Path) -> None:
    """A note is worth keeping even when there is nothing to attach it to."""
    result = runner.invoke(app, ["note", "--last", "--text", "just a thought"])

    assert result.exit_code == 0, result.output
    assert notes.search()[0].signature is None
