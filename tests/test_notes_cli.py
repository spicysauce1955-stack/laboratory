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


def test_note_agent_with_a_name_is_recorded_as_that_author(tmp_path: Path) -> None:
    """The actual root cause behind every "note fails on long, detailed text" report on file
    (ledger forensics 2026-08-26/27): `--agent` marks the note as agent-written and takes no
    value, but real calls write `--agent claude` / `--agent ws-trainer` as if it named the
    agent. The stray name then had nowhere to bind — `job_id` was already taken — and click
    rejected it as an extra argument (`usage_error`, exit 2), regardless of how short or plain
    the `-m` text was. A long, punctuation-rich, multi-sentence write-up (commas, parentheses,
    apostrophes — the shape of a real incident note) must succeed here exactly like a short one;
    length and punctuation were never the cause.
    """
    message = (
        "During today's sweep we noticed something odd (a real gotcha, not a fluke): jobs that "
        "used the 'cpu' backend on DigitalOcean kept failing with a 422 from the provisioner, "
        "and it wasn't obvious why at first, but after digging through the SkyPilot logs, "
        "checking the account's resource limits, and comparing against a few other fresh "
        "accounts, we found the actual cause -- a brand-new DO account is capped at a much "
        "smaller instance tier by default, so requesting anything above 4 vCPUs (or a volume "
        "bigger than 50GB) gets rejected outright, no matter how the request is phrased."
    )

    result = runner.invoke(
        app,
        ["note", "j-1", "--kind", "GOTCHA", "--agent", "ws-trainer", "-m", message],
    )

    assert result.exit_code == 0, result.output
    written = notes.search()[0]
    assert written.author == "ws-trainer"
    assert written.text == message


def test_note_agent_name_before_job_id_swaps_them_back(tmp_path: Path) -> None:
    """The other stray-token ordering: `--agent <name> <job_id>` (name first, real job id
    second). `--agent` takes no value, so both `<name>` and `<job_id>` are plain positional
    tokens on the command line; click fills `job_id` (the first *declared* positional) with
    whichever one it meets first -- here that's the agent name -- and the real job id lands in
    `extra_args` instead. Before the fix this recorded job_id="ws-trainer" (not a real job id)
    and author=<the real job id> (also wrong) -- exactly the live bug report. A real job id
    always matches `_JOB_ID_SHAPE`; recovering requires swapping the two back, not just
    recovering the author the way the existing (job-id-first) case does.
    """
    result = runner.invoke(
        app,
        ["note", "--agent", "ws-trainer", "20260906-101112-abcdef", "-m", "hello"],
    )

    assert result.exit_code == 0, result.output
    written = notes.search()[0]
    assert written.author == "ws-trainer"
    assert written.job_id == "20260906-101112-abcdef"
    assert written.text == "hello"


def test_note_agent_with_a_name_and_no_job_id_is_recorded_as_that_author(tmp_path: Path) -> None:
    """Same misuse as the test above, but with NO job id at all -- a documented, valid use (a
    submit that died before provisioning never gets a job id). With no job id, `job_id` is the
    first positional argument click will fill, so the lone stray `--agent <name>` token binds
    straight to it instead of landing in the `extra_args` catch-all (which is what let the
    two-positional case above be recovered). Before the fix this silently recorded a bogus
    job_id="claude" and reverted `author` to plain "agent" -- no error, no clue anything was
    wrong. It must instead behave exactly as if `--agent` were a plain boolean flag with no job
    id given: `author` is the name, and `job_id` is absent.
    """
    result = runner.invoke(
        app, ["note", "--agent", "claude", "-m", "submit died before provisioning"]
    )

    assert result.exit_code == 0, result.output
    written = notes.search()[0]
    assert written.author == "claude"
    assert written.job_id is None
    assert written.text == "submit died before provisioning"


def test_note_extra_argument_without_agent_is_a_clear_usage_error(tmp_path: Path) -> None:
    """A stray positional that is *not* the `--agent <name>` misuse (the flag was never given)
    is a genuine mistake — most often `-m`/`--text` left off entirely. This must still fail, but
    with a specific, actionable message naming the stray token, not a bare `usage_error` with no
    explanation of what was wrong."""
    result = runner.invoke(app, ["note", "j-1", "stray-token", "-m", "the real message"])

    assert result.exit_code != 0
    assert "stray-token" in result.output
    assert "-m/--text" in result.output


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
