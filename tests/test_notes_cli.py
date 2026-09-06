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
    """`--agent` is a value-bearing option now (never a boolean flag), so a value-less marker
    is spelled `--agent=` (empty value via `=`) rather than a bare trailing `--agent` — a bare
    `--agent` with nothing after it (not even `=`) is rejected, since it would otherwise have to
    guess whether the next token was its value or an unrelated flag. `--agent=` still records
    plain `"agent"` as the author, exactly like the old boolean-true behavior."""
    runner.invoke(app, ["note", "--text", "surprised me", "--agent="])

    assert notes.search()[0].author == "agent"


def test_note_bare_agent_with_nothing_after_it_is_a_clear_usage_error(tmp_path: Path) -> None:
    """A trailing `--agent` with no value at all (not even `--agent=`) is exactly the shape that
    used to swallow the next flag's token silently. It must fail loudly instead."""
    result = runner.invoke(app, ["note", "--text", "surprised me", "--agent"])

    assert result.exit_code != 0


def test_note_agent_with_a_name_is_recorded_as_that_author(tmp_path: Path) -> None:
    """`--agent NAME` now binds `NAME` directly as `--agent`'s own value at parse time — an
    ordinary, unambiguous option — regardless of the text's length or punctuation. A long,
    punctuation-rich, multi-sentence write-up (commas, parentheses, apostrophes — the shape of a
    real incident note) must succeed here exactly like a short one; length and punctuation were
    never the cause of the old failure mode.
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


def test_note_agent_name_before_job_id(tmp_path: Path) -> None:
    """`--agent <name>` before the job id positional. Since `--agent` now consumes `<name>` as
    its own value directly at parse time, it never competes with `job_id`'s positional slot at
    all -- no swap logic is needed, unlike the old boolean-flag design where both tokens were
    plain positionals and `job_id` (the first *declared* one) greedily absorbed whichever came
    first on the command line."""
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
    """`--agent <name>` with NO job id at all -- a documented, valid use (a submit that died
    before provisioning never gets a job id). `--agent` consumes `<name>` as its own value
    directly, so there is no stray token left over to (mis)bind to `job_id`'s positional slot:
    `author` is the name, and `job_id` is simply absent, with no special-casing needed.
    """
    result = runner.invoke(
        app, ["note", "--agent", "claude", "-m", "submit died before provisioning"]
    )

    assert result.exit_code == 0, result.output
    written = notes.search()[0]
    assert written.author == "claude"
    assert written.job_id is None
    assert written.text == "submit died before provisioning"


def test_note_agent_boolean_default_keeps_a_non_standard_job_id(tmp_path: Path) -> None:
    """`lab note <job_id> --agent= -m ...` for a job id that does NOT match the lab's own
    `_new_job_id()` shape (a short/legacy/test-fixture id) -- the exact live bug: a job id parsed
    positionally is never touched by `--agent`'s own parsing, whatever shape it has, so it must
    survive unchanged and `author` must fall back to the plain agent-marker default."""
    result = runner.invoke(app, ["note", "j-1", "--agent=", "-m", "hello world"])

    assert result.exit_code == 0, result.output
    written = notes.search()[0]
    assert written.job_id == "j-1"
    assert written.author == "agent"


def test_note_agent_with_a_name_keeps_a_non_standard_job_id_regardless_of_order(
    tmp_path: Path,
) -> None:
    """`--agent NAME` (value directly on the flag) records NAME as author and leaves a
    non-standard-shaped job id untouched, whether the job id or `--agent` comes first -- there is
    no ordering concern left with `--agent` parsed as an ordinary option."""
    result_job_first = runner.invoke(
        app, ["note", "j-1", "--agent", "claude", "-m", "hello job-first"]
    )
    result_agent_first = runner.invoke(
        app, ["note", "--agent", "claude", "j-2", "-m", "hello agent-first"]
    )

    assert result_job_first.exit_code == 0, result_job_first.output
    assert result_agent_first.exit_code == 0, result_agent_first.output
    written = notes.search()
    by_text = {n.text: n for n in written}
    assert by_text["hello job-first"].job_id == "j-1"
    assert by_text["hello job-first"].author == "claude"
    assert by_text["hello agent-first"].job_id == "j-2"
    assert by_text["hello agent-first"].author == "claude"


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
