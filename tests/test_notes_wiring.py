"""Where notes reach their two readers: the next run, and whoever reads the result later."""

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


def _job(tmp_path: Path, job_id: str) -> Path:
    """A succeeded job on disk, in its own runs/ tree."""
    from helpers import make_manifest

    from lab.models import JobState
    from lab.store import JobStore

    home = tmp_path / "runs"
    store = JobStore(home)
    store.create(make_manifest(job_id, "python census.py").model_copy(
        update={"status": JobState.succeeded}
    ))
    return home


# ------------------------------------------------------- the push, at failure time


def test_a_failing_call_shows_a_note_matching_its_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reader who will never go looking: an agent that just hit this exact failure."""
    from lab.events.stats import signature

    error = {"type": "LabError", "message": "no offer under $0.66/hr"}
    notes.write(text="it was the accelerator name, not the price", signature=signature(error))

    shown = notes.push_for_error(error)

    assert shown is not None
    assert "it was the accelerator name" in shown


def test_a_failing_call_with_no_matching_note_says_nothing(tmp_path: Path) -> None:
    assert notes.push_for_error({"type": "LabError", "message": "something new"}) is None


def test_the_push_survives_a_broken_notes_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A note lookup on the error path must never turn a clean failure into a crash."""
    monkeypatch.setenv("LAB_NOTES_DIR", "/proc/nonexistent/nope")

    assert notes.push_for_error({"type": "LabError", "message": "x"}) is None


def test_an_error_with_no_message_is_not_pushed_on(tmp_path: Path) -> None:
    """signature(None) is the literal string 'unknown' — it must never match every note."""
    notes.write(text="unrelated advice", signature="unknown")

    assert notes.push_for_error(None) is None


# ------------------------------------------------------- notes on the job's own record


def test_status_reports_how_many_notes_a_job_carries(tmp_path: Path) -> None:
    """`lab status` is 81% of real usage — a note nobody sees there is a note nobody sees."""
    from lab.core import job_status_view

    home = _job(tmp_path, "j-1")
    notes.write(text="this one billed over cap", job_id="j-1", home=home)

    view = job_status_view(home, tmp_path, "j-1")

    assert view["notes"] == 1


def test_status_reports_zero_notes_without_inventing_a_key(tmp_path: Path) -> None:
    from lab.core import job_status_view

    home = _job(tmp_path, "j-1")

    assert job_status_view(home, tmp_path, "j-1")["notes"] == 0


# ------------------------------------------------------- notes travel with the result


def test_export_carries_a_jobs_notes(tmp_path: Path) -> None:
    """runs/ is git-ignored; the bundle is how a result reaches the repo where it is written up.
    A note left out of it is lost exactly when someone is trying to explain the number."""
    home = _job(tmp_path, "j-1")
    notes.write(text="two near-threshold cells flipped", job_id="j-1", home=home)

    from lab.core import default_lab

    lab = default_lab(home=home, backend="local")
    dest = tmp_path / "bundle"
    report = lab.export("j-1", dest)

    exported = dest / "j-1" / "notes.jsonl"
    assert exported.exists(), report
    assert "two near-threshold cells flipped" in exported.read_text()


def test_export_of_a_job_with_no_notes_is_unchanged(tmp_path: Path) -> None:
    from lab.core import default_lab

    home = _job(tmp_path, "j-1")

    dest = tmp_path / "bundle"
    default_lab(home=home, backend="local").export("j-1", dest)

    assert not (dest / "j-1" / "notes.jsonl").exists()


# ------------------------------------------------------- the ledger record


def test_writing_a_note_leaves_a_ledger_record(tmp_path: Path) -> None:
    """So `lab history` and `lab report` can find notes without being told a job id."""
    result = runner.invoke(app, ["note", "--text", "billed over cap", "--kind", "BUDGET EVENT"])
    assert result.exit_code == 0, result.output

    records = [
        json.loads(line)
        for path in sorted((tmp_path / "events").glob("*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    opens = [r for r in records if r.get("phase") == "open" and r.get("action") == "note"]
    assert opens, records
