"""The user-note channel: what a researcher writes down, keyed so the next run finds it.

The behaviours under test are the ones the 2026-08-26 ledger review said the tool was missing:
a note must survive the project it was written in (snn-research writes notes from
tempotron-capacity), must be retrievable by what *recurs* rather than by the job id it was
attached to, and must carry the version it was written at so stale advice reads as stale.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab import notes


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point both stores at tmp_path — never touch the real ~/.lab or the real runs/."""
    monkeypatch.setenv("LAB_NOTES_DIR", str(tmp_path / "notes"))
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))


# --------------------------------------------------------------------------- write


def test_note_lands_in_the_job_dir_next_to_its_logs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "j-1").mkdir(parents=True)

    notes.write(text="billed over cap", job_id="j-1", kind="BUDGET EVENT", home=runs)

    written = notes.for_job("j-1", home=runs)
    assert [n.text for n in written] == ["billed over cap"]
    assert [n.kind for n in written] == ["BUDGET EVENT"]


def test_note_survives_the_project_it_was_written_in(tmp_path: Path) -> None:
    """The global index is the point: notes written from one project must be readable from
    another. snn-research runs `lab` from tempotron-capacity but does its thinking elsewhere."""
    runs = tmp_path / "runs"
    (runs / "j-1").mkdir(parents=True)

    notes.write(text="RTX_4090 is the wrong name", job_id="j-1", home=runs)

    # A different project, with no access to that runs/ tree at all.
    assert [n.text for n in notes.search()] == ["RTX_4090 is the wrong name"]


def test_note_without_a_job_id_is_still_recorded(tmp_path: Path) -> None:
    """The highest-value notes have no job: a submit that dies pre-provision never gets one."""
    notes.write(text="the catalog error blames price, but it was the name", kind="GOTCHA")

    found = notes.search()
    assert [n.text for n in found] == ["the catalog error blames price, but it was the name"]
    assert found[0].job_id is None


def test_note_records_the_version_it_was_written_at(tmp_path: Path) -> None:
    """Without this a reader cannot tell advice from folklore."""
    notes.write(text="do not trust --timeout")

    assert notes.search()[0].lab_version == notes.current_version()


def test_note_masks_a_secret_pasted_into_its_text(tmp_path: Path) -> None:
    """Free text is a much bigger exfiltration surface than argv (FR-J1)."""
    notes.write(text="failed with --api-key=sk-live-abcdef1234567890 in the command")

    assert "sk-live-abcdef1234567890" not in notes.search()[0].text


def test_a_note_is_never_lost_to_an_unwritable_store(tmp_path: Path) -> None:
    """Best-effort like the ledger: a note that cannot be filed must not fail the command."""
    runs = tmp_path / "runs"
    (runs / "j-1").mkdir(parents=True)
    (runs / "j-1" / "notes.jsonl").mkdir()  # a directory where the file should go

    notes.write(text="still fine", job_id="j-1", home=runs)  # must not raise


# --------------------------------------------------------------------------- facets


def test_note_carries_the_facets_of_the_job_it_annotates(tmp_path: Path) -> None:
    """Retrieval is by what recurs, not by the job id — the next run has a different id."""
    notes.write(
        text="needs --with scipy",
        job_id="j-1",
        facets={"entrypoint": "census.py", "cloud": "do", "accelerators": None},
    )

    assert notes.search()[0].facets["entrypoint"] == "census.py"


def test_matching_finds_a_note_by_error_signature(tmp_path: Path) -> None:
    """The highest-precision case: the same failure recurring."""
    notes.write(text="it was the accelerator name, not the price", signature="LabError: no offer")

    hits = notes.match(signature="LabError: no offer")

    assert [n.text for n in hits] == ["it was the accelerator name, not the price"]


def test_matching_ignores_a_note_from_a_different_signature(tmp_path: Path) -> None:
    notes.write(text="unrelated", signature="LabError: something else")

    assert notes.match(signature="LabError: no offer") == []


def test_matching_finds_a_note_by_entrypoint_and_cloud(tmp_path: Path) -> None:
    notes.write(text="DO caps out at 8 boxes", facets={"entrypoint": "census.py", "cloud": "do"})

    hits = notes.match(facets={"entrypoint": "census.py", "cloud": "do"})

    assert [n.text for n in hits] == ["DO caps out at 8 boxes"]


def test_matching_never_fires_on_cloud_alone(tmp_path: Path) -> None:
    """A cloud-only match is noise, and noise is what makes an alarm ignorable."""
    notes.write(text="DO caps out at 8 boxes", facets={"entrypoint": "census.py", "cloud": "do"})

    assert notes.match(facets={"cloud": "do"}) == []


def test_matching_returns_the_newest_first_and_caps_the_count(tmp_path: Path) -> None:
    """Five old notes on every submit and nobody reads any of them."""
    for i in range(5):
        notes.write(text=f"note {i}", signature="LabError: no offer")

    hits = notes.match(signature="LabError: no offer", limit=2)

    assert [n.text for n in hits] == ["note 4", "note 3"]


# --------------------------------------------------------------------------- lifecycle


def test_a_retired_note_stops_being_pushed(tmp_path: Path) -> None:
    """Without retirement this feature distributes obsolete folklore at scale — the exact
    failure it exists to fix (a watchdog still guarding a bug fixed in v0.1.0)."""
    notes.write(text="do not trust --timeout", signature="LabError: no offer")
    note_id = notes.search()[0].id

    notes.retire(note_id, reason="enforced on-box since v0.1.0")

    assert notes.match(signature="LabError: no offer") == []


def test_a_retired_note_is_still_history(tmp_path: Path) -> None:
    notes.write(text="do not trust --timeout")
    note_id = notes.search()[0].id

    notes.retire(note_id, reason="enforced on-box since v0.1.0")

    retired = notes.search(include_retired=True)
    assert len(retired) == 1
    assert retired[0].retired is not None
    assert retired[0].retired["reason"] == "enforced on-box since v0.1.0"


def test_retiring_an_unknown_note_is_an_error_the_caller_can_report(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        notes.retire("nope", reason="whatever")


# --------------------------------------------------------------------------- read shapes


def test_search_filters_by_project(tmp_path: Path) -> None:
    notes.write(text="from A", project="A")
    notes.write(text="from B", project="B")

    assert [n.text for n in notes.search(project="A")] == ["from A"]


def test_notes_render_as_a_team_log_row(tmp_path: Path) -> None:
    """The adoption bribe: they already hand-write this table at submit time."""
    notes.write(text="billed over cap", job_id="j-1", kind="BUDGET EVENT", usd=11.88)

    md = notes.as_markdown(notes.search())

    assert "| j-1 |" in md
    assert "BUDGET EVENT" in md
    assert "billed over cap" in md
    assert "11.88" in md


def test_index_lines_are_one_json_object_each(tmp_path: Path) -> None:
    """Same shape as the job index and the ledger: greppable, append-only, one record a line."""
    notes.write(text="one")
    notes.write(text="two")

    lines = notes.index_path().read_text().strip().split("\n")
    assert len(lines) == 2
    assert [json.loads(line)["text"] for line in lines] == ["one", "two"]
    assert all(json.loads(line)["v"] == 1 for line in lines)
