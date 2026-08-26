"""`lab init` must say when the *skill* changed, not just that some file did.

The delivery failure this closes: `--row-key seed,alpha` shipped on 2026-08-06 and the consuming
project recorded it as impossible on 2026-08-14. `lab init` had refreshed the skill file in
between and said so only as a path inside a JSON list. Nobody diffs a skill, so a refreshed file
nobody re-reads is not delivery.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.init import scaffold


def _fresh_project(root: Path) -> None:
    (root / ".git").mkdir(parents=True)


def test_report_names_the_version_the_scaffold_moved_from(tmp_path: Path) -> None:
    _fresh_project(tmp_path)
    scaffold(tmp_path)

    state = json.loads((tmp_path / ".lab-scaffold.json").read_text())
    state["lab_version"] = "0.1.0"
    (tmp_path / ".lab-scaffold.json").write_text(json.dumps(state))
    # Make the shipped skill differ from what the project holds, as an upgrade would.
    skill = next(tmp_path.rglob("skills/laboratory/SKILL.md"))
    skill.write_text("stale\n")

    report = scaffold(tmp_path)

    assert report["from_version"] == "0.1.0"
    assert report["to_version"] and report["to_version"] != "0.1.0"


def test_report_flags_that_the_skill_itself_changed(tmp_path: Path) -> None:
    _fresh_project(tmp_path)
    scaffold(tmp_path)
    skill = next(tmp_path.rglob("skills/laboratory/SKILL.md"))
    recorded = json.loads((tmp_path / ".lab-scaffold.json").read_text())
    skill.write_text("stale\n")
    # Re-record the hash so the file reads as ours-and-untouched, i.e. refreshable.
    import hashlib

    rel = skill.relative_to(tmp_path).as_posix()
    recorded["files"][rel] = hashlib.sha256(b"stale\n").hexdigest()
    (tmp_path / ".lab-scaffold.json").write_text(json.dumps(recorded))

    report = scaffold(tmp_path)

    assert report["skill_changed"] is True


def test_report_does_not_flag_the_skill_when_it_is_current(tmp_path: Path) -> None:
    _fresh_project(tmp_path)
    scaffold(tmp_path)

    report = scaffold(tmp_path)

    assert report["skill_changed"] is False


def test_check_reports_the_skill_without_writing(tmp_path: Path) -> None:
    _fresh_project(tmp_path)

    report = scaffold(tmp_path, check=True)

    assert report["skill_changed"] is True
    assert not (tmp_path / ".lab-scaffold.json").exists()


def test_cli_init_tells_the_reader_to_re_read_the_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: a line a human or agent will actually see."""
    from typer.testing import CliRunner

    from lab.cli import app

    _fresh_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "ev"))

    result = CliRunner().invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    assert "skill" in result.output.lower()
    assert "corrections" in result.output.lower()
