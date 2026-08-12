"""`lab init` scaffolds a researcher's project and stays honest on re-runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lab import __version__
from lab.init import STATE_FILE, scaffold


def test_scaffolds_a_fresh_project(tmp_path: Path) -> None:
    report = scaffold(tmp_path)
    assert (tmp_path / ".mcp.json").is_file()
    assert (tmp_path / ".env.example").is_file()
    assert (tmp_path / ".gitignore").is_file()
    assert (tmp_path / ".skyignore").is_file()
    assert (tmp_path / "experiments" / "example.py").is_file()
    assert (tmp_path / ".claude" / "skills" / "laboratory" / "SKILL.md").is_file()
    assert ".mcp.json" in report["created"] or ".mcp.json" in report["merged"]
    state = json.loads((tmp_path / STATE_FILE).read_text())
    assert state["lab_version"] == __version__


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    scaffold(tmp_path)
    report = scaffold(tmp_path)
    assert report["created"] == []
    assert report["conflicts"] == []


def test_unmodified_file_is_refreshed(tmp_path: Path) -> None:
    """A file we own and the user has not touched is brought up to the installed version."""
    scaffold(tmp_path)
    target = tmp_path / "experiments" / "example.py"
    state_path = tmp_path / STATE_FILE
    state = json.loads(state_path.read_text())
    target.write_text("# pretend the previous version shipped this\n")
    state["files"]["experiments/example.py"] = hashlib.sha256(target.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state))

    report = scaffold(tmp_path)
    assert "experiments/example.py" in report["refreshed"]
    assert "get_overrides" in target.read_text()


def test_user_modified_file_is_never_clobbered(tmp_path: Path) -> None:
    scaffold(tmp_path)
    target = tmp_path / "experiments" / "example.py"
    target.write_text("# my own experiment\n")

    report = scaffold(tmp_path)
    assert "experiments/example.py" in report["conflicts"]
    assert target.read_text() == "# my own experiment\n"
    assert (tmp_path / "experiments" / "example.py.new").is_file()


def test_mcp_json_merge_preserves_other_servers(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    scaffold(tmp_path)
    cfg = json.loads((tmp_path / ".mcp.json").read_text())
    assert cfg["mcpServers"]["other"] == {"command": "x"}
    assert cfg["mcpServers"]["lab"]["args"] == ["run", "lab", "mcp"]


def test_mcp_json_user_edited_lab_entry_is_left_alone(tmp_path: Path) -> None:
    scaffold(tmp_path)
    path = tmp_path / ".mcp.json"
    cfg = json.loads(path.read_text())
    cfg["mcpServers"]["lab"]["args"] = ["run", "lab", "mcp", "--my-flag"]
    path.write_text(json.dumps(cfg))

    report = scaffold(tmp_path)
    assert ".mcp.json" in report["conflicts"]
    assert json.loads(path.read_text())["mcpServers"]["lab"]["args"][-1] == "--my-flag"


def test_gitignore_merge_appends_only_missing_lines(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.pyc\nruns/\n")
    scaffold(tmp_path)
    lines = (tmp_path / ".gitignore").read_text().splitlines()
    assert lines.count("runs/") == 1
    assert "*.pyc" in lines
    assert ".env" in lines


def test_check_reports_ok_on_a_fresh_scaffold(tmp_path: Path) -> None:
    scaffold(tmp_path)
    assert scaffold(tmp_path, check=True)["ok"] is True


def test_check_fails_on_a_missing_file_and_writes_nothing(tmp_path: Path) -> None:
    scaffold(tmp_path)
    (tmp_path / ".env.example").unlink()
    report = scaffold(tmp_path, check=True)
    assert report["ok"] is False
    assert ".env.example" in report["created"]
    assert not (tmp_path / ".env.example").exists()


def test_skill_lands_under_dot_claude(tmp_path: Path) -> None:
    scaffold(tmp_path)
    assert (tmp_path / ".claude" / "skills" / "laboratory" / "SKILL.md").is_file()
    assert (tmp_path / ".claude" / "skills" / "laboratory" / "examples").is_dir()


def test_never_touches_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='mine'\n")
    scaffold(tmp_path)
    assert (tmp_path / "pyproject.toml").read_text() == "[project]\nname='mine'\n"
