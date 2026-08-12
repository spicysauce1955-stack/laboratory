"""The scaffold must travel inside the package, not beside it in the repo."""

from __future__ import annotations

import json
from importlib.resources import files


def _payload():
    return files("lab") / "_scaffold" / "project"


def test_payload_is_importable_package_data() -> None:
    assert (_payload() / "dot-mcp.json").is_file()
    assert (_payload() / "experiments" / "example.py").is_file()


def test_mcp_config_shells_the_console_script() -> None:
    cfg = json.loads((_payload() / "dot-mcp.json").read_text())
    assert cfg["mcpServers"]["lab"]["args"] == ["run", "lab", "mcp"]


def test_skill_ships_in_the_payload() -> None:
    assert (_payload() / "skills" / "laboratory" / "SKILL.md").is_file()


def test_skill_does_not_tell_the_agent_it_is_in_the_lab_repo() -> None:
    """The skill is read inside a researcher's project; every instruction must be true there."""
    text = (_payload() / "skills" / "laboratory" / "SKILL.md").read_text()
    assert "in this repo" not in text
    assert "inside the `laboratory` repo" not in text
    assert "python -m lab.mcp_server" not in text
    # Paths that only exist in the lab's own checkout must not be presented as local files.
    assert "`src/lab/" not in text
    assert "`docs/guides/" not in text


def test_env_example_carries_no_real_secrets() -> None:
    text = (_payload() / "dot-env.example").read_text()
    assert "BEGIN PRIVATE KEY" not in text
    for line in text.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            _, _, value = line.partition("=")
            assert value.strip() in {"", '""'} or value.strip().startswith("<"), line
