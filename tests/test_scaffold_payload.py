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


def test_scaffold_skill_version_matches_the_lab_version() -> None:
    """The scaffolded skill declares its own version in frontmatter, and that is the number a
    consumer project reads to decide whether its copy is current.

    It is a hand-written literal, so nothing stopped it drifting: written at v0.9.0, it still
    said `0.9.0` while the body documented v0.12.0 behaviour — a doc asserting its own staleness.
    `scripts/release.sh` now rewrites it during the version bump; this pins the invariant so the
    two cannot silently separate again.
    """
    import re
    from pathlib import Path

    from lab import __version__

    skill = (
        Path(__file__).resolve().parents[1]
        / "src/lab/_scaffold/project/skills/laboratory/SKILL.md"
    ).read_text(encoding="utf-8")
    declared = re.search(r'^  version: "([^"]+)"', skill, re.MULTILINE)
    assert declared is not None, "the scaffolded skill lost its frontmatter version"
    assert declared.group(1) == __version__, (
        f"scaffolded skill declares {declared.group(1)}, lab is {__version__} — "
        "release.sh's bump step should have rewritten it"
    )
