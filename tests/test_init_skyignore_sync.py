"""`lab init` keeps a `.skyignore` it created in step with what the project git-ignores LATER (M7).

The stress run: `lab init`, then append `private_keys/` to `.gitignore`. `lab init --check` said
`ok`, and re-running `lab init` left `.skyignore` without it — and since SkyPilot reads ONLY
`.skyignore` once one exists, that directory would be rsynced to a third-party box.

Now a `.skyignore` that `lab init` created carries a marker line, and while the marker is there
`lab init` adds what the git-ignore gained (`--check` reports it as stale, so CI catches it). A
`.skyignore` the user wrote — or one whose marker the user deleted — is never rewritten: it is
the user's deliberate override, often precisely to upload git-ignored data, so init only warns
and names the git-ignored paths it will upload (`skyignore_uploads`). All of it is asserted
against SkyPilot's own exclusion logic, not against our intent.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from lab.init import _SKYIGNORE_MARKER, scaffold


def _git_project(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("gcp-key.json\n")
    (root / "gcp-key.json").write_text("{}")
    (root / "main.py").write_text("print('hi')\n")


def _excluded(root: Path) -> set[str]:
    from sky.data.storage_utils import get_excluded_files

    return {e.rstrip("/") for e in get_excluded_files(str(root))}


def _uploaded(root: Path, path: str) -> bool:
    parts = path.rstrip("/").split("/")
    excluded = _excluded(root)
    return not any("/".join(parts[: i + 1]) in excluded for i in range(len(parts)))


def _add_private_keys(root: Path) -> None:
    with (root / ".gitignore").open("a") as f:
        f.write("private_keys/\n")
    (root / "private_keys").mkdir()
    (root / "private_keys" / "id_rsa").write_text("-----BEGIN KEY-----")


def test_a_created_skyignore_carries_the_marker(tmp_path: Path) -> None:
    _git_project(tmp_path)

    scaffold(tmp_path)

    assert (tmp_path / ".skyignore").read_text().splitlines()[0] == _SKYIGNORE_MARKER


def test_check_reports_a_pattern_git_ignored_after_init(tmp_path: Path) -> None:
    _git_project(tmp_path)
    scaffold(tmp_path)
    _add_private_keys(tmp_path)
    assert _uploaded(tmp_path, "private_keys/id_rsa")  # the leak, as SkyPilot sees it

    report = scaffold(tmp_path, check=True)

    assert report["ok"] is False
    assert ".skyignore" in report["merged"]
    assert "private_keys/" in report["skyignore_uploads"]
    assert any("private_keys/" in w for w in report["warnings"])
    assert "private_keys" not in (tmp_path / ".skyignore").read_text()  # check writes nothing


def test_init_syncs_it_and_then_check_is_ok(tmp_path: Path) -> None:
    _git_project(tmp_path)
    scaffold(tmp_path)
    _add_private_keys(tmp_path)

    report = scaffold(tmp_path)

    assert ".skyignore" in report["merged"]
    assert not _uploaded(tmp_path, "private_keys/id_rsa")
    assert report["skyignore_uploads"] == []
    again = scaffold(tmp_path, check=True)
    assert again["ok"] is True and ".skyignore" in again["unchanged"]


def test_a_pattern_is_synced_before_anything_matches_it(tmp_path: Path) -> None:
    """The directory need not exist yet: the pattern itself is what keeps it off the box."""
    _git_project(tmp_path)
    scaffold(tmp_path)
    with (tmp_path / ".gitignore").open("a") as f:
        f.write("*.p12\n")

    scaffold(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "cert.p12").write_text("k")

    assert not _uploaded(tmp_path, "sub/cert.p12")


def test_deleting_the_marker_hands_the_file_to_the_user(tmp_path: Path) -> None:
    _git_project(tmp_path)
    scaffold(tmp_path)
    sky = tmp_path / ".skyignore"
    sky.write_text("\n".join(ln for ln in sky.read_text().splitlines() if ln != _SKYIGNORE_MARKER))
    before = sky.read_text()
    _add_private_keys(tmp_path)

    report = scaffold(tmp_path)

    assert sky.read_text() == before
    assert "private_keys/" in report["skyignore_uploads"]
    assert any("private_keys/" in w and "WILL be uploaded" in w for w in report["warnings"])


def test_a_user_authored_skyignore_is_never_rewritten_and_check_stays_ok(tmp_path: Path) -> None:
    """Uploading git-ignored data is often the point of writing one: warn, never fail CI."""
    _git_project(tmp_path)
    scaffold(tmp_path)
    mine = "# mine\n.env\n.venv\nruns\n.git\n.mypy_cache\n.ruff_cache\n.pytest_cache\n"
    mine += "__pycache__\nmlruns\n"
    (tmp_path / ".skyignore").write_text(mine)

    report = scaffold(tmp_path, check=True)

    assert (tmp_path / ".skyignore").read_text() == mine
    assert report["skyignore_uploads"] == ["gcp-key.json"]
    assert ".skyignore" not in report["merged"]
    assert report["ok"] is True


def test_a_legacy_lab_created_skyignore_is_recognised_and_marked(tmp_path: Path) -> None:
    """Created before the marker existed (no marker, and no ownership in .lab-scaffold.json):
    recognisable by the header lab init wrote into it, and taken over with the marker."""
    _git_project(tmp_path)
    scaffold(tmp_path)
    sky = tmp_path / ".skyignore"
    legacy = "\n".join(ln for ln in sky.read_text().splitlines() if ln != _SKYIGNORE_MARKER) + "\n"
    sky.write_text(legacy)
    state_file = tmp_path / ".lab-scaffold.json"
    state = json.loads(state_file.read_text())
    del state["skyignore"]  # what a pre-marker lab init wrote
    state_file.write_text(json.dumps(state))
    _add_private_keys(tmp_path)

    report = scaffold(tmp_path)

    text = sky.read_text()
    assert text.splitlines()[0] == _SKYIGNORE_MARKER
    assert not _uploaded(tmp_path, "private_keys/id_rsa")
    assert ".skyignore" in report["merged"]


def test_a_path_only_a_nested_gitignore_hides_is_synced_too(tmp_path: Path) -> None:
    _git_project(tmp_path)
    scaffold(tmp_path)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / ".gitignore").write_text("token.txt\n")
    (tmp_path / "lib" / "token.txt").write_text("secret")
    assert _uploaded(tmp_path, "lib/token.txt")

    scaffold(tmp_path)

    assert not _uploaded(tmp_path, "lib/token.txt")
