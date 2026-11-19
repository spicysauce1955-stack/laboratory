"""`lab init` must not widen what SkyPilot uploads to a remote box.

SkyPilot's ``get_excluded_files`` is an if/else, not a union: with no ``.skyignore`` it excludes
what git ignores, and the moment one exists it reads *only* that file. ``lab init`` used to create
a ``.skyignore`` holding just the scaffold's own lines, so every git-ignored secret a project had
(``gcp-key.json``, ``secrets/``, ``.env.local``) silently started shipping to third-party Vast
hosts. Asserted against SkyPilot's own exclusion logic, before and after, so it tracks what really
syncs rather than what we intended.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lab.init import scaffold


def _git_project(root: Path, gitignore: str) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text(gitignore)
    (root / "gcp-key.json").write_text("{}")
    (root / "secrets").mkdir()
    (root / "secrets" / "token").write_text("t")
    (root / ".env.local").write_text("K=v")
    (root / "data").mkdir()
    (root / "data" / "big.bin").write_text("x")
    (root / "nested" / "deep").mkdir(parents=True)
    (root / "nested" / "deep" / "x.pem").write_text("k")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "y.pem").write_text("k")
    (root / "main.py").write_text("print('hi')\n")


def _excluded(root: Path) -> set[str]:
    from sky.data.storage_utils import get_excluded_files

    return {e.rstrip("/") for e in get_excluded_files(str(root))}


def _covered(path: str, excluded: set[str]) -> bool:
    """rsync drops an excluded directory's whole subtree, so an ancestor counts."""
    parts = path.rstrip("/").split("/")
    return any("/".join(parts[: i + 1]) in excluded for i in range(len(parts)))


GITIGNORE = "gcp-key.json\nsecrets/\n.env.local\n/data/\n*.pem\n"


def test_creating_skyignore_keeps_every_git_ignored_path_off_the_box(tmp_path: Path) -> None:
    _git_project(tmp_path, GITIGNORE)
    before = _excluded(tmp_path)
    assert {"gcp-key.json", "secrets", ".env.local", "data"} <= before  # the fixture is honest

    report = scaffold(tmp_path)

    assert ".skyignore" in report["created"]
    after = _excluded(tmp_path)
    leaked = sorted(p for p in before if not _covered(p, after))
    assert leaked == [], f"lab init started uploading git-ignored paths: {leaked}"
    assert "main.py" not in after  # still excludes secrets, not everything
    # The scaffold's own lines are still there.
    assert ".env" in (tmp_path / ".skyignore").read_text().splitlines()


def test_seeded_patterns_also_cover_files_created_after_init(tmp_path: Path) -> None:
    """The carried-over lines are patterns, not just a snapshot of today's files."""
    _git_project(tmp_path, GITIGNORE)
    scaffold(tmp_path)
    (tmp_path / "later.pem").write_text("k")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "gcp-key.json").write_text("{}")

    after = _excluded(tmp_path)

    assert "later.pem" in after
    assert "sub/gcp-key.json" in after


def test_creation_is_announced_as_a_warning(tmp_path: Path) -> None:
    _git_project(tmp_path, GITIGNORE + "!keep.pem\n")

    report = scaffold(tmp_path)

    text = "\n".join(report["warnings"])
    assert ".skyignore" in text
    assert "gcp-key.json" in text  # names what was carried over
    assert "!keep.pem" in text  # and what could not be expressed


def test_an_existing_skyignore_is_never_rewritten(tmp_path: Path) -> None:
    """A user's own .skyignore may deliberately upload git-ignored data — leave it be."""
    _git_project(tmp_path, GITIGNORE)
    mine = "# mine\n.env\n.venv\nruns\n.git\n.mypy_cache\n.ruff_cache\n.pytest_cache\n"
    mine += "__pycache__\nmlruns\n"
    (tmp_path / ".skyignore").write_text(mine)

    report = scaffold(tmp_path)

    assert (tmp_path / ".skyignore").read_text() == mine
    # ...but say out loud which git-ignored paths will now be uploaded.
    text = "\n".join(report["warnings"])
    assert "gcp-key.json" in text and "secrets" in text


def test_check_mode_writes_no_skyignore(tmp_path: Path) -> None:
    _git_project(tmp_path, GITIGNORE)
    scaffold(tmp_path, check=True)
    assert not (tmp_path / ".skyignore").exists()


def test_not_a_git_repo_still_seeds_from_the_gitignore(tmp_path: Path) -> None:
    """No git, no snapshot — the pattern translation alone must still carry the lines over."""
    (tmp_path / ".gitignore").write_text("gcp-key.json\n/data/\nsub/dir/\n")

    scaffold(tmp_path)

    lines = (tmp_path / ".skyignore").read_text().splitlines()
    assert "gcp-key.json" in lines
    assert "/data/" in lines
    assert "/sub/dir/" in lines  # a middle slash anchors the pattern in git


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("foo", "foo"),
        ("foo/", "foo/"),
        ("/foo", "/foo"),
        ("a/b", "/a/b"),
        ("**/a/b", "a/b"),
        ("*.pem", "*.pem"),
        ("  ", None),
        ("# comment", None),
    ],
)
def test_gitignore_line_translation(line: str, expected: str | None) -> None:
    from lab.init import _translate_ignore_line

    assert _translate_ignore_line(line) == expected
