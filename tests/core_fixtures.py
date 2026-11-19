"""A throwaway git repo + local-backend Lab, for tests that must not touch the dev checkout."""

from __future__ import annotations

import subprocess
from pathlib import Path

from lab.backends.local import LocalBackend
from lab.core import Lab


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def make_repo(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    """A committed repo with a uv.lock and runs/ ignored (so submits don't dirty it)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    (repo / "uv.lock").write_text("lock\n")
    (repo / ".gitignore").write_text("runs/\n")
    for rel, text in (files or {}).items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    return repo


def local_lab(repo: Path) -> Lab:
    home = repo / "runs"
    return Lab(backend=LocalBackend(home=home, repo=repo), repo=repo, home=home)
