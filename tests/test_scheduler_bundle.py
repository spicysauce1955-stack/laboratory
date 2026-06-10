"""Bundle = exact runnable tree: committed files + dirty diff + untracked (non-ignored) files."""

import subprocess
from pathlib import Path

from lab.scheduler.bundle import create_bundle, extract_bundle


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "uv.lock").write_text("lockfile\n")
    (repo / "exp.py").write_text("print('v1')\n")
    (repo / ".gitignore").write_text("runs/\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def test_clean_tree_bundle_roundtrip(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tar, code = create_bundle(repo, tmp_path / "out")
    assert tar.exists() and tar.suffixes[-2:] == [".tar", ".gz"]
    assert not code.git_dirty and len(code.git_commit) == 40
    dest = extract_bundle(tar, tmp_path / "x")
    assert (dest / "exp.py").read_text() == "print('v1')\n"
    assert (dest / "uv.lock").exists()


def test_dirty_and_untracked_files_included(tmp_path: Path):
    repo = _make_repo(tmp_path)
    (repo / "exp.py").write_text("print('v2')\n")  # modified tracked
    (repo / "new_exp.py").write_text("print('new')\n")  # untracked
    (repo / "runs").mkdir()
    (repo / "runs" / "big.bin").write_text("x" * 10)  # ignored -> excluded
    tar, code = create_bundle(repo, tmp_path / "out")
    assert code.git_dirty
    dest = extract_bundle(tar, tmp_path / "x")
    assert (dest / "exp.py").read_text() == "print('v2')\n"
    assert (dest / "new_exp.py").read_text() == "print('new')\n"
    assert not (dest / "runs").exists()
