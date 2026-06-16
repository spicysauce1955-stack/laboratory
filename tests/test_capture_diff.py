import subprocess
from pathlib import Path

from lab.manifest import apply_diff, capture_diff, current_commit


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "tracked.txt").write_text("original\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")


def test_clean_tree_returns_none(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert capture_diff(repo, tmp_path / "dest") is None


def test_roundtrip_restores_dirty_state(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    commit = current_commit(repo)
    # Make the tree dirty: edit a tracked file, add an untracked file.
    (repo / "tracked.txt").write_text("CHANGED\n")
    (repo / "new_script.py").write_text("print('hi')\n")

    blob = capture_diff(repo, tmp_path / "dest")
    assert blob is not None and Path(blob).exists()

    # Reconstruct: fresh checkout of the commit + apply the captured diff.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    _git(repo, "worktree", "add", "-q", "--detach", str(fresh), commit)
    apply_diff(Path(blob), fresh)

    assert (fresh / "tracked.txt").read_text() == "CHANGED\n"
    assert (fresh / "new_script.py").read_text() == "print('hi')\n"
