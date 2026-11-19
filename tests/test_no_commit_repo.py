"""A repo with no commits yet is where `lab init` is most likely to be run first (stress run L6).

Every CLI call probes the project's HEAD for the event ledger; in a repo with no commits git
answers `fatal: ambiguous argument 'HEAD': unknown revision...` — and that went straight to the
user's terminal, on `lab init` of all commands, looking like the init itself had failed. The
probe already tolerated the failure; only git's stderr leaked. `current_commit` now captures it
and carries it in the exception instead, so a caller that does fail on it (submit) still says why.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lab.manifest import current_commit


def _empty_repo(root: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def test_lab_init_in_a_repo_with_no_commits_is_quiet(tmp_path: Path) -> None:
    repo = _empty_repo(tmp_path / "fresh")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAB_")}
    env.update(LAB_DOTENV="0", LAB_EVENTS_DIR=str(tmp_path / "events"),
               LAB_JOBS_INDEX_DIR=str(tmp_path / "jobs"), LAB_NOTES_DIR=str(tmp_path / "notes"))

    proc = subprocess.run(
        [sys.executable, "-c", "from lab.cli import main; main()", "init"],
        cwd=repo, env=env, capture_output=True, text=True, timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    assert "fatal:" not in proc.stderr
    assert "ambiguous argument" not in proc.stderr
    assert (repo / ".mcp.json").is_file()


def test_current_commit_keeps_gits_reason_off_the_terminal(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    repo = _empty_repo(tmp_path / "fresh")

    with pytest.raises(subprocess.CalledProcessError) as info:
        current_commit(repo)

    assert capfd.readouterr().err == ""
    assert "HEAD" in str(info.value)  # the reason travels with the exception instead
