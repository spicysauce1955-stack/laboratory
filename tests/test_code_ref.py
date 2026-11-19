"""`--code-ref` runs the commit it names (and the manifest records), never the working tree."""

from __future__ import annotations

from pathlib import Path

import pytest
from core_fixtures import git, local_lab, make_repo
from helpers import PYTHON, wait_terminal

from lab.core import LabError
from lab.models import JobSpec, JobState

_EXP = """\
import os
from pathlib import Path
Path(os.environ["LAB_RUN_DIR"], "version.txt").write_text("{v}")
"""


def _two_commits(tmp_path: Path) -> tuple[Path, str, str]:
    repo = make_repo(tmp_path, {"exp.py": _EXP.format(v="v1")})
    first = git(repo, "rev-parse", "HEAD")
    (repo / "exp.py").write_text(_EXP.format(v="v2"))
    git(repo, "commit", "-qam", "v2")
    return repo, first, git(repo, "rev-parse", "HEAD")


def _ran(lab, jid: str) -> str:
    assert wait_terminal(lab.backend, jid) is JobState.succeeded, lab.manifest(jid).end_reason
    return (lab.store.output_dir(jid) / "version.txt").read_text()


@pytest.mark.parametrize("ref", ["sha", "HEAD~1"])
def test_code_ref_runs_the_named_commit(tmp_path: Path, monkeypatch, ref: str):
    repo, first, _ = _two_commits(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    jid = lab.submit(
        JobSpec(command=f"{PYTHON} exp.py", code_ref=first if ref == "sha" else ref)
    )
    m = lab.manifest(jid)
    assert m.code.git_commit == first and m.code.git_dirty is False
    assert _ran(lab, jid) == "v1"


def test_code_ref_ignores_a_dirty_working_tree(tmp_path: Path, monkeypatch):
    repo, first, _ = _two_commits(tmp_path)
    (repo / "exp.py").write_text(_EXP.format(v="dirty"))
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    jid = lab.submit(JobSpec(command=f"{PYTHON} exp.py", code_ref=first), allow_dirty=False)
    m = lab.manifest(jid)
    assert m.code.git_dirty is False and m.code.diff_ref is None
    assert _ran(lab, jid) == "v1"


def test_code_ref_naming_head_is_the_ordinary_path(tmp_path: Path, monkeypatch):
    repo, _, head = _two_commits(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    jid = lab.submit(JobSpec(command=f"{PYTHON} exp.py", code_ref=head))
    assert lab.manifest(jid).code.git_commit == head
    assert _ran(lab, jid) == "v2"
    assert not (lab.home / "_code_ref").exists()  # no snapshot needed for the current commit


@pytest.mark.parametrize("ref", ["no-such-branch", "--output=/tmp/x", "0" * 40])
def test_unknown_code_ref_fails_loud(tmp_path: Path, monkeypatch, ref: str):
    repo, _, _ = _two_commits(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    with pytest.raises(LabError, match="code_ref"):
        lab.submit(JobSpec(command=f"{PYTHON} exp.py", code_ref=ref))
    assert lab.list_jobs() == []


def test_a_sweep_at_a_code_ref_shares_one_snapshot(tmp_path: Path, monkeypatch):
    repo, first, _ = _two_commits(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    _, jids = lab.sweep(f"{PYTHON} exp.py", {"N": [1, 2]}, code_ref=first)
    assert [_ran(lab, j) for j in jids] == ["v1", "v1"]
    assert [p.name for p in (lab.home / "_code_ref").iterdir()] == [first]


def test_find_cached_keys_on_the_code_ref_commit(tmp_path: Path, monkeypatch):
    repo, first, _ = _two_commits(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    at_head = lab.submit(JobSpec(command=f"{PYTHON} exp.py"))
    _ran(lab, at_head)
    assert lab.find_cached(JobSpec(command=f"{PYTHON} exp.py")) == at_head
    assert lab.find_cached(JobSpec(command=f"{PYTHON} exp.py", code_ref=first)) is None
