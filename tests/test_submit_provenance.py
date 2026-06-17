import subprocess
from pathlib import Path

import pytest

from lab.core import Lab, LabError
from lab.backends.local import LocalBackend
from lab.models import JobSpec


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo_with_lockfile(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "uv.lock").write_text("lock\n")
    (repo / "tracked.txt").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _lab(repo: Path) -> Lab:
    home = repo / "runs"
    return Lab(backend=LocalBackend(home=home, repo=repo), repo=repo, home=home)


def test_dirty_submit_captures_diff_ref(tmp_path: Path, monkeypatch):
    repo = _repo_with_lockfile(tmp_path)
    (repo / "tracked.txt").write_text("DIRTY\n")  # make the tree dirty
    monkeypatch.delenv("LAB_R2_ENDPOINT", raising=False)  # isolate: local diff_ref, no R2 mirror
    monkeypatch.chdir(repo)
    lab = _lab(repo)
    job_id = lab.submit(JobSpec(command="true"))
    m = lab.manifest(job_id)
    assert m.code.git_dirty is True
    assert m.code.diff_ref is not None
    assert Path(m.code.diff_ref).exists()  # local path resolves (no R2 in this test)


def test_clean_submit_has_no_diff_ref(tmp_path: Path, monkeypatch):
    repo = _repo_with_lockfile(tmp_path)
    monkeypatch.chdir(repo)
    lab = _lab(repo)
    m = lab.manifest(lab.submit(JobSpec(command="true")))
    assert m.code.git_dirty is False and m.code.diff_ref is None


def test_no_dirty_refuses(tmp_path: Path, monkeypatch):
    repo = _repo_with_lockfile(tmp_path)
    (repo / "tracked.txt").write_text("DIRTY\n")
    monkeypatch.chdir(repo)
    lab = _lab(repo)
    with pytest.raises(LabError, match="dirty"):
        lab.submit(JobSpec(command="true"), allow_dirty=False)


def test_toctou_dirty_then_clean_fails_loud(tmp_path: Path, monkeypatch):
    # The tree reads dirty, but capture_diff finds nothing (a concurrent stash/checkout cleaned
    # it). Rather than writing a Gap-B manifest (raw ValueError at create), submit fails loud.
    repo = _repo_with_lockfile(tmp_path)
    (repo / "tracked.txt").write_text("DIRTY\n")
    monkeypatch.chdir(repo)
    lab = _lab(repo)
    monkeypatch.setattr("lab.core.capture_diff", lambda *a, **k: None)
    with pytest.raises(LabError, match="changed during submit"):
        lab.submit(JobSpec(command="true"))


def test_r2_upload_failure_falls_back_to_local(tmp_path: Path, monkeypatch):
    # A transient R2 error on the diff mirror must not fail the submit — the local diff_ref is
    # still fail-closed-valid, so we keep it and warn.
    repo = _repo_with_lockfile(tmp_path)
    (repo / "tracked.txt").write_text("DIRTY\n")
    monkeypatch.chdir(repo)
    lab = _lab(repo)

    class _BadR2:
        def upload_file(self, local, key):
            raise RuntimeError("network down")

        def uri(self, prefix):  # pragma: no cover - never reached on failure
            return f"r2://b/{prefix}"

    monkeypatch.setattr("lab.core.r2_enabled", lambda: True)
    monkeypatch.setattr("lab.core.R2Store.from_env", classmethod(lambda cls: _BadR2()))
    job_id = lab.submit(JobSpec(command="true"))
    m = lab.manifest(job_id)
    assert m.code.git_dirty is True
    assert m.code.diff_ref is not None and Path(m.code.diff_ref).exists()  # local fallback
