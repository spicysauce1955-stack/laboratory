"""Diagnostics never go to stdout: the CLI's stdout is parsed JSON and MCP's is the stdio stream."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from core_fixtures import local_lab, make_repo
from helpers import make_manifest

import lab.core as core_mod
from lab.models import JobSpec, JobState


class _FailingR2:
    """An R2 store whose every network call fails (unreachable endpoint / bad credentials)."""

    def upload_file(self, *a, **k):
        raise ConnectionError("endpoint unreachable")

    def download_dir(self, *a, **k):
        raise ConnectionError("endpoint unreachable")

    def uri(self, rel: str) -> str:
        return f"r2://bucket/{rel}"


@pytest.fixture
def failing_r2(monkeypatch):
    monkeypatch.setattr(core_mod, "r2_enabled", lambda: True)
    monkeypatch.setattr(core_mod.R2Store, "from_env", staticmethod(lambda: _FailingR2()))


def test_dirty_diff_upload_failure_goes_to_stderr(tmp_path: Path, monkeypatch, capsys, failing_r2):
    repo = make_repo(tmp_path, {"tracked.txt": "v1\n"})
    (repo / "tracked.txt").write_text("DIRTY\n")
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    jid = lab.submit(JobSpec(command="true"))
    out, err = capsys.readouterr()
    assert out == ""
    assert "diff R2 upload failed" in err
    assert lab.manifest(jid).code.diff_ref is not None  # local copy kept (fail-closed)


def test_aggregate_mirror_failure_goes_to_stderr(tmp_path: Path, monkeypatch, capsys, failing_r2):
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    # Build the sweep with R2 off so submit itself is quiet; the mirror is what fails.
    monkeypatch.setattr(core_mod, "r2_enabled", lambda: False)
    sweep_id, _ = lab.sweep("true", {"N": [1]}, seeds="0", shard_size=1)
    jid = lab.sweep_plan(sweep_id).cells[0].shard_job_ids[0]
    lab.store.write_manifest(lab.manifest(jid).model_copy(update={"status": JobState.succeeded}))
    (lab.store.output_dir(jid) / "results.csv").write_text("seed,acc\n0,0.5\n")
    monkeypatch.setattr(core_mod, "r2_enabled", lambda: True)
    capsys.readouterr()
    lab.aggregate_sweep(sweep_id)
    out, err = capsys.readouterr()
    assert out == ""
    assert "aggregate R2 mirror failed" in err


def test_r2_fallback_unavailable_goes_to_stderr(tmp_path: Path, monkeypatch, capsys):
    lab = local_lab(make_repo(tmp_path))
    m = make_manifest("jr", "python x.py").model_copy(
        update={"status": JobState.succeeded}
    )
    m.backend.provisioner = "skypilot"  # a remote job: the R2 fallback is meaningful for it
    lab.store.create(m)
    monkeypatch.setattr(core_mod, "r2_enabled", lambda: True)

    class _NoBoto:
        @staticmethod
        def from_env():
            raise ImportError("No module named 'boto3'")

    monkeypatch.setattr(core_mod, "R2Store", _NoBoto)
    monkeypatch.setattr(lab.backend, "collect_artifacts", lambda *a, **k: [])
    lab.fetch_artifacts("jr")
    out, err = capsys.readouterr()
    assert out == ""
    assert "R2 fallback unavailable" in err


def test_export_fetch_failure_goes_to_stderr(tmp_path: Path, monkeypatch, capsys):
    lab = local_lab(make_repo(tmp_path))
    lab.store.create(
        make_manifest("je", "python x.py").model_copy(update={"status": JobState.succeeded})
    )

    def _boom(*a, **k):
        raise RuntimeError("collect blew up")

    monkeypatch.setattr(lab, "fetch_artifacts", _boom)
    lab.export("je", tmp_path / "bundle")
    out, err = capsys.readouterr()
    assert out == ""
    assert "fetch_artifacts(je) failed" in err


def test_no_bare_print_to_stdout_in_core_store_manifest():
    """Every print in the library core names a stream — a bare print is a stdout write."""
    src = Path(core_mod.__file__).parent
    offenders = []
    for name in ("core.py", "store.py", "manifest.py"):
        text = (src / name).read_text()
        for m in re.finditer(r"\bprint\(", text):
            depth, i = 1, m.end()
            while depth and i < len(text):
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
            if "file=sys.stderr" not in text[m.start() : i]:
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{name}:{line}")
    assert offenders == []
