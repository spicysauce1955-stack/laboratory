"""Regression: `lab wait` must exit non-zero when it gives up on a timeout (LAB-BUGS §1)."""

import json
import time

from typer.testing import CliRunner

import lab.cli as cli_mod
from helpers import make_manifest
from lab.cli import app
from lab.models import BackendInfo, JobState


def _patch_store(monkeypatch, tmp_path, fake_lab):
    """Wire `lab wait` to a fake lab + a store whose manifest paths all 'exist'."""

    class _FakeStore:
        def __init__(self, home):
            pass

        def manifest_path(self, job_id):
            p = tmp_path / f"{job_id}.json"
            p.touch()
            return p

    monkeypatch.setattr(cli_mod, "_lab_for", lambda job_id: fake_lab)
    monkeypatch.setattr(cli_mod, "JobStore", _FakeStore)


def test_wait_exits_1_on_timeout_without_completion(monkeypatch, tmp_path):
    running = make_manifest("j1", "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running}
    )

    class _FakeLab:
        def wait(self, ids, *, interval, timeout):
            return [running]  # never reached terminal -> the timeout path

    class _FakeStore:
        def __init__(self, home):
            pass

        def manifest_path(self, job_id):
            p = tmp_path / f"{job_id}.json"
            p.touch()  # exists -> passes the "unknown job id" guard
            return p

    monkeypatch.setattr(cli_mod, "_lab_for", lambda job_id: _FakeLab())
    monkeypatch.setattr(cli_mod, "JobStore", _FakeStore)

    result = CliRunner().invoke(app, ["wait", "j1", "--timeout", "0.5"])
    assert result.exit_code == 1


def _terminal(job_id, *, provisioner, teardown):
    return make_manifest(job_id, "python x.py").model_copy(
        update={
            "status": JobState.succeeded,
            "backend": BackendInfo(provisioner=provisioner),
            "teardown_status": teardown,
        }
    )


def test_wait_flags_unconfirmed_teardown_on_remote_job(monkeypatch, tmp_path):
    """A terminal remote job whose teardown_status never settles (null, not 'failed') must be
    surfaced as teardown_unconfirmed + a warning — not pass as a silent clean exit 0."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    term = _terminal("j1", provisioner="skypilot", teardown=None)

    class _FakeLab:
        def wait(self, ids, *, interval, timeout):
            return [term]

        def manifest(self, job_id):
            return term  # stays unconfirmed across the settle re-reads

    _patch_store(monkeypatch, tmp_path, _FakeLab())
    done = tmp_path / "done.json"
    result = CliRunner().invoke(app, ["wait", "j1", "--done-file", str(done)])

    assert result.exit_code == 0  # not a confirmed leak (that's exit 3) — but no longer silent
    summary = json.loads(done.read_text())
    assert summary["teardown_unconfirmed"] == ["j1"]
    assert summary["teardown_leaks"] == []
    assert "reconcile" in result.output  # actionable warning surfaced


def test_wait_does_not_flag_local_job(monkeypatch, tmp_path):
    """A local job has nothing to tear down (teardown_status is always null) — never flag it."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    term = _terminal("j1", provisioner="local", teardown=None)

    class _FakeLab:
        def wait(self, ids, *, interval, timeout):
            return [term]

        def manifest(self, job_id):
            return term

    _patch_store(monkeypatch, tmp_path, _FakeLab())
    done = tmp_path / "done.json"
    result = CliRunner().invoke(app, ["wait", "j1", "--done-file", str(done)])

    assert result.exit_code == 0
    assert json.loads(done.read_text())["teardown_unconfirmed"] == []


def test_wait_settles_a_lagging_teardown(monkeypatch, tmp_path):
    """A teardown that is merely lagging (null at terminal, then recorded) settles on re-read and is
    NOT flagged — avoids false positives from mirror lag."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    at_terminal = _terminal("j1", provisioner="skypilot", teardown=None)
    settled = _terminal("j1", provisioner="skypilot", teardown="succeeded")

    class _FakeLab:
        def wait(self, ids, *, interval, timeout):
            return [at_terminal]  # null at the moment it went terminal

        def manifest(self, job_id):
            return settled  # teardown recorded a tick later

    _patch_store(monkeypatch, tmp_path, _FakeLab())
    done = tmp_path / "done.json"
    result = CliRunner().invoke(app, ["wait", "j1", "--done-file", str(done)])

    assert result.exit_code == 0
    assert json.loads(done.read_text())["teardown_unconfirmed"] == []
