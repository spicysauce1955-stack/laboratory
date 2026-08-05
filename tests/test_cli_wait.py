"""Regression: `lab wait` must exit non-zero when it gives up on a timeout (LAB-BUGS §1)."""

import json
import time

from typer.testing import CliRunner

import lab.cli as cli_mod
from helpers import make_manifest
from lab.cli import app
from lab.core import Lab
from lab.models import BackendInfo, JobState


class _SummaryMixin:
    """Borrow the real summary/settle logic so the fakes exercise Lab.wait_summary."""

    wait_summary = Lab.wait_summary
    _wait_summary_dict = Lab._wait_summary_dict
    _settle_teardown = Lab._settle_teardown


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

    class _FakeLab(_SummaryMixin):
        def wait(self, ids, *, interval, timeout, **kwargs):
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

    class _FakeLab(_SummaryMixin):
        def wait(self, ids, *, interval, timeout, **kwargs):
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

    class _FakeLab(_SummaryMixin):
        def wait(self, ids, *, interval, timeout, **kwargs):
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

    class _FakeLab(_SummaryMixin):
        def wait(self, ids, *, interval, timeout, **kwargs):
            return [at_terminal]  # null at the moment it went terminal

        def manifest(self, job_id):
            return settled  # teardown recorded a tick later

    _patch_store(monkeypatch, tmp_path, _FakeLab())
    done = tmp_path / "done.json"
    result = CliRunner().invoke(app, ["wait", "j1", "--done-file", str(done)])

    assert result.exit_code == 0
    assert json.loads(done.read_text())["teardown_unconfirmed"] == []


# ---------------------------------------------------------------------------
# Fail-fast + incremental done-file (field-report #3)
# ---------------------------------------------------------------------------


def _two_job_lab(tmp_path, statuses):
    """A fake Lab whose per-job status/manifest follow a scripted dict {job_id: JobState}."""

    class _Lab(_SummaryMixin):
        wait = Lab.wait  # real loop over the scripted statuses

        def status(self, job_id):
            return statuses[job_id]

        def manifest(self, job_id):
            return make_manifest(job_id, "python x.py").model_copy(
                update={
                    "status": statuses[job_id],
                    "backend": BackendInfo(provisioner="skypilot"),
                    "teardown_status": "succeeded"
                    if statuses[job_id]
                    in {JobState.succeeded, JobState.failed, JobState.timed_out}
                    else None,
                }
            )

    return _Lab()


def test_wait_fail_fast_returns_on_failed_job(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    statuses = {"j1": JobState.failed, "j2": JobState.running}
    lab = _two_job_lab(tmp_path, statuses)
    summary = lab.wait_summary(["j2", "j1"], interval=0.01, timeout=5, fail_fast=True)
    assert summary["failed_fast"] is True
    assert summary["all_terminal"] is False
    assert summary["pending"] == ["j2"]
    assert summary["jobs"][0]["job_id"] == "j1"  # offender first


def test_wait_fail_fast_ignores_preempted_and_cancelled(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    statuses = {"j1": JobState.preempted, "j2": JobState.cancelled, "j3": JobState.succeeded}
    lab = _two_job_lab(tmp_path, statuses)
    summary = lab.wait_summary(["j1", "j2", "j3"], interval=0.01, timeout=5, fail_fast=True)
    assert summary["failed_fast"] is False
    assert summary["all_terminal"] is True


def test_wait_on_update_emits_incremental_then_final(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    statuses = {"j1": JobState.succeeded, "j2": JobState.succeeded}
    lab = _two_job_lab(tmp_path, statuses)
    updates: list = []
    summary = lab.wait_summary(["j1", "j2"], interval=0.01, timeout=5, on_update=updates.append)
    assert len(updates) >= 2  # at least one per-job update + the final one
    assert updates[-1] == summary
    assert summary["pending"] == [] and summary["all_terminal"] is True


def test_wait_callback_error_never_aborts(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    statuses = {"j1": JobState.succeeded}
    lab = _two_job_lab(tmp_path, statuses)

    def _boom(_s):
        raise RuntimeError("watcher crashed")

    summary = lab.wait_summary(["j1"], interval=0.01, timeout=5, on_update=_boom)
    assert summary["all_terminal"] is True


def test_cli_wait_fail_fast_exits_4(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    failed = make_manifest("j1", "python x.py").model_copy(
        update={"status": JobState.failed, "backend": BackendInfo(provisioner="skypilot"),
                "teardown_status": "succeeded"}
    )

    class _FakeLab(_SummaryMixin):
        wait = Lab.wait

        def status(self, job_id):
            return JobState.failed if job_id == "j1" else JobState.running

        def manifest(self, job_id):
            if job_id == "j1":
                return failed
            return make_manifest("j2", "python x.py").model_copy(
                update={"status": JobState.running,
                        "backend": BackendInfo(provisioner="skypilot")}
            )

    _patch_store(monkeypatch, tmp_path, _FakeLab())
    done = tmp_path / "done.json"
    result = CliRunner().invoke(
        app, ["wait", "j1", "j2", "--fail-fast", "--done-file", str(done), "--timeout", "5"]
    )
    assert result.exit_code == 4
    summary = json.loads(done.read_text())
    assert summary["failed_fast"] is True and summary["pending"] == ["j2"]


def test_cli_wait_done_file_written_incrementally(monkeypatch, tmp_path):
    """The done-file must be valid, current JSON after each terminal event, not only at exit."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    snapshots: list = []
    term = _terminal("j1", provisioner="local", teardown=None)

    class _FakeLab(_SummaryMixin):
        def wait(self, ids, *, interval, timeout, fail_fast=False, on_terminal=None, **kw):
            if on_terminal is not None:
                on_terminal(term)
            return [term]

        def manifest(self, job_id):
            return term

    import lab.cli as _cli

    real_write = _cli.atomic_write_text

    def _spy(path, text):
        snapshots.append(json.loads(text))
        real_write(path, text)

    monkeypatch.setattr(_cli, "atomic_write_text", _spy)
    _patch_store(monkeypatch, tmp_path, _FakeLab())
    done = tmp_path / "done.json"
    result = CliRunner().invoke(app, ["wait", "j1", "--done-file", str(done)])
    assert result.exit_code == 0
    assert len(snapshots) >= 2  # incremental snapshot(s) + final
    assert all("pending" in s for s in snapshots)


def test_cli_wait_timeout_accepts_duration_string(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    term = _terminal("j1", provisioner="local", teardown=None)

    class _FakeLab(_SummaryMixin):
        seen: dict = {}

        def wait(self, ids, *, interval, timeout, **kw):
            _FakeLab.seen["timeout"] = timeout
            return [term]

        def manifest(self, job_id):
            return term

    _patch_store(monkeypatch, tmp_path, _FakeLab())
    result = CliRunner().invoke(app, ["wait", "j1", "--timeout", "2m"])
    assert result.exit_code == 0
    assert _FakeLab.seen["timeout"] == 120.0


def test_cli_wait_bad_timeout_string_is_usage_error(monkeypatch, tmp_path):
    term = _terminal("j1", provisioner="local", teardown=None)

    class _FakeLab(_SummaryMixin):
        def wait(self, ids, *, interval, timeout, **kw):
            return [term]

        def manifest(self, job_id):
            return term

    _patch_store(monkeypatch, tmp_path, _FakeLab())
    result = CliRunner().invoke(app, ["wait", "j1", "--timeout", "2hr"])
    assert result.exit_code == 2


def test_cli_wait_fail_fast_with_confirmed_leak_exits_3(monkeypatch, tmp_path):
    """A confirmed teardown leak coincident with --fail-fast must keep the URGENT exit 3 —
    the leak alarm outranks the fail-fast signal (CR finding; FR-C2)."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    leaked = make_manifest("j1", "python x.py").model_copy(
        update={"status": JobState.failed, "backend": BackendInfo(provisioner="skypilot"),
                "teardown_status": "failed"}
    )

    class _FakeLab(_SummaryMixin):
        wait = Lab.wait

        def status(self, job_id):
            return JobState.failed if job_id == "j1" else JobState.running

        def manifest(self, job_id):
            if job_id == "j1":
                return leaked
            return make_manifest("j2", "python x.py").model_copy(
                update={"status": JobState.running,
                        "backend": BackendInfo(provisioner="skypilot")}
            )

    _patch_store(monkeypatch, tmp_path, _FakeLab())
    result = CliRunner().invoke(app, ["wait", "j1", "j2", "--fail-fast", "--timeout", "5"])
    assert result.exit_code == 3  # leak alarm wins over fail-fast's 4


def test_wait_summary_settles_offender_teardown_on_fail_fast(monkeypatch, tmp_path):
    """On the fail-fast path the offender's lagging teardown_status must still settle —
    a merely-lagging null would otherwise read as unconfirmed instead of its real value."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    lagging = make_manifest("j1", "python x.py").model_copy(
        update={"status": JobState.failed, "backend": BackendInfo(provisioner="skypilot"),
                "teardown_status": None}
    )
    settled = lagging.model_copy(update={"teardown_status": "failed"})
    reads = {"n": 0}

    class _FakeLab(_SummaryMixin):
        wait = Lab.wait

        def status(self, job_id):
            return JobState.failed if job_id == "j1" else JobState.running

        def manifest(self, job_id):
            if job_id != "j1":
                return make_manifest("j2", "python x.py").model_copy(
                    update={"status": JobState.running,
                            "backend": BackendInfo(provisioner="skypilot")}
                )
            reads["n"] += 1
            return lagging if reads["n"] <= 2 else settled  # settles on a later re-read

    summary = _FakeLab().wait_summary(["j1", "j2"], interval=0.01, timeout=5, fail_fast=True)
    assert summary["failed_fast"] is True
    assert summary["teardown_leaks"] == ["j1"]  # settled to its real value, not unconfirmed
