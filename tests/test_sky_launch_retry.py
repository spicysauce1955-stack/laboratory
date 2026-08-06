"""Transient launch retry + submit stagger (field-report #4): a connection refusal from the
submitter's OWN local SkyPilot API server must not become a terminal job failure."""

from pathlib import Path

import lab.core as core_mod
import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.models import JobState
from lab.sky_runner import TransientLaunchError, _launch_with_retry, is_transient_launch_error
from lab.store import JobStore

FIELD_REPORT_ERROR = RuntimeError(
    "HTTPConnectionPool(host='127.0.0.1', port=46580): "
    "Max retries exceeded with url: /api/stream?x (Caused by NewConnectionError: "
    "Failed to establish a new connection: [Errno 111] Connection refused)"
)


def test_classifier_local_connection_refused_is_transient():
    assert is_transient_launch_error(FIELD_REPORT_ERROR) is True


def test_classifier_remote_connection_error_is_not_transient():
    e = RuntimeError(
        "HTTPConnectionPool(host='34.120.10.2', port=443): Max retries exceeded ... "
        "Connection refused"
    )
    assert is_transient_launch_error(e) is False  # cloud endpoint — fail toward alarm


def test_classifier_ordinary_error_is_not_transient():
    assert is_transient_launch_error(ValueError("boom")) is False
    assert is_transient_launch_error(RuntimeError("127.0.0.1 said no quota")) is False


def test_launch_retry_recovers_and_reuses_cluster_name():
    calls: list[str] = []
    sleeps: list[float] = []

    class _Sky:
        def launch(self, task, cluster_name, **kw):
            calls.append(cluster_name)
            if len(calls) < 3:
                raise FIELD_REPORT_ERROR
            return "request-1"

    out = _launch_with_retry(
        _Sky(), "task", "lab-x", attempts=3, base_s=0.01, sleep=sleeps.append
    )
    assert out == "request-1"
    assert calls == ["lab-x", "lab-x", "lab-x"]  # same cluster name every attempt
    assert len(sleeps) == 2 and all(s > 0 for s in sleeps)


def test_launch_retry_exhaustion_raises_transient():
    class _Sky:
        def launch(self, task, cluster_name, **kw):
            raise FIELD_REPORT_ERROR

    try:
        _launch_with_retry(_Sky(), "task", "lab-x", attempts=2, base_s=0.01, sleep=lambda s: None)
    except TransientLaunchError as e:
        assert "Connection refused" in str(e)
    else:
        raise AssertionError("expected TransientLaunchError")


def test_non_transient_launch_error_does_not_retry():
    calls: list[int] = []

    class _Sky:
        def launch(self, task, cluster_name, **kw):
            calls.append(1)
            raise ValueError("bad accelerator spec")

    try:
        _launch_with_retry(_Sky(), "task", "lab-x", attempts=3, base_s=0.01, sleep=lambda s: None)
    except ValueError:
        pass
    assert len(calls) == 1  # no retry on non-transient errors


def test_run_job_transient_exhaustion_marks_transient_and_tears_down(tmp_path, monkeypatch):
    import sys
    import types

    home = tmp_path / "runs"
    store = JobStore(home)
    m = make_manifest("jt1", "python x.py", timeout="10m")
    store.create(m)

    fake_sky = types.ModuleType("sky")
    monkeypatch.setitem(sys.modules, "sky", fake_sky)

    def _launch(*a, **k):
        raise FIELD_REPORT_ERROR

    monkeypatch.setattr(fake_sky, "launch", _launch, raising=False)
    monkeypatch.setattr(runner_mod, "build_task", lambda *a, **k: "task")
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
    teardowns: list[str] = []
    monkeypatch.setattr(
        runner_mod, "tear_down_and_record",
        lambda sky, cluster, st, jid, cloud, **kw: teardowns.append(cluster) or True,
    )

    rc = runner_mod.run_job(home / "jt1")

    assert rc == 1
    final = store.read_manifest("jt1")
    assert final.status is JobState.failed
    assert (final.end_reason or "").startswith("transient:")
    assert len(teardowns) == 1  # still torn down (FR-C2)


def test_sweep_staggers_between_remote_submits(tmp_path, monkeypatch):
    submitted: list[str] = []
    sleeps: list[float] = []

    class _FakeBackend:
        name = "skypilot"

        def submit(self, manifest):
            submitted.append(manifest.job_id)
            return manifest.job_id

    lab = Lab(backend=_FakeBackend(), repo=tmp_path, home=tmp_path)  # type: ignore[arg-type]
    monkeypatch.setattr(core_mod, "current_commit", lambda repo: "0" * 40)
    monkeypatch.setattr(core_mod, "commit_exists", lambda repo, ref: True)
    monkeypatch.setattr(core_mod, "is_dirty", lambda repo: False)
    monkeypatch.setattr(core_mod, "uv_lock_sha256", lambda p: "x")
    monkeypatch.setattr(core_mod.time, "sleep", sleeps.append)
    monkeypatch.setenv("LAB_SUBMIT_STAGGER_S", "1.5")

    lab.sweep("python x.py", {"a": ["1", "2", "3"]})
    assert len(submitted) == 3
    assert sleeps == [1.5, 1.5]  # between submits, not after the last


def test_sweep_no_stagger_for_local_backend(tmp_path, monkeypatch):
    sleeps: list[float] = []
    repo = Path.cwd()
    from lab.manifest import repo_root

    repo = repo_root(repo)
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    monkeypatch.setattr(core_mod.time, "sleep", sleeps.append)
    monkeypatch.setenv("LAB_SUBMIT_STAGGER_S", "1.5")
    lab.sweep("true", {"a": ["1", "2"]})
    assert sleeps == []  # local backend never staggers
