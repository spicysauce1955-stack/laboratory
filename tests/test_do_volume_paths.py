"""Every DO teardown path verifies volumes the same way — including the ones outside the supervisor.

``lab cancel`` tears down in the *caller's* process (and SIGTERMs the supervisor, which tears down
too), and a supervisor that died is finished by ``status()``'s recovery branch. Neither can see the
supervisor's memory except through ``_runtime.json``, and both must recognise DO's reworded detach
rejection. Job F (2026-11-19) was exactly such a cancel: its volume was mid-detach, DO said
"cannot delete attached volume", and the teardown recorded ``failed``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from fake_do import NEW_DETACH, FakeDo, FakeSky
from helpers import make_manifest

from lab.backends import skypilot as m
from lab.backends.skypilot import SkyPilotBackend
from lab.models import BackendInfo, JobState

JOB = "20261119-111134-b048d2"
CLUSTER = f"lab-smoke-{JOB}"
VOLUME = f"{CLUSTER}-3dd12990-ef65-head"


def _backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, do: FakeDo) -> SkyPilotBackend:
    backend = SkyPilotBackend(home=tmp_path, repo=tmp_path)
    manifest = make_manifest(JOB, "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running, "backend": BackendInfo(provisioner="skypilot")}
    )
    manifest.resources.cloud = "do"
    backend.store.create(manifest)
    # A supervisor pid that is not alive: `cancel` signals nothing, `status` sees it dead.
    backend.store.write_runtime(JOB, runner_pid=999999999, cluster=CLUSTER)
    monkeypatch.setattr(m, "_get_do_client", lambda: do)
    monkeypatch.setattr(m.time, "sleep", do.sleep)
    monkeypatch.setitem(sys.modules, "sky", FakeSky(do))
    return backend


def _job_f(**kw: Any) -> FakeDo:
    """Job F's account: a live droplet with its volume attached; once the droplet goes, DO's own
    background delete of the volume does not land, and the volume detaches 12 s later."""
    kw.setdefault("detach_s", 12.0)
    kw.setdefault("associated_delete", False)
    return FakeDo(
        droplets=[{"id": 31, "name": VOLUME}],
        volumes=[{"id": "vol-f", "name": VOLUME, "droplet_ids": [31]}],
        detach_message=NEW_DETACH,
        **kw,
    )


class TestLabCancel:
    def test_job_f_a_cancelled_jobs_detaching_volume_is_not_a_false_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _job_f()
        backend = _backend(tmp_path, monkeypatch, do)

        assert backend.cancel(JOB) is JobState.cancelled

        manifest = backend.store.read_manifest(JOB)
        assert manifest.teardown_status == "succeeded", manifest.end_reason
        assert "vol-f" not in do.vols
        assert manifest.teardown_detail is not None
        assert manifest.teardown_detail["do_volumes"]["gone"] == ["vol-f"]

    def test_cancel_verifies_the_volume_the_supervisor_remembered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _job_f(detach_s=0.0)
        do.hidden_until["vol-f"] = 1e9  # no listing will ever show it
        backend = _backend(tmp_path, monkeypatch, do)
        backend.store.write_runtime(JOB, do_volume_ids=["vol-f"])

        backend.cancel(JOB)

        assert backend.store.read_manifest(JOB).teardown_status == "succeeded"
        assert "vol-f" not in do.vols


class TestDeadSupervisorRecovery:
    def test_recovery_verifies_the_volume_the_supervisor_remembered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _job_f(detach_s=0.0)
        do.hidden_until["vol-f"] = 1e9
        backend = _backend(tmp_path, monkeypatch, do)
        backend.store.write_runtime(JOB, do_volume_ids=["vol-f"])

        backend.status(JOB)

        manifest = backend.store.read_manifest(JOB)
        assert manifest.teardown_status == "succeeded"
        assert "vol-f" not in do.vols

    def test_recovery_that_cannot_verify_a_remembered_volume_is_not_succeeded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _job_f(detach_s=0.0)
        do.hidden_until["vol-f"] = 1e9
        do.get_error = RuntimeError("read timeout")
        do.refuse["vol-f"] = RuntimeError("connection reset by peer")
        backend = _backend(tmp_path, monkeypatch, do)
        backend.store.write_runtime(JOB, do_volume_ids=["vol-f"])

        backend.status(JOB)

        assert backend.store.read_manifest(JOB).teardown_status == "unknown"
