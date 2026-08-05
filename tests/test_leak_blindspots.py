"""The three correlated leak-detector blind spots (review roadmap item 2): a dead supervisor
must not make reconcile, backend.status, or the scheduler watchdog read a billing box as safe."""

from datetime import timedelta


from helpers import make_manifest
from lab._util import now
from lab.backends.local import LocalBackend
from lab.backends.skypilot import SkyPilotBackend, cluster_name_for
from lab.core import Lab
from lab.models import BackendInfo, JobState


def _dead_running_manifest(job_id: str, *, started_ago_s: float = 3600.0):
    """A skypilot job stuck `running` whose supervisor is gone."""
    return make_manifest(job_id, "python x.py", timeout="1h").model_copy(
        update={
            "status": JobState.running,
            "started_at": now() - timedelta(seconds=started_ago_s),
            "backend": BackendInfo(provisioner="skypilot"),
        }
    )


def _patch_empty_sky(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("sky")
    fake.get = lambda x: x
    fake.status = lambda refresh=None: []
    fake.StatusRefreshMode = types.SimpleNamespace(AUTO="AUTO", FORCE="FORCE", NONE="NONE")
    monkeypatch.setitem(sys.modules, "sky", fake)


# ---------------------------------------------------------------------------
# A) reconcile: dead-supervisor `running` jobs must not protect their cluster
# ---------------------------------------------------------------------------


def test_reconcile_flags_unsupervised_job_and_frees_its_rental(tmp_path, monkeypatch):
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
    m = _dead_running_manifest("jdead")
    lab.store.create(m)
    lab.store.write_runtime("jdead", runner_pid=999999999, cluster=cluster_name_for("jdead"))

    label = f"{cluster_name_for('jdead')}-1a2b-head"
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances", lambda *a, **k: [{"id": 7, "label": label}]
    )
    _patch_empty_sky(monkeypatch)
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])

    report = lab.reconcile()

    assert report["unsupervised"] == [
        {"job_id": "jdead", "cluster": cluster_name_for("jdead")}
    ]
    # The still-billing rental is now visible as an orphan instead of counted healthy.
    assert [o["id"] for o in report["orphans"]] == [7]


def test_reconcile_grace_period_protects_fresh_jobs(tmp_path, monkeypatch):
    """A just-started job whose runtime/pid isn't settled yet must NOT be torn down."""
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
    m = _dead_running_manifest("jfresh", started_ago_s=30)
    lab.store.create(m)
    lab.store.write_runtime("jfresh", runner_pid=999999999, cluster=cluster_name_for("jfresh"))

    label = f"{cluster_name_for('jfresh')}-1a2b-head"
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances", lambda *a, **k: [{"id": 9, "label": label}]
    )
    _patch_empty_sky(monkeypatch)
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])

    report = lab.reconcile()
    assert report["unsupervised"] == []
    assert report["orphans"] == []  # still protected during the grace window


def test_reconcile_live_supervisor_still_protects(tmp_path, monkeypatch):
    """A running job with a LIVE supervisor pid keeps protecting its cluster (no regression)."""
    import os

    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
    m = _dead_running_manifest("jlive")
    lab.store.create(m)
    lab.store.write_runtime("jlive", runner_pid=os.getpid(), cluster=cluster_name_for("jlive"))

    label = f"{cluster_name_for('jlive')}-1a2b-head"
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances", lambda *a, **k: [{"id": 4, "label": label}]
    )
    _patch_empty_sky(monkeypatch)
    monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])

    report = lab.reconcile()
    assert report["unsupervised"] == []
    assert report["orphans"] == []


# ---------------------------------------------------------------------------
# B) SkyPilotBackend.status: the dead-pid branch must attempt teardown
# ---------------------------------------------------------------------------


def test_status_dead_supervisor_attempts_teardown(tmp_path, monkeypatch):
    backend = SkyPilotBackend(home=tmp_path)
    m = _dead_running_manifest("jd2")
    m.resources.cloud = "gcp"
    backend.store.create(m)
    backend.store.write_runtime("jd2", runner_pid=999999999, cluster=cluster_name_for("jd2"))

    calls: list[tuple] = []

    def _fake_tdr(sky_mod, cluster, store, job_id, cloud="vast", **kw):
        calls.append((cluster, job_id, cloud))
        store.update_manifest(job_id, teardown_status="succeeded")
        return True

    monkeypatch.setattr("lab.backends.skypilot.tear_down_and_record", _fake_tdr)

    state = backend.status("jd2")

    assert state is JobState.failed
    assert calls == [(cluster_name_for("jd2"), "jd2", "gcp")]
    final = backend.store.read_manifest("jd2")
    assert final.teardown_status == "succeeded"
    # Terminal now — polling again must not re-run teardown.
    assert backend.status("jd2") is JobState.failed
    assert len(calls) == 1


def test_status_dead_supervisor_teardown_crash_flags_failed(tmp_path, monkeypatch):
    """If the teardown attempt itself blows up, the manifest must carry the leak alarm
    (teardown_status='failed') so `lab wait` exits 3 instead of 0."""
    backend = SkyPilotBackend(home=tmp_path)
    m = _dead_running_manifest("jd3")
    backend.store.create(m)
    backend.store.write_runtime("jd3", runner_pid=999999999, cluster=cluster_name_for("jd3"))

    monkeypatch.setattr(
        "lab.backends.skypilot.tear_down_and_record",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sky exploded")),
    )

    state = backend.status("jd3")
    assert state is JobState.failed
    assert backend.store.read_manifest("jd3").teardown_status == "failed"


# ---------------------------------------------------------------------------
# C) scheduler watchdog liveness: label matching, uncertainty reads as alive
# ---------------------------------------------------------------------------


def _make_sched(tmp_path):
    from lab.scheduler.queue import LocalQueueStore
    from lab.scheduler.tick import Scheduler

    return Scheduler(LocalQueueStore(tmp_path / "queue"), home=tmp_path / "runs")


def test_cluster_alive_vast_matches_labels_not_price(tmp_path, monkeypatch):
    sched = _make_sched(tmp_path)
    monkeypatch.setattr(
        "lab.backends.skypilot.vast_hourly_for_cluster",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("price used for liveness")),
    )
    # Rental present but with no/unparseable dph_total — must still read ALIVE.
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances",
        lambda *a, **k: [{"id": 1, "label": "lab-x-1a2b-head", "dph_total": None}],
    )
    assert sched._cluster_alive("lab-x", cloud="vast") is True

    monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
    assert sched._cluster_alive("lab-x", cloud="vast") is False


def test_cluster_alive_vast_api_error_reads_alive(tmp_path, monkeypatch):
    """Listing failure = unknown; must NOT read as 'gone' (which triggers failed-with-no-teardown)."""
    sched = _make_sched(tmp_path)
    monkeypatch.setattr(
        "lab.backends.skypilot.list_vast_instances",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down")),
    )
    assert sched._cluster_alive("lab-x", cloud="vast") is True
