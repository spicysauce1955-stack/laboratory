"""`lab cancel` must not report a job finished before its machine is actually released (R9).

Live evidence, 2026-08-20. Seven `lab cancel` calls fired 90 seconds apart, each from a distinct
session -- an external watchdog with a 90s timeout. Every one of them was killed while blocked
inside ``robust_teardown``'s multi-minute retry ladder, and the ledger recorded all seven as
``running-or-died`` (opened, never closed). What they left behind on disk was worse than nothing:

    status           = cancelled          <- terminal: `lab wait` is satisfied, dashboards go quiet
    teardown_status  = None               <- no leak signal at all
    end_reason       = "cancelled by user"

A machine that may still be billing, wearing the manifest of a cleanly finished job. The cause is
ordering: ``cancel()`` wrote the terminal status *first*, then did the slow part. Anything that
interrupted the slow part froze that lie in place.

The contract these tests pin: a cancel that does not complete leaves the job **non-terminal**, so
the existing dead-supervisor recovery in ``status()`` can finish the job later. The terminal
status is written last, once teardown has actually been attempted.
"""

import signal

import pytest
from helpers import make_manifest

from lab.backends.skypilot import SkyPilotBackend, cluster_name_for
from lab.models import BackendInfo, JobState


def _backend_with_running_job(tmp_path, job_id, *, cloud="do"):
    backend = SkyPilotBackend(home=tmp_path, repo=tmp_path)
    m = make_manifest(job_id, "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running, "backend": BackendInfo(provisioner="skypilot")}
    )
    m.resources.cloud = cloud
    backend.store.create(m)
    backend.store.write_runtime(job_id, runner_pid=999999999, cluster=cluster_name_for(job_id))
    return backend


def _fake_sky(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("sky")
    fake.get = lambda x: x
    fake.cancel = lambda cluster, all=False: None
    fake.down = lambda cluster: None
    monkeypatch.setitem(sys.modules, "sky", fake)
    return fake


class TestInterruptedCancelDoesNotLookClean:
    def test_a_cancel_killed_during_teardown_leaves_the_job_non_terminal(
        self, tmp_path, monkeypatch
    ):
        """The exact 2026-08-20 shape: the caller dies inside the retry ladder."""
        backend = _backend_with_running_job(tmp_path, "jc1")
        _fake_sky(monkeypatch)

        def _killed(*a, **k):
            raise KeyboardInterrupt("watchdog timeout at 90s")

        monkeypatch.setattr("lab.backends.skypilot.tear_down_and_record", _killed)

        with pytest.raises(KeyboardInterrupt):
            backend.cancel("jc1")

        m = backend.store.read_manifest("jc1")
        assert m.status is not JobState.cancelled, (
            "a terminal status with no teardown record is what hid seven possibly-billing machines"
        )
        assert m.teardown_status != "succeeded"

    def test_the_cancel_intent_survives_so_recovery_can_finish_it(self, tmp_path, monkeypatch):
        """Non-terminal alone is not enough -- something has to know a cancel was wanted."""
        backend = _backend_with_running_job(tmp_path, "jc2")
        _fake_sky(monkeypatch)
        monkeypatch.setattr(
            "lab.backends.skypilot.tear_down_and_record",
            lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

        with pytest.raises(KeyboardInterrupt):
            backend.cancel("jc2")

        assert backend.store.read_runtime("jc2").get("cancelling") is True

    def test_a_sigterm_mid_teardown_is_handled_the_same_way(self, tmp_path, monkeypatch):
        """`timeout` sends SIGTERM, not SIGINT -- the watchdog's actual mechanism."""
        backend = _backend_with_running_job(tmp_path, "jc3")
        _fake_sky(monkeypatch)

        def _sigtermed(*a, **k):
            raise SystemExit(signal.SIGTERM)

        monkeypatch.setattr("lab.backends.skypilot.tear_down_and_record", _sigtermed)

        with pytest.raises(SystemExit):
            backend.cancel("jc3")

        assert backend.store.read_manifest("jc3").status is not JobState.cancelled


class TestCompletedCancelIsUnchanged:
    def test_a_cancel_that_finishes_records_cancelled_last(self, tmp_path, monkeypatch):
        """No regression: the happy path still ends `cancelled` with teardown recorded."""
        backend = _backend_with_running_job(tmp_path, "jc4")
        _fake_sky(monkeypatch)
        order: list[str] = []

        def _teardown(sky_mod, cluster, store, job_id, cloud, **kw):
            order.append("teardown")
            store.update_manifest(job_id, teardown_status="succeeded")
            return True

        monkeypatch.setattr("lab.backends.skypilot.tear_down_and_record", _teardown)

        assert backend.cancel("jc4") is JobState.cancelled

        m = backend.store.read_manifest("jc4")
        assert order == ["teardown"], "teardown must run before the terminal status is written"
        assert m.status is JobState.cancelled
        assert m.teardown_status == "succeeded"
        assert m.end_reason == "cancelled by user"

    def test_an_already_terminal_job_is_a_no_op(self, tmp_path, monkeypatch):
        backend = _backend_with_running_job(tmp_path, "jc5")
        backend.store.update_manifest("jc5", status=JobState.succeeded)
        monkeypatch.setattr(
            "lab.backends.skypilot.tear_down_and_record",
            lambda *a, **k: pytest.fail("must not tear down an already-terminal job"),
        )

        assert backend.cancel("jc5") is JobState.succeeded

    def test_teardown_uses_the_cluster_recorded_at_launch(self, tmp_path, monkeypatch):
        """Names gained a project slug; recomputing one can address the wrong machine."""
        backend = _backend_with_running_job(tmp_path, "jc6")
        backend.store.write_runtime("jc6", cluster="lab-legacy-name-from-an-older-release")
        _fake_sky(monkeypatch)
        seen: list[str] = []
        monkeypatch.setattr(
            "lab.backends.skypilot.tear_down_and_record",
            lambda sky_mod, cluster, store, job_id, cloud, **kw: seen.append(cluster) or True,
        )

        backend.cancel("jc6")

        assert seen == ["lab-legacy-name-from-an-older-release"]


class TestInterruptedCancelIsRecoverable:
    def test_status_finishes_an_interrupted_cancel(self, tmp_path, monkeypatch):
        """Leaving the job non-terminal is only safe if something later completes it.

        The supervisor was already SIGTERMed by the interrupted cancel, so `status()`'s existing
        dead-supervisor branch is the natural place: it already attempts teardown and records a
        terminal state. It must attribute the outcome to the cancel, not to a mystery crash.
        """
        backend = _backend_with_running_job(tmp_path, "jc7")
        _fake_sky(monkeypatch)
        monkeypatch.setattr(
            "lab.backends.skypilot.tear_down_and_record",
            lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            backend.cancel("jc7")

        torn: list[str] = []
        monkeypatch.setattr(
            "lab.backends.skypilot.tear_down_and_record",
            lambda sky_mod, cluster, store, job_id, cloud, **kw: torn.append(cluster) or True,
        )

        state = backend.status("jc7")

        assert torn, "recovery must attempt the teardown the interrupted cancel never finished"
        assert state is JobState.cancelled
        m = backend.store.read_manifest("jc7")
        assert m.end_reason is not None and "cancel" in m.end_reason.lower()
