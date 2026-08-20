"""A cluster that disappears mid-run must end the job now, not at the timeout (R8).

Live evidence (2026-08-20, job 20260820-071913-be3c72, still on disk at
``tempotron-capacity/runs/20260820-071913-be3c72/logs.txt``): SkyPilot answered
``Cluster 'lab-20260820-071913-be3c72' does not exist.`` **65 consecutive times** and the
supervisor kept polling, because every poll exception was printed and swallowed. ``max_wait``
is ``timeout + 300``, so the job stayed ``running`` for a possible 125 minutes and only stopped
because an external watchdog cancelled it.

The counterweight these tests pin just as hard: a poll error that is *not* definitive must
still be tolerated. Turning a blip into a terminal failure would abandon a healthy — and still
billing — machine, which is the same class of bug in the other direction.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab._util import now
from lab.models import CostInfo, JobState
from lab.sky_runner import _wait_terminal
from lab.store import JobStore


# The live wire message, ANSI colouring and all — sky yellows its warnings, and the supervisor
# stores whatever it is handed.
LIVE_MESSAGE = "\x1b[33mCluster 'lab-20260820-071913-be3c72' does not exist.\x1b[0m"


def _named(name: str) -> type[Exception]:
    """An exception class standing in for one of sky's, matched by type name.

    ``lab._skycompat.classify_sky_error`` deliberately matches on ``type(exc).__name__`` so it
    works without importing the optional ``skypilot`` extra; these tests exercise it the same
    way, which keeps them hermetic and independent of the installed sky version.
    """
    return type(name, (Exception,), {})


ClusterDoesNotExist = _named("ClusterDoesNotExist")
ApiServerConnectionError = _named("ApiServerConnectionError")


class _Status:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSky:
    """Minimal stand-in for the ``sky`` module as ``_wait_terminal`` uses it."""

    def __init__(self, exc: BaseException | None = None, names: list[str] | None = None) -> None:
        self.exc = exc
        self.names = names or []
        self.polls = 0

    def get(self, x: Any) -> Any:
        return x

    def queue(self, cluster: str, skip_finished: bool = False) -> Any:
        self.polls += 1
        if self.exc is not None:
            raise self.exc
        name = self.names[min(self.polls - 1, len(self.names) - 1)]
        return [{"job_id": 1, "status": _Status(name)}]


# ---------------------------------------------------------------------------
# _wait_terminal: which poll errors are definitive
# ---------------------------------------------------------------------------


def test_cluster_gone_ends_the_wait_immediately(monkeypatch) -> None:
    monkeypatch.setattr(runner_mod.time, "sleep", lambda _s: None)
    sky = _FakeSky(ClusterDoesNotExist("Cluster 'lab-x' does not exist."))

    state, reached, lost = _wait_terminal(sky, "lab-x", None, max_wait=7200, poll_s=0.01)

    assert sky.polls <= 3, "must not keep polling a cluster that is definitively gone"
    assert state is JobState.failed
    assert reached is False
    assert lost is not None and "does not exist" in lost


def test_the_exact_live_signature_ends_the_wait(monkeypatch) -> None:
    """The 65-times-repeated message from the 2026-08-20 incident, verbatim."""
    monkeypatch.setattr(runner_mod.time, "sleep", lambda _s: None)
    sky = _FakeSky(ClusterDoesNotExist(LIVE_MESSAGE))

    _, _, lost = _wait_terminal(sky, "lab-20260820-071913-be3c72", 1, max_wait=7200, poll_s=0.01)

    assert sky.polls == 1
    assert lost is not None and "does not exist" in lost


def test_a_wrapped_cluster_gone_is_still_definitive(monkeypatch) -> None:
    """sky re-wraps freely, so the verdict must follow ``__cause__``/``__context__``."""
    monkeypatch.setattr(runner_mod.time, "sleep", lambda _s: None)
    wrapped = RuntimeError("request failed")
    wrapped.__cause__ = ClusterDoesNotExist("Cluster 'lab-x' does not exist.")
    sky = _FakeSky(wrapped)

    state, reached, lost = _wait_terminal(sky, "lab-x", None, max_wait=7200, poll_s=0.01)

    assert sky.polls == 1
    assert state is JobState.failed and reached is False
    assert lost is not None and "does not exist" in lost


def test_a_transient_poll_error_is_still_tolerated() -> None:
    """Only a *definitive* answer ends the wait; a blip must not fail a healthy job."""
    sky = _FakeSky(TimeoutError("read timed out"))

    state, reached, lost = _wait_terminal(sky, "lab-x", None, max_wait=0.05, poll_s=0.01)

    assert sky.polls > 1
    assert lost is None
    assert reached is False
    assert state is JobState.failed  # deadline reached with no terminal status: unchanged


def test_an_api_server_blip_does_not_end_the_wait() -> None:
    """A local API server that is down says nothing about the remote box.

    ``classify_sky_error`` calls ``ApiServerConnectionError`` a ``failed`` *call* — correct for a
    destroy (the request never left the client, so nothing was destroyed), wrong as evidence
    about a cluster. Ending the wait here would mark a healthy job failed and walk away from a
    machine that is still billing, so the cluster-gone test is narrower than "the verdict is
    failed".
    """
    sky = _FakeSky(ApiServerConnectionError("Could not connect to SkyPilot API server"))

    _, _, lost = _wait_terminal(sky, "lab-x", None, max_wait=0.05, poll_s=0.01)

    assert sky.polls > 1
    assert lost is None


def test_a_normal_run_still_reaches_terminal(monkeypatch) -> None:
    """Regression: the happy path's contract is unchanged apart from the new third element."""
    monkeypatch.setattr(runner_mod.time, "sleep", lambda _s: None)
    sky = _FakeSky(names=["RUNNING", "RUNNING", "SUCCEEDED"])

    state, reached, lost = _wait_terminal(sky, "lab-x", 1, max_wait=10_000, poll_s=1.0)

    assert state is JobState.succeeded
    assert reached is True
    assert lost is None


# ---------------------------------------------------------------------------
# run_job: how a lost cluster is recorded, and that teardown still happens
# ---------------------------------------------------------------------------


def _prep(tmp_path: Path, job_id: str) -> JobStore:
    store = JobStore(tmp_path / "runs")
    store.create(
        make_manifest(job_id, "python x.py", timeout="120s").model_copy(
            update={
                "status": JobState.running,
                "started_at": now(),
                "cost": CostInfo(hourly_usd=0.2, estimated_usd=0.2),
            }
        )
    )
    store.write_runtime(job_id, runner_pid=1, cluster=f"lab-{job_id}")
    return store


def test_a_lost_cluster_finalises_failed_and_still_tears_down(tmp_path: Path, monkeypatch) -> None:
    """End to end through the real ``_wait_terminal``: the manifest must stop lying, and the
    teardown path must not be skipped — a vanished SkyPilot registration is exactly when a
    provider-side rental is most likely to survive (FR-C2)."""
    store = _prep(tmp_path, "cl1")

    fake_sky = types.ModuleType("sky")
    fake_sky.get = lambda x: x  # type: ignore[attr-defined]
    fake_sky.tail_logs = lambda *a, **k: None  # type: ignore[attr-defined]

    def _queue(cluster: str, skip_finished: bool = False) -> Any:
        raise ClusterDoesNotExist(LIVE_MESSAGE)

    fake_sky.queue = _queue  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky", fake_sky)
    monkeypatch.setattr(runner_mod, "_resolve_hourly", lambda *a, **k: 0.2)
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "r2_enabled", lambda: False)

    teardown_calls: list[str] = []

    def _spy(sky_mod: Any, cluster: str, st: Any, jid: str, cloud: str = "vast") -> bool:
        teardown_calls.append(cluster)
        st.update_manifest(jid, teardown_status="succeeded")
        return True

    monkeypatch.setattr(runner_mod, "tear_down_and_record", _spy)

    rc = runner_mod.run_job(store.job_dir("cl1"), adopt=True)

    m = store.read_manifest("cl1")
    assert m.status is JobState.failed
    assert m.ended_at is not None
    assert m.end_reason is not None and m.end_reason.startswith("cluster disappeared mid-run")
    assert "does not exist" in m.end_reason
    assert len(m.end_reason) <= 300
    # Distinct from the neighbouring terminal reasons, so the ledger can tell them apart.
    assert "provisioning exceeded" not in m.end_reason
    assert m.end_reason != "cancelled by user"
    # The leak net still ran: a job may not end with teardown unattempted.
    assert teardown_calls == ["lab-cl1"]
    assert m.teardown_status == "succeeded"
    assert rc == 1


def test_a_lost_cluster_records_a_leak_when_teardown_fails(tmp_path: Path, monkeypatch) -> None:
    """If the machine really is unreachable the destroy may fail — that must surface as a leak,
    not be swallowed because the cluster was "already gone"."""
    store = _prep(tmp_path, "cl2")

    fake_sky = types.ModuleType("sky")
    fake_sky.get = lambda x: x  # type: ignore[attr-defined]
    fake_sky.tail_logs = lambda *a, **k: None  # type: ignore[attr-defined]

    def _queue(cluster: str, skip_finished: bool = False) -> Any:
        raise ClusterDoesNotExist(LIVE_MESSAGE)

    fake_sky.queue = _queue  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky", fake_sky)
    monkeypatch.setattr(runner_mod, "_resolve_hourly", lambda *a, **k: 0.2)
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "r2_enabled", lambda: False)

    def _spy(sky_mod: Any, cluster: str, st: Any, jid: str, cloud: str = "vast") -> bool:
        st.update_manifest(jid, teardown_status="failed")
        return False

    monkeypatch.setattr(runner_mod, "tear_down_and_record", _spy)

    runner_mod.run_job(store.job_dir("cl2"), adopt=True)

    m = store.read_manifest("cl2")
    assert m.status is JobState.failed
    assert m.teardown_status == "failed"  # `lab wait` exits 3 on this
