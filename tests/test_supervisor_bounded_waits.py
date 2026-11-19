"""No SkyPilot call may block the supervisor without a bound (review finding).

``sky.tail_logs(follow=True)`` and ``sky.get(sky.queue(...))`` had no timeout. A half-open stream
(the API server's connection silently dead) or a wedged queue request blocked the supervisor
forever — and the local wall-clock backstop, which is only checked *between* those calls, never
fired: the manifest said ``running`` and nothing stopped the box but its own watchdog.

The stream now runs on a daemon thread joined against the remaining wall budget, and is probed
periodically so even an uncapped job notices that its run ended under a stream that did not;
every queue/status poll is bounded and a poll that times out is just "no answer yet".
"""

from __future__ import annotations

import sys
import threading
import types
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab._util import now
from lab.backends.skypilot import SUCCESS_SENTINEL
from lab.models import CostInfo, JobState
from lab.store import JobStore


class _Status:
    def __init__(self, name: str) -> None:
        self.name = name


def _store(tmp_path: Path, *, timeout: str | None, started_ago: timedelta) -> JobStore:
    store = JobStore(tmp_path / "runs")
    m = make_manifest("j1", "python x.py", timeout=timeout).model_copy(
        update={
            "status": JobState.running,
            "started_at": now() - started_ago,
            "cost": CostInfo(hourly_usd=0.5),
        }
    )
    store.create(m)
    store.write_runtime("j1", runner_pid=1, cluster="lab-j1")
    out = store.output_dir("j1")
    out.mkdir(parents=True, exist_ok=True)
    (out / SUCCESS_SENTINEL).write_text("1")
    return store


@pytest.fixture
def release(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Everything a hung call waits on; set at teardown so no test thread outlives its test."""
    ev = threading.Event()
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: 0.5)
    monkeypatch.setattr(runner_mod, "r2_enabled", lambda: False)
    monkeypatch.setattr(runner_mod, "POLL_S", 0.01)
    yield ev
    ev.set()


def _run_bounded(store: JobStore, seconds: float = 10.0) -> bool:
    """Run the supervisor on a thread; True iff it finished within ``seconds``."""
    t = threading.Thread(
        target=runner_mod.run_job, args=(store.job_dir("j1"),), kwargs={"adopt": True}, daemon=True
    )
    t.start()
    t.join(seconds)
    return not t.is_alive()


def _sky(*, tail_logs: Any, queue: Any, status: Any = None) -> types.ModuleType:
    fake = types.ModuleType("sky")
    fake.get = lambda x: x  # type: ignore[attr-defined]
    fake.tail_logs = tail_logs  # type: ignore[attr-defined]
    fake.queue = queue  # type: ignore[attr-defined]
    fake.status = status or (lambda *a, **k: [{"status": "UP"}])  # type: ignore[attr-defined]
    return fake


def test_a_half_open_stream_cannot_outlive_the_wall_clock_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release: threading.Event
) -> None:
    """The reviewer's reproduction: the budget was spent long ago, tail_logs never returns."""
    store = _store(tmp_path, timeout="1s", started_ago=timedelta(hours=2))
    running = [{"job_id": 1, "status": _Status("RUNNING")}]
    monkeypatch.setitem(
        sys.modules,
        "sky",
        _sky(tail_logs=lambda *a, **k: release.wait(), queue=lambda *a, **k: running),
    )

    assert _run_bounded(store), "the supervisor is still blocked inside tail_logs"
    assert store.read_manifest("j1").status is JobState.timed_out


def test_a_spent_budget_does_not_sleep_a_poll_interval_before_giving_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer's own reproduction runs at the real POLL_S: one look, then out — not one
    look, a 10s nap, and only then a deadline check."""
    import time

    monkeypatch.setattr(runner_mod, "POLL_S", 30.0)
    running = [{"job_id": 1, "status": _Status("RUNNING")}]
    sky = _sky(tail_logs=lambda *a, **k: None, queue=lambda *a, **k: running)

    t0 = time.monotonic()
    state, reached, lost = runner_mod._wait_terminal(sky, "lab-j1", None, 0.0)

    assert time.monotonic() - t0 < 5.0
    assert state is JobState.running and reached is False and lost is None


def test_a_wedged_queue_poll_cannot_outlive_the_wall_clock_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release: threading.Event
) -> None:
    store = _store(tmp_path, timeout="1s", started_ago=timedelta(hours=2))
    monkeypatch.setattr(runner_mod, "SKY_CALL_TIMEOUT_S", 0.2)

    def _hang(*a: Any, **k: Any) -> Any:
        release.wait()
        return []

    monkeypatch.setitem(
        sys.modules, "sky", _sky(tail_logs=lambda *a, **k: None, queue=_hang, status=_hang)
    )

    assert _run_bounded(store), "the supervisor is still blocked inside sky.queue"
    assert store.read_manifest("j1").status in (JobState.failed, JobState.timed_out)


def test_an_uncapped_job_notices_its_run_ended_under_a_stream_that_did_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release: threading.Event
) -> None:
    """No budget to run out, so the stream is probed: the queue is the source of truth."""
    store = _store(tmp_path, timeout=None, started_ago=timedelta(minutes=5))
    monkeypatch.setattr(runner_mod, "STREAM_PROBE_S", 0.05)
    done = [{"job_id": 1, "status": _Status("SUCCEEDED")}]
    monkeypatch.setitem(
        sys.modules,
        "sky",
        _sky(tail_logs=lambda *a, **k: release.wait(), queue=lambda *a, **k: done),
    )

    assert _run_bounded(store), "an uncapped job waited forever on a dead log stream"
    assert store.read_manifest("j1").status is JobState.succeeded


def test_a_healthy_stream_is_not_cut_short_by_the_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release: threading.Event
) -> None:
    """The probe only ends the wait on a terminal answer; a RUNNING job keeps streaming."""
    store = _store(tmp_path, timeout=None, started_ago=timedelta(minutes=5))
    monkeypatch.setattr(runner_mod, "STREAM_PROBE_S", 0.02)
    statuses = iter(["RUNNING"] * 5)
    streamed = threading.Event()

    def _tail(*a: Any, **k: Any) -> None:
        streamed.wait(5.0)  # streams until the job's last line

    def _queue(*a: Any, **k: Any) -> list[dict[str, Any]]:
        name = next(statuses, None)
        if name is None:
            streamed.set()
            name = "SUCCEEDED"
        return [{"job_id": 1, "status": _Status(name)}]

    monkeypatch.setitem(sys.modules, "sky", _sky(tail_logs=_tail, queue=_queue))

    assert _run_bounded(store)
    assert store.read_manifest("j1").status is JobState.succeeded
