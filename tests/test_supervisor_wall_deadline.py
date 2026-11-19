"""The supervisor's local wall-clock backstop must never kill a healthy run (review finding).

Two defects, one mechanism — the backstop's budget ran out while the remote job was still
RUNNING, and the classifier mapped ``running`` to ``failed``:

* **Uncapped jobs got an invented budget.** ``remaining_wall_budget`` gave a job submitted with no
  ``--timeout`` 3600+300s. Once that passed — typically after an API-server restart made
  ``tail_logs`` return early — a single poll saw RUNNING, the job was recorded ``failed`` and its
  healthy machine torn down: every uncapped job older than 65 minutes, on any stream drop. A job
  that asked for no wall-clock bound must be watched until it actually ends.
* **Capped jobs were charged their boot phase.** The budget was anchored at ``started_at`` —
  before provisioning, workdir sync and ``uv sync`` — with only 300s of slack, while the box's
  own cap wraps only the entrypoint and its watchdog allows a whole boot allowance before it.
  A provision+setup over 300s ended a healthy capped run early, as ``failed``. The local deadline
  is now anchored at run start (provisioning complete) plus the same boot allowance the on-box
  watchdog uses, and a job still RUNNING at that deadline is ``timed_out`` — the backstop enforced
  the cap — never ``failed``.
"""

from __future__ import annotations

import math
import sys
import types
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab._util import now
from lab.backends.skypilot import SUCCESS_SENTINEL, boot_allowance_s
from lab.models import BackendInfo, CostInfo, JobState
from lab.store import JobStore


class _Status:
    def __init__(self, name: str) -> None:
        self.name = name


def _sky(statuses: list[str], *, tail_logs: Any = None) -> types.ModuleType:
    """A sky whose queue answers each of ``statuses`` once, then repeats the last."""
    seq = list(statuses)
    fake = types.ModuleType("sky")
    fake.get = lambda x: x  # type: ignore[attr-defined]

    def _queue(cluster: str, skip_finished: bool = False) -> list[dict[str, Any]]:
        name = seq.pop(0) if len(seq) > 1 else seq[0]
        return [{"job_id": 1, "status": _Status(name)}]

    def _drop(*a: Any, **k: Any) -> None:
        raise ConnectionError("API server connection reset")  # the stream drops at once

    fake.queue = _queue  # type: ignore[attr-defined]
    fake.tail_logs = tail_logs or _drop  # type: ignore[attr-defined]
    fake.status = lambda *a, **k: [{"name": "lab-j1", "status": "UP"}]  # type: ignore[attr-defined]
    return fake


def _job(
    tmp_path: Path, *, timeout: str | None, started_ago: timedelta, runtime: dict[str, Any]
) -> JobStore:
    store = JobStore(tmp_path / "runs")
    m = make_manifest("j1", "python train.py", timeout=timeout).model_copy(
        update={
            "status": JobState.running,
            "started_at": now() - started_ago,
            "cost": CostInfo(hourly_usd=0.5, estimated_usd=None),
            "backend": BackendInfo(provisioner="skypilot"),
        }
    )
    store.create(m)
    store.write_runtime("j1", runner_pid=1, cluster="lab-j1", **runtime)
    out = store.output_dir("j1")
    out.mkdir(parents=True, exist_ok=True)
    (out / SUCCESS_SENTINEL).write_text("1")  # the run script's clean-exit marker
    return store


@pytest.fixture
def torn(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def _teardown(sky_mod: Any, cluster: str, *a: Any, **k: Any) -> bool:
        calls.append(cluster)
        return True

    monkeypatch.setattr(runner_mod, "tear_down_and_record", _teardown)
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: 0.5)
    monkeypatch.setattr(runner_mod, "r2_enabled", lambda: False)
    monkeypatch.setattr(runner_mod, "POLL_S", 0.01)
    return calls


def test_an_uncapped_job_is_watched_until_it_ends_not_failed_at_65_minutes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torn: list[str]
) -> None:
    """The reviewer's reproduction: no --timeout, 2h in, the log stream drops, the job is RUNNING."""
    store = _job(tmp_path, timeout=None, started_ago=timedelta(hours=2), runtime={})
    monkeypatch.setitem(sys.modules, "sky", _sky(["RUNNING"] * 5 + ["SUCCEEDED"]))

    rc = runner_mod.run_job(store.job_dir("j1"), adopt=True)

    final = store.read_manifest("j1")
    assert final.status is JobState.succeeded, final.end_reason
    assert rc == 0 and torn == ["lab-j1"]  # torn down once, after it really finished


def test_an_uncapped_job_has_no_local_budget() -> None:
    assert runner_mod.remaining_wall_budget(None, now() - timedelta(days=3)) == math.inf


@pytest.mark.parametrize("anchored", [True, False], ids=["run-anchor", "legacy-adopt"])
def test_a_slow_boot_does_not_eat_a_capped_jobs_run_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torn: list[str], anchored: bool
) -> None:
    """20m cap, 30m since the launch request, 5m since the box came up. The old budget
    (started_at + 20m + 300s) was spent before the run had used a third of its cap."""
    runtime = {"run_anchor_at": (now() - timedelta(minutes=5)).isoformat()} if anchored else {}
    store = _job(tmp_path, timeout="20m", started_ago=timedelta(minutes=30), runtime=runtime)
    monkeypatch.setitem(sys.modules, "sky", _sky(["RUNNING"] * 3 + ["SUCCEEDED"]))

    runner_mod.run_job(store.job_dir("j1"), adopt=True)

    assert store.read_manifest("j1").status is JobState.succeeded


def test_still_running_at_the_local_deadline_is_timed_out_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torn: list[str]
) -> None:
    """The backstop is allowed to end a run — the cap says so — but what it ends is a timeout."""
    store = _job(tmp_path, timeout="1s", started_ago=timedelta(hours=3), runtime={})
    monkeypatch.setitem(sys.modules, "sky", _sky(["RUNNING"], tail_logs=lambda *a, **k: None))

    runner_mod.run_job(store.job_dir("j1"), adopt=True)

    final = store.read_manifest("j1")
    assert final.status is JobState.timed_out
    assert (final.end_reason or "").startswith("timed out after 1s wall-clock cap")
    assert torn == ["lab-j1"]


def test_the_capped_budget_carries_the_boot_allowance() -> None:
    m = make_manifest("j1", "python x.py", timeout="1h")
    boot = boot_allowance_s(m)

    remaining = runner_mod.remaining_wall_budget("1h", now(), boot_s=boot)

    assert 3600 + runner_mod.WALL_CLOCK_GRACE_S + boot - 5 <= remaining
    assert remaining <= 3600 + runner_mod.WALL_CLOCK_GRACE_S + boot


def test_a_fresh_launch_records_its_run_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torn: list[str]
) -> None:
    """The anchor has to survive a supervisor restart, or an adopted job is budgeted from the
    launch request again."""
    store = _job(tmp_path, timeout="1h", started_ago=timedelta(0), runtime={})
    store.update_manifest("j1", status=JobState.queued)
    fake = _sky(["SUCCEEDED"], tail_logs=lambda *a, **k: None)
    fake.launch = lambda *a, **k: "req-1"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setattr(runner_mod, "build_task", lambda *a, **k: "task")
    monkeypatch.setattr(
        runner_mod,
        "provision_with_watchdog",
        lambda *a, **k: (1, types.SimpleNamespace(launched_resources=None)),
    )
    monkeypatch.setattr(runner_mod, "resolve_cost", lambda *a, **k: CostInfo())
    before = now()

    runner_mod.run_job(store.job_dir("j1"))

    anchor = store.read_runtime("j1").get("run_anchor_at")
    assert anchor is not None and anchor >= before.isoformat()
