"""A job stopped by the abort path still billed, and spend must say how much.

Smoke run, 2026-11-19: job F was ``lab cancel``led about four minutes into a DigitalOcean run. Its
droplet and 50 GB volume existed and billed for those minutes, but the box had not yet reported UP
— the moment the normal path prices it (``resolve_cost`` reads the launched handle) — so no
``CostInfo`` was ever written. ``lab list``'s spend named it under ``unknown_cost_jobs`` and left a
real rental out of the total.

The abort path now records the cost the way the normal path does, ``hourly x supervisor wall``
(``started_at`` to ``ended_at``), but only when the teardown is evidence that a machine existed:
pricing a launch that never created anything would put money in the total that nobody spent. With
no launched handle to price, the rate comes from the catalog for the requested spec, at the top of
its band (exact on DO, whose prices do not vary by region), and ``hourly_basis`` says so. A rate
that was already recorded at UP is finalized rather than replaced.
"""

from __future__ import annotations

import signal
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fake_do import FakeDo, FakeSky
from helpers import make_manifest

import lab.placement as placement
import lab.sky_runner as runner_mod
from lab._util import actual_cost, duration_seconds
from lab.backends import skypilot as sp
from lab.models import BackendInfo, CostInfo, JobState, ResourceRequest
from lab.pricing import realized_spend
from lab.store import JobStore

JOB = "20261119-111134-b048d2"
CLUSTER = f"lab-smoke-{JOB}"
VOLUME = f"{CLUSTER}-3dd12990-ef65-head"
T0 = datetime(2026, 11, 19, 11, 11, 34, tzinfo=UTC)
DO_4CPU = placement.Estimate(
    instance_type="s-4vcpu-8gb",
    low_usd=0.0714,
    high_usd=0.0714,
    storage_usd=0.0069,
    regions=8,
    excluded_zones=0,
    basis="do catalog s-4vcpu-8gb on-demand 8 regions + 50GiB disk ($0.0069/hr)",
)


class _Sky(FakeSky):
    """``sky`` for a supervisor run: launches, cancels, and tears down against the fake DO."""

    def launch(self, task: Any, cluster_name: str, **kw: Any) -> str:
        return "req-1"

    def api_cancel(self, request_id: Any) -> None:
        pass


def _clock(monkeypatch: pytest.MonkeyPatch, wall_s: float) -> None:
    """The supervisor starts at T0; every later reading is `wall_s` on."""
    readings = iter([T0])
    monkeypatch.setattr(runner_mod, "now", lambda: next(readings, T0 + timedelta(seconds=wall_s)))


def _cancel_mid_provision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, do: FakeDo
) -> JobStore:
    store = JobStore(tmp_path / "runs")
    store.create(
        make_manifest(JOB, "python x.py", resources=ResourceRequest(cloud="do", cpus=4)).model_copy(
            update={"backend": BackendInfo(provisioner="skypilot")}
        )
    )
    store.write_runtime(JOB, cluster=CLUSTER)
    monkeypatch.setattr(sp, "_get_do_client", lambda: do)
    monkeypatch.setattr(sp.time, "sleep", do.sleep)
    monkeypatch.setitem(sys.modules, "sky", _Sky(do))
    monkeypatch.setattr(runner_mod, "build_task", lambda *a, **k: "task")

    def _cancelled(*a: Any, **k: Any) -> Any:
        store.write_runtime(JOB, cancelling=True)  # `lab cancel`, then its SIGTERM
        raise runner_mod.SupervisorTerminated(signal.SIGTERM)

    monkeypatch.setattr(runner_mod, "provision_with_watchdog", _cancelled)
    with pytest.raises(SystemExit):
        runner_mod.run_job(store.job_dir(JOB))
    return store


def _job_f_account() -> FakeDo:
    return FakeDo(
        droplets=[{"id": 31, "name": VOLUME}],
        volumes=[{"id": "vol-f", "name": VOLUME, "droplet_ids": [31]}],
        detach_s=0.0,
    )


class TestJobF:
    def test_a_cancel_mid_provision_records_hourly_times_supervisor_wall(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(placement, "estimate", lambda res, **k: DO_4CPU)
        _clock(monkeypatch, wall_s=240)

        store = _cancel_mid_provision(tmp_path, monkeypatch, _job_f_account())

        m = store.read_manifest(JOB)
        assert m.status is JobState.cancelled
        assert m.cost is not None, "a droplet billed for four minutes and no cost was recorded"
        wall = duration_seconds(m.started_at, m.ended_at)
        assert wall == pytest.approx(240)
        assert m.cost.hourly_usd == pytest.approx(DO_4CPU.worst_hourly_usd)
        assert m.cost.duration_seconds == pytest.approx(wall)
        assert m.cost.actual_usd == actual_cost(m.cost.hourly_usd, wall)
        assert "catalog" in (m.cost.hourly_basis or "") and "UP" in (m.cost.hourly_basis or "")

        spend = realized_spend([m], at=T0 + timedelta(hours=1))
        assert spend["unknown_cost_jobs"] == [], "the rental is still missing from spend"
        assert spend["realized_usd"] == pytest.approx(m.cost.actual_usd)

    def test_nothing_found_by_the_teardown_records_no_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No droplet, no volume: nothing was provisioned, so no money goes in the total."""
        monkeypatch.setattr(placement, "estimate", lambda res, **k: DO_4CPU)
        _clock(monkeypatch, wall_s=240)

        store = _cancel_mid_provision(tmp_path, monkeypatch, FakeDo())

        assert store.read_manifest(JOB).cost is None

    def test_an_unpriceable_spec_records_nothing_and_never_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_catalog(res: Any, **k: Any) -> Any:
            raise RuntimeError("catalog unavailable")

        monkeypatch.setattr(placement, "estimate", _no_catalog)
        _clock(monkeypatch, wall_s=240)

        store = _cancel_mid_provision(tmp_path, monkeypatch, _job_f_account())

        m = store.read_manifest(JOB)
        assert m.cost is None and m.status is JobState.cancelled
        assert m.teardown_status == "succeeded", "pricing must never get in the teardown's way"


class TestAfterUp:
    def test_a_rate_recorded_at_up_is_finalized_not_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = JobStore(tmp_path / "runs")
        started = datetime.now(UTC) - timedelta(seconds=600)
        store.create(
            make_manifest("up1", "python x.py", resources=ResourceRequest(cloud="do")).model_copy(
                update={
                    "status": JobState.running,
                    "started_at": started,
                    "backend": BackendInfo(provisioner="skypilot"),
                    "cost": CostInfo(hourly_usd=0.0783, hourly_basis="do s-4vcpu-8gb (launched)"),
                }
            )
        )
        store.write_runtime("up1", cancelling=True)
        monkeypatch.setitem(sys.modules, "sky", types.ModuleType("sky"))
        monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)
        monkeypatch.setattr(
            placement, "estimate", lambda *a, **k: pytest.fail("a recorded rate was re-priced")
        )

        runner_mod._teardown_on_abort(
            store, "up1", "lab-up1", "do",
            why="supervisor terminated by SIGTERM", launched=True, torn_down=False,
        )

        m = store.read_manifest("up1")
        assert m.cost is not None and m.cost.hourly_basis == "do s-4vcpu-8gb (launched)"
        wall = duration_seconds(m.started_at, m.ended_at)
        assert m.cost.duration_seconds == pytest.approx(wall)
        assert m.cost.actual_usd == actual_cost(0.0783, wall)
