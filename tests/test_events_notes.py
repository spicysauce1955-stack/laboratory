"""The notes that make a failure explicable. Each asserts the note fires at the site that
already prints the same thing to stderr — the print is the live UX, the note is the record.

Every note here is driven through the real production code path (the module's own function,
not ``events.note`` called directly), per the design: a note is buffered on the current call and
only survives into the trace if that call ends non-ok, so each test opens a ``record()``, drives
the site, then raises to force the flush.
"""

from __future__ import annotations

import subprocess
import time as _time
import types
from datetime import timedelta
from pathlib import Path

import pytest
from helpers import PYTHON, wait_terminal
from test_scheduler_tick import T0, FakeClock, make_sched, put_reg

import lab.backends.skypilot as SKY
import lab.core as core_mod
import lab.doctor as D
from lab import events, placement
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.events import store
from lab.manifest import repo_root
from lab.models import JobSpec, JobState, ResourceRequest
from lab.scheduler.models import Triggers


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


def _trace_kinds() -> list[str]:
    close = [r for r in store.iter_records(store.day_files()) if r["phase"] == "close"][-1]
    return [n["k"] for n in close.get("trace", [])]


# --------------------------------------------------------------------------------------------
# placement.py
# --------------------------------------------------------------------------------------------


def test_placement_note_also_notes() -> None:
    """The real function is ``placement._note`` (the brief's ``_warn`` does not exist in this
    codebase); it is the shared diagnostic used by the memo-write, catalog-resolve, and
    region-lookup failure paths, so hooking it once covers all three."""
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            placement._note("zone europe-west1-b exhausted")
            raise RuntimeError("boom")
    assert "placement.warn" in _trace_kinds()


def test_a_note_never_escapes_into_a_successful_record() -> None:
    with events.record("cli", "submit", {}):
        placement._note("harmless")
    close = [r for r in store.iter_records(store.day_files()) if r["phase"] == "close"][-1]
    assert "trace" not in close


def _fake_placement_catalog() -> types.SimpleNamespace:
    """A minimal fake shaped like ``sky.catalog``: one region, two zones, a flat price."""
    zones = ["us-central1-a", "us-central1-b"]
    region = types.SimpleNamespace(
        name="us-central1", zones=[types.SimpleNamespace(name=z) for z in zones]
    )
    return types.SimpleNamespace(
        get_default_instance_type=lambda **kw: "n4-standard-4",
        get_instance_type_for_accelerator=lambda *a, **kw: (["n1-highmem-4"], []),
        get_region_zones_for_instance_type=lambda *a, **kw: [region],
        get_region_zones_for_accelerators=lambda *a, **kw: [region],
        get_hourly_cost=lambda instance_type, use_spot, region, zone, clouds=None: 0.18,
        get_accelerator_hourly_cost=lambda *a, **kw: 0.0,
        validate_region_zone=lambda r, z, clouds=None: (r, z),
    )


def test_zone_skipped_and_priced_also_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drives ``placement.candidates()``: one of the two zones is pre-excluded by a fake memo,
    so the survivor is priced and the excluded one is reported skipped."""
    monkeypatch.setattr(placement, "_catalog", _fake_placement_catalog)
    memo = types.SimpleNamespace(exhausted_zones=lambda cloud, it, **kw: {"us-central1-a"})
    res = ResourceRequest(cloud="gcp", cpus=4)
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            out = placement.candidates(res, instance_type="n4-standard-4", memo=memo)
            assert out and out[0].hourly_usd == 0.18
            raise RuntimeError("boom")
    kinds = _trace_kinds()
    assert "placement.zone_skipped" in kinds
    assert "placement.priced" in kinds


def test_zone_exhausted_also_notes(tmp_path: Path) -> None:
    memo = placement.CapacityMemo(tmp_path / "memo.json")
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            memo.record("gcp", "n4-standard-4", ["us-central1-a"])
            raise RuntimeError("boom")
    assert "placement.zone_exhausted" in _trace_kinds()


def test_disk_override_also_notes() -> None:
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            applied = placement.effective_disk_gb(ResourceRequest(cloud="gcp"))
            assert applied == placement.CPU_DEFAULT_DISK_GB
            raise RuntimeError("boom")
    assert "placement.disk_override" in _trace_kinds()


# --------------------------------------------------------------------------------------------
# doctor.py
# --------------------------------------------------------------------------------------------


def test_doctor_check_also_notes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(D._REGISTRY, "gcp", (("adc", lambda res, ctx: D._ok("adc", "fine")),))
    with pytest.raises(RuntimeError):
        with events.record("cli", "doctor", {}):
            results = D.run_checks("gcp", ResourceRequest(cloud="gcp"), home=tmp_path)
            assert results[0].status == "ok"
            raise RuntimeError("boom")
    assert "doctor.check" in _trace_kinds()


# --------------------------------------------------------------------------------------------
# backends/skypilot.py
# --------------------------------------------------------------------------------------------


class _FakeSkyDownFails:
    """Every ``sky.down`` fails; drives robust_teardown's retry + fallback paths."""

    def down(self, cluster: str) -> str:
        raise RuntimeError("sky.down boom")

    def get(self, x: object) -> object:
        return x


def test_teardown_attempt_retry_and_fallback_also_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SKY, "_vast_destroy_matching", lambda c: ([123], []))
    with pytest.raises(RuntimeError):
        with events.record("cli", "cancel", {}):
            out = SKY.robust_teardown(_FakeSkyDownFails(), "lab-x", backoffs=(), cloud="vast")
            assert out["status"] == "succeeded"
            raise RuntimeError("boom")
    kinds = _trace_kinds()
    assert "teardown.attempt" in kinds
    assert "teardown.retry" in kinds
    assert "teardown.fallback" in kinds


def test_teardown_fallback_failure_also_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SKY, "_vast_destroy_matching", lambda c: ([], ["999: still running"])
    )
    with pytest.raises(RuntimeError):
        with events.record("cli", "cancel", {}):
            out = SKY.robust_teardown(_FakeSkyDownFails(), "lab-x", backoffs=(), cloud="vast")
            assert out["status"] == "failed"
            raise RuntimeError("boom")
    assert "teardown.fallback" in _trace_kinds()


class _BoomVastClient:
    def show_user(self) -> dict:
        raise RuntimeError("vast api boom")


def test_vast_balance_failed_also_notes() -> None:
    with pytest.raises(RuntimeError):
        with events.record("cli", "doctor", {}):
            assert SKY.vast_balance(client=_BoomVastClient()) is None
            raise RuntimeError("boom")
    assert "vast.balance_failed" in _trace_kinds()


class _FakeLaunchSky:
    """Stand-in for `sky` covering provisioning: a ``sleep_s`` that outlasts the watchdog."""

    def __init__(self, *, sleep_s: float = 0.0) -> None:
        self.sleep_s = sleep_s
        self.api_cancel_calls: list[object] = []

    def stream_and_get(self, request_id: object) -> tuple:
        if self.sleep_s:
            _time.sleep(self.sleep_s)
        return (1, "handle")

    def api_cancel(self, request_id: object) -> None:
        self.api_cancel_calls.append(request_id)


def test_provision_timeout_also_notes() -> None:
    sky = _FakeLaunchSky(sleep_s=1.0)  # never finishes before the 0.05s watchdog
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            with pytest.raises(SKY.ProvisionTimeout):
                SKY.provision_with_watchdog(sky, "req-1", timeout_s=0.05)
            raise RuntimeError("boom")
    assert "provision.timeout" in _trace_kinds()


# --------------------------------------------------------------------------------------------
# core.py
# --------------------------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "uv.lock").write_text("lock\n")
    (repo / "run.py").write_text("print(1)\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def test_dirty_snapshot_also_notes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "run.py").write_text("print(2)\n")  # dirty now
    lab = Lab(backend=LocalBackend(home=tmp_path / "runs", repo=repo), repo=repo, home=tmp_path / "runs")
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            lab.submit(JobSpec(command=f"{PYTHON} -c 'pass'"), allow_dirty=True)
            raise RuntimeError("boom")
    assert "core.dirty_snapshot" in _trace_kinds()


class _BoomR2:
    """Enabled + reachable, but every upload fails — drives the diff-mirror except branch."""

    @classmethod
    def from_env(cls) -> "_BoomR2":
        return cls()

    def upload_file(self, local: Path, key: str) -> None:
        raise RuntimeError("r2 boom")

    def uri(self, key: str) -> str:
        return f"r2://x/{key}"


def test_storage_upload_failed_also_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core_mod, "r2_enabled", lambda: True)
    monkeypatch.setattr(core_mod, "R2Store", _BoomR2)
    repo = _make_repo(tmp_path)
    (repo / "run.py").write_text("print(2)\n")  # dirty -> triggers the diff-capture + mirror
    lab = Lab(backend=LocalBackend(home=tmp_path / "runs", repo=repo), repo=repo, home=tmp_path / "runs")
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            lab.submit(JobSpec(command=f"{PYTHON} -c 'pass'"), allow_dirty=True)
            raise RuntimeError("boom")
    assert "storage.upload_failed" in _trace_kinds()


def test_cache_hit_also_notes(tmp_path: Path) -> None:
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    cmd = f"{PYTHON} experiments/example_capacity.py"
    jid = lab.submit(JobSpec(code_ref="HEAD", command=cmd, seed=5, config={"K": 1}))
    assert wait_terminal(backend, jid) == JobState.succeeded
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            hit = lab.find_cached(
                JobSpec(command=cmd, seed=5, config={"K": 1}), require_clean=False
            )
            assert hit == jid
            raise RuntimeError("boom")
    assert "core.cache_hit" in _trace_kinds()


class _FakeRemoteBackend:
    """``name`` != "local" is all ``sweep()`` needs to engage the submit stagger."""

    name = "vast"

    def submit(self, manifest) -> str:  # noqa: ANN001
        return manifest.job_id


def test_submit_stagger_also_notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_SUBMIT_STAGGER_S", "0.01")
    repo = _make_repo(tmp_path)
    lab = Lab(
        backend=_FakeRemoteBackend(), repo=repo, home=tmp_path / "runs"  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError):
        with events.record("cli", "sweep", {}):
            lab.sweep(
                f"{PYTHON} -c 'pass'", {"a": [1, 2]}, resources=ResourceRequest(), max_jobs=8
            )
            raise RuntimeError("boom")
    assert "core.submit_stagger" in _trace_kinds()


# --------------------------------------------------------------------------------------------
# scheduler/tick.py
# --------------------------------------------------------------------------------------------


def test_scheduler_trigger_fires_also_notes(tmp_path: Path) -> None:
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a")
    with pytest.raises(RuntimeError):
        with events.record("scheduler", "tick", {}):
            sched.tick()
            raise RuntimeError("boom")
    assert "scheduler.trigger" in _trace_kinds()


def test_scheduler_trigger_blocked_also_notes(tmp_path: Path) -> None:
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    put_reg(q, tmp_path, "reg-a", triggers=Triggers(not_before=T0 + timedelta(hours=1)))
    with pytest.raises(RuntimeError):
        with events.record("scheduler", "tick", {}):
            sched.tick()
            raise RuntimeError("boom")
    assert "scheduler.trigger" in _trace_kinds()
