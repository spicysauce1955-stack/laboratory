import time
from pathlib import Path

import pytest
from helpers import make_manifest

import lab.backends.skypilot as skypilot_mod
from lab.backends.skypilot import (
    TIMEOUT_SENTINEL,
    ProvisionTimeout,
    build_run_script,
    build_setup_script,
    build_task,
    cluster_name_for,
    map_job_status,
    promote_timeout,
    provision_with_watchdog,
    robust_teardown,
    tear_down_and_record,
    vast_hourly_for_cluster,
)
from lab.models import JobState
from lab.store import JobStore


def test_map_job_status():
    assert map_job_status("SUCCEEDED") == JobState.succeeded
    assert map_job_status("RUNNING") == JobState.running
    assert map_job_status("PENDING") == JobState.queued
    assert map_job_status("SETTING_UP") == JobState.queued
    assert map_job_status("FAILED_SETUP") == JobState.failed
    assert map_job_status("CANCELLED") == JobState.cancelled
    assert map_job_status("WAT") == JobState.failed  # unknown -> fail-loud


def test_cluster_name_for():
    assert cluster_name_for("20260527-114219-c8e4e5") == "lab-20260527-114219-c8e4e5"
    assert cluster_name_for("Weird_Name!!") == "lab-weird-name"
    long = cluster_name_for("x" * 100)
    assert long.startswith("lab-") and len(long) <= 60


def test_build_scripts_and_timeout():
    setup = build_setup_script()
    assert "uv sync --frozen" in setup and "astral.sh/uv/install" in setup
    assert "--no-default-groups" in setup

    m = make_manifest("j1", "python experiments/example_capacity.py", timeout="30m")
    run = build_run_script(m)
    # Entrypoint runs in its own session so the timer can group-kill the whole tree (§6).
    assert "setsid --wait bash -c" in run
    assert "python experiments/example_capacity.py" in run
    assert "source .venv/bin/activate" in run
    # No timeout scaffolding when there is no cap.
    bare = build_run_script(make_manifest("j2", "python x.py"))
    assert "setsid --wait" not in bare and "poweroff" not in bare
    assert "python x.py" in bare


def test_run_script_group_kill_and_sentinel():
    run = build_run_script(make_manifest("j", "python x.py", timeout="30m"))
    assert "sleep 1800" in run                 # 30m wall
    assert "kill -TERM -$$" in run             # TERM the whole process group
    assert "kill -KILL -$$" in run             # then KILL after the grace
    assert f"sleep {skypilot_mod.TIMEOUT_KILL_GRACE_S}" in run
    assert TIMEOUT_SENTINEL in run             # killer drops the sentinel for promote_timeout


def test_run_script_self_destruct_watchdog():
    run = build_run_script(make_manifest("j", "python x.py", timeout="30m"))
    margin = skypilot_mod.SELF_DESTRUCT_MARGIN_S
    assert f"sleep {1800 + margin}" in run     # poweroff at wall + margin
    assert "poweroff" in run
    assert "nohup setsid bash -c" in run       # detached, survives the supervisor


def test_promote_timeout(tmp_path):
    assert promote_timeout(JobState.failed, tmp_path) == JobState.failed
    (tmp_path / TIMEOUT_SENTINEL).touch()
    assert promote_timeout(JobState.failed, tmp_path) == JobState.timed_out
    # only a failed run is promoted
    assert promote_timeout(JobState.succeeded, tmp_path) == JobState.succeeded


def test_build_task_fields(tmp_path: Path):
    pytest.importorskip("sky")  # build_task needs the optional `skypilot` extra
    m = make_manifest("j1", "python run.py", seed=5)
    task = build_task(m, workdir=tmp_path)
    assert task.envs["LAB_RUN_ID"] == "j1"
    assert task.envs["LAB_SEED"] == "5"
    assert task.envs["LAB_RUN_DIR"]
    assert "run.py" in task.run
    assert "uv sync" in task.setup

    # accelerators flow through to the Resources (required for Vast)
    gpu_task = build_task(make_manifest("j2", "python r.py", accelerators="T4:1"), workdir=tmp_path)
    res = next(iter(gpu_task.resources))
    assert "T4" in str(res.accelerators)


# ----------------------------------------------------------------------------
# robust_teardown — FR-C2 leak prevention (sky.down retries + vast-sdk fallback)
# ----------------------------------------------------------------------------


class _FakeSky:
    """Minimal stand-in for the `sky` module. ``down(cluster)`` records the call and returns a
    sentinel "request id"; ``get(req)`` either succeeds or raises depending on ``fail_times``.
    """

    def __init__(self, fail_times: int = 0, exc: type[Exception] = ConnectionError) -> None:
        self.fail_times = fail_times
        self.exc = exc
        self.down_calls: list[str] = []
        self.get_calls: int = 0

    def down(self, cluster: str) -> str:
        self.down_calls.append(cluster)
        return f"req-{len(self.down_calls)}"

    def get(self, req: str) -> None:
        self.get_calls += 1
        if self.get_calls <= self.fail_times:
            raise self.exc(f"transient #{self.get_calls}")


class _FakeVast:
    def __init__(self, instances: list[dict] | None = None) -> None:
        self.instances = instances or []
        self.destroyed: list[int] = []

    def show_instances(self) -> list[dict]:
        return list(self.instances)

    def destroy_instance(self, id: int) -> dict:  # noqa: A002 - mirror real vastai-sdk arg name
        self.destroyed.append(int(id))
        return {"id": id, "destroyed": True}


@pytest.fixture
def _no_sleep(monkeypatch: pytest.MonkeyPatch):
    """Make retries instantaneous in tests."""
    monkeypatch.setattr(skypilot_mod.time, "sleep", lambda _s: None)


def test_robust_teardown_first_attempt_succeeds(_no_sleep):
    sky = _FakeSky(fail_times=0)
    out = robust_teardown(sky, "lab-abc")
    assert out["status"] == "succeeded"
    assert out["attempts"] == 1
    assert out["vast_fallback_used"] is False
    assert sky.down_calls == ["lab-abc"]


def test_robust_teardown_retries_then_succeeds(_no_sleep):
    sky = _FakeSky(fail_times=3)  # 3 transient errors, then succeeds on the 4th attempt
    out = robust_teardown(sky, "lab-abc")
    assert out["status"] == "succeeded"
    assert out["attempts"] == 4
    assert out["vast_fallback_used"] is False
    assert out["error"] is None  # no error retained on eventual sky.down success


def test_robust_teardown_falls_back_to_vast_when_sky_exhausted(
    _no_sleep, monkeypatch: pytest.MonkeyPatch
):
    sky = _FakeSky(fail_times=99)  # always fails
    vast = _FakeVast(
        instances=[
            {"id": 7, "label": "sky-lab-abc-xyz"},  # matches cluster -> destroyed
            {"id": 8, "label": "someone-elses"},  # not ours -> left alone
        ]
    )
    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: vast)
    out = robust_teardown(sky, "lab-abc")
    assert out["status"] == "succeeded"  # fallback succeeded
    assert out["vast_fallback_used"] is True
    assert out["vast_destroyed"] == [7]
    assert out["attempts"] == 6  # 1 + 5 retries (cf. TEARDOWN_BACKOFFS)
    assert vast.destroyed == [7]  # only the matching instance was destroyed


def test_robust_teardown_failed_when_both_sky_and_vast_error(
    _no_sleep, monkeypatch: pytest.MonkeyPatch
):
    sky = _FakeSky(fail_times=99)

    class _BrokenVast:
        def show_instances(self) -> list[dict]:
            raise RuntimeError("vast api down")

    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: _BrokenVast())
    out = robust_teardown(sky, "lab-abc")
    assert out["status"] == "failed"
    assert out["vast_fallback_used"] is True
    assert "sky.down" in out["error"] and "vast-direct" in out["error"]


def test_tear_down_and_record_records_failure_loudly(
    _no_sleep, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A persistent teardown leak must show up in the manifest as teardown_status='failed' and
    an actionable end_reason annotation so `lab status`/`lab wait` can flag it."""
    store = JobStore(tmp_path)
    m = make_manifest("job-1", "python x.py")
    store.create(m)
    store.update_manifest("job-1", end_reason="some prior reason")  # confirm we append, not clobber
    sky = _FakeSky(fail_times=99)

    class _BrokenVast:
        def show_instances(self) -> list[dict]:
            raise RuntimeError("vast api down")

    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: _BrokenVast())
    ok = tear_down_and_record(sky, "lab-job-1", store, "job-1")
    assert ok is False
    persisted = store.read_manifest("job-1")
    assert persisted.teardown_status == "failed"
    assert persisted.end_reason is not None
    assert "TEARDOWN FAILED" in persisted.end_reason
    assert "some prior reason" in persisted.end_reason  # annotation appended, not overwritten


def test_tear_down_and_record_marks_success_when_sky_down_succeeds(_no_sleep, tmp_path: Path):
    store = JobStore(tmp_path)
    store.create(make_manifest("job-2", "python x.py"))
    sky = _FakeSky(fail_times=0)
    ok = tear_down_and_record(sky, "lab-job-2", store, "job-2")
    assert ok is True
    assert store.read_manifest("job-2").teardown_status == "succeeded"


# ----------------------------------------------------------------------------
# provision_with_watchdog — bound the blocking stream_and_get so a dead Vast host
# stuck in "loading" can't hang the supervisor forever (FR-I1 provisioning guard).
# ----------------------------------------------------------------------------


class _FakeLaunchSky:
    """Stand-in for `sky` covering the provisioning surface: ``stream_and_get`` blocks/returns/
    raises per config; ``api_cancel`` records the abort call."""

    def __init__(
        self,
        *,
        sleep_s: float = 0.0,
        raise_exc: Exception | None = None,
        result: tuple = (1, "handle"),
    ) -> None:
        self.sleep_s = sleep_s
        self.raise_exc = raise_exc
        self.result = result
        self.api_cancel_calls: list[object] = []

    def stream_and_get(self, request_id: object) -> tuple:
        if self.sleep_s:
            time.sleep(self.sleep_s)  # real sleep — simulates a host stuck provisioning
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.result

    def api_cancel(self, request_id: object) -> None:
        self.api_cancel_calls.append(request_id)


def test_provision_watchdog_returns_on_fast_launch():
    sky = _FakeLaunchSky(result=(42, "handle"))
    job_id, handle = provision_with_watchdog(sky, "req-1", timeout_s=5.0)
    assert (job_id, handle) == (42, "handle")
    assert sky.api_cancel_calls == []  # healthy launch: never aborted


def test_provision_watchdog_times_out_on_hang():
    sky = _FakeLaunchSky(sleep_s=1.0)  # "host" never finishes provisioning
    with pytest.raises(ProvisionTimeout):
        provision_with_watchdog(sky, "req-2", timeout_s=0.05)
    assert sky.api_cancel_calls == ["req-2"]  # best-effort abort fired


def test_provision_watchdog_reraises_real_error():
    sky = _FakeLaunchSky(raise_exc=RuntimeError("offer disappeared"))
    with pytest.raises(RuntimeError, match="offer disappeared"):
        provision_with_watchdog(sky, "req-3", timeout_s=5.0)
    assert sky.api_cancel_calls == []  # a genuine error is not a timeout abort


# ----------------------------------------------------------------------------
# vast_hourly_for_cluster — bill at the rental's real dph_total, not SkyPilot's
# catalog get_cost() estimate which under-reports Vast prices ~4x (FR-I2).
# ----------------------------------------------------------------------------


def test_vast_hourly_for_cluster_returns_dph_total(monkeypatch: pytest.MonkeyPatch):
    instances = [
        {"id": 1, "label": "sky-lab-abc-xyz", "dph_total": 1.57},  # the rental for our cluster
        {"id": 2, "label": "someone-elses", "dph_total": 9.99},  # unrelated -> ignored
    ]
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda client=None: instances)
    assert vast_hourly_for_cluster("lab-abc") == 1.57


def test_vast_hourly_for_cluster_coerces_string_price(monkeypatch: pytest.MonkeyPatch):
    instances = [{"id": 1, "label": "sky-lab-abc-xyz", "dph_total": "1.57"}]  # API may stringify
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda client=None: instances)
    assert vast_hourly_for_cluster("lab-abc") == 1.57


def test_vast_hourly_for_cluster_none_when_no_match(monkeypatch: pytest.MonkeyPatch):
    instances = [{"id": 1, "label": "other-rental", "dph_total": 1.0}]
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda client=None: instances)
    assert vast_hourly_for_cluster("lab-abc") is None


def test_vast_hourly_for_cluster_none_when_price_missing(monkeypatch: pytest.MonkeyPatch):
    instances = [{"id": 1, "label": "sky-lab-abc-xyz"}]  # matches, but no dph_total field
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda client=None: instances)
    assert vast_hourly_for_cluster("lab-abc") is None


from lab.backends.skypilot import vast_balance


def test_vast_balance_reads_credit(monkeypatch):
    class _V:
        def show_user(self):
            return {"credit": -1.46, "balance": -1.46}

    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: _V())
    assert vast_balance() == -1.46


def test_vast_balance_none_on_error(monkeypatch):
    class _V:
        def show_user(self):
            raise RuntimeError("api down")

    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: _V())
    assert vast_balance() is None


from lab.sky_runner import provision_failure_reason


def test_provision_failure_reason_flags_negative_balance(monkeypatch):
    import lab.sky_runner as sr

    monkeypatch.setattr(sr, "vast_balance", lambda: -1.46)
    reason = provision_failure_reason("launch error: Failed to provision all possible resources")
    assert "balance is $-1.46" in reason and "top up" in reason


def test_provision_failure_reason_keeps_generic_when_funded(monkeypatch):
    import lab.sky_runner as sr

    monkeypatch.setattr(sr, "vast_balance", lambda: 25.0)
    generic = "launch error: Failed to provision all possible resources"
    assert provision_failure_reason(generic) == generic


def test_provision_failure_reason_keeps_generic_when_balance_unknown(monkeypatch):
    import lab.sky_runner as sr

    monkeypatch.setattr(sr, "vast_balance", lambda: None)
    generic = "launch error: Failed to provision all possible resources"
    assert provision_failure_reason(generic) == generic
