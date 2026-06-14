import json
from pathlib import Path

from helpers import PYTHON, make_manifest

import lab.sky_runner as sky_runner
from lab.models import JobState
from lab.runner import run_job
from lab.store import JobStore


def _prep(tmp_path: Path, job_id: str, command: str, **kw) -> JobStore:
    store = JobStore(tmp_path)
    store.create(make_manifest(job_id, command, **kw))
    return store


def test_success_injects_env_and_records_state(tmp_path: Path):
    # The example experiment reads $LAB_SEED and writes into $LAB_RUN_DIR (Experiment Contract).
    store = _prep(tmp_path, "s1", f"{PYTHON} experiments/example_capacity.py", seed=7)
    rc = run_job(store.job_dir("s1"))
    assert rc == 0
    m = store.read_manifest("s1")
    assert m.status == JobState.succeeded
    assert m.exit_code == 0
    assert m.started_at is not None and m.ended_at is not None
    result = json.loads((store.output_dir("s1") / "result.json").read_text())
    assert result["seed"] == 7  # LAB_SEED reached the experiment
    assert (store.output_dir("s1") / "metrics.jsonl").exists()


def test_failure_records_exit_code(tmp_path: Path):
    store = _prep(tmp_path, "f1", f'{PYTHON} -c "import sys; sys.exit(3)"')
    rc = run_job(store.job_dir("f1"))
    assert rc == 3
    m = store.read_manifest("f1")
    assert m.status == JobState.failed
    assert m.exit_code == 3


def test_timeout_terminates(tmp_path: Path):
    store = _prep(tmp_path, "t1", f'{PYTHON} -c "import time; time.sleep(30)"', timeout="1s")
    run_job(store.job_dir("t1"))
    m = store.read_manifest("t1")
    assert m.status == JobState.timed_out
    assert m.end_reason == "wall-clock timeout"


def test_wait_terminal_fires_heartbeat(monkeypatch):
    # Fake sky_mod whose queue reports RUNNING for several polls, then SUCCEEDED.
    polls = {"n": 0}

    class _Status:
        def __init__(self, name):
            self.name = name

    class _FakeSky:
        def get(self, x):
            return x

        def queue(self, cluster, skip_finished=False):
            polls["n"] += 1
            name = "RUNNING" if polls["n"] < 7 else "SUCCEEDED"
            return [{"job_id": 1, "status": _Status(name)}]

    monkeypatch.setattr(sky_runner.time, "sleep", lambda _s: None)  # no real waiting
    beats = {"n": 0}

    final, reached = sky_runner._wait_terminal(
        _FakeSky(),
        "lab-x",
        1,
        max_wait=10_000,
        poll_s=1.0,
        heartbeat_s=3.0,
        on_heartbeat=lambda: beats.__setitem__("n", beats["n"] + 1),
    )
    from lab.models import JobState

    assert final == JobState.succeeded
    assert reached is True  # broke on SUCCEEDED, not the deadline
    # 6 RUNNING polls before terminal, heartbeat every 3 polls -> fired at poll 3 and 6.
    assert beats["n"] == 2


class _StatusRec:
    def __init__(self, name):
        self.status = type("S", (), {"name": name})()


class _StatusSky:
    def __init__(self, recs=None, raise_exc=False):
        self._recs = recs
        self._raise = raise_exc

    def get(self, x):
        if self._raise:
            raise RuntimeError("api down")
        return x

    def status(self, cluster_names=None):
        return self._recs


def test_cluster_up_true_when_record_up():
    assert sky_runner._cluster_up(_StatusSky([_StatusRec("UP")]), "lab-x") is True


def test_cluster_up_false_when_empty():
    assert sky_runner._cluster_up(_StatusSky([]), "lab-x") is False


def test_cluster_up_false_when_not_up():
    assert sky_runner._cluster_up(_StatusSky([_StatusRec("STOPPED")]), "lab-x") is False


def test_cluster_up_false_on_exception_conservative():
    # Uncertainty must read as "gone" (False) so the classifier can fall through to its safe paths.
    assert sky_runner._cluster_up(_StatusSky(raise_exc=True), "lab-x") is False
