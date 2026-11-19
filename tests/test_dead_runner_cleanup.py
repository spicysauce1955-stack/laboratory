"""A local job whose runner died must not leave its experiment running unbounded.

Stress run 2026-11-19: SIGKILL the local runner of a running job, and the next status read marks
the job ``failed`` ("runner exited without recording status") — but the experiment, which runs in
its own session precisely so it can be killed as a group, kept running with nobody watching its
wall-clock cap, and ``lab cancel`` then refused it as terminal. ``lab ps`` meanwhile reported it
``running``/``n/a`` because it only checked supervisor liveness for skypilot jobs.

Finalizing a dead-runner job now stops its recorded process group — only when the group is
provably still the job's (the leader's recorded start time; a runtime file without one is never
signalled) — and ``lab ps`` checks local runners too.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import pytest
from helpers import PYTHON, make_manifest

import lab.runner as runner_mod
from lab import attribution
from lab._util import now, pid_alive, process_start_time
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.models import JobState
from lab.store import JobStore

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(), reason="process identity is read from /proc (Linux)"
)

_SLEEPER = """\
import os, sys, time
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(60)
"""


def _wait_for(predicate, timeout: float = 10.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


def _dead_pid() -> tuple[int, int]:
    """A pid (with its real start time) that has certainly exited and been reaped."""
    p = subprocess.Popen([PYTHON, "-c", "pass"])
    start = process_start_time(p.pid)
    p.wait()
    assert start is not None
    return p.pid, start


def _reap(pid: int | None) -> None:
    if pid is not None:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _running_job(tmp_path: Path, job_id: str, **runtime: object) -> LocalBackend:
    store = JobStore(tmp_path / "runs")
    store.create(make_manifest(job_id, "python x.py"))
    store.update_manifest(job_id, status=JobState.running)
    store.write_runtime(job_id, **runtime)
    return LocalBackend(home=tmp_path / "runs", repo=tmp_path)


class TestFinalizingADeadRunnerStopsItsExperiment:
    def test_a_killed_runner_does_not_leave_its_experiment_running(self, tmp_path, monkeypatch):
        """The end-to-end shape of the stress finding, against the real runner."""
        monkeypatch.setattr(runner_mod, "KILL_GRACE_S", 2.0)
        script = tmp_path / "sleeper.py"
        script.write_text(_SLEEPER)
        pidfile = tmp_path / "experiment.pid"
        store = JobStore(tmp_path / "runs")
        store.create(make_manifest("dr-real", f"{PYTHON} {script} {pidfile}"))
        backend = LocalBackend(home=tmp_path / "runs")  # repo = cwd: the runner needs `lab`
        backend.submit(store.read_manifest("dr-real"))
        exp_pid = None
        try:
            _wait_for(pidfile.exists, what="the experiment to start")
            exp_pid = int(pidfile.read_text())
            _wait_for(
                lambda: store.read_runtime("dr-real").get("command_start_time"),
                what="the runner to record the command group",
            )
            os.kill(store.read_runtime("dr-real")["runner_pid"], signal.SIGKILL)
            _wait_for(
                lambda: not pid_alive(
                    store.read_runtime("dr-real")["runner_pid"],
                    start_time=store.read_runtime("dr-real")["runner_start_time"],
                ),
                what="the runner to die",
            )
            assert pid_alive(exp_pid), "precondition: the experiment outlives its runner"

            assert backend.status("dr-real") is JobState.failed
            _wait_for(lambda: not pid_alive(exp_pid), timeout=5.0, what="the experiment to stop")
            reason = store.read_manifest("dr-real").end_reason or ""
            assert reason.startswith("runner exited without recording status")
            assert "stopped" in reason
        finally:
            _reap(exp_pid)

    def test_a_group_whose_leader_identity_does_not_match_is_never_signalled(self, tmp_path):
        runner_pid, runner_start = _dead_pid()
        decoy = subprocess.Popen([PYTHON, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            real = process_start_time(decoy.pid)
            assert real is not None
            backend = _running_job(
                tmp_path, "dr-recycled",
                runner_pid=runner_pid, runner_start_time=runner_start,
                command_pgid=decoy.pid, command_start_time=real + 1,
            )
            assert backend.status("dr-recycled") is JobState.failed
            time.sleep(0.3)
            assert decoy.poll() is None, "finalizing killed an unrelated process group"
        finally:
            decoy.kill()
            decoy.wait()

    def test_a_group_without_a_recorded_identity_is_never_signalled(self, tmp_path):
        """A runtime file from before the start time was recorded cannot prove the group is
        still ours; an automatic kill from a status poll stays out of it."""
        runner_pid, runner_start = _dead_pid()
        decoy = subprocess.Popen([PYTHON, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            backend = _running_job(
                tmp_path, "dr-legacy",
                runner_pid=runner_pid, runner_start_time=runner_start, command_pgid=decoy.pid,
            )
            assert backend.status("dr-legacy") is JobState.failed
            time.sleep(0.3)
            assert decoy.poll() is None
            assert "not signalled" in (backend.store.read_manifest("dr-legacy").end_reason or "")
        finally:
            decoy.kill()
            decoy.wait()

    def test_an_experiment_that_had_already_exited_is_still_failed_not_succeeded(self, tmp_path):
        """The runner died after its experiment finished but before recording the verdict. The
        outputs may be complete, but the exit code is gone: nothing proves success, so the job
        stays ``failed`` (its rows still aggregate as partial, so nothing paid for is lost) and
        the reason says the outputs were kept."""
        runner_pid, runner_start = _dead_pid()
        exp_pid, exp_start = _dead_pid()
        backend = _running_job(
            tmp_path, "dr-finished",
            runner_pid=runner_pid, runner_start_time=runner_start,
            command_pgid=exp_pid, command_start_time=exp_start,
        )
        out = backend.store.output_dir("dr-finished")
        (out / "results.csv").write_text("seed,acc\n0,0.9\n")
        assert backend.status("dr-finished") is JobState.failed
        reason = backend.store.read_manifest("dr-finished").end_reason or ""
        assert reason.startswith("runner exited without recording status")
        assert "exit code" in reason and "not marked succeeded" in reason
        assert (out / "results.csv").exists()

    def test_a_runner_that_never_launched_its_command_keeps_the_plain_reason(self, tmp_path):
        runner_pid, runner_start = _dead_pid()
        backend = _running_job(
            tmp_path, "dr-early", runner_pid=runner_pid, runner_start_time=runner_start
        )
        assert backend.status("dr-early") is JobState.failed
        assert (
            backend.store.read_manifest("dr-early").end_reason
            == "runner exited without recording status"
        )


class TestPsChecksLocalRunners:
    @pytest.fixture(autouse=True)
    def _registry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LAB_JOBS_INDEX_DIR", str(tmp_path / "lab-jobs"))

    def _lab(self, home: Path) -> Lab:
        return Lab(backend=LocalBackend(home=home, repo=home), repo=home, home=home)

    def _seed(self, lab: Lab, job_id: str, *, started_ago_s: float, **runtime: object) -> None:
        m = make_manifest(job_id, "python x.py").model_copy(
            update={
                "status": JobState.running,
                "started_at": now() - timedelta(seconds=started_ago_s),
            }
        )
        lab.store.create(m)
        attribution.record_job(job_id, project=lab.home.name, runs_dir=lab.store.home)
        if runtime:
            lab.store.write_runtime(job_id, **runtime)

    def test_a_local_job_whose_runner_died_is_unsupervised(self, tmp_path: Path) -> None:
        lab = self._lab(tmp_path / "proj")
        pid, start = _dead_pid()
        self._seed(lab, "ps-dead", started_ago_s=3600.0, runner_pid=pid, runner_start_time=start)
        (job,) = lab.ps()["jobs"]
        assert job["backend"] == "local"
        assert job["supervised"] == "unsupervised"

    def test_a_young_local_job_whose_recorded_runner_died_is_not_starting(
        self, tmp_path: Path
    ) -> None:
        """The grace covers a runner that has not recorded its pid yet. A local runner's pid is
        recorded before `submit` returns, so a recorded pid that is dead is a dead runner."""
        lab = self._lab(tmp_path / "proj")
        pid, start = _dead_pid()
        self._seed(lab, "ps-young-dead", started_ago_s=2.0, runner_pid=pid, runner_start_time=start)
        (job,) = lab.ps()["jobs"]
        assert job["supervised"] == "unsupervised"

    def test_a_local_job_with_a_live_runner_is_supervised(self, tmp_path: Path) -> None:
        lab = self._lab(tmp_path / "proj")
        me = os.getpid()
        self._seed(
            lab, "ps-alive", started_ago_s=3600.0,
            runner_pid=me, runner_start_time=process_start_time(me),
        )
        (job,) = lab.ps()["jobs"]
        assert job["supervised"] == "local"

    def test_a_fresh_local_job_without_a_runner_pid_yet_is_starting(self, tmp_path: Path) -> None:
        lab = self._lab(tmp_path / "proj")
        self._seed(lab, "ps-fresh", started_ago_s=1.0)
        (job,) = lab.ps()["jobs"]
        assert job["supervised"] == "starting"

    def test_ps_never_finalizes_or_signals(self, tmp_path: Path) -> None:
        """`ps` is the look-before-you-touch command: it reports, it does not act."""
        lab = self._lab(tmp_path / "proj")
        pid, start = _dead_pid()
        self._seed(lab, "ps-ro", started_ago_s=3600.0, runner_pid=pid, runner_start_time=start)
        lab.ps()
        assert lab.manifest("ps-ro").status is JobState.running
