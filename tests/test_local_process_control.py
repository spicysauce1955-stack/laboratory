"""Local process control: what `timed_out`/`cancelled` record must actually be true.

Two holes, both reproduced against the real runner before this was fixed:

* **Nothing ever escalated to SIGKILL for the experiment.** The entrypoint runs under ``sh -c``,
  so the process the runner holds is the shell. On a timeout the group got SIGTERM, the shell died
  of it at once, ``proc.wait`` returned, and the ``proc.kill()`` fallback (which only ever targeted
  the shell anyway) never ran. ``cancel`` sent SIGTERM and nothing else. An experiment that ignores
  SIGTERM was still running 16 s after both ``timed_out`` and ``cancelled`` were recorded — on a
  shared box it keeps burning the CPU the next job was promised, and it can keep writing into a
  run dir whose verdict is already final.
* **``cancel`` signalled a bare PID.** ``runner_pid`` is recorded with its start time precisely so
  a recycled number can be told apart, but ``cancel`` never checked it: a reused PID killed an
  unrelated decoy process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from helpers import PYTHON, make_manifest

import lab.runner as runner_mod
from lab._util import pid_alive, process_start_time
from lab.backends.local import LocalBackend
from lab.models import JobState
from lab.runner import run_job
from lab.store import JobStore

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(), reason="process identity is read from /proc (Linux)"
)

_IGNORER = """\
import os, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(60)
"""


def _ignorer_command(tmp_path: Path) -> tuple[str, Path]:
    """A compound shell command (so ``sh`` stays the parent) around a SIGTERM-ignoring python."""
    script = tmp_path / "ignore_sigterm.py"
    script.write_text(_IGNORER)
    pidfile = tmp_path / "experiment.pid"
    return f"{PYTHON} {script} {pidfile} ; echo done", pidfile


def _read_pid(pidfile: Path, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pidfile.exists():
            return int(pidfile.read_text())
        time.sleep(0.05)
    raise AssertionError("the experiment never started")


def _dies_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid)


def _reap(pid: int | None) -> None:
    """Never leave a sleeper behind if an assertion fires first."""
    if pid is not None:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


class TestTimeoutEscalates:
    def test_a_sigterm_ignoring_experiment_is_killed_after_the_grace(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod, "KILL_GRACE_S", 0.5, raising=False)
        command, pidfile = _ignorer_command(tmp_path)
        store = JobStore(tmp_path / "runs")
        store.create(make_manifest("t-ignore", command, timeout="1s"))
        pid = None
        try:
            started = time.monotonic()
            run_job(store.job_dir("t-ignore"))
            elapsed = time.monotonic() - started
            pid = _read_pid(pidfile)
            assert store.read_manifest("t-ignore").status == JobState.timed_out
            assert _dies_within(pid, 2.0), "timed_out was recorded but the experiment still runs"
            assert elapsed < 10, f"the runner should give up after the grace, took {elapsed:.1f}s"
        finally:
            _reap(pid)

    def test_a_well_behaved_experiment_still_ends_on_sigterm(self, tmp_path, monkeypatch):
        """The grace is a ceiling, not a sleep: a job that honours SIGTERM ends at once."""
        monkeypatch.setattr(runner_mod, "KILL_GRACE_S", 30.0, raising=False)
        store = JobStore(tmp_path / "runs")
        store.create(
            make_manifest("t-ok", f'{PYTHON} -c "import time; time.sleep(60)"', timeout="1s")
        )
        started = time.monotonic()
        run_job(store.job_dir("t-ok"))
        assert time.monotonic() - started < 10
        assert store.read_manifest("t-ok").status == JobState.timed_out

    def test_the_command_group_identity_is_recorded(self, tmp_path):
        """`cancel` (another process) needs the leader's start time to verify the group."""
        store = JobStore(tmp_path / "runs")
        store.create(make_manifest("t-rt", f'{PYTHON} -c "pass"'))
        run_job(store.job_dir("t-rt"))
        rt = store.read_runtime("t-rt")
        assert rt.get("command_pgid")
        assert isinstance(rt.get("command_start_time"), int)


def _running_job(tmp_path: Path, job_id: str, **runtime: object) -> LocalBackend:
    store = JobStore(tmp_path / "runs")
    store.create(make_manifest(job_id, "python x.py"))
    store.update_manifest(job_id, status=JobState.running)
    if runtime:
        store.write_runtime(job_id, **runtime)
    return LocalBackend(home=tmp_path / "runs", repo=tmp_path)


class TestCancel:
    def test_cancel_kills_a_sigterm_ignoring_experiment(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod, "KILL_GRACE_S", 0.5, raising=False)
        command, pidfile = _ignorer_command(tmp_path)
        store = JobStore(tmp_path / "runs")
        store.create(make_manifest("c-ignore", command))
        backend = LocalBackend(home=tmp_path / "runs")  # repo = cwd: the runner needs `lab`
        backend.submit(store.read_manifest("c-ignore"))
        pid = None
        try:
            pid = _read_pid(pidfile)
            deadline = time.monotonic() + 10
            while not store.read_runtime("c-ignore").get("command_pgid"):
                assert time.monotonic() < deadline, "the runner never recorded the command group"
                time.sleep(0.05)
            assert backend.cancel("c-ignore") == JobState.cancelled
            assert _dies_within(pid, 2.0), "cancelled was recorded but the experiment still runs"
        finally:
            _reap(pid)

    def test_cancel_does_not_signal_a_recycled_runner_pid(self, tmp_path):
        decoy = subprocess.Popen([PYTHON, "-c", "import time; time.sleep(30)"])
        try:
            real = process_start_time(decoy.pid)
            assert real is not None
            backend = _running_job(
                tmp_path, "c-recycled-runner", runner_pid=decoy.pid, runner_start_time=real + 1
            )
            assert backend.cancel("c-recycled-runner") == JobState.cancelled
            time.sleep(0.3)
            assert decoy.poll() is None, "cancel killed an unrelated process holding a reused PID"
        finally:
            decoy.kill()
            decoy.wait()

    def test_cancel_does_not_signal_a_recycled_command_group(self, tmp_path):
        decoy = subprocess.Popen(
            [PYTHON, "-c", "import time; time.sleep(30)"], start_new_session=True
        )
        try:
            real = process_start_time(decoy.pid)
            assert real is not None
            backend = _running_job(
                tmp_path, "c-recycled-group",
                command_pgid=decoy.pid, command_start_time=real + 1,
            )
            assert backend.cancel("c-recycled-group") == JobState.cancelled
            time.sleep(0.3)
            assert decoy.poll() is None, "cancel killed an unrelated process group"
        finally:
            decoy.kill()
            decoy.wait()

    def test_cancel_still_signals_a_runner_whose_identity_matches(self, tmp_path):
        """The identity check must not turn cancel into a no-op for the real runner."""
        stand_in = subprocess.Popen([PYTHON, "-c", "import time; time.sleep(30)"])
        try:
            backend = _running_job(
                tmp_path, "c-real-runner",
                runner_pid=stand_in.pid, runner_start_time=process_start_time(stand_in.pid),
            )
            backend.cancel("c-real-runner")
            assert stand_in.wait(timeout=5) == -signal.SIGTERM
        finally:
            if stand_in.poll() is None:
                stand_in.kill()
                stand_in.wait()
