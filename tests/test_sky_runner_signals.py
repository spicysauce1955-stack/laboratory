"""A SIGTERMed supervisor must leave a labelled event and a torn-down machine (R13 / F1).

Measured over the whole live ledger on the dev box on 2026-08-20: **15 of 52** ``supervisor/run``
calls never wrote a close line, so they read as ``running-or-died`` — "opened, never recorded an
outcome". The field report called that "DO supervisors die silently"; the re-adjudication (F1) is
that they were *spinning* on a lost cluster (R8, fixed in ``0822fae``) until something SIGTERMed
them. Python's default SIGTERM disposition terminates the process without raising, so ``run_job``'s
``except BaseException`` — the one thing that reliably records a close — never runs. Both the
ledger row **and any teardown that was in flight** are abandoned, and the abandoned teardown is
the expensive half: ``lab cancel`` and the scheduler watchdog both stop a supervisor by sending
exactly this signal (``backends/skypilot.py`` ``os.kill(runner_pid, SIGTERM)``,
``scheduler/tick.py`` ``os.killpg(pgid, SIGTERM)``).

These tests drive the real entrypoint in a real subprocess and send a real signal, because the
whole defect lives in the interpreter's *default* disposition — a mocked ``signal.signal`` would
prove nothing about it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest

import lab.sky_runner as runner_mod
from helpers import PYTHON, make_manifest
from lab._util import now
from lab.events import fold
from lab.events import store as events_store
from lab.models import CostInfo, JobState
from lab.store import JobStore

# The supervisor stubs sky out and then blocks where a real one blocks — in `_wait_terminal`,
# polling a remote queue — so the signal lands mid-run, exactly as it did live. `--adopt` is used
# only to skip `sky.launch`: it is the shortest real path to "a machine exists and we are waiting
# on it", which is the state that makes an abandoned teardown cost money.
_DRIVER = '''\
import json, sys, time, types
from pathlib import Path

job_dir, ready, marker, delay = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])

import lab.sky_runner as runner
from lab.models import CostInfo

fake_sky = types.ModuleType("sky")
fake_sky.tail_logs = lambda *a, **k: None
sys.modules["sky"] = fake_sky


def _teardown(sky_mod, cluster, store, job_id, cloud="vast", **kw):
    with marker.open("a") as fh:
        fh.write(json.dumps({"event": "start", "cluster": cluster, "job_id": job_id,
                             "cloud": cloud, "backoffs": kw.get("backoffs")}) + "\\n")
    time.sleep(delay)
    with marker.open("a") as fh:
        fh.write(json.dumps({"event": "done"}) + "\\n")
    return True


def _block(*a, **k):
    ready.write_text("waiting")
    while True:
        time.sleep(0.02)


runner.tear_down_and_record = _teardown
runner.resolve_cost = lambda *a, **k: CostInfo(hourly_usd=0.5, estimated_usd=0.5)
runner._rsync_down = lambda *a, **k: None
runner._wait_terminal = _block

runner.main([str(job_dir), "--adopt"])
'''


def _seed_running_job(tmp_path: Path, job_id: str = "j1") -> JobStore:
    home = tmp_path / "runs"
    store = JobStore(home)
    manifest = make_manifest(job_id, "python x.py", timeout="1h").model_copy(
        update={
            "status": JobState.running,
            "started_at": now(),
            "cost": CostInfo(hourly_usd=0.5, estimated_usd=0.5),
        }
    )
    store.create(manifest)
    store.write_runtime(job_id, cluster=f"lab-{job_id}")
    return store


class _Supervisor:
    """A real supervisor subprocess, parked in its wait loop and ready to be signalled."""

    def __init__(self, proc: subprocess.Popen[str], marker: Path, log: Path) -> None:
        self.proc = proc
        self.marker = marker
        self.log = log

    def teardowns(self) -> list[dict[str, object]]:
        if not self.marker.exists():
            return []
        return [json.loads(line) for line in self.marker.read_text().splitlines() if line.strip()]

    def wait(self, timeout: float = 30.0) -> int:
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - only on a regression
            self.proc.kill()
            raise AssertionError(
                f"supervisor did not exit within {timeout}s after SIGTERM — deadlock.\n"
                f"job log:\n{self.log.read_text() if self.log.exists() else '(none)'}"
            ) from None


def _start_supervisor(tmp_path: Path, store: JobStore, *, teardown_delay: float = 0.0):
    """Launch the driver and block until it is parked in the wait loop."""
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)
    ready, marker = tmp_path / "ready", tmp_path / "teardown.jsonl"
    proc = subprocess.Popen(
        [PYTHON, str(driver), str(store.home / "j1"), str(ready), str(marker),
         str(teardown_delay)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),  # carries the per-test LAB_EVENTS_DIR / LAB_JOBS_INDEX_DIR
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        if ready.exists():
            return _Supervisor(proc, marker, store.logs_path("j1"))
        if proc.poll() is not None:
            raise AssertionError(f"supervisor exited early ({proc.returncode}): {proc.communicate()[0]}")
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("supervisor never reached its wait loop")


def _supervisor_events():
    records = list(events_store.iter_records(events_store.day_files()))
    return [e for e in fold(records) if e.surface == "supervisor" and e.action == "run"]


# ---------------------------------------------------------------------------
# The defect itself: a signalled supervisor must close its ledger call
# ---------------------------------------------------------------------------


def test_sigterm_closes_the_ledger_call(tmp_path: Path) -> None:
    """The 29% never-closed rate, reproduced and pinned at zero for this path."""
    store = _seed_running_job(tmp_path)
    sup = _start_supervisor(tmp_path, store)

    sup.proc.send_signal(signal.SIGTERM)
    code = sup.wait()

    assert code == 143, "128 + SIGTERM — the conventional exit code for a signalled process"
    events = _supervisor_events()
    assert len(events) == 1, events
    event = events[0]
    assert event.status != "running-or-died", "the close line was never written"
    assert event.status == "interrupted"
    assert event.exit_code == 143
    assert event.refs.get("job_id") == "j1"
    # The trace says *why* it ended — an unattributable death is the thing being fixed.
    assert [n.d for n in event.trace if n.k == "signal"] == [{"sig": "TERM"}]
    assert event.error is not None and "TERM" in event.error["message"]


def test_sigterm_still_tears_the_machine_down(tmp_path: Path) -> None:
    """The expensive half: a signal must not skip teardown (FR-C2)."""
    store = _seed_running_job(tmp_path)
    sup = _start_supervisor(tmp_path, store)

    sup.proc.send_signal(signal.SIGTERM)
    sup.wait()

    events = [t for t in sup.teardowns() if t["event"] == "start"]
    assert len(events) == 1, f"teardown ran {len(events)} time(s): {sup.teardowns()}"
    assert events[0]["cluster"] == "lab-j1"
    assert events[0]["job_id"] == "j1"


def test_sigterm_marks_the_job_terminal(tmp_path: Path) -> None:
    """A supervisor that is gone must not leave the job reading `running` forever — that is what
    made `lab wait` hang and drove the operator's 5-minute status poller (P2)."""
    store = _seed_running_job(tmp_path)
    sup = _start_supervisor(tmp_path, store)

    sup.proc.send_signal(signal.SIGTERM)
    sup.wait()

    manifest = store.read_manifest("j1")
    assert manifest.status is JobState.failed
    assert manifest.ended_at is not None
    assert "SIGTERM" in (manifest.end_reason or "")


def test_a_second_sigterm_neither_deadlocks_nor_double_closes(tmp_path: Path) -> None:
    """Whoever sends one SIGTERM usually sends another when the process doesn't vanish. The
    second must not re-enter the handler: the first is already unwinding *through the teardown*,
    and interrupting that is precisely the leak this task exists to close."""
    store = _seed_running_job(tmp_path)
    sup = _start_supervisor(tmp_path, store, teardown_delay=1.5)

    sup.proc.send_signal(signal.SIGTERM)
    deadline = time.time() + 30
    while time.time() < deadline and not sup.teardowns():
        time.sleep(0.02)
    assert sup.teardowns(), "teardown never started"
    sup.proc.send_signal(signal.SIGTERM)  # lands mid-teardown

    assert sup.wait() == 143
    assert [t["event"] for t in sup.teardowns()] == ["start", "done"], "teardown was cut short"
    events = _supervisor_events()
    assert len(events) == 1
    assert events[0].status == "interrupted"
    closes = [
        r
        for r in events_store.iter_records(events_store.day_files())
        if r.get("phase") == "close" and r.get("id") == events[0].id
    ]
    assert len(closes) == 1, f"expected exactly one close line, got {closes}"


# ---------------------------------------------------------------------------
# Scoping: the handler belongs to the entrypoint, not to whoever imports the module
# ---------------------------------------------------------------------------


def test_importing_the_module_installs_no_handler() -> None:
    """`lab.sky_runner` is imported by the backend, the scheduler and the tests. Hijacking
    SIGTERM in a process that merely imported it would break `lab cancel`'s own kill path."""
    probe = (
        "import signal, sys;"
        "import lab.sky_runner;"
        "sys.stdout.write(repr(signal.getsignal(signal.SIGTERM)))"
    )
    out = subprocess.run(
        [PYTHON, "-c", probe], capture_output=True, text=True, check=True, env=os.environ.copy()
    ).stdout
    assert "SIG_DFL" in out, out


def test_installing_off_the_main_thread_degrades_instead_of_crashing() -> None:
    """`signal.signal` raises off the main thread. A supervisor that cannot install the handler
    must still supervise — losing the labelled close is worse than nothing, not fatal."""
    result: list[object] = []

    def _try() -> None:
        try:
            result.append(runner_mod.install_signal_handlers())
        except BaseException as e:  # noqa: BLE001 — the point is that nothing escapes
            result.append(e)

    thread = threading.Thread(target=_try)
    thread.start()
    thread.join(10)
    assert result == [[]], result


@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_the_supervisor_claims_the_silent_killers(name: str) -> None:
    """SIGINT is deliberately absent: Python already turns it into `KeyboardInterrupt`, which
    `run_job` catches. These two terminate the process without raising anything at all."""
    assert getattr(signal, name) in runner_mod._TERMINATION_SIGNALS
    assert signal.SIGINT not in runner_mod._TERMINATION_SIGNALS
