"""What the supervisor's death can and cannot be made to prove (F5), plus label hygiene (R11).

**F5 — the honest scope.** ``submit`` spawns the supervisor with ``start_new_session=True`` and
never waits on the handle, on purpose: the job must outlive the CLI. The cost is that the OS-level
exit status -- clean exit vs ``SIGTERM`` vs the ``SIGKILL`` an OOM kill sends -- is a one-shot
value that only the *parent* can collect, and only until the child is reaped. When the parent is
``lab submit``, it exits within milliseconds, the supervisor reparents to init, and init reaps it
the instant it dies. **In that case the exit status is gone before any lab process could look, and
no amount of later polling recovers it.** A design that pretends otherwise would record nothing
while looking like it worked.

So this pins a two-tier capture and, explicitly, its floor:

* **``waitpid``** -- the spawning process is still alive (the MCP server, ``lab sweep``,
  ``lab submit --wait``, the scheduler). Its ``Popen`` handle is kept in a registry, and a poll
  yields the *exact* status, signal included. This is the agent-facing path, so it is the common
  one.
* **``proc_zombie``** -- somebody else's live parent holds the corpse unreaped. ``/proc/<pid>/stat``
  field 52 carries the waitpid-form status and is readable by any same-uid process, so a
  ``lab status`` in another terminal reads the exact signal too.
* **``recycled`` / ``disappeared``** -- nobody was holding it. The exit status is unrecoverable and
  is recorded *as* unrecoverable, with the reason, rather than left blank. Recorded-and-unknown and
  never-looked-at must not read the same, which is the whole complaint behind F5.

``recycled`` is separable from ``disappeared`` only because the spawn already records
``runner_start_time`` (F4): a PID whose kernel start-time changed is provably not our supervisor.

**R11** is unrelated and small: ``_instance_label`` padded its output with the separators of the
fields an instance did not carry, so a Vast rental with only ``label`` rendered with three trailing
spaces in reconcile reports and teardown log lines.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap
import types

import pytest
from helpers import make_manifest

from lab.backends import skypilot as m
from lab.backends.skypilot import SkyPilotBackend, cluster_name_for
from lab.models import BackendInfo, JobState

CLUSTER = "lab-laboratory-20260820-071905-771110"
RENTAL = f"{CLUSTER}-3dd12990-f5bf-head"


# ---------------------------------------------------------------------------
# R11 — label hygiene
# ---------------------------------------------------------------------------


class TestInstanceLabel:
    def test_a_single_field_instance_is_exactly_the_bare_name(self):
        """The live shape: a Vast rental carries `label` and nothing else.

        It used to render as `"...-head   "` -- the three empty candidate fields joined in --
        which leaks into every reconcile report and teardown log line an operator reads.
        """
        assert m._instance_label({"label": RENTAL}) == RENTAL.lower()

    @pytest.mark.parametrize(
        "inst",
        [
            {"label": RENTAL},
            {"name": RENTAL},
            {"instance_label": RENTAL},
            {"machine_name": RENTAL},
            {"label": RENTAL, "machine_name": RENTAL},
            {"label": "", "name": RENTAL},
            {"label": None, "name": RENTAL},
            {},
        ],
    )
    def test_a_label_never_carries_edge_whitespace(self, inst):
        label = m._instance_label(inst)
        assert label == label.strip()

    def test_every_candidate_field_is_still_probed(self):
        """The multi-field probe is the point of the function -- a Vast SDK rename must not
        silently disable matching. Trimming must not turn into "read `label` only"."""
        for field in ("label", "name", "instance_label", "machine_name"):
            assert RENTAL.lower() in m._instance_label({field: RENTAL})

    def test_present_fields_stay_separated(self):
        """Joining without a separator would let two adjacent fields form a substring match that
        neither field actually contains -- matching here is an `in` test."""
        assert m._instance_label({"label": "aaa", "name": "bbb"}) == "aaa bbb"


# ---------------------------------------------------------------------------
# F5 — supervisor exit capture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is module-global by necessity (it holds this process's own children). Leaving
    a test's entries in it would leak a real process into the next test."""
    yield
    for _job_id, (proc, _store) in list(m._SUPERVISORS.items()):
        proc.kill()
        proc.wait()
    m._SUPERVISORS.clear()


def _registered_job(tmp_path, job_id="jexit", *, script="import time; time.sleep(60)"):
    """A job whose supervisor is a real local process this test owns."""
    backend = SkyPilotBackend(home=tmp_path, repo=tmp_path)
    manifest = make_manifest(job_id, "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running, "backend": BackendInfo(provisioner="skypilot")}
    )
    backend.store.create(manifest)
    proc = subprocess.Popen([sys.executable, "-c", script])
    backend.store.write_runtime(
        job_id,
        runner_pid=proc.pid,
        runner_start_time=m.process_start_time(proc.pid),
        cluster=cluster_name_for(job_id),
    )
    m._register_supervisor(job_id, proc, backend.store)
    return backend, proc


def _await_exit(proc: subprocess.Popen) -> None:
    """Wait for the OS to finish killing it without reaping -- the record under test is what
    `observe_supervisor_exit` collects, so the test must not collect it first."""
    os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)


class TestExactCaptureWhileTheSpawnerLives:
    """Tier 1. The MCP server, `lab sweep` and `lab submit --wait` all outlive their supervisors,
    and there the exit status is exact -- signal included."""

    def test_a_sigkill_is_named(self, tmp_path):
        """The case F5 exists for: an OOM kill is currently unprovable after the fact."""
        backend, proc = _registered_job(tmp_path)
        os.kill(proc.pid, signal.SIGKILL)
        _await_exit(proc)

        rec = m.observe_supervisor_exit(backend.store, "jexit")

        assert rec is not None
        assert rec["source"] == "waitpid"
        assert rec["signal"] == "SIGKILL"
        assert rec["returncode"] is None
        assert "OOM" in rec["detail"]

    def test_a_sigterm_is_distinguished_from_a_sigkill(self, tmp_path):
        """`lab cancel` sends SIGTERM; the OOM killer sends SIGKILL. Conflating them is how a
        deliberate cancel and a machine running out of memory read the same in the ledger."""
        backend, proc = _registered_job(tmp_path)
        os.kill(proc.pid, signal.SIGTERM)
        _await_exit(proc)

        rec = m.observe_supervisor_exit(backend.store, "jexit")

        assert rec is not None and rec["signal"] == "SIGTERM"

    def test_a_clean_exit_reports_its_code(self, tmp_path):
        backend, proc = _registered_job(tmp_path, script="raise SystemExit(7)")
        _await_exit(proc)

        rec = m.observe_supervisor_exit(backend.store, "jexit")

        assert rec is not None
        assert (rec["returncode"], rec["signal"]) == (7, None)

    def test_a_live_supervisor_records_nothing(self, tmp_path):
        """No record means "still running". Writing a speculative one would make every poll of a
        healthy job look like a death."""
        backend, proc = _registered_job(tmp_path)
        try:
            assert m.observe_supervisor_exit(backend.store, "jexit") is None
            assert "runner_exit" not in backend.store.read_runtime("jexit")
        finally:
            proc.kill()
            proc.wait()

    def test_the_record_is_durable_and_written_once(self, tmp_path):
        """It has to survive the process that observed it -- that is the entire requirement --
        and a second observation must not overwrite the exact answer with a guess."""
        backend, proc = _registered_job(tmp_path)
        os.kill(proc.pid, signal.SIGKILL)
        _await_exit(proc)

        first = m.observe_supervisor_exit(backend.store, "jexit")
        on_disk = json.loads((tmp_path / "jexit" / "_runtime.json").read_text())["runner_exit"]
        assert on_disk == first

        # The pid is reaped and gone now; a fresh reader (no registry, no /proc) must still see
        # the exact answer rather than degrading it to "disappeared".
        m._SUPERVISORS.pop("jexit", None)
        assert m.observe_supervisor_exit(backend.store, "jexit") == first

    def test_polling_reaps_the_child(self, tmp_path):
        """Holding the handle is what makes the status collectable, but it also stops CPython's
        opportunistic `subprocess._cleanup` from reaping. The poll must do that job instead, or a
        long-lived MCP server accumulates one zombie per job it ever submitted."""
        backend, proc = _registered_job(tmp_path)
        os.kill(proc.pid, signal.SIGKILL)
        _await_exit(proc)

        m.observe_supervisor_exit(backend.store, "jexit")

        assert m.process_start_time(proc.pid) is None, "the corpse was never reaped"
        assert "jexit" not in m._SUPERVISORS


class TestCaptureFromAnUnreapedCorpse:
    """Tier 2. Somebody else's parent is holding it -- a `lab status` from another terminal while
    the MCP server holds the handle. `/proc/<pid>/stat` field 52 is the waitpid-form status and is
    readable by any process of the same uid."""

    @pytest.fixture
    def zombie(self):
        holder = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent("""
                import os, signal, subprocess, sys, time
                c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                time.sleep(0.2)
                os.kill(c.pid, signal.SIGKILL)
                time.sleep(0.2)
                print(c.pid, flush=True)
                time.sleep(30)
            """)],
            stdout=subprocess.PIPE, text=True,
        )
        assert holder.stdout is not None
        pid = int(holder.stdout.readline().strip())
        yield pid
        holder.kill()
        holder.wait()

    def test_the_signal_is_read_out_of_proc(self, tmp_path, zombie):
        backend = SkyPilotBackend(home=tmp_path, repo=tmp_path)
        backend.store.create(
            make_manifest("jzomb", "python x.py").model_copy(
                update={"status": JobState.running, "backend": BackendInfo(provisioner="skypilot")}
            )
        )
        backend.store.write_runtime(
            "jzomb", runner_pid=zombie, runner_start_time=m.process_start_time(zombie)
        )

        rec = m.observe_supervisor_exit(backend.store, "jzomb")

        assert rec is not None
        assert rec["source"] == "proc_zombie"
        assert rec["signal"] == "SIGKILL"


class TestWhatCannotBeProved:
    """The floor, pinned deliberately. After a bare `lab submit`, init reaps the supervisor the
    moment it dies and the exit status is unrecoverable. These tests exist so that fact stays
    *recorded* rather than silently becoming an empty field."""

    def _job(self, tmp_path, job_id, pid, start_time):
        backend = SkyPilotBackend(home=tmp_path, repo=tmp_path)
        backend.store.create(
            make_manifest(job_id, "python x.py").model_copy(
                update={"status": JobState.running, "backend": BackendInfo(provisioner="skypilot")}
            )
        )
        backend.store.write_runtime(job_id, runner_pid=pid, runner_start_time=start_time)
        return backend

    def test_a_reaped_pid_records_that_the_status_is_unrecoverable(self, tmp_path):
        backend = self._job(tmp_path, "jgone", 999999999, 12345)

        rec = m.observe_supervisor_exit(backend.store, "jgone")

        assert rec is not None
        assert rec["source"] == "disappeared"
        assert (rec["returncode"], rec["signal"]) == (None, None)
        assert "unrecoverable" in rec["detail"] and "999999999" in rec["detail"]

    def test_a_recycled_pid_is_not_confused_with_a_reaped_one(self, tmp_path):
        """Only the recorded start-time (F4) separates these. Without it the number is held by
        *something*, and the supervisor reads alive forever."""
        recorded = m.process_start_time(os.getpid())
        assert recorded is not None
        backend = self._job(tmp_path, "jrecyc", os.getpid(), recorded + 1_000_000)

        rec = m.observe_supervisor_exit(backend.store, "jrecyc")

        assert rec is not None
        assert rec["source"] == "recycled"
        assert (rec["returncode"], rec["signal"]) == (None, None)

    def test_a_job_that_never_recorded_a_pid_is_not_declared_dead(self, tmp_path):
        """Legacy runtime files and jobs still queued have no pid. Unknown must stay unknown."""
        backend = SkyPilotBackend(home=tmp_path, repo=tmp_path)
        backend.store.create(make_manifest("jnopid", "python x.py"))

        assert m.observe_supervisor_exit(backend.store, "jnopid") is None

    def test_a_live_matching_pid_is_alive(self, tmp_path):
        backend = self._job(tmp_path, "jself", os.getpid(), m.process_start_time(os.getpid()))

        assert m.observe_supervisor_exit(backend.store, "jself") is None

    def test_no_procfs_is_not_a_death(self, tmp_path, monkeypatch):
        """The whole `/proc` tier is Linux-only. Somewhere without procfs every lookup comes back
        empty, and reading that as "gone" would flip every healthy job to failed and tear its
        machine down -- the 2026-08-20 failure pointing the other way. Signal 0 decides."""
        monkeypatch.setattr(m, "_proc_stat_tail", lambda pid: None)
        backend = self._job(tmp_path, "jnoproc", os.getpid(), None)

        assert m.observe_supervisor_exit(backend.store, "jnoproc") is None
        assert "runner_exit" not in backend.store.read_runtime("jnoproc")


class TestStatusSurfacesTheCause:
    """`_runtime.json` is durable but nothing reads it. The manifest's `end_reason` is what
    `lab status` prints, so the cause has to land there -- otherwise the evidence exists and
    no operator ever sees it."""

    def _fake_sky(self, monkeypatch):
        fake = types.ModuleType("sky")
        fake.get = lambda x: x  # type: ignore[attr-defined]
        fake.down = lambda cluster: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sky", fake)
        monkeypatch.setattr(m, "tear_down_and_record", lambda *a, **k: True)

    def test_a_sigkilled_supervisor_names_sigkill_in_end_reason(self, tmp_path, monkeypatch):
        self._fake_sky(monkeypatch)
        backend, proc = _registered_job(tmp_path)
        os.kill(proc.pid, signal.SIGKILL)
        _await_exit(proc)

        state = backend.status("jexit")

        assert state is JobState.failed
        reason = backend.store.read_manifest("jexit").end_reason
        assert reason is not None
        assert "supervisor exited without recording status" in reason
        assert "SIGKILL" in reason

    def test_an_unreaped_supervisor_does_not_read_as_alive(self, tmp_path, monkeypatch):
        """A killed-but-unreaped supervisor must not read as alive, by either route.

        A zombie answers `os.kill(pid, 0)` happily and keeps its start-time, so both of
        `pid_alive`'s original checks called it alive -- for as long as its parent declined to
        reap it, which under a long-lived MCP server is indefinitely, and the dead-supervisor
        teardown never fired while the box billed.

        That is now closed at the root: `pid_alive` consults the process state and reads `Z` as
        dead. `status()` reaching the same verdict is asserted alongside it, because it must hold
        even where the exit record is what supplies the answer.
        """
        self._fake_sky(monkeypatch)
        backend, proc = _registered_job(tmp_path)
        os.kill(proc.pid, signal.SIGKILL)
        _await_exit(proc)
        rt = backend.store.read_runtime("jexit")
        assert m.pid_alive(rt["runner_pid"], start_time=rt["runner_start_time"]) is False

        assert backend.status("jexit") is JobState.failed

    def test_a_cancelled_job_keeps_its_own_wording(self, tmp_path, monkeypatch):
        """A cancel sends SIGTERM itself; reporting that back as the cause of death would dress a
        deliberate act up as a supervisor failure."""
        self._fake_sky(monkeypatch)
        backend, proc = _registered_job(tmp_path)
        backend.store.write_runtime("jexit", cancelling=True)
        os.kill(proc.pid, signal.SIGTERM)
        _await_exit(proc)

        assert backend.status("jexit") is JobState.cancelled
        reason = backend.store.read_manifest("jexit").end_reason
        assert reason is not None and reason.startswith("cancelled by user")
