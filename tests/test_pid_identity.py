"""A recycled PID must not keep a dead supervisor alive forever (F4).

``pid_alive`` was a bare ``os.kill(pid, 0)``: it answers "does *some* process hold this number",
never "is it still the process we spawned". A supervisor's PID is freed quickly -- its parent, the
short-lived ``lab submit`` CLI, exits almost immediately, so the child reparents to init and is
reaped there -- and on a busy machine that number gets recycled. When it does, every liveness
check for that job reports "alive" **for good**, and the self-heal paths that depend on it never
fire: ``SkyPilotBackend.status()``'s dead-supervisor teardown, reconcile's ``unsupervised`` pass,
and the scheduler watchdog all quietly stop working for that job while its machine bills.

The fix records process *identity*, not just its number: the start-time from ``/proc/<pid>/stat``
field 22, which the kernel guarantees differs between a process and any later reuser of its PID.

Identity is only compared when it was recorded. A runtime file written by an older release has no
start-time, and there "unknown" must read as *alive* -- the conservative direction, since calling
a live supervisor dead would let reconcile destroy the machine out from under a running job, which
is precisely the 2026-08-20 failure pointing the other way.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lab._util import pid_alive, process_start_time


class TestProcessStartTime:
    def test_it_reads_a_real_running_process(self):
        assert process_start_time(os.getpid()) is not None

    def test_a_pid_that_does_not_exist_has_no_start_time(self):
        assert process_start_time(999999999) is None

    def test_it_never_raises_on_a_nonsense_pid(self):
        """Liveness feeds leak detection; a probe that throws is worse than one that shrugs."""
        for pid in (0, -1, None):
            assert process_start_time(pid) is None  # type: ignore[arg-type]

    def test_the_same_process_reads_the_same_value_twice(self):
        assert process_start_time(os.getpid()) == process_start_time(os.getpid())

    def test_two_live_processes_are_distinguishable(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert process_start_time(proc.pid) is not None
        finally:
            proc.kill()
            proc.wait()


class TestPidAliveWithIdentity:
    def test_a_live_process_with_a_matching_start_time_is_alive(self):
        assert pid_alive(os.getpid(), start_time=process_start_time(os.getpid())) is True

    def test_a_recycled_pid_reads_as_dead(self):
        """The blind spot, directly: the number is held, but by somebody else.

        Simulated by pointing the recorded identity at a value the live process cannot have --
        exactly what a PID-reuse victim's runtime file looks like after the reuse.
        """
        recorded = process_start_time(os.getpid())
        assert recorded is not None

        assert pid_alive(os.getpid(), start_time=recorded + 1_000_000) is False

    def test_a_dead_pid_is_dead_whatever_identity_was_recorded(self):
        assert pid_alive(999999999, start_time=12345) is False

    def test_no_recorded_identity_falls_back_to_the_old_behaviour(self):
        """Legacy runtime files have no start-time. Unknown must mean *alive*, not dead."""
        assert pid_alive(os.getpid()) is True
        assert pid_alive(os.getpid(), start_time=None) is True
        assert pid_alive(999999999) is False

    def test_an_unreadable_identity_does_not_kill_a_live_process(self):
        """If /proc is unavailable the check must degrade to "alive", never to "dead"."""
        import lab._util as util

        original = util.process_start_time
        try:
            util.process_start_time = lambda pid: None  # type: ignore[assignment]
            assert util.pid_alive(os.getpid(), start_time=999) is True
        finally:
            util.process_start_time = original  # type: ignore[assignment]


class TestSpawnRecordsIdentity:
    """Recording the identity at spawn is what makes the comparison possible at all.

    Asserted as an invariant on the write rather than by driving a whole `submit`: any code path
    that records a `runner_pid` without its `runner_start_time` re-opens the blind spot, and there
    is more than one such path (the local backend and the skypilot backend).
    """

    def test_every_runner_pid_write_carries_its_identity(self):
        """Scans the whole tree, not a hand-listed pair of modules.

        The first version of this test was parametrised over `lab.backends.local` and
        `lab.backends.skypilot` — the two spawn sites that existed when it was written — and
        therefore said nothing about `lab/scheduler/tick.py`, which respawns a supervisor on the
        adopt path and did *not* record the identity. That omission is worse than the original
        blind spot it was meant to guard: `write_runtime` **merges**, so the previous supervisor's
        `runner_start_time` survived beside the new pid, `pid_alive` compared them, mismatched,
        and reported a freshly respawned supervisor dead on the very next check — which tears down
        its cluster and fails a live job. Enumerating the tree is the only version of this test
        that can catch the next such site.
        """
        import ast

        offenders = []
        for path in sorted(Path("src/lab").rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "write_runtime"):
                    continue
                kwargs = {kw.arg for kw in node.keywords}
                if "runner_pid" in kwargs and "runner_start_time" not in kwargs:
                    offenders.append(f"{path}:{node.lineno}")
        assert not offenders, (
            f"runner_pid recorded without runner_start_time at {offenders} — write_runtime merges, "
            "so a stale identity survives beside the new pid and pid_alive reports the live "
            "supervisor dead (F4)"
        )


class TestReconcileRemediatesUnsupervised:
    """Detection without remediation is where F3 actually sits.

    Dry-run reconcile already *reports* `unsupervised` -- non-terminal jobs whose supervisor is
    gone. But flipping such a job to terminal and attempting its teardown still only happened if
    somebody ran `lab status` on that exact job id. On an unattended box nobody does, so a dead
    supervisor's machine sits unnoticed. `--apply` should finish the job the same way `status()`
    would, since it is already the destructive, opted-in path.
    """

    def test_apply_tears_down_and_finalises_an_unsupervised_job(self, tmp_path, monkeypatch):
        from datetime import timedelta

        from helpers import make_manifest

        from lab._util import now
        from lab.backends.local import LocalBackend
        from lab.backends.skypilot import cluster_name_for
        from lab.core import Lab
        from lab.models import BackendInfo, JobState

        lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
        m = make_manifest("jdead", "python x.py", timeout="1h").model_copy(
            update={
                "status": JobState.running,
                "started_at": now() - timedelta(seconds=3600),
                "backend": BackendInfo(provisioner="skypilot"),
            }
        )
        lab.store.create(m)
        lab.store.write_runtime("jdead", runner_pid=999999999, cluster=cluster_name_for("jdead"))

        monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
        monkeypatch.setattr("lab.core.attribute_jobs", lambda ids: {})
        monkeypatch.setattr("lab.core.local_project", lambda: "laboratory")
        _patch_empty_sky(monkeypatch)
        torn: list[str] = []
        monkeypatch.setattr(
            "lab.backends.skypilot.tear_down_and_record",
            lambda sky_mod, cluster, store, job_id, cloud, **kw: torn.append(job_id) or True,
        )

        report = lab.reconcile(apply=True)

        assert [u["job_id"] for u in report["unsupervised"]] == ["jdead"]
        assert torn == ["jdead"], "an unattended box has nobody to run `lab status <id>`"
        assert lab.store.read_manifest("jdead").status in (JobState.failed, JobState.cancelled)

    def test_a_dry_run_remediates_nothing(self, tmp_path, monkeypatch):
        """Dry-run must stay read-only; it is the mode people run to *look*."""
        from datetime import timedelta

        from helpers import make_manifest

        from lab._util import now
        from lab.backends.local import LocalBackend
        from lab.backends.skypilot import cluster_name_for
        from lab.core import Lab
        from lab.models import BackendInfo, JobState

        lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
        m = make_manifest("jdead2", "python x.py", timeout="1h").model_copy(
            update={
                "status": JobState.running,
                "started_at": now() - timedelta(seconds=3600),
                "backend": BackendInfo(provisioner="skypilot"),
            }
        )
        lab.store.create(m)
        lab.store.write_runtime("jdead2", runner_pid=999999999, cluster=cluster_name_for("jdead2"))

        monkeypatch.setattr("lab.backends.skypilot.list_vast_instances", lambda *a, **k: [])
        monkeypatch.setattr("lab.backends.skypilot.list_do_volumes", lambda *a, **k: [])
        monkeypatch.setattr("lab.core.attribute_jobs", lambda ids: {})
        monkeypatch.setattr("lab.core.local_project", lambda: "laboratory")
        _patch_empty_sky(monkeypatch)
        monkeypatch.setattr(
            "lab.backends.skypilot.tear_down_and_record",
            lambda *a, **k: pytest.fail("a dry run must not tear anything down"),
        )

        report = lab.reconcile(apply=False)

        assert [u["job_id"] for u in report["unsupervised"]] == ["jdead2"]
        assert lab.store.read_manifest("jdead2").status is JobState.running


def _patch_empty_sky(monkeypatch):
    import sys as _sys
    import types

    fake = types.ModuleType("sky")
    fake.get = lambda x: x
    fake.status = lambda refresh=None: []
    fake.down = lambda cluster: None
    fake.StatusRefreshMode = types.SimpleNamespace(AUTO="AUTO", FORCE="FORCE", NONE="NONE")
    monkeypatch.setitem(_sys.modules, "sky", fake)


def test_every_liveness_check_passes_the_recorded_identity():
    """A `pid_alive(pid)` that forgets `start_time=` silently keeps the blind spot open.

    The bug is invisible at the call site -- the code reads fine and the tests pass, because a
    recycled PID is rare and non-deterministic. So the invariant is enforced structurally: every
    caller in `src/lab` must pass the identity it recorded. `_util.py` itself is exempt (it
    defines the function and its own default).
    """
    import ast

    src = Path("src/lab")
    offenders = []
    for path in sorted(src.rglob("*.py")):
        if path.name == "_util.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "pid_alive":
                continue
            if not any(kw.arg == "start_time" for kw in node.keywords):
                offenders.append(f"{path}:{node.lineno}")
    assert not offenders, (
        f"pid_alive called without start_time= at {offenders} — a recycled PID will report these "
        "as alive forever, disabling the self-heal that depends on them (F4)"
    )


class TestZombiesAreNotAlive:
    """An exited-but-unreaped process must not read as a live supervisor.

    `os.kill(pid, 0)` succeeds for a zombie and its start-time is unchanged, so both of the checks
    above call it alive -- for as long as its parent declines to reap it. Under a long-lived parent
    (the MCP server, `lab sweep`, the scheduler tick) that is indefinitely, and every self-heal
    keyed on supervisor liveness quietly stops firing while the machine bills. This is the same
    defect F4 is about, reached by a different route.
    """

    def test_an_unreaped_child_is_not_alive(self):
        proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
        try:
            deadline = time.time() + 10
            while time.time() < deadline and proc.poll() is None:
                time.sleep(0.02)
            # Deliberately not reaped: `proc.poll()` above only checks, and the corpse is ours.
            recorded = process_start_time(proc.pid)

            assert pid_alive(proc.pid, start_time=recorded) is False
            assert pid_alive(proc.pid) is False
        finally:
            proc.wait()

    def test_a_live_child_is_still_alive(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert pid_alive(proc.pid, start_time=process_start_time(proc.pid)) is True
        finally:
            proc.kill()
            proc.wait()

    def test_an_unreadable_proc_entry_does_not_read_as_dead(self, monkeypatch):
        """No /proc must never mean "everything is dead" -- that would destroy live machines."""
        import lab._util as util

        monkeypatch.setattr(util, "_is_zombie", lambda pid: False)
        assert util.pid_alive(os.getpid()) is True
