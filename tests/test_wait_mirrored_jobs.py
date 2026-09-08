"""`lab wait` on a scheduler-launched (deferred) job — the gap behind a 98,654-call polling storm.

A five-day campaign made 98,654 `lab status` CLI calls (a flat ~1,050/hour) because ~40
hand-rolled shell loops polled `lab status <job_id>` every two minutes: `lab wait` could not
watch a scheduler-launched job at all, so a loop around `lab status` was the only way to learn
when one finished. Deferred jobs never get a record in this project's local `runs/` — the
scheduler launches them from its own host and only mirrors their manifests into the queue
(spec §4.3) — and `lab wait` read the local store exclusively, so it rejected those ids up
front ("unknown job id(s)") even though `lab status` on the same id answered fine. The storm
also blew the event ledger past its size cap and destroyed a day of forensic records.

`Lab._resolve_manifest` is the fix: local store first, queue mirror second, the same fallback
`job_status_view` has always given `lab status`. It is used by the whole wait path (`wait`,
`wait_summary`, `_settle_teardown`) and nothing else — `status`/`cancel`/`fetch_artifacts` stay
local-only on purpose (see `cli.cancel`'s docstring and the 2026-08-20 attribution incident).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from helpers import make_manifest
from typer.testing import CliRunner

import lab.core as core_mod
from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import BackendInfo, JobManifest, JobState
from lab.scheduler.queue import LocalQueueStore
from lab.store import JobStore

runner = CliRunner()


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def _lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    home = tmp_path / "runs"
    return Lab(backend=LocalBackend(home=home, repo=repo), repo=repo, home=home)


def _manifest(
    job_id: str, state: JobState, *, provisioner: str = "local", teardown: str | None = None
) -> JobManifest:
    return make_manifest(job_id, "python x.py").model_copy(
        update={
            "status": state,
            "backend": BackendInfo(provisioner=provisioner),
            "teardown_status": teardown,
        }
    )


def _mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *manifests: JobManifest) -> Path:
    """Mirror manifests into a real `LocalQueueStore` and make it the default queue — the exact
    shape of a job the scheduler launched on another machine: mirrored, never in local runs/."""
    qdir = tmp_path / "queue"
    q = LocalQueueStore(qdir)
    for m in manifests:
        q.mirror_manifest(m)
    monkeypatch.setenv("LAB_QUEUE_DIR", str(qdir))
    return qdir


class _ScriptedQueue:
    """A QueueStore double whose `read_mirrored` walks a per-job script, so a mirrored job can be
    observed *changing* (running -> succeeded) across polls the way the real mirror does on the
    scheduler's next tick. The last entry repeats forever; an unscripted id reads as absent."""

    def __init__(self, script: dict[str, list[JobManifest]]) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.reads: list[str] = []

    def read_mirrored(self, job_id: str) -> JobManifest | None:
        self.reads.append(job_id)
        seq = self.script.get(job_id)
        if not seq:
            return None
        return seq.pop(0) if len(seq) > 1 else seq[0]


def _use_queue(monkeypatch: pytest.MonkeyPatch, queue: Any) -> None:
    monkeypatch.setattr("lab.scheduler.queue.default_queue", lambda: queue)


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _isolate_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI's repo (and therefore `runs/`) at a scratch tree, and chdir out of this
    checkout so `_warn_if_repo_override_shadows_cwd` stays quiet (it would prepend a stderr
    warning to the JSON payload). Mirrors `tests/test_mirrored_job_commands.py::_isolate`."""
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_REPO_DIR", str(repo))
    return repo / "runs"


# ---------------------------------------------------------------------------
# Lab.wait / wait_summary on a mirror-only job
# ---------------------------------------------------------------------------


class TestWaitResolvesAMirrorOnlyJob:
    def test_wait_reaches_terminal_with_no_local_runs_record(self, tmp_path, monkeypatch):
        """The whole point: no local manifest exists, and `wait` still returns the real one."""
        lab = _lab(tmp_path)
        _mirror(tmp_path, monkeypatch, _manifest("jmir", JobState.succeeded))

        (m,) = lab.wait(["jmir"], interval=0.01, timeout=5)

        assert m.job_id == "jmir"
        assert m.status is JobState.succeeded
        # Never seeded into the real store: `cancel` and reconcile's `unsupervised` pass both
        # rely on "in runs/" meaning "this machine supervises it" (2026-08-20 incident).
        assert not JobStore(tmp_path / "runs").manifest_path("jmir").exists()

    def test_wait_polls_the_mirror_until_the_job_goes_terminal(self, tmp_path, monkeypatch):
        """A deferred job is normally still running when the wait starts; the mirror only turns
        terminal on a later scheduler tick."""
        lab = _lab(tmp_path)
        queue = _ScriptedQueue(
            {"jmir": [_manifest("jmir", JobState.running), _manifest("jmir", JobState.succeeded)]}
        )
        _use_queue(monkeypatch, queue)
        _record_sleeps(monkeypatch)

        (m,) = lab.wait(["jmir"], interval=0.01, timeout=5)

        assert m.status is JobState.succeeded
        assert queue.reads.count("jmir") >= 2  # polled again after the non-terminal read

    def test_on_terminal_fires_once_for_a_mirrored_job(self, tmp_path, monkeypatch):
        lab = _lab(tmp_path)
        queue = _ScriptedQueue(
            {"jmir": [_manifest("jmir", JobState.running), _manifest("jmir", JobState.failed)]}
        )
        _use_queue(monkeypatch, queue)
        _record_sleeps(monkeypatch)
        seen: list[str] = []

        lab.wait(["jmir"], interval=0.01, timeout=5, on_terminal=lambda m: seen.append(m.job_id))

        assert seen == ["jmir"]

    def test_fail_fast_trips_on_a_mirrored_failure(self, tmp_path, monkeypatch):
        lab = _lab(tmp_path)
        _mirror(
            tmp_path,
            monkeypatch,
            _manifest("jm1", JobState.failed, provisioner="skypilot", teardown="succeeded"),
            _manifest("jm2", JobState.running, provisioner="skypilot"),
        )
        _record_sleeps(monkeypatch)

        summary = lab.wait_summary(["jm2", "jm1"], interval=0.01, timeout=5, fail_fast=True)

        assert summary["failed_fast"] is True
        assert summary["pending"] == ["jm2"]  # still running — and still billing
        assert summary["jobs"][0]["job_id"] == "jm1"  # offender first

    def test_wait_summary_classifies_a_mirrored_teardown_leak(self, tmp_path, monkeypatch):
        """The FR-C2 money alarm has to work for deferred jobs too — they are the ones nobody is
        watching."""
        lab = _lab(tmp_path)
        _mirror(
            tmp_path,
            monkeypatch,
            _manifest("jmir", JobState.failed, provisioner="skypilot", teardown="failed"),
        )
        _record_sleeps(monkeypatch)

        summary = lab.wait_summary(["jmir"], interval=0.01, timeout=5)

        assert summary["all_terminal"] is True
        assert summary["teardown_leaks"] == ["jmir"]

    def test_a_lagging_mirrored_teardown_reads_as_unconfirmed_not_clean(
        self, tmp_path, monkeypatch
    ):
        """Mirror lag makes `teardown_status` read null for a tick or two after success. A null
        must stay "unconfirmed" (warn, exit 0), never be promoted to clean."""
        lab = _lab(tmp_path)
        _mirror(
            tmp_path, monkeypatch, _manifest("jmir", JobState.succeeded, provisioner="skypilot")
        )
        _record_sleeps(monkeypatch)

        summary = lab.wait_summary(["jmir"], interval=0.01, timeout=5)

        assert summary["teardown_unconfirmed"] == ["jmir"]
        assert summary["teardown_leaks"] == [] and summary["teardown_unknown"] == []

    def test_settle_teardown_re_reads_the_mirror(self, tmp_path, monkeypatch):
        """`_settle_teardown` used to call `self.manifest` directly, which raises for a
        mirror-only job — so the settle pass had to go through the resolver as well."""
        lab = _lab(tmp_path)
        lagging = _manifest("jmir", JobState.succeeded, provisioner="skypilot")
        settled = _manifest("jmir", JobState.succeeded, provisioner="skypilot", teardown="failed")
        _use_queue(monkeypatch, _ScriptedQueue({"jmir": [lagging, lagging, settled]}))
        _record_sleeps(monkeypatch)

        summary = lab.wait_summary(["jmir"], interval=0.01, timeout=5)

        assert summary["teardown_leaks"] == ["jmir"]  # settled to its real value
        assert summary["teardown_unconfirmed"] == []


def test_a_mirrored_job_never_reaches_the_live_backend(tmp_path, monkeypatch):
    """Cross-machine safety, structurally: this machine does not supervise a scheduler-launched
    job, and SkyPilot client/server skew (`lab._skycompat`) makes asking a live backend about one
    a real correctness risk — the reason `cancel` refuses mirror-only jobs outright. The resolver
    reads the local store *first* so the backend is never consulted for a mirrored id. (Ordering
    it the other way happens to work only because both real backends read the manifest on their
    first line; `Backend` promises no such thing.)"""

    class _RefusingBackend:
        name = "refusing"

        def status(self, job_id: str) -> JobState:
            raise AssertionError(f"live backend consulted for mirror-only job {job_id!r}")

    lab = Lab(backend=_RefusingBackend(), repo=repo_root(Path.cwd()), home=tmp_path / "runs")
    _mirror(tmp_path, monkeypatch, _manifest("jmir", JobState.succeeded, provisioner="skypilot"))
    _record_sleeps(monkeypatch)

    (m,) = lab.wait(["jmir"], interval=0.01, timeout=5)

    assert m.status is JobState.succeeded


class TestMirroredJobsAreNamedInTheSummary:
    def test_the_summary_names_the_mirrored_job_ids(self, tmp_path, monkeypatch):
        """A watcher reading the summary (or the --done-file) has to be able to tell that this
        data came from the mirror and can be a scheduler tick stale."""
        lab = _lab(tmp_path)
        local = _manifest("jloc", JobState.succeeded)
        lab.store.create(local)
        _mirror(tmp_path, monkeypatch, _manifest("jmir", JobState.succeeded))
        _record_sleeps(monkeypatch)

        summary = lab.wait_summary(["jloc", "jmir"], interval=0.01, timeout=5)

        assert summary["mirrored"] == ["jmir"]

    def test_a_purely_local_wait_reports_no_mirrored_ids(self, tmp_path, monkeypatch):
        lab = _lab(tmp_path)
        lab.store.create(_manifest("jloc", JobState.succeeded))
        _mirror(tmp_path, monkeypatch)  # an empty mirror, so a stray lookup would still be safe
        _record_sleeps(monkeypatch)

        summary = lab.wait_summary(["jloc"], interval=0.01, timeout=5)

        assert summary["mirrored"] == []


# ---------------------------------------------------------------------------
# poll cadence: the mirror is only as fresh as the scheduler's tick
# ---------------------------------------------------------------------------


class TestMirroredPollCadence:
    def test_an_all_mirrored_wait_floors_the_interval(self, tmp_path, monkeypatch, capsys):
        """The mirror refreshes on the scheduler's tick (60s default), so a 10s poll costs one
        R2 GET per job to learn nothing."""
        lab = _lab(tmp_path)
        _use_queue(
            monkeypatch,
            _ScriptedQueue(
                {
                    "jmir": [
                        _manifest("jmir", JobState.running),
                        _manifest("jmir", JobState.succeeded),
                    ]
                }
            ),
        )
        sleeps = _record_sleeps(monkeypatch)

        lab.wait(["jmir"], interval=1.0, timeout=600)

        assert sleeps == [core_mod._MIRROR_MIN_INTERVAL_S]
        assert "mirror" in capsys.readouterr().err.lower()  # said so, on stderr (stdout is JSON)

    def test_a_mixed_wait_keeps_the_callers_interval(self, tmp_path, monkeypatch, capsys):
        """One local job means local polls are cheap and fresh — flooring them would slow a
        normal wait down for no reason."""
        lab = _lab(tmp_path)
        lab.store.create(_manifest("jloc", JobState.succeeded))
        _use_queue(
            monkeypatch,
            _ScriptedQueue(
                {
                    "jmir": [
                        _manifest("jmir", JobState.running),
                        _manifest("jmir", JobState.succeeded),
                    ]
                }
            ),
        )
        sleeps = _record_sleeps(monkeypatch)

        lab.wait(["jloc", "jmir"], interval=1.0, timeout=600)

        assert sleeps == [1.0]
        assert "mirror" not in capsys.readouterr().err.lower()

    def test_an_interval_already_above_the_floor_is_left_alone(self, tmp_path, monkeypatch):
        lab = _lab(tmp_path)
        _use_queue(
            monkeypatch,
            _ScriptedQueue(
                {
                    "jmir": [
                        _manifest("jmir", JobState.running),
                        _manifest("jmir", JobState.succeeded),
                    ]
                }
            ),
        )
        sleeps = _record_sleeps(monkeypatch)

        lab.wait(["jmir"], interval=90.0, timeout=600)

        assert sleeps == [90.0]


# ---------------------------------------------------------------------------
# degraded / absent mirrors
# ---------------------------------------------------------------------------


class TestUnresolvableJobs:
    def test_a_version_skewed_mirrored_manifest_reads_as_not_found(self, tmp_path, monkeypatch):
        """`read_mirrored` answers `None` for a partial/stub manifest (a version-skewed scheduler
        host, or a read racing an in-progress write). That is "not found", not a crash — and it
        must not read as a job either, or `wait` would blow up on `.status`."""
        lab = _lab(tmp_path)
        qdir = tmp_path / "queue"
        (qdir / "jobs").mkdir(parents=True)
        (qdir / "jobs" / "jskew.json").write_text(json.dumps({"job_id": "jskew"}))
        monkeypatch.setenv("LAB_QUEUE_DIR", str(qdir))

        with pytest.raises(FileNotFoundError):
            lab.wait(["jskew"], interval=0.01, timeout=5)

    def test_a_job_in_neither_place_raises_file_not_found(self, tmp_path, monkeypatch):
        lab = _lab(tmp_path)
        _mirror(tmp_path, monkeypatch)  # empty mirror

        with pytest.raises(FileNotFoundError):
            lab.wait(["nope"], interval=0.01, timeout=5)


# ---------------------------------------------------------------------------
# the local path must not regress: a dead supervisor still finalizes
# ---------------------------------------------------------------------------


def test_wait_still_finalizes_a_local_job_whose_supervisor_died(tmp_path, monkeypatch):
    """The poll loop cannot become a plain manifest read. `backend.status` is what notices a
    supervisor that died without recording terminal state and *writes* the terminal manifest;
    without that call a dead-supervisor job stays `running` forever and every wait on it burns
    its full timeout (~40% of DO supervisors died silently in the 2026-08 campaign)."""
    lab = _lab(tmp_path)
    lab.store.create(_manifest("jdead", JobState.running))
    lab.store.write_runtime("jdead", runner_pid=999999999)  # a pid that cannot exist
    _record_sleeps(monkeypatch)

    (m,) = lab.wait(["jdead"], interval=0.01, timeout=5)

    assert m.status is JobState.failed
    assert m.end_reason == "runner exited without recording status"


# ---------------------------------------------------------------------------
# CLI gate
# ---------------------------------------------------------------------------


class TestCliWaitGate:
    def test_a_mirror_only_job_id_is_accepted(self, tmp_path, monkeypatch):
        _isolate_cli(tmp_path, monkeypatch)
        _mirror(tmp_path, monkeypatch, _manifest("jmir", JobState.succeeded))

        result = runner.invoke(app, ["wait", "jmir", "--timeout", "10"])

        assert result.exit_code == 0, result.output
        summary = json.loads(result.stdout)
        assert summary["all_terminal"] is True
        assert summary["mirrored"] == ["jmir"]
        assert summary["jobs"][0]["state"] == "succeeded"

    def test_a_mirror_only_job_id_is_never_seeded_into_local_runs(self, tmp_path, monkeypatch):
        home = _isolate_cli(tmp_path, monkeypatch)
        _mirror(tmp_path, monkeypatch, _manifest("jmir", JobState.succeeded))

        assert runner.invoke(app, ["wait", "jmir", "--timeout", "10"]).exit_code == 0
        assert not JobStore(home).manifest_path("jmir").exists()

    def test_a_mirrored_skypilot_job_resolves_without_a_local_manifest(
        self, tmp_path, monkeypatch
    ):
        """The realistic shape: the scheduler launches on a cloud, so the mirrored manifest's
        provisioner is `skypilot`. `_lab_for_wait` has to build that backend from the *mirrored*
        manifest (`_lab_for` raises for a mirror-only id) while keeping the real `runs/` as its
        home — no cloud call happens, because the resolver reads the local store first and the
        job is already terminal."""
        _isolate_cli(tmp_path, monkeypatch)
        _mirror(
            tmp_path,
            monkeypatch,
            _manifest("jmir", JobState.succeeded, provisioner="skypilot", teardown="succeeded"),
        )

        result = runner.invoke(app, ["wait", "jmir", "--timeout", "10"])

        assert result.exit_code == 0, result.output
        summary = json.loads(result.stdout)
        assert summary["mirrored"] == ["jmir"] and summary["teardown_unconfirmed"] == []

    def test_a_mirrored_teardown_leak_still_exits_3(self, tmp_path, monkeypatch):
        """The money alarm has to survive the new path: a deferred job is the one nobody is
        watching, so exit 3 here is the whole point of being able to wait on it at all."""
        _isolate_cli(tmp_path, monkeypatch)
        _mirror(
            tmp_path,
            monkeypatch,
            _manifest("jmir", JobState.failed, provisioner="skypilot", teardown="failed"),
        )

        result = runner.invoke(app, ["wait", "jmir", "--timeout", "10"])

        assert result.exit_code == 3, result.output

    def test_an_unknown_job_id_still_exits_2(self, tmp_path, monkeypatch):
        _isolate_cli(tmp_path, monkeypatch)
        _mirror(tmp_path, monkeypatch)  # nothing mirrored either

        result = runner.invoke(app, ["wait", "nope", "--timeout", "10"])

        assert result.exit_code == 2
        assert "unknown job id(s)" in result.output
        assert "nope" in result.output

    def test_a_mirror_that_cannot_answer_exits_2_rather_than_crashing(self, tmp_path, monkeypatch):
        """`read_mirrored` can still raise outright on some paths (a queue-store I/O failure).
        The guard must degrade to the same structured exit 2 (`_read_mirrored`'s documented
        degradation), not accept the id and then blow up mid-wait, and not print a traceback."""
        _isolate_cli(tmp_path, monkeypatch)

        class _RaisingQueue:
            def read_mirrored(self, job_id: str) -> JobManifest | None:
                raise ValueError("queue store timeout")

        _use_queue(monkeypatch, _RaisingQueue())

        result = runner.invoke(app, ["wait", "jmir", "--timeout", "10"])

        assert result.exit_code == 2
        assert "unknown job id(s)" in result.output
        assert "Traceback" not in result.output

    def test_the_done_file_carries_the_mirrored_ids(self, tmp_path, monkeypatch):
        """A watcher reading the done-file snapshot has to see that the data is mirror-sourced."""
        _isolate_cli(tmp_path, monkeypatch)
        _mirror(tmp_path, monkeypatch, _manifest("jmir", JobState.succeeded))
        done = tmp_path / "done.json"

        result = runner.invoke(
            app, ["wait", "jmir", "--timeout", "10", "--done-file", str(done)]
        )

        assert result.exit_code == 0, result.output
        assert json.loads(done.read_text())["mirrored"] == ["jmir"]
