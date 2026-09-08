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


class _FlakyQueue:
    """A mirror that raises on the reads named in ``raise_on`` (0-based) and otherwise walks
    ``_ScriptedQueue``'s script. Models a transient object-store failure — `R2Store.get_text`
    re-raises every non-NoSuchKey boto error, so a single 5xx or reset lands here as an
    exception mid-wait."""

    def __init__(self, script: dict[str, list[JobManifest]], raise_on: set[int]) -> None:
        self.inner = _ScriptedQueue(script)
        self.raise_on = set(raise_on)
        self.n = 0
        self.raised: list[int] = []

    def read_mirrored(self, job_id: str) -> JobManifest | None:
        i, self.n = self.n, self.n + 1
        if i in self.raise_on:
            self.raised.append(i)
            raise OSError("connection reset by peer (simulated R2 5xx)")
        return self.inner.read_mirrored(job_id)

    @property
    def reads(self) -> list[str]:
        return self.inner.reads


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


# ---------------------------------------------------------------------------
# a transient mirror read must not kill a long wait (review finding 1)
# ---------------------------------------------------------------------------


class TestATransientMirrorFailureIsNotTheEndOfTheWait:
    """`R2Store.get_text` re-raises every non-NoSuchKey boto error and `default_queue()` can fail
    on its own, so one 5xx / connection reset on one poll used to unwind `wait_summary`. The CLI's
    generic handler turns that into **exit 1** — the documented code for "gave up on --timeout" —
    so a six-hour wait ended indistinguishable from a real timeout because of a network hiccup.
    A poll that cannot read the mirror means "no news yet"; the caller's own --timeout is the
    bound, and no retry budget is invented."""

    def test_a_mirror_error_mid_wait_is_survived_and_the_verdict_is_still_right(
        self, tmp_path, monkeypatch, capsys
    ):
        lab = _lab(tmp_path)
        queue = _FlakyQueue(
            {
                "jmir": [
                    _manifest("jmir", JobState.running),
                    _manifest("jmir", JobState.succeeded),
                ]
            },
            raise_on={1},  # not the first read: the wait is already six hours old
        )
        _use_queue(monkeypatch, queue)
        _record_sleeps(monkeypatch)

        (m,) = lab.wait(["jmir"], interval=0.01, timeout=600)

        assert queue.raised == [1]  # the failure really happened mid-poll
        assert m.status is JobState.succeeded  # the real verdict, not a crash
        assert "mirror could not be read" in capsys.readouterr().err  # said so, on stderr

    def test_a_mirror_that_briefly_reads_as_absent_keeps_waiting(self, tmp_path, monkeypatch):
        """A version-skewed manifest written by a mid-wait scheduler redeploy makes
        `read_mirrored` answer `None`, which the resolver turns into a bare FileNotFoundError.
        Mid-wait that is still just "not yet available"."""

        class _BlinkingQueue:
            def __init__(self) -> None:
                self.n = 0

            def read_mirrored(self, job_id: str) -> JobManifest | None:
                self.n += 1
                if self.n == 2:
                    return None  # partial/stub manifest, e.g. a redeployed scheduler host
                if self.n < 4:
                    return _manifest("jmir", JobState.running)
                return _manifest("jmir", JobState.succeeded)

        _use_queue(monkeypatch, _BlinkingQueue())
        _record_sleeps(monkeypatch)

        (m,) = _lab(tmp_path).wait(["jmir"], interval=0.01, timeout=600)

        assert m.status is JobState.succeeded

    def test_an_unknown_id_still_fails_on_the_first_poll(self, tmp_path, monkeypatch):
        """The degradation must not swallow the "this id exists nowhere" signal: it has to fail
        immediately (never sleep its way to the deadline), which is what keeps the CLI gate's
        exit 2 honest."""
        lab = _lab(tmp_path)
        _mirror(tmp_path, monkeypatch)  # empty mirror
        sleeps = _record_sleeps(monkeypatch)

        with pytest.raises(FileNotFoundError):
            lab.wait(["nope"], interval=0.01, timeout=600)

        assert sleeps == []

    def test_the_cli_does_not_report_a_mirror_blip_as_a_timeout(self, tmp_path, monkeypatch):
        """The end-to-end shape of the bug: exit 1 with no way to tell a network hiccup from a
        real `--timeout`."""
        _isolate_cli(tmp_path, monkeypatch)
        running = _manifest("jmir", JobState.running)
        # Reads 0 and 1 are the CLI's own pre-wait pair (`_mirror_knows`, then `_lab_for_wait`
        # learning which backend to build); read 2 is the wait's first poll. So read 3 is the
        # first one that lands *inside* the poll loop, hours in.
        queue = _FlakyQueue(
            {"jmir": [running, running, running, _manifest("jmir", JobState.succeeded)]},
            raise_on={3},
        )
        _use_queue(monkeypatch, queue)
        _record_sleeps(monkeypatch)

        result = runner.invoke(app, ["wait", "jmir", "--interval", "1", "--timeout", "8h"])

        assert queue.raised == [3]
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["all_terminal"] is True


# ---------------------------------------------------------------------------
# the teardown settle has to outlast a scheduler tick for mirrored jobs (finding 5)
# ---------------------------------------------------------------------------


class _TickingMirror:
    """A mirror that publishes `teardown_status` only once a scheduler tick has passed — what the
    real one does. `tick.py::_sync` mirrors the terminal manifest, and the supervisor's later
    `teardown_status` only reaches the mirror on a *following* tick (its `RE_MIRROR_TERMINAL_S`
    grace window)."""

    def __init__(self, lagging: JobManifest, settled: JobManifest, *, clock, tick_s=60.0) -> None:
        self.lagging, self.settled, self.clock, self.tick_s = lagging, settled, clock, tick_s

    def read_mirrored(self, job_id: str) -> JobManifest | None:
        return self.settled if self.clock() >= self.tick_s else self.lagging


def _ticking(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    _use_queue(
        monkeypatch,
        _TickingMirror(
            _manifest("jmir", JobState.succeeded, provisioner="skypilot"),
            _manifest("jmir", JobState.succeeded, provisioner="skypilot", teardown="succeeded"),
            clock=lambda: sum(sleeps),
        ),
    )


class TestTheSettleSpansASchedulerTickForMirroredJobs:
    """The settle window was 3 x min(interval, 5) = 15s and explicitly exempt from the mirrored
    poll floor. A mirrored manifest cannot refresh faster than the scheduler's 60s tick, so every
    clean deferred wait ended with `teardown_status: None` -> `teardown_unconfirmed` -> the "run
    `lab reconcile` to be sure no machine is still billing" warning. That is the money alarm
    firing on every clean run: the R10 failure mode that gets an alarm ignored."""

    def test_a_mirrored_teardown_settles_instead_of_reading_unconfirmed(
        self, tmp_path, monkeypatch
    ):
        lab = _lab(tmp_path)
        sleeps = _record_sleeps(monkeypatch)
        _ticking(monkeypatch, sleeps)

        summary = lab.wait_summary(["jmir"], interval=10.0, timeout=600)

        assert summary["teardown_unconfirmed"] == []
        assert summary["jobs"][0]["teardown_status"] == "succeeded"
        assert sum(sleeps) >= 60.0  # the old 15s window could not have spanned the tick

    def test_a_clean_deferred_wait_prints_no_teardown_warning(self, tmp_path, monkeypatch):
        """The user-visible half: exit 0 *and* silence. A warning here is the one nobody would
        ever act on again."""
        _isolate_cli(tmp_path, monkeypatch)
        sleeps = _record_sleeps(monkeypatch)
        _ticking(monkeypatch, sleeps)

        result = runner.invoke(app, ["wait", "jmir", "--timeout", "600"])

        assert result.exit_code == 0, result.output
        assert "teardown not confirmed" not in result.stderr

    def test_a_mirrored_teardown_that_never_lands_is_still_unconfirmed(self, tmp_path, monkeypatch):
        """Do not suppress the alarm — a real leak still has to surface, just after a window long
        enough to make it mean something."""
        lab = _lab(tmp_path)
        sleeps = _record_sleeps(monkeypatch)
        _use_queue(
            monkeypatch,
            _ScriptedQueue({"jmir": [_manifest("jmir", JobState.succeeded, provisioner="skypilot")]}),
        )

        summary = lab.wait_summary(["jmir"], interval=10.0, timeout=600)

        assert summary["teardown_unconfirmed"] == ["jmir"]
        assert sum(sleeps) == core_mod._MIRROR_SETTLE_S  # bounded, not a second poll loop

    def test_a_purely_local_wait_keeps_the_fast_settle(self, tmp_path, monkeypatch):
        """A local job's teardown_status is written to the very file being re-read, seconds after
        terminal — flooring that at a scheduler tick would add 75s to every ordinary remote
        wait's exit."""
        lab = _lab(tmp_path)
        lab.store.create(_manifest("jloc", JobState.succeeded, provisioner="skypilot"))
        sleeps = _record_sleeps(monkeypatch)
        _mirror(tmp_path, monkeypatch)  # empty: nothing here resolves from the mirror

        summary = lab.wait_summary(["jloc"], interval=10.0, timeout=600)

        assert summary["teardown_unconfirmed"] == ["jloc"]
        assert sleeps == [5.0, 5.0, 5.0]


# ---------------------------------------------------------------------------
# one queue store per Lab, not one per poll (finding 4)
# ---------------------------------------------------------------------------


class TestTheQueueStoreIsBuiltOnce:
    def test_a_polling_wait_builds_the_queue_store_once(self, tmp_path, monkeypatch):
        """`default_queue()` -> `R2QueueStore.from_env()` -> `boto3.client("s3", ...)`: botocore
        service models re-loaded and a fresh TLS connection, per mirrored job per poll. An 8h wait
        at the 30s floor built ~960 of them — in the change whose point is cutting queue read
        cost."""
        lab = _lab(tmp_path)
        running = _manifest("jmir", JobState.running)
        queue = _ScriptedQueue(
            {"jmir": [running, running, running, _manifest("jmir", JobState.succeeded)]}
        )
        builds: list[int] = []
        monkeypatch.setattr(
            "lab.scheduler.queue.default_queue", lambda: (builds.append(1), queue)[1]
        )
        _record_sleeps(monkeypatch)

        lab.wait_summary(["jmir"], interval=0.01, timeout=600)

        assert len(queue.reads) >= 4  # it really did poll several times
        assert builds == [1]  # ... through one client

    def test_a_second_lab_does_not_inherit_the_first_labs_queue(self, tmp_path, monkeypatch):
        """The cache is per-Lab, never process-global: tests swap `LAB_QUEUE_DIR` between cases
        and a leaked store would make one case answer for the next."""
        _mirror(tmp_path / "a", monkeypatch, _manifest("ja", JobState.succeeded))
        lab_a = _lab(tmp_path / "a")
        assert lab_a.wait(["ja"], interval=0.01, timeout=5)[0].job_id == "ja"

        _mirror(tmp_path / "b", monkeypatch, _manifest("jb", JobState.succeeded))
        lab_b = _lab(tmp_path / "b")

        assert lab_b.wait(["jb"], interval=0.01, timeout=5)[0].job_id == "jb"
        with pytest.raises(FileNotFoundError):  # `ja` lives only in the *first* queue
            lab_b.wait(["ja"], interval=0.01, timeout=5)


# ---------------------------------------------------------------------------
# per-job backend routing in a mixed wait (finding 6)
# ---------------------------------------------------------------------------


def test_a_mixed_wait_refreshes_each_job_through_its_own_backend(tmp_path, monkeypatch):
    """`cli._lab_for_wait(ids[0])` builds one Lab for the whole id list. Now that a mirror-only
    id is accepted, `lab wait <deferred_skypilot_job> <local_job>` builds a SkyPilot-backed Lab —
    and refreshing the *local* job through it hits `SkyPilotBackend.status`'s dead-supervisor
    branch, which tears down `cluster_name_for(<local job id>)`, a cluster that never existed,
    records `teardown_status="failed"` and exits 3. A "a paid machine may still be billing" alarm
    for a job that has no machine at all."""

    class _SkyLikeBackend:
        """Stands in for `SkyPilotBackend`: asked about a non-terminal job whose runner is gone,
        it attempts a teardown of a cluster named after the job and records the failure."""

        name = "skypilot"

        def __init__(self, home: Path) -> None:
            self.store = JobStore(home)
            self.asked: list[str] = []

        def status(self, job_id: str) -> JobState:
            self.asked.append(job_id)
            m = self.store.read_manifest(job_id)
            if m.status is JobState.running:
                return self.store.update_manifest(
                    job_id, status=JobState.failed, teardown_status="failed"
                ).status
            return m.status

    home = tmp_path / "runs"
    backend = _SkyLikeBackend(home)
    lab = Lab(backend=backend, repo=repo_root(Path.cwd()), home=home)
    lab.store.create(_manifest("jloc", JobState.running, provisioner="local"))
    lab.store.write_runtime("jloc", runner_pid=999999999)  # a pid that cannot exist
    _mirror(
        tmp_path,
        monkeypatch,
        _manifest("jmir", JobState.succeeded, provisioner="skypilot", teardown="succeeded"),
    )
    _record_sleeps(monkeypatch)

    summary = lab.wait_summary(["jmir", "jloc"], interval=0.01, timeout=600)

    assert backend.asked == []  # never asked about a job it did not run
    assert summary["teardown_leaks"] == []  # no phantom cluster, no money alarm
    # ...and the local job still got finalized, by the backend that actually ran it.
    assert summary["jobs"][1] == {
        "job_id": "jloc", "state": "failed", "exit_code": None, "teardown_status": None
    }
    assert lab.manifest("jloc").end_reason == "runner exited without recording status"


# ---------------------------------------------------------------------------
# callback failures never touch stdout (finding 7)
# ---------------------------------------------------------------------------


def _boom(_arg: Any) -> None:
    raise RuntimeError("watcher exploded")


class TestCallbackFailuresStayOffStdout:
    def test_a_failing_on_terminal_callback_prints_to_stderr(self, tmp_path, monkeypatch, capsys):
        lab = _lab(tmp_path)
        _mirror(tmp_path, monkeypatch, _manifest("jmir", JobState.succeeded))
        _record_sleeps(monkeypatch)

        lab.wait(["jmir"], interval=0.01, timeout=5, on_terminal=_boom)

        cap = capsys.readouterr()
        assert cap.out == ""  # stdout carries only the summary JSON, which callers pipe to jq
        assert "on_terminal callback failed" in cap.err

    def test_a_failing_on_update_callback_prints_to_stderr(self, tmp_path, monkeypatch, capsys):
        lab = _lab(tmp_path)
        _mirror(tmp_path, monkeypatch, _manifest("jmir", JobState.succeeded))
        _record_sleeps(monkeypatch)

        lab.wait_summary(["jmir"], interval=0.01, timeout=5, on_update=_boom)

        cap = capsys.readouterr()
        assert cap.out == ""
        assert "on_update callback failed" in cap.err

    def test_an_unwritable_done_file_leaves_the_summary_json_parseable(self, tmp_path, monkeypatch):
        """`lab wait <job> --done-file /nope/done.json | jq` — the message must not land ahead of
        the summary on stdout."""
        _isolate_cli(tmp_path, monkeypatch)
        _mirror(tmp_path, monkeypatch, _manifest("jmir", JobState.succeeded))
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")  # mkdir(parents=True) over a regular file fails

        result = runner.invoke(
            app, ["wait", "jmir", "--timeout", "10", "--done-file", str(blocker / "done.json")]
        )

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["all_terminal"] is True
        assert "callback failed" in result.stderr
