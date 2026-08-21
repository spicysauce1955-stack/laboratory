"""The partial-results safety net must actually run, and must say so when it doesn't.

Incident, 2026-08-20 (all four run dirs are still on disk under
``tempotron-capacity/runs/``): eight jobs launched at 20:00 UTC, four were killed at 07:43 after
11.7 h, and **all four have a completely empty ``output/``** — mtime unchanged from the second
the run dir was created. The experiment appends+fsyncs ``results.csv`` after every row, so ~6 h
of finished rows existed on each box and none of it came home.

Two independent defects produced that, and this module pins both.

**1. The heartbeat ran only in the window where it could not possibly work.**
``run_job`` calls ``sky.tail_logs(cluster, job, follow=True)``, which blocks until the remote job
is terminal; ``_wait_terminal`` — the only thing that ever called ``on_heartbeat`` — runs *after*
it returns. So:

* succeeded job ``20260820-200053-530f1a`` streamed for 16152 s and reached
  ``Job finished (status: SUCCEEDED)`` with **zero** ``heartbeat rsync skipped`` lines and zero
  poll errors: across 4.5 healthy hours the heartbeat fired **not once**. Its files arrived from
  the single post-wait rsync.
* the four lost jobs have their log stream ending at the experiment's *second* output line,
  immediately followed by ``ssh: connect to host … Network is unreachable``. The first heartbeat
  attempt therefore happened after connectivity was already gone, and all 46 of them failed with
  ``returned non-zero exit status 255``.

``tail_logs`` returning *is* the loss event. A fetch that only starts then is a fetch that never
runs while the box is healthy.

**2. A fetch that delivered nothing was indistinguishable from one that worked.** ``rsync``
exits 0 having copied zero files, success was silent, and failure was one unstructured line
among 1597 that nobody reads. Nothing durable recorded whether partial results were coming back.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any

import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab._util import now
from lab.models import CostInfo, JobState
from lab.store import JobStore


class _Status:
    def __init__(self, name: str) -> None:
        self.name = name


def _prep(tmp_path: Path, job_id: str) -> JobStore:
    store = JobStore(tmp_path / "runs")
    store.create(
        make_manifest(job_id, "python x.py", timeout="120s").model_copy(
            update={
                "status": JobState.running,
                "started_at": now(),
                "cost": CostInfo(hourly_usd=0.2, estimated_usd=0.2),
            }
        )
    )
    store.write_runtime(job_id, runner_pid=1, cluster=f"lab-{job_id}")
    return store


def _fake_sky(tail_logs: Any) -> types.ModuleType:
    """A ``sky`` whose queue answers SUCCEEDED at once, so the only wait is ``tail_logs``."""
    mod = types.ModuleType("sky")
    mod.get = lambda x: x  # type: ignore[attr-defined]
    mod.tail_logs = tail_logs  # type: ignore[attr-defined]
    mod.queue = lambda cluster, skip_finished=False: [  # type: ignore[attr-defined]
        {"job_id": 1, "status": _Status("SUCCEEDED")}
    ]
    return mod


def _neutralise(monkeypatch: Any) -> None:
    monkeypatch.setattr(runner_mod, "_resolve_hourly", lambda *a, **k: 0.2)
    monkeypatch.setattr(runner_mod, "r2_enabled", lambda: False)
    monkeypatch.setattr(
        runner_mod,
        "tear_down_and_record",
        lambda sky_mod, cluster, st, jid, cloud="vast", **k: True,
    )


# ---------------------------------------------------------------------------
# Defect 1: the fetch must overlap the run, not follow it
# ---------------------------------------------------------------------------


def test_partials_are_fetched_while_tail_logs_is_still_blocking(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The regression that lost the data.

    ``tail_logs`` here stands in for a healthy 4.5 h stream: it blocks until it sees a fetch.
    Before the fix nothing fetched until it returned, so this deadlocks out to the timeout with
    ``during == 0`` — exactly the 2026-08-20 shape, where the only window with both partial
    results on disk *and* a reachable box was the window the heartbeat sat out.
    """
    store = _prep(tmp_path, "hb1")
    monkeypatch.setattr(runner_mod, "HEARTBEAT_S", 0.02)

    fetched = threading.Event()
    during = 0

    def _rsync(cluster: str, remote: str, local: Path) -> runner_mod.RsyncStats:
        nonlocal during
        if not returned.is_set():
            during += 1
            fetched.set()
        local.mkdir(parents=True, exist_ok=True)
        return runner_mod.RsyncStats(files=1, bytes=64)

    returned = threading.Event()

    def _tail_logs(*a: Any, **k: Any) -> None:
        fetched.wait(timeout=10.0)
        returned.set()

    monkeypatch.setattr(runner_mod, "_rsync_down", _rsync)
    monkeypatch.setitem(sys.modules, "sky", _fake_sky(_tail_logs))
    _neutralise(monkeypatch)

    runner_mod.run_job(store.job_dir("hb1"), adopt=True)

    assert during >= 1, "no partial-results fetch happened while the job was still streaming"


def test_the_fetch_thread_does_not_outlive_the_job(tmp_path: Path, monkeypatch: Any) -> None:
    """Teardown is cost-critical (FR-C2) — a background fetcher must not still be running (nor
    still writing into ``output/``) once the supervisor has moved on to destroying the box."""
    store = _prep(tmp_path, "hb2")
    monkeypatch.setattr(runner_mod, "HEARTBEAT_S", 0.01)
    monkeypatch.setattr(
        runner_mod, "_rsync_down", lambda c, r, local: runner_mod.RsyncStats(files=0, bytes=0)
    )
    monkeypatch.setitem(sys.modules, "sky", _fake_sky(lambda *a, **k: None))
    _neutralise(monkeypatch)

    runner_mod.run_job(store.job_dir("hb2"), adopt=True)

    leaked = [t.name for t in threading.enumerate() if t.name.startswith("lab-partials-")]
    assert leaked == [], f"fetch thread outlived the job: {leaked}"


# ---------------------------------------------------------------------------
# Defect 2: a fetch that delivers nothing must not look like one that does
# ---------------------------------------------------------------------------


def test_a_fetch_that_delivers_nothing_says_so_in_the_job_record(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``rsync`` exits 0 when it successfully copies nothing. That is the quiet half of the
    incident: had the network held, an empty ``REMOTE_RUN_DIR`` would have produced the same
    empty ``output/`` with not one line of complaint anywhere."""
    store = _prep(tmp_path, "hb3")
    monkeypatch.setattr(runner_mod, "HEARTBEAT_S", 0.01)
    monkeypatch.setattr(
        runner_mod, "_rsync_down", lambda c, r, local: runner_mod.RsyncStats(files=0, bytes=0)
    )
    monkeypatch.setitem(sys.modules, "sky", _fake_sky(lambda *a, **k: None))
    _neutralise(monkeypatch)

    runner_mod.run_job(store.job_dir("hb3"), adopt=True)

    p = store.read_runtime("hb3")["partials"]
    assert p["attempts"] >= 1
    assert p["failed"] == 0  # rsync itself was fine...
    assert p["files_total"] == 0  # ...and delivered nothing
    assert p["delivered"] is False
    assert p["last_delivery_at"] is None
    assert p["last_ok_at"] is not None


def test_every_fetch_failing_is_recorded_with_the_reason(tmp_path: Path, monkeypatch: Any) -> None:
    """The live shape: 46 consecutive ``exit status 255``. The record must carry the count and
    the reason, and must be readable without parsing 1597 lines of job log."""
    store = _prep(tmp_path, "hb4")
    monkeypatch.setattr(runner_mod, "HEARTBEAT_S", 0.01)

    def _boom(cluster: str, remote: str, local: Path) -> runner_mod.RsyncStats:
        raise subprocess.CalledProcessError(
            255, ["rsync", "-az", f"{cluster}:{remote}/"], stderr="ssh: Network is unreachable\n"
        )

    monkeypatch.setattr(runner_mod, "_rsync_down", _boom)

    stalled = threading.Event()

    def _tail_logs(*a: Any, **k: Any) -> None:
        stalled.wait(timeout=10.0)

    monkeypatch.setitem(sys.modules, "sky", _fake_sky(_tail_logs))
    _neutralise(monkeypatch)

    # Let a few beats fail, then release the stream.
    def _release() -> None:
        for _ in range(500):
            if store.read_runtime("hb4").get("partials", {}).get("failed", 0) >= 2:
                break
            threading.Event().wait(0.01)
        stalled.set()

    releaser = threading.Thread(target=_release)
    releaser.start()
    runner_mod.run_job(store.job_dir("hb4"), adopt=True)
    releaser.join(timeout=10.0)

    p = store.read_runtime("hb4")["partials"]
    assert p["failed"] >= 2
    assert p["ok"] == 0
    assert p["delivered"] is False
    assert p["consecutive_failures"] >= 2
    assert "255" in (p["last_error"] or "") or "unreachable" in (p["last_error"] or "")


def test_a_stalled_fetch_warns_once_per_escalation_not_once_per_beat(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """46 identical lines are not a signal. The operator gets a *stateful* warning naming the
    consequence — nothing has been retrieved — not one indistinguishable line per attempt."""
    store = _prep(tmp_path, "hb5")
    monkeypatch.setattr(runner_mod, "HEARTBEAT_S", 0.01)
    monkeypatch.setattr(runner_mod, "PARTIALS_REWARN_S", 3600.0)

    def _boom(cluster: str, remote: str, local: Path) -> runner_mod.RsyncStats:
        raise subprocess.CalledProcessError(255, ["rsync"], stderr="Network is unreachable\n")

    monkeypatch.setattr(runner_mod, "_rsync_down", _boom)

    stalled = threading.Event()
    monkeypatch.setitem(
        sys.modules, "sky", _fake_sky(lambda *a, **k: stalled.wait(timeout=10.0))
    )
    _neutralise(monkeypatch)

    def _release() -> None:
        for _ in range(500):
            if store.read_runtime("hb5").get("partials", {}).get("failed", 0) >= 5:
                break
            threading.Event().wait(0.01)
        stalled.set()

    releaser = threading.Thread(target=_release)
    releaser.start()
    runner_mod.run_job(store.job_dir("hb5"), adopt=True)
    releaser.join(timeout=10.0)

    out = capsys.readouterr().out
    warnings = [ln for ln in out.splitlines() if "partial-results fetch" in ln]
    failed = store.read_runtime("hb5")["partials"]["failed"]
    assert failed >= 5
    assert warnings, "a fetch that never delivers must be visible in the job log"
    assert len(warnings) < failed, "one line per beat is the noise that hid the 2026-08-20 loss"
    assert any("not" in w.lower() or "nothing" in w.lower() for w in warnings)


def test_an_empty_remote_dir_is_not_alarming_until_it_is(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """``uv sync`` leaves ``REMOTE_RUN_DIR`` genuinely empty for the first minutes of every run
    (~5 on the 2026-08-20 boxes). Warning there would fire on every healthy job, and an alarm
    that is usually wrong is one nobody reads (R10) — which is the whole failure being fixed.
    The *final* fetch gets no such grace: by then empty means the run produced nothing."""
    store = _prep(tmp_path, "hb7")
    monkeypatch.setattr(runner_mod, "HEARTBEAT_S", 0.01)
    monkeypatch.setattr(runner_mod, "PARTIALS_EMPTY_GRACE_S", 3600.0)  # the whole test is "early"
    monkeypatch.setattr(
        runner_mod, "_rsync_down", lambda c, r, local: runner_mod.RsyncStats(files=0, bytes=0)
    )

    seen = threading.Event()

    def _tail_logs(*a: Any, **k: Any) -> None:
        for _ in range(500):
            if store.read_runtime("hb7").get("partials", {}).get("attempts", 0) >= 3:
                break
            threading.Event().wait(0.01)
        seen.set()

    monkeypatch.setitem(sys.modules, "sky", _fake_sky(_tail_logs))
    _neutralise(monkeypatch)

    runner_mod.run_job(store.job_dir("hb7"), adopt=True)

    assert seen.is_set()
    empties = [ln for ln in capsys.readouterr().out.splitlines() if "copied nothing" in ln]
    assert store.read_runtime("hb7")["partials"]["attempts"] >= 4  # beats were still recorded
    assert len(empties) == 1, "exactly one complaint, from the final fetch"
    assert "produced no output at all" in empties[0]


def test_the_final_fetch_is_recorded_too(tmp_path: Path, monkeypatch: Any) -> None:
    """The post-wait rsync is the one that worked on 2026-08-20 (it is how the succeeded job got
    its files). It belongs in the same record, so "we got everything at the end" and "we got
    nothing at all" are distinguishable states rather than both being silence."""
    store = _prep(tmp_path, "hb6")
    monkeypatch.setattr(runner_mod, "HEARTBEAT_S", 3600.0)  # no mid-run beat will fire
    monkeypatch.setattr(
        runner_mod, "_rsync_down", lambda c, r, local: runner_mod.RsyncStats(files=3, bytes=999)
    )
    monkeypatch.setitem(sys.modules, "sky", _fake_sky(lambda *a, **k: None))
    _neutralise(monkeypatch)

    runner_mod.run_job(store.job_dir("hb6"), adopt=True)

    p = store.read_runtime("hb6")["partials"]
    assert p["delivered"] is True
    assert p["files_total"] >= 3
    assert p["last_delivery_at"] is not None


# ---------------------------------------------------------------------------
# _rsync_down itself: it must be able to report what it moved
# ---------------------------------------------------------------------------


def test_rsync_down_reports_what_it_transferred(monkeypatch: Any, tmp_path: Path) -> None:
    """Parsed from ``--stats``. Thousands separators are locale-dependent, so digits only."""
    captured: dict[str, Any] = {}

    def _run(cmd: list[str], **kw: Any) -> Any:
        captured["cmd"] = cmd
        captured["kw"] = kw
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "Number of files: 4 (reg: 2, dir: 2)\n"
                "Number of regular files transferred: 2\n"
                "Total file size: 1,234 bytes\n"
                "Total transferred file size: 1,234 bytes\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(runner_mod.subprocess, "run", _run)

    stats = runner_mod._rsync_down("lab-x", "/tmp/lab_run", tmp_path / "out")

    assert stats.files == 2
    assert stats.bytes == 1234
    assert "--stats" in captured["cmd"]
    # ssh transport noise (3 lines per failure, ~326 lines on 2026-08-20) stays out of the job log.
    assert captured["kw"].get("capture_output") is True


def test_rsync_down_still_raises_on_a_transport_failure(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Exit 255 must stay an exception — "copied nothing" and "could not connect" are different
    answers and the record has to tell them apart."""

    def _run(cmd: list[str], **kw: Any) -> Any:
        raise subprocess.CalledProcessError(255, cmd, stderr="ssh: Network is unreachable\n")

    monkeypatch.setattr(runner_mod.subprocess, "run", _run)

    try:
        runner_mod._rsync_down("lab-x", "/tmp/lab_run", tmp_path / "out")
    except subprocess.CalledProcessError as e:
        assert e.returncode == 255
    else:  # pragma: no cover - the assertion below is the failure message
        raise AssertionError("a failed rsync must not read as an empty one")


def test_lab_status_surfaces_the_partials_record():
    """The record is only useful if a caller can read it.

    On 2026-08-20 an operator had no way to tell "partial results are being retrieved" from
    "nothing has been retrieved for eleven hours" short of listing the output directory by hand.
    An agent driving the MCP tools reasons from this JSON.
    """
    from helpers import make_manifest

    from lab.core import _status_fields
    from lab.models import JobState

    manifest = make_manifest("jp", "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running}
    )
    record = {"attempts": 12, "ok": 0, "failed": 12, "delivered": False, "files_total": 0}

    fields = _status_fields(manifest, state="running", mirrored=False, partials=record)

    assert fields["partials"] == record


def test_lab_status_partials_is_absent_when_nothing_was_recorded():
    from helpers import make_manifest

    from lab.core import _status_fields
    from lab.models import JobState

    manifest = make_manifest("jq", "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running}
    )

    assert _status_fields(manifest, state="running", mirrored=False)["partials"] is None
