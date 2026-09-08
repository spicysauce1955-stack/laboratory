"""A job that dies mid-run must still return the rows it had already finished (§6c salvage).

Artifacts reach the object store exactly once, in ``run_job``, after the wait. Anything that
stops the supervisor before that line -- a timeout at the cap on a box whose supervisor has since
died, a SIGKILLed supervisor (~40% of DO supervisors on this box), a lost cluster -- returns an
*empty* R2 prefix, even though the entrypoint had been appending and fsyncing ``results.csv`` per
row for hours. Three zero-row timeouts in the 2026-09 campaign cost $10.4 between them and
returned nothing.

The rows were not unreachable: ``PartialsFetcher`` has been rsyncing the remote run dir down
every 60 s since 2026-08-21. They were on the *supervisor's* disk -- which for a
scheduler-launched job is the always-on droplet's, a machine nobody reads and the next
blue-green deploy replaces. Mirroring what has already been fetched is what turns "the rows exist
somewhere" into "the rows survive".

Three properties this must not trade away, one per section below:

* a mid-run copy can never be read as a finished one;
* a mid-run copy can never contain half a row;
* a failure to mirror is not a failure of the job.
"""

from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
from helpers import make_manifest

import lab.partials as P
import lab.sky_runner as runner_mod
from lab._util import now
from lab.models import CostInfo, JobState
from lab.store import JobStore


class FakeR2:
    """An in-memory object store with the two verbs the mirror uses. No network, no boto3."""

    def __init__(self, *, fail: bool = False) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts = 0
        self.fail = fail

    def put_bytes(self, key: str, data: bytes) -> None:
        if self.fail:
            raise OSError("r2 is down")
        self.puts += 1
        self.objects[key] = data

    def put_text(self, key: str, text: str) -> None:
        self.put_bytes(key, text.encode())

    def uri(self, prefix: str) -> str:
        return f"r2://fake/{prefix}"


def _mirror(tmp_path: Path, r2: FakeR2, **kw: Any) -> tuple[P.PartialMirror, Path]:
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    return P.PartialMirror("job-1", out, r2_factory=lambda: r2, **kw), out


# ---------------------------------------------------------------------------------------------
# 1. A partial can never be mistaken for a complete artifact
# ---------------------------------------------------------------------------------------------


class TestPartialIsAlwaysDistinguishableFromComplete:
    def test_it_never_writes_the_jobs_real_artifact_keys(self, tmp_path: Path) -> None:
        """``fetch_artifacts``, ``lab export`` and ``sweep-aggregate`` all read
        ``<job_id>/<name>``. A mid-run copy must be somewhere else, full stop."""
        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2)
        (out / "results.csv").write_text("seed,acc\n0,0.5\n")

        assert mirror.maybe_mirror("running") > 0
        assert r2.objects
        for key in r2.objects:
            assert key.startswith(f"job-1/{P.PARTIAL_DIR}/"), key
        assert "job-1/results.csv" not in r2.objects

    def test_the_marker_travels_with_the_rows(self, tmp_path: Path) -> None:
        """The mark rides *inside* the prefix, so it survives a plain ``download_dir`` or a hand
        copy — a bare directory of rows with no marker is what "mistaken for complete" means."""
        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2)
        (out / "results.csv").write_text("seed,acc\n0,0.5\n1,0.7\n")

        mirror.maybe_mirror("running")

        marker = json.loads(r2.objects[f"job-1/{P.PARTIAL_DIR}/{P.PARTIAL_MARKER}"])
        assert marker["complete"] is False
        assert marker["job_id"] == "job-1"
        assert [f["name"] for f in marker["files"]] == ["results.csv"]
        assert marker["files"][0]["lines"] == 3

    def test_the_mark_is_the_producing_jobs_status_not_a_new_vocabulary(
        self, tmp_path: Path
    ) -> None:
        """``lab.aggregate`` decides partiality from the producing job's status and stamps it
        into ``_shard_status``; the marker carries the same field from the same source, so a
        salvaged table merges under the rules that already exist rather than new ones."""
        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2)
        (out / "results.csv").write_text("seed\n0\n")

        mirror.maybe_mirror("running")

        from lab.aggregate import STATUS_COLUMN

        marker = json.loads(r2.objects[f"job-1/{P.PARTIAL_DIR}/{P.PARTIAL_MARKER}"])
        assert marker[STATUS_COLUMN] == "running"
        assert marker[STATUS_COLUMN] != JobState.succeeded.value

    def test_only_result_tables_are_mirrored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checkpoints are the expensive half of an output dir and the useless half of a
        salvage. Line-oriented tables at the top level only — nothing recursive, nothing huge."""
        monkeypatch.setattr(P, "MIRROR_MAX_BYTES", 64)
        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2)
        (out / "results.csv").write_text("seed\n0\n")
        (out / "metrics.jsonl").write_text('{"name":"acc"}\n')
        (out / "model.pt").write_bytes(b"\x00" * 4096)
        (out / "big.csv").write_bytes(b"x,y\n" + b"1,2\n" * 64)
        (out / "ckpt").mkdir()
        (out / "ckpt" / "inner.csv").write_text("a\n1\n")

        mirror.maybe_mirror("running")

        names = {k.rsplit("/", 1)[-1] for k in r2.objects}
        assert names == {"results.csv", "metrics.jsonl", P.PARTIAL_MARKER}

    def test_the_success_sentinel_can_never_be_mirrored(self, tmp_path: Path) -> None:
        """``confirm_success`` reads ``.lab_success`` out of ``output/`` to decide whether a run
        may be labelled succeeded. If a mid-run copy of it could be salvaged back into a run dir,
        this would have turned a dead job into a green one — the one way this feature could have
        cost more than it saves. Hidden files are never eligible."""
        from lab.backends.skypilot import SUCCESS_SENTINEL, TIMEOUT_SENTINEL

        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2)
        (out / "results.csv").write_text("seed\n0\n")
        (out / SUCCESS_SENTINEL).touch()
        (out / TIMEOUT_SENTINEL).touch()

        mirror.maybe_mirror("running")

        assert not any(SUCCESS_SENTINEL in k or TIMEOUT_SENTINEL in k for k in r2.objects)


# ---------------------------------------------------------------------------------------------
# 2. Torn reads
# ---------------------------------------------------------------------------------------------


class TestNoHalfRowEverReachesTheStore:
    def test_a_row_still_being_appended_is_left_behind(self, tmp_path: Path) -> None:
        """rsync copies a file that is being appended to; the tail can be half a row. The
        mirrored object stops at the last newline, so the object is always a whole table."""
        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2)
        (out / "results.csv").write_text("seed,acc\n0,0.5\n1,0.7\n2,0.9")  # no trailing newline

        mirror.maybe_mirror("running")

        body = r2.objects[f"job-1/{P.PARTIAL_DIR}/results.csv"].decode()
        assert body == "seed,acc\n0,0.5\n1,0.7\n"
        assert body.endswith("\n")

    def test_a_file_with_nothing_complete_yet_is_not_mirrored(self, tmp_path: Path) -> None:
        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2)
        (out / "results.csv").write_text("seed,acc")  # header still being written

        assert mirror.maybe_mirror("running") == 0
        assert r2.objects == {}

    def test_whole_lines_is_pure_and_total(self) -> None:
        assert P.whole_lines(b"") == b""
        assert P.whole_lines(b"abc") == b""
        assert P.whole_lines(b"a\nb") == b"a\n"
        assert P.whole_lines(b"a\nb\n") == b"a\nb\n"

    def test_a_salvaged_table_merges_under_the_existing_partial_rules(
        self, tmp_path: Path
    ) -> None:
        """End of the chain: the mirrored bytes, fed to the real merge with the status the
        marker records, produce rows stamped partial rather than an exception."""
        from lab.aggregate import STATUS_COLUMN, merge_seed_rows

        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2)
        (out / "results.csv").write_text("seed,acc\n0,0.5\n1,0.7\n2,0.")

        mirror.maybe_mirror("running")
        body = r2.objects[f"job-1/{P.PARTIAL_DIR}/results.csv"].decode()
        merged, present, partial = merge_seed_rows([(body, "timed_out")], "seed")

        assert present == [0, 1] and partial == [0, 1]
        assert STATUS_COLUMN in merged.splitlines()[0]


# ---------------------------------------------------------------------------------------------
# 3. Cost: an interval, and no upload for an unchanged file
# ---------------------------------------------------------------------------------------------


class TestItDoesNotPayForNothing:
    def test_an_unchanged_table_is_not_uploaded_again(self, tmp_path: Path) -> None:
        """The no-op case. A finished-but-still-running job (or a stalled entrypoint) would
        otherwise re-PUT the same bytes for the rest of the rental."""
        clock = {"t": 0.0}
        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2, interval_s=10.0, clock=lambda: clock["t"])
        (out / "results.csv").write_text("seed\n0\n")

        assert mirror.maybe_mirror("running") > 0
        first = r2.puts
        for _ in range(20):
            clock["t"] += 60.0
            mirror.maybe_mirror("running")

        assert r2.puts == first, "an unchanged output dir was mirrored again"

    def test_a_grown_table_is_mirrored_again(self, tmp_path: Path) -> None:
        clock = {"t": 0.0}
        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2, interval_s=10.0, clock=lambda: clock["t"])
        (out / "results.csv").write_text("seed\n0\n")
        mirror.maybe_mirror("running")

        clock["t"] += 60.0
        (out / "results.csv").write_text("seed\n0\n1\n")
        assert mirror.maybe_mirror("running") > 0
        assert r2.objects[f"job-1/{P.PARTIAL_DIR}/results.csv"] == b"seed\n0\n1\n"

    def test_the_interval_bounds_how_often_it_can_upload(self, tmp_path: Path) -> None:
        """The rsync heartbeat is 60 s; mirroring on every beat would be 240 uploads of a
        growing table over a 4 h run for at most 4 extra minutes of saved rows."""
        clock = {"t": 0.0}
        r2 = FakeR2()
        mirror, out = _mirror(tmp_path, r2, interval_s=300.0, clock=lambda: clock["t"])

        uploads = 0
        for beat in range(1, 61):  # one hour of 60s heartbeats
            (out / "results.csv").write_text("seed\n" + "".join(f"{i}\n" for i in range(beat)))
            uploads += 1 if mirror.maybe_mirror("running") else 0
            clock["t"] += 60.0

        assert uploads == 12, uploads  # 3600s / 300s

    def test_a_missing_output_dir_costs_nothing(self, tmp_path: Path) -> None:
        r2 = FakeR2()
        mirror = P.PartialMirror("job-1", tmp_path / "nope", r2_factory=lambda: r2)

        assert mirror.maybe_mirror("running") == 0
        assert r2.puts == 0

    def test_no_object_store_configured_is_a_silent_no_op(self, tmp_path: Path) -> None:
        mirror, out = _mirror(tmp_path, FakeR2())
        mirror._r2_factory = lambda: None  # r2_enabled() is False in the field
        (out / "results.csv").write_text("seed\n0\n")

        assert mirror.maybe_mirror("running") == 0
        assert mirror.state["failed"] == 0  # not configured is not a failure


# ---------------------------------------------------------------------------------------------
# 4. Failure is never fatal
# ---------------------------------------------------------------------------------------------


class TestAFailedMirrorIsNotAFailedJob:
    def test_it_records_the_error_and_returns(self, tmp_path: Path) -> None:
        r2 = FakeR2(fail=True)
        mirror, out = _mirror(tmp_path, r2)
        (out / "results.csv").write_text("seed\n0\n")

        assert mirror.maybe_mirror("running") == 0  # no raise
        assert mirror.state["failed"] == 1
        assert "r2 is down" in (mirror.state["last_error"] or "")

    def test_a_broken_store_does_not_retry_every_beat(self, tmp_path: Path) -> None:
        """A store that is down stays down; hammering it burns the interval budget on nothing."""
        clock = {"t": 0.0}
        r2 = FakeR2(fail=True)
        mirror, out = _mirror(tmp_path, r2, interval_s=300.0, clock=lambda: clock["t"])
        (out / "results.csv").write_text("seed\n0\n")

        for _ in range(10):
            mirror.maybe_mirror("running")
            clock["t"] += 60.0

        assert mirror.state["failed"] == 2  # t=0 and t=300, not ten


# ---------------------------------------------------------------------------------------------
# 5. Wired into the supervisor
# ---------------------------------------------------------------------------------------------


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
    mod = types.ModuleType("sky")
    mod.get = lambda x: x  # type: ignore[attr-defined]
    mod.tail_logs = tail_logs  # type: ignore[attr-defined]
    mod.queue = lambda cluster, skip_finished=False: [  # type: ignore[attr-defined]
        {"job_id": 1, "status": _Status("SUCCEEDED")}
    ]
    return mod


def _neutralise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_mod, "_resolve_hourly", lambda *a, **k: 0.2)
    monkeypatch.setattr(runner_mod, "r2_enabled", lambda: False)  # end-of-run upload off
    monkeypatch.setattr(
        runner_mod,
        "tear_down_and_record",
        lambda sky_mod, cluster, st, jid, cloud="vast", **k: True,
    )


class TestTheSupervisorMirrorsWhileTheJobRuns:
    def test_rows_reach_the_store_before_the_run_ends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: the object store holds rows while ``tail_logs`` is still blocking,
        i.e. before anything that could kill the supervisor has happened."""
        store = _prep(tmp_path, "pm1")
        r2 = FakeR2()
        monkeypatch.setattr(runner_mod, "HEARTBEAT_S", 0.02)
        monkeypatch.setattr(P, "MIRROR_INTERVAL_S", 0.0)
        monkeypatch.setattr(P, "default_r2", lambda: r2)

        mirrored = threading.Event()

        def _rsync(cluster: str, remote: str, local: Path) -> runner_mod.RsyncStats:
            local.mkdir(parents=True, exist_ok=True)
            (local / "results.csv").write_text("seed,acc\n0,0.5\n")
            if r2.objects:
                mirrored.set()
            return runner_mod.RsyncStats(files=1, bytes=32)

        monkeypatch.setattr(runner_mod, "_rsync_down", _rsync)

        def _tail_logs(*a: Any, **k: Any) -> None:
            mirrored.wait(timeout=10.0)

        monkeypatch.setitem(sys.modules, "sky", _fake_sky(_tail_logs))
        _neutralise(monkeypatch)

        runner_mod.run_job(store.job_dir("pm1"), adopt=True)

        assert mirrored.is_set(), "nothing was mirrored while the job was still streaming"
        assert r2.objects[f"pm1/{P.PARTIAL_DIR}/results.csv"] == b"seed,acc\n0,0.5\n"
        assert store.read_runtime("pm1")["partials"]["mirror"]["ok"] >= 1

    def test_a_mirror_that_fails_leaves_the_job_healthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Degrade silently to the end-of-run upload: no non-zero exit, no teardown alarm, and
        the reason recorded where the other partial-results state already lives."""
        store = _prep(tmp_path, "pm2")
        monkeypatch.setattr(runner_mod, "HEARTBEAT_S", 0.01)
        monkeypatch.setattr(P, "MIRROR_INTERVAL_S", 0.0)

        tried = threading.Event()

        def _broken_store() -> FakeR2:
            tried.set()
            return FakeR2(fail=True)

        monkeypatch.setattr(P, "default_r2", _broken_store)

        def _rsync(cluster: str, remote: str, local: Path) -> runner_mod.RsyncStats:
            local.mkdir(parents=True, exist_ok=True)
            (local / "results.csv").write_text("seed\n0\n")
            (local / ".lab_success").touch()
            return runner_mod.RsyncStats(files=1, bytes=8)

        monkeypatch.setattr(runner_mod, "_rsync_down", _rsync)
        # Hold the stream until a mirror has been attempted; `stop()` then joins the fetch
        # thread, so the record read below is settled rather than raced.
        monkeypatch.setitem(
            sys.modules, "sky", _fake_sky(lambda *a, **k: tried.wait(timeout=10.0))
        )
        _neutralise(monkeypatch)

        code = runner_mod.run_job(store.job_dir("pm2"), adopt=True)

        m = store.read_manifest("pm2")
        assert code == 0
        assert m.status is JobState.succeeded
        assert m.teardown_status != "failed"
        assert store.read_runtime("pm2")["partials"]["mirror"]["failed"] >= 1

    def test_an_exploding_mirror_cannot_escape_into_the_supervisor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not just store errors: anything this code can raise, including its own bugs."""
        store = _prep(tmp_path, "pm3")
        monkeypatch.setattr(runner_mod, "HEARTBEAT_S", 0.01)
        exploded = threading.Event()

        def _boom() -> Any:
            exploded.set()
            raise RuntimeError("bug in the mirror itself")

        monkeypatch.setattr(P, "default_r2", _boom)
        monkeypatch.setattr(P, "MIRROR_INTERVAL_S", 0.0)

        def _rsync(cluster: str, remote: str, local: Path) -> runner_mod.RsyncStats:
            local.mkdir(parents=True, exist_ok=True)
            (local / "results.csv").write_text("seed\n0\n")
            return runner_mod.RsyncStats(files=1, bytes=8)

        monkeypatch.setattr(runner_mod, "_rsync_down", _rsync)
        monkeypatch.setitem(
            sys.modules, "sky", _fake_sky(lambda *a, **k: exploded.wait(timeout=10.0))
        )
        _neutralise(monkeypatch)

        assert runner_mod.run_job(store.job_dir("pm3"), adopt=True) == 0
        assert exploded.is_set(), "the raising path was never reached — the test proves nothing"


# ---------------------------------------------------------------------------------------------
# 6. Getting it back
# ---------------------------------------------------------------------------------------------


class TestSalvage:
    def test_fetch_artifacts_falls_back_to_the_partial_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A supervisor that died never set ``artifacts_uri`` and never uploaded, so the normal
        R2 fallback finds nothing. The mirrored rows are the only copy left."""
        from lab.core import Lab

        home = tmp_path / "runs"
        store = JobStore(home)
        store.create(
            make_manifest("pm4", "python x.py", timeout="1h").model_copy(
                update={"status": JobState.timed_out, "started_at": now()}
            )
        )

        class _Downloader:
            def download_dir(self, prefix: str, local: Path) -> int:
                if not prefix.endswith(P.PARTIAL_DIR):
                    return 0
                local.mkdir(parents=True, exist_ok=True)
                (local / "results.csv").write_text("seed,acc\n0,0.5\n")
                (local / P.PARTIAL_MARKER).write_text('{"complete": false}')
                return 2

        monkeypatch.setattr("lab.core.r2_enabled", lambda: True)
        monkeypatch.setattr("lab.core.R2Store.from_env", classmethod(lambda cls: _Downloader()))

        from lab.backends.local import LocalBackend

        lab = Lab(backend=LocalBackend(home=home, repo=tmp_path), repo=tmp_path, home=home)
        records = lab.fetch_artifacts("pm4")

        names = {r.name for r in records}
        assert "results.csv" in names
        assert (store.output_dir("pm4") / P.PARTIAL_MARKER).exists(), "the mark must come too"
