"""Machine-wide job attribution.

The incident these tests pin: ``lab reconcile --apply --yes`` destroyed seven *running* clusters
belonging to a different project on the same machine, because the "is it ours?" side of the check
read a project-local ``runs/`` while the "does it exist?" side read a user-global
``~/.sky/state.db``. Every assertion here is about the module refusing to answer rather than
answering wrongly: a false ``unknown`` costs a warning line, a false attribution costs money.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab import attribution
from lab.events import store

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the developer's real ``~/.lab/jobs``. The event ledger is already redirected
    by the autouse fixture in ``conftest.py``."""
    monkeypatch.setenv("LAB_JOBS_INDEX_DIR", str(tmp_path / "lab-jobs"))


# ---------------------------------------------------------------- ledger fixtures


def _open(id_: str, *, project: str | dict | None = "other-project", **over: object) -> dict:
    base: dict = {
        "id": id_, "ts": NOW.isoformat(), "phase": "open", "session": "s", "seq": 0,
        "surface": "cli", "action": "submit", "params": {}, "lab_version": "0.6.2",
    }
    if isinstance(project, str):
        base["project"] = {"name": project, "commit": "0" * 40, "dirty": False}
    elif project is not None:
        base["project"] = project
    return {**base, **over}


def _close(id_: str, **refs: object) -> dict:
    return {"id": id_, "ts": NOW.isoformat(), "phase": "close", "outcome": "ok",
            "exit_code": 0, "duration_ms": 10, "refs": dict(refs), "result": {}}


def write_day(day: str, records: list[dict]) -> Path:
    """Write one ``YYYY-MM-DD.jsonl`` day file into the isolated ledger dir."""
    path = store.events_dir() / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return path


# ---------------------------------------------------------------- registry


def test_registry_hit_attributes_project_and_runs_dir(tmp_path: Path) -> None:
    attribution.record_job("j1", project="other-project", runs_dir=tmp_path / "other" / "runs")
    got = attribution.attribute_jobs(["j1"])["j1"]
    assert got.project == "other-project"
    assert got.source == "registry"
    assert got.runs_dir == (tmp_path / "other" / "runs").resolve()
    assert got.known is True


def test_known_jobs_lists_every_registry_entry_with_its_runs_dir(tmp_path: Path) -> None:
    """The cross-project 'what's running' scan (`lab ps`) starts here: every id the registry has
    ever heard of, plus enough to go read its manifest, regardless of which project it's in."""
    attribution.record_job("j1", project="proj-a", runs_dir=tmp_path / "a" / "runs", created_at=NOW)
    attribution.record_job("j2", project="proj-b", runs_dir=tmp_path / "b" / "runs", created_at=NOW)
    got = {kj.job_id: kj for kj in attribution.known_jobs()}
    assert set(got) == {"j1", "j2"}
    assert got["j1"].project == "proj-a"
    assert got["j1"].runs_dir == (tmp_path / "a" / "runs").resolve()
    assert got["j1"].created_at == NOW


def test_known_jobs_on_an_empty_registry_is_empty() -> None:
    assert attribution.known_jobs() == []


def test_known_jobs_never_raises_on_a_corrupt_registry_line(tmp_path: Path) -> None:
    attribution.record_job("j1", project="p", runs_dir=tmp_path / "runs", created_at=NOW)
    with attribution.index_path().open("a", encoding="utf-8") as f:
        f.write("not json at all\n")
    ids = {kj.job_id for kj in attribution.known_jobs()}
    assert ids == {"j1"}


def test_known_jobs_skips_one_bad_record_without_dropping_the_rest(tmp_path: Path) -> None:
    """A single malformed field (valid JSON, unparseable `created_at`) must not abort the whole
    scan — every record after it in the registry is a real job `lab ps` still needs to see."""
    attribution.record_job("j1", project="p", runs_dir=tmp_path / "runs", created_at=NOW)
    with attribution.index_path().open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"v": 1, "job_id": "j-bad", "project": "p", "runs_dir": str(tmp_path / "runs"),
                 "created_at": "not-a-real-timestamp"}
            )
            + "\n"
        )
    attribution.record_job("j3", project="p", runs_dir=tmp_path / "runs", created_at=NOW)

    ids = {kj.job_id for kj in attribution.known_jobs()}

    assert "j3" in ids, "a record after the malformed one must still be seen"
    assert "j1" in ids


def test_record_job_creates_the_index_under_the_env_override(tmp_path: Path) -> None:
    attribution.record_job("j1", project="p", runs_dir=tmp_path / "runs")
    assert attribution.index_path() == tmp_path / "lab-jobs" / "index.jsonl"
    (record,) = [json.loads(line) for line in attribution.index_path().read_text().splitlines()]
    assert record["job_id"] == "j1"
    assert record["project"] == "p"
    assert Path(record["runs_dir"]).is_absolute()
    assert record["created_at"]


def test_record_job_is_idempotent(tmp_path: Path) -> None:
    for _ in range(3):
        attribution.record_job("j1", project="p", runs_dir=tmp_path / "runs", created_at=NOW)
    assert attribution.index_path().read_text().count("\n") == 1
    assert attribution.attribute_jobs(["j1"])["j1"].project == "p"


def test_re_recording_with_a_new_runs_dir_takes_the_newest_record(tmp_path: Path) -> None:
    """Not a conflict to resolve — the job genuinely moved. Last write wins, and the older line
    stays on disk (append-only), so the read side must not pick it up."""
    attribution.record_job("j1", project="p", runs_dir=tmp_path / "old", created_at=NOW)
    attribution.record_job("j1", project="p2", runs_dir=tmp_path / "new", created_at=NOW)
    got = attribution.attribute_jobs(["j1"])["j1"]
    assert got.project == "p2"
    assert got.runs_dir == (tmp_path / "new").resolve()


def test_registry_record_without_a_project_is_unknown_not_a_guess(tmp_path: Path) -> None:
    attribution.record_job("j1", project=None, runs_dir=tmp_path / "runs")
    got = attribution.attribute_jobs(["j1"])["j1"]
    assert got.project is None
    assert got.source == "unknown"
    assert got.known is False
    # the runs_dir is still a useful hint even when ownership is unproven
    assert got.runs_dir == (tmp_path / "runs").resolve()


# ---------------------------------------------------------------- ledger fallback


def test_ledger_fallback_attributes_a_job_created_before_the_registry_existed() -> None:
    write_day("2026-08-19", [_open("e1", project="tempotron-capacity"),
                             _close("e1", job_id="20260819-181113-6ab6d7")])
    got = attribution.attribute_jobs(["20260819-181113-6ab6d7"])["20260819-181113-6ab6d7"]
    assert got.project == "tempotron-capacity"
    assert got.source == "ledger"
    assert got.runs_dir is None  # the ledger never recorded one — do not invent it


def test_ledger_attributes_every_shard_of_a_sweep() -> None:
    write_day("2026-08-19", [_open("e1", project="sweeper"),
                             _close("e1", sweep_id="sw1", job_ids=["a", "b", "c"])])
    got = attribution.attribute_jobs(["a", "c"])
    assert {k: v.project for k, v in got.items()} == {"a": "sweeper", "c": "sweeper"}


def test_ledger_pair_split_across_two_day_files_still_resolves() -> None:
    """An overnight supervisor opens on day N and closes on day N+1. Scanning newest-first must
    carry the unmatched close back to the older file that holds its open."""
    write_day("2026-08-20", [_close("e1", job_id="j1")])
    write_day("2026-08-19", [_open("e1", project="overnight", surface="supervisor",
                                   action="run")])
    got = attribution.attribute_jobs(["j1"])["j1"]
    assert (got.project, got.source) == ("overnight", "ledger")


def test_ledger_open_and_close_in_file_order_within_one_day() -> None:
    """Within a file the open precedes its close, so a single forward pass that only looks
    backwards would never pair them."""
    write_day("2026-08-20", [_open("e1", project="p"), _close("e1", job_id="j1")])
    assert attribution.attribute_jobs(["j1"])["j1"].project == "p"


def test_ledger_event_without_a_project_is_unknown() -> None:
    write_day("2026-08-19", [_open("e1", project=None), _close("e1", job_id="j1")])
    got = attribution.attribute_jobs(["j1"])["j1"]
    assert got.project is None and got.source == "unknown"


def test_ledger_event_with_an_empty_project_name_is_unknown() -> None:
    write_day("2026-08-19", [_open("e1", project={"name": "  "}), _close("e1", job_id="j1")])
    assert attribution.attribute_jobs(["j1"])["j1"].project is None


def test_a_close_with_no_open_attributes_nothing() -> None:
    write_day("2026-08-19", [_close("e1", job_id="j1")])
    assert attribution.attribute_jobs(["j1"])["j1"].source == "unknown"


def test_two_projects_claiming_one_job_id_is_unknown_never_a_coin_flip() -> None:
    write_day("2026-08-19", [_open("e1", project="alpha"), _close("e1", job_id="j1"),
                             _open("e2", project="beta"), _close("e2", job_id="j1")])
    got = attribution.attribute_jobs(["j1"])["j1"]
    assert got.project is None and got.source == "unknown"


def test_the_registry_wins_over_the_ledger(tmp_path: Path) -> None:
    write_day("2026-08-19", [_open("e1", project="stale"), _close("e1", job_id="j1")])
    attribution.record_job("j1", project="authoritative", runs_dir=tmp_path / "runs")
    got = attribution.attribute_jobs(["j1"])["j1"]
    assert (got.project, got.source) == ("authoritative", "registry")


# ---------------------------------------------------------------- unknown / degradation


def test_unknown_when_neither_source_has_it() -> None:
    got = attribution.attribute_jobs(["nobody-knows"])["nobody-knows"]
    assert got == attribution.Attribution("nobody-knows", None, None, "unknown")


def test_every_requested_id_gets_an_entry(tmp_path: Path) -> None:
    attribution.record_job("j1", project="p", runs_dir=tmp_path / "runs")
    assert set(attribution.attribute_jobs(["j1", "j2", "j3"])) == {"j1", "j2", "j3"}


def test_no_ids_requested_touches_nothing() -> None:
    assert attribution.attribute_jobs([]) == {}


def test_a_missing_lab_home_is_all_unknown_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LAB_JOBS_INDEX_DIR", str(tmp_path / "nope" / "jobs"))
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "nope" / "events"))
    assert attribution.attribute_jobs(["j1"])["j1"].source == "unknown"
    assert attribution.known_job_ids() == set()


def test_a_corrupt_registry_line_is_skipped_and_the_rest_still_resolves(tmp_path: Path) -> None:
    attribution.record_job("j1", project="p1", runs_dir=tmp_path / "runs")
    with attribution.index_path().open("a", encoding="utf-8") as f:
        f.write('{"job_id": "j2", "project": "trunca\n')  # torn write
        f.write("not json at all\n")
        f.write("[1, 2, 3]\n")  # parseable JSON, wrong shape
    attribution.record_job("j3", project="p3", runs_dir=tmp_path / "runs")
    got = attribution.attribute_jobs(["j1", "j2", "j3"])
    assert got["j1"].project == "p1"
    assert got["j3"].project == "p3"
    assert got["j2"].project is None  # the torn line must not be half-believed


def test_a_corrupt_ledger_line_is_skipped(tmp_path: Path) -> None:
    write_day("2026-08-19", [_open("e1", project="p"), _close("e1", job_id="j1")])
    path = store.events_dir() / "2026-08-19.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write('{"id": "e2", "phase": "close", "refs": {"job_i\n')
    assert attribution.attribute_jobs(["j1"])["j1"].project == "p"


def test_wrongly_typed_ledger_fields_do_not_raise() -> None:
    write_day("2026-08-19", [
        {"id": 17, "phase": "open", "project": {"name": "p"}},           # id not a string
        {"id": "e1", "phase": "open", "project": "a-string-not-a-dict"},
        {"id": "e1", "phase": "close", "refs": "not-a-mapping"},
        {"id": "e2", "phase": "open", "project": {"name": ["nested"]}},
        {"id": "e2", "phase": "close", "refs": {"job_id": 42, "job_ids": [None, "j1"]}},
    ])
    got = attribution.attribute_jobs(["j1"])["j1"]
    assert got.project is None and got.source == "unknown"


def test_a_registry_path_that_is_a_directory_degrades(tmp_path: Path) -> None:
    attribution.index_path().mkdir(parents=True)
    assert attribution.attribute_jobs(["j1"])["j1"].source == "unknown"


def test_record_job_never_raises_when_the_index_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory\n")
    monkeypatch.setenv("LAB_JOBS_INDEX_DIR", str(blocker / "jobs"))
    attribution.record_job("j1", project="p", runs_dir=tmp_path / "runs")  # must not raise


# ---------------------------------------------------------------- known_job_ids


def test_known_job_ids_unions_both_sources(tmp_path: Path) -> None:
    attribution.record_job("j1", project="p", runs_dir=tmp_path / "runs")
    write_day("2026-08-19", [_open("e1", project="p"), _close("e1", job_id="j2"),
                             _open("e2", project="p"), _close("e2", job_ids=["j3", "j4"])])
    assert attribution.known_job_ids() == {"j1", "j2", "j3", "j4"}


def test_known_job_ids_includes_ids_it_cannot_attribute(tmp_path: Path) -> None:
    """Knowing an id *exists* is a different question from knowing who owns it."""
    write_day("2026-08-19", [_close("e1", job_id="j2")])  # no open, so no project
    assert attribution.known_job_ids() == {"j2"}
    assert attribution.attribute_jobs(["j2"])["j2"].project is None


# ---------------------------------------------------------------- bounds


def test_the_scan_stops_at_the_newest_file_that_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconcile runs this on every invocation, so the ledger walk must not read 90 days of
    JSONL once the answer is in hand."""
    write_day("2026-08-20", [_open("e1", project="p"), _close("e1", job_id="j1")])
    for day in ("2026-08-19", "2026-08-18", "2026-08-17"):
        write_day(day, [_open(f"o-{day}", project="p"), _close(f"o-{day}", job_id=f"x-{day}")])
    read: list[Path] = []
    real = store.iter_records

    def spy(paths):  # type: ignore[no-untyped-def]
        read.extend(paths)
        return real(paths)

    monkeypatch.setattr(store, "iter_records", spy)
    assert attribution.attribute_jobs(["j1"])["j1"].project == "p"
    days = [p.stem for p in read if p.parent == store.events_dir()]
    assert days == ["2026-08-20"]


def test_the_index_is_capped_and_dropping_records_only_costs_knowledge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unbounded growth in a file reconcile reads on every run is the smell; forgetting the
    oldest jobs is the safe direction to fail (unknown => do not destroy)."""
    monkeypatch.setenv("LAB_JOBS_INDEX_MAX_RECORDS", "10")
    for i in range(25):
        attribution.record_job(f"j{i:02d}", project="p", runs_dir=tmp_path / "runs")
    lines = attribution.index_path().read_text().splitlines()
    assert len(lines) <= 11  # capped, modulo the record that triggered the compaction
    got = attribution.attribute_jobs(["j00", "j24"])
    assert got["j24"].project == "p"  # the newest survives
    assert got["j00"].project is None  # the oldest degrades to unknown, not to a wrong answer


# ---------------------------------------------------------------- concurrency


WRITER = """
import sys
from pathlib import Path
from lab.attribution import record_job
tag = sys.argv[1]
for i in range(100):
    record_job(f"{tag}-{i:03d}", project=f"proj-{tag}", runs_dir=Path("/tmp/runs") / tag)
"""


def test_concurrent_writers_produce_only_whole_lines(tmp_path: Path) -> None:
    """A sharded sweep creates many manifests at once, each from its own process. The per-day
    lock-file pattern borrowed from ``lab.events.store`` is what rules out torn lines here."""
    env = {**os.environ, "LAB_JOBS_INDEX_DIR": str(tmp_path / "jobs"),
           "LAB_JOBS_INDEX_MAX_RECORDS": "100000"}
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, f"w{n}"], env=env) for n in range(6)]
    for p in procs:
        assert p.wait() == 0
    lines = (tmp_path / "jobs" / "index.jsonl").read_text().splitlines()
    assert len(lines) == 6 * 100
    ids = {json.loads(line)["job_id"] for line in lines}  # every line whole and parseable
    assert len(ids) == 6 * 100


COMPACTING_WRITER = """
import sys
from pathlib import Path
from lab.attribution import record_job
tag = sys.argv[1]
for i in range(150):
    record_job(f"{tag}-{i:03d}", project="p", runs_dir=Path("/tmp/runs") / tag)
"""


def test_the_cap_holds_while_writers_race_the_compaction(tmp_path: Path) -> None:
    """Same nuance ``test_events_concurrency`` documents for the ledger: a bare concurrent
    append is *already* safe here — Linux serializes one ``write(2)`` to a regular file, and a
    record is a couple of hundred bytes — so the lock is not what keeps lines whole.

    Its load-bearing job is the read-modify-write in ``_locked_append``: over the cap, the writer
    reads every record, keeps a suffix and ``os.replace``s the file. Two of those racing each
    other silently drop one of the two new records into the orphaned inode. That loss is
    intermittent (observed at a cap of 50 with the lock neutralised) and its exact victim depends
    on interleaving, so this test pins the invariants that must hold *every* time instead: the
    cap is never exceeded, no record is torn, and compaction never duplicates one it kept.
    """
    cap = 50
    env = {**os.environ, "LAB_JOBS_INDEX_DIR": str(tmp_path / "jobs"),
           "LAB_JOBS_INDEX_MAX_RECORDS": str(cap)}
    procs = [subprocess.Popen([sys.executable, "-c", COMPACTING_WRITER, f"w{n}"], env=env)
             for n in range(6)]
    for p in procs:
        assert p.wait() == 0
    lines = (tmp_path / "jobs" / "index.jsonl").read_text().splitlines()
    assert len(lines) == cap
    ids = [json.loads(line)["job_id"] for line in lines]  # every line whole and parseable
    assert len(set(ids)) == cap
