from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from lab import events
from lab.events import store
from lab.events.annotate import digest_of, refs_from

# `lab.events.__init__` re-exports the `record` *function* under the name `record`, shadowing
# the `lab.events.record` submodule as a package attribute — so `import lab.events.record` binds
# to the function, not the module. Go through `sys.modules` (via `importlib`) to reach the
# module itself and reset its process-global counters between tests.
record_module = importlib.import_module("lab.events.record")


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    monkeypatch.setenv("LAB_SESSION_ID", "sess_test")
    # The writer's sequence counter, cached session id and prune-once flag are process-global
    # by design (`_seq` must be monotonic across a real process's calls). Reset them per test so
    # one test's calls don't leak into the next test's assertions.
    monkeypatch.setattr(record_module, "_seq", 0)
    monkeypatch.setattr(record_module, "_session", None)
    monkeypatch.setattr(record_module, "_pruned", False)
    return tmp_path / "events"


def _records() -> list[dict]:
    return list(store.iter_records(store.day_files()))


def test_a_successful_call_writes_an_open_and_a_close_sharing_an_id() -> None:
    with events.record("cli", "status", {"job_id": "j-1"}):
        pass
    opened, closed = _records()
    assert opened["phase"] == "open" and closed["phase"] == "close"
    assert opened["id"] == closed["id"]
    assert opened["action"] == "status" and opened["surface"] == "cli"
    assert closed["outcome"] == "ok"
    assert closed["duration_ms"] >= 0


def test_the_open_line_is_written_before_the_body_runs() -> None:
    with events.record("cli", "submit", {}):
        assert [r["phase"] for r in _records()] == ["open"]


def test_params_are_sanitized_on_the_way_in() -> None:
    with events.record("cli", "submit", {"api_key": "x" * 40}):
        pass
    assert _records()[0]["params"]["api_key"] == "…REDACTED…"


def test_an_exception_records_a_crash_with_the_error_and_reraises() -> None:
    with pytest.raises(ValueError):
        with events.record("mcp", "submit", {}):
            raise ValueError("no capacity")
    closed = _records()[1]
    assert closed["outcome"] == "crash"
    assert closed["error"]["type"] == "ValueError"
    assert closed["error"]["message"] == "no capacity"
    assert "test_events_record.py" in closed["error"]["where"]


def test_a_designated_error_type_records_error_not_crash_and_reraises() -> None:
    """The spec: "A ToolError records outcome:'error'; anything else propagating out records
    crash." `record()` must let a caller designate a handled-exception type (what the MCP
    middleware does for FastMCP's `ToolError`) without letting it *declare* success for an
    arbitrary exception — only a pre-named type is ever reclassified, everything else still
    falls to `crash`, preserving derive-don't-declare."""

    class FakeToolError(Exception):
        pass

    with pytest.raises(FakeToolError):
        with events.record("mcp", "status", {}, error_types=(FakeToolError,)):
            raise FakeToolError("job 'j-nope' not found")
    closed = _records()[1]
    assert closed["outcome"] == "error"
    assert closed["error"]["type"] == "FakeToolError"
    assert "j-nope" in closed["error"]["message"]

    # An exception NOT in error_types still falls through to "crash" — the designation doesn't
    # widen into "any exception is now an error".
    with pytest.raises(ValueError):
        with events.record("mcp", "status", {}, error_types=(FakeToolError,)):
            raise ValueError("unexpected")
    assert _records()[3]["outcome"] == "crash"


def test_keyboard_interrupt_records_interrupted() -> None:
    with pytest.raises(KeyboardInterrupt):
        with events.record("cli", "wait", {}):
            raise KeyboardInterrupt
    assert _records()[1]["outcome"] == "interrupted"


def test_notes_are_discarded_on_success() -> None:
    with events.record("cli", "submit", {}):
        events.note("provision.attempt", zone="europe-west1-b")
    assert _records()[1].get("trace") is None


def test_notes_are_flushed_into_the_trace_on_failure() -> None:
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            events.note("provision.attempt", zone="europe-west1-b")
            events.note("teardown.retry", attempt=2)
            raise RuntimeError("boom")
    trace = _records()[1]["trace"]
    assert [n["k"] for n in trace] == ["provision.attempt", "teardown.retry"]
    assert trace[0]["d"] == {"zone": "europe-west1-b"}
    assert all(isinstance(n["t"], int) for n in trace)


def test_note_outside_a_call_is_a_no_op() -> None:
    events.note("orphan", x=1)  # must not raise, must not write
    assert _records() == []


def test_the_ring_buffer_is_bounded() -> None:
    with pytest.raises(RuntimeError):
        with events.record("cli", "sweep", {}):
            for i in range(500):
                events.note("tick", i=i)
            raise RuntimeError("boom")
    trace = _records()[1]["trace"]
    assert len(trace) == 200
    assert trace[-1]["d"] == {"i": 499}  # the newest are the ones kept


def test_finish_restores_the_outer_call_when_record_blocks_nest() -> None:
    """`finish()` must reset the ContextVar to the token `begin()` captured, not blindly clear it
    to `None`. With `lab mcp` shipped, the MCP middleware's `record()` genuinely nests inside
    whatever the surrounding context already has current — harmless today only because FastMCP
    happens to run each request in a copied context, a third-party implementation detail this
    test does not rely on. Nesting two `record()` blocks directly here proves the outer call is
    still current once the inner one closes, regardless of what runs them."""
    with events.record("cli", "outer", {}) as outer:
        assert events.current() is outer
        with events.record("mcp", "inner", {}) as inner:
            assert events.current() is inner
        assert events.current() is outer
    assert events.current() is None


def test_ref_and_result_land_on_the_close_record() -> None:
    with events.record("cli", "submit", {}) as call:
        call.ref(job_id="j-4f2a")
        call.result(state="failed", cost_usd=0.29)
    closed = _records()[1]
    assert closed["refs"] == {"job_id": "j-4f2a"}
    assert closed["result"] == {"state": "failed", "cost_usd": 0.29}


def test_seq_increments_within_a_process() -> None:
    with events.record("cli", "a", {}):
        pass
    with events.record("cli", "b", {}):
        pass
    assert [r["seq"] for r in _records() if r["phase"] == "open"] == [0, 1]


def test_session_id_comes_from_the_environment() -> None:
    with events.record("cli", "status", {}):
        pass
    assert _records()[0]["session"] == "sess_test"


def test_disabled_writes_nothing_and_still_yields_a_usable_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAB_EVENTS", "0")
    with events.record("cli", "status", {}) as call:
        call.ref(job_id="j-1")
        events.note("x")
    assert _records() == []


def test_refs_from_pulls_known_ids_out_of_a_payload() -> None:
    assert refs_from({"job_id": "j-1", "irrelevant": 5}) == {"job_id": "j-1"}
    assert refs_from("not a mapping") == {}


def test_refs_from_keeps_sweeps_job_ids_findable_by_job() -> None:
    """`lab sweep` returns {sweep_id, count, job_ids:[str]} — a top-level list of job-id
    strings. Before the fix these were lost entirely (refs only ever pulled `job_ids` out of a
    nested list of job-*shaped dicts*, which a plain list of strings is not), so the call that
    created the shards was not findable by `lab history --job <shard-id>`."""
    payload = {"sweep_id": "s-1", "count": 2, "job_ids": ["j-1", "j-2"]}
    assert refs_from(payload) == {"sweep_id": "s-1", "job_ids": ["j-1", "j-2"]}


def test_refs_from_does_not_claim_every_job_from_a_listing_shaped_payload() -> None:
    """`lab list` returns {jobs: [{job_id: ...}, ...]} — *every* job that exists, not every job
    this call touched. The old nested-dict harvest walked any list of job-shaped dicts and
    claimed them all as refs, so a call that touched nothing matched every job in the store —
    bloating records and producing false positives in `lab history --job <id>`."""
    payload = {"jobs": [{"job_id": "j-1"}, {"job_id": "j-2"}, {"job_id": "j-3"}]}
    assert refs_from(payload) == {}


def test_refs_from_maps_confirms_orig_and_confirm_id_onto_job_ids() -> None:
    """`lab confirm` returns {orig_id, confirm_id, verdict}. Both ids are themselves job ids (a
    confirm run submits `confirm_id` as a real job re-deriving `orig_id`) — not a separate id
    namespace — so mapping them onto `job_ids` makes a confirm call findable by `--job
    <orig_id>` or `--job <confirm_id>`, closing the "empty refs" hole the guide used to document
    as a known gap."""
    payload = {"orig_id": "j-1", "confirm_id": "j-2", "verdict": "match"}
    assert refs_from(payload) == {"job_ids": ["j-1", "j-2"]}


def test_id_keys_has_no_dead_entries() -> None:
    """`run_id` used to sit in `_ID_KEYS` even though no shipped call site ever populates it —
    a dead entry the guide had to call out as a known gap. Every key `refs_from` looks for by
    name must be one some real payload actually uses."""
    from lab.events.annotate import _ID_KEYS

    assert "run_id" not in _ID_KEYS


def test_refs_from_caps_job_ids_with_a_truncation_marker() -> None:
    ids = [f"j-{i}" for i in range(100)]
    refs = refs_from({"job_ids": ids})
    assert len(refs["job_ids"]) == 65  # 64 kept + one marker
    assert refs["job_ids"][:64] == ids[:64]
    assert refs["job_ids"][64] == "…36 more"


def test_digest_of_keeps_a_small_summary_not_the_payload() -> None:
    payload = {"state": "succeeded", "actual_cost_usd": 1.25, "series": list(range(1000))}
    assert digest_of(payload) == {"state": "succeeded", "cost_usd": 1.25, "series_n": 1000}
