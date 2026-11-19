from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from lab.events.models import Event
from lab.events.stats import signature, stats, stats_dict

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def _event(id_: str, *, action="submit", outcome="ok", error=None, usd=None, ms=1000,
           ts=NOW) -> Event:
    return Event(id=id_, ts=ts, session="s", seq=0, surface="cli", action=action,
                 outcome=outcome, duration_ms=ms, error=error,
                 result={"cost_usd": usd} if usd else {})


def test_signature_normalizes_ids_numbers_paths_and_zones() -> None:
    a = {"type": "ProvisionTimeout", "message": "host never reached UP in 20m (europe-west1-b)"}
    b = {"type": "ProvisionTimeout", "message": "host never reached UP in 45m (us-central1-a)"}
    assert signature(a) == signature(b)


def test_signature_keeps_different_bugs_apart() -> None:
    a = {"type": "ProvisionTimeout", "message": "host never reached UP in 20m"}
    b = {"type": "TeardownFailed", "message": "sky.down exhausted 3 retries"}
    assert signature(a) != signature(b)


def test_signature_keeps_different_messages_apart_within_the_same_type() -> None:
    # Same error type, different content-bearing wording — the normalizer must not collapse
    # these into one row just because `type` matches. (A test differing in `type` alone would
    # pass regardless of whether the message half of the signature preserves anything at all.)
    a = {"type": "TeardownFailed", "message": "sky.down exhausted 3 retries"}
    b = {"type": "TeardownFailed", "message": "vastai-sdk fallback exhausted 3 retries"}
    assert signature(a) != signature(b)


def test_signature_of_a_missing_error_is_stable() -> None:
    assert signature(None) == "unknown"


def test_signature_collapses_a_path_even_when_a_segment_looks_like_a_sha() -> None:
    # "abc1234567" is a 10-char hex-only directory name: if the sha rule ran before the path
    # rule, it would eat that one segment and leave the rest of the path as literal text,
    # so two failures against different config files would no longer collapse to one signature.
    a = {"type": "IOError",
         "message": "read failed at /home/user/abc1234567/config.yaml"}
    b = {"type": "IOError",
         "message": "read failed at /home/user/abc1234567/other.yaml"}
    assert signature(a) == signature(b)
    assert "<path>" in signature(a)
    assert "config.yaml" not in signature(a)


@pytest.mark.parametrize(
    ("first", "second", "placeholder"),
    [
        # A real job id is `YYYYMMDD-HHMMSS-<6 hex>`. Only the legacy `j-…` shape used to be
        # recognised, so the date became `<sha>`, the time `<n>`, and the hex suffix survived —
        # every "unknown job id" was its own signature.
        ("20261119-102456-7486e1", "20261120-081512-a0b1c2", "<job>"),
        ("20261119-102456-748612", "20261120-081512-abcdef", "<job>"),  # all-digit / all-alpha
        ("sweep-20261119-102809-b2ca49", "sweep-20261121-000000-ffee00", "<sweep>"),
        ("reg-20261119-102809-b2ca49", "reg-20261121-000000-ffee00", "<reg>"),
        (
            "lab-proj-20261119-102456-7486e1",
            "lab-proj-20261120-081512-a0b1c2",
            "<cluster>",
        ),
        ("lab-20261119-102456-7486e1", "lab-20261120-081512-a0b1c2", "<cluster>"),  # legacy
        (
            "lab-tempotron-capacity-20261119-102456-7486e1-head-1a2b3c4d-compute",
            "lab-tempotron-capacity-20261120-081512-a0b1c2-head-c0h9pkx0-compute",
            "<cluster>",
        ),
        (  # GCP truncates past 35 chars: no job id survives, only the node suffix
            "lab-laboratory-20260820-ef-3dd12990-head-c0h9pkx0-compute",
            "lab-laboratory-20260821-a1-77aa0011-worker-zz00yy11-tpu",
            "<cluster>",
        ),
    ],
)
def test_signature_normalizes_lab_ids_and_cluster_names(
    first: str, second: str, placeholder: str
) -> None:
    """The next run has a different id. A signature that keeps any of it can never recur, so
    `lab report` gets one triage row per failure and a note filed on it never fires again."""
    for template in (
        "unknown job id '{}' — not in local runs/",
        "teardown of {} did not confirm",
        "{}: sky.down raised ClusterNotUpError",
    ):
        a = signature({"type": "Exit", "message": template.format(first)})
        b = signature({"type": "Exit", "message": template.format(second)})
        assert a == b, (a, b)
        assert placeholder in a, a


def test_signature_still_tells_a_job_id_from_ordinary_numbers() -> None:
    """Normalising ids must not collapse the message around them: two different failures about
    the same job stay two signatures."""
    job = "20261119-102456-7486e1"
    a = signature({"type": "Exit", "message": f"unknown job id '{job}'"})
    b = signature({"type": "Exit", "message": f"job '{job}' is still running"})
    assert a != b
    assert signature({"type": "Exit", "message": f"unknown job id '{job}'"}) == (
        "Exit: unknown job id '<job>'"
    )


@pytest.mark.parametrize(
    ("legacy", "message"),
    [
        # Signatures the pre-fix normaliser produced (and notes were filed under), each next to
        # a message the same failure produces for a *different* job today.
        (
            "Exit: unknown job id '<sha>-<n>-7486e1' — not in local runs/",
            "unknown job id '20261120-081512-a0b1c2' — not in local runs/",
        ),
        (
            "Exit: unknown job id '<sha>-<n>-<n>' — not in local runs/",
            "unknown job id '20261120-081512-a0b1c2' — not in local runs/",
        ),
        (
            "Exit: teardown of lab-proj-<sha>-<n>-7486e1 did not confirm",
            "teardown of lab-proj-20261120-081512-a0b1c2 did not confirm",
        ),
        (
            "Exit: sweep sweep-<sha>-<n>-b2ca49 has no plan",
            "sweep sweep-20261121-000000-ffee00 has no plan",
        ),
        (
            "Exit: lab-proj-<sha>-<n>-7486e1-head-<sha>-compute is gone",
            "lab-proj-20261120-081512-a0b1c2-head-c0h9pkx0-compute is gone",
        ),
        (
            "Exit: lab-laboratory-<sha>-ef-<sha>-head-c0h9pkx0-compute is gone",
            "lab-laboratory-20260821-a1-77aa0011-head-zz00yy11-compute is gone",
        ),
        (
            "Exit: registration reg-<sha>-<n>-<n> not found",
            "registration reg-20261121-000000-ffee00 not found",
        ),
    ],
)
def test_a_legacy_signature_upgrades_onto_todays(legacy: str, message: str) -> None:
    """Notes store the signature they were filed under. Upgrading the stored key — rather than
    leaving it — is what lets a note written against the old normaliser fire for the next job."""
    from lab.events.stats import upgrade_signature

    current = signature({"type": "Exit", "message": message})
    assert upgrade_signature(legacy) == current
    assert upgrade_signature(current) == current  # idempotent on today's form


def test_upgrading_leaves_a_signature_with_no_ids_alone() -> None:
    from lab.events.stats import upgrade_signature

    sig = signature({"type": "ProvisionTimeout", "message": "host never reached UP in 20m"})
    assert upgrade_signature(sig) == sig
    assert upgrade_signature("unknown") == "unknown"


def test_one_command_reached_from_both_surfaces_is_one_action_row() -> None:
    """The CLI records `sweep-aggregate` (and a group leaf as `queue list`); the MCP tool of the
    same name records `sweep_aggregate` / `queue_list`. Same command, same row — split, each
    half under-reports the failure rate and the failure looks half as common as it is."""
    err = {"type": "LabError", "message": "no shard plan"}
    events = [
        Event(id="1", ts=NOW, session="s", seq=0, surface="cli", action="sweep-aggregate",
              outcome="error", duration_ms=10, error=err),
        Event(id="2", ts=NOW, session="s", seq=1, surface="mcp", action="sweep_aggregate",
              outcome="ok", duration_ms=30),
        Event(id="3", ts=NOW, session="s", seq=2, surface="cli", action="queue list",
              outcome="ok", duration_ms=5),
        Event(id="4", ts=NOW, session="s", seq=3, surface="mcp", action="queue_list",
              outcome="ok", duration_ms=5),
        Event(id="5", ts=NOW, session="s", seq=4, surface="mcp", action="sweep_aggregate",
              outcome="error", duration_ms=20, error=err),
    ]
    view = stats(events)
    rows = {a.action: a for a in view.actions}
    # The CLI's spelling labels the row: it is what a person types to repeat the command.
    assert set(rows) == {"sweep-aggregate", "queue list"}
    assert rows["sweep-aggregate"].calls == 3 and rows["sweep-aggregate"].failures == 2
    assert rows["queue list"].calls == 2
    assert view.signatures[0].actions == ["sweep-aggregate"]
    # The raw records are untouched: grouping is a read-side concern.
    assert [e.action for e in events][:2] == ["sweep-aggregate", "sweep_aggregate"]
    assert [e.surface for e in events][:2] == ["cli", "mcp"]


def test_an_action_seen_only_over_mcp_keeps_its_mcp_spelling() -> None:
    view = stats([Event(id="1", ts=NOW, session="s", seq=0, surface="mcp",
                        action="fetch_artifacts", outcome="ok", duration_ms=1)])
    assert [a.action for a in view.actions] == ["fetch_artifacts"]


def test_stats_counts_calls_failures_and_failure_rate() -> None:
    view = stats([_event("1"), _event("2", outcome="error"), _event("3", action="doctor")])
    submit = next(a for a in view.actions if a.action == "submit")
    assert submit.calls == 2 and submit.failures == 1 and submit.failure_rate == 0.5
    assert view.total == 3 and view.failures == 1


def test_stats_counts_dangling_opens_as_failures() -> None:
    view = stats([_event("1", outcome=None)])
    assert view.dangling == 1 and view.failures == 1


def test_stats_sums_dollars_burned_in_failed_calls_only() -> None:
    view = stats([_event("1", outcome="error", usd=0.29), _event("2", outcome="ok", usd=5.0)])
    assert view.usd_burned == 0.29


def test_stats_treats_a_non_numeric_cost_usd_as_absent_on_both_aggregation_paths() -> None:
    # `result` comes straight off disk and can hold anything JSON can express. A malformed
    # `cost_usd` must degrade to "no cost" rather than crash `stats()` — and must degrade the
    # same way whether it's summed into the view-level total or the per-signature total.
    err = {"type": "ProvisionTimeout", "message": "host never reached UP in 20m"}
    bad = Event(id="1", ts=NOW, session="s", seq=0, surface="cli", action="submit",
                outcome="error", duration_ms=1000, error=err, result={"cost_usd": "oops"})
    view = stats([bad])
    assert view.usd_burned == 0.0
    assert view.signatures[0].usd == 0.0


def test_stats_ranks_signatures_by_count_and_records_the_window_seen() -> None:
    err = {"type": "ProvisionTimeout", "message": "host never reached UP in 20m"}
    other = {"type": "TeardownFailed", "message": "sky.down exhausted 3 retries"}
    events = [_event(str(i), outcome="error", error=err, ts=NOW - timedelta(hours=i))
              for i in range(3)]
    events.append(_event("x", outcome="error", error=other))
    view = stats(events)
    assert view.signatures[0].count == 3
    assert view.signatures[0].first_seen < view.signatures[0].last_seen
    assert view.signatures[0].actions == ["submit"]


def test_median_duration_is_reported_per_action() -> None:
    # Skewed on purpose: mean of [1, 2, 1000] is ~334, median is 2 — an implementation that
    # accidentally used `statistics.mean` would pass a symmetric fixture unchanged, so the
    # fixture has to make mean and median diverge sharply to prove which one is computed.
    view = stats([_event("1", ms=1), _event("2", ms=1000), _event("3", ms=2)])
    assert view.actions[0].median_ms == 2


def test_median_duration_averages_the_middle_pair_for_an_even_count() -> None:
    # statistics.median averages the two middle values when there's an even number of samples;
    # [1, 3, 5, 1000] sorts to the same list, and the middle pair (3, 5) averages to 4 — nowhere
    # close to the mean (~252), so this is pinning the even-count averaging behaviour specifically.
    view = stats([_event(str(i), ms=ms) for i, ms in enumerate([1, 3, 5, 1000])])
    assert view.actions[0].median_ms == 4


def test_stats_dict_shapes_the_view_as_json_ready_data() -> None:
    err = {"type": "ProvisionTimeout", "message": "host never reached UP in 20m"}
    view = stats([_event("1", outcome="error", error=err)], since=NOW)
    d = stats_dict(view)
    assert d["since"] == NOW.isoformat()
    assert d["total"] == 1 and d["failures"] == 1
    assert d["actions"][0]["action"] == "submit"
    sig = d["signatures"][0]
    assert sig["first_seen"] == NOW.isoformat() and sig["last_seen"] == NOW.isoformat()
    assert isinstance(sig["signature"], str)
