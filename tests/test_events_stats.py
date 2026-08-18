from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
