"""Scheduler host / client lab-version skew — the pure predicate (no queue, no network).

The always-on host is deployed independently of every project's venv and has been observed running
a pre-v0.5.0 lab against v0.11.0 clients, silently dropping registration fields it did not know
(`--price-cap`). These tests pin the three answers that matters: older (dangerous), newer, and
cannot-tell — the last of which must never collapse into "fine".
"""

from lab.scheduler.skew import (
    HEARTBEAT_VERSION_KEY,
    classify_skew,
    skew_from_heartbeat,
)


class TestClassify:
    def test_same_version_is_not_skew(self):
        s = classify_skew("0.11.0", "0.11.0")
        assert s.verdict == "same"
        assert s.may_drop_fields is False
        assert "0.11.0" in s.detail

    def test_older_host_is_the_dangerous_direction(self):
        s = classify_skew("0.4.2", "0.11.0")
        assert s.verdict == "host_older"
        assert s.may_drop_fields is True
        # The warning has to name the consequence and the remedy, or it is just trivia.
        assert "OLDER" in s.detail and "--price-cap" in s.detail
        assert "deploy/scheduler/deploy.sh v0.11.0" in s.detail

    def test_newer_host_is_reported_but_is_the_other_problem(self):
        s = classify_skew("0.12.0", "0.11.0")
        assert s.verdict == "host_newer"
        assert s.may_drop_fields is False
        assert "NEWER" in s.detail

    def test_patch_releases_are_compared_too(self):
        # lab ships field-bearing changes in patch releases (v0.7.1), so a major.minor-only
        # compare would call a host that is genuinely behind a match.
        assert classify_skew("0.11.0", "0.11.1").verdict == "host_older"
        assert classify_skew("0.11.2", "0.11.1").verdict == "host_newer"

    def test_comparison_is_numeric_not_lexicographic(self):
        # "0.9.0" > "0.11.0" as strings; the host is older, not newer.
        assert classify_skew("0.9.0", "0.11.0").verdict == "host_older"
        assert classify_skew("0.11.0", "0.9.0").verdict == "host_newer"
        assert classify_skew("1.2.0", "0.99.0").verdict == "host_newer"

    def test_decoration_around_a_real_version_still_parses(self):
        assert classify_skew("v0.11.0", "0.11.0").verdict == "same"
        assert classify_skew("0.11.0.dev3", "0.11.0").verdict == "same"
        assert classify_skew("0.11.0+g1234abc", "0.12.0").verdict == "host_older"

    def test_unparseable_host_version_cannot_tell(self):
        s = classify_skew("nightly", "0.11.0")
        assert s.verdict == "unknown"
        assert s.may_drop_fields is True  # cannot rule out the dangerous direction
        assert "'nightly'" in s.detail
        assert s.host_version == "nightly"  # echoed back: this is how "unparseable" is told from

    def test_unparseable_client_version_cannot_tell(self):
        # A source tree with no installed metadata reports lab.__version__ == "0.0.0+unknown",
        # which parses as 0.0.0 and would otherwise fabricate "the host is newer".
        for client in ("0.0.0+unknown", "", None, "banana"):
            s = classify_skew("0.11.0", client)
            assert s.verdict == "unknown", client

    def test_never_raises_on_anything(self):
        for host, client in (
            (None, None),
            ("", ""),
            ("...", "0.11.0"),
            ("0.11", "0.11.0"),  # two components only: not a version this compares
            ("a.b.c", "0.11.0"),
            ("999999999999999999999.0.0", "0.11.0"),
        ):
            assert classify_skew(host, client).verdict in (
                "same", "host_older", "host_newer", "unknown"
            )
        assert classify_skew("0.11", "0.11.0").verdict == "unknown"
        assert classify_skew("999999999999999999999.0.0", "0.11.0").verdict == "host_newer"


class TestFromHeartbeat:
    def test_reads_the_version_the_tick_writes(self):
        hb = {"at": "2026-09-08T00:00:00+00:00", HEARTBEAT_VERSION_KEY: "0.4.0"}
        assert skew_from_heartbeat(hb, "0.11.0").verdict == "host_older"

    def test_heartbeat_without_the_field_is_cannot_tell_but_smells_old(self):
        """An old host predates the field entirely — the absence IS the evidence."""
        hb = {"at": "2026-09-08T00:00:00+00:00", "host": "sched-1", "tick_count": 42}
        s = skew_from_heartbeat(hb, "0.11.0")
        assert s.verdict == "unknown"
        assert s.host_version is None  # distinguishes "missing" from "unparseable"
        assert s.may_drop_fields is True
        assert "no lab_version" in s.detail and "older" in s.detail
        assert "deploy/scheduler/deploy.sh v0.11.0" in s.detail

    def test_heartbeat_with_an_unparseable_field(self):
        s = skew_from_heartbeat({HEARTBEAT_VERSION_KEY: "unknown"}, "0.11.0")
        assert s.verdict == "unknown"
        assert s.host_version == "unknown"
        assert "could not compare" in s.detail

    def test_no_heartbeat_at_all(self):
        s = skew_from_heartbeat(None, "0.11.0")
        assert s.verdict == "unknown" and s.host_version is None

    def test_a_heartbeat_that_is_not_a_mapping_never_raises(self):
        # heartbeat.json is external JSON; a truncated or hand-edited one can be anything.
        for junk in ([], "0.11.0", 7):
            assert skew_from_heartbeat(junk, "0.11.0").verdict == "unknown"  # type: ignore[arg-type]

    def test_non_string_version_field(self):
        assert skew_from_heartbeat({HEARTBEAT_VERSION_KEY: 0.11}, "0.11.0").verdict == "unknown"
        assert skew_from_heartbeat({HEARTBEAT_VERSION_KEY: None}, "0.11.0").host_version is None

    def test_client_defaults_to_this_process(self):
        from lab import __version__

        s = skew_from_heartbeat({HEARTBEAT_VERSION_KEY: __version__})
        # In an installed tree that is "same"; in a bare source tree lab.__version__ is
        # "0.0.0+unknown" on BOTH sides, which is unknown -- never a fabricated verdict.
        assert s.verdict in ("same", "unknown")
        assert s.client_version == __version__
