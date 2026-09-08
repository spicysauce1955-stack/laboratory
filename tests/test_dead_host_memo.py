"""The dead-offer memo and accelerator rotation (lab.placement).

Hermetic: the catalog is faked, nothing here touches a cloud, a credential, or the network.

The numbers quoted throughout were measured on 2026-09-08 over the 1,502 run directories in
``~/.superset/projects/tempotron-capacity/runs`` (the campaign this feature comes from), by
pairing each manifest's ``end_reason`` with the ``⚙︎ Launching on …`` line in its ``logs.txt``:

    vast launches with a resolvable placement          n=571
    of those, died at the provisioning watchdog        240 (42%)

    Maryland, US, NA  / 1x-RTX_4090-32-65536   n=432   160 dead (37%)
    Romania, RO, EU   / 1x-RTX_3090-32-65536   n=114    63 dead (55%)
    , CA, NA          / 1x-RTX_5090-32-65536   n= 20    15 dead (75%)
    Sweden, SE, EU    / 1x-RTX_4090-32-65536   n=  5     2 dead

Those numbers are the reason the memo below *records* a placement but never excludes one.
"""

from __future__ import annotations

import json
import types

import pytest

from lab import placement as P
from lab.models import ResourceRequest


# --------------------------------------------------------------------------------------------
# The real log of a real dead provision. Verbatim from
# ~/.superset/projects/tempotron-capacity/runs/20260903-200140-f01cf6/logs.txt — a job whose
# manifest reads "provisioning exceeded 240s". ANSI codes stripped; nothing else changed.
#
# This is the whole evidence base for the identity question: it is everything the lab saw about
# the host that took its 240s. There is no instance id, no machine id, no host id in it.
# --------------------------------------------------------------------------------------------
REAL_DEAD_PROVISION_LOG = """\
2026-09-03T20:01:41.765Z Running on cluster: lab-tempotr-5171-20260903-200140-f01cf6
2026-09-03T20:01:42.246Z Considered resources (1 node):
2026-09-03T20:01:42.246Z -------------------------------------------------------------------
2026-09-03T20:01:42.246Z  INFRA                    INSTANCE               vCPUs   Mem(GB)   \
GPUS        COST ($)   CHOSEN
2026-09-03T20:01:42.246Z -------------------------------------------------------------------
2026-09-03T20:01:42.246Z  Vast (Romania, RO, EU)   1x-RTX_3090-32-65536   32      64        \
RTX3090:1   0.25          ✔
2026-09-03T20:01:42.246Z -------------------------------------------------------------------
2026-09-03T20:01:42.548Z ⚙︎ Launching on Vast Romania, RO, EU.
2026-09-03T20:05:42.139Z Cancelling 1 request: '9ce01952-a1ca-4747-a88e-f766cd7112fd'...
2026-09-03T20:05:42.681Z Request 9ce01952-a1ca-4747-a88e-f766cd7112fd cancelled by user
2026-09-03T20:05:42.983Z 'sky.launch' request 9ce01952-a1ca-4747-a88e-f766cd7112fd cancelled
"""

# A GCP failover log names a zone per attempt; the last one is the one that was being waited on.
REAL_GCP_LAUNCH_LOG = "⚙️ Launching on GCP us-central1 (us-central1-a).\n"


# --------------------------------------------------------------------------------------------
# What the lab can actually key on
# --------------------------------------------------------------------------------------------


class TestPlacementParsing:
    def test_reads_the_instance_type_and_region_a_real_dead_launch_named(self):
        drawn = P.parse_drawn_placement(REAL_DEAD_PROVISION_LOG)
        assert drawn == P.DrawnPlacement(
            instance_type="1x-RTX_3090-32-65536", region="Romania, RO, EU"
        )

    def test_a_log_with_nothing_in_it_yields_nothing(self):
        assert P.parse_drawn_placement("") == P.DrawnPlacement(None, None)
        assert P.parse_drawn_placement("no launch here\n") == P.DrawnPlacement(None, None)

    def test_the_last_launching_on_line_wins(self):
        """SkyPilot prints one per failover hop; the one being waited on when we gave up is the
        last, not the first."""
        text = REAL_GCP_LAUNCH_LOG + "⚙️ Launching on GCP us-east1 (us-east1-b).\n"
        assert P.parse_drawn_placement(text).region == "us-east1 (us-east1-b)"

    def test_a_region_without_a_considered_table_still_parses(self):
        drawn = P.parse_drawn_placement(REAL_GCP_LAUNCH_LOG)
        assert drawn.region == "us-central1 (us-central1-a)"
        assert drawn.instance_type is None


class TestKeys:
    def test_a_machine_key_names_one_physical_host(self):
        assert P.machine_key("vast", 12345) == "vast|machine:12345"
        assert P.machine_key("vast", "12345") == "vast|machine:12345"

    def test_a_placement_key_is_coarse_and_says_so_in_its_kind(self):
        assert (
            P.placement_key("vast", "1x-RTX_3090-32-65536", "Romania, RO, EU")
            == "vast|placement:1x-RTX_3090-32-65536@Romania, RO, EU"
        )

    def test_price_is_never_part_of_a_key(self):
        """The campaign's own workaround keyed on price ("skip the $0.3585 host") and its log
        flags that as fragile: good hosts recur at the same price as bad ones."""
        key = P.placement_key("vast", "1x-RTX_3090-32-65536", "Romania, RO, EU")
        assert "0.25" not in key and "$" not in key

    def test_a_missing_half_is_recorded_as_unknown_not_invented(self):
        assert P.placement_key("gcp", None, "us-central1") == "gcp|placement:?@us-central1"
        assert P.placement_key("gcp", "n1-highmem-4", None) == "gcp|placement:n1-highmem-4@?"


# --------------------------------------------------------------------------------------------
# The memo itself
# --------------------------------------------------------------------------------------------


class TestDeadHostMemo:
    def test_a_failed_provision_records_the_offender(self, tmp_path):
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=100.0, strikes=2)
        key = P.machine_key("vast", 12345)
        assert memo.strikes(key, now_s=1000.0) == 0
        memo.record(key, now_s=1000.0)
        assert memo.strikes(key, now_s=1000.0) == 1

    def test_one_strike_is_not_enough_to_call_a_host_dead(self, tmp_path):
        """A first failure is indistinguishable from bad luck, and the campaign data says a
        placement that just failed still succeeds ~half the time."""
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=100.0, strikes=2)
        key = P.machine_key("vast", 12345)
        memo.record(key, now_s=1000.0)
        assert memo.is_dead(key, now_s=1000.0) is False
        memo.record(key, now_s=1001.0)
        assert memo.is_dead(key, now_s=1001.0) is True

    def test_the_next_placement_skips_a_struck_out_host(self, tmp_path):
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=100.0, strikes=2)
        dead, alive = P.machine_key("vast", 1), P.machine_key("vast", 2)
        memo.record(dead, now_s=1000.0)
        memo.record(dead, now_s=1001.0)
        memo.record(alive, now_s=1001.0)
        assert memo.dead_keys("vast", now_s=1002.0) == {dead}
        assert memo.is_dead(alive, now_s=1002.0) is False

    def test_an_entry_expires_after_its_ttl(self, tmp_path):
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=100.0, strikes=2)
        key = P.machine_key("vast", 12345)
        memo.record(key, now_s=1000.0)
        memo.record(key, now_s=1001.0)
        assert memo.is_dead(key, now_s=1050.0) is True
        assert memo.is_dead(key, now_s=1200.0) is False  # past the TTL, forgotten
        assert memo.strikes(key, now_s=1200.0) == 0

    def test_strikes_expire_one_at_a_time_not_all_at_once(self, tmp_path):
        """Each failure carries its own timestamp, so a host that failed twelve hours ago and
        once just now is one strike from dead, not two."""
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=100.0, strikes=2)
        key = P.machine_key("vast", 12345)
        memo.record(key, now_s=1000.0)
        memo.record(key, now_s=1150.0)
        assert memo.strikes(key, now_s=1160.0) == 1

    def test_the_ttl_comes_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAB_DEAD_HOST_TTL_S", "7")
        assert P.DeadHostMemo.for_home(tmp_path).ttl_s == 7.0

    def test_a_garbage_ttl_in_the_environment_falls_back_to_the_default(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LAB_DEAD_HOST_TTL_S", "soon")
        assert P.DeadHostMemo.for_home(tmp_path).ttl_s == P.DEFAULT_DEAD_HOST_TTL_S

    def test_the_strike_threshold_comes_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAB_DEAD_HOST_STRIKES", "1")
        memo = P.DeadHostMemo.for_home(tmp_path)
        key = P.machine_key("vast", 1)
        memo.record(key, now_s=1000.0)
        assert memo.is_dead(key, now_s=1000.0) is True

    def test_a_threshold_below_one_is_ignored(self, tmp_path, monkeypatch):
        """`LAB_DEAD_HOST_STRIKES=0` would make every host dead on sight, including hosts that
        have never failed. A memo may slow a launch; it may not blacklist the world."""
        monkeypatch.setenv("LAB_DEAD_HOST_STRIKES", "0")
        memo = P.DeadHostMemo.for_home(tmp_path)
        assert memo.strike_limit >= 1
        assert memo.is_dead(P.machine_key("vast", 1), now_s=1000.0) is False


class TestTheMemoIsAdvisory:
    """A launch blocked by a bad memo would be a far worse bug than the one this fixes."""

    def test_a_corrupt_memo_reads_as_empty(self, tmp_path):
        path = tmp_path / P.DeadHostMemo.FILENAME
        path.write_text("{not json at all")
        memo = P.DeadHostMemo(path)
        assert memo.strikes(P.machine_key("vast", 1)) == 0
        assert memo.is_dead(P.machine_key("vast", 1)) is False
        assert memo.dead_keys("vast") == set()
        assert memo.has_any("vast") is False

    def test_a_memo_of_the_wrong_shape_reads_as_empty(self, tmp_path):
        path = tmp_path / P.DeadHostMemo.FILENAME
        path.write_text(json.dumps({"entries": ["not", "a", "dict"]}))
        assert P.DeadHostMemo(path).strikes("vast|machine:1") == 0

    def test_unparseable_timestamps_are_dropped_not_fatal(self, tmp_path):
        path = tmp_path / P.DeadHostMemo.FILENAME
        path.write_text(
            json.dumps({"version": 1, "entries": {"vast|machine:1": ["soon", 1000.0]}})
        )
        assert P.DeadHostMemo(path, ttl_s=1e9).strikes("vast|machine:1", now_s=1000.0) == 1

    def test_a_missing_memo_reads_as_empty(self, tmp_path):
        memo = P.DeadHostMemo.for_home(tmp_path / "never-created")
        assert memo.dead_keys("vast") == set()

    def test_a_write_to_an_unwritable_path_does_not_raise(self, tmp_path):
        memo = P.DeadHostMemo(tmp_path / "nope" / "\0bad" / "memo.json")
        memo.record(P.machine_key("vast", 1))  # must not raise

    def test_a_write_that_fails_is_reported_on_stderr_only(self, tmp_path, capsys):
        memo = P.DeadHostMemo(tmp_path / "nope" / "\0bad" / "memo.json")
        memo.record(P.machine_key("vast", 1))
        out = capsys.readouterr()
        assert out.out == ""  # stdout is a JSON channel
        assert "dead-host memo" in out.err

    def test_the_recorded_host_survives_the_ledger_sanitizer(self, tmp_path, monkeypatch):
        """`lab.events.sanitize` is a deny-list that masks a param literally named ``key`` as a
        probable secret, so the note records ``host=``. If that regresses, every dead-host record
        in the ledger reads ``…REDACTED…`` and the evidence this whole feature collects is
        unreadable in `lab history` / `lab report`."""
        from lab import events
        from lab.events.sanitize import sanitize_params

        seen: list[dict[str, object]] = []
        monkeypatch.setattr(events, "note", lambda kind, **kw: seen.append({"kind": kind, **kw}))
        P.DeadHostMemo.for_home(tmp_path).record(P.machine_key("vast", 41234))

        assert seen and seen[0]["kind"] == "placement.dead_host"
        assert "key" not in seen[0]
        assert sanitize_params(seen[0])["host"] == "vast|machine:41234"

    def test_expired_entries_are_pruned_on_write(self, tmp_path):
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=100.0)
        memo.record(P.machine_key("vast", "old"), now_s=1000.0)
        memo.record(P.machine_key("vast", "new"), now_s=2000.0)
        entries = json.loads((tmp_path / P.DeadHostMemo.FILENAME).read_text())["entries"]
        assert list(entries) == ["vast|machine:new"]

    def test_a_single_key_cannot_grow_without_bound(self, tmp_path):
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=1e9)
        key = P.machine_key("vast", 1)
        for i in range(P.MAX_STRIKES_KEPT * 3):
            memo.record(key, now_s=1000.0 + i)
        stored = json.loads((tmp_path / P.DeadHostMemo.FILENAME).read_text())["entries"][key]
        assert len(stored) == P.MAX_STRIKES_KEPT
        assert memo.strikes(key, now_s=1000.0) == P.MAX_STRIKES_KEPT

    def test_the_memo_never_grows_past_its_key_cap(self, tmp_path):
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=1e9)
        for i in range(P.MAX_MEMO_KEYS + 25):
            memo.record(P.machine_key("vast", i), now_s=1000.0 + i)
        entries = json.loads((tmp_path / P.DeadHostMemo.FILENAME).read_text())["entries"]
        assert len(entries) == P.MAX_MEMO_KEYS
        assert P.machine_key("vast", 0) not in entries  # oldest evicted first


class TestCoarseKeysAreEvidenceNotAVerdict:
    """The measurement that decided this, in the file's docstring: 42% of the campaign's Vast
    launches died provisioning, and the region+instance-type key barely beats that base rate
    (48% after three strikes in 30 min). Worse, the placement with the *most* failures — Maryland
    4090, 160 of them — is also the one with the best success rate (37% dead against Romania's
    55% and CA's 75%). Excluding it would have steered the campaign into its worse supply.
    """

    def test_a_struck_out_placement_key_does_not_exclude_a_region(self, tmp_path, pool_catalog):
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=1e9, strikes=1)
        key = P.placement_key("vast", "1x-RTX4090", "a")
        for _ in range(10):
            memo.record(key)
        assert memo.is_dead(key) is True

        names = [c.region for c in P.candidates(_vast(), instance_type="1x-RTX4090")]
        assert names == ["a", "b"]  # candidates() does not consult this memo at all

    def test_a_dead_machine_is_not_confused_with_its_placement(self, tmp_path):
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=1e9, strikes=1)
        memo.record(P.machine_key("vast", 12345))
        assert memo.dead_keys("vast", kind="machine") == {"vast|machine:12345"}
        assert memo.dead_keys("vast", kind="placement") == set()


# --------------------------------------------------------------------------------------------
# Accelerator rotation
# --------------------------------------------------------------------------------------------


_POOL_PRICES = {"RTX4090:1": 0.40, "RTX3090:1": 0.25, "L40S:1": 1.10}


def _fake_catalog(prices: dict[str, float] | None = None) -> types.SimpleNamespace:
    """Prices per accelerator spec, flat across two regions. Enough to test the cap rule."""
    table = _POOL_PRICES if prices is None else prices

    def get_instance_type_for_accelerator(name, count, **kw):
        key = f"{name}:{count}"
        return ([f"1x-{name}"], []) if key in table else ([], [])

    def get_hourly_cost(instance_type, use_spot, region, zone, clouds=None):
        for spec, price in table.items():
            if instance_type == f"1x-{spec.split(':')[0]}":
                return price
        raise ValueError(f"no price for {instance_type}")

    def region(name):
        return types.SimpleNamespace(name=name, zones=[])

    return types.SimpleNamespace(
        get_default_instance_type=lambda **kw: None,
        get_instance_type_for_accelerator=get_instance_type_for_accelerator,
        get_region_zones_for_instance_type=lambda *a, **kw: [region("a"), region("b")],
        get_region_zones_for_accelerators=lambda *a, **kw: [region("a"), region("b")],
        get_hourly_cost=get_hourly_cost,
        get_accelerator_hourly_cost=lambda *a, **kw: 0.0,
        validate_region_zone=lambda r, z, clouds=None: (r, z),
    )


@pytest.fixture
def pool_catalog(monkeypatch):
    cat = _fake_catalog()
    monkeypatch.setattr(P, "_catalog", lambda: cat)
    return cat


def _vast(**kw) -> ResourceRequest:
    return ResourceRequest(cloud="vast", accelerators="RTX4090:1", **kw)


class TestParsePool:
    def test_a_comma_list_becomes_a_pool(self):
        assert P.parse_pool("RTX4090:1,RTX3090:1") == ["RTX4090:1", "RTX3090:1"]

    def test_whitespace_and_blanks_are_forgiven(self):
        assert P.parse_pool(" RTX4090:1 , ,RTX3090:1 ") == ["RTX4090:1", "RTX3090:1"]

    def test_duplicates_collapse_so_an_attempt_is_never_spent_twice(self):
        assert P.parse_pool("A:1,B:1,A:1") == ["A:1", "B:1"]

    def test_nothing_means_nothing(self):
        assert P.parse_pool(None) == []
        assert P.parse_pool("") == []
        assert P.parse_pool([]) == []

    def test_a_list_is_accepted_as_given(self):
        assert P.parse_pool(["A:1", "B:1"]) == ["A:1", "B:1"]


class TestRotationAdvancesOnlyOnTheRightFailures:
    def test_it_returns_the_next_untried_entry(self, pool_catalog):
        pool = ["RTX4090:1", "RTX3090:1", "L40S:1"]
        res = _vast()
        assert P.next_accelerator(pool=pool, tried=["RTX4090:1"], res=res) == "RTX3090:1"
        assert (
            P.next_accelerator(pool=pool, tried=["RTX4090:1", "RTX3090:1"], res=res) == "L40S:1"
        )

    def test_an_exhausted_pool_returns_none_rather_than_cycling(self, pool_catalog):
        """A bound that wraps around is not a bound. The caller stops here and reports the
        failure it already has."""
        pool = ["RTX4090:1", "RTX3090:1"]
        assert P.next_accelerator(pool=pool, tried=pool, res=_vast()) is None

    def test_an_empty_pool_never_rotates(self, pool_catalog):
        assert P.next_accelerator(pool=[], tried=[], res=_vast()) is None

    def test_a_known_dead_entry_is_deferred_not_dropped(self, pool_catalog, tmp_path):
        """Skipped while a healthy alternative exists…"""
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=1e9, strikes=1)
        memo.record(P.accelerator_key("vast", "RTX3090:1"))
        pool = ["RTX4090:1", "RTX3090:1", "L40S:1"]
        assert (
            P.next_accelerator(pool=pool, tried=["RTX4090:1"], res=_vast(), memo=memo)
            == "L40S:1"
        )

    def test_but_a_pool_that_is_entirely_dead_still_yields_a_candidate(
        self, pool_catalog, tmp_path
    ):
        """…and never deadlocks: a memo that would exclude everything is ignored, exactly as
        `narrowed_regions` ignores a capacity memo that excludes every region."""
        memo = P.DeadHostMemo.for_home(tmp_path, ttl_s=1e9, strikes=1)
        for spec in ("RTX4090:1", "RTX3090:1", "L40S:1"):
            memo.record(P.accelerator_key("vast", spec))
        got = P.next_accelerator(
            pool=["RTX4090:1", "RTX3090:1", "L40S:1"], tried=["RTX4090:1"], res=_vast(), memo=memo
        )
        assert got == "RTX3090:1"  # the first untried entry, in the user's own order


class TestRotationNeverRaisesAPriceCap:
    def test_an_entry_that_cannot_fit_under_the_cap_is_skipped(self, pool_catalog):
        """L40S is $1.10 in every region the catalog knows; under a $0.50 cap it can never be
        launched, so spending an attempt on it would only buy a slower failure."""
        res = _vast(max_hourly_usd=0.50)
        assert P.next_accelerator(pool=["RTX4090:1", "L40S:1"], tried=["RTX4090:1"], res=res) is None

    def test_the_cap_itself_is_never_touched(self, pool_catalog):
        res = _vast(max_hourly_usd=0.50)
        P.next_accelerator(pool=["RTX4090:1", "RTX3090:1"], tried=["RTX4090:1"], res=res)
        assert res.max_hourly_usd == 0.50

    def test_the_rotated_spec_inherits_the_cap_unchanged(self, pool_catalog):
        res = _vast(max_hourly_usd=0.50)
        rotated = P.rotated_resources(res, "RTX3090:1")
        assert rotated.accelerators == "RTX3090:1"
        assert rotated.max_hourly_usd == 0.50
        assert rotated.cloud == "vast"

    def test_an_unpriceable_entry_is_not_blocked_by_the_cap(self, monkeypatch):
        """A check that cannot answer never blocks (the `lab doctor` rule). An accelerator the
        catalog has never heard of is offered to SkyPilot, which enforces the cap itself."""
        monkeypatch.setattr(P, "_catalog", lambda: _fake_catalog({}))
        res = _vast(max_hourly_usd=0.01)
        assert (
            P.next_accelerator(pool=["RTX4090:1", "MYSTERY:1"], tried=["RTX4090:1"], res=res)
            == "MYSTERY:1"
        )

    def test_no_cap_admits_everything(self, pool_catalog):
        res = _vast()
        assert (
            P.next_accelerator(pool=["RTX4090:1", "L40S:1"], tried=["RTX4090:1"], res=res)
            == "L40S:1"
        )

    def test_affordability_is_judged_on_the_floor_not_the_ceiling(self, pool_catalog):
        """The module's other rule is "guardrails check the top of the band" — this is the one
        place the opposite polarity is correct. The question here is *feasibility* ("could this
        ever launch under the cap?"), and refusing an accelerator whose cheapest region fits
        would be the memo blocking a launch the cap allows."""
        cat = _fake_catalog({"A:1": 0.10})
        cat.get_hourly_cost = lambda instance_type, use_spot, region, zone, clouds=None: (
            0.10 if region == "a" else 9.99
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(P, "_catalog", lambda: cat)
            assert P.affordable_under_cap(_vast(max_hourly_usd=0.5), "A:1") is True
