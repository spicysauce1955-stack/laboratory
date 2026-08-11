"""Placement + pricing (lab.placement).

Everything here is hermetic: the catalog is faked, so these run with no credentials and no
network. The one test that reaches the real catalog is marked and skips when sky is absent.
"""

from __future__ import annotations

import json
import types

import pytest

from lab import placement as P
from lab.models import ResourceRequest


# --------------------------------------------------------------------------------------------
# A fake sky.catalog. Prices are shaped like GCP's real ones: on-demand nearly flat, spot with a
# wide regional spread, because that spread is the thing the design exists to handle.
# --------------------------------------------------------------------------------------------

_ONDEMAND = {"us-central1": 0.18, "us-east1": 0.18, "europe-west1": 0.20, "asia-east2": 0.29}
_SPOT = {"us-central1": 0.12, "us-east1": 0.10, "europe-west1": 0.034, "asia-east2": 0.05}
_ZONES = {
    "us-central1": ["us-central1-a", "us-central1-b", "us-central1-c"],
    "us-east1": ["us-east1-b", "us-east1-c"],
    "europe-west1": ["europe-west1-b", "europe-west1-c"],
    "asia-east2": ["asia-east2-a"],
}


def _region(name: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        name=name, zones=[types.SimpleNamespace(name=z) for z in _ZONES[name]]
    )


def fake_catalog(*, accel_price: float = 0.0) -> types.SimpleNamespace:
    def get_hourly_cost(instance_type, use_spot, region, zone, clouds=None):
        table = _SPOT if use_spot else _ONDEMAND
        if region not in table:
            raise ValueError(f"no price for {region}")
        return table[region]

    def get_accelerator_hourly_cost(name, count, use_spot, region, zone, clouds=None):
        return accel_price * count

    return types.SimpleNamespace(
        get_default_instance_type=lambda **kw: "n4-standard-4",
        get_instance_type_for_accelerator=lambda *a, **kw: (["n1-highmem-4"], []),
        get_region_zones_for_instance_type=lambda *a, **kw: [_region(r) for r in _ZONES],
        get_region_zones_for_accelerators=lambda *a, **kw: [_region(r) for r in _ZONES],
        get_hourly_cost=get_hourly_cost,
        get_accelerator_hourly_cost=get_accelerator_hourly_cost,
        validate_region_zone=lambda r, z, clouds=None: (r, z),
    )


@pytest.fixture
def catalog(monkeypatch):
    cat = fake_catalog()
    monkeypatch.setattr(P, "_catalog", lambda: cat)
    return cat


def _res(**kw) -> ResourceRequest:
    return ResourceRequest(cloud="gcp", cpus=4, disk_size=50, **kw)


# --------------------------------------------------------------------------------------------
# Price band polarity — the regression this whole module was written for
# --------------------------------------------------------------------------------------------


def test_estimate_ceiling_is_the_max_not_the_min(catalog):
    """The bug: `Resources.get_cost()` unpinned returns the cheapest region's price, and that was
    feeding admission control. A guardrail that checks the best case is not a guardrail."""
    est = P.estimate(_res(use_spot=True, spot_fallback=False))
    assert est is not None
    assert est.low_usd == pytest.approx(0.034)  # europe-west1, the cheapest
    assert est.high_usd == pytest.approx(0.12)  # us-central1, the dearest — what we check
    assert est.high_usd > est.low_usd


def test_spot_fallback_ceiling_is_priced_on_demand(catalog):
    """`--spot` defaults to falling back to on-demand, so the worst case the user has actually
    authorised is an on-demand box — ~8x the spot price they were budgeting for."""
    with_fallback = P.estimate(_res(use_spot=True, spot_fallback=True))
    spot_only = P.estimate(_res(use_spot=True, spot_fallback=False))
    assert with_fallback is not None and spot_only is not None
    assert with_fallback.low_usd == spot_only.low_usd == pytest.approx(0.034)
    assert with_fallback.high_usd == pytest.approx(0.29)  # the dearest ON-DEMAND region
    assert spot_only.high_usd == pytest.approx(0.12)
    assert "fallback" in with_fallback.basis


def test_price_cap_lowers_the_ceiling_and_narrows_the_field(catalog):
    """A cap is enforced by SkyPilot's optimizer, so it makes the worst case provable.

    The reported ceiling is the dearest *surviving* candidate, which is tighter than the cap
    itself: with a $0.06 cap only europe-west1 ($0.034) and asia-east2 ($0.05) are admissible, so
    $0.05 is the most that can actually be billed — the cap bounds it, the catalog sharpens it.
    """
    est = P.estimate(_res(use_spot=True, spot_fallback=False, max_hourly_usd=0.06))
    assert est is not None
    assert est.regions == 2
    assert est.high_usd == pytest.approx(0.05)
    assert est.high_usd <= 0.06


def test_price_cap_binds_when_it_is_below_every_candidate_ceiling(catalog):
    """When the cap sits under the dearest survivor, the cap is the ceiling."""
    est = P.estimate(_res(use_spot=True, spot_fallback=False, max_hourly_usd=0.045))
    assert est is not None
    assert est.regions == 1  # europe-west1 only
    assert est.high_usd == pytest.approx(0.034)


def test_worst_hourly_includes_storage(catalog):
    est = P.estimate(_res(use_spot=True, spot_fallback=False))
    assert est is not None
    assert est.storage_usd == pytest.approx(50 * 0.000109589)
    assert est.worst_hourly_usd == pytest.approx(est.high_usd + est.storage_usd)


def test_pinned_region_collapses_the_band(catalog):
    est = P.estimate(_res(use_spot=True, spot_fallback=False, region="europe-west1"))
    assert est is not None
    assert est.regions == 1
    assert est.low_usd == est.high_usd == pytest.approx(0.034)


def test_accelerator_price_is_added_to_the_instance(monkeypatch):
    """GCP bills GPUs as a separate SKU, so pricing a T4 box as a bare n1 under-reports it."""
    monkeypatch.setattr(P, "_catalog", lambda: fake_catalog(accel_price=0.35))
    est = P.estimate(ResourceRequest(cloud="gcp", accelerators="T4:1", disk_size=100))
    assert est is not None
    assert est.instance_type == "n1-highmem-4"
    assert est.low_usd == pytest.approx(0.18 + 0.35)


def test_unpriceable_spec_returns_none_rather_than_raising(monkeypatch):
    """A missing price must degrade the guardrail to its old behaviour, never block a launch."""
    monkeypatch.setattr(
        P, "_catalog", lambda: types.SimpleNamespace(
            get_default_instance_type=lambda **kw: None,
            get_instance_type_for_accelerator=lambda *a, **kw: ([], []),
        )
    )
    assert P.estimate(_res()) is None


def test_catalog_import_failure_is_not_an_error(monkeypatch):
    def _boom():
        raise ImportError("no skypilot extra here")

    monkeypatch.setattr(P, "_catalog", _boom)
    assert P.resolve_instance_type(_res()) is None
    assert P.estimate(_res()) is None


# --------------------------------------------------------------------------------------------
# Storage pricing
# --------------------------------------------------------------------------------------------


def test_storage_rate_follows_the_machine_family():
    """n4 supports only hyperdisk-balanced; n1 gets pd-balanced. Different rates, so the two
    profiles genuinely bill differently and guessing one rate would be wrong for the other."""
    hyper = P.storage_hourly_usd("gcp", 100, "n4-standard-4")
    pd = P.storage_hourly_usd("gcp", 100, "n1-highmem-4")
    assert hyper == pytest.approx(100 * 0.000109589)
    assert pd == pytest.approx(100 * 0.000136986)
    assert pd > hyper


@pytest.mark.parametrize(
    "instance_type,hyperdisk",
    [
        ("n4-standard-4", True),
        ("a3-ultragpu-8g", True),
        ("a4-highgpu-8g", True),
        ("n1-highmem-4", False),
        ("e2-standard-4", False),
        ("n4a-standard-4", False),  # a distinct family — a prefix match would price it 20% low
        ("a3-highgpu-8g", False),  # only a3-ultragpu is hyperdisk-only, not all of a3
        ("", False),
    ],
)
def test_hyperdisk_families_are_matched_on_whole_tokens(instance_type, hyperdisk):
    rate = P.storage_hourly_usd("gcp", 100, instance_type) / 100
    expected = 0.000109589 if hyperdisk else 0.000136986
    assert rate == pytest.approx(expected)


def test_skypilot_default_disk_roughly_doubles_a_spot_cpu_bill():
    """The number that reframed GCP-COST-2 from a leak footnote into a live distortion.

    A 256 GB boot disk is 82% of a $0.034/hr spot n4-standard-4 on hyperdisk (what n4 must use)
    and 103% of it on pd-balanced. Either way the disk is the same order as the machine, which is
    why leaving it to SkyPilot's default was never harmless.
    """
    spot_n4 = 0.034
    hyperdisk_256 = P.storage_hourly_usd("gcp", 256, "n4-standard-4")
    pd_256 = P.storage_hourly_usd("gcp", 256, "n1-highmem-4")
    assert hyperdisk_256 == pytest.approx(0.02806, abs=1e-4)
    assert pd_256 == pytest.approx(0.03507, abs=1e-4)
    assert 0.8 < hyperdisk_256 / spot_n4 < 1.0
    assert pd_256 > spot_n4

    # What the lab actually provisions instead.
    assert P.storage_hourly_usd("gcp", 50, "n4-standard-4") == pytest.approx(0.00548, abs=1e-4)
    assert P.storage_hourly_usd("gcp", 100, "n1-highmem-4") == pytest.approx(0.0137, abs=1e-4)


def test_vast_storage_is_zero_because_dph_total_already_includes_it():
    """Adding a term for Vast would double-count: we price Vast from the rental's billed rate."""
    assert P.storage_hourly_usd("vast", 100, "whatever") == 0.0
    assert P.storage_hourly_usd("do", 100, None) == pytest.approx(100 * 0.000136986)
    assert P.storage_hourly_usd("gcp", None, "n4-standard-4") == 0.0


# --------------------------------------------------------------------------------------------
# Exhaustion parsing
# --------------------------------------------------------------------------------------------


def test_parse_exhausted_zones_reads_real_skypilot_output():
    text = (
        "⚙️ Launching on GCP us-central1 (us-central1-a).\n"
        "W 10-11 18:25:57 instance_utils.py:112] Got return codes 'VM_MIN_COUNT_NOT_REACHED', "
        "'ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS' in us-central1-a: 'Requested minimum count "
        "of 1 VMs could not be created'; \"The zone 'projects/xxx/zones/us-central1-a' does not "
        "have enough resources available to fulfill the request. '(resource type:compute)'\"\n"
        "⚙️ Launching on GCP us-central1 (us-central1-b). Instance is up.\n"
    )
    assert P.parse_exhausted_zones(text) == ["us-central1-a"]


def test_progress_lines_alone_never_poison_the_memo():
    """"Launching on GCP us-central1 (us-central1-b)" names a zone that may have worked fine."""
    assert P.parse_exhausted_zones("⚙️ Launching on GCP us-central1 (us-central1-b).") == []


def test_parse_handles_long_region_names():
    text = (
        "ZONE_RESOURCE_POOL_EXHAUSTED in northamerica-northeast1-a and "
        "'zones/europe-west4-b' does not have enough resources"
    )
    assert P.parse_exhausted_zones(text) == ["northamerica-northeast1-a", "europe-west4-b"]


def test_quota_errors_are_not_treated_as_capacity():
    """Quota is regional and persistent; a 30-minute zone exclusion neither helps nor expires
    correctly, and would mask the real (actionable) diagnosis."""
    assert P.parse_exhausted_zones("Quota 'CPUS' exceeded. Limit: 8.0 in region us-central1") == []


# --------------------------------------------------------------------------------------------
# Capacity memo
# --------------------------------------------------------------------------------------------


def test_memo_round_trip_and_ttl(tmp_path):
    memo = P.CapacityMemo.for_home(tmp_path, ttl_s=100.0)
    memo.record("gcp", "n4-standard-4", ["us-central1-a", "us-central1-b"], now_s=1000.0)
    assert memo.exhausted_zones("gcp", "n4-standard-4", now_s=1050.0) == {
        "us-central1-a", "us-central1-b"
    }
    # past the TTL the zones are forgotten
    assert memo.exhausted_zones("gcp", "n4-standard-4", now_s=1200.0) == set()


def test_memo_is_keyed_on_instance_type(tmp_path):
    """n4-standard-4 being out in a zone says nothing about n1-highmem-4 there."""
    memo = P.CapacityMemo.for_home(tmp_path, ttl_s=100.0)
    memo.record("gcp", "n4-standard-4", ["us-central1-a"], now_s=1000.0)
    assert memo.exhausted_zones("gcp", "n1-highmem-4", now_s=1000.0) == set()
    assert memo.exhausted_zones("do", "n4-standard-4", now_s=1000.0) == set()


def test_corrupt_memo_reads_as_empty(tmp_path):
    """Advisory, never authoritative: a broken memo may slow a launch, never fail one."""
    path = tmp_path / P.CapacityMemo.FILENAME
    path.write_text("{not json at all")
    assert P.CapacityMemo(path).exhausted_zones("gcp", "n4-standard-4") == set()


def test_memo_write_to_an_unwritable_path_does_not_raise(tmp_path):
    memo = P.CapacityMemo(tmp_path / "nope" / "\0bad" / "memo.json")
    memo.record("gcp", "n4-standard-4", ["us-central1-a"])  # must not raise


def test_memo_prunes_expired_entries_on_write(tmp_path):
    memo = P.CapacityMemo.for_home(tmp_path, ttl_s=100.0)
    memo.record("gcp", "n4-standard-4", ["old-zone1-a"], now_s=1000.0)
    memo.record("gcp", "n4-standard-4", ["new-zone1-a"], now_s=2000.0)
    entries = json.loads((tmp_path / P.CapacityMemo.FILENAME).read_text())["entries"]
    assert list(entries) == ["gcp|n4-standard-4|new-zone1-a"]


# --------------------------------------------------------------------------------------------
# Candidate narrowing
# --------------------------------------------------------------------------------------------


def test_memo_excludes_only_fully_dead_regions(catalog, tmp_path):
    """One dead zone still leaves its region usable — SkyPilot can try the siblings."""
    memo = P.CapacityMemo.for_home(tmp_path, ttl_s=1e9)
    memo.record("gcp", "n4-standard-4", ["us-central1-a"])
    names = [c.region for c in P.candidates(_res(), instance_type="n4-standard-4", memo=memo)]
    assert "us-central1" in names

    memo.record("gcp", "n4-standard-4", _ZONES["us-central1"])
    names = [c.region for c in P.candidates(_res(), instance_type="n4-standard-4", memo=memo)]
    assert "us-central1" not in names
    assert "europe-west1" in names


def test_candidates_are_sorted_cheapest_first(catalog):
    cands = P.candidates(_res(use_spot=True), instance_type="n4-standard-4")
    assert [c.region for c in cands] == ["europe-west1", "asia-east2", "us-east1", "us-central1"]


def test_validate_placement_rejects_a_bad_region(monkeypatch):
    def _raise(r, z, clouds=None):
        raise ValueError("Invalid region 'eu-west-nope'")

    cat = fake_catalog()
    cat.validate_region_zone = _raise
    monkeypatch.setattr(P, "_catalog", lambda: cat)
    with pytest.raises(P.PlacementError):
        P.validate_placement(_res(region="eu-west-nope"))


def test_validate_placement_is_a_noop_without_pins(catalog):
    P.validate_placement(_res())  # must not raise or call anything


@pytest.mark.parametrize("memory,expected", [("32GB", "32"), ("32", "32"), ("8+", "8+"), (None, None)])
def test_catalog_memory_normalisation(memory, expected):
    assert P._catalog_memory(ResourceRequest(memory=memory)) == expected
