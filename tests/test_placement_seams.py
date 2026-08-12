"""Where placement meets the rest of the lab: the disk invariant, build_task's search space,
per-cloud provision budgets, cost recording, and the registration worst case."""

from __future__ import annotations

import pytest
from helpers import make_manifest

from lab import placement as P
from lab.backends import skypilot as S
from lab.core import (
    CPU_DEFAULT_DISK_GB,
    GPU_DEFAULT_DISK_GB,
    default_disk_gb,
    resolve_backend_profile,
)
from lab.models import ResourceRequest


# --------------------------------------------------------------------------------------------
# The disk invariant (GCP-COST-2)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cloud,accelerators,expected",
    [
        ("gcp", None, CPU_DEFAULT_DISK_GB),
        ("gcp", "T4:1", GPU_DEFAULT_DISK_GB),
        ("do", None, CPU_DEFAULT_DISK_GB),
        ("vast", "RTX_4090:1", None),  # Vast folds storage into dph_total
        (None, None, None),
    ],
)
def test_default_disk_by_cloud_and_shape(cloud, accelerators, expected):
    res = ResourceRequest(cloud=cloud, accelerators=accelerators)
    assert default_disk_gb(res) == expected


def test_explicit_disk_size_always_wins():
    res = ResourceRequest(cloud="gcp", accelerators="T4:1", disk_size=17)
    assert default_disk_gb(res) == 17


def test_no_gcp_skypilot_spec_can_inherit_skypilots_256gb_default():
    """The invariant. SkyPilot's default is $0.0351/hr — more than a $0.0340/hr spot n4 — and it
    reached the GPU path only because that path was never given a number of its own."""
    _, resolved = resolve_backend_profile(
        "skypilot", ResourceRequest(cloud="gcp", accelerators="T4:1")
    )
    assert resolved.disk_size == GPU_DEFAULT_DISK_GB
    assert resolved.disk_size != 256


def test_build_task_applies_the_disk_default_even_when_the_profile_never_ran(tmp_path):
    """The invariant must hold on the DEFERRED path too.

    `resolve_backend_profile` runs on the CLI/MCP submit path only — the scheduler launches a
    registration straight through `Lab.submit`, so a registered GCP spec (which carries no
    disk_size) reached SkyPilot with none and inherited its 256 GB default. Caught by noticing
    that `lab register --cloud gcp` quoted a worst case with storage of exactly $0.
    """
    m = make_manifest("j1", "echo hi", resources=ResourceRequest(cloud="gcp", cpus=4))
    assert m.resources.disk_size is None  # exactly what a registration looks like
    got = _resources_of(S.build_task(m, workdir=tmp_path, memo=None))
    assert got[0].disk_size == CPU_DEFAULT_DISK_GB

    gpu = make_manifest("j2", "echo hi", resources=ResourceRequest(cloud="gcp", accelerators="T4:1"))
    got = _resources_of(S.build_task(gpu, workdir=tmp_path, memo=None))
    assert got[0].disk_size == GPU_DEFAULT_DISK_GB


def test_registration_worst_case_prices_the_disk_it_will_actually_get(monkeypatch):
    """A registration names no disk, but it will launch with one, so the exposure quoted to the
    user must include it."""
    from lab import placement

    est = placement.estimate(ResourceRequest(cloud="gcp", cpus=4))
    if est is None:
        pytest.skip("catalog unavailable")
    assert est.storage_usd > 0, "a GCP registration's worst case must price its disk"
    assert f"{CPU_DEFAULT_DISK_GB}GiB" in est.basis


def test_cpu_profile_disk_is_unchanged():
    _, resolved = resolve_backend_profile("cpu", ResourceRequest(cloud="gcp"))
    assert resolved.disk_size == CPU_DEFAULT_DISK_GB


def test_local_backend_is_untouched():
    backend, resolved = resolve_backend_profile("local", ResourceRequest())
    assert backend == "local"
    assert resolved.disk_size is None


# --------------------------------------------------------------------------------------------
# Per-cloud provisioning budgets (GCP-PROV-2)
# --------------------------------------------------------------------------------------------


def test_provision_timeout_is_per_cloud():
    """The live failure was failover TRUNCATED by a Vast-shaped 8-minute watchdog, not missing
    failover — GCP spends its provisioning time walking zones, not waiting on one host."""
    assert S.provision_timeout_min("vast") == 8
    assert S.provision_timeout_min("gcp") == 20
    assert S.provision_timeout_min("gcp") > S.provision_timeout_min("vast")
    assert S.provision_timeout_min(None) == 8
    assert S.provision_timeout_min("unknown-cloud") == S.DEFAULT_PROVISION_TIMEOUT_MIN


# --------------------------------------------------------------------------------------------
# build_task's search space
# --------------------------------------------------------------------------------------------


def _resources_of(task):
    got = task.resources
    return list(got) if not isinstance(got, dict) else list(got)


def _manifest(tmp_path, **res):
    return make_manifest("j1", "echo hi", resources=ResourceRequest(cloud="gcp", **res))


def test_unpinned_launch_has_a_single_unconstrained_resource(tmp_path):
    task = S.build_task(_manifest(tmp_path, cpus=4, disk_size=50), workdir=tmp_path, memo=None)
    got = _resources_of(task)
    assert len(got) == 1
    assert got[0].region is None and got[0].zone is None


def test_pins_reach_sky_resources(tmp_path):
    m = _manifest(tmp_path, cpus=4, disk_size=50, region="europe-west1", zone="europe-west1-b")
    got = _resources_of(S.build_task(m, workdir=tmp_path, memo=None))
    assert got[0].region == "europe-west1"
    assert got[0].zone == "europe-west1-b"


def test_price_cap_reaches_sky_resources(tmp_path):
    """The cap is enforced by the optimizer, which is what makes the quoted worst case a ceiling
    rather than a prediction."""
    m = _manifest(tmp_path, cpus=4, disk_size=50, max_hourly_usd=0.05)
    got = _resources_of(S.build_task(m, workdir=tmp_path, memo=None))
    assert got[0]._max_hourly_cost == 0.05


def test_spot_with_fallback_offers_both_kinds(tmp_path):
    m = _manifest(tmp_path, cpus=4, disk_size=50, use_spot=True, spot_fallback=True)
    got = _resources_of(S.build_task(m, workdir=tmp_path, memo=None))
    assert sorted(bool(r.use_spot) for r in got) == [False, True]


def test_spot_only_offers_one_kind(tmp_path):
    m = _manifest(tmp_path, cpus=4, disk_size=50, use_spot=True, spot_fallback=False)
    got = _resources_of(S.build_task(m, workdir=tmp_path, memo=None))
    assert [bool(r.use_spot) for r in got] == [True]


# --------------------------------------------------------------------------------------------
# Narrowing decisions (pure — no sky involved)
# --------------------------------------------------------------------------------------------


def test_no_memo_means_no_narrowing():
    assert S.narrowed_regions(ResourceRequest(cloud="gcp", cpus=4), None) == [None]


def test_an_empty_memo_means_no_narrowing(tmp_path):
    memo = P.CapacityMemo.for_home(tmp_path)
    assert S.narrowed_regions(ResourceRequest(cloud="gcp", cpus=4), memo) == [None]


def test_an_explicit_pin_beats_the_memo(tmp_path, monkeypatch):
    """The memo must never silently override what the user asked for."""
    monkeypatch.setattr(P, "resolve_instance_type", lambda res: "n4-standard-4")
    memo = P.CapacityMemo.for_home(tmp_path)
    memo.record("gcp", "n4-standard-4", ["europe-west1-b"])
    res = ResourceRequest(cloud="gcp", cpus=4, region="europe-west1")
    assert S.narrowed_regions(res, memo) == [None]


def _fake_candidates(monkeypatch, *, with_memo, without_memo):
    """Fake `candidates` so the memo-aware and memo-free calls can differ."""
    def _c(res, instance_type, memo=None):
        names = with_memo if memo is not None else without_memo
        return [
            P.Candidate(region=n, zones=(f"{n}-b",), hourly_usd=0.01 * i)
            for i, n in enumerate(names, start=1)
        ]

    monkeypatch.setattr(P, "candidates", _c)


def test_memo_narrows_to_surviving_regions(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "resolve_instance_type", lambda res: "n4-standard-4")
    _fake_candidates(
        monkeypatch,
        with_memo=["europe-west1", "us-east1"],
        without_memo=["europe-west1", "us-east1", "us-central1"],
    )
    memo = P.CapacityMemo.for_home(tmp_path)
    memo.record("gcp", "n4-standard-4", ["us-central1-a"])
    assert S.narrowed_regions(ResourceRequest(cloud="gcp", cpus=4), memo) == [
        "europe-west1", "us-east1"
    ]


def test_a_memo_that_costs_no_region_does_not_narrow(tmp_path, monkeypatch):
    """A dead zone in a multi-zone region leaves that region usable, so the memo can be non-empty
    while changing nothing. Narrowing anyway would trade 40 regions of failover for 10 and buy
    nothing — the design's promise is that an idle memo is invisible."""
    monkeypatch.setattr(P, "resolve_instance_type", lambda res: "n4-standard-4")
    survivors = ["europe-west1", "us-east1", "us-central1"]
    _fake_candidates(monkeypatch, with_memo=survivors, without_memo=survivors)
    memo = P.CapacityMemo.for_home(tmp_path)
    memo.record("gcp", "n4-standard-4", ["us-central1-a"])
    assert S.narrowed_regions(ResourceRequest(cloud="gcp", cpus=4), memo) == [None]


def test_a_memo_that_would_exclude_everything_is_ignored(tmp_path, monkeypatch):
    """Better to launch into a possibly-dead zone than to refuse to launch at all."""
    monkeypatch.setattr(P, "resolve_instance_type", lambda res: "n4-standard-4")
    monkeypatch.setattr(P, "candidates", lambda res, instance_type, memo=None: [])
    memo = P.CapacityMemo.for_home(tmp_path)
    memo.record("gcp", "n4-standard-4", ["us-central1-a"])
    assert S.narrowed_regions(ResourceRequest(cloud="gcp", cpus=4), memo) == [None]


def test_narrowing_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "resolve_instance_type", lambda res: "n4-standard-4")
    _fake_candidates(
        monkeypatch,
        with_memo=[f"r{i}" for i in range(40)],
        without_memo=[f"r{i}" for i in range(41)],
    )
    memo = P.CapacityMemo.for_home(tmp_path)
    memo.record("gcp", "n4-standard-4", ["us-central1-a"])
    got = S.narrowed_regions(ResourceRequest(cloud="gcp", cpus=4), memo)
    assert len(got) == P.MAX_NARROWED_REGIONS


# --------------------------------------------------------------------------------------------
# Cost recording
# --------------------------------------------------------------------------------------------


def test_resolve_cost_folds_storage_into_the_headline_number(tmp_path, monkeypatch):
    """Everything downstream reads hourly_usd, so the disk has to live there to reach them."""
    from lab import sky_runner

    monkeypatch.setattr(sky_runner, "_resolve_hourly", lambda cluster, handle, cloud: 0.034)
    m = make_manifest(
        "j1", "echo hi", resources=ResourceRequest(cloud="gcp", disk_size=256), timeout="1h"
    )
    cost = sky_runner.resolve_cost(
        "lab-j1", None, m, "gcp", instance_type="n4-standard-4"
    )
    assert cost.compute_hourly_usd == pytest.approx(0.034)
    assert cost.storage_hourly_usd == pytest.approx(256 * 0.000109589)
    assert cost.hourly_usd == pytest.approx(cost.compute_hourly_usd + cost.storage_hourly_usd)
    assert cost.estimated_usd == pytest.approx(cost.hourly_usd, abs=1e-6)  # 1h wall, rounded
    assert "256GiB" in (cost.hourly_basis or "")


def test_resolve_cost_survives_an_unknown_compute_price(tmp_path, monkeypatch):
    from lab import sky_runner

    monkeypatch.setattr(sky_runner, "_resolve_hourly", lambda cluster, handle, cloud: None)
    m = make_manifest("j1", "echo hi", resources=ResourceRequest(cloud="gcp", disk_size=50))
    cost = sky_runner.resolve_cost("lab-j1", None, m, "gcp", instance_type="n4-standard-4")
    assert cost.hourly_usd is None
    assert cost.storage_hourly_usd > 0
    assert "compute unknown" in (cost.hourly_basis or "")


def test_capacity_exhaustion_is_memoised_from_a_failed_launch(tmp_path, monkeypatch):
    from lab import sky_runner

    monkeypatch.setattr(P, "resolve_instance_type", lambda res: "n4-standard-4")
    m = make_manifest("j1", "echo hi", resources=ResourceRequest(cloud="gcp", cpus=4))
    zones = sky_runner.record_capacity_exhaustion(
        tmp_path,
        m,
        "gcp",
        error_text="ResourcesUnavailableError: failed",
        log_text="ZONE_RESOURCE_POOL_EXHAUSTED in us-central1-a: no capacity",
    )
    assert zones == ["us-central1-a"]
    memo = P.CapacityMemo.for_home(tmp_path)
    assert memo.exhausted_zones("gcp", "n4-standard-4") == {"us-central1-a"}


def test_a_non_capacity_failure_memoises_nothing(tmp_path, monkeypatch):
    from lab import sky_runner

    monkeypatch.setattr(P, "resolve_instance_type", lambda res: "n4-standard-4")
    m = make_manifest("j1", "echo hi", resources=ResourceRequest(cloud="gcp", cpus=4))
    assert sky_runner.record_capacity_exhaustion(
        tmp_path, m, "gcp", error_text="Quota 'CPUS' exceeded", log_text=""
    ) == []


# --------------------------------------------------------------------------------------------
# stdout stays parseable
# --------------------------------------------------------------------------------------------


def test_placement_diagnostics_go_to_stderr(capsys, monkeypatch):
    """The CLI emits JSON on stdout and callers parse it, so a "[lab] catalog price unavailable"
    printed there is not a log line — it is a corrupted payload. Caught by `lab register` failing
    to produce valid JSON once pricing started running on that path."""
    monkeypatch.setattr(P, "_catalog", lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    assert P.resolve_instance_type(ResourceRequest(cloud="gcp", cpus=4)) is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[lab]" in captured.err


def test_catalog_hourly_diagnostics_go_to_stderr(capsys, monkeypatch):
    monkeypatch.setattr(P, "estimate", lambda res, memo=None: None)
    assert S.catalog_hourly(ResourceRequest(cloud="gcp", cpus=4)) is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "catalog price unavailable" in captured.err


# --------------------------------------------------------------------------------------------
# Failure diagnosis survives truncation
# --------------------------------------------------------------------------------------------

# The real message SkyPilot raised on the live GPU launch. It is 290 characters before any
# diagnosis is added, against a 300-character `end_reason` budget.
_REAL_SKY_ERROR = (
    "launch error: Failed to provision all possible launchable resources. Relax the task's "
    "resource requirements: 1x GCP({'T4': 1}, disk_size=100)\n"
    "To keep retrying until the cluster is up, use the `--retry-until-up` flag.\n"
    "Reasons for provision failures (for details, please check the log above):\n"
)


def test_the_diagnosis_survives_the_end_reason_truncation():
    """Found live. The hint used to be appended after SkyPilot's message, which alone exceeds the
    300-char manifest budget — so the actionable sentence was reliably cut off and the real cause
    (GPUS_ALL_REGIONS=0) never reached the manifest. Leading with it is the whole fix."""
    from lab.sky_runner import provision_failure_reason

    reason = provision_failure_reason(
        _REAL_SKY_ERROR + "Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally", "gcp"
    )
    assert len(_REAL_SKY_ERROR) > 250  # the generic part really is that long
    assert "GPUs (all regions)" in reason[:300]


def test_global_gpu_quota_is_diagnosed_ahead_of_the_generic_quota_hint():
    """GCP enforces GPU quota at two levels, and a fresh project fails the global one while its
    regional quota reads fine — so pointing at NVIDIA_T4_GPUS would send the user to the wrong
    console page."""
    from lab.sky_runner import _gcp_failure_hint

    hint = _gcp_failure_hint("Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally")
    assert "GLOBAL" in hint
    assert "GPUS_ALL_REGIONS" in hint


def test_an_unsatisfiable_spec_is_diagnosed_as_such_not_as_a_setup_problem():
    """Observed live with `--price-cap 0.001`: nothing provisioned (correct), but the manifest
    blamed credentials/API/quota. The optimizer had rejected the spec before touching the cloud,
    and SkyPilot's own log said so — 'Max hourly cost limit may be too restrictive'."""
    from lab.sky_runner import _gcp_failure_hint

    hint = _gcp_failure_hint(
        "launch error: Catalog does not contain any instances satisfying the request: 1"
    )
    assert "price-cap" in hint
    assert "not billed" in hint
    assert "sky check gcp" not in hint  # the generic checklist must not win here


def test_capacity_exhaustion_still_diagnoses_as_capacity():
    from lab.sky_runner import _gcp_failure_hint

    assert "capacity" in _gcp_failure_hint("ZONE_RESOURCE_POOL_EXHAUSTED in us-central1-a")


# --------------------------------------------------------------------------------------------
# Registration worst case (GCP-COST-4)
# --------------------------------------------------------------------------------------------


def test_worst_case_cost_falls_back_to_the_catalog_for_non_vast_clouds(monkeypatch):
    """It used to be None for every GCP/DO registration, because price triggers are Vast-only.
    "Run this overnight, worst case null dollars" — and blank reads as free, not as unknown."""
    from lab.scheduler import register as R
    from lab.scheduler.models import Triggers

    monkeypatch.setattr(S, "catalog_hourly", lambda res: 0.20)
    res = ResourceRequest(cloud="gcp", cpus=4, timeout="2h")
    assert R.worst_case_cost(Triggers(), res) == pytest.approx(0.40)


def test_worst_case_cost_prefers_a_real_offer_price(monkeypatch):
    from lab.scheduler import register as R
    from lab.scheduler.models import Triggers

    monkeypatch.setattr(S, "catalog_hourly", lambda res: 99.0)
    res = ResourceRequest(cloud="vast", accelerators="RTX_4090:1", timeout="2h")
    assert R.worst_case_cost(Triggers(max_hourly_usd=0.5), res) == pytest.approx(1.0)


def test_worst_case_cost_is_still_none_when_there_is_nothing_to_price():
    """A plain local registration has no cloud and no accelerators — unchanged behaviour."""
    from lab.scheduler import register as R
    from lab.scheduler.models import Triggers

    assert R.worst_case_cost(Triggers(), ResourceRequest(timeout="2h")) is None
    assert R.worst_case_cost(Triggers(), ResourceRequest(cloud="gcp")) is None


def test_catalog_chatter_never_reaches_stdout(monkeypatch, capsys):
    """SkyPilot prints "Updating <cloud> catalog: ..." on STDOUT the first time a machine needs a
    catalog CSV. Our stdout is a JSON payload, so on a cold catalog — a fresh laptop, a new
    scheduler droplet — that corrupts what callers parse: `json.loads(lab register ...)` failed on
    CI's first-ever run and would have worked on every run after."""
    import lab.placement as placement_mod

    class _NoisyCatalog:
        def list_accelerators(self, *a, **k):
            print("Updating Vast catalog: vast/vms.csv")
            return {}

        def get_instance_type_for_accelerator(self, *a, **k):
            print("Updating Vast catalog: vast/vms.csv")
            return (["x1.large"], None)

    monkeypatch.setattr(placement_mod, "_catalog", lambda: _NoisyCatalog())
    placement_mod.resolve_instance_type(
        ResourceRequest(cloud="vast", accelerators="RTX_4090:1")
    )

    captured = capsys.readouterr()
    assert captured.out == "", f"catalog chatter leaked onto stdout: {captured.out!r}"
    assert "Updating Vast catalog" in captured.err
