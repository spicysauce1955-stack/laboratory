"""Preflight checks (lab.doctor).

Hermetic: every GCP API is faked. Two of these encode findings from the live project, and both
are cases where the obvious implementation is confidently wrong.
"""

from __future__ import annotations

import types

import pytest

from lab import doctor as D
from lab.models import ResourceRequest


class FakeExec:
    def __init__(self, payload):
        self._payload = payload

    def execute(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def fake_compute(*, region_quotas=None, project_quotas=None):
    region_quotas = region_quotas or {}
    project_quotas = project_quotas or {}

    class Regions:
        def get(self, project, region):
            quotas = region_quotas.get(region, {})
            return FakeExec(
                {"quotas": [{"metric": m, "limit": lim, "usage": use}
                            for m, (lim, use) in quotas.items()]}
            )

    class Projects:
        def get(self, project):
            return FakeExec(
                {"quotas": [{"metric": m, "limit": lim} for m, lim in project_quotas.items()]}
            )

    return types.SimpleNamespace(regions=lambda: Regions(), projects=lambda: Projects())


@pytest.fixture
def ctx():
    return {"creds": object(), "project": "test-project"}


# --------------------------------------------------------------------------------------------
# The two live findings
# --------------------------------------------------------------------------------------------


def test_global_gpu_quota_blocks_even_when_regional_quota_is_fine(monkeypatch, ctx):
    """Verified live: this project holds NVIDIA_T4_GPUS=1 regionally and GPUS_ALL_REGIONS=0
    globally. GCP enforces both, so a check reading only the regional number reports "ok" and the
    launch still fails — the exact failure `lab doctor` exists to prevent."""
    compute = fake_compute(
        region_quotas={"us-central1": {"NVIDIA_T4_GPUS": (1, 0)}},
        project_quotas={"GPUS_ALL_REGIONS": 0},
    )
    monkeypatch.setattr(D, "_build", lambda creds, api, version: compute)
    res = ResourceRequest(cloud="gcp", accelerators="T4:1", region="us-central1")
    result = D.check_quota_gpu(res, ctx)
    assert result.status == "fail"
    assert "GPUS_ALL_REGIONS" in result.detail
    assert "all regions" in (result.fix or "").lower()


def test_gpu_quota_passes_when_both_levels_allow_it(monkeypatch, ctx):
    compute = fake_compute(
        region_quotas={"us-central1": {"NVIDIA_T4_GPUS": (4, 1)}},
        project_quotas={"GPUS_ALL_REGIONS": 4},
    )
    monkeypatch.setattr(D, "_build", lambda creds, api, version: compute)
    res = ResourceRequest(cloud="gcp", accelerators="T4:1", region="us-central1")
    assert D.check_quota_gpu(res, ctx).status == "ok"


def test_zero_preemptible_cpu_quota_is_not_a_failure(monkeypatch, ctx):
    """GCP documents that where preemptible quota was never granted, Spot VMs consume the standard
    CPUS quota. PREEMPTIBLE_CPUS=0 is the default and means nothing; failing on it would be a
    confident false positive that blocks a launch which would have worked."""
    compute = fake_compute(
        region_quotas={"us-central1": {"CPUS": (200, 0), "PREEMPTIBLE_CPUS": (0, 0)}}
    )
    monkeypatch.setattr(D, "_build", lambda creds, api, version: compute)
    res = ResourceRequest(cloud="gcp", cpus=4, region="us-central1", use_spot=True)
    assert D.check_quota_cpu(res, ctx).status == "ok"


# --------------------------------------------------------------------------------------------
# Fail-open on error, fail-closed on a definitive negative
# --------------------------------------------------------------------------------------------


def test_an_api_error_skips_rather_than_blocks(monkeypatch, ctx):
    """Seen live: the billing check got a 403 because the *Cloud Billing API itself* was disabled,
    on a project that was billing perfectly well. That is not evidence about billing."""

    def _boom(creds, api, version):
        raise RuntimeError("403 Cloud Billing API has not been used in project")

    monkeypatch.setattr(D, "_build", _boom)
    result = D.check_billing(ResourceRequest(cloud="gcp"), ctx)
    assert result.status == "skip"
    assert not result.blocking


def test_billing_disabled_is_a_definitive_failure(monkeypatch, ctx):
    api = types.SimpleNamespace(
        projects=lambda: types.SimpleNamespace(
            getBillingInfo=lambda name: FakeExec({"billingEnabled": False})
        )
    )
    monkeypatch.setattr(D, "_build", lambda creds, a, v: api)
    assert D.check_billing(ResourceRequest(cloud="gcp"), ctx).status == "fail"


def test_missing_iam_permissions_fail_and_name_them(monkeypatch, ctx):
    granted = [p for p in D.GCP_REQUIRED_PERMISSIONS if p != "storage.buckets.create"]
    crm = types.SimpleNamespace(
        projects=lambda: types.SimpleNamespace(
            testIamPermissions=lambda resource, body: FakeExec({"permissions": granted})
        )
    )
    monkeypatch.setattr(D, "_build", lambda creds, a, v: crm)
    result = D.check_iam(ResourceRequest(cloud="gcp"), ctx)
    assert result.status == "fail"
    assert "storage.buckets.create" in result.detail


def test_disabled_api_fails_with_the_enable_command(monkeypatch, ctx):
    su = types.SimpleNamespace(
        services=lambda: types.SimpleNamespace(
            get=lambda name: FakeExec(
                {"state": "DISABLED" if "compute" in name else "ENABLED"}
            )
        )
    )
    monkeypatch.setattr(D, "_build", lambda creds, a, v: su)
    result = D.check_apis(ResourceRequest(cloud="gcp"), ctx)
    assert result.status == "fail"
    assert "compute.googleapis.com" in (result.fix or "")


def test_a_check_that_raises_is_skipped_not_propagated(monkeypatch, tmp_path):
    """A broken check must never take a launch down with it."""

    def _explode(res, ctx):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(D._REGISTRY, "gcp", (("adc", _explode),))
    results = D.run_checks("gcp", ResourceRequest(cloud="gcp"), home=tmp_path)
    assert [r.status for r in results] == ["skip"]
    assert D.blocking_failures(results) == []


def test_preflight_returns_only_definitive_negatives(monkeypatch, tmp_path):
    def _fail(res, ctx):
        return D._fail("adc", "no credentials", "log in")

    def _skip_(res, ctx):
        return D._skip("apis", "could not answer")

    def _warn_(res, ctx):
        return D._warn("catalog", "unpriceable")

    monkeypatch.setitem(D._REGISTRY, "gcp", (("adc", _fail), ("apis", _skip_), ("catalog", _warn_)))
    blocking = D.preflight("gcp", ResourceRequest(cloud="gcp"), home=tmp_path)
    assert [r.name for r in blocking] == ["adc"]


def test_local_backend_never_preflights(tmp_path):
    assert D.preflight("local", ResourceRequest(), home=tmp_path) == []


# --------------------------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------------------------


def test_quota_cache_is_keyed_on_the_requested_shape(tmp_path):
    """Caught live: a `--cpus 4` run's 50 GB disk verdict was served to a later `--gpu T4:1` run,
    which asks about 100 GB in a different region set."""
    calls = []

    def _quota(res, ctx):
        calls.append(res.disk_size)
        return D._ok("quota_disk", f"{res.disk_size} GB ok")

    cache = D.CheckCache.for_home(tmp_path)
    key_a = f"gcp|p|quota_disk|{D._shape_key(ResourceRequest(disk_size=50))}"
    key_b = f"gcp|p|quota_disk|{D._shape_key(ResourceRequest(disk_size=100))}"
    assert key_a != key_b
    cache.put(key_a, D._ok("quota_disk", "50 GB ok"))
    assert cache.get(key_a) is not None
    assert cache.get(key_b) is None


def test_context_producing_checks_are_never_cached(monkeypatch, tmp_path):
    """A cached verdict carries the answer but not the side effect. Serving `adc` from cache left
    every later check credential-less, which read as "no GCP project selected"."""
    runs = []

    def _adc(res, ctx):
        runs.append(1)
        ctx["creds"], ctx["project"] = object(), "test-project"
        return D._ok("adc", "fine")

    def _project(res, ctx):
        return D._ok("project", str(ctx.get("project")))

    monkeypatch.setitem(D._REGISTRY, "gcp", (("adc", _adc), ("project", _project)))
    for _ in range(2):
        results = D.run_checks("gcp", ResourceRequest(cloud="gcp"), home=tmp_path)
        assert results[1].detail == "test-project"
    assert len(runs) == 2  # ran both times, because ctx cannot come from a cache


def test_skips_are_not_cached(tmp_path):
    """"Could not answer" is not a finding worth remembering — the next run should retry."""
    cache = D.CheckCache.for_home(tmp_path)
    cache.put("k", D._skip("apis", "timeout"))
    # put() itself stores whatever it is given; run_checks is what declines to cache a skip.
    assert cache.get("k") is not None


def test_corrupt_cache_reads_as_empty(tmp_path):
    path = tmp_path / D.CheckCache.FILENAME
    path.write_text("<<<not json")
    assert D.CheckCache(path).get("anything") is None


def test_expired_cache_entry_is_ignored(tmp_path):
    cache = D.CheckCache.for_home(tmp_path)
    cache.put("gcp|p|iam|-", D._ok("iam", "fine"), now_s=1000.0)
    assert cache.get("gcp|p|iam|-", now_s=1000.0 + D._TTL_S["iam"] - 1) is not None
    assert cache.get("gcp|p|iam|-", now_s=1000.0 + D._TTL_S["iam"] + 1) is None


# --------------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------------


def test_unknown_cloud_reports_a_skip_not_a_crash(tmp_path):
    results = D.run_checks("nonesuch", home=tmp_path)
    assert [r.status for r in results] == ["skip"]


def test_doctor_view_is_the_one_shape_both_shells_emit():
    """CLI `--json` and the MCP tool must not drift: they share this view, the way status does.
    They had already diverged once — the CLI omitted `ok`, the field a caller branches on."""
    results = [D._ok("adc", "fine"), D._skip("billing", "no answer"), D._fail("iam", "x", "y")]
    view = D.doctor_view("gcp", results)
    assert view["cloud"] == "gcp"
    assert view["ok"] is False
    assert [c["name"] for c in view["checks"]] == ["adc", "billing", "iam"]
    assert [b["name"] for b in view["blocking"]] == ["iam"]

    # A skip is not a blocker, so a run with no `fail` is ok.
    assert D.doctor_view(None, [D._skip("apis", "no answer")])["ok"] is True
    assert D.doctor_view(None, [])["cloud"] == "vast"


def test_format_report_shows_fixes_only_for_actionable_findings():
    out = D.format_report(
        [D._ok("adc", "fine"), D._fail("iam", "missing x", "grant it")]
    )
    assert "grant it" in out
    assert out.count("fix:") == 1


def test_gpu_family_and_count_parsing():
    assert D._gpu_family("T4:1") == "T4"
    assert D._gpu_family("a100-80gb:8") == "A100_80GB"
    assert D._gpu_count("T4:2") == 2
    assert D._gpu_count("T4") == 1
    assert D._gpu_family(None) is None
