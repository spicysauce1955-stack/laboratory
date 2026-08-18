"""Preflight — "will this launch work?", asked before it costs a provision.

LAB-BUGS §8 closed with "a `lab doctor` readout would also pre-empt it." It was never built, and
the live GCP bring-up paid for that: APIs disabled, then six missing IAM roles, then the ADC daemon
gotcha — each discovered by a failed command, in series, each costing a provisioning round trip.

Every check here maps to a failure this project actually hit. Two rules keep the cure from being
worse than the disease:

* **Fail-open on error.** A check that cannot answer — a 5xx, a timeout, a missing library, an API
  that is itself disabled — is ``skip``, and ``skip`` never blocks a launch. A preflight that
  refused a job because the *preflight* broke would be strictly worse than no preflight.
* **Fail-closed only on a definitive negative.** Only a check that positively established "this
  cannot work" (quota is zero, a required API is off, a permission is absent) blocks.

That distinction is not theoretical. On the first live run the billing check got a 403 saying the
*Cloud Billing API* was disabled — which says nothing about whether billing is enabled, on a
project that was demonstrably billing fine. Under a naive reading that is a hard failure; under
these rules it is a ``skip``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from lab import events
from lab._util import atomic_write_text
from lab.models import ResourceRequest
from pydantic import BaseModel

Status = Literal["ok", "warn", "fail", "skip"]

# Per-check cache lifetimes. Credentials and grants change rarely; quota moves when a VM starts,
# so it gets a short one. The cache exists so the auto-preflight costs nothing on the common path.
_TTL_S: dict[str, float] = {
    "adc": 3600.0,
    "sky_daemon": 3600.0,
    "project": 3600.0,
    "billing": 86400.0,
    "apis": 86400.0,
    "iam": 86400.0,
    "quota_cpu": 3600.0,
    "quota_gpu": 3600.0,
    "quota_disk": 3600.0,
}
_DEFAULT_TTL_S = 3600.0

# Checks cheap and decisive enough to run automatically before every remote launch. `sky_daemon`
# is excluded: it shells out to `sky check`, which takes seconds, and its finding is a warning.
QUICK_CHECKS = ("adc", "project", "apis", "iam", "quota_cpu", "quota_gpu", "quota_disk", "catalog")

# Checks that are never served from cache. Two reasons, both load-bearing:
#   * they publish into ``ctx`` (credentials, the project, the resolved instance type) that later
#     checks read — a cached verdict carries the answer but not the side effect, which silently
#     left every downstream check credential-less;
#   * they are local anyway (a file read and a catalog lookup), so caching buys nothing.
_NEVER_CACHE = frozenset({"adc", "project", "catalog"})

# Checks whose verdict depends on the *shape* being launched, not just the account. Their cache
# key carries a fingerprint of that shape; without it a `--cpus 4` run's disk verdict was served
# to a later `--gpu T4:1` run, which asks about a different size in different regions.
_SHAPE_DEPENDENT = frozenset({"quota_cpu", "quota_gpu", "quota_disk"})


def _shape_key(res: ResourceRequest) -> str:
    """Fingerprint of everything that can change a shape-dependent verdict.

    ``max_hourly_usd`` belongs here even though no check reads it directly: it changes which
    regions are *candidates*, and the quota checks ask about the cheapest candidates. Caught live —
    a `--price-cap 0.001` run left no candidates at all, fell back to checking us-central1, and
    cached that under the key an uncapped run of the same size would look up.
    """
    return (
        f"{res.cpus}/{res.disk_size}/{res.accelerators}/{res.region}/{res.zone}"
        f"/{res.use_spot}/{res.spot_fallback}/{res.max_hourly_usd}"
    )


# Project-level permissions SkyPilot needs on GCP. Verified against the live project: GCP's
# projects.testIamPermissions accepts all of these and returns the subset the caller holds.
# Permissions, not roles, so a custom role granting the same access does not read as broken.
GCP_REQUIRED_PERMISSIONS = (
    "compute.instances.create",
    "compute.instances.delete",
    "compute.instances.list",
    "compute.disks.create",
    "compute.disks.delete",
    "iam.serviceAccounts.actAs",
    "serviceusage.services.use",
    "resourcemanager.projects.getIamPolicy",
    "storage.buckets.create",
    "storage.buckets.delete",
)

GCP_REQUIRED_APIS = ("compute.googleapis.com", "cloudresourcemanager.googleapis.com")

# How many of the cheapest candidate regions to check regional quota in. Quota is per-region and
# the optimizer may pick any of ~40, so checking one proves little and checking all is far too
# slow; a handful establishes whether the project has quota *somewhere* it would plausibly land.
_QUOTA_REGIONS_TO_CHECK = 3


class CheckResult(BaseModel):
    """One preflight finding. ``fix`` is the command or console action that resolves it."""

    name: str
    status: Status
    detail: str
    fix: str | None = None

    @property
    def blocking(self) -> bool:
        return self.status == "fail"


def _ok(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="ok", detail=detail)


def _fail(name: str, detail: str, fix: str) -> CheckResult:
    return CheckResult(name=name, status="fail", detail=detail, fix=fix)


def _warn(name: str, detail: str, fix: str | None = None) -> CheckResult:
    return CheckResult(name=name, status="warn", detail=detail, fix=fix)


def _skip(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="skip", detail=detail)


# --------------------------------------------------------------------------------------------
# Result cache
# --------------------------------------------------------------------------------------------


class CheckCache:
    """Per-(cloud, project, check) results with per-check TTLs. Best-effort; never raises.

    Without this the auto-preflight would add half a dozen API round trips to every submit, which
    is exactly the kind of tax that gets a safety feature switched off.
    """

    FILENAME = "doctor_cache.json"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @classmethod
    def for_home(cls, home: Path) -> CheckCache:
        return cls(Path(home) / cls.FILENAME)

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text())
            return raw if isinstance(raw, dict) else {}
        except Exception:  # noqa: BLE001 — absent/corrupt means "nothing cached"
            return {}

    def get(self, key: str, *, now_s: float | None = None) -> CheckResult | None:
        now_s = time.time() if now_s is None else now_s
        entry = self._load().get(key)
        if not isinstance(entry, dict):
            return None
        try:
            name = str(entry["result"]["name"])
            if now_s - float(entry["at"]) >= _TTL_S.get(name, _DEFAULT_TTL_S):
                return None
            return CheckResult.model_validate(entry["result"])
        except Exception:  # noqa: BLE001
            return None

    def put(self, key: str, result: CheckResult, *, now_s: float | None = None) -> None:
        now_s = time.time() if now_s is None else now_s
        try:
            data = self._load()
            data[key] = {"at": now_s, "result": result.model_dump()}
            atomic_write_text(self.path, json.dumps(data, sort_keys=True))
        except Exception:  # noqa: BLE001 — a cache write must never fail a submit
            pass


# --------------------------------------------------------------------------------------------
# GCP checks
# --------------------------------------------------------------------------------------------


def _gcp_clients() -> tuple[Any, str]:
    """(credentials, project). Raises GcpNotConfigured when GCP isn't set up here."""
    from lab.backends.skypilot import _gcp_default_credentials

    creds, project = _gcp_default_credentials()
    if not project:
        from lab.backends.skypilot import GcpNotConfigured

        raise GcpNotConfigured("no GCP project selected")
    return creds, str(project)


def _build(creds: Any, api: str, version: str) -> Any:
    from googleapiclient import discovery  # type: ignore[import-untyped]

    return discovery.build(api, version, credentials=creds, cache_discovery=False)


@contextmanager
def _quiet_google_http() -> Iterator[None]:
    """Silence googleapiclient's own WARNING for HTTP errors while checks run.

    A check that gets a 403 reports it as a `skip` or a `fail` with a fix — in the user's terms,
    on the right line of the table. The library's raw "Encountered 403 Forbidden with reason
    PERMISSION_DENIED" adds nothing and reads as an alarm above output that is deliberately calm
    about it. Scoped to this block, so nothing else in the process loses the warning.
    """
    logger = logging.getLogger("googleapiclient.http")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _principal(creds: Any) -> str:
    """A human-readable identity for the ADC in use, without ever printing the credential."""
    for attr in ("service_account_email", "signer_email", "quota_project_id"):
        value = getattr(creds, attr, None)
        if value:
            return str(value)
    return type(creds).__name__


def check_adc(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    try:
        creds, project = _gcp_clients()
    except Exception as e:  # noqa: BLE001
        return _fail(
            "adc",
            f"no usable application-default credentials: {e}",
            "gcloud auth application-default login — or set GOOGLE_APPLICATION_CREDENTIALS "
            "in .env to a service-account key path",
        )
    ctx["creds"], ctx["project"] = creds, project
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    where = f" (key {key_path})" if key_path else ""
    return _ok("adc", f"{_principal(creds)}{where}")


def check_project(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    project = ctx.get("project")
    if not project:
        return _fail(
            "project",
            "no GCP project selected",
            "gcloud config set project <id>, or GOOGLE_CLOUD_PROJECT=<id> in .env",
        )
    return _ok("project", str(project))


def check_sky_daemon(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    """Ask SkyPilot's own daemon whether GCP works *for it*.

    `.env` configures the lab process; SkyPilot runs a long-lived API server that does not inherit
    it, so the daemon can disagree with everything the checks above just verified. That divergence
    is the documented GCP-CREDS-3 gotcha, and this is the only check that can see it.
    """
    try:
        proc = subprocess.run(
            ["sky", "check", "gcp"], capture_output=True, text=True, timeout=120
        )
    except Exception as e:  # noqa: BLE001 — sky missing or slow: not our verdict to give
        return _skip("sky_daemon", f"could not run `sky check gcp`: {e}")
    out = f"{proc.stdout}\n{proc.stderr}"
    if "GCP: enabled" in out:
        return _ok("sky_daemon", "SkyPilot's API server can use GCP")
    if "GCP: disabled" in out:
        return _fail(
            "sky_daemon",
            "SkyPilot's API server reports GCP disabled even though ADC resolves here — the "
            "daemon does not inherit .env",
            "uv run sky api stop  (the next lab command restarts it with this environment), or "
            "symlink the key to ~/.config/gcloud/application_default_credentials.json",
        )
    return _skip("sky_daemon", "could not parse `sky check gcp` output")


def check_apis(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    creds, project = ctx.get("creds"), ctx.get("project")
    if creds is None or not project:
        return _skip("apis", "no credentials to check with")
    try:
        su = _build(creds, "serviceusage", "v1")
        off = []
        for api in GCP_REQUIRED_APIS:
            state = su.services().get(name=f"projects/{project}/services/{api}").execute()
            if str(state.get("state")) != "ENABLED":
                off.append(api)
    except Exception as e:  # noqa: BLE001 — cannot answer -> do not block
        return _skip("apis", f"could not read service states: {str(e)[:160]}")
    if off:
        return _fail(
            "apis",
            f"disabled: {', '.join(off)}",
            f"gcloud services enable {' '.join(off)} (needs roles/serviceusage.serviceUsageAdmin)",
        )
    return _ok("apis", ", ".join(GCP_REQUIRED_APIS))


def check_billing(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    creds, project = ctx.get("creds"), ctx.get("project")
    if creds is None or not project:
        return _skip("billing", "no credentials to check with")
    try:
        api = _build(creds, "cloudbilling", "v1")
        info = api.projects().getBillingInfo(name=f"projects/{project}").execute()
    except Exception as e:  # noqa: BLE001
        # Seen live: a 403 because the *Cloud Billing API itself* is disabled. That is not
        # evidence about billing — the project was billing fine — so it must not block. Say that
        # in one line rather than pasting the library's URL-laden 403 into the table.
        if "has not been used in project" in str(e) or "SERVICE_DISABLED" in str(e):
            return CheckResult(
                name="billing",
                status="skip",
                detail="not checked — the Cloud Billing API is off, which says nothing about "
                "whether billing itself is enabled",
                fix="gcloud services enable cloudbilling.googleapis.com (optional; only makes "
                "this check possible)",
            )
        return _skip("billing", f"could not read billing info: {str(e)[:160]}")
    if not info.get("billingEnabled"):
        return _fail(
            "billing",
            "the project has no active billing account",
            "attach a billing account in the console; provisioning fails without one",
        )
    return _ok("billing", str(info.get("billingAccountName", "enabled")))


def check_iam(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    creds, project = ctx.get("creds"), ctx.get("project")
    if creds is None or not project:
        return _skip("iam", "no credentials to check with")
    try:
        crm = _build(creds, "cloudresourcemanager", "v1")
        got = crm.projects().testIamPermissions(
            resource=project, body={"permissions": list(GCP_REQUIRED_PERMISSIONS)}
        ).execute()
        have = set(got.get("permissions", []))
    except Exception as e:  # noqa: BLE001
        return _skip("iam", f"could not test permissions: {str(e)[:160]}")
    missing = [p for p in GCP_REQUIRED_PERMISSIONS if p not in have]
    if missing:
        return _fail(
            "iam",
            f"missing {len(missing)}: {', '.join(missing)}",
            "grant roles/compute.admin, roles/iam.serviceAccountUser, "
            "roles/iam.serviceAccountAdmin, roles/serviceusage.serviceUsageAdmin, "
            "roles/iam.securityReviewer, roles/storage.admin",
        )
    return _ok("iam", f"all {len(GCP_REQUIRED_PERMISSIONS)} required permissions present")


def _quota_regions(res: ResourceRequest, ctx: dict[str, Any]) -> list[str]:
    """The regions worth checking quota in: the pin, else the cheapest few candidates."""
    if res.region:
        return [res.region]
    if res.zone:
        return [res.zone.rsplit("-", 1)[0]]
    est = ctx.get("estimate")
    cands = ctx.get("candidates") or []
    if not cands and est is None:
        return ["us-central1"]
    return [c.region for c in cands[:_QUOTA_REGIONS_TO_CHECK]] or ["us-central1"]


def _region_quotas(compute: Any, project: str, region: str) -> dict[str, tuple[float, float]]:
    data = compute.regions().get(project=project, region=region).execute()
    return {
        str(q["metric"]): (float(q.get("limit", 0)), float(q.get("usage", 0)))
        for q in data.get("quotas", [])
        if "metric" in q
    }


def _gpu_family(accelerators: str | None) -> str | None:
    if not accelerators:
        return None
    return accelerators.partition(":")[0].upper().replace("-", "_")


def _gpu_count(accelerators: str | None) -> int:
    if not accelerators:
        return 0
    _, _, count = accelerators.partition(":")
    try:
        return int(count) if count else 1
    except ValueError:
        return 1


def check_quota_gpu(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    """GPU quota, at **both** levels GCP enforces.

    A fresh project has a regional per-family quota *and* a separate global ``GPUS_ALL_REGIONS``
    ceiling, and a launch needs both. Verified live on this project: regional ``NVIDIA_T4_GPUS`` is
    1 in us-central1 and europe-west1 while ``GPUS_ALL_REGIONS`` is 0 — so a check reading only the
    regional number reports "ok" and the launch still fails. Reading only one level is the exact
    shape of bug this command exists to prevent.
    """
    family = _gpu_family(res.accelerators)
    if family is None:
        return _skip("quota_gpu", "no accelerators requested")
    creds, project = ctx.get("creds"), ctx.get("project")
    if creds is None or not project:
        return _skip("quota_gpu", "no credentials to check with")
    want = _gpu_count(res.accelerators)
    try:
        compute = _build(creds, "compute", "v1")
        info = compute.projects().get(project=project).execute()
        global_limit = next(
            (
                float(q.get("limit", 0))
                for q in info.get("quotas", [])
                if str(q.get("metric")) == "GPUS_ALL_REGIONS"
            ),
            None,
        )
        regions = _quota_regions(res, ctx)
        metric = f"NVIDIA_{family}_GPUS"
        regional = {r: _region_quotas(compute, project, r).get(metric) for r in regions}
    except Exception as e:  # noqa: BLE001
        return _skip("quota_gpu", f"could not read GPU quota: {str(e)[:160]}")

    if global_limit is not None and global_limit < want:
        return _fail(
            "quota_gpu",
            f"global GPUS_ALL_REGIONS is {global_limit:.0f}, need {want} — this blocks every "
            "region regardless of per-region quota",
            "request an increase for 'GPUs (all regions)' (Global) in IAM & Admin > Quotas; "
            "a fresh project starts at 0 and approval can take up to 48h",
        )
    usable = {r: q for r, q in regional.items() if q is not None and q[0] - q[1] >= want}
    if not usable:
        seen = ", ".join(
            f"{r}={q[0]:.0f}" if q else f"{r}=none" for r, q in regional.items()
        )
        return _fail(
            "quota_gpu",
            f"no checked region has {want}x {family} free ({seen})",
            f"request NVIDIA_{family}_GPUS quota in a region you intend to use",
        )
    where = ", ".join(f"{r} {q[0] - q[1]:.0f} free" for r, q in usable.items())
    detail = f"{family}x{want}: {where}"
    if global_limit is not None:
        detail += f"; global GPUS_ALL_REGIONS={global_limit:.0f}"
    return _ok("quota_gpu", detail)


def _resolved_vcpus(res: ResourceRequest, ctx: dict[str, Any]) -> int | None:
    """vCPUs of the instance type this spec resolves to, from the catalog. None if unknown."""
    est = ctx.get("estimate")
    instance_type = getattr(est, "instance_type", None)
    if not instance_type:
        return None
    try:
        from sky import catalog

        vcpus, _mem = catalog.get_vcpus_mem_from_instance_type(
            str(instance_type), clouds=res.cloud or "vast"
        )
        return int(vcpus) if vcpus else None
    except Exception:  # noqa: BLE001 — derivation is a bonus, never a failure
        return None


def check_quota_cpu(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    """Regional vCPU quota.

    Note what is *not* checked: ``PREEMPTIBLE_CPUS``. It is 0 on this project, and GCP documents
    that where preemptible quota was never granted, Spot VMs consume the standard ``CPUS`` quota
    instead. Failing on it would be a confident false positive.
    """
    # Prefer the vCPU count of the instance type that will actually provision. A `--gpu T4:1` job
    # names no vCPUs but lands on n1-highmem-4 and consumes 4 of them; checking only what the user
    # typed would skip the check on exactly the specs most likely to be quota-blocked.
    want = res.cpus or _resolved_vcpus(res, ctx) or 0
    if not want:
        return _skip("quota_cpu", "no vCPU count requested and none derivable from the catalog")
    creds, project = ctx.get("creds"), ctx.get("project")
    if creds is None or not project:
        return _skip("quota_cpu", "no credentials to check with")
    try:
        compute = _build(creds, "compute", "v1")
        regions = _quota_regions(res, ctx)
        quotas = {r: _region_quotas(compute, project, r).get("CPUS") for r in regions}
    except Exception as e:  # noqa: BLE001
        return _skip("quota_cpu", f"could not read CPU quota: {str(e)[:160]}")
    usable = {r: q for r, q in quotas.items() if q is not None and q[0] - q[1] >= want}
    if not usable:
        return _fail(
            "quota_cpu",
            f"no checked region has {want} vCPUs free ({', '.join(quotas)})",
            "request a CPUS quota increase for the region you intend to use",
        )
    return _ok("quota_cpu", f"{want} vCPU: {', '.join(usable)} ok")


def check_quota_disk(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    want = res.disk_size or 0
    if not want:
        return _skip("quota_disk", "no explicit disk size requested")
    creds, project = ctx.get("creds"), ctx.get("project")
    if creds is None or not project:
        return _skip("quota_disk", "no credentials to check with")
    try:
        compute = _build(creds, "compute", "v1")
        regions = _quota_regions(res, ctx)
        quotas = {r: _region_quotas(compute, project, r).get("DISKS_TOTAL_GB") for r in regions}
    except Exception as e:  # noqa: BLE001
        return _skip("quota_disk", f"could not read disk quota: {str(e)[:160]}")
    usable = {r: q for r, q in quotas.items() if q is not None and q[0] - q[1] >= want}
    if not usable:
        return _fail(
            "quota_disk",
            f"no checked region has {want} GB of disk quota free",
            "request a DISKS_TOTAL_GB increase for the region you intend to use",
        )
    return _ok("quota_disk", f"{want} GB: {', '.join(usable)} ok")


def check_catalog(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    """What the spec resolves to and what it will cost — the only check needing no credentials."""
    from lab import placement

    try:
        est = placement.estimate(res)
    except Exception as e:  # noqa: BLE001
        return _skip("catalog", f"catalog unavailable: {str(e)[:160]}")
    if est is None:
        return _warn(
            "catalog",
            "SkyPilot's catalog cannot price this spec, so cost guardrails will not apply to it",
            "check --cpus/--memory/--accelerators name a shape this cloud offers",
        )
    ctx["estimate"] = est
    # Reuse the type the estimate already resolved rather than asking the catalog again: this
    # runs on every remote submit, and a 32-shard sweep pays it 32 times.
    ctx["candidates"] = placement.candidates(res, instance_type=est.instance_type)
    detail = (
        f"{est.instance_type}, {est.regions} region(s), "
        f"${est.best_hourly_usd:.4f}-${est.worst_hourly_usd:.4f}/hr incl. "
        f"${est.storage_usd:.4f} disk"
    )
    if est.excluded_zones:
        detail += f" ({est.excluded_zones} zone(s) memoised as exhausted)"
    return _ok("catalog", detail)


def check_vast_balance(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    from lab.backends.skypilot import vast_balance

    try:
        bal = vast_balance()
    except Exception as e:  # noqa: BLE001
        return _skip("balance", f"could not read Vast balance: {str(e)[:160]}")
    if bal is None:
        return _skip("balance", "Vast balance unavailable")
    if bal <= 0:
        return _fail(
            "balance",
            f"Vast account balance is ${bal:.2f}",
            "top up at https://cloud.vast.ai/billing — rentals are rejected at zero",
        )
    return _ok("balance", f"${bal:.2f}")


def check_do_token(res: ResourceRequest, ctx: dict[str, Any]) -> CheckResult:
    path = Path.home() / ".config" / "doctl" / "config.yaml"
    if os.environ.get("DIGITALOCEAN_TOKEN") or path.exists():
        return _ok("token", "DigitalOcean credentials present")
    return _fail(
        "token",
        "no DigitalOcean token found",
        "doctl auth init (writes ~/.config/doctl/config.yaml), or set DIGITALOCEAN_TOKEN",
    )


# Ordered: later checks read what earlier ones put in ``ctx`` (credentials, the price estimate
# that tells the quota checks which regions are worth asking about).
_REGISTRY: dict[str, tuple[tuple[str, Any], ...]] = {
    "gcp": (
        ("adc", check_adc),
        ("project", check_project),
        ("catalog", check_catalog),
        ("apis", check_apis),
        ("billing", check_billing),
        ("iam", check_iam),
        ("quota_cpu", check_quota_cpu),
        ("quota_gpu", check_quota_gpu),
        ("quota_disk", check_quota_disk),
        ("sky_daemon", check_sky_daemon),
    ),
    "vast": (("balance", check_vast_balance), ("catalog", check_catalog)),
    "do": (("token", check_do_token), ("catalog", check_catalog)),
    "local": (),
}

def run_checks(
    cloud: str | None,
    resources: ResourceRequest | None = None,
    *,
    home: Path | None = None,
    quick: bool = False,
    use_cache: bool = True,
) -> list[CheckResult]:
    """Run a cloud's preflight checks in order.

    ``quick`` restricts to :data:`QUICK_CHECKS` — the subset cheap and decisive enough to run on
    the submit path. Results are cached per (cloud, project, check) with per-check TTLs so the
    automatic preflight usually costs no API calls at all.
    """
    key = cloud or "vast"
    checks = _REGISTRY.get(key)
    if checks is None:
        return [_skip("cloud", f"no checks defined for {key!r}")]
    res = resources or ResourceRequest(cloud=cloud)
    cache = CheckCache.for_home(home) if (home is not None and use_cache) else None

    ctx: dict[str, Any] = {}
    out: list[CheckResult] = []
    for name, fn in checks:
        if quick and name not in QUICK_CHECKS:
            continue
        cacheable = cache if name not in _NEVER_CACHE else None
        shape = _shape_key(res) if name in _SHAPE_DEPENDENT else "-"
        cache_key = f"{key}|{ctx.get('project', '-')}|{name}|{shape}"
        cached = cacheable.get(cache_key) if cacheable is not None else None
        if cached is not None:
            out.append(cached)
            events.note("doctor.check", name=name, status=cached.status, detail=cached.detail)
            continue
        try:
            with _quiet_google_http():
                result = fn(res, ctx)
        except Exception as e:  # noqa: BLE001 — a broken check must never block a launch
            result = _skip(name, f"check errored: {str(e)[:160]}")
        out.append(result)
        events.note("doctor.check", name=name, status=result.status, detail=result.detail)
        # A `skip` means "could not answer", which is not a finding worth remembering.
        if cacheable is not None and result.status != "skip":
            cacheable.put(cache_key, result)
    return out


def blocking_failures(results: list[CheckResult]) -> list[CheckResult]:
    """The subset that should stop a launch — definitive negatives only."""
    return [r for r in results if r.blocking]


def preflight(
    cloud: str | None, resources: ResourceRequest, *, home: Path | None = None
) -> list[CheckResult]:
    """The automatic pre-launch pass. Returns the findings that should block; [] means proceed."""
    if (cloud or "vast") == "local":
        return []
    return blocking_failures(run_checks(cloud, resources, home=home, quick=True))


def doctor_view(cloud: str | None, results: list[CheckResult]) -> dict[str, Any]:
    """One structured shape for both shells (FR-F2), so `lab doctor --json` and the MCP tool
    cannot drift apart. ``ok`` is the single field a caller needs to branch on."""
    return {
        "cloud": cloud or "vast",
        "ok": not any(r.status == "fail" for r in results),
        "checks": [r.model_dump() for r in results],
        "blocking": [r.model_dump() for r in results if r.status == "fail"],
    }


def format_report(results: list[CheckResult]) -> str:
    """Human-readable table. Machine consumers use the models directly / ``--json``."""
    if not results:
        return "no checks ran"
    width = max(len(r.name) for r in results)
    icon = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "skip": "--  "}
    lines = []
    for r in results:
        lines.append(f"  {r.name.ljust(width)}  {icon[r.status]}  {r.detail}")
        if r.fix and r.status in ("fail", "warn"):
            lines.append(f"  {' ' * width}        fix: {r.fix}")
    return "\n".join(lines)
