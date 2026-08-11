"""Placement + pricing — "where will this land, and what will it really cost?"

The lab shipped three remote clouds without ever asking SkyPilot's catalog a question. That
catalog is a local CSV: it needs no credentials, makes no cloud calls, and already knows which
instance type a spec resolves to, which regions offer it, and what each one charges at each
spot-ness. Every gap this module closes came from not asking it.

Two facts drive the design.

**Prices vary by region far more than anyone budgeted for.** A `n4-standard-4` is $0.1814/hr
on-demand almost everywhere, but its *spot* price runs from $0.0340 (europe-west1) to $0.1226 — a
3.6x spread. ``sky.Resources.get_cost()`` with no region pinned deliberately returns the *minimum*
across every region offering the resource, so a spec priced that way is priced at its best case.
That is the wrong direction for a guardrail: an admission check that under-estimates admits jobs
it should refuse. :func:`estimate` therefore reports a *band*, and its consumers check the top.

**SkyPilot's optimizer is already good at this, so this module does not replace it.** It narrows
the search space (a pinned region, a price cap, zones that just ran out of capacity) and prices
what is left. Ordering, failover, and the actual choice stay SkyPilot's. With no pins, no cap, and
an empty memo, the search space handed to SkyPilot is exactly what it is today.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from lab._util import atomic_write_text
from lab.models import ResourceRequest

# --------------------------------------------------------------------------------------------
# Storage pricing
# --------------------------------------------------------------------------------------------
# SkyPilot never passes a disk tier, so its default (MEDIUM) applies, which its GCP adapter maps
# to `pd-balanced` — except n4/a3-ultragpu/a4, which support *only* `hyperdisk-balanced`. Our two
# GCP profiles therefore bill at different rates: the cpu profile lands on n4 (hyperdisk) and the
# GPU path on n1 (pd-balanced). Rates are GCP's published per-GiB-hour prices.
_GCP_PD_BALANCED_USD_GIB_HR = 0.000136986  # $0.10/GiB/month
_GCP_HYPERDISK_BALANCED_USD_GIB_HR = 0.000109589  # $0.08/GiB/month
# Machine families that cannot use pd-balanced and so always bill at the hyperdisk rate.
_GCP_HYPERDISK_ONLY_FAMILIES = ("n4", "a3-ultragpu", "a4")

# DigitalOcean block volumes are $0.10/GiB/month, the same shape as pd-balanced.
_DO_VOLUME_USD_GIB_HR = 0.000136986

# Vast is deliberately 0: we price Vast from the rental's own ``dph_total``, which is the real
# billed rate and already includes its storage. Adding a term here would double-count it.
_VAST_STORAGE_USD_GIB_HR = 0.0

# How long a zone stays excluded after it reports a capacity exhaustion. GCP capacity comes back
# on the order of minutes-to-hours; 30 min is long enough to steer a sweep's remaining shards away
# and short enough that a recovered zone is not blacklisted for a whole session.
DEFAULT_MEMO_TTL_S = 1800.0

# Upper bound on how many regions we hand SkyPilot when narrowing. The optimizer prices every
# candidate locally, so this is about keeping the failover walk bounded, not about API cost.
MAX_NARROWED_REGIONS = 10

# A GCE zone: <region>-<letter>, where a region is <continent-ish>-<direction><digit>.
# Matches us-central1-a, europe-west4-b, northamerica-northeast1-c, me-west1-a.
_ZONE_RE = re.compile(r"\b([a-z]{2,}-[a-z]+\d+-[a-z])\b")

# Markers that mean "this zone has no capacity right now" — a transient, zone-scoped condition
# worth remembering. Quota errors are deliberately NOT here: quota is regional and persistent, so
# excluding a zone for 30 minutes would neither help nor expire correctly.
_EXHAUSTION_MARKERS = (
    "zone_resource_pool_exhausted",
    "does not have enough resources",
    "vm_min_count_not_reached",
)


class PlacementError(ValueError):
    """A placement constraint the user gave is not satisfiable — e.g. an unknown region name.

    Distinct from "we could not price this", which is never an error: an unpriceable spec returns
    None and every guardrail degrades to its previous behaviour. This is raised only for input
    that is wrong on its face, and only before anything can bill.
    """


def storage_hourly_usd(cloud: str | None, disk_gb: int | None, instance_type: str | None) -> float:
    """USD/hour for a job's boot/attached disk. Zero when we have no rate for the cloud.

    This term has never been on the manifest, and it is not small: SkyPilot's default 256 GB disk
    costs $0.0351/hr on pd-balanced, against a $0.0340/hr spot n4-standard-4. An untuned boot disk
    can cost more than the machine it is attached to.
    """
    if not disk_gb:
        return 0.0
    key = cloud or "vast"
    if key == "gcp":
        family = (instance_type or "").split("-")[0:2]
        joined = "-".join(family)
        hyperdisk = (instance_type or "").startswith(
            _GCP_HYPERDISK_ONLY_FAMILIES
        ) or joined in _GCP_HYPERDISK_ONLY_FAMILIES
        rate = (
            _GCP_HYPERDISK_BALANCED_USD_GIB_HR if hyperdisk else _GCP_PD_BALANCED_USD_GIB_HR
        )
        return disk_gb * rate
    if key == "do":
        return disk_gb * _DO_VOLUME_USD_GIB_HR
    return disk_gb * _VAST_STORAGE_USD_GIB_HR


# --------------------------------------------------------------------------------------------
# Capacity memo
# --------------------------------------------------------------------------------------------


class CapacityMemo:
    """Zones that recently reported "no capacity", shared across every job on this host.

    The problem it solves is a sweep: 32 shards submit within seconds of each other, and without a
    shared memory each one independently discovers that us-central1 is exhausted, spending a full
    provision-failover walk to learn what shard 1 already knew.

    **Advisory, never authoritative.** A missing, unreadable, corrupt, or expired memo reads as
    empty, and a failed write is swallowed. The worst thing a broken memo can do is make a launch
    take the path it would have taken anyway; it can never fail one. That is why every method here
    catches broadly instead of propagating.
    """

    FILENAME = "capacity_memo.json"

    def __init__(self, path: Path, *, ttl_s: float | None = None) -> None:
        self.path = Path(path)
        if ttl_s is None:
            try:
                ttl_s = float(os.environ.get("LAB_CAPACITY_MEMO_TTL_S", DEFAULT_MEMO_TTL_S))
            except ValueError:
                ttl_s = DEFAULT_MEMO_TTL_S
        self.ttl_s = ttl_s

    @classmethod
    def for_home(cls, home: Path, *, ttl_s: float | None = None) -> CapacityMemo:
        return cls(Path(home) / cls.FILENAME, ttl_s=ttl_s)

    @staticmethod
    def _key(cloud: str, instance_type: str, zone: str) -> str:
        # Keyed on the instance type because exhaustion is per-machine-shape: n4-standard-4 being
        # out in us-central1-a says nothing about whether n1-highmem-4 is.
        return f"{cloud}|{instance_type}|{zone}"

    def _load(self) -> dict[str, float]:
        try:
            raw = json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001 — absent/corrupt/unreadable all mean "nothing known"
            return {}
        entries = raw.get("entries") if isinstance(raw, dict) else None
        if not isinstance(entries, dict):
            return {}
        out: dict[str, float] = {}
        for k, v in entries.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def _live(self, entries: dict[str, float], *, now_s: float) -> dict[str, float]:
        return {k: t for k, t in entries.items() if now_s - t < self.ttl_s}

    def exhausted_zones(
        self, cloud: str, instance_type: str, *, now_s: float | None = None
    ) -> set[str]:
        """Zones for this (cloud, instance_type) whose exhaustion is still within the TTL."""
        now_s = time.time() if now_s is None else now_s
        prefix = f"{cloud}|{instance_type}|"
        return {
            k[len(prefix) :]
            for k in self._live(self._load(), now_s=now_s)
            if k.startswith(prefix)
        }

    def record(
        self, cloud: str, instance_type: str, zones: Iterable[str], *, now_s: float | None = None
    ) -> None:
        """Remember that these zones just ran out of capacity. Best-effort; never raises."""
        zones = list(zones)
        if not zones:
            return
        now_s = time.time() if now_s is None else now_s
        try:
            entries = self._live(self._load(), now_s=now_s)  # prune expired while we are here
            for z in zones:
                entries[self._key(cloud, instance_type, z)] = now_s
            atomic_write_text(
                self.path, json.dumps({"version": 1, "entries": entries}, sort_keys=True)
            )
        except Exception as e:  # noqa: BLE001 — a memo write must never fail a job
            print(f"[lab] capacity memo write skipped: {e}")


def parse_exhausted_zones(text: str) -> list[str]:
    """Zone names that a failed launch's log blamed for a *capacity* shortfall. Pure.

    SkyPilot surfaces GCE's exhaustion two ways in the same line — ``in us-central1-a:`` and
    ``'projects/x/zones/us-central1-a'`` — so we scan any line carrying an exhaustion marker and
    take every zone-shaped token on it. Lines without a marker are ignored, which keeps the
    "⚙️ Launching on GCP us-central1 (us-central1-a)" progress lines (which name zones that may
    have worked fine) from poisoning the memo.
    """
    found: list[str] = []
    for line in text.splitlines():
        low = line.lower()
        if not any(m in low for m in _EXHAUSTION_MARKERS):
            continue
        for zone in _ZONE_RE.findall(low):
            if zone not in found:
                found.append(zone)
    return found


# --------------------------------------------------------------------------------------------
# Catalog access
# --------------------------------------------------------------------------------------------


def _catalog() -> Any:
    """Import ``sky.catalog`` lazily. Test seam — monkeypatch me to inject a fake catalog.

    Function-local so this module imports on a host without the skypilot extra (the scheduler
    reaches for pricing there, and the CLI imports it unconditionally).
    """
    from sky import catalog

    return catalog


def _catalog_cpus(res: ResourceRequest) -> str | None:
    return str(res.cpus) if res.cpus else None


def _catalog_memory(res: ResourceRequest) -> str | None:
    """Normalise ``"32GB"`` to the bare ``"32"`` the catalog wants, preserving a trailing ``+``."""
    if not res.memory:
        return None
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(?:gb|gib|g)?\s*(\+?)\s*$", str(res.memory), re.I)
    return f"{m.group(1)}{m.group(2)}" if m else str(res.memory)


def _split_accelerators(spec: str | None) -> tuple[str, int] | None:
    if not spec:
        return None
    name, _, count = spec.partition(":")
    try:
        return name, int(count) if count else 1
    except ValueError:
        return name, 1


def resolve_instance_type(res: ResourceRequest) -> str | None:
    """The instance type this spec will actually land on, or None if the catalog can't say.

    None is a normal answer, not an error: a spec the catalog cannot resolve simply goes unpriced,
    and every consumer degrades to the behaviour it had before this module existed.
    """
    cloud = res.cloud or "vast"
    try:
        cat = _catalog()
        accel = _split_accelerators(res.accelerators)
        if accel is not None:
            types, _fuzzy = cat.get_instance_type_for_accelerator(
                accel[0],
                accel[1],
                cpus=_catalog_cpus(res),
                memory=_catalog_memory(res),
                use_spot=res.use_spot,
                region=res.region,
                zone=res.zone,
                clouds=cloud,
            )
            return str(types[0]) if types else None
        found = cat.get_default_instance_type(
            cpus=_catalog_cpus(res),
            memory=_catalog_memory(res),
            region=res.region,
            zone=res.zone,
            use_spot=res.use_spot,
            clouds=cloud,
        )
        return str(found) if found else None
    except Exception as e:  # noqa: BLE001 — unpriceable is a normal outcome, not a failure
        print(f"[lab] catalog could not resolve an instance type for {cloud}: {e}")
        return None


def validate_placement(res: ResourceRequest) -> None:
    """Reject a region/zone the cloud does not have, before anything provisions.

    The one hard failure in this module, and deliberately so: it is a user typo, and catching it
    here costs nothing while catching it at launch costs a provision.
    """
    if res.region is None and res.zone is None:
        return
    cloud = res.cloud or "vast"
    try:
        _catalog().validate_region_zone(res.region, res.zone, clouds=cloud)
    except ImportError:  # no skypilot extra here; nothing to validate against
        return
    except Exception as e:  # noqa: BLE001 — sky raises bare ValueError with a helpful message
        raise PlacementError(f"invalid --region/--zone for {cloud}: {e}") from e


@dataclass(frozen=True)
class Candidate:
    region: str
    zones: tuple[str, ...]
    hourly_usd: float


def _price(cat: Any, instance_type: str, res: ResourceRequest, region: str, spot: bool) -> float:
    """Compute-only price for one region: the instance plus, on clouds that bill them apart, the
    accelerators. GCP attaches GPUs as a separate SKU, so omitting the second term would price a
    T4 box as a bare n1."""
    total = float(
        cat.get_hourly_cost(
            instance_type, use_spot=spot, region=region, zone=None, clouds=res.cloud or "vast"
        )
    )
    accel = _split_accelerators(res.accelerators)
    if accel is not None:
        try:
            total += float(
                cat.get_accelerator_hourly_cost(
                    accel[0],
                    accel[1],
                    use_spot=spot,
                    region=region,
                    zone=None,
                    clouds=res.cloud or "vast",
                )
            )
        except Exception:  # noqa: BLE001 — some catalogs fold the GPU into the instance price
            pass
    return total


def candidates(
    res: ResourceRequest,
    *,
    instance_type: str,
    memo: CapacityMemo | None = None,
    spot: bool | None = None,
) -> list[Candidate]:
    """Regions that can host this spec, priced and sorted cheapest-first.

    Applies, in order: the user's region/zone pin, the capacity memo (a region drops out only when
    *every* one of its zones is excluded — a single dead zone still leaves the region usable), and
    the price cap. Returns [] when nothing survives, which callers treat as "do not narrow".
    """
    cloud = res.cloud or "vast"
    use_spot = res.use_spot if spot is None else spot
    try:
        cat = _catalog()
        accel = _split_accelerators(res.accelerators)
        if accel is not None:
            regions = cat.get_region_zones_for_accelerators(
                accel[0], accel[1], use_spot=use_spot, clouds=cloud
            )
        else:
            regions = cat.get_region_zones_for_instance_type(
                instance_type, use_spot=use_spot, clouds=cloud
            )
    except Exception as e:  # noqa: BLE001 — no candidates is a valid degraded answer
        print(f"[lab] catalog region lookup failed for {cloud}: {e}")
        return []

    excluded = memo.exhausted_zones(cloud, instance_type) if memo is not None else set()
    out: list[Candidate] = []
    for region in regions:
        name = str(getattr(region, "name", ""))
        if not name:
            continue
        if res.region is not None and name != res.region:
            continue
        zone_names = tuple(str(z.name) for z in (getattr(region, "zones", None) or []))
        if res.zone is not None and res.zone not in zone_names:
            continue
        live = tuple(z for z in zone_names if z not in excluded)
        # A region with no zone list at all (some catalogs omit them) is kept: we have nothing to
        # exclude on, so excluding it would be guessing.
        if zone_names and not live:
            continue
        try:
            price = _price(cat, instance_type, res, name, use_spot)
        except Exception:  # noqa: BLE001 — region simply has no listed price for this shape
            continue
        if res.max_hourly_usd is not None and price > res.max_hourly_usd:
            continue
        out.append(Candidate(region=name, zones=live or zone_names, hourly_usd=price))
    out.sort(key=lambda c: c.hourly_usd)
    return out


@dataclass(frozen=True)
class Estimate:
    """A priced view of where a spec can land.

    ``low_usd`` is the best case at the spot-ness the user asked for; ``high_usd`` is the **worst
    admissible** compute rate. They are different numbers for a reason — the consumers of this
    estimate are cost guardrails, and a guardrail that checks the best case is not a guardrail.
    """

    instance_type: str
    low_usd: float
    high_usd: float
    storage_usd: float
    regions: int
    excluded_zones: int
    basis: str

    @property
    def worst_hourly_usd(self) -> float:
        """What admission control checks: the worst compute rate plus storage."""
        return self.high_usd + self.storage_usd

    @property
    def best_hourly_usd(self) -> float:
        return self.low_usd + self.storage_usd


def estimate(res: ResourceRequest, *, memo: CapacityMemo | None = None) -> Estimate | None:
    """Price a spec as a band. None when the catalog cannot price it (a normal outcome).

    The worst case deliberately accounts for **spot fallback**. ``spot_fallback`` defaults True, so
    ``--spot`` means "spot, or on-demand if spot is scarce" — and on GCP on-demand is ~5x spot. A
    worst case computed at spot prices would therefore be an estimate of the outcome the user hopes
    for rather than the one they have authorised, so when fallback is live the ceiling is priced
    on-demand.
    """
    instance_type = resolve_instance_type(res)
    if instance_type is None:
        return None
    cloud = res.cloud or "vast"

    best = candidates(res, instance_type=instance_type, memo=memo)
    if not best:
        return None

    # The ceiling is priced at the most expensive kind we could actually be billed for.
    ceiling_is_spot = res.use_spot and not res.spot_fallback
    worst_pool = (
        best
        if ceiling_is_spot == res.use_spot
        else candidates(res, instance_type=instance_type, memo=memo, spot=ceiling_is_spot)
    )
    if not worst_pool:
        worst_pool = best

    low = best[0].hourly_usd
    high = max(c.hourly_usd for c in worst_pool)
    if res.max_hourly_usd is not None:
        # The cap is enforced by SkyPilot's optimizer, so it is a real ceiling, not a hope.
        high = min(high, res.max_hourly_usd)

    excluded = 0
    if memo is not None:
        excluded = len(memo.exhausted_zones(cloud, instance_type))

    kind = "spot" if res.use_spot else "on-demand"
    if res.use_spot and res.spot_fallback:
        kind = "spot w/ on-demand fallback"
    disk = storage_hourly_usd(cloud, res.disk_size, instance_type)
    where = res.zone or res.region or f"{len(best)} regions"
    basis = (
        f"{cloud} catalog {instance_type} {kind} {where}"
        f" + {res.disk_size or 0}GiB disk (${disk:.4f}/hr)"
    )
    return Estimate(
        instance_type=instance_type,
        low_usd=low,
        high_usd=high,
        storage_usd=disk,
        regions=len(best),
        excluded_zones=excluded,
        basis=basis,
    )
