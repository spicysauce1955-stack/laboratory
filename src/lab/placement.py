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

Two memos live here, and they are not the same shape. :class:`CapacityMemo` latches a *zone* that
said "no capacity" for 30 minutes. :class:`DeadHostMemo` counts strikes against a *host* that
took a whole provisioning slot and never came up — a failure that bills $0, which is precisely
why nothing else in the system notices it. Both are advisory in the same strict sense: a broken,
missing or unwritable memo reads as no memory and can never be the reason a launch fails.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar, cast

from lab._util import atomic_write_text
from lab.models import ResourceRequest

_F = TypeVar("_F", bound=Callable[..., Any])


def _note(message: str) -> None:
    """Diagnostics go to **stderr**, never stdout.

    The CLI emits its results as JSON on stdout and callers parse it. A stray "[lab] catalog price
    unavailable" on stdout is not a log line, it is a corrupted payload — which is exactly what
    happened once pricing started running on the `lab register` path.

    The same line is buffered into the event ledger, where it survives the terminal scrolling.
    """
    from lab import events

    print(message, file=sys.stderr)
    events.note("placement.warn", message=message)


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

# Clouds that bill attached storage as a separate line item. On these, letting `disk_size` fall
# through to SkyPilot's 256 GB default is never right: on DO it hard-failed with a 422 (which is
# why the cpu profile pins 50), and on GCP it succeeds quietly at $0.028/hr (hyperdisk, what n4
# must use) to $0.035/hr (pd-balanced) — 82-103% of the $0.034/hr spot n4-standard-4 it is
# attached to, so it roughly doubles a cheap job's bill. Loud on one cloud, silent on the other.
STORAGE_BILLING_CLOUDS = ("gcp", "do")
# Keep the cpu default inside a fresh DigitalOcean account's tier: 8-vCPU sizes and a 256 GB block
# volume are both tier-restricted and fail provisioning until a tier bump.
CPU_DEFAULT_DISK_GB = 50
# Accelerated jobs need room for a CUDA image plus wheels and a checkpoint; 100 GB is comfortable
# without paying for SkyPilot's 256.
GPU_DEFAULT_DISK_GB = 100


def effective_disk_gb(res: ResourceRequest) -> int | None:
    """The ``disk_size`` a spec should actually launch with, or None to leave it alone.

    Only clouds that bill storage separately get a default: on Vast the disk is part of the
    rental's ``dph_total``, so imposing a size there would change provisioning behaviour to fix a
    cost problem Vast does not have.

    This lives here, not only in ``resolve_backend_profile``, because that function is on the
    CLI/MCP submit path and **the scheduler's launch path does not call it** — a registered GCP
    job would otherwise still inherit SkyPilot's 256 GB. Pure.
    """
    if res.disk_size is not None:
        return res.disk_size
    if (res.cloud or "vast") not in STORAGE_BILLING_CLOUDS:
        return None
    applied = GPU_DEFAULT_DISK_GB if res.accelerators else CPU_DEFAULT_DISK_GB
    from lab import events

    events.note("placement.disk_override", requested=res.disk_size, applied=applied)
    return applied


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
    costs $0.028/hr (hyperdisk) to $0.035/hr (pd-balanced) — 82-103% of a $0.034/hr spot
    n4-standard-4. An untuned boot disk is the same order of cost as the machine it hangs off.
    """
    if not disk_gb:
        return 0.0
    key = cloud or "vast"
    if key == "gcp":
        # Match on whole family tokens, not a prefix: `n4-standard-4` is hyperdisk-only but
        # `n4a-standard-4` is a different family that is not, and a `startswith("n4")` would
        # quietly price it 20% low.
        tokens = (instance_type or "").split("-")
        hyperdisk = tokens[0] in _GCP_HYPERDISK_ONLY_FAMILIES or (
            "-".join(tokens[:2]) in _GCP_HYPERDISK_ONLY_FAMILIES
        )
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
        from lab import events

        zones = list(zones)
        if not zones:
            return
        now_s = time.time() if now_s is None else now_s
        try:
            entries = self._live(self._load(), now_s=now_s)  # prune expired while we are here
            for z in zones:
                entries[self._key(cloud, instance_type, z)] = now_s
                events.note("placement.zone_exhausted", zone=z)
            atomic_write_text(
                self.path, json.dumps({"version": 1, "entries": entries}, sort_keys=True)
            )
        except Exception as e:  # noqa: BLE001 — a memo write must never fail a job
            _note(f"[lab] capacity memo write skipped: {e}")


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
# Dead-host memo
# --------------------------------------------------------------------------------------------
# A dead offer bills $0 and so nothing in the cost machinery notices it, but it costs a whole
# provisioning slot: over five days of the 2026-09 campaign, 240 of 571 Vast launches (42%) died
# at the provisioning watchdog, ~4-8 minutes each.
#
# **What identity is available is the whole design question**, and the honest answer is: very
# little. Everything the lab saw about the host that took its 240s is in the job log, and a real
# one (runs/20260903-200140-f01cf6/logs.txt, a "provisioning exceeded 240s" job) reads in full:
#
#     Running on cluster: lab-tempotr-5171-20260903-200140-f01cf6
#     Considered resources (1 node):
#      INFRA                    INSTANCE               ... GPUS        COST ($)   CHOSEN
#      Vast (Romania, RO, EU)   1x-RTX_3090-32-65536   ... RTX3090:1   0.25          ✔
#     ⚙︎ Launching on Vast Romania, RO, EU.
#     Cancelling 1 request: '9ce01952-…'
#
# No instance id, no machine id, no host id — only a *placement*: cloud, instance type, region.
# The one per-host identity that exists at all is the Vast rental's ``machine_id``, and it can
# only be had by asking Vast while the rental is still alive (see ``lab.sky_runner``); the
# catalog, which is all this module may touch, does not know it.
#
# So the memo takes three kinds of key and trusts them very differently:
#
#   ``machine``    a real physical host (Vast ``machine_id``). Sharp. Decides.
#   ``accel``      the accelerator spec that was asked for. What rotation reads.
#   ``placement``  (instance_type, region). Coarse. Recorded as evidence, never used to exclude.
#
# The second half of that is a measurement, not a preference. Replaying the campaign's 571 Vast
# launches: after three failures on one placement key within 30 minutes, the next launch on it
# died 48% of the time against a 42% base rate — barely a signal. And the placement with by far
# the most failures (Maryland, US, NA / 1x-RTX_4090, 160 dead) is also the *best* pool anyone had:
# 37% dead, against Romania's 55% and CA's 75%. A memo that excluded the loudest offender would
# have steered the campaign out of its healthiest supply and into its worst. Recording a
# placement is worth doing — it is what ``lab report`` can group on — but excluding on it is not.

# How long a strike is remembered. Longer than :data:`DEFAULT_MEMO_TTL_S` (30 min) on purpose:
# a zone's capacity comes back on its own, while a host with a bad driver, a wedged disk or a GPU
# it cannot actually expose is *broken*, and the campaign's two named offenders were re-drawn
# across days. Six hours keeps it inside one working session, so a machine its owner repairs
# overnight is not blacklisted forever.
DEFAULT_DEAD_HOST_TTL_S = 21600.0

# How many strikes inside the TTL make a host dead. Two, because one failure is indistinguishable
# from bad luck — the same campaign's data has placements failing ~40% of the time and succeeding
# on the next attempt — and because the cost of being wrong is asymmetric: a wrongly-skipped host
# costs one extra launch attempt, a wrongly-trusted one costs a whole provisioning slot.
DEFAULT_DEAD_HOST_STRIKES = 2

# Bounds on the file. Both exist so a runaway sweep cannot turn an advisory cache into a disk
# problem; both drop the *oldest* evidence first.
MAX_STRIKES_KEPT = 10
MAX_MEMO_KEYS = 200

# "⚙︎ Launching on Vast Romania, RO, EU." / "⚙️ Launching on GCP us-central1 (us-central1-a)."
# Both shapes are real, taken from this machine's logs; the trailing period is SkyPilot's.
_LAUNCHING_RE = re.compile(r"Launching on\s+(?:Vast|GCP|DO|AWS|Azure|Kubernetes|\w+)\s+(.+?)\.?\s*$")
# The CHOSEN row of SkyPilot's "Considered resources" table. Vast instance types are
# ``1x-RTX_3090-32-65536``; GCP's are ``n1-highmem-4`` (sometimes ``…[Spot]``).
_CONSIDERED_RE = re.compile(
    r"^\s*(?:Vast|GCP|DO|AWS|Azure)\s*(?:\([^)]*\))?\s+([A-Za-z0-9][\w.\-]*(?:\[Spot\])?)\s+\d"
)


@dataclass(frozen=True)
class DrawnPlacement:
    """What a launch log says the optimizer actually drew. Either half may be None."""

    instance_type: str | None
    region: str | None


def parse_drawn_placement(text: str) -> DrawnPlacement:
    """The instance type and region a launch was waiting on when it died. Pure.

    The **last** ``Launching on`` line wins: SkyPilot prints one per failover hop, so on GCP the
    first names a zone that already failed while the last names the one we were still waiting for.
    Returns ``DrawnPlacement(None, None)`` when the log says nothing — a normal outcome (a launch
    that never got as far as choosing), never an error.
    """
    instance_type: str | None = None
    region: str | None = None
    for raw in text.splitlines():
        line = _ANSI_RE.sub("", raw).rstrip()
        m = _LAUNCHING_RE.search(line)
        if m:
            region = m.group(1).strip() or None
            continue
        c = _CONSIDERED_RE.search(_strip_log_prefix(line))
        if c:
            instance_type = c.group(1)
    return DrawnPlacement(instance_type=instance_type, region=region)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Job logs are timestamped by `install_log_redaction` ("2026-09-03T20:01:42.246Z …"); the
# "Considered resources" table is column-aligned, so the stamp has to come off before matching.
_LOG_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s")


def _strip_log_prefix(line: str) -> str:
    return _LOG_PREFIX_RE.sub("", line)


def machine_key(cloud: str, machine_id: str | int) -> str:
    """A key naming one physical host. The only *sharp* identity the lab can obtain."""
    return f"{cloud}|machine:{machine_id}"


def placement_key(cloud: str, instance_type: str | None, region: str | None) -> str:
    """A key naming a (cloud, instance type, region) — evidence, not a verdict (see above).

    Never a price. The campaign's own workaround keyed on one ("skip the $0.3585 host") and its
    retrospective flags that as fragile: good hosts recur at the same price as bad ones, and a
    price is a property of an offer at an instant, not of a machine.
    """
    return f"{cloud}|placement:{instance_type or '?'}@{region or '?'}"


def accelerator_key(cloud: str, accelerators: str) -> str:
    """A key naming an accelerator *request* — what the user asked for, not where it landed.

    The rotation policy needs a "how has asking for this been going lately?" question, and at the
    moment it has to choose there is no placement yet: the region and instance type are the
    optimizer's answer, not the caller's input. So this keys on the SkyPilot accelerator spec
    exactly as written on the manifest (``"RTX4090:1"``), which is a real string the lab holds,
    rather than on a guess at the instance type it will resolve to.
    """
    return f"{cloud}|accel:{accelerators}"


class DeadHostMemo:
    """Hosts that failed to come up, remembered with strikes and a TTL. Shared across jobs.

    The same contract as :class:`CapacityMemo`, and for the same reason: **advisory, never
    authoritative**. A missing, unreadable, corrupt, expired or unwritable memo reads as empty and
    a failed write is swallowed, so the worst a broken memo can do is let a launch take the path
    it would have taken anyway. It can never fail one, and it can never be the reason a job does
    not get a machine.

    Unlike the capacity memo this counts, rather than latches: one failure is bad luck (a
    placement that just failed still succeeded ~half the time in the campaign data), so a key is
    "dead" only after :data:`DEFAULT_DEAD_HOST_STRIKES` failures inside the TTL. Strikes expire
    individually, so a host that failed twelve hours ago and once just now is one strike from
    dead, not two.
    """

    FILENAME = "dead_host_memo.json"

    def __init__(self, path: Path, *, ttl_s: float | None = None, strikes: int | None = None):
        self.path = Path(path)
        if ttl_s is None:
            ttl_s = _env_float("LAB_DEAD_HOST_TTL_S", DEFAULT_DEAD_HOST_TTL_S)
        self.ttl_s = ttl_s
        if strikes is None:
            strikes = int(_env_float("LAB_DEAD_HOST_STRIKES", DEFAULT_DEAD_HOST_STRIKES))
        # A threshold below 1 would make every host dead on sight, including hosts that have
        # never failed — the one thing this must never do.
        self.strike_limit = max(1, strikes)

    @classmethod
    def for_home(
        cls, home: Path, *, ttl_s: float | None = None, strikes: int | None = None
    ) -> DeadHostMemo:
        return cls(Path(home) / cls.FILENAME, ttl_s=ttl_s, strikes=strikes)

    def _load(self) -> dict[str, list[float]]:
        try:
            raw = json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001 — absent/corrupt/unreadable all mean "nothing known"
            return {}
        entries = raw.get("entries") if isinstance(raw, dict) else None
        if not isinstance(entries, dict):
            return {}
        out: dict[str, list[float]] = {}
        for k, v in entries.items():
            if not isinstance(v, list):
                continue
            stamps: list[float] = []
            for t in v:
                try:
                    stamps.append(float(t))
                except (TypeError, ValueError):
                    continue  # one bad stamp drops itself, not the entry
            if stamps:
                out[str(k)] = stamps
        return out

    def _live(self, entries: dict[str, list[float]], *, now_s: float) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for k, stamps in entries.items():
            fresh = [t for t in stamps if now_s - t < self.ttl_s]
            if fresh:
                out[k] = fresh[-MAX_STRIKES_KEPT:]
        return out

    def strikes(self, key: str, *, now_s: float | None = None) -> int:
        """How many failures this key has collected inside the TTL."""
        now_s = time.time() if now_s is None else now_s
        return len(self._live(self._load(), now_s=now_s).get(key, []))

    def is_dead(self, key: str, *, now_s: float | None = None) -> bool:
        return self.strikes(key, now_s=now_s) >= self.strike_limit

    def dead_keys(
        self, cloud: str, *, kind: str | None = None, now_s: float | None = None
    ) -> set[str]:
        """Every struck-out key for this cloud, optionally narrowed to one kind."""
        now_s = time.time() if now_s is None else now_s
        prefix = f"{cloud}|" if kind is None else f"{cloud}|{kind}:"
        return {
            k
            for k, stamps in self._live(self._load(), now_s=now_s).items()
            if k.startswith(prefix) and len(stamps) >= self.strike_limit
        }

    def has_any(self, cloud: str, *, kind: str | None = None, now_s: float | None = None) -> bool:
        """Is there anything to avoid on this cloud at all?

        The cheap gate callers use to keep the no-memory case byte-identical to the behaviour
        before this existed: a launch with nothing recorded must not pay for a single extra call.
        """
        return bool(self.dead_keys(cloud, kind=kind, now_s=now_s))

    def record(self, key: str, *, now_s: float | None = None) -> None:
        """Add a strike against ``key``. Best-effort; never raises."""
        from lab import events

        now_s = time.time() if now_s is None else now_s
        try:
            entries = self._live(self._load(), now_s=now_s)  # prune expired while we are here
            entries.setdefault(key, []).append(now_s)
            entries[key] = entries[key][-MAX_STRIKES_KEPT:]
            if len(entries) > MAX_MEMO_KEYS:
                # Evict whole keys oldest-last-strike first: the least recent evidence is the
                # least useful, and an unbounded cache is a different bug from the one we fix.
                ordered = sorted(entries.items(), key=lambda kv: max(kv[1]), reverse=True)
                entries = dict(ordered[:MAX_MEMO_KEYS])
            # The field is `host`, not `key`: `lab.events.sanitize` is a deny-list and masks a
            # param literally named "key" as a probable secret — which would have redacted the
            # one thing this note exists to record.
            events.note("placement.dead_host", host=key, strikes=len(entries.get(key, [])))
            atomic_write_text(
                self.path, json.dumps({"version": 1, "entries": entries}, sort_keys=True)
            )
        except Exception as e:  # noqa: BLE001 — a memo write must never fail a job
            _note(f"[lab] dead-host memo write skipped: {e}")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# --------------------------------------------------------------------------------------------
# Accelerator rotation
# --------------------------------------------------------------------------------------------
# The campaign's operating rule was a human: under a dead-host storm, hand-rotate the accelerator
# (RTX4090 -> 3090 -> 5090 -> L40S) because a different `gpu_name` makes SkyPilot's Vast adapter
# search a different pool of offers. This promotes that rule to an option — and *only* an option.
#
# The campaign's own data does not support making it a default. Replaying those 571 launches:
# during a storm (3+ dead provisions on one accelerator within 30 min), launches that stayed on
# the storming accelerator died 51% of the time and launches that rotated away died **69%** of the
# time (n=16, and confounded — the human rotated precisely when things were worst, and rotated
# into 3090/5090 pools that are worse in general: 55% and 75% dead against the 4090's 37%). That
# is far too weak to turn on for everyone, and strong enough to warn about. So: no pool, no
# rotation, and the default remains exactly one launch attempt.


def parse_pool(spec: str | Iterable[str] | None) -> list[str]:
    """Normalise ``"RTX4090:1,RTX3090:1"`` (or a list) into an ordered, de-duplicated pool. Pure.

    Duplicates collapse because the attempt bound is spent per *entry*: trying the same
    accelerator twice under the guise of rotation would burn the budget on the thing that just
    failed.
    """
    if spec is None:
        return []
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return out


def rotated_resources(res: ResourceRequest, accelerators: str) -> ResourceRequest:
    """``res`` with a different accelerator and **nothing else changed**.

    Specifically not ``max_hourly_usd``: rotating *what* you ask for is the feature; quietly
    agreeing to pay more is not. Escalating a price cap under a dead-host storm was a deliberate
    human decision every time in the campaign, and it stays one.
    """
    return res.model_copy(update={"accelerators": accelerators})


def affordable_under_cap(res: ResourceRequest, accelerators: str) -> bool:
    """Could ``accelerators`` ever launch under ``res``'s price cap? True when unsure.

    Two deliberate polarities:

    * **Unsure never blocks.** An accelerator this catalog cannot price is offered to SkyPilot
      anyway, which enforces the cap itself. (The ``lab doctor`` rule: only definitive negatives.)
    * **The floor, not the ceiling.** Everywhere else in this module a guardrail checks the top of
      the price band, because an admission check that under-estimates admits jobs it should
      refuse. Here the question is *feasibility* — "is there any region where this fits?" — and
      checking the ceiling would refuse a rotation the cap actually permits.
    """
    cap = res.max_hourly_usd
    if cap is None:
        return True
    # Priced with the cap *removed*: `estimate` filters candidates by the cap itself, so a capped
    # estimate answers None both when nothing fits and when the catalog has never heard of the
    # accelerator — and those two must not be treated alike (one is a definitive no, the other is
    # "we cannot say", which never blocks).
    probe = res.model_copy(update={"accelerators": accelerators, "max_hourly_usd": None})
    est = estimate(probe)
    if est is None:
        return True  # the catalog cannot say; SkyPilot will enforce the cap
    return est.best_hourly_usd <= cap


def next_accelerator(
    *,
    pool: Iterable[str],
    tried: Iterable[str],
    res: ResourceRequest,
    memo: DeadHostMemo | None = None,
    now_s: float | None = None,
) -> str | None:
    """The next accelerator to try, or None when the pool is spent.

    Order is the user's. Entries already tried on this launch are skipped (the bound must not
    wrap — a bound that cycles is not a bound), as are entries the catalog says can never fit
    under the price cap: spending an attempt on those buys a slower failure, not a machine.

    Entries whose placement is currently struck out in ``memo`` are *deferred*, not dropped — if
    every remaining entry is struck out, the first untried one is returned anyway. A memo that
    would exclude every candidate must degrade to "no memory", exactly as
    :func:`lab.backends.skypilot.narrowed_regions` ignores a capacity memo that would exclude
    every region. Refusing to launch is never the memo's call to make.
    """
    already = set(tried)
    remaining = [
        a for a in pool if a not in already and affordable_under_cap(res, a)
    ]
    if not remaining:
        return None
    if memo is not None:
        cloud = res.cloud or "vast"
        healthy = [
            a for a in remaining if not memo.is_dead(accelerator_key(cloud, a), now_s=now_s)
        ]
        if healthy:
            return healthy[0]
    return remaining[0]


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


def _quiet(fn: _F) -> _F:
    """Keep SkyPilot's own chatter off stdout for the duration of a catalog call.

    The first time a machine needs a catalog CSV, SkyPilot downloads it and announces itself on
    **stdout** ("Updating Vast catalog: vast/vms.csv"). Our stdout is a JSON payload, so on any
    host with a cold catalog — a fresh laptop, a newly provisioned scheduler droplet — that
    chatter corrupts what callers parse: `json.loads` of `lab register`'s output fails on the very
    first run of a machine's life and works ever after, which is the worst shape a bug can have.

    ``_note`` already holds this line for our own diagnostics; this extends it to the vendor's.
    Decorating the whole function (rather than the one call) also covers anything a future edit
    adds to the body.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)

    return cast(_F, wrapper)


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


@_quiet
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
        _note(f"[lab] catalog could not resolve an instance type for {cloud}: {e}")
        return None


@_quiet
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


@_quiet
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
    from lab import events

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
        _note(f"[lab] catalog region lookup failed for {cloud}: {e}")
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
        for _skipped in (z for z in zone_names if z in excluded):
            events.note("placement.zone_skipped", zone=_skipped, reason="exhausted_memo")
        # A region with no zone list at all (some catalogs omit them) is kept: we have nothing to
        # exclude on, so excluding it would be guessing.
        if zone_names and not live:
            continue
        try:
            price = _price(cat, instance_type, res, name, use_spot)
        except Exception:  # noqa: BLE001 — region simply has no listed price for this shape
            continue
        events.note("placement.priced", instance=instance_type, region=name, hourly_usd=price)
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
    # Price the disk the launch will actually get, not the one the spec happens to name — a
    # registration carries no disk_size and would otherwise be quoted as if storage were free.
    disk_gb = effective_disk_gb(res)
    disk = storage_hourly_usd(cloud, disk_gb, instance_type)
    where = res.zone or res.region or f"{len(best)} regions"
    basis = (
        f"{cloud} catalog {instance_type} {kind} {where}"
        f" + {disk_gb or 0}GiB disk (${disk:.4f}/hr)"
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
