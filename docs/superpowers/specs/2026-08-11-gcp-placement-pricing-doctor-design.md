# GCP placement, pricing, and preflight — design

**Date:** 2026-08-11
**Closes:** `GCP-PROV-1` (region/zone control), `GCP-PROV-2` (Vast-calibrated watchdog),
`GCP-PROV-4` (GPU path never run), `GCP-PROV-5` (no preflight), `GCP-COST-1` (booked-price
accuracy), `GCP-COST-2` (GPU disk default), `GCP-COST-4` (`worst_case_cost_usd: null`),
`GCP-PREEMPT-2` (silent spot→on-demand), `GCP-DOC-1` (stale price table).
**Basis:** `docs/proposals/2026-08-11-gcp-backend-gap-schema.md`, the code at `18dd24bb`, the
live GCP project `myproject-505213`, and SkyPilot 0.12.3 as installed.

---

## 1. What the investigation changed

Four items were raised — region control, the GPU disk default, booked-price accuracy, and
`lab doctor`. Reading the code and probing the installed SkyPilot turned them into one system with
a single root cause, and turned up three facts the gap schema did not have.

**The lab never asks the catalog anything.** `sky.catalog` ships with SkyPilot and answers, from a
local CSV with no credentials and no cloud calls: which instance type a spec resolves to
(`get_default_instance_type`, `get_instance_type_for_accelerator`), which regions and zones offer
it (`get_region_zones_for_instance_type`, `…_for_accelerators`), what it costs in a *named* region
at a *named* spot-ness (`get_hourly_cost`), and whether a region/zone string is real
(`validate_region_zone`). Verified live: the cpu profile resolves to `n4-standard-4` across 41 GCP
regions, on-demand $0.1814 in the cheap tier rising to $0.2902, and **spot from $0.0340
(europe-west1) to $0.1226** — the 5× spread the gap schema said "the manifest cannot express" was
in a local file the whole time. `T4:1` resolves to `n1-highmem-4` across 23 regions.

**The cost guardrail is optimistic, which is the wrong direction.** `Resources.get_cost()` with no
region pinned returns the *minimum* across every region that offers the resource — confirmed in
SkyPilot's source, which does this deliberately so that a multi-component price cannot mix regions.
`catalog_hourly` builds its `Resources` without a region. So the number `GCP-COST-3` just wired
into the scheduler's admission control is the globally cheapest price, under-estimating by up to
3.6× on spot. An admission check that under-estimates admits jobs it should refuse. This is a new
finding; the gap schema records `catalog_hourly` as the fix, not as a defect.

**The disk is a first-order cost, not a footnote.** GCP Hyperdisk Balanced provisioned space is
$0.000109589/GiB/hr. SkyPilot's `DEFAULT_DISK_SIZE_GB` is 256, so an untuned boot disk is
**$0.0281/hr** — against a $0.0340/hr spot `n4-standard-4`, the disk nearly doubles the bill. The
gap schema scored `GCP-COST-2` as a leak risk worth ~$25/mo; it is really the largest live
distortion in the cost model, and it is invisible because `hourly_usd` has never included storage.

**Two live quota findings that shape the preflight** (project `myproject-505213`, read through the
compute API):

- Regional `NVIDIA_T4_GPUS` is 1 in both us-central1 and europe-west1, but global
  `GPUS_ALL_REGIONS` is **0**. GCP enforces both; a check reading only regional quota reports "ok"
  and the launch still fails. GPU quota must be checked at both levels.
- Regional `PREEMPTIBLE_CPUS` is **0**, and this is *not* a blocker: GCP documents that in regions
  where preemptible quota was never granted, Spot VMs consume the standard `CPUS` quota. A naive
  check would flag it as fatal and be wrong.

**Already correct, so out of scope.** Post-launch compute pricing. `handle.launched_resources`
carries the region and `get_cost` honours it, so the *actual* per-hour compute rate on a finished
manifest is already region-accurate. `GCP-COST-1`'s real residue is the missing storage line, the
unrecorded zone, and the pre-launch estimate — not the post-launch compute figure.

**And PROV-1 is not what it looks like.** SkyPilot's optimizer already fails over across
regions and zones, cheapest-first. The live run did not lack failover — it had failover *truncated*
by `DEFAULT_PROVISION_TIMEOUT_MIN = 8`, a constant whose docstring justifies it by Vast host
behaviour. Region control that does not also fix that timeout walks into the same wall.

---

## 2. Approach

**Constrain and observe SkyPilot's optimizer; do not rebuild it.**

The rejected alternative was a full planner: compute an ordered, priced candidate list and hand
SkyPilot an explicit ordering. It duplicates an optimizer that is already good at this and will
drift from the catalog as SkyPilot evolves. The other rejected alternative was thin pass-through —
flags only — which leaves every shard of a sweep to rediscover a dead zone independently.

So the lab contributes exactly the three things SkyPilot cannot do for itself:

1. **Say where**, when the user knows better than the optimizer (`--region` / `--zone`).
2. **Say how much is too much** (`--price-cap`), and have the optimizer itself enforce it via
   `Resources(max_hourly_cost=…)` — a ceiling, not an estimate.
3. **Remember what just failed**, across jobs. A capacity memo shared by every submit means shards
   2..32 of a sweep do not each march into the zone that exhausted for shard 1.

Everything else — ordering, failover, the actual choice — stays SkyPilot's. With an empty memo and
no flags, the search space handed to SkyPilot is byte-for-byte what it is today.

---

## 3. Components

### 3.1 `lab/placement.py` — where, and at what rate

The only module that talks to `sky.catalog`. No credentials, no cloud calls, no I/O beyond the
memo file. That property is load-bearing: `lab register`'s worst-case number and `lab doctor`'s
price section must work on a machine with no GCP access.

| Symbol | Contract |
|---|---|
| `resolve_instance_type(res) -> str \| None` | accelerators → `get_instance_type_for_accelerator`; else `get_default_instance_type`. `None` when the catalog cannot resolve the spec. |
| `Candidate` | `region: str`, `zones: list[str]`, `hourly_usd: float` |
| `candidates(res, *, memo) -> list[Candidate]` | every region offering the resolved type at the requested spot-ness, minus zones the memo lists as exhausted, minus regions whose price exceeds `res.max_hourly_usd`. Sorted by price. |
| `price_band(res, *, memo) -> tuple[float, float] \| None` | `(min, max)` compute-only over the survivors. |
| `storage_hourly_usd(cloud, gb) -> float` | per-cloud $/GiB/hr constant × GB. GCP `0.000109589` (Hyperdisk Balanced). |
| `estimate(res, *, memo) -> Estimate` | the public entry point: `instance_type`, `low`, `high`, `storage`, `basis`. `high + storage` is what admission control checks. |
| `parse_exhausted_zones(text) -> list[str]` | pure; regex over a failed launch's log text. |
| `CapacityMemo` | see below. |

**Polarity.** `estimate().high` is the *worst admissible* rate, not the expected one, because its
consumers are guardrails and the repo's cost-safety philosophy is fail-toward-alarm. When a
`--price-cap` is set, `high` is `min(catalog max, cap)` — the cap makes the worst case provable
rather than estimated. When the catalog cannot price a spec, `estimate` returns `None` and every
guardrail degrades to exactly today's behaviour; a missing price must never block a launch.

**`CapacityMemo`.** A JSON file under the lab home mapping `(cloud, instance_type, zone)` to an
`exhausted_at` timestamp, TTL 30 minutes (`LAB_CAPACITY_MEMO_TTL_S`). Written by the supervisor on
a provision failure, read at `build_task` time. It is **advisory**: a missing, unreadable, corrupt,
or stale memo reads as empty and is never an error. Worst case it makes a launch slower; it can
never make one fail. Entries are keyed on instance type because exhaustion is per-machine-shape —
`n4-standard-4` being out in us-central1-a says nothing about `n1-highmem-4` there.

**How exclusion reaches SkyPilot.** `Resources` can pin a zone but cannot exclude one. So when —
and only when — the memo has live exclusions or a price cap narrows the field, `build_task` hands
`task.set_resources([...])` the surviving regions (capped at 10). SkyPilot still orders and
chooses; we have only narrowed its search space. With nothing to say, a single unpinned `Resources`
goes out exactly as today.

### 3.2 `lab/doctor.py` — will this launch work

A per-cloud check registry. `CheckResult(name, status: ok|warn|fail|skip, detail, fix)`;
`run_checks(cloud, resources, *, quick) -> list[CheckResult]`.

GCP checks, each mapping to a failure this project actually hit:

| Check | Method | Fails when |
|---|---|---|
| `adc` | `google.auth.default()` | no credentials resolve |
| `sky_daemon` | `sky check gcp` view vs this process's principal | the daemon disagrees with `.env` (GCP-CREDS-3, made visible) |
| `project` | ADC project / `GOOGLE_CLOUD_PROJECT` | unset |
| `billing` | `cloudbilling.projects.getBillingInfo` | `billingEnabled` false |
| `apis` | `serviceusage.services.list(filter=state:ENABLED)` | compute or cloudresourcemanager off |
| `iam` | `projects.testIamPermissions` | any required permission absent |
| `quota_cpu` | `compute.regions().get` → `CPUS` | requested vCPUs exceed the limit |
| `quota_gpu` | regional `NVIDIA_<F>_GPUS` **and** global `GPUS_ALL_REGIONS` | either is below the request |
| `quota_disk` | regional `DISKS_TOTAL_GB` | requested disk exceeds it |
| `catalog` | `placement.estimate` | the spec does not resolve |

`iam` tests **permissions, not roles**, so a custom role that grants the right permissions does not
read as broken. `quota_gpu` checks both levels — the live project passes regionally and fails
globally, which is precisely the case that burns a launch. `PREEMPTIBLE_CPUS = 0` is explicitly
*not* a failure, per GCP's documented fallback to standard `CPUS`.

Vast reuses `vast_balance`; DO checks for a doctl token; local has nothing to check. Exit 1 on any
`fail`, 0 otherwise; `--json` emits the structured list. Exposed on both the CLI and MCP.

### 3.3 Auto-preflight

The cheap subset runs before a remote launch, so a misconfiguration costs 2 seconds instead of a
provision. Two rules keep it from becoming a new failure mode:

- **Fail-open on error.** A check that cannot answer — API 5xx, timeout, missing library — is
  `skip`, and `skip` never blocks. The preflight refusing a launch because the preflight itself
  broke would be strictly worse than no preflight.
- **Fail-closed only on a definitive negative.** Only a check that positively answered "this cannot
  work" (quota is 0, the API is disabled, a required permission is absent) blocks, and
  `--no-preflight` overrides.

Results cache under the lab home per `(cloud, project)`: 24 h for IAM / APIs / billing, 1 h for
quota. So the common case adds no API calls at all.

### 3.4 Seam edits

No new logic in `cli.py` or `mcp_server.py` — both stay thin shells.

- `ResourceRequest` += `region`, `zone`, `max_hourly_usd`.
- `BackendInfo` += `zone`. `CostInfo` += `compute_hourly_usd`, `storage_hourly_usd`,
  `hourly_basis`; **`hourly_usd` becomes the total**, so `estimated_usd`, admission control, and
  the dashboard pick up storage with no further change. `hourly_basis` is a human-readable
  provenance string, e.g. `"gcp catalog n4-standard-4 spot europe-west1-b + 50GiB hyperdisk"`.
- `core.resolve_backend_profile` — the invariant: **no skypilot job on a storage-billing cloud may
  reach SkyPilot's 256 GB default.** cpu profile 50 GB (unchanged), GPU path 100 GB
  (`GPU_DEFAULT_DISK_GB`). The two profiles bill at *different rates*: SkyPilot's GCP adapter maps
  its default disk tier to `pd-balanced`, except n4/a3-ultragpu/a4 which support only
  `hyperdisk-balanced`. So the cpu profile's n4 is $0.000109589/GiB/hr and the GPU path's n1 is
  $0.000136986 — 100 GB on the GPU path is **$0.0137/hr**, against $0.0351/hr had the 256 GB
  default applied there.
- `backends/skypilot.build_task` — pin region/zone when given; narrow when the memo says so; pass
  `max_hourly_cost`.
- `backends/skypilot.catalog_hourly` — delegates to `placement.estimate`, worst-case basis,
  storage included.
- `sky_runner` — per-cloud provision timeouts (`vast` 8, `do` 12, **`gcp` 20**), records the zone,
  writes the memo on provision failure, prices with storage, and surfaces
  `use_spot and not launched_spot` in the end reason (GCP-PREEMPT-2).
- CLI: `--region`, `--zone`, `--price-cap` on `submit` / `sweep` / `register` / `register-sweep`;
  `lab doctor`. `--price-cap` is deliberately not `--max-hourly`, which already means the Vast
  price *trigger* on `register` — a wait-until condition, not a ceiling.

---

## 4. Data flow

```
submit --cloud gcp --spot [--region R] [--zone Z] [--price-cap C]
  │
  ├─ resolve_backend_profile ── disk_size always explicit (50 cpu / 100 gpu)
  │
  ├─ doctor.run_checks(quick=True) ── cached; fail-closed only on a definitive negative
  │
  ├─ placement.estimate(res, memo) ── instance_type, (low, high), storage, basis
  │        │                           └─ high+storage → CostInfo.estimated_usd, admission control
  │        └─ candidates(res, memo) ── regions minus exhausted zones minus over-cap
  │
  ├─ build_task ── pin | narrow | unpinned, + max_hourly_cost
  │
  ├─ sky.launch ── SkyPilot orders and fails over within that space
  │        │
  │        ├─ success → BackendInfo{machine_type, region, zone, launched_spot}
  │        │            CostInfo{hourly_usd = compute+storage, breakdown, basis}
  │        │
  │        └─ failure → parse_exhausted_zones(log) → memo.record()
  │                     provision_failure_reason(...) ── unchanged, already diagnoses
  │
  └─ next submit reads the memo and skips those zones
```

---

## 5. Error handling

The governing rule is that **nothing added here may create a new way for a launch to fail.** Every
new component degrades to current behaviour:

| Component | Degraded mode |
|---|---|
| catalog unavailable / spec unpriceable | `estimate` → `None`; guardrails behave exactly as today |
| memo missing / corrupt / unreadable | empty memo; full unpinned search space |
| memo would exclude every region | ignored entirely; full search space, logged |
| preflight check errors | `skip`; never blocks |
| `--region`/`--zone` invalid | rejected at parse time via `validate_region_zone` — the one hard failure, and it is the user's typo, caught before anything bills |
| storage price unknown for a cloud | storage term is 0 and `hourly_basis` says so, rather than guessing |

The single deliberate exception to fail-open is the preflight's definitive negatives — a quota of
0 or a disabled API, where launching is *known* to be wasted money and time.

## 6. Testing

**Unit (pure, no cloud).** `parse_exhausted_zones` against real SkyPilot log text; memo TTL,
corruption, and round-trip; `candidates` exclusion and cap filtering; `estimate` polarity — that
`high` is the max and not the min, the regression that motivated this; `storage_hourly_usd`
arithmetic; the disk invariant, that no gcp/do resolution leaves `disk_size` `None`; per-cloud
provision timeouts; `build_task` shape under pin / narrow / unpinned; every doctor check against
faked API payloads, including the two live findings — `GPUS_ALL_REGIONS = 0` must fail while
regional quota passes, and `PREEMPTIBLE_CPUS = 0` must not fail.

**Integration, double-locked** as `tests/test_gcp_backend_integration.py` already is
(`RUN_GCP_INTEGRATION=1` plus real credentials, so a plain `pytest` can never bill).

**Live verification** on the real project, which is the acceptance gate:

1. `lab doctor --cloud gcp` — expect `ok` throughout and `catalog` reporting the price band.
2. `lab doctor --cloud gcp --gpu T4:1` — expect **fail** on `quota_gpu` citing
   `GPUS_ALL_REGIONS = 0`, not regional quota.
3. A GPU submit — expect the preflight to refuse it, then `--no-preflight` to confirm the real
   launch fails the same way the preflight predicted. This validates the whole diagnosis chain and
   costs nothing, because no VM ever starts.
4. A real spot CPU job end to end: succeeded, `teardown_status == "succeeded"`, and a manifest
   carrying zone, spot-ness, and an `hourly_usd` that includes storage and matches the catalog for
   the region it actually landed in.
5. `lab reconcile` clean afterwards.


---

## 7. Outcome (2026-08-11)

Built, tested, and verified live against project `myproject-505213`. 106 new unit tests; the full
suite, `ruff`, and `mypy --strict` are clean.

**What the live runs proved.** A real spot CPU job ran end to end and landed in **europe-west1-b**
— the cheapest spot region — recording `compute $0.03401/hr` against a catalog price of $0.0340
for that exact region, plus `storage $0.00548/hr` (50 GiB hyperdisk, to the cent), for a total of
$0.0395/hr and an actual spend of $0.0014 over 131 seconds. Storage is 14% of that bill and was
invisible before. Teardown succeeded and `lab reconcile` was clean.

The GPU path was exercised as a *prediction test*: `lab doctor --cloud gcp --gpu T4:1` failed on
`GPUS_ALL_REGIONS = 0`, the submit was refused in seconds, and the same launch run with
`--no-preflight` failed after attempting to provision with exactly
`Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally`. The preflight is a true positive, not a
guess — and it cost nothing to establish, because no VM ever started.

**Four defects the live work found that reading the code did not.** All are fixed and tested:

1. **`GCP-PROV-6`** — the failure diagnosis was appended after SkyPilot's ~290-character message
   and truncated off the 300-character `end_reason`. GCP-PROV-3 was marked fixed and its output
   was being discarded. The diagnosis now leads.
2. **The doctor cache dropped context.** A cached `adc` result carried the verdict but not the
   credentials it publishes, so every later check reported "no GCP project selected" on the second
   invocation. Context-producing checks are never cached now.
3. **The doctor cache ignored shape.** A `--cpus 4` run's 50 GB disk verdict was served to a
   `--gpu T4:1` run that asks about 100 GB in a different region set. Quota keys carry a shape
   fingerprint now.
4. **Diagnostics polluted stdout.** Pricing began running on the `lab register` path, whose stdout
   is JSON that callers parse; `[lab] catalog price unavailable` corrupted the payload. All
   placement diagnostics go to stderr.

Two smaller ones came from self-review: the hyperdisk family check matched on a prefix, so
`n4a-*` would have been priced 20% low; and a memo entry that cost no region still collapsed the
search space from 41 regions to 10.

**Deliberately not done.** `--instance-type` (the catalog resolves it from `--cpus`/`--memory`, and
a third way to say the same thing invites conflicts); egress and sustained-use discounts in the
cost model (both need usage data the lab does not hold — the guide now states the exclusion rather
than the old claim that the estimate was "accurate"); and `GCP-LEAK-7`/`-8`/`-9`, `GCP-PREEMPT-1`,
`GCP-CREDS-2`…`-5`, which remain open in the gap schema.
