# GCP backend — capability & gap schema

**Status (2026-08-11, second pass).** **Fixed:** `GCP-LEAK-1`…`GCP-LEAK-6`, `VAST-LEAK-1`,
`GCP-COST-1`…`GCP-COST-6`, `GCP-PROV-1`, `GCP-PROV-2`, `GCP-PROV-3`, `GCP-PROV-5`, `GCP-PROV-6`,
`GCP-PREEMPT-2`, `GCP-PREEMPT-3`, `GCP-TEST-1`, `GCP-TEST-2`, `GCP-DOC-1`.
**Partly:** `GCP-PROV-4` (GPU path attempted live; blocked by a project quota, not a lab defect —
now predicted by `lab doctor` instead of discovered by a failed launch), `GCP-CREDS-1` (runbook
fixed, live host unverified). Remaining analysis: `GCP-LEAK-7`…`-9`, `GCP-PREEMPT-1`,
`GCP-CREDS-2`…`-5`.

The second pass added two records that did not exist when this document was written, both found
by instrumenting or running the thing rather than reading it: `GCP-COST-5` (the pre-launch
estimate was the cheapest region's price, so the guardrail checked the best case) and
`GCP-PROV-6` (the failure diagnosis was truncated off the manifest before anyone could read it).

Six further defects surfaced only once the code ran against the live project — two in the doctor's
cache, one in stdout hygiene, one where the disk invariant missed the *deferred* launch path
entirely, one where a working `--price-cap` was diagnosed as a credentials problem, plus the
truncation above. They are itemised in
`docs/superpowers/specs/2026-08-11-gcp-placement-pricing-doctor-design.md` §7. The pattern is
worth naming: **every one of them was invisible to code reading and obvious within a minute of
running the thing.**
**Scope:** everything `--cloud gcp` touches — provisioning, cost, teardown, leak detection,
preemption, the scheduler, credentials, and test coverage.
**Basis:** the code as of `96b02b3` + the first live GCP run (2026-08-11, job
`20260811-145235-d5a46c`) + the vendor precedents in `LAB-BUGS.md` and the Vast/DO test suites.

The GCP backend was built by mirroring the DO CPU backend, which was itself built by mirroring
Vast. Each mirroring step dropped something. This document names what.

---

## 1. Schema

Every gap below is a record with these fields.

| Field | Type | Meaning |
|---|---|---|
| `id` | `GCP-<AREA>-<n>` | stable handle |
| `area` | enum | `leak` \| `cost` \| `provision` \| `preempt` \| `creds` \| `surface` \| `test` |
| `severity` | enum | `critical` (money leaks or a guardrail silently no-ops) \| `high` \| `medium` \| `low` |
| `confidence` | enum | `confirmed` (read in the code) \| `observed` (hit live) \| `suspected` (needs a check) |
| `precedent` | ref | the Vast/DO bug or test this mirrors — every gap here has one |
| `failure_mode` | prose | what a user loses, concretely |
| `evidence` | `file:line` | |
| `test` | prose | the test that would close it |
| `fix` | prose | sketch only |

**Severity is money-weighted.** `critical` is reserved for two shapes: a resource that can bill
unnoticed, and a cost guardrail the user believes is on when it is off. Both have precedent in
this repo (`LAB-BUGS.md` §4 → ~$50 leaked; §6 → $146 overrun).

---

## 2. Vendor parity matrix

The fastest way to see the shape of the gaps. Each column is a capability Vast has; ✅ = present,
⚠️ = present but weaker, ❌ = absent.

| Capability | Vast | DO | GCP | Gap |
|---|:--:|:--:|:--:|---|
| Provider-direct instance listing | ✅ | ❌ | ✅ | — |
| Provider-direct destroy fallback in `robust_teardown` | ✅ | ❌ | ✅ | — |
| Fallback reports partial failure honestly | ✅ | n/a | ✅ | ~~GCP-LEAK-4~~, ~~VAST-LEAK-1~~ fixed |
| Destroy is confirmed, not fire-and-forget | ✅ | n/a | ✅ | ~~GCP-LEAK-6~~ fixed |
| Post-teardown "is it really gone" confirm | ✅ | ❌ | ✅ | ~~GCP-LEAK-5~~ fixed |
| Detached-storage leak pass | n/a | ✅ | ✅ | — |
| Storage pass survives an instance-API failure | n/a | ✅ | ✅ | ~~GCP-LEAK-3~~ fixed |
| Listing failure is loud, not silently "clean" | ✅ | ❌ | ✅ | ~~GCP-LEAK-2~~ fixed |
| Orphans wired into `reconcile`'s exit code | ✅ | ✅ | ✅ | ~~GCP-LEAK-1~~ fixed |
| Real booked price on the manifest | ✅ | ⚠️ | ✅ | ~~GCP-COST-1~~ fixed |
| Pre-launch estimate is a ceiling, not a floor | ⚠️ | ✅ | ✅ | ~~GCP-COST-5~~ fixed |
| Storage on the billed rate | n/a | ❌ | ✅ | ~~GCP-COST-2~~ fixed |
| Region / zone control | n/a | ❌ | ✅ | ~~GCP-PROV-1~~ fixed |
| Capacity memory across jobs | ❌ | ❌ | ✅ | ~~GCP-PROV-1~~ fixed |
| Pre-launch preflight (`lab doctor`) | ⚠️ | ⚠️ | ✅ | ~~GCP-PROV-5~~ fixed |
| Failure diagnosed from the real cause | ✅ | ❌ | ✅ | ~~GCP-PROV-3~~ fixed |
| Pre-launch budget estimate for the scheduler | ✅ | ✅ | ✅ | ~~GCP-COST-3~~ fixed |
| Liveness-unknown fails safe (assume alive) | ✅ | ✅ | ✅ | ~~GCP-PREEMPT-3~~ fixed |
| Live integration test | ⚠️ | ✅ | ✅ | ~~GCP-TEST-1~~ written |

Two things fell out when this was written. **Vast was the only cloud whose leak story was
complete**, because it was the only one that got a second, provider-direct opinion on every
question — GCP had all the machinery for that second opinion (`list_gcp_instances`) and used it in
exactly one place. And **the DO column's ❌s were inherited wholesale by GCP**: GCP-LEAK-1, -2 and
GCP-COST-3 were DO bugs that were never GCP bugs specifically; GCP just doubled the blast radius.

After the leak-honesty pass the leak half of the matrix is closed, and one cell inverted:

> **`VAST-LEAK-1` — `_vast_destroy_matching` reported success when every destroy failed**
> ✅ **FIXED 2026-08-11**
> `area: leak` · `severity: medium` · `confidence: confirmed` · `precedent: GCP-LEAK-4`
>
> The Vast fallback kept the shape GCP-LEAK-4 had just lost: per-rental `destroy_instance`
> failures were caught, printed, and dropped, and `robust_teardown` returned `succeeded`
> regardless. Vast is partly covered downstream by `confirm_no_rental`, so this was `medium`
> rather than `high` — but it left GCP strictly more honest than Vast, which is backwards for the
> cloud that bills the most per hour. Now returns `(destroyed, failures)` like its GCP twin. A
> matching rental carrying **no id** also alarms now: we can see it and cannot kill it.

---

## 3. Gaps

### Leak detection (FR-C2)

---
**`GCP-LEAK-1` — `lab reconcile` exits 0 on a GCP-only leak** ✅ **FIXED 2026-08-11**
`area: leak` · `severity: critical` · `confidence: confirmed`
· `precedent: LAB-BUGS §4; tests/test_gcp_backend.py:335`

The dry-run exit code is `if (report["orphans"] or report["sky_orphans"]) and not apply`. It reads
two of the six orphan lists. `gcp_orphans`, `gcp_disk_orphans` and `do_volume_orphans` never trip
exit 3 — they are printed in the JSON and ignored by the exit status.

**failure_mode:** the GCP compute pass exists precisely for the case where SkyPilot's registry
lost the cluster (LAB-BUGS §4, verbatim). In that case `sky_orphans` is empty by definition and
`gcp_orphans` is the only list with anything in it — so the one scenario the pass was written for
is the one where the alarm doesn't sound. A wrapper script or an agent checking `$?` sees "clean".
The skill and the guide both tell users exit 3 means action required.

**evidence:** `src/lab/cli.py:524`
**test:** a report carrying only `gcp_orphans` exits 3; same for `gcp_disk_orphans` and
`do_volume_orphans`. Assert on the *union* so a seventh pass can't be added and forgotten.
**fix:** exit 3 if any orphan list is non-empty. One line, and it retires the whole class.

---
**`GCP-LEAK-2` — any GCP API failure reads as "GCP not configured → clean"** ✅ **FIXED 2026-08-11**
`area: leak` · `severity: critical` · `confidence: confirmed` · `precedent: core.py:1011-1015`

`except Exception: gcp_instances = None`, commented "GCP not configured/unavailable: skip the
pass". That catch cannot tell *unconfigured* from *broken*: a revoked IAM role, an expired key, a
disabled API, a wrong project, an API quota — all produce a silent, empty, clean-looking pass.

The Vast pass is deliberately built the other way: `ImportError` → skip with a `vast_pass`
breadcrumb in the report; **anything else raises**, because (its own words) "when the SDK *is*
present there is no safe degraded mode for a leak-detection command." GCP got the degraded mode.

**failure_mode:** the user rotates the service-account key, `reconcile` reports clean forever
after, and the next leaked VM bills until someone opens the console. This is worse than no pass:
the report *claims* coverage.

**evidence:** `src/lab/core.py:1113-1115` vs `src/lab/core.py:1011-1015`
**test:** a `DefaultCredentialsError`-shaped failure skips with a `gcp_pass` breadcrumb; a 403 or
any other error raises `LabError`.
**fix:** add `gcp_pass` to the report mirroring `vast_pass`, and narrow the catch to the genuine
not-configured signals (`google.auth.exceptions.DefaultCredentialsError`, `ImportError`).

---
**`GCP-LEAK-3` — the disk pass is nested inside the instance pass** ✅ **FIXED 2026-08-11**
`area: leak` · `severity: high` · `confidence: confirmed` · `precedent: the DO volume pass, which is independent`

Disk listing runs only `if gcp_instances is not None`, and its own failure yields `gcp_disks = []`
silently. So one hiccup on `instances.aggregatedList` hides *both* passes, and a disk-API failure
is indistinguishable from "no leaked disks".

**failure_mode:** the unattached-disk leak is the slow, quiet one — a 50 GB balanced PD is ~$5/mo
forever, below anyone's noticing threshold, and it survives every instance-level cleanup. It is
also exactly what a preempted spot VM can leave behind.

**evidence:** `src/lab/core.py:1118-1130`
**test:** instance listing raises, disk listing succeeds → disk orphans still reported.
**fix:** hoist the disk pass to its own try, keyed off its own configured-ness.

---
**`GCP-LEAK-4` — `_gcp_destroy_matching` reports success when every delete failed** ✅ **FIXED 2026-08-11**
`area: leak` · `severity: high` · `confidence: confirmed`
· `precedent: _vast_destroy_matching shares the shape — see note`

Per-instance delete exceptions are caught, printed, and dropped; the function returns only the
names it *did* destroy. `robust_teardown` then returns `status: "succeeded"` unconditionally,
commented "destroyed-or-none-found are both safe outcomes". True — but the third outcome,
*found-and-failed-to-destroy*, takes the same branch and is not safe.

**note on the precedent:** Vast has the identical shape, but Vast is covered downstream by
`confirm_no_rental`, which independently re-lists and returns False under any uncertainty. GCP's
equivalent is stubbed out (GCP-LEAK-5), so on GCP this is unbacked.

**failure_mode:** `sky.down` exhausted its retries, the gcp-direct fallback also failed on every
instance, and the manifest records `teardown_status: "succeeded"`. `lab wait` exits 0. This is the
false-clean that FR-C2 exists to prevent.

**evidence:** `src/lab/backends/skypilot.py:357-371`, `:510-523`
**test:** all deletes raise → `status: "failed"` with the errors in `error`.
**fix:** count failures; any failure → `failed`.

---
**`GCP-LEAK-5` — `preempted_teardown_confirmed` never checks GCP, though it can** ✅ **FIXED 2026-08-11**
`area: leak` · `severity: high` · `confidence: confirmed`
· `precedent: tests/test_teardown_confirm.py (the Vast trio)`

```python
if cloud != "vast":
    return True
```
with the rationale "other clouds have none [no provider-direct listing]". That was true when the
function was written for DO. It is **false for GCP** — `list_gcp_instances` is that listing, in
the same module, already used by `reconcile` and by the teardown fallback.

**failure_mode:** unmanaged spot preemption is the likeliest way a GCP box outlives its job, and
it is the exact path that skips confirmation. `test_gcp_backend.py:513` currently *asserts* the
unconditional `True`, so the gap is pinned in place by a passing test.

**evidence:** `src/lab/backends/skypilot.py:385-391`
**test:** cloud=gcp with a matching live instance → False; with none → True; with a listing error
→ False (never claim "gone" under uncertainty, matching `confirm_no_rental`).
**fix:** a `confirm_no_instance(cluster)` alongside `confirm_no_rental`, same fail-toward-alarm
contract. Rewrite the pinning test.

---
**`GCP-LEAK-6` — deletes are fire-and-forget; the Operation is never polled** ✅ **FIXED 2026-08-11**
`area: leak` · `severity: high` · `confidence: confirmed` · `precedent: none — GCP-specific`

`compute.instances().delete(...).execute()` returns a GCE **Operation**, not a completed delete.
The lab treats a 200 on the *request* as a destroyed VM. GCE deletes routinely take 30-60s and can
fail after acceptance (`RESOURCE_IN_USE_BY_ANOTHER_RESOURCE`, a stuck zonal operation, a quota
error on the disk detach).

**failure_mode:** `gcp_destroyed: ["lab-x-head"]` in the report and `teardown_status: "succeeded"`
on the manifest, while the VM is still RUNNING. Every downstream signal — `lab wait`'s exit code,
the dashboard's `teardown` column, the scheduler's leak sweep — inherits the lie. Vast's
`destroy_instance` is comparatively synchronous, so this failure mode has no vendor precedent and
no existing test to copy.

**evidence:** `src/lab/backends/skypilot.py:339-354`
**test:** a fake compute whose returned operation carries `error` → the name is not counted as
destroyed and the teardown reports failed.
**fix:** poll `zoneOperations.wait` with a short bound; on timeout report unconfirmed rather than
destroyed. Cheap, and it also makes `reconcile --apply` honest.

---
**`GCP-LEAK-7` — `lab-` prefix matching is both too broad and unanchored to a project**
`area: leak` · `severity: medium` · `confidence: confirmed` · `precedent: do_volume_orphans, same shape`

Two halves, both from `_get_gcp_compute()` resolving the project ambiently via
`google.auth.default()`:

- **Too broad.** `reconcile --apply` deletes any GCE instance or unattached disk in the project
  whose name starts with `lab-` and doesn't substring-match a running cluster. In a shared
  project, someone's `lab-notebook` VM matches. This is a destructive false positive, and
  `--apply` does not ask.
- **Unanchored.** SkyPilot can be pinned to a different project via `~/.sky/config.yaml`. If it
  is, reconcile sweeps a project the lab never launches into and reports clean. The report never
  says which project it swept.

**evidence:** `src/lab/backends/skypilot.py:241-251`, `:304-336`
**test:** a `lab-notebook`-style name is not an orphan; the report records the swept project.
**fix:** match the real cluster shape (`lab-<job_id>` + SkyPilot's suffix), not a bare prefix; put
`project` in the report.

---
**`GCP-LEAK-8` — uncovered billable GCP resources**
`area: leak` · `severity: medium` · `confidence: suspected` · `precedent: the DO volume pass`

Covered: instances, unattached disks. Not covered, and each bills:

- **Reserved-but-unattached static external IPs** — GCP charges *more* for an idle reserved IP
  than an attached one, by design.
- **SkyPilot's staging bucket** — `roles/storage.admin` is in the required-roles table precisely
  because SkyPilot creates one. Nothing ever reaps it.
- Snapshots and custom images, if a future path creates them.

**fix:** an addresses pass and a bucket pass, both cheap `aggregatedList` calls. Confirm first that
SkyPilot's GCP provisioner actually reserves static IPs on our path — it may use ephemeral ones,
which would drop the first bullet.

---
**`GCP-LEAK-9` — the `poweroff` backstop does not stop GCP billing the way it stops Vast billing**
`area: leak` · `severity: medium` · `confidence: confirmed` · `precedent: skypilot.py:128-130`

The on-box watchdog runs `sudo poweroff -f` at `wall + 600s`, documented as "a hard backstop". Its
effect is per-cloud and the code says so nowhere:

| Cloud | `poweroff` does | Still billing after |
|---|---|---|
| Vast | ends the rental | nothing |
| GCP | TERMINATEs the VM; compute billing stops | **the persistent disk, indefinitely** |
| DO | powers the droplet off | **the whole droplet, at full price** |

**failure_mode:** on GCP the backstop converts a compute leak into a storage leak — a real
improvement, but not the "hard backstop" the docstring promises, and the residue is invisible
until someone runs `reconcile` (which, per GCP-LEAK-1, exits 0 about it).

**fix:** documentation, primarily. The real teardown path is `down=True` + autostop; state that
the poweroff backstop is compute-only on GCP and does not release storage.

---

### Cost (FR-I2)

---
**`GCP-COST-1` — no booked-price verification; the catalog is trusted** ✅ **FIXED 2026-08-11**
`area: cost` · `severity: medium` · `confidence: confirmed` · `precedent: LAB-BUGS §5`

`_resolve_hourly` queries the real rate for Vast only: "for every other cloud (DO/GCP) SkyPilot's
catalog estimate is accurate, so use it." LAB-BUGS §5 is the story of trusting exactly that
catalog and being wrong by 20×.

GCP's catalog is far more trustworthy than Vast's — prices are published, not auctioned — so this
is `medium`, not `critical`. But the estimate still omits: sustained-use discounts (over-charges),
**spot price by zone** (our own live run: $0.034/hr in europe-west1-b against the $0.18 the guide
quotes for us-central1 — a 5× spread the manifest cannot express), the disk line item entirely,
and egress.

**fix:** price the *launched* resource rather than the requested one — `machine_type` and `region`
are already captured on `BackendInfo`, and `_hourly_cost(handle)` already reads the launched
resources. Then state the exclusions (disk, egress, SUD) in the guide instead of calling it
"accurate".

---
**`GCP-COST-2` — the GPU path inherits SkyPilot's 256 GB default boot disk** ✅ **FIXED 2026-08-11**
> **Wider than recorded.** The gap named the GPU path, but the hole was the *deferred* path too:
> `resolve_backend_profile` is only on the CLI/MCP submit path, and the scheduler launches
> registrations straight through `Lab.submit`, so any registered GCP job — cpu profile included —
> inherited the 256 GB default. Found by noticing `lab register --cloud gcp` quoted a worst case
> whose storage term was exactly $0. The rule now lives in `placement.effective_disk_gb` and is
> applied in `build_task`, which every launch goes through.
`area: cost` · `severity: medium` · `confidence: confirmed` · `precedent: the DO 422 (memory: DO tier limits)`

The `cpu` profile pins `disk_size=50`. `--backend skypilot --cloud gcp --accelerators T4:1` leaves
`disk_size=None`, so SkyPilot's 256 GB default applies. On DO that same default **hard-failed with
a 422**, which is why the cpu profile pins it — the failure was loud and got fixed. On GCP it
succeeds quietly and bills ~$25/mo per volume while it lives, and consumes fresh-project SSD
quota.

**failure_mode:** a leaked GPU-job disk (GCP-LEAK-3's blind spot) is 5× more expensive than a
leaked cpu-profile disk, for no reason anyone chose.

**test:** a gcp spec with accelerators and no explicit `disk_size` resolves to an explicit default.
**fix:** default `disk_size` for the gcp GPU path the way the cpu profile does.

---
**`GCP-COST-4` — `lab register` reports `worst_case_cost_usd: null` for non-Vast clouds** ✅ **FIXED 2026-08-11**
`area: cost` · `severity: medium` · `confidence: observed` · `precedent: GCP-COST-3, its sibling`

Found while registering a live GCP probe (2026-08-11): the registration returned
`"worst_case_cost_usd": null`. `worst_case_cost()` returns None unless `triggers.max_hourly_usd`
is set — and price triggers are Vast-only, so it is None for every GCP/DO registration.

GCP-COST-3 fixed the *scheduler's* launch-time admission control. This is the **registration-time
authorization** number: the figure the user is shown when they commit to a deferred job, and (for
sweeps) the `per_point_cap` fallback when `max_cost_usd` isn't given — so a GCP sweep registered
without an explicit cap admits against `per_point_cap=None`.

**failure_mode:** "run this overnight, worst case null dollars." The one number whose entire job
is to make the user's exposure legible before they close the laptop is blank on two of three
clouds — and blank reads as "free", not as "unknown".

**fix:** same source as GCP-COST-3 — fall back to `catalog_hourly(resources)` when there is no
offer price. The helper already exists; this is one call site.

---
**`GCP-COST-3` — the scheduler's daily budget and per-job cost cap silently no-op on GCP** ✅ **FIXED 2026-08-11**
`area: cost` · `severity: high` · `confidence: confirmed`
· `precedent: memory "cost-safety design philosophy" — admission control is the guardrail`

`_estimate_cost` returns `None` unless `_best_hourly_seen[reg_id]` was populated, and that cache is
populated **only by a Vast price trigger**. Price triggers are rejected for GCP by design
(Vast-only offer feed). Therefore `est is None` for every GCP registration, and both admission
checks are guarded on `est is not None`:

```python
if est is not None and cap is not None and est > cap:            # max_cost_usd  — skipped
if budget is not None and est is not None and committed + est > budget:  # daily budget — skipped
```

**failure_mode:** a user sets `--max-cost 3` on a GCP registration and `budget_usd_per_day` in
control.json, and neither ever fires. The docstring calls this out for "CPU/local jobs" — but it
now silently covers an entire *cloud*, including its GPUs. The guardrail is advertised in the CLI
help and the skill; it does not hold. `lab register --cloud gcp --gpu T4:1` is unbounded.

**evidence:** `src/lab/scheduler/tick.py:179-189`, `:437-447`
**test:** a gcp registration with `max_cost_usd` below its worst case is skipped, not launched.
**fix:** for non-Vast clouds, derive the hourly from the SkyPilot catalog pre-launch —
`sky.Resources(...).get_cost(3600)` is a pure catalog lookup, no provisioning, and it is the same
call `_hourly_cost` already makes post-launch.

---
**`GCP-COST-5` — the pre-launch estimate was the cheapest region's price** ✅ **FIXED 2026-08-11**
`area: cost` · `severity: high` · `confidence: confirmed` · `precedent: GCP-COST-3, which introduced it`

Found while wiring the region system, not by reading the gap list. `catalog_hourly` built a
`sky.Resources` with no region and called `get_cost()`. SkyPilot's source is explicit about what
that returns: the **minimum** across every region that offers the shape, chosen deliberately so a
multi-component price cannot mix regions.

So the number GCP-COST-3 had just wired into the scheduler's admission control — the fix for a
guardrail that silently did nothing — was the globally cheapest price. On GCP spot that is
**$0.0340 against a real worst case of $0.1226**, a 3.6× under-estimate; with `spot_fallback` live
the true ceiling is the on-demand $0.2902, 8.5×.

**failure_mode:** an admission check that under-estimates admits jobs it should refuse. `--max-cost`
and the daily budget both compare against this number, so both were systematically too permissive
— the same shape of failure as GCP-COST-3 itself, one layer down and harder to see, because the
guardrail now *did* return a number and the number looked plausible.

**fix:** price a *band*. `placement.estimate()` returns `(low, high)` over the candidate regions
and every guardrail checks `high`; the ceiling is priced on-demand whenever `spot_fallback` could
land there. A `--price-cap` maps to `sky.Resources(max_hourly_cost=)`, which makes the ceiling
enforced rather than predicted.

---
**`GCP-PROV-6` — the failure diagnosis was truncated off the manifest** ✅ **FIXED 2026-08-11**
`area: provision` · `severity: medium` · `confidence: observed` · `precedent: GCP-PROV-3, which it silently undid`

Observed on a live GPU launch. `provision_failure_reason` appended its diagnosis *after* SkyPilot's
generic message, and `end_reason` is capped at 300 characters on the manifest. SkyPilot's generic
message alone is ~290:

```
Failed to provision all possible launchable resources. Relax the task's resource requirements: …
To keep retrying until the cluster is up, use the `--retry-until-up` flag.
Reasons for provision failures (for details, please check the log above):
```

The real cause — `Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally`, sitting in the log — never
reached the manifest at all. GCP-PROV-3 was marked fixed and its output was being thrown away.

**failure_mode:** the user reads `lab status`, sees a generic "failed to provision", and goes
looking in the wrong place. This is the exact experience GCP-PROV-3 was written to end.

**test:** the real 290-character SkyPilot error plus a quota line still yields a diagnosis inside
the first 300 characters.
**fix:** lead with the diagnosis. Also added a `GPUS_ALL_REGIONS`-specific hint ahead of the generic
quota one, because the generic hint names the *regional* metric and would send the user to the
wrong console page.

---

### Provisioning

---
**`GCP-PROV-1` — no region / zone / instance-type control** ✅ **FIXED 2026-08-11**
`area: provision` · `severity: high` · `confidence: observed` · `precedent: none — the live blocker`

`ResourceRequest` exposes `cpus`, `gpus`, `memory`, `disk_size`, `accelerators`, `cloud`,
timeouts, spot flags. No `region`, `zone`, or `instance_type`. Every cpu-profile shape resolves to
the **n4** family on GCP.

**observed 2026-08-11:** `ZONE_RESOURCE_POOL_EXHAUSTED` for `n4-standard-4` across all four
us-central1 zones *and* us-east1-b. There is no flag that steers around it. The only lever that
worked was `use_spot=true`, which re-prices the optimizer's search and happened to reach
europe-west1-b — a workaround that changes the *billing model* to route around a *capacity*
problem, and which the guide now recommends.

**fix:** pass `region`/`zone` through to `sky.Resources` (it already accepts them); consider
`--instance-type`. Both are additive.

---
**`GCP-PROV-2` — the provision watchdog is calibrated to Vast** ✅ **FIXED 2026-08-11**
`area: provision` · `severity: medium` · `confidence: suspected`
· `precedent: skypilot.py:42-44, written for Vast`

`DEFAULT_PROVISION_TIMEOUT_MIN = 8`, justified as "a healthy Vast host reaches UP in ~2-4 min". A
GCP launch that fails over across five zones spends its time in the optimizer, not in a stuck
host. If that search exceeds 8 minutes the watchdog kills a launch that was about to succeed —
**and a launch killed mid-provision is the original LAB-BUGS §4 leak scenario**, where autostop
isn't set yet.

**fix:** per-cloud defaults.

---
**`GCP-PROV-3` — `provision_failure_reason` for GCP is a static leaflet, not a diagnosis** ✅ **FIXED 2026-08-11**
`area: provision` · `severity: medium-high` · `confidence: observed` · `precedent: LAB-BUGS §8`

Vast got a *dynamic* diagnosis out of §8: on failure, query the balance and say
`"Vast account balance is $X — top up to provision"`. GCP got a fixed string listing three
possible causes, returned regardless of what actually happened.

The real causes are all unambiguously identifiable from the error text we already hold:

| Marker in the launch error | Real cause | Actionable message |
|---|---|---|
| `ZONE_RESOURCE_POOL_EXHAUSTED` | capacity | try another region, or `--spot` |
| `QUOTA_EXCEEDED` / `Quota .* exceeded` | quota | request an increase for this family/region |
| `SERVICE_DISABLED` / `has not been used in project` | API off | enable compute/cloudresourcemanager |
| `billing` / `BILLING_DISABLED` | billing | attach a billing account |

**observed:** capacity exhaustion surfaced as `Failed to set up SkyPilot runtime on cluster` /
`Could not find any head instance` — a *downstream* symptom that reads like a lab bug. It cost
real time in the live session and is currently mitigated only by a "grep the log for
ZONE_RESOURCE_POOL_EXHAUSTED" note in the guide's gotchas.

**fix:** pattern-match the error text, the same shape as `is_transient_launch_error`. Pure
function, trivially testable.

---
**`GCP-PROV-4` — the GPU path has never run** ⚠️ **ATTEMPTED LIVE 2026-08-11 — blocked by project quota, not by the lab**
`area: provision` · `severity: high` · `confidence: observed` · `precedent: none`

> **Status.** A real `--accelerators T4:1` launch was attempted on 2026-08-11. It failed with
> `Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally` — a project-level authorisation the lab
> cannot grant itself, not a defect in the GPU path. Everything upstream of provisioning is now
> exercised: the catalog resolves `T4:1` to `n1-highmem-4` across 23 regions and prices it
> ($0.60-$0.87/hr on demand incl. disk), the GPU disk default (100 GB) reaches `sky.Resources`,
> and `lab doctor --cloud gcp --gpu T4:1` **predicts the failure in seconds** rather than
> discovering it by burning a provision. What remains untested is strictly what happens *after* a
> GPU VM boots. Reopen and finish this once `GPUS_ALL_REGIONS` is raised (request it in
> IAM & Admin > Quotas; first-time GPU approvals take up to 48h).

Zero GPU jobs have run on GCP. A fresh project has **0 GPU quota** and nothing pre-checks it.
`--accelerators T4:1` additionally constrains the machine family (T4 needs n1) and the zone set,
neither of which the lab reasons about. This is *unknown*, not *broken* — but "GPU is P1" and the
guide advertises T4/L4/A100 with a price table.

**fix:** run one. Then a preflight (below).

---
**`GCP-PROV-5` — no preflight; every misconfiguration is discovered by burning a launch** ✅ **FIXED 2026-08-11**
`area: provision` · `severity: medium` · `confidence: confirmed` · `precedent: LAB-BUGS §8 asked for it`

§8 closed with "a `lab status`/`lab doctor` balance readout would also pre-empt it." It was never
built. The live GCP bring-up rediscovered the cost of that: APIs disabled, then six missing IAM
roles, then the ADC daemon gotcha — each found by a failed command, in series.

**fix:** `lab doctor --cloud gcp` — ADC resolves? project set? both APIs enabled? required roles
present? quota for the requested accelerator? Each is one API call, and each maps to a failure we
actually hit.

---

### Preemption

---
**`GCP-PREEMPT-1` — preemption is inferred, though GCE reports it**
`area: preempt` · `severity: medium` · `confidence: confirmed` · `precedent: preemption.py:29-31`

`classify_terminal` infers preemption from "spot + cluster vanished + no authoritative terminal".
GCE states it outright: the instance goes `TERMINATED` with `scheduling.preemptible=true`, and the
guest gets a 30-second ACPI warning via the metadata server. We hold the compute API that can read
this.

**failure_mode:** a genuinely *failed* spot job whose box happened to disappear classifies as
`preempted` → the scheduler auto-resubmits → the same failure is paid for twice. The classifier
correctly refuses to override an authoritative terminal; the point is that on GCP an authoritative
answer exists and isn't fetched.

**fix:** an optional cloud probe feeding `sky_state`, leaving the pure classifier untouched.

---
**`GCP-PREEMPT-2` — `spot_fallback` can land on-demand at ~5× with no warning** ✅ **FIXED 2026-08-11**
`area: preempt` · `severity: low-medium` · `confidence: confirmed`

`spot_fallback` defaults True, and the guide now recommends `--spot` as the n4 capacity
workaround (GCP-PROV-1) — so users will increasingly submit spot expecting spot pricing and
silently land on-demand. `launched_spot` records what happened; nothing surfaces it.

**fix:** surface the divergence in `lab status` when `use_spot and launched_spot is False`.

---
**`GCP-PREEMPT-3` — unknown liveness reads as "gone" on GCP, and as "alive" on Vast** ✅ **FIXED 2026-08-11**
`area: preempt` · `severity: medium` · `confidence: confirmed` · `precedent: tick.py:67-87, explicitly`

The scheduler watchdog's Vast branch documents its fail-safe: a listing error "must read as
'unknown, assume alive'". The non-Vast branch is `except Exception: return False` — unknown reads
as **gone**.

**failure_mode:** a transient `sky.status` error on a healthy GCP job marks it `failed` with
`end_reason="supervisor died; instance gone"`, losing the run. Mitigated — the gone branch calls
`_teardown` first, so it doesn't leak money, just work — but it is the opposite polarity from the
documented contract two branches up.

**evidence:** `src/lab/scheduler/tick.py:79-87`
**fix:** distinguish "sky says no such cluster" from "sky failed to answer"; the latter → alive.

---

### Credentials (FR-J1)

---
**`GCP-CREDS-1` — the scheduler droplet has no GCP credential path** ⚠️ **RUNBOOK FIXED 2026-08-11; live host still unverified**
`area: creds` · `severity: high` · `confidence: suspected` · `precedent: deploy/scheduler/, predates GCP`

The guide states "`lab register --cloud gcp` works like any registration — the cloud rides in the
registered spec and the scheduler launches on GCP." The always-on droplet was provisioned before
GCP existed in the lab. Nothing has put a service-account key or ADC on it.

**failure_mode:** deferred GCP jobs queue successfully, pass their triggers, and fail at launch on
the droplet — at 3am, unattended, which is the entire point of the feature.

**fix:** verify on the live droplet; if absent, add the ADC symlink to `deploy/scheduler/` and its
runbook. This is the one gap that is a *deployment* action, not a code change.

---
**`GCP-CREDS-2` — `.env` discovery ignores `LAB_REPO_DIR`**
`area: creds` · `severity: medium` · `confidence: confirmed` · `precedent: cli.py:528-532`

The Typer callback loads from `repo_root()` — cwd-derived. Every other repo-rooted path in the CLI
goes through `_repo()`, which honours `LAB_REPO_DIR`. The scheduler host is the documented user of
that override *and* the host that most needs a service-account key.

**failure_mode:** a systemd unit whose WorkingDirectory isn't the repo silently loads no `.env`,
and the failure appears one layer down as an opaque auth error.

**test:** `LAB_REPO_DIR` set → `.env` is read from there.
**fix:** use the same resolution as `_repo()`.

---
**`GCP-CREDS-3` — `.env` is not actually the source of truth it reads as** ✅ **FIXED 2026-08-11**
`area: creds` · `severity: low` · `confidence: observed` · `precedent: documented in the guide's gotcha`

`load_lab_env` mutates the lab process only. SkyPilot's long-lived API-server daemon does not
inherit it, so whichever process started the daemon determines the credentials until
`sky api stop`. Documented, with the well-known-ADC symlink as the robust workaround — but the
mental model `.env` invites ("this file configures the lab") is not quite true.

**fix:** none needed beyond the existing docs; consider having `lab doctor` (GCP-PROV-5) report
the *daemon's* view of credentials rather than the process's.

**done 2026-08-11:** that is exactly what `lab doctor`'s `sky_daemon` check now does — it shells
out to `sky check gcp` and reports a **fail** when the daemon says GCP is disabled while ADC
resolves fine here, naming `sky api stop` and the well-known-ADC symlink as the fixes. The
divergence is no longer something you have to know to look for.

---
**`GCP-CREDS-4` — `.env` exclusion from the workdir sync is believed, not asserted**
`area: creds` · `severity: low-medium` · `confidence: suspected` · `precedent: LAB-BUGS §7`

`build_task(..., workdir=Path.cwd())` rsyncs the repo root to the remote. `.env` is git-ignored and
SkyPilot honours `.gitignore`/`.skyignore`, so it is *believed* excluded — never asserted.

Today `.env` holds only paths, so the blast radius is a disclosed filesystem layout rather than a
key. But it is precisely the file a user will paste an R2 secret into, and §7 is this repo's
history of a secret reaching a persisted artifact.

**test:** assert `.env` is not in the synced file set.
**fix:** an explicit `.skyignore` entry — belt and braces, one line.

---
**`GCP-CREDS-5` — redaction covers GCP tokens but not signed URLs**
`area: creds` · `severity: low` · `confidence: suspected` · `precedent: LAB-BUGS §7`

`redact.py` already handles `access_token` / `refresh_token` / `private_key` / bare `ya29.` — added
for GCP, and good. Not covered: GCS signed-URL credentials (`X-Goog-Signature=`,
`X-Goog-Credential=`), which SkyPilot's bucket staging can emit.

**fix:** two more patterns and a test using real-shaped GCP log lines.

---

### Test coverage & hygiene

---
**`GCP-TEST-1` — no live integration test** ✅ **FIXED 2026-08-11 (written; not yet run)**
`area: test` · `severity: medium` · `confidence: confirmed` · `precedent: tests/test_cpu_backend_integration.py`

DO has one, with a double lock: `RUN_DO_INTEGRATION=1` **and** real creds present, so it can never
bill in CI or a plain `pytest` run. GCP has none, even though the live run is now reproducible and
costs $0.0013.

**fix:** `tests/test_gcp_backend_integration.py`, same two locks, asserting: job succeeded,
`teardown_status == "succeeded"`, remote interpreter matches `.python-version`, and a follow-up
`reconcile` is clean.

---
**`GCP-TEST-2` — `.python-version` is untracked** ✅ **FIXED 2026-08-11**
`area: test` · `severity: high` · `confidence: observed` · `precedent: FR-B2`

The pin reached the remote only via the dirty-tree diff bundle. On a clean tree it does not exist,
and the remote resolves the newest interpreter satisfying `requires-python = ">=3.12"` — which
live meant **Python 3.14.7**, no cp314 wheels for `numpy 1.26.4`, no C compiler on the image,
`FAILED_SETUP`.

This is not a GCP gap. `uv.lock` pins packages; nothing pinned the interpreter that resolves them,
on **any** remote backend. It is latent for Vast and DO too — they just haven't drawn a fresh
enough image yet.

**fix:** `git add .python-version`. Then a test that the pin is present in the synced workdir, so
it can't be dropped again.

---
**`GCP-TEST-3` — two tests pin current behaviour that the fixes must change** ✅ **RESOLVED 2026-08-11**

- `test_gcp_backend.py:227` `test_reconcile_gcp_pass_skips_when_unconfigured` — asserts the
  over-broad swallow of GCP-LEAK-2.
- `test_gcp_backend.py:513` `test_preempted_teardown_confirmed_gcp_never_touches_vast` — asserts
  the unconditional `True` of GCP-LEAK-5. The *intent* (never touch Vast for a GCP job) is right
  and should survive; the assertion should become "consults the GCP listing, never the Vast one".

Neither is a bug. Both are load-bearing for the current design and will fail loudly when it
changes — which is what they're for.

**Resolved in the leak-honesty pass.** They did exactly their job: both failed, and both were
rewritten rather than deleted. `test_reconcile_gcp_pass_skips_when_unconfigured` now asserts the
narrow behaviour (skip **only** on `GcpNotConfigured`, and say so in the report), with a sibling
asserting that any other API failure raises. The unconditional-`True` test became the trio
`test_preempted_teardown_confirmed_gcp_{true_when_instance_is_gone,false_when_instance_survives,
false_when_listing_fails}` — keeping the original intent (never consult Vast for a GCP job) while
asserting the GCP listing is consulted and that uncertainty reads as "still there".

---
**`GCP-DOC-1` — the guide's price table is stale** ✅ **FIXED 2026-08-11**
`area: surface` · `severity: low` · `confidence: observed`

The table quotes `e2-standard-4` at ~$0.13 as "cpu profile pick". The catalog now resolves the
profile to `n4-standard-4` (~$0.18 on-demand; $0.034 spot in europe-west1-b, per the live run).
Users budget from this number.

---

## 4. Reading of the whole

**The critical two were both one-line fixes** — now made. GCP-LEAK-1 (exit code read two of six
orphan lists) and GCP-LEAK-2 (every API error read as "not configured") together meant the FR-C2
money alarm did not fire for the cloud we had just shipped. Neither was deep; both were omissions
from wiring, and they mirrored DO omissions that were never noticed because DO has no
provider-direct pass to omit. Fixing LEAK-1 closed the DO volume hole for free.

**The most expensive one was invisible** — now fixed. GCP-COST-3, a `--max-cost` and a daily
budget that silently didn't apply to an entire cloud, GPUs included, is the failure shape that
produced this repo's two worst incidents (§4, §6). It read as working: nothing in `lab status`,
the report, or the CLI help said the guardrail had been skipped. The estimate now falls back to
SkyPilot's catalog when there is no Vast offer feed — a local lookup, no provisioning — and
returns None only for specs with nothing to price (a plain local registration), preserving the
old behaviour exactly where it was already correct.

**The pattern across all of them:** Vast got a *second, provider-direct opinion* on every
money-critical question — is it really gone, what is it really costing, why did it really fail.
DO could not have one. GCP can, and has the client for it, and uses it in exactly one place
(`robust_teardown`'s fallback). Most of the leak and cost gaps close by extending that one
existing capability to the three other places that already have a Vast-shaped hole:
`preempted_teardown_confirmed`, `_resolve_hourly`, and `provision_failure_reason`.

**Sequencing.** ~~GCP-TEST-2 (`git add`) and GCP-LEAK-1 first — minutes, and they stop active
bleeding of trust in the exit code. Then GCP-LEAK-2/-3/-4/-6 as one leak-honesty pass.~~ **Done
2026-08-11** (GCP-LEAK-5 came along with it — the confirm function was the natural home for the
listing the other fixes already needed). Next: **GCP-COST-3**, the budget that silently doesn't
apply. GCP-PROV-1 and GCP-PROV-4 are feature work and belong in their own cycle; GCP-CREDS-1 is a
deployment errand that should be checked before anyone relies on deferred GCP jobs.

**What the leak-honesty pass changed**, for a reader diffing against the records above:

| Change | Gap |
|---|---|
| `reconcile`'s exit 3 reads the union of all five orphan lists (`_ORPHAN_FIELDS`) | LEAK-1 |
| `GcpNotConfigured` separates "GCP isn't set up here" from "the API failed"; the latter raises `LabError` | LEAK-2 |
| `gcp_pass` / `gcp_disk_pass` breadcrumbs in the report, mirroring `vast_pass` | LEAK-2 |
| instance and disk passes are independent, each with its own configured-ness | LEAK-3 |
| `_gcp_destroy_matching` returns `(destroyed, failures)`; any failure → `teardown_status="failed"` | LEAK-4 |
| `_await_zone_operation` waits for deletes to actually complete and raises on operation errors | LEAK-6 |
| `confirm_no_instance` gives GCP the same post-teardown second opinion Vast has | LEAK-5 |

19 new tests; 515 pass, ruff and mypy clean. Two tests that pinned the old behaviour were
rewritten (GCP-TEST-3), and one pre-existing test was made hermetic — it had begun reaching the
real GCP project once `preempted_teardown_confirmed` started consulting the compute API.
