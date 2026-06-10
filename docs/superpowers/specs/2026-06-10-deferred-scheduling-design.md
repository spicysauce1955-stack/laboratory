# Deferred Experiment Scheduling — Design

**Date:** 2026-06-10
**Status:** Approved (brainstorming session)
**Spec refs:** FR-H2 (queueing), FR-H3 (partial — no priorities), FR-I3 (price-aware launching),
FR-C2/FR-I2 (cost safety), FR-J1 (secrets)

## 1. Problem & goals

`lab submit` is synchronous: it launches immediately from the caller's machine. There is no way to
say "run this tonight", "run this when a 4090 drops below $0.25/hr", or "run B after A succeeds"
without keeping a terminal open and babysitting. This feature adds **registrations**: deferred job
submissions evaluated and launched by a small always-on cloud scheduler.

Goals (v1):
- Triggers: **time window** (nightly, tz-aware), **price threshold** (Vast.ai offers), and
  **dependency chaining** (after another registration succeeds). AND semantics when combined.
- Multiple concurrent launches (each job on its own machine, as today).
- Guardrails: per-job cost cap, daily budget cap, required expiry deadline, global pause +
  per-entry hold.
- Management: list/show/cancel/hold registrations from laptop CLI and MCP.

Non-goals (v1): recurring/cron registrations (each entry runs at most once), priorities,
cross-machine dependencies on laptop-launched job_ids (dependencies reference registrations in the
same queue only), multi-user quotas.

## 2. Architecture

```
laptop                          R2 (bus + source of truth)      Droplet (playground-provisioned)
──────                          ──────────────────────────      ────────────────────────────────
lab register ──writes──▶  queue/entries/<reg_id>.json      systemd timer (60s)
             ──uploads─▶  queue/bundles/<reg_id>.tar.gz         │
lab queue list/cancel/    queue/control.json                    ▼
  pause ◀──reads/writes──   {paused, budget_usd_per_day,   lab scheduler tick
                             max_concurrent}                    │ claims, launches via the
                          queue/entries/… (state updates) ◀─────┘ existing SkyPilot backend
                          queue/cancelled/, queue/held/  (laptop-owned markers)
                          queue/heartbeat.json
                          (job manifests + artifacts as today)
                                                                ▼
                                                           Vast.ai instances (N concurrent)
```

Decisions made during brainstorming (with the chosen option):
- **Execution model: cloud-side scheduler** on a cheap DigitalOcean Droplet provisioned via the
  playground repo (`cloud-digitalocean` backend). Runs an **idempotent `lab scheduler tick`**
  driven by a systemd timer (~60s) — no long-lived daemon, restart-proof, every tick re-derives
  state from R2. The same command runs on the laptop for Droplet-less use.
- **State/comms: object store as bus.** Queue entries, control, heartbeat, and bundles live under
  an R2 prefix. No inbound ports, no API server; laptop and Droplet only share R2 credentials.
- **Code delivery: bundle to R2.** Registration snapshots the repo (git archive of the commit +
  dirty diff applied), compresses, uploads. The Droplet extracts and submits from the bundle. The
  manifest's `CodeRef` records the original commit + dirty flag, preserving today's provenance.

## 3. Components

New module `src/lab/scheduler/` (CLI and MCP stay thin shells over it):

- **`models.py`** — Pydantic models below; reuses `JobSpec`, `ResourceRequest`, `CodeRef` as-is.
- **`queue.py`** — `QueueStore`: CRUD over the R2 prefix, mirroring `JobStore`/`storage.py`
  patterns. Also a local-filesystem implementation for tests and laptop-only mode.
- **`tick.py`** — `tick(queue, lab, clock, price_feed) -> TickReport`: all scheduling logic,
  dependency-injected (fake clock/prices/store in tests).
- **`bundle.py`** — create/extract code bundles (`git archive` + dirty diff + untracked
  non-ignored files → `.tar.gz`; stdlib `tarfile`, no new dependency). Registration state lives
  **inside** each `entries/<reg_id>.json` (flat layout) — object stores have no atomic move, so
  state-as-prefix would race; the laptop only ever writes markers, never entry state.

### Data model

```python
class DailyWindow(BaseModel):
    start: time          # e.g. 23:00
    end: time            # e.g. 07:00 (may cross midnight)
    tz: str              # IANA name, e.g. "Europe/Berlin"

class Triggers(BaseModel):
    not_before: datetime | None = None   # absolute earliest start
    window: DailyWindow | None = None    # recurring daily eligibility window
    max_hourly_usd: float | None = None  # launch only if a matching offer is at/below this
    offer_query: str | None = None       # extra vastai filter; default derived from accelerators
    after: list[str] = []                # reg_ids in this queue that must reach `succeeded`

class Guardrails(BaseModel):
    expires_at: datetime                 # REQUIRED — past this the entry expires, never launches
    max_cost_usd: float | None = None    # per-job: offer hourly × timeout must fit

RegState = Literal[
    "pending", "launching", "launched", "succeeded", "failed", "expired", "cancelled", "held"
]

class Registration(BaseModel):
    reg_id: str
    created_at: datetime
    spec: JobSpec                        # exactly what Lab.submit takes today
    triggers: Triggers
    guardrails: Guardrails
    bundle_key: str                      # R2 key of the code snapshot
    code: CodeRef                        # commit + dirty captured at registration
    state: RegState
    job_id: str | None = None            # set at launch
    launched_at: datetime | None = None
    last_skip_reason: str | None = None  # why the last tick didn't launch it
```

Trigger semantics: **all present triggers must hold simultaneously** (AND). A registration with
**no triggers is eligible immediately** — this is the supported way to run an "ASAP" job under
Droplet supervision instead of a laptop-bound supervisor (laptop power/network loss then cannot
orphan it). The window gates
**starting only** — a job may start near the window's end; its own wall-clock timeout (FR-I1)
bounds cost. Once `job_id` is set the job is an ordinary lab job: `status`, `logs`, `metrics`,
`fetch`, `cancel`, `wait`, `reconcile` work unchanged. The scheduler's responsibility ends at
launch plus one post-launch price verification.

## 4. Tick algorithm

Each tick:

1. **Heartbeat** — write `queue/heartbeat.json` (timestamp + host id). `lab queue list` warns if
   the heartbeat is stale (>10 min).
2. **Load control** — `queue/control.json`. If `paused`: stop (heartbeat still written).
3. **Sync & watchdog** — for each `launched` registration:
   - Read the job manifest; mirror terminal state (`succeeded`/`failed`/…) back onto the
     registration (this is what advances `after` chains) and **copy the manifest to
     `queue/jobs/<job_id>.json`** so the laptop can `status`/inspect scheduler-launched jobs and
     the Droplet stays truly stateless.
   - If the manifest is non-terminal but the supervisor is dead (`runner_pid` no longer alive —
     the backend already records it), **adopt the job**: the cluster name derives from `job_id`
     (`cluster_name_for`), so query the cluster; if the instance is gone, mark the job `failed`
     (`"supervisor died"`); if it is still running and within `started_at + timeout`, respawn the
     supervisor in adopt mode (re-wait, rsync, teardown); past the timeout, tear down via the
     existing robust path and mark `timed_out`. A Droplet reboot mid-run therefore costs at most
     one tick interval of supervision, never an orphaned rental.
4. **Expire** — `pending`/`held` entries past `expires_at` → `expired` (with reason).
5. **Evaluate triggers** for `pending` entries:
   - *Clock:* `now >= not_before`; if `window` set, local time (in `window.tz`) is inside it
     (handles midnight crossing).
   - *Price:* one `search_offers` call per distinct accelerator spec per tick (deduped). Query is
     derived from `resources.accelerators` plus `offer_query`, defaults include
     `rentable=true rented=false reliability>0.95`. Eligible iff `min(dph_total) <=
     max_hourly_usd`. CPU-only registrations skip the price trigger.
   - *Dependency:* all `after` reg_ids are `succeeded`. If any reached
     `failed`/`expired`/`cancelled`, the dependent is `cancelled` with reason
     `"dependency <id> ended <state>"` (fail-fast; no zombie waits).
6. **Guardrails** on the eligible set:
   - *Per-job:* estimated cost (best offer hourly × timeout) ≤ `max_cost_usd` if set.
   - *Daily budget:* sum of `estimated_usd` of scheduler-launched jobs in the trailing 24h plus
     this launch ≤ `budget_usd_per_day`. Over-budget entries are **skipped** (retried next tick),
     not cancelled.
   - *Concurrency:* scheduler-launched jobs currently running < `max_concurrent`.
7. **Launch** survivors oldest-first: re-check the cancel marker, mark `launching`, download +
   extract bundle to a workdir, `Lab.submit()` via the SkyPilot backend, record `job_id`, mark
   `launched`.
8. **Post-launch price verify** — performed during step 3 of **subsequent ticks** (the rental's
   real `dph_total` is only known once provisioning completes, and the tick must never block):
   if a `launched` entry's manifest shows `cost.hourly_usd` exceeding `max_hourly_usd` by >15%
   (offer raced away), cancel + teardown (existing FR-C2 path) and return the entry to `pending`
   with a skip reason.

Budget arithmetic reuses the `estimated_usd` the manifest already records (FR-I2) — no new cost
model.

## 5. Failure handling

- **Tick crash mid-launch:** an entry stuck in `launching` older than 10 min is repaired on the
  next tick — if a job manifest exists the launch happened (→ `launched`), else revert to
  `pending`. Ticks are idempotent by construction; overlapping ticks are prevented by the systemd
  timer's non-reentrancy plus the `launching` claim state.
- **Cancel race:** laptop cancellation writes a `queue/cancelled/<reg_id>` marker; the tick
  re-checks it immediately before `submit`. If launch wins the race, the entry is `launched` and
  the *job* is cancelled normally.
- **API errors** (`search_offers`, R2): log, set `last_skip_reason`, skip affected entries this
  tick, never crash the tick. Persistent failure is visible via skip reasons + heartbeat age.
- **Single writer for claims:** only the scheduler transitions `pending → launching → launched`;
  the laptop only creates entries and writes cancel/hold markers and control. No concurrent-writer
  conflict on the same key.
- **Registration write ordering:** `lab register` uploads the **bundle first**, then writes the
  queue entry — the entry is the commit point, so the scheduler can never see an entry whose code
  is missing. An interrupted registration (laptop lost network mid-upload) leaves at most an
  unreferenced bundle, never a broken entry.
- **Reconcile sweep:** every ~30 ticks the scheduler runs the existing reconcile check (Vast
  `lab-*` rentals vs. known jobs) and records orphans in the tick report / heartbeat. Flag-only by
  default; `control.json: auto_reconcile=true` lets it destroy confirmed orphans (`reconcile
  --apply` semantics). `lab reconcile` from the laptop remains the independent ground truth for
  "what is billing me right now" regardless of any state-file loss.

## 6. User surface

CLI (mirrored as MCP tools `register`, `queue_list`, `queue_show`, `queue_cancel`, `queue_pause`):

```bash
lab register "<uv run command>" \
    [--gpu RTX_4090:1] --timeout 2h [--seed N] [--config k=v ...] \
    [--window 23:00-07:00] [--not-before <ts>] [--max-hourly 0.25] \
    [--max-cost 1.50] --expires "+3d" [--after <reg_id>] [--hold]
# prints {reg_id, bundle_key, worst-case cost = max_hourly × timeout}

lab queue list            # state + last_skip_reason per entry + heartbeat age
lab queue show <reg_id>   # full registration + trigger evaluation trace
lab queue cancel <reg_id>
lab queue hold <reg_id> / release <reg_id>
lab queue pause / resume  # global switch in control.json
lab queue budget --per-day 5 [--max-concurrent 4]
lab scheduler tick        # what the Droplet timer runs; also usable on the laptop
```

`--expires` is required at registration (guardrail). `register` prints the worst-case cost so the
user authorizes spend at registration time, not at 3am. `--timeout` is required for GPU
registrations (it already is the cost bound, FR-I1).

## 7. Deployment (Droplet)

- New playground lab config (`config/labs/lab-scheduler.yaml`, `cloud-digitalocean` backend,
  smallest droplet — tick is tiny and I/O-bound) + an Ansible role that:
  installs uv, clones laboratory, `uv sync`, writes `/etc/lab/scheduler.env` (Vast API key, R2
  credentials — delivered via Ansible, never in repo/manifests/logs, FR-J1), installs
  `lab-scheduler.service` + `lab-scheduler.timer` (60s, `Persistent=true`, after
  `network-online.target` — ticks resume automatically after a Droplet reboot).
- The Droplet is **stateless** — queue entries, control, bundles, and (via the §4 manifest
  mirroring) job manifests all live in R2 — so destroy/recreate (`playground suspend` / `apply`)
  is always safe: a fresh Droplet reads R2 and the watchdog re-adopts any still-running clusters.
  ~$4–6/mo; suspend when the queue is idle.
- The laptop never needs the Droplet to be reachable; all interaction goes through R2.

## 8. Testing

- **Unit (bulk):** `tick()` with fake clock, fake price feed, in-memory/tmpdir QueueStore, and the
  local/fake backend — table-driven over: window edges incl. midnight crossing and timezones;
  price below/at/above threshold; dependency success and failure propagation; budget exhaustion
  and recovery; concurrency cap; expiry; pause/hold; cancel race; orphaned-`launching` repair;
  post-launch price-verify rollback; watchdog adoption (dead supervisor × instance
  alive/gone/over-timeout); manifest mirroring. Mirrors the fake-cloud style of
  `tests/test_skypilot.py`.
- **Integration:** bundle round-trip (archive + dirty diff → extract → `Lab.submit` on the `local`
  backend); end-to-end `not_before`-in-the-past → tick → job succeeds → dependent launches on the
  next tick.
- **Live smoke (manual, documented):** one real cheap registration through the Droplet overnight,
  verifying heartbeat, launch, price verify, teardown, artifact fetch.
- `mypy --strict` on `src/lab`, ruff line-length 100, as project-wide.

## 9. Scenario coverage (failure walkthroughs)

| Scenario | What happens | Covered by |
|---|---|---|
| Laptop loses power/internet **after registering** | Nothing — Droplet launches and supervises; artifacts land in R2 | Cloud-side scheduler (§2) |
| Laptop dies mid **direct `lab submit`** (non-scheduled) | Pre-existing weakness: laptop supervisor dies; remote autostop/in-instance timeout + `lab reconcile` are the nets | Unchanged; mitigation: register with **no triggers** → ASAP launch under Droplet supervision (§3) |
| Laptop offline during `lab register` | Bundle-first write ordering → at worst an unreferenced bundle, never a code-less entry | §5 |
| **Droplet** reboots/dies mid-run | Timer resumes (`Persistent=true`); watchdog detects dead `runner_pid`, adopts the cluster, enforces timeout, tears down, records true state | §4.3, §7 |
| Droplet destroyed permanently | All state in R2; recreate via playground, watchdog re-adopts live clusters | §7 |
| Vast instance dies / preempted mid-run | Supervisor sees terminal/missing job → `failed`, robust teardown; registration mirrors it; dependents fail fast. No auto-retry in v1 | §4.3, §5 (existing sky_runner) |
| Scheduler silently broken | Heartbeat age shown in `lab queue list`; `last_skip_reason` per entry | §4.1, §6 |
| "What is actually billing me right now?" | `lab reconcile` queries Vast directly — independent of every state file | §5 |
| "Which experiments are still running?" | `lab queue list` (registrations) + R2-mirrored manifests (`lab status <job_id>` works from the laptop) + reconcile sweep for orphans | §4.3, §5 |
| R2 unreachable from Droplet | Tick logs and exits without action; in-flight jobs unaffected (supervisor is local); next tick retries | §5 |

## 10. Build order (suggested)

1. Models + QueueStore (tmpdir backend) + bundle create/extract — pure, fully unit-testable.
2. `tick()` with clock/dependency/expiry/pause + local backend launches (no price, no R2).
3. R2 QueueStore + `lab register` / `lab queue *` CLI + MCP tools.
4. Price feed (`search_offers`) + budget/concurrency guardrails + post-launch verify.
5. Playground lab config + Ansible role + systemd units; live smoke.
