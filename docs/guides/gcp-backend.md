# Google Cloud backend (`--cloud gcp`, CPU + GPU)

Run lab jobs on Google Cloud — cheap CPU boxes (an alternative to the DigitalOcean `cpu`
profile) and on-demand/spot GPUs (a more reliable alternative to the Vast spot market).

```bash
# CPU: the cpu profile (4 vCPU + 50 GB) on GCP instead of DO
uv run lab submit --backend cpu --cloud gcp -c "python experiments/x.py" --timeout 1h

# GPU: T4 / L4 on demand
uv run lab submit --backend skypilot --cloud gcp --accelerators T4:1 -c "..." --timeout 2h

# GPU spot (preemptible) with automatic on-demand fallback if spot is scarce
uv run lab submit --backend skypilot --cloud gcp --accelerators L4:1 --spot -c "..." --timeout 2h

# Sweeps and deferred jobs take the same flag
uv run lab sweep --backend cpu --cloud gcp -c "..." --grid lr=0.1,0.01 --timeout 1h
uv run lab register --cloud gcp --gpu T4:1 --timeout 2h --expires +3d -c "..."
```

`--cloud` selects the SkyPilot cloud (`vast` default | `do` | `gcp`) on `submit`, `sweep`,
`register`, and `register-sweep` (and as `cloud=` on the MCP tools). `--backend cpu --cloud gcp`
keeps the cpu profile's defaults (4 vCPU, 50 GB disk, no accelerators) but provisions on GCP —
and unlike DO, GCP CPU jobs may use `--spot` (preemptible).

### What it costs

Don't budget from a table — ask the catalog, which is a local file and always current:

```bash
uv run lab doctor --cloud gcp --cpus 4          # -> n4-standard-4, 41 regions, $0.19-$0.30/hr
uv run lab doctor --cloud gcp --gpu T4:1 --spot # -> n1-highmem-4, 23 regions, $0.17-$0.36/hr
```

For orientation only (catalog, 2026-08-11), **compute plus the lab's default disk**:

| Shape | resolves to | on-demand $/hr | spot $/hr (cheapest-dearest region) |
|---|---|---:|---|
| cpu profile (4 vCPU) | `n4-standard-4` | 0.181-0.290 | **0.034**-0.123 |
| `--accelerators T4:1` | `n1-highmem-4` | 0.600-0.869 | 0.157-0.346 |

Two things that table makes visible and the old one hid:

- **Region matters more than anything else you can choose.** Spot `n4-standard-4` is 3.6x dearer
  in its worst region than its best. The optimizer already shops for you; `--region` overrides it.
- **`--spot` without `--no-fallback` can bill on-demand.** `spot_fallback` defaults on, so the
  worst case for a spot cpu job is **$0.290/hr**, not $0.123 — about 8.5x the price you were
  probably budgeting. That is the number `lab register` now authorises against, and
  `lab status` reports `spot_downgraded: true` when it actually happens.

Storage is billed separately and is now on the manifest. It is not a rounding error: SkyPilot's
default 256 GB boot disk costs **$0.028/hr** on hyperdisk-balanced (what n4 must use) or
**$0.035/hr** on pd-balanced (everything else) — that is 82-103% of a $0.034/hr spot
`n4-standard-4`, so the untuned disk roughly *doubles* a cheap job's bill. The lab therefore never
lets that default apply: every GCP job carries an explicit size (**50 GB** cpu profile,
**100 GB** GPU), costing $0.0055 and $0.0137/hr respectively.

### Choosing where it lands

```bash
--region europe-west1        # pin a region (validated against the catalog before anything bills)
--zone   europe-west1-b      # pin a zone
--price-cap 0.06             # refuse any instance above $0.06/hr compute
```

`--price-cap` is a **ceiling enforced by SkyPilot's optimizer**, not an estimate: it will not
select an option above it. Set it below everything available and the job fails in about a minute
with *"no instance type matches this spec, so nothing was provisioned (you were not billed)"* —
the optimizer rejects the spec before touching the cloud, so an over-tight cap costs time, never
money. Note it is not `--max-hourly` on `lab register`, which is a Vast-only *wait-until* price
trigger — a scheduling condition, not a limit.

With no pins the optimizer searches every region cheapest-first, as before. The one thing the lab
adds is a **capacity memo**: when a launch fails with `ZONE_RESOURCE_POOL_EXHAUSTED`, the zones
named are remembered for 30 minutes (`LAB_CAPACITY_MEMO_TTL_S`) and excluded from the next
launch's search space. This matters most for sweeps — without it, all 32 shards independently
walk into the zone the first shard already found empty. The memo is advisory: if it is missing,
corrupt, or would exclude every region, it is ignored rather than allowed to block a launch.

Preempted spot jobs are classified `preempted` (not `failed`) and the scheduler's
auto-resubmit applies to them like any spot job.

## One-time setup

- `uv sync --extra skypilot --extra gcp`
- **Authenticate.** There is no GCP "API key" for provisioning: SkyPilot and the lab's own
  compute-API passes both go through **Application Default Credentials** (`google.auth.default()`).
  Pick one of two paths — see *Credentials* below.
- Select a project (`gcloud config set project <id>`, or `GOOGLE_CLOUD_PROJECT` in `.env`) with
  billing enabled, and enable the Compute Engine API.
- **Check it with `uv run lab doctor --cloud gcp`** — one command in place of the chain of manual
  checks below. It verifies credentials (including SkyPilot's API-server daemon, which does *not*
  inherit `.env`), the project, billing, both required APIs, every IAM permission SkyPilot needs,
  and quota for the shape you name; then prints what the catalog says it will cost. Exit 1 means
  something would fail. Add `--gpu T4:1` before your first GPU job.
  **The cheap half of these checks also runs automatically before every remote launch**, so a
  missing permission or an exhausted quota costs about a second instead of a provisioning round
  trip. Only a check that positively establishes "this cannot work" blocks; one that merely fails
  to answer is skipped, because a preflight that refused a job because the *preflight* broke would
  be worse than none. `--no-preflight` opts out.
- **GPU quota is enforced at two levels and you need both.** A fresh project has 0 GPU quota.
  Request per-family *regional* quota (e.g. `NVIDIA_T4_GPUS` in your region) **and** the separate
  *global* `GPUS_ALL_REGIONS`. These are independent: a project with `NVIDIA_T4_GPUS = 1` and
  `GPUS_ALL_REGIONS = 0` looks fine on the quota page and fails every GPU launch with
  `Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally`. `lab doctor --cloud gcp --gpu T4:1`
  checks both. First-time GPU quota requests can take up to 48 hours.
- Ignore `PREEMPTIBLE_CPUS = 0` — it is the default and is not a blocker. Where preemptible quota
  was never granted, GCP runs Spot VMs against the standard `CPUS` quota.

### Credentials

**(a) User login — interactive, nothing to configure in the repo.**

```bash
gcloud auth login
gcloud auth application-default login   # SkyPilot reads these ADC creds
gcloud config set project <project-id>
```

**(b) Service-account key — non-interactive (CI, the always-on scheduler host, headless boxes).**
Create a JSON key (Console → IAM & Admin → Service Accounts; roles **Compute Admin** +
**Service Account User**), save it **outside the repo**, and point at it from `.env`:

```bash
cp .env.example .env          # .env is git-ignored (FR-J1); .env.example is the committed template
chmod 600 ~/.config/gcloud/lab-sa.json
# then in .env:
#   GOOGLE_APPLICATION_CREDENTIALS=/home/you/.config/gcloud/lab-sa.json
#   GOOGLE_CLOUD_PROJECT=my-project-id
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
```

The last line matters: SkyPilot shells out to `gcloud` for some GCP operations, so activating the
service account there too avoids a half-authenticated state where `sky check gcp` passes but
provisioning fails.

> **Gotcha — `.env` alone does not reach SkyPilot.** `load_lab_env` sets the variables inside the
> *lab* process, but SkyPilot runs a long-lived **API server daemon** that does not inherit them:
> a daemon started before (or outside) a lab command reports `GCP: disabled — Your default
> credentials were not found` even though `.env` is correct. Two fixes, in order of robustness:
>
> 1. **Symlink the key to the well-known ADC path** (best for headless / service-account setups,
>    including the systemd scheduler host) — then *every* process finds it with no env plumbing:
>    ```bash
>    ln -s ~/.config/gcloud/lab-sa.json ~/.config/gcloud/application_default_credentials.json
>    ```
>    `google.auth.default()` honours a service-account JSON at that path. Don't do this if you
>    already have user ADC there — it would shadow your login.
> 2. **Restart the daemon** after changing credentials: `uv run sky api stop` (the next lab command
>    restarts it, inheriting the current environment).

### Required APIs

Enable **both** on the project before the first job — SkyPilot's `check_credentials` calls
Resource Manager, and everything else calls Compute:

- `compute.googleapis.com` (Compute Engine API)
- `cloudresourcemanager.googleapis.com` (Cloud Resource Manager API)

A `SERVICE_DISABLED` / `403 … has not been used in project … before or it is disabled` from either
means one is off. `gcloud services enable <api>` works only if the caller holds
`roles/serviceusage.serviceUsageAdmin` — a stock **Compute Admin + Service Account User** service
account gets `AUTH_PERMISSION_DENIED`, so enable them as a project owner in the Console (or grant
that role first).

### Required IAM roles

Compute Admin + Service Account User is **not** enough for SkyPilot: `sky check gcp` reports
`GCP: disabled` listing missing `serviceusage.services.use/enable`,
`resourcemanager.projects.getIamPolicy`, `iam.roles.get`, and
`storage.buckets.create/delete`. Grant the service account all of:

| Role | Covers |
|---|---|
| `roles/compute.admin` | provisioning VMs, disks |
| `roles/iam.serviceAccountUser` | attaching the VM's service account |
| `roles/iam.serviceAccountAdmin` | creating the VM service account at launch |
| `roles/serviceusage.serviceUsageAdmin` | `serviceusage.services.use` / `.enable` |
| `roles/iam.securityReviewer` | `resourcemanager.projects.getIamPolicy`, `iam.roles.get` |
| `roles/storage.admin` | `sky check gcp`'s `storage.buckets.create/delete` probe |

`roles/storage.admin` is needed to make `sky check gcp` pass, **not** because this lab stages
anything into object storage. SkyPilot only creates a `skypilot-filemounts-*` bucket from
`maybe_translate_local_file_mounts_and_sync_up`, which runs for controller-backed launches
(`sky jobs launch` / `sky serve up`) and translates `file_mounts` / `storage_mounts`. The lab
calls plain `sky.launch` with a `workdir`, rsynced over SSH — so no bucket is ever created and
none needs reaping. `test_the_launch_path_creates_no_object_storage` fails if that changes.

Grant them **as a project owner** — a service account cannot grant itself roles (it can't even
`getIamPolicy`, so `add-iam-policy-binding` fails at the read):

```bash
gcloud auth login                       # as an owner, not the service account
for R in compute.admin iam.serviceAccountUser iam.serviceAccountAdmin \
         serviceusage.serviceUsageAdmin iam.securityReviewer storage.admin; do
  gcloud projects add-iam-policy-binding <project-id> \
    --member="serviceAccount:<sa>@<project-id>.iam.gserviceaccount.com" \
    --role="roles/$R" --condition=None >/dev/null
done
```

Switching gcloud's active account does **not** disturb the lab: ADC comes from the well-known
file, not from `gcloud config get account`.

`.env` is loaded by `lab.env.load_lab_env` at CLI/MCP startup (`src/lab/env.py`). Values already
exported in your shell **win** over the file, so `GOOGLE_CLOUD_PROJECT=other uv run lab submit …`
overrides for a single command. Blank entries are treated as unset — an unfilled
`GOOGLE_APPLICATION_CREDENTIALS` simply falls through to path (a). A path that doesn't exist fails
loudly at startup instead of surfacing as an opaque auth error mid-provision. The file holds the
key's **path**, never the key itself, and never reaches a manifest or a remote box (the remote
syncs without the `cli` group).

## Deferred jobs (scheduler)

`lab register --cloud gcp` works like any registration — the cloud rides in the registered spec
and the scheduler launches on GCP. **Exception:** `--max-hourly` / `--offer-query` price triggers
query the **Vast offer feed only** and are rejected (fail-loud) for non-Vast clouds; use
time-window/dependency triggers instead. Cost guardrails still apply: `--max-cost` and the daily
budget price non-Vast clouds from SkyPilot's catalog.

> **The scheduler host needs its own GCP credentials.** It is a different machine, and nothing is
> copied to it — a host provisioned before you started using GCP has neither the `gcp` extra nor
> any ADC, so a deferred GCP job queues fine, passes its triggers, and fails at launch overnight.
> Set it up (and run `sky check gcp` there) before relying on it: `deploy/scheduler/README.md`,
> section *Google Cloud credentials*.

## Cost-safety

GCP has **two teardown channels**, mirroring the Vast design:

- In-band: `sky.down` with retries + idle autostop + the on-box poweroff backstop. If every
  `sky.down` attempt fails, `robust_teardown` falls back to a **gcp-direct destroy** via the
  compute API (bypassing SkyPilot's registry), same as the vastai-sdk fallback on Vast. Only if
  *that* also fails does `teardown_status="failed"` land on the manifest (`lab wait` exits 3).
- Out-of-band: `lab reconcile` runs (a) the cloud-agnostic `sky.status` orphan pass and (b) a
  **GCP compute-API pass** listing `lab-*` instances and **unattached `lab-*` persistent disks**
  (a disk that outlives its VM keeps billing — the GCP analogue of the DO volume leak).
  `--apply` deletes both, waiting for each delete operation to actually complete rather than
  trusting the accepted request. The instance and disk passes are independent, and each reports
  its own outcome in the report (`gcp_pass`, `gcp_disk_pass`).
  **Any orphan from any pass exits 3** in dry-run mode — the alarm reads the union, not a subset.
- **"Skipped" vs "failed" matters.** The GCP passes skip only when GCP genuinely isn't set up on
  this machine (no ADC, no project, extra not installed) — the report says so. An actual API
  failure (revoked role, expired key, disabled API, 5xx) **raises** instead of reporting clean:
  a leak-detection pass that swallows an error claims coverage it doesn't have.
- **The on-box `poweroff` backstop is compute-only on GCP.** It fires at `wall + 600s` and is
  described elsewhere as "a hard backstop"; on GCP that is only half true. `poweroff` puts the VM
  in `TERMINATED`, which stops **compute** billing — but the persistent disk survives a
  TERMINATED instance and **keeps billing indefinitely**. So on GCP the backstop converts a
  compute leak into a storage leak. That is a real improvement (compute is the expensive part),
  but it is not a release of resources, and nothing on the box will ever release the disk. The
  actual teardown path is `down=True` + autostop; `lab reconcile`'s unattached-disk pass is the
  net behind it. (For contrast: on Vast `poweroff` ends the rental outright and nothing survives;
  on DO it powers the droplet off and you keep paying full price for the whole droplet.)
- **The GCP passes only claim what SkyPilot actually created.** A resource is ours only if it
  matches the real node shape `lab-…-<head|worker>-<uuid8>-<compute|tpu|mig>` — e.g.
  `lab-20260811-144501-c5b340-3dd12990-head-c0h9pkx0-compute`, where `3dd12990` is the SkyPilot
  user hash and the job id is *not* reliably recoverable (`make_cluster_name_on_cloud` truncates
  past GCP's 35-char limit, and our names fit by exactly zero characters); a bare
  `lab-` prefix is not enough, because `--apply` deletes without prompting and a shared project's
  `lab-notebook` would have matched. Anything else named `lab-*` is listed under `gcp_unmatched`
  and never destroyed — check that list if you expected a leak and saw none.
- **Check `gcp_project` in the report.** The passes resolve the project ambiently from ADC, while
  SkyPilot can be pinned to a *different* project in `~/.sky/config.yaml`. If those disagree,
  reconcile sweeps a project the lab never launches into and truthfully reports it clean. The
  report names the project it swept so you can tell that apart from a real all-clear.
- Manual double-check if you suspect a leak:
  `gcloud compute instances list --filter="name~'^lab-'"` and
  `gcloud compute disks list --filter="name~'^lab-' AND -users:*"`.

## Gotchas (live-learned 2026-08-11)

- **The interpreter must be pinned (`.python-version` = `3.12`).** Without it `requires-python
  = ">=3.12"` lets the remote `uv sync` grab the newest interpreter it can find — a GCP image
  resolved **Python 3.14.7**, for which the `numpy<2` pin (1.26.4) publishes no wheels. uv fell
  back to building NumPy from sdist and the image has **no C compiler** (`cc`/`gcc`/`clang` all
  absent), so setup died with `FAILED_SETUP`. Pinning 3.12 restores the wheel path (verified:
  remote came up on 3.12.13, zero sdist builds). This is latent for every remote backend, not
  just GCP — the lockfile pins packages but not the interpreter that resolves them.
- **`n4` capacity is tight.** SkyPilot's GCP catalog resolves every `cpus`/`memory` combination
  the cpu profile can express to the **n4** family (`n4-standard-4` for the 4-vCPU default). When
  n4 is exhausted you get `ZONE_RESOURCE_POOL_EXHAUSTED` — observed across all four `us-central1`
  zones *and* `us-east1-b` in one run. Three levers now exist: `--region`/`--zone` to steer
  directly, the capacity memo to stop repeating a known-empty zone, and a **20-minute** GCP
  provisioning budget (was 8, a figure calibrated to Vast host behaviour). That last one was the
  real bug in the original run: SkyPilot's failover *was* working, and the watchdog killed it
  part-way through. `--spot` remains a good idea, but as a price choice, not a capacity workaround.
- **A failed provision reports as a runtime-setup error.** When no zone yields a VM, the surfaced
  message is `Failed to set up SkyPilot runtime on cluster` /
  `Could not find any head instance` — that is the *downstream* symptom. Grep the log for
  `ZONE_RESOURCE_POOL_EXHAUSTED` before chasing a runtime bug.
- **Teardown is asynchronous, and `lab reconcile` can catch it mid-flight.** `lab wait` returns
  when the *job* is terminal; `sky.down` and GCE's own delete operation run after that. So a
  `lab wait` that exits 0 can be followed by `teardown_status: null` (reported as
  `teardown_unconfirmed`) and — measured 2026-08-11 — by a `lab reconcile` that lists the head node
  as `RUNNING`. It was gone ~40s later. Reconcile *is* ground truth, but ground truth includes
  "still shutting down": give it a minute and re-run before treating a fresh orphan as a leak. An
  instance still listed several minutes after the job ended is real, and `lab reconcile --apply`
  is the fix.

## Spot preemption is read, not guessed

On the unmanaged spot path the lab normally *infers* preemption: spot, plus the box vanished,
plus no authoritative terminal status. GCE does not require that guess — a preempted Spot VM goes
`TERMINATED` with `scheduling.preemptible = true` — so on GCP the lab asks the compute API before
falling back to the inference.

This matters because the inference is wrong in the expensive direction: a job that genuinely
*failed* on a box that happened to disappear looked identical to a preemption, and the scheduler
auto-resubmits preempted jobs — so the same failure got paid for twice. GCE's answer wins over
the inference in both directions, and if the probe can't answer (no ADC, revoked role, a 5xx, or
the instance already deleted) the old inference stands unchanged.

## Limitations (v1)

- Billed cost is compute (catalog, priced for the region actually launched into) plus disk. It
  excludes **egress** and **sustained-use discounts**, so it reads slightly high on a long
  on-demand run and does not model data transfer at all.
- Quota preflight checks the cheapest three candidate regions, not all ~40; a launch can still
  fail on quota in a region the optimizer reached and `lab doctor` did not sample.
