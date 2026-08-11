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

Rough on-demand prices (us-central1; spot is ~60-70% off but reclaimable):

| Resource | ~$/hr |
|---|---|
| `e2-standard-4` (4 vCPU, cpu profile pick) | ~$0.13 |
| `n1-standard-4 + T4:1` | ~$0.55 |
| `g2-standard-4 + L4:1` | ~$0.70 |
| `a2-highgpu-1g + A100:1` | ~$3.70 |

Preempted spot jobs are classified `preempted` (not `failed`) and the scheduler's
auto-resubmit applies to them like any spot job.

## One-time setup

- `uv sync --extra skypilot --extra gcp`
- **Authenticate.** There is no GCP "API key" for provisioning: SkyPilot and the lab's own
  compute-API passes both go through **Application Default Credentials** (`google.auth.default()`).
  Pick one of two paths — see *Credentials* below.
- Select a project (`gcloud config set project <id>`, or `GOOGLE_CLOUD_PROJECT` in `.env`) with
  billing enabled, and enable the Compute Engine API.
- Confirm `uv run sky check gcp` shows **GCP: enabled**.
- **GPU quota:** a fresh project has 0 GPU quota. Request per-family regional quota
  (e.g. `NVIDIA_T4_GPUS` / `NVIDIA_L4_GPUS` in your region) in the console before the first GPU
  job, or provisioning fails with a quota error (`lab` surfaces a `sky check gcp`/quota hint).

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
| `roles/storage.admin` | SkyPilot's staging bucket (`storage.buckets.create/delete`) |

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
time-window/dependency triggers instead.

## Cost-safety

GCP has **two teardown channels**, mirroring the Vast design:

- In-band: `sky.down` with retries + idle autostop + the on-box poweroff backstop. If every
  `sky.down` attempt fails, `robust_teardown` falls back to a **gcp-direct destroy** via the
  compute API (bypassing SkyPilot's registry), same as the vastai-sdk fallback on Vast. Only if
  *that* also fails does `teardown_status="failed"` land on the manifest (`lab wait` exits 3).
- Out-of-band: `lab reconcile` runs (a) the cloud-agnostic `sky.status` orphan pass and (b) a
  **GCP compute-API pass** listing `lab-*` instances and **unattached `lab-*` persistent disks**
  (a disk that outlives its VM keeps billing — the GCP analogue of the DO volume leak).
  `--apply` deletes both. The pass skips silently when GCP isn't configured (no ADC).
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
- **`n4` capacity is tight, and the lab cannot steer around it.** SkyPilot's GCP catalog resolves
  every `cpus`/`memory` combination the cpu profile can express to the **n4** family
  (`n4-standard-4` for the 4-vCPU default). When n4 is exhausted you get
  `ZONE_RESOURCE_POOL_EXHAUSTED` — observed across all four `us-central1` zones *and* `us-east1-b`
  in one run — and `ResourceRequest` exposes no instance-type or region override. **`use_spot=true`
  (with `spot_fallback`) is the practical workaround**: it re-prices the search and pushed the
  optimizer to `europe-west1-b`, which had capacity, at $0.034/hr instead of $0.18.
- **A failed provision reports as a runtime-setup error.** When no zone yields a VM, the surfaced
  message is `Failed to set up SkyPilot runtime on cluster` /
  `Could not find any head instance` — that is the *downstream* symptom. Grep the log for
  `ZONE_RESOURCE_POOL_EXHAUSTED` before chasing a runtime bug.
- **`teardown_status` lags the terminal state.** `lab wait` can return with the job terminal but
  `teardown_status: null` (reported as `teardown_unconfirmed`), which resolves to `"succeeded"` a
  few seconds later. Re-read `lab status` before treating it as a leak — and confirm with
  `lab reconcile`, which queries the live compute API and is ground truth.

## Limitations (v1)

- No region/zone pinning — SkyPilot's optimizer picks the cheapest feasible region; the launched
  region is recorded in the manifest (`backend.region`).
- Billed cost uses SkyPilot's catalog estimate (accurate for on-demand; spot varies by zone).
