# 15 — Compute Shape & Storage (spec §13.4, §13.5)

The P0 workload (tempotron-capacity) is **CPU-bound and embarrassingly parallel** over seeds/α/K.
That changes the compute question from "which GPU" to "cheapest cores/RAM, ideally interruptible."

## Compute shape (§13.5): single fat node first

- **Recommendation: single multicore node for P0.** A sweep over K runs as parallel processes on
  one node (joblib/`multiprocessing`/Hydra multirun), which is simplest (FR-H1) and avoids
  per-job provisioning overhead.
- **P1 `sweep` = job-per-point on spot:** fan out N managed-spot jobs (one per K/α), each
  independently monitorable under a `sweep_id` (FR-A5, §9 `sweep`). Spot fits because each point is
  short, fault-tolerant, and checkpointable; SkyPilot managed spot auto-recovers.

## Where to get the cores

| Path | $ (approx) | Notes |
|---|---|---|
| **Hetzner dedicated (CCX)** | CCX13 8 vCPU/80GB SSD ≈ $17/mo; larger CCX scale up | **Best value** for a steady fat node. Dedicated vCPU (not oversubscribed like their shared line). **Not** in SkyPilot's provider list → use via `local`/manual backend. |
| **AWS/GCP/Azure spot CPU** | **75–90% off** on-demand (e.g. C7i.large spot ≈ $25/mo; Azure ARM spot ≈ $11–15/mo) | The SkyPilot-managed path for elastic sweeps; needs interruption tolerance (we have it). |
| **Vast.ai CPU / Oracle A1 ARM** | very cheap; Oracle has a free 4× ARM tier | Vast.ai is SkyPilot-supported; Oracle A1 great value, generous free tier. |
| **Vultr / cloud on-demand** | 8 vCPU/32GB ≈ $0.22/hr; 24 vCPU/96GB ≈ $0.88/hr | Simple hourly if you want predictable, no-spot. |

**Guidance for this lab:** keep two profiles —
1. **`local`** backend on a cheap **Hetzner dedicated** node (or this machine) for steady,
   cheapest CPU crunching and offline dev (NFR-4).
2. **`skypilot`** backend on **spot CPU** (AWS/GCP/Azure or Vast.ai) for elastic P1 sweep fan-out
   with auto-teardown + auto-recovery.

GPU (P1) reuses the same `skypilot` backend with a GPU resource request — RTX 4090/L40S tier is
plenty (see `01-compute-providers.md`); H100s only for real DL.

## Storage (§13.4): object store, not rsync-only

- **Recommendation: object store (Cloudflare R2 / S3) as canonical**, because remote VMs are torn
  down (FR-C2) — artifacts must be **pushed off-box before teardown** to survive. `fetch_artifacts`
  then pulls into `runs/<job_id>/`.
- **Cloudflare R2 / Backblaze B2 = no egress fees** — important since the agent re-pulls artifacts
  and re-pulls input data to fresh boxes; hyperscaler egress would add up.
- **rsync** is fine for the **`local`** backend (artifacts already on disk) and as a fast path, but
  it does not survive teardown for remote jobs → object store is the portable default. SkyPilot can
  mount the bucket (`file_mounts`/Storage) so the run writes straight to durable storage.

## Auth/secrets (§13.6)

- Cloud creds live only in the SkyPilot **API server env / `~/.sky`** on the control box;
  least-privilege per provider (FR-J1/J2).
- Tracker key (if W&B) and object-store keys are injected into the remote as **SkyPilot
  secrets/`envs`** at launch — never committed, never logged (FR-J1).
- The MCP server reads its own creds from a gitignored `.env`. **Manifests record URIs, not keys.**
- Per-job isolation + destroy-after-use (FR-J3) is inherent to ephemeral managed jobs.

Sources: `sources/lab-system-raw.md` (CPU VM perf/price 2026, Hetzner/spot, R2 no-egress).
