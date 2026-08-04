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
- `gcloud auth login` **and** `gcloud auth application-default login` (SkyPilot uses ADC), with a
  project selected (`gcloud config set project <id>`) and billing enabled.
- Enable the Compute Engine API; confirm `uv run sky check gcp` shows **GCP: enabled**.
- **GPU quota:** a fresh project has 0 GPU quota. Request per-family regional quota
  (e.g. `NVIDIA_T4_GPUS` / `NVIDIA_L4_GPUS` in your region) in the console before the first GPU
  job, or provisioning fails with a quota error (`lab` surfaces a `sky check gcp`/quota hint).

## Deferred jobs (scheduler)

`lab register --cloud gcp` works like any registration — the cloud rides in the registered spec
and the scheduler launches on GCP. **Exception:** `--max-hourly` / `--offer-query` price triggers
query the **Vast offer feed only** and are rejected (fail-loud) for non-Vast clouds; use
time-window/dependency triggers instead.

## Cost-safety

- Teardown is `sky.down` with retries + idle autostop + the on-box poweroff backstop. There is
  **no provider-direct kill fallback** for GCP (that exists for Vast only); a persistent teardown
  failure flips `teardown_status="failed"` on the manifest and `lab wait` exits 3.
- `lab reconcile` covers GCP via the cloud-agnostic `sky.status` orphan pass
  (`sky_orphans`/`sky_destroyed` in its report); the Vast-direct pass is skipped when vastai-sdk
  isn't installed. Dry-run exits 3 when either pass finds orphans.
- Manual double-check if you suspect a leak:
  `gcloud compute instances list --filter="name~'^lab-'"`.
- SkyPilot's GCP teardown deletes the boot disk with the VM; no static IPs are allocated per
  cluster, so there is no GCP analogue of the DO volume-leak pass.

## Limitations (v1)

- No region/zone pinning — SkyPilot's optimizer picks the cheapest feasible region; the launched
  region is recorded in the manifest (`backend.region`).
- Billed cost uses SkyPilot's catalog estimate (accurate for on-demand; spot varies by zone).
