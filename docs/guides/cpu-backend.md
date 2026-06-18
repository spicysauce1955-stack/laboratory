# CPU backend (`--backend cpu`, DigitalOcean)

Run CPU-bound jobs (e.g. solver/oracle work) on a cheap multi-core DigitalOcean droplet instead
of paying for an idle GPU.

```bash
uv run lab submit --backend cpu -c "uv run analysis/v12_existence_oracle.py --milp" --timeout 1h
uv run lab submit --backend cpu --cpus 32 -c "..." --timeout 2h    # up to 48 vCPU
```

`--backend cpu` provisions a DO droplet via SkyPilot: it clears accelerators, defaults to **4
vCPU** with a **50 GB** attached volume, and runs **on-demand** (DO has no spot). SkyPilot picks
the smallest droplet meeting `--cpus`:

| Instance | vCPU | RAM | $/hr |
|---|---|---|---|
| `g-8vcpu-32gb` | 8 | 32 GB | ~$0.36 |
| `g-16vcpu-64gb` | 16 | 64 GB | $0.75 |
| `g-32vcpu-128gb` | 32 | 128 GB | $1.50 |
| `g-48vcpu-192gb-intel` | 48 | 192 GB | $2.70 |

**48 vCPU is the single-box ceiling**; for more, shard across jobs. The launched instance type is
recorded in the manifest (`backend.machine_type`).

## Account tier (defaults are deliberately small)
A **fresh DO account** is tier-restricted in two ways that both surface as an opaque SkyPilot
"Failed to provision … in all zones" — the real cause is in `sky_logs/.../provision.log`:
- **Large droplet sizes** (8-vCPU `s-8vcpu-16gb` and the `g-*` sizes above) → DO `422 This size is
  currently restricted, please open a ticket to increase your account tier`.
- **SkyPilot's default 256 GB volume** → DO `422 failed to create volume: invalid size specified`
  (the DO provisioner always attaches a block volume of `disk_size` GB; fresh-tier accounts cap it
  well under 256). That's why the `cpu` profile defaults `disk_size` to **50 GB** (override with
  `--disk-size` once explicitly via the API; the field lives on `ResourceRequest`).

So the defaults (4 vCPU / 50 GB) are chosen to provision on an untouched DO account. To go bigger
(`--cpus 32`, larger volumes) **open a DO ticket to raise your account tier** first.

## One-time setup
- `uv sync --extra skypilot --extra do`
- `doctl auth init` (writes a token to `~/.config/doctl/config.yaml`); confirm `sky check` shows
  **DO: enabled**.

## Cost-safety
Teardown is `sky.down` + idle autostop + the on-box poweroff backstop. `lab reconcile` covers DO
via a cloud-agnostic `sky.status` orphan pass (`sky_orphans`/`sky_destroyed` in its report).
> **Volume caveat:** `reconcile`'s orphan pass checks **instances, not block volumes**. SkyPilot's
> DO teardown deletes the attached volume together with the droplet, but if a teardown ever leaves
> the droplet gone and the volume behind, `reconcile` will not flag it — check
> `doctl compute volume list` for stray `lab-*` volumes if you suspect a leak.
