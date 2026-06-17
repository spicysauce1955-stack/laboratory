# CPU backend (`--backend cpu`, DigitalOcean)

Run CPU-bound jobs (e.g. solver/oracle work) on a cheap multi-core DigitalOcean droplet instead
of paying for an idle GPU.

```bash
uv run lab submit --backend cpu -c "uv run analysis/v12_existence_oracle.py --milp" --timeout 1h
uv run lab submit --backend cpu --cpus 32 -c "..." --timeout 2h    # up to 48 vCPU
```

`--backend cpu` provisions a DO droplet via SkyPilot: it clears accelerators, defaults to **8
vCPU**, and runs **on-demand** (DO has no spot). SkyPilot picks the smallest droplet meeting
`--cpus`:

| Instance | vCPU | RAM | $/hr |
|---|---|---|---|
| `g-8vcpu-32gb` | 8 | 32 GB | ~$0.36 |
| `g-16vcpu-64gb` | 16 | 64 GB | $0.75 |
| `g-32vcpu-128gb` | 32 | 128 GB | $1.50 |
| `g-48vcpu-192gb-intel` | 48 | 192 GB | $2.70 |

**48 vCPU is the single-box ceiling**; for more, shard across jobs. The launched instance type is
recorded in the manifest (`backend.machine_type`).

## One-time setup
- `uv sync --extra skypilot --extra do`
- `doctl auth init` (writes a token to `~/.config/doctl/config.yaml`); confirm `sky check` shows
  **DO: enabled**.
- A DO vCPU/droplet **quota increase** may be needed for 32/48-vCPU droplets.

## Cost-safety
Teardown is `sky.down` + idle autostop + the on-box poweroff backstop. `lab reconcile` covers DO
via a cloud-agnostic `sky.status` orphan pass (`sky_orphans`/`sky_destroyed` in its report).
