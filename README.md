# Laboratory — Remote Experiment Runner

Run computational experiments on remote machines, decoupled from the local session: submit heavy
jobs and keep working, watch metrics live and kill early if off-track, and get results back
**reproducibly**. Experiment-agnostic core — any script honoring the Experiment Contract runs
unchanged.

- **Spec:** [`LAB-REQUIREMENTS.md`](LAB-REQUIREMENTS.md) (RFC-2119, phased P0/P1/P2)
- **Research & design decisions:** [`research/`](research/) — start at `research/README.md`,
  decisions in `research/16-decisions.md`, architecture in `research/10-architecture.md`.

## Status

P0 in progress. **Working today (local backend, no credentials):** submit → run → status/logs →
fetch loop via both the CLI and the MCP server, with reproducible per-job manifests.
- `lab.core` + `lab.store` + `lab.runner` + `lab.backends.local` (detached subprocess supervisor:
  env injection, wall-clock timeout, auto-recorded terminal state).
- `lab.cli` (Typer) and `lab.mcp_server` (FastMCP, structured returns) — thin shells over the
  same `Lab` core, both with backend selection (local | skypilot).
- **`skypilot` remote backend** — validated end-to-end on Vast.ai: provision → remote `uv sync`
  → run → durable artifacts on Cloudflare **R2** → autostop teardown.
- 28 tests, ruff-clean.

Next (see `research/16-decisions.md`): push notifications; sweeps; leaner remote dep group.

## Quickstart (dev)

```bash
uv sync                                # local backend + CLI/MCP (lean default)
uv sync --extra skypilot --extra r2    # + remote (Vast) backend & durable R2 (full dev)

# CLI
uv run lab submit -c "python experiments/example_capacity.py" --seed 42
uv run lab list
uv run lab status <job_id>
uv run lab logs <job_id>
uv run lab metrics <job_id> --since-step 7   # live incremental series (early-kill loop)
uv run lab fetch <job_id>
uv run lab sweep -c "python experiments/example_capacity.py" -g "seed=1,2,3"   # grid → job-per-point
uv run lab wait --sweep <sweep_id>           # block until done; run in background → push-notify (FR-G1)
uv run lab dashboard                         # live terminal dashboard: status + cost + metrics (FR-D3)

# MCP server (stdio) — register this command in your MCP client config
uv run python -m lab.mcp_server

# Remote backend (Vast.ai via SkyPilot): uv sync --extra skypilot, set a Vast API key, then:
uv run lab submit -c "python experiments/example_capacity.py" \
  --backend skypilot --accelerators RTX4090:1 --timeout 20m

# Durable artifacts on Cloudflare R2 (optional): uv sync --extra r2, creds in
# ~/.cloudflare/r2.credentials, then export before submitting/fetching:
export LAB_R2_ENDPOINT="https://<account>.r2.cloudflarestorage.com"
export LAB_R2_BUCKET="lab-artifacts"
```

## Layout

```
src/lab/            # the lab package (core + backends + interfaces)
experiments/        # experiment entrypoints (Experiment Contract §7)
runs/               # fetched artifacts + manifests (git-ignored)
research/           # research notes backing the spec
LAB-REQUIREMENTS.md # the spec
```
