# Laboratory — Remote Experiment Runner

Run computational experiments on remote machines, decoupled from the local session: submit heavy
jobs and keep working, watch metrics live and kill early if off-track, and get results back
**reproducibly**. Experiment-agnostic core — any script honoring the Experiment Contract runs
unchanged.

- **Getting started:** [`docs/guides/getting-started.md`](docs/guides/getting-started.md)
- **What a release freezes:** [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) ·
  **what changed:** [`CHANGELOG.md`](CHANGELOG.md)
- **Spec:** [`LAB-REQUIREMENTS.md`](LAB-REQUIREMENTS.md) (RFC-2119, phased P0/P1/P2)
- **Research & design decisions:** [`research/`](research/) — start at `research/README.md`,
  decisions in `research/16-decisions.md`, architecture in `research/10-architecture.md`.

## Install it into your project

The lab is released as tagged versions you install as a dependency. Your repo holds the
experiment code, its git history is the provenance every manifest pins, and results land under
your `runs/`.

```bash
# --python 3.12 matters: bare `uv init` writes requires-python = ">=3.11" and the add then fails.
uv init --python 3.12 my-experiments && cd my-experiments && git init
uv add "laboratory[skypilot,gcp,r2] @ git+https://github.com/spicysauce1955-stack/laboratory@v0.5.0"
uv run lab init          # .mcp.json, the laboratory skill, .env.example, ignores, an example entrypoint

uv run lab submit -c "python experiments/example.py steps=5" --seed 0
uv run lab wait <job_id>
```

Full walkthrough — credentials, remote backends, writing an entrypoint, upgrading — in
[`docs/guides/getting-started.md`](docs/guides/getting-started.md).

## Developing the lab itself

Working *on* the lab, rather than using it:

```bash
uv sync                                # local backend + CLI/MCP
uv sync --extra skypilot --extra r2    # + remote (Vast) backend & durable R2
uv sync --extra do --extra gcp         # + DigitalOcean / Google Cloud (--cloud, --backend cpu)

uv run ruff check && uv run mypy --strict && uv run pytest -q
uv run pytest -m packaging             # the installed-wheel guard (excluded from the default run)
scripts/release.sh v0.5.1              # cut a release (CI publishes on the tag)
```

Note several tests `import sky` at module scope, so the cloud extras are required for a full run.

### Command reference

```bash
uv run lab submit -c "python experiments/example_capacity.py" --seed 42
uv run lab submit -c "python experiments/x.py" --with scipy --with scikit-learn   # per-job runtime deps
uv run lab list
uv run lab status <job_id>
uv run lab logs <job_id>
uv run lab metrics <job_id> --since-step 7   # live incremental series (early-kill loop)
uv run lab fetch <job_id>
uv run lab sweep -c "python experiments/example_capacity.py" -g "seed=1,2,3"   # grid → job-per-point
uv run lab wait --sweep <sweep_id>           # block until done; run in background → push-notify (FR-G1)
uv run lab dashboard                         # live terminal dashboard: status + cost + metrics (FR-D3)

# MCP server (stdio) — `lab init` writes this into a project's .mcp.json for you
uv run lab mcp

# Remote backend (Vast.ai via SkyPilot): uv sync --extra skypilot, set a Vast API key, then:
uv run lab submit -c "python experiments/example_capacity.py" \
  --backend skypilot --accelerators RTX4090:1 --timeout 20m

# Other clouds via --cloud (vast default | do | gcp; docs/guides/gcp-backend.md):
uv run lab submit -c "python experiments/x.py" \
  --backend skypilot --cloud gcp --accelerators T4:1 --timeout 20m

# Durable artifacts on Cloudflare R2 (optional): uv sync --extra r2, creds in
# ~/.cloudflare/r2.credentials, then export before submitting/fetching:
export LAB_R2_ENDPOINT="https://<account>.r2.cloudflarestorage.com"
export LAB_R2_BUCKET="lab-artifacts"
```

## Layout

```
src/lab/              # the lab package (core + backends + interfaces)
src/lab/_scaffold/    # what `lab init` writes into a project — ships inside the wheel
experiments/          # the lab's own example/fixture entrypoints (Experiment Contract §7)
runs/                 # fetched artifacts + manifests (git-ignored)
research/             # research notes backing the spec
scripts/release.sh    # cut a release
LAB-REQUIREMENTS.md   # the spec
```

The `laboratory` skill's source of truth is `src/lab/_scaffold/project/skills/laboratory/` — it is
packaged and installed into projects by `lab init`, not read from this repo's `.claude/`.
