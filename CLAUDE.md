# CLAUDE.md — Laboratory

A **Remote Experiment Runner**: turns "run this experiment" into a reproducible remote job with an
agent-usable **MCP** interface + a CLI, live observability, and cost-bounded auto-teardown.

## Read first
- `LAB-REQUIREMENTS.md` — the spec (RFC-2119, phased). The source of truth.
- `research/16-decisions.md` — chosen design + **P0 build order**.
- `research/10-architecture.md` — architecture + FR→component traceability.

## Key facts
- **Env is fixed:** Python via **uv** (`uv.lock` committed; **NumPy `<2`** pin), config via
  **Hydra+Pydantic**, metrics via **MLflow**, outputs to git-ignored `runs/`.
- **First workload:** tempotron-capacity — CPU-bound, embarrassingly parallel (seeds/α/K). GPU is P1.
- **Chosen stack (to confirm):** provisioner = **SkyPilot** + a **`local`** subprocess backend
  (NFR-4); tracker = **MLflow** self-hosted (`get_metric_history` = live series); interface =
  **FastMCP**; artifacts = object store (Cloudflare R2/S3) → `runs/<job_id>/`.
- **Experiment Contract (§7):** any committed `uv run` entrypoint, determined by config+seed,
  writes to `$LAB_RUN_DIR`, logs metrics via `log_metric(name, value, step)`, exits non-zero on fail.

## Conventions
- `ruff` (line length 100), `mypy --strict` on `src/lab`. CLI and MCP server are thin shells over
  the `lab.core.Lab` library — never duplicate logic between them.
- **Secrets** never in repo/manifest/logs (FR-J1); manifests record URIs, not keys.
