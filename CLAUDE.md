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
- **Teardown is cost-critical (FR-C2):** every skypilot job runs through `robust_teardown`
  (sky.down retries → vastai-sdk fallback). A persistent failure flips `teardown_status="failed"`
  on the manifest and makes `lab wait` exit 3. **Recovery: `uv run lab reconcile [--apply]`**
  finds orphaned `lab-*` Vast rentals not tied to a running job and destroys them.
- **Deferred scheduling:** `lab register` + `lab queue …` queue jobs (night window / price /
  dependency triggers); an always-on host runs `lab scheduler tick` every 60s (systemd timer,
  `deploy/scheduler/`). Spec: `docs/superpowers/specs/2026-06-10-deferred-scheduling-design.md`.
- **Fail-closed provenance (FR-B1):** `JobStore.create` rejects any manifest whose `code` can't
  reproduce the run (null SHA, or `git_dirty` without a `diff_ref`) — enforced on create only, so
  legacy manifests still read. A dirty `submit` auto-snapshots the diff+untracked into
  `code_diff.tar.gz` (`capture_diff`/`apply_diff`), mirrored to R2; `--no-dirty`/`allow_dirty=false`
  refuses instead. Deferred paths set `diff_ref` to the bundle key. Timeout `end_reason` carries the
  wall ("timed out after Ns wall-clock cap"). Guide: `docs/guides/provenance-and-timeouts.md`.

## Conventions
- `ruff` (line length 100), `mypy --strict` on `src/lab`. CLI and MCP server are thin shells over
  the `lab.core.Lab` library — never duplicate logic between them.
- **Secrets** never in repo/manifest/logs (FR-J1); manifests record URIs, not keys.
