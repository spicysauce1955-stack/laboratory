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
  finds orphaned rentals/instances not tied to a running job and destroys them. `--apply` lists
  what it will destroy and **asks**; add `--yes` for unattended use (with no tty it refuses and
  exits 4 rather than prompting). Only the approved set is destroyed. The GCP passes claim only
  SkyPilot's real node shape `lab-…-<head|worker>-<uuid8>-<compute|tpu|mig>` — anything else
  named `lab-*` is listed under `gcp_unmatched`, warned about on stderr, and never destroyed.
- **Deferred scheduling:** `lab register` + `lab queue …` queue jobs (night window / price /
  dependency triggers); an always-on host runs `lab scheduler tick` every 60s (systemd timer,
  `deploy/scheduler/`). Spec: `docs/superpowers/specs/2026-06-10-deferred-scheduling-design.md`.
- **Fail-closed provenance (FR-B1):** `JobStore.create` rejects any manifest whose `code` can't
  reproduce the run (null SHA, or `git_dirty` without a `diff_ref`) — enforced on create only, so
  legacy manifests still read. A dirty `submit` auto-snapshots the diff+untracked into
  `code_diff.tar.gz` (`capture_diff`/`apply_diff`), mirrored to R2; `--no-dirty`/`allow_dirty=false`
  refuses instead. Deferred paths set `diff_ref` to the bundle key. Timeout `end_reason` carries the
  wall ("timed out after Ns wall-clock cap"). Guide: `docs/guides/provenance-and-timeouts.md`.
- **CPU backend (FR P1-1):** `lab submit --backend cpu` provisions a cheap multi-core **DigitalOcean**
  droplet (default **4 vCPU + 50 GB volume**, up to 48; on-demand) via SkyPilot — sugar over skypilot +
  `cloud="do"`, resolved in `resolve_backend_profile`. Defaults stay inside a **fresh DO account tier**:
  8-vCPU sizes AND SkyPilot's default 256 GB volume both `422` on an untouched account (size
  restricted / "invalid size specified") — bigger needs a DO tier-increase ticket. `disk_size` lives on
  `ResourceRequest` → `sky.Resources`. The cloud is selectable via `--cloud vast|do|gcp` on
  submit/sweep/register (validated in `validate_cloud`); `--backend cpu --cloud gcp` runs the cpu
  profile on GCP (spot allowed there — only DO forces spot off). `lab reconcile` runs a
  cloud-agnostic `sky.status` orphan pass (Vast-direct pass skipped without vastai-sdk), a GCP
  compute-API pass (`lab-*` instances + unattached `lab-*` disks), and flags `unsupervised`
  running jobs whose supervisor pid is dead; DO volumes remain uncovered. `robust_teardown` has a
  gcp-direct fallback mirroring the vastai one. Price/offer triggers
  (`--max-hourly`/`--offer-query`) are Vast-only and rejected for other clouds. Guides:
  `docs/guides/cpu-backend.md`, `docs/guides/gcp-backend.md`.
- **Placement & pricing (`lab.placement`):** the only module that talks to `sky.catalog` (a local
  CSV; no credentials, no cloud calls). It resolves the instance type a spec lands on, prices every
  region, and remembers zones that just returned `ZONE_RESOURCE_POOL_EXHAUSTED` (30 min TTL,
  advisory — a broken memo is ignored, never fatal) so a sweep's later shards skip them.
  `--region`/`--zone` pin; `--price-cap` maps to `sky.Resources(max_hourly_cost=)`, a ceiling the
  optimizer enforces. **Estimates are bands and guardrails check the top** — `get_cost()` on an
  unpinned `Resources` returns the cheapest region's price, which made admission control
  systematically permissive; the ceiling is priced on-demand whenever `spot_fallback` could land
  there. `CostInfo.hourly_usd` is **compute + storage**; no gcp/do job may inherit SkyPilot's
  256 GB disk (50 GB cpu, 100 GB GPU — `placement.effective_disk_gb`, applied in **`build_task`**
  because the scheduler launches registrations without calling `resolve_backend_profile`).
  Provision timeouts are per-cloud (gcp 20m). Diagnostics from these paths go to **stderr**:
  stdout carries only JSON, which callers parse.
- **Preflight (`lab doctor`, `lab.doctor`):** checks credentials (incl. SkyPilot's daemon, which
  does not inherit `.env`), project, billing, APIs, IAM permissions and quota before a launch
  costs a provision; the cheap subset runs automatically on submit (`--no-preflight` opts out).
  **Only definitive negatives block** — a check that cannot answer is `skip` and never blocks.
  GCP GPU quota is checked at both levels Google enforces: a project can hold regional
  `NVIDIA_*_GPUS` and still be blocked by a global `GPUS_ALL_REGIONS` of 0. `PREEMPTIBLE_CPUS=0`
  is *not* a blocker (spot falls back to standard `CPUS` quota).
- **Sharded sweeps (FR P1-2):** `lab sweep --seeds 0-31 --shard-size 8` splits each grid cell's
  seeds into independently-bounded shard jobs (own timeout + teardown), then `lab sweep-aggregate`
  row-concatenates the succeeded shards into one per-cell `results.csv` (seed column overridable),
  reporting `seeds_present` vs expected and naming missing seeds on partial failure;
  `lab sweep-retry` resubmits only the missing shards. A `SweepPlan` under the `sweep_id` is the
  cell→shards map. Guide: `docs/guides/sharded-sweeps.md`.

- **Agent-UX hardening (field report 2026-08-05):** entrypoints report consumed config via
  `$LAB_RUN_DIR/effective_config.json` (helper: `lab.experiment.get_overrides`); a succeeded job
  with argv overrides it never consumed **flips to failed** (`--allow-unknown-config` opts out;
  `lab lint` pre-checks legacy scripts). `sweep-aggregate` includes partial rows from terminal
  non-succeeded shards by default (`_shard_status` column, `seeds_partial` in the view,
  `--strict` opts out) and `sweep-retry` resubmits only missing seeds. `lab wait` gains
  `--fail-fast` (exit 4), an incrementally-rewritten `--done-file` (with `pending`), and
  duration-string `--timeout`. Transient local-API launch errors retry with backoff
  (`LAB_LAUNCH_RETRIES`, `end_reason` prefix `transient:`); remote sweep submits stagger
  (`LAB_SUBMIT_STAGGER_S`, default 1.5s). `lab export <job|sweep> --to DIR` writes the
  committable provenance bundle (manifests + tables + diffs + index.json). `lab status` shows
  `estimated_running_usd` + `last_log_line`. Grid is optional when `--seeds` is given.

## Conventions
- `ruff` (line length 100), `mypy --strict` on `src/lab`. CLI and MCP server are thin shells over
  the `lab.core.Lab` library — never duplicate logic between them.
- **Secrets** never in repo/manifest/logs (FR-J1); manifests record URIs, not keys. Machine-local
  settings (GCP project + service-account key **path**, R2 endpoint/bucket) go in a git-ignored
  `.env` at the repo root, loaded by `lab.env.load_lab_env` at CLI/MCP startup; the committed
  `.env.example` is placeholders-only. Real env wins over the file; blank means unset.
