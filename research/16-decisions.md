# 16 — Recommendations for the Open Decisions (spec §13)

A concrete call on each open decision, plus a P0 build order. These are recommendations to
confirm, not final — each links to the file with the full rationale.

## §13.1 Provisioner → **SkyPilot + a `local` backend** (see `11`)
SkyPilot's async SDK, managed jobs, autostop/`down:true`, managed spot, and ability to run an
**arbitrary `uv run` entrypoint unchanged** match the FRs and the Experiment Contract directly.
Add a **`local` subprocess backend** behind the same interface for NFR-4 (fallback + dev loop).
Keep **Modal** documented as a P1/P2 alternative backend (great DX, but image/SDK lock-in).

## §13.2 Tracker / live metrics → **MLflow self-hosted** (see `12`)
`MlflowClient.get_metric_history(run_id, key)` returns the full live series — the exact fit for the
`metrics` tool and the early-kill loop (FR-D2). Already in our stack; self-hostable; token-cheap.
**Re-evaluate W&B in P1** if we want a zero-ops live dashboard (FR-D3).

## §13.3 Completion signaling → **poll in P0, push sentinel in P1** (see `13`)
`status` is cheap (FR-G2) → poll on a run-length-matched interval for the MVP. In P1, emit a
terminal **sentinel** that trips a Claude Code background-task notification so the agent stops
polling. Keep the mechanism outside the experiment.

## §13.4 Artifact storage → **object store (R2/S3) canonical + pull to `runs/`** (see `15`)
Remote VMs are torn down, so artifacts must be pushed off-box before teardown; **Cloudflare R2 /
B2 (no egress)** is the durable home. `fetch_artifacts` pulls into `runs/<job_id>/`. rsync is the
fast path for the `local` backend only.

## §13.5 Compute shape → **single fat node first; spot job-per-point in P1** (see `15`)
P0: one multicore node (local backend on a cheap Hetzner dedicated box or this machine), sweep as
parallel processes. P1: SkyPilot managed-**spot** fan-out, one job per K/α under a `sweep_id`.

## §13.6 Auth/secrets → **creds in control-plane env only; URIs (not keys) in manifests** (see `15`)
Cloud creds in the SkyPilot API-server env/`~/.sky`; tracker/storage keys injected as SkyPilot
secrets at launch; MCP server reads a gitignored `.env`. Nothing secret in repo/manifest/logs
(FR-J1). Ephemeral per-job isolation covers FR-J2/J3.

---

## Suggested P0 build order (maps to roadmap §12 "MVP")

1. **Lab core + backend interface** — `submit/status/tail_logs/cancel/collect_artifacts` Protocol;
   implement the **`local`** backend first (fastest to a working loop, satisfies NFR-4).
2. **Manifest writer** — commit pin (+ dirty policy), `uv.lock` hash, resolved config + seed,
   timestamps; write `runs/<job_id>/manifest.json` (FR-B).
3. **`skypilot` backend** — async `sky.jobs.launch` with `setup: uv sync --frozen`,
   `run: uv run python …`, `autostop {idle_minutes, down:true}`, wall-clock timeout (FR-C/I).
4. **MCP server (FastMCP)** — typed tools `submit/status/logs/fetch_artifacts/cancel/list` with
   structured outputs + fail-loud errors (FR-F); **CLI** mirror over the same core.
5. **Artifacts** — push `$LAB_RUN_DIR` → R2 on completion; `fetch_artifacts` → `runs/` (FR-E).
6. **Acceptance pass** — verify AC-1/2/5/6/7 on a seeded tempotron run (local + one SkyPilot CPU job).

**Then P1:** live `metrics` tool (MLflow `get_metric_history`) + early-kill, push notify, `sweep`
fan-out on spot, GPU resource requests, result caching, dashboard.

## Things explicitly de-scoped from the research (per spec §14 non-goals)
- General DAG orchestration (Airflow/Dagster), data lake/warehouse, model serving.
- **Tabular-GPU acceleration (RAPIDS/cuML/cudf.pandas)** — irrelevant to a CPU spiking-neuron
  simulation; `06` is trimmed accordingly. GPU support is a generic resource request (P1), not a
  RAPIDS adoption.
- Heavyweight HPO frameworks as a core dependency — the lab's `sweep` is a parameter **grid**
  (job-per-point), generatable with Hydra multirun; Optuna stays optional for true HPO (`05`).
