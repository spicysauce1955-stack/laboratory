# Laboratory — Delivery Document

**To:** the researcher who authored `LAB-REQUIREMENTS.md`
**Re:** the Remote Experiment Runner is built and ready to use for the tempotron-capacity work (and anything else).
**Status:** P0 (MVP) + the P1 roadmap (§12) **complete and validated** — including a real end-to-end run on a Vast.ai GPU. Two individually-P1-tagged FRs and the P2 items are not built yet (see §8 below — stated honestly).

---

## 1. What you got, in one paragraph

`lab` turns *"run this experiment"* into a **reproducible remote job** with a **structured agent-usable (MCP) interface**, a **human CLI**, **live observability**, and **cost-bounded auto-teardown** — exactly the spec's purpose (§1). It runs your experiment either **locally** (no credentials) or on a **remote GPU** (Vast.ai via SkyPilot), captures a **per-job manifest** that regenerates the run (commit + `uv.lock` + config + seed), streams **live metrics** so you can kill early, fetches artifacts back into `runs/<job_id>/` with **durable copies on Cloudflare R2**, reports **cost**, supports **parameter sweeps** and **result caching**, and a **live dashboard**. Both you-the-human (CLI) and you-the-agent (MCP) drive the same core.

---

## 2. Requirements traceability (§5/§6/§11)

Honest accounting against the spec. **Met** = built + tested; **Partial** = built with a stated simplification; **Not yet** = deferred (P2 or a P1 SHOULD not built).

| Req | Status | How / where |
|---|---|---|
| FR-A1 non-blocking submit | ✅ Met | detached supervisor; `submit` returns a `job_id` in <~1s |
| FR-A2 observable states | ✅ Met | `queued→running→{succeeded,failed,cancelled,timed_out}` in the manifest |
| FR-A3 cancel | ✅ Met | `lab cancel` — kills the process group / `sky cancel`+`down` |
| FR-A4 exit code + end reason | ✅ Met | `manifest.exit_code`, `manifest.end_reason` |
| FR-A5 sweep | ✅ Met | `lab sweep --grid k=v1,v2` → job-per-point under one `sweep_id` |
| FR-B1 pinned commit + dirty policy | ⚠️ Partial | records `git_commit` + `git_dirty`; **runs the working tree** (not a checkout) and the dirty-diff *snapshot* (`diff_ref`) is **not** implemented; `code_ref` is recorded but `submit` always pins HEAD |
| FR-B2 env from `uv.lock` | ✅ Met | remote runs `uv sync --frozen`; `manifest.env.uv_lock_sha256` recorded |
| FR-B3 manifest | ✅ Met | `runs/<job_id>/manifest.json` (see §6) |
| FR-B4 explicit seed, reproducible | ✅ Met | `manifest.run.seed`; same manifest → same numbers (deterministic experiments) |
| FR-B5 result caching | ✅ Met | `lab submit --cache` (commit+command+config+seed, clean-tree gated) |
| FR-C1 provision+env+run+capture | ✅ Met | `skypilot` backend on Vast |
| FR-C2 auto-teardown | ✅ Met | `autostop down=true` + explicit `sky.down`; **verified no orphans** |
| FR-C3 GPU / multi-node | ✅ GPU / ❌ multi-node (P2) | GPU via `--accelerators`; multi-node is P2 |
| FR-C4 env to the job | ✅ Met | `$LAB_RUN_ID`, `$LAB_RUN_DIR`, `$LAB_SEED` |
| FR-D1 logs | ✅ Met | `lab logs` |
| FR-D2 live incremental metrics | ✅ Met | `lab metrics --since-step N` (the early-kill loop) |
| FR-D3 dashboard | ✅ Met | `lab dashboard` (live terminal: status + cost + metrics) |
| FR-E1/E2/E3 artifacts | ✅ Met | collected to `runs/<job_id>/output/`; each has a `sha256`+`bytes`; durable on R2 |
| FR-F1 MCP / F2 CLI / F3 fail-loud | ✅ Met | FastMCP server (structured JSON) + Typer CLI; errors are actionable |
| FR-G1 push / G2 cheap poll | ✅ Met | `lab wait` as a background task = push; `status` is a cheap (no-cloud) poll |
| FR-H1 concurrent + list | ✅ Met | many jobs; `lab list` |
| FR-H2 queue under limited capacity + position | ❌ Not yet (P1 SHOULD) | jobs launch immediately; no global capacity queue (sweep has a `max_jobs` cap only) |
| FR-I1 timeout + teardown | ✅ Met | `--timeout`; remote wraps in `timeout`, supervisor enforces |
| FR-I2 cost/compute (estimated+actual) | ✅ Met | `manifest.cost` (estimate at launch, actual at end); shown in `status`/dashboard |
| FR-I3 spot/preemptible + auto-resume | ❌ Not yet | we use on-demand cluster jobs + autostop; SkyPilot managed-spot not wired (§12 lists this under P2) |
| FR-J1/J2/J3 secrets / least-priv / isolation | ✅ Met | creds live outside the repo; manifests store **URIs, not keys**; ephemeral per-job instances |
| AC-1..AC-7 (§11) | ✅ Met | reproduce-in-`runs/`, <5s submit, live partial metrics, terminal-learned+fetch, cancel/timeout teardown, regenerable manifest, no leaked secrets — all exercised |
| NFR-1..7 | ✅ Met (NFR-5 partial) | reproducibility, survive-disconnect, low-token agent calls, local fallback, auditability, cost-safety; NFR-5 scale is single-node+GPU (multi-node is P2) |

**Bottom line:** everything you marked **P0**, plus the **P1 roadmap in §12**, works. The gaps to know about are **FR-B1 dirty-snapshot / `code_ref` pinning**, **FR-H2 capacity queueing**, **FR-I3 managed-spot**, and **multi-node** — all noted in §8.

---

## 3. Setup (one time)

```bash
# Local only (CLI + local backend; no credentials needed):
uv sync

# + remote (Vast) backend and durable R2 artifacts (full setup):
uv sync --extra skypilot --extra r2
```

**Remote on Vast.ai** (only for `--backend skypilot`):
- Put your Vast API key at `~/.config/vastai/vast_api_key` (one line). No SSH key needed — SkyPilot manages its own.
- Verify: `uv run sky check vast` → should print `Vast: enabled`.

**Durable artifacts on Cloudflare R2** (optional; remote artifacts survive teardown):
- Put R2 **S3 credentials** (Access Key ID + Secret) at `~/.cloudflare/r2.credentials` (AWS-format `[default]` block).
- `export LAB_R2_ENDPOINT="https://<account>.r2.cloudflarestorage.com"` and `export LAB_R2_BUCKET="lab-artifacts"` before submitting/fetching. Without these, the lab falls back to local-only artifacts.

> 🔐 The Vast key, Cloudflare token, and R2 secret were pasted in chat during development — **please rotate them** (ideally a scoped R2 token).

---

## 4. Make your experiment lab-compatible (the Experiment Contract, §7)

This is the only thing **you** have to do to run the tempotron-capacity work. A script is "lab-compatible" if it obeys this thin contract — **no lab imports required**:

1. It's a committed, runnable entrypoint (e.g. `python experiments/capacity.py`) fully determined by **config + seed**.
2. It reads the seed and output dir the lab injects via env:
   - `$LAB_SEED` — the seed (also passed as a `seed=...` override if the seed is in a sweep grid).
   - `$LAB_RUN_DIR` — write **all** outputs here.
   - `$LAB_RUN_ID` — the job id, if you want it.
   - Sweep parameters arrive **appended to the command** as shell-quoted `key=value` tokens (Hydra-style), e.g. `python capacity.py K=30 alpha=0.1`.
3. It logs **incremental metrics** so progress is observable live — write JSON lines to `$LAB_RUN_DIR/metrics.jsonl`:
   ```json
   {"name": "capacity", "value": 0.83, "step": 12, "wall_time": 1779880000.1}
   ```
   (Optional convenience: `from lab.metrics import log_metric; log_metric("capacity", 0.83, step=12)` — it writes the same file. Using it is fine; it only needs stdlib.)
4. It writes outputs (figures `.png/.pdf`, tables `.csv/.json/.parquet`, checkpoints) under `$LAB_RUN_DIR`.
5. It **exits non-zero on failure** (fail-loud).

**Minimal tempotron-shaped skeleton:**

```python
import json, os, sys

def main() -> int:
    run_dir = os.environ["LAB_RUN_DIR"]
    seed = int(os.environ.get("LAB_SEED", "0"))
    # sweep overrides arrive as key=value argv tokens:
    overrides = dict(tok.split("=", 1) for tok in sys.argv[1:] if "=" in tok)
    K = int(overrides.get("K", 100))
    alpha = float(overrides.get("alpha", 0.1))

    # ... run the capacity experiment for (seed, K, alpha) ...
    with open(f"{run_dir}/metrics.jsonl", "w") as f:
        for step, frac in enumerate(train_curve):           # your loop
            f.write(json.dumps({"name": "frac_correct", "value": frac, "step": step}) + "\n")
    with open(f"{run_dir}/result.json", "w") as f:
        json.dump({"seed": seed, "K": K, "alpha": alpha, "capacity": capacity}, f)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

**Extra runtime deps per job.** The remote env is intentionally lean (numpy + pydantic + hydra-core). If your experiment needs more — e.g. `scipy` for LP feasibility — pass `--with PKG` to `lab submit`/`sweep` (repeatable). The lab layers them via `uv run --with`, so no project changes are required:

```bash
uv run lab submit -c "python experiments/capacity.py" --with scipy --with scikit-learn
```

The same works if you'd rather hand-write it in the command (`uv run --with scipy python …`). Validated end-to-end on the local backend; the remote path uses the identical wrapper and Vast instances have PyPI access.

See `experiments/example_capacity.py` for a runnable reference. (If you prefer Hydra+Pydantic configs, the `key=value` overrides are Hydra-native — just consume them through Hydra.)

---

## 5. Using it

### CLI (you, the human)

```bash
# Submit (returns immediately with a job_id):
uv run lab submit -c "python experiments/capacity.py" --seed 7 --timeout 2h
#   --backend skypilot --accelerators RTX4090:1   → run on Vast (GPU required there)
#   --cache                                        → reuse a prior identical succeeded job
#   --with scipy --with scikit-learn               → extra runtime deps for this job
#   --cpus / --memory / --gpus / --code-ref        → resources / pinning

uv run lab list                       # all jobs (job_id, sweep_id, status)
uv run lab status <job_id>            # state + cost (duration / hourly / estimated / actual)
uv run lab logs <job_id> --tail 100   # stdout+stderr
uv run lab metrics <job_id> --since-step 20   # only what's new — the "are we on track?" query
uv run lab cancel <job_id>            # stop + tear down
uv run lab fetch <job_id>             # pull artifacts into runs/<job_id>/ (from R2 if remote)

# Sweep a grid → one job per point under a shared sweep_id:
uv run lab sweep -c "python experiments/capacity.py" -g "K=50,100,200" -g "seed=1,2,3"
#   (a `seed=...` grid key sets each job's seed)

# Block until done, then exit — run in the background so its completion notifies you:
uv run lab wait --sweep <sweep_id> --done-file runs/<sweep>.done
#   Exit code: 0 = all terminal, 1 = timed out before all done. If invoked via a wrapper that
#   may mask the exit code, parse the JSON / --done-file — `"all_terminal"` is the truth.

# Live dashboard (status + cost + latest metrics), Ctrl-C to exit:
uv run lab dashboard           # or  --sweep <sweep_id>
```

### MCP (you, the agent)

Register the server with your MCP client (stdio):

```bash
uv run python -m lab.mcp_server
```

Tools (structured JSON, fail-loud): **`submit`, `sweep`, `status`, `logs`, `metrics`, `fetch_artifacts`, `cancel`, `list`** (same params as the CLI; `submit`/`sweep` take `backend`, `cache`, `accelerators`, …).

> The agent's **push** path (FR-G1) is the **CLI `lab wait` run as a background task** — when it exits, your session is notified. There is intentionally **no blocking `wait`/`dashboard` MCP tool** (the §9 contract lists none, and a long-blocking tool would defeat the non-blocking design). For polling, the MCP `status` tool is cheap (no per-call cloud cost).

---

## 6. Reproducibility & the manifest (§8)

Every job writes `runs/<job_id>/manifest.json` — the contract that regenerates the run:

```jsonc
{
  "job_id": "...", "sweep_id": null, "created_at": "...", "submitted_by": "human|agent",
  "code":  { "git_commit": "…", "git_dirty": false, "diff_ref": null },
  "env":   { "uv_lock_sha256": "…", "python_version": "3.12.3" },
  "run":   { "entrypoint_command": "python …", "resolved_config": { "K": 30 }, "seed": 7 },
  "resources": { "cpus": null, "gpus": null, "memory": null, "accelerators": "RTX4090:1", "timeout": "2h" },
  "backend": { "provisioner": "skypilot", "machine_type": "…", "region": "…" },
  "status": "succeeded", "started_at": "…", "ended_at": "…", "exit_code": 0, "end_reason": "succeeded",
  "cost": { "duration_seconds": 152.3, "hourly_usd": 0.40, "estimated_usd": 0.80, "actual_usd": 0.017 },
  "metrics_uri": null, "logs_uri": null, "artifacts_uri": "r2://lab-artifacts/<job_id>",
  "artifacts": [ { "name": "result.json", "type": "table", "path": "…", "sha256": "…", "bytes": 25 } ]
}
```

To **reproduce**: re-submit the same command/seed at the same (clean) commit — or use `--cache` to reuse the existing result instead of recomputing (FR-B5). Layout: `runs/<job_id>/{manifest.json, logs.txt, output/}` (`runs/` is git-ignored).

---

## 7. Backends — which to use

- **`local`** (default): runs as a subprocess on this machine. No credentials, fastest iteration, your NFR-4 fallback. Great for development and CPU-bound capacity sweeps on a fat node.
- **`skypilot`** (`--backend skypilot`): provisions a Vast.ai instance → `uv sync` the locked env → run → fetch → durable R2 → **auto-teardown**. Vast is GPU-only, so pass `--accelerators` (e.g. `RTX4090:1`). Cost is reported per job. Provisioning installs only experiment-runtime deps (~11 packages), not the lab control plane.

> Operational note: Vast is a marketplace — an occasional host won't accept SSH or an offer vanishes; the backend fails fast and a resubmit lands a healthy host. Validation runs cost ~$0.02–0.07 each.

---

## 8. Known limitations / not-yet-built (be aware)

- **FR-B1 (partial):** jobs run the **working tree**, recording `git_commit` + `git_dirty`. The dirty-tree **diff snapshot** (`diff_ref`) and **honoring a non-HEAD `--code-ref`** are not implemented (we always pin HEAD). For strict reproducibility, **commit before submitting** (caching also requires a clean tree).
- **FR-H2 (not yet):** no capacity-based **queue / queue position** — jobs launch immediately (sweeps have a `max_jobs` safety cap only).
- **FR-I3 (not yet):** managed **spot/preemptible with auto-resume** — we use on-demand cluster jobs + autostop. (§12 places spot under P2.)
- **Multi-node / distributed (P2):** single-node only.
- **Tracker:** live metrics use a `metrics.jsonl` convention (no MLflow server wired); `mlflow` is available as an opt-in `tracking` extra if you later want its UI.
- **Non-goals (unchanged, §13):** not a DAG orchestrator, data lake, or model-serving system.

---

## 9. Where things live

```
LAB-REQUIREMENTS.md   your spec (source of truth)
DELIVERY.md           this document
README.md             quickstart
CLAUDE.md             notes for AI sessions
src/lab/              core: models, store, runner, sky_runner, backends/{local,skypilot},
                      core (Lab), cli, mcp_server, metrics, storage (R2), dashboard
experiments/          experiment entrypoints (example_capacity.py)
runs/                 per-job manifests + artifacts (git-ignored)
research/             the design behind the build — start at research/README.md;
                      decisions in research/16-decisions.md, architecture in research/10-architecture.md
tests/                43 tests (pytest); run: `uv run pytest`
```

Quality bar: **43 tests passing, `ruff` clean**, each feature reviewed (architect-reviewer / code-review) before merge. The git history is one commit per feature with the FR it satisfies.

---

## 10. Suggested next steps (your call)

If you want the lab to go further than the delivered scope: **(a)** strict commit pinning + dirty-diff snapshot (closes FR-B1); **(b)** capacity queue with positions (FR-H2); **(c)** managed-spot + auto-resume for cheaper long sweeps (FR-I3); **(d)** multi-node (FR-C3/P2). None are needed to start the tempotron-capacity work today.
