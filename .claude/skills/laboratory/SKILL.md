---
name: laboratory
description: "Run reproducible ML experiments on local or remote (Vast.ai via SkyPilot) machines via the lab MCP/CLI. Use when the user asks to: run/submit an experiment, sweep over K/alpha/seeds, watch metrics live and kill early if off-track, fetch artifacts, reproduce a prior run, or any tempotron-capacity / GPU-on-Vast / cost-bounded compute task. Triggers: lab submit, lab sweep, lab wait, run experiment remote, sweep parameters, watch metrics live, fetch artifacts, kill early if off-track, reproduce job."
metadata:
  version: "0.1.0"
  last_updated: "2026-05-28"
  status: active
---

# Laboratory — Remote Experiment Runner

This skill teaches you to drive the **lab** — an experiment-agnostic remote job
runner — from inside the `laboratory` repo. The lab handles: submitting a job
without blocking, watching its metrics live, sweeping a grid, fetching durable
artifacts, and pinning everything to a reproducible manifest (commit + uv.lock
+ config + seed).

You will normally use the **MCP tools** (`mcp__lab__*`) registered by the repo's
`.mcp.json`. For push-notify (block-until-done as a background task), use the
**CLI** `uv run lab wait` — there is no MCP `wait` tool by design.

## 1. When to use this skill

Invoke this skill when the user asks (in any phrasing):

- "Run / submit this experiment" (especially with a seed, grid, or GPU).
- "Sweep over K / alpha / seeds" (parameter grid).
- "Watch the run live and stop it if it's off-track" (live early-kill).
- "Fetch the results / artifacts of job `<id>`."
- "Reproduce job `<id>` / re-run with the same config."
- Anything that wants a remote GPU on Vast.ai, or a cost-bounded job, or a
  manifest-tracked run.

**Don't** invoke this skill for a one-off local sanity check that doesn't need
tracking — just running `uv run python experiments/foo.py` is fine.

## 2. Prerequisites (verify before first use)

Run from the lab repo root.

- **Sync deps.**
  - `uv sync` — local backend + CLI + MCP server (lean default).
  - `uv sync --extra skypilot --extra r2` — also enables remote (Vast.ai via
    SkyPilot) and durable artifacts on Cloudflare R2.
- **Remote creds (only for `--backend skypilot`).**
  - Vast API key in `~/.config/vastai/vast_api_key`.
  - R2 (optional, for durable artifacts):
    - Creds in `~/.cloudflare/r2.credentials` (S3-compat: Access Key + Secret).
    - Env: `LAB_R2_ENDPOINT="https://<account>.r2.cloudflarestorage.com"` and
      `LAB_R2_BUCKET="lab-artifacts"`.
- **MCP server.** Registered by `.mcp.json` at the repo root. Opening the repo
  in Claude Code should offer the `lab` server; once enabled, the tools below
  appear as `mcp__lab__submit`, etc.
- **Commit before submitting.** Manifests pin `HEAD`. The lab accepts a dirty
  tree (records `git_dirty: true`), but the cache (`cache=true`) will not hit
  on a dirty tree. Commit when you want reproducibility + cache reuse.

## 3. The Experiment Contract (spec §7)

If asked to *write* a new experiment, the entrypoint MUST:

| What the entrypoint does                          | How                                                |
|---------------------------------------------------|----------------------------------------------------|
| Read its run dir and seed from env                | `os.environ["LAB_RUN_DIR"]`, `LAB_RUN_ID`, `LAB_SEED` |
| Write outputs under `$LAB_RUN_DIR`                | All files (figures, checkpoints, tables) go here   |
| Log metrics incrementally                         | One JSON object per line into `$LAB_RUN_DIR/metrics.jsonl` (helper: `lab.metrics.log_metric(name, value, step)`) |
| Exit non-zero on failure                          | `sys.exit(1)` or raise                             |
| Accept grid overrides as `key=value` argv         | e.g. Hydra picks up `seed=3 K=200` from `sys.argv` |

A metric line is `{"name": "...", "value": <float>, "step": <int>, "wall_time": <float>}`.
The lab tolerates a half-written trailing line, so you can write line-by-line
and the metrics tool will still read cleanly.

**Reference template:** `experiments/example_capacity.py` (50 lines, contract-compliant).

## 4. The MCP tool surface

All registered by `build_server` in `src/lab/mcp_server.py`. Each returns a
JSON-serializable dict.

### `mcp__lab__submit`
Submit one job. Non-blocking — returns immediately with the `job_id`.

| Input            | Type           | Notes |
|------------------|----------------|-------|
| `command`        | str (required) | e.g. `"python experiments/example_capacity.py"` |
| `backend`        | str            | `"local"` (default) or `"skypilot"` |
| `cache`          | bool           | If true and a prior identical-`(commit, command, config, seed)` succeeded job exists on a clean tree, reuse it |
| `seed`           | int            | Recorded + injected as `$LAB_SEED` |
| `code_ref`       | str            | Git ref to pin; default `"HEAD"` |
| `cpus` / `memory`/ `gpus` | int/str/int | Resource hints (memory like `"8"` GB) |
| `accelerators`   | str            | **Required for skypilot.** e.g. `"RTX4090:1"` |
| `timeout`        | str            | Wall-clock cap; `"30m"`, `"2h"`, `"45s"` |
| `with_pkg`       | list[str]      | Per-job extra runtime deps (e.g. `["scipy", "scikit-learn>=1.4"]`) — layered via `uv run --with` |

Returns: `{"job_id": "...", "cached": bool, "status": "queued"|"succeeded"|...}`.

### `mcp__lab__sweep`
Submit a Cartesian-product grid of jobs under one `sweep_id`. Same kwargs as
`submit`, plus:

| Input | Type | Notes |
|-------|------|-------|
| `grid` | dict[str, list] | e.g. `{"seed": [1,2,3], "K": [100, 200, 500]}` → 9 jobs |

Grid values become `key=value` overrides on the experiment's argv (string-valued
— Hydra/typer coerce). If `seed` is a grid key, it sets each job's recorded seed.

Returns: `{"sweep_id": "...", "job_ids": [...]}`.

### `mcp__lab__status`
`{job_id}` → `{state, started_at, ended_at, exit_code, end_reason, cost}`.
States: `queued`, `running`, `succeeded`, `failed`, `cancelled`, `timed_out`.
Cheap to poll.

### `mcp__lab__metrics`
`{job_id, names?, since_step?}` → `{"series": {name: [{step, value, wall_time}, ...]}}`.
Incremental — pass `since_step=<last_step_seen>` to fetch only new points.
Designed for live polling at ~5–15s cadence (the early-kill loop).

### `mcp__lab__logs`
`{job_id, tail=100}` → `{"lines": [...]}`. The stdout/stderr of the job.

### `mcp__lab__fetch_artifacts`
`{job_id}` → `{"local_paths": [...], "artifacts": [...]}`. Pulls artifacts into
`runs/<job_id>/output/`. For skypilot jobs with R2 enabled, falls back to R2 if
the local output is empty (e.g. after a fresh clone).

### `mcp__lab__cancel`
`{job_id}` → `{"state": "cancelled"}`. Stops the job and tears the machine down.

### `mcp__lab__list`
`{}` → `{"jobs": [{job_id, sweep_id, status, created_at}, ...]}`. All jobs in
`runs/`.

## 5. The CLI surface (and the one CLI-only command)

Every MCP tool has a matching CLI command (`uv run lab submit / sweep / status
/ logs / metrics / fetch / cancel / list`). The `lab` CLI prints JSON
mirroring the MCP returns.

**`uv run lab wait` is CLI-only and is the push-notify primitive.**
Block until one or more jobs reach a terminal state:

```bash
uv run lab wait <job_id_1> <job_id_2> --done-file done.json
# or:
uv run lab wait --sweep <sweep_id> --done-file done.json
```

Why CLI-only: the right pattern is to run `lab wait` as a **Claude Code
background task**, keep working in the foreground, and let the task's
process-exit notify the harness — at which point you read `done.json` and
proceed. Exit code is non-zero on timeout (use that as a retry signal).

`uv run lab dashboard` (optional, interactive) — live terminal table of all
jobs with state, cost, and latest metric. Ctrl-C to exit.

## 6. Canonical workflows

Pick the one that matches the user's intent. Each has a copy-pasteable
walkthrough under `examples/`.

### A. Submit one job, keep working, get notified
See **`examples/01-submit-and-watch.md`**. Pattern: `mcp__lab__submit` →
background `uv run lab wait <id> --done-file done.json` → keep working →
on wake, read `done.json`, `mcp__lab__fetch_artifacts`, `mcp__lab__metrics`.

### B. Sweep a grid and aggregate
See **`examples/02-sweep-and-wait.md`**. Pattern: `mcp__lab__sweep` over the
grid → background `lab wait --sweep <sweep_id>` → on wake, `mcp__lab__list`,
filter to `sweep_id`, fetch each, summarize succeeded vs failed.

### C. Live early-kill (watch and stop if off-track)
See **`examples/03-live-early-kill.md`**. Pattern: submit → poll
`mcp__lab__metrics(job_id, since_step=last)` every ~10s → if the divergence
criterion fires, `mcp__lab__cancel(job_id)`.

### D. Reuse cached results
Pass `cache=true` to `mcp__lab__submit`. The lab hashes
`(commit, command, normalized_config, seed)` (config leaves coerced to strings,
so CLI grids and API ints hit the same cache). On a hit, returns the existing
`job_id` with `cached: true` — no new job runs. **Cache only hits on a clean
tree** (commit your changes first).

### E. Per-job extra runtime dep
The remote env is lean (numpy/pydantic/hydra only). For a one-off dep:
`mcp__lab__submit(command="python experiments/needs_scipy.py", with_pkg=["scipy"])`.
This wraps the command as `uv run --with scipy python experiments/needs_scipy.py`
(see `tests/test_util.py:12-23`). Same on the CLI: `lab submit -c "..." --with scipy`.

## 7. Backend selection

| Backend | When to use | Required kwargs |
|---------|-------------|-----------------|
| `local` | Dev, smoke tests, CPU experiments on the local machine. Free. | none |
| `skypilot` | GPU work, parallel jobs, anything that shouldn't tie up the local machine. Costs money (Vast.ai). | `accelerators` (e.g. `"RTX4090:1"`) AND `timeout` (e.g. `"30m"`) |

If `accelerators` is omitted on `skypilot`, SkyPilot may land you on a non-GPU
host — pass it explicitly. `timeout` is a hard wall-clock cap; the job is killed
and marked `timed_out` if it overruns, and the machine is torn down.

## 8. Reproducibility & manifests

Every job writes `runs/<job_id>/manifest.json` (model in `src/lab/models.py`)
recording: created_at, git commit (+ dirty flag), uv.lock sha256, command,
resolved config, seed, backend + machine type + region, status timeline,
exit code + end reason, cost (estimated + actual), artifact URIs.

`runs/` is git-ignored. For artifacts that must survive a clean clone (and
for cross-machine `mcp__lab__fetch_artifacts`), enable R2 (see §2). Manifests
record artifact **URIs**, never credentials (spec FR-J1).

## 9. Common gotchas

- **Dirty tree disables cache.** Cache lookups skip when `git_dirty: true`.
- **Vast marketplace flakiness.** A single failed launch (machine vanished
  mid-provision) is not "the pipeline is broken" — resubmit; SkyPilot will
  pick a different offer.
- **`lab wait` timeout exits non-zero.** Use the exit code as the retry signal.
  (If a wrapper script swallows it, you'll mistakenly see "ok" — check the
  job's `state` via `mcp__lab__status` to be sure.)
- **No MCP `wait` tool.** By design: agents shouldn't block an MCP call for
  hours. Use the CLI as a background task.
- **Skypilot jobs need explicit `accelerators` and `timeout`.** Missing
  either is the most common mistake.
- **Grid values are strings on the argv.** The experiment (Hydra/typer/argparse)
  coerces types — the lab doesn't guess.

## 10. Pointers

- **Full reference (human-facing):** `DELIVERY.md` at the repo root.
- **Spec:** `LAB-REQUIREMENTS.md` (RFC-2119, FR/AC/NFR).
- **Design decisions:** `research/16-decisions.md`.
- **MCP tool source:** `src/lab/mcp_server.py`.
- **CLI source:** `src/lab/cli.py`.
- **Example experiment:** `experiments/example_capacity.py`.
