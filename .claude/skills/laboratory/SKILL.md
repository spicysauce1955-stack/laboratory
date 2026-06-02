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

## 5. The CLI surface (and the CLI-only commands)

Every MCP tool has a matching CLI command (`uv run lab submit / sweep / status
/ logs / metrics / fetch / cancel / list`). The `lab` CLI prints JSON
mirroring the MCP returns.

Two commands are CLI-only by design:

### `uv run lab wait` — the push-notify primitive

Block until one or more jobs reach a terminal state:

```bash
uv run lab wait <job_id_1> <job_id_2> --done-file done.json
# or:
uv run lab wait --sweep <sweep_id> --done-file done.json
# with a deadline — NOTE: --timeout here is SECONDS (a number), not "30m":
uv run lab wait <job_id> --timeout 1800 --done-file done.json
```

> **Gotcha — `wait --timeout` is in seconds, not a duration string.** Unlike
> `submit`/`sweep` `--timeout`, which accept `"30m"`/`"2h"`, `lab wait --timeout`
> takes a raw number of seconds (e.g. `1800` for 30 min). Passing `30m` here
> exits `2` (bad args). Convert first.

Why CLI-only: the right pattern is to run `lab wait` as a **Claude Code
background task**, keep working in the foreground, and let the task's
process-exit notify the harness — at which point you read `done.json` and
proceed.

**Exit codes:**
- `0` — all jobs terminal AND all teardowns clean.
- `1` — gave up on `--timeout` (some jobs still not terminal).
- `2` — bad arguments (no job ids / unknown id / empty sweep).
- `3` — all terminal BUT at least one **teardown leaked** (`teardown_status: "failed"`).
  Treat as an urgent signal — a paid GPU rental may still be running. Run
  `lab reconcile` immediately (see §6.F below).

### `uv run lab reconcile [--apply]` — leak detection & cleanup (FR-C2)

Lists active Vast.ai rentals directly via the vastai-sdk and cross-checks
them against the local job DB. **Always run this after seeing
`teardown_status: "failed"` or `lab wait` exiting 3.**

```bash
uv run lab reconcile             # dry-run: print orphans + ghosts
uv run lab reconcile --apply     # destroy the orphans
```

- **Orphans** = Vast.ai rentals labelled `lab-*` but not tied to any running
  lab job (probable leaks; bill until destroyed).
- **Ghosts** = lab jobs whose manifest still says `running` but Vast has no
  matching rental (supervisor probably died; safe to investigate via
  `lab status <id>`).

Exit code: `0` if nothing to do, `3` if orphans were found in dry-run
mode (re-run with `--apply`), `2` on error.

### `uv run lab dashboard` — live terminal view

Live table of all jobs with state, cost, latest metric, and a **`teardown`**
column that flags `LEAK` rows loudly. Ctrl-C to exit.

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
criterion fires, `mcp__lab__cancel(job_id)`. The returned points live under the
**`series`** key (`result["series"]["loss"]`), not at the top level — index
into `series` before reading values.

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

### F. Recover from a teardown leak (FR-C2)
See **`examples/04-reconcile-leak.md`**. Pattern: `lab wait` exits 3 (or you
see `teardown_status: "failed"` in `lab status`) → `lab reconcile` (dry-run)
→ inspect the orphans → `lab reconcile --apply` to destroy them. The lab
already retries `sky.down` and falls back to vastai-sdk directly on failure,
so leaks are rare — but `reconcile` is the operational safety net when even
that fails.

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
exit code + end reason, cost (estimated + actual), artifact URIs, and
**`teardown_status`** (`"succeeded" | "failed" | null`) — the FR-C2 leak
signal. A `"failed"` value means a paid rental may still be billing; the
`end_reason` field is annotated with an actionable instruction in that case.

`runs/` is git-ignored. For artifacts that must survive a clean clone (and
for cross-machine `mcp__lab__fetch_artifacts`), enable R2 (see §2). Manifests
record artifact **URIs**, never credentials (spec FR-J1).

## 9. Common gotchas

- **Dirty tree disables cache.** Cache lookups skip when `git_dirty: true`.
- **Vast marketplace flakiness.** A single failed launch (machine vanished
  mid-provision) is not "the pipeline is broken" — resubmit; SkyPilot will
  pick a different offer.
- **Provisioning watchdog → `failed` with "provisioning exceeded …".** A dead
  Vast host stuck in "loading" used to hang the job forever. The lab now aborts
  any host that doesn't reach UP within the provision timeout (default **8m**,
  override with `--provision-timeout 10m` / `provision_timeout="10m"`), tears it
  down, and marks the job `failed` with `end_reason` `provisioning exceeded
  <N>s (… likely a dead Vast offer)`. **This is a dead-host signal, not a code
  failure — just resubmit** (a fresh offer usually comes up healthy). Distinct
  from a run-time `timed_out`, which means your experiment itself ran too long.
- **`lab wait` exit codes are meaningful.** `0` = clean; `1` = timed out;
  `3` = **a teardown leaked** (paid rental may still be running — run
  `lab reconcile` now); `2` = bad args. If a wrapper script swallows the
  exit code, you'll mistakenly see "ok" — check `teardown_status` via
  `mcp__lab__status` to be sure.
- **`teardown_status: "failed"` is a money alarm.** The lab already retries
  `sky.down` for ~3.5 min and falls back to direct vastai-sdk destroy; a
  `"failed"` value means even that failed. **Always follow up with
  `lab reconcile --apply`** to stop the bleed.
- **No MCP `wait` or `reconcile` tools.** By design: agents shouldn't block
  an MCP call for hours, and reconcile is an operational/destructive command.
  Both live on the CLI; use `lab wait` as a background task.
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
