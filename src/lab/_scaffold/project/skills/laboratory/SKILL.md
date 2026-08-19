---
name: laboratory
description: "Run/execute a reproducible ML or compute experiment via the lab runner (MCP tools / `lab` CLI) — in this project this is the right way to actually launch a training/experiment job, not running the script directly. Use when the user wants the work done, not just discussed: run, submit, or kick off an experiment; sweep a grid over hyperparameters/seeds and report which config won; shard a large-seed sweep into independently-bounded per-seed sub-jobs and aggregate one per-cell result (sweep-aggregate / sweep-retry); put a job on a remote GPU (RTX 4090 on Vast.ai via SkyPilot; T4/L4 on GCP) or a cheap remote CPU box (DigitalOcean/GCP, --backend cpu), cap its cost or runtime; REGISTER/schedule an experiment for later — run tonight/off-hours, run when a GPU price drops, run after another job, queue/hold/cancel deferred runs while the laptop is closed; stream live metrics and kill a diverging run early; fetch results/artifacts; reproduce a prior run or verify a result still reproduces (lab confirm); export a committable provenance bundle for the paper (lab export); diagnose a billing/teardown leak ('am I still being charged?', stuck Vast rental, `lab wait` exit 3); or read back the lab's own event ledger — what have I already tried this session, why did that submit/sweep actually fail, which failures keep recurring and what have they cost (lab history / lab report). Triggers: lab submit, lab sweep, lab sweep-aggregate, lab sweep-retry, lab wait, lab confirm, lab export, lab lint, lab register, lab queue, lab scheduler, lab reconcile, lab history, lab report. Skip for merely writing an experiment script or reading saved results."
metadata:
  version: "0.9.0"
  last_updated: "2026-08-19"
  status: active
---

# Laboratory — Remote Experiment Runner

This skill teaches you to drive the **lab** — an experiment-agnostic remote job
runner installed into this project as a dependency. The lab handles: submitting
a job without blocking, watching its metrics live, sweeping a grid, fetching
durable artifacts, and pinning everything to a reproducible manifest (commit +
uv.lock + config + seed). Provenance is pinned against **this project's** git
history, not the lab's.

You will normally use the **MCP tools** (`mcp__lab__*`) registered by this
project's `.mcp.json` (written by `lab init`). For push-notify (block-until-done as a background task), use the
**CLI** `uv run lab wait` — the MCP `wait` tool is deliberately bounded (default
600s) and is only for short waits.

## 1. When to use this skill

Invoke this skill when the user asks (in any phrasing):

- "Run / submit this experiment" (especially with a seed, grid, or GPU).
- "Sweep over K / alpha / seeds" (parameter grid).
- "Shard a big-seed sweep so each chunk has its own timeout" / "split the 32
  seeds into sub-jobs and stitch one result back together" (sharded sweep — §4
  `sweep(seeds=…, shard_size=…)` + `sweep-aggregate`).
- "Watch the run live and stop it if it's off-track" (live early-kill).
- "Fetch the results / artifacts of job `<id>`."
- "Reproduce job `<id>` / re-run with the same config" / "does this result
  still hold?" (`lab confirm`).
- "Export these results so they can be committed / cited" (`lab export`).
- Anything that wants a remote GPU (Vast.ai or GCP) or a cheap remote CPU box,
  or a cost-bounded job, or a manifest-tracked run.
- "What have I already tried?" / "why did that submit fail?" / "what keeps
  breaking and what has it cost me?" — the event ledger (§4b, §6 I).

**Don't** invoke this skill for a one-off local sanity check that doesn't need
tracking — just running `uv run python experiments/foo.py` is fine.

## 2. Prerequisites (verify before first use)

Run from **this project's** root — the git repo whose commits the lab pins as
provenance and under whose `runs/` results land.

- **Sync deps.** `uv sync` installs what this project's `pyproject.toml` pins,
  including the `laboratory` version it depends on. Remote backends need the
  matching extras on that dependency; if a backend is missing, re-add it with
  the extras you need, e.g.
  `uv add "laboratory[skypilot,gcp,r2] @ git+https://github.com/spicysauce1955-stack/laboratory@vX.Y.Z"`
  (`skypilot` = Vast.ai, `do` = DigitalOcean, `gcp` = Google Cloud, `r2` =
  durable artifacts).
- **Remote creds (only for remote backends).** Machine-local, never committed —
  see `.env.example` in this project.
  - Vast API key in `~/.config/vastai/vast_api_key` (`--cloud vast`, the default).
  - GCP: gcloud ADC auth + project + GPU quota — see §7. Run `uv run lab doctor
    --cloud gcp` before spending money; it checks creds, project, billing, APIs,
    IAM and quota.
  - R2 (optional, for durable artifacts):
    - Creds in `~/.cloudflare/r2.credentials` (S3-compat: Access Key + Secret).
    - Env: `LAB_R2_ENDPOINT="https://<account>.r2.cloudflarestorage.com"` and
      `LAB_R2_BUCKET="lab-artifacts"`.
- **MCP server.** Registered by `.mcp.json` at this project's root, written by
  `uv run lab init`. Opening the project in Claude Code should offer the `lab`
  server; once enabled, the tools below appear as `mcp__lab__submit`, etc. If
  they are missing, run `uv run lab init --check`.
- **Dirty trees are captured, not lost (fail-closed provenance, FR-B1).**
  Manifests pin `HEAD`; if the tree is dirty the lab **auto-snapshots the
  uncommitted changes** (tracked diff + untracked files) and records a
  `diff_ref`, so the exact code is always reconstructable — it will never write
  `git_commit: null` or `git_dirty: true, diff_ref: null`. Pass `--no-dirty`
  (CLI) / `allow_dirty=false` (MCP) to refuse a dirty submit instead. Note the
  cache (`cache=true`) still only hits on a **clean** tree, and `lab confirm`
  still refuses a dirty producer — commit when you want cache reuse or
  confirmability. See the [provenance & timeouts guide](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/guides/provenance-and-timeouts.md).

## 3. The Experiment Contract (spec §7)

If asked to *write* a new experiment, the entrypoint MUST:

| What the entrypoint does                          | How                                                |
|---------------------------------------------------|----------------------------------------------------|
| Read its run dir and seed from env                | `os.environ["LAB_RUN_DIR"]`, `LAB_RUN_ID`, `LAB_SEED` |
| Write outputs under `$LAB_RUN_DIR`                | All files (figures, checkpoints, tables) go here   |
| Log metrics incrementally                         | One JSON object per line into `$LAB_RUN_DIR/metrics.jsonl` (helper: `lab.metrics.log_metric(name, value, step)`) |
| Exit non-zero on failure                          | `sys.exit(1)` or raise                             |
| Accept grid overrides as `key=value` argv         | e.g. Hydra picks up `seed=3 K=200` from `sys.argv` |
| Report the config it actually consumed            | Write `$LAB_RUN_DIR/effective_config.json` (one call: `lab.experiment.get_overrides(known={...})`) |

**The config-consumption handshake (field-report #1).** A *succeeded* job whose argv
overrides were never consumed (no `effective_config.json`, or the file omits submitted keys)
is **flipped to `failed`** at the succeeded transition — a typo'd or stale key must not
silently run a different experiment than requested. `allow_unknown_config=true`
(CLI `--allow-unknown-config`) opts a legacy entrypoint out; better, pre-check it with
`uv run lab lint -c "<cmd>" --grid k=v` (heuristic grep for override keys the script never
references; exits 1 on findings).

A metric line is `{"name": "...", "value": <float>, "step": <int>, "wall_time": <float>}`.
The lab tolerates a half-written trailing line, so you can write line-by-line
and the metrics tool will still read cleanly.

**Reference template:** `experiments/example.py` (~40 lines, contract-compliant).

**For sharded sweeps (§4 `sweep` with `seeds`/`shard_size`)** the entrypoint additionally must:
read its assigned seed subset from a config key (default `seeds`, a comma list like `seeds=0,1,2,3`)
and emit **one row per seed** into a row-structured result file (default `results.csv`) that includes
a column identifying the seed (default `seed`). The lab row-concatenates those per-shard files into one
per-cell aggregate and uses the seed column to report present-vs-expected and name missing seeds. Both
the file and the column are overridable (`--results-file` / `--seed-column`). **Multiple rows per
seed** (an axis swept inside the job, e.g. one row per (seed, α)) are supported by declaring the
row identity at sweep time: `--row-key seed,alpha` — duplicates are then judged on the full key.
Guide: the [sharded sweeps guide](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/guides/sharded-sweeps.md).

## 4. The MCP tool surface

All registered by the lab's MCP server (`lab mcp`). Each returns a
JSON-serializable dict.

### `mcp__lab__submit`
Submit one job. Non-blocking — returns immediately with the `job_id`.

| Input            | Type           | Notes |
|------------------|----------------|-------|
| `command`        | str (required) | e.g. `"python experiments/example.py"` |
| `backend`        | str            | `"local"` (default), `"skypilot"`, or `"cpu"` (cheap CPU box: 4 vCPU + 50 GB volume, DO by default) |
| `cloud`          | str            | SkyPilot cloud: `"vast"` (default) \| `"do"` \| `"gcp"`. `--backend cpu --cloud gcp` runs the cpu profile on GCP (spot allowed there; DO has none). GCP GPUs: `accelerators="T4:1"`/`"L4:1"`. |
| `disk_size`      | int            | Boot/attached volume GB (skypilot; sizes the DO volume) |
| `cache`          | bool           | If true and a prior identical-`(commit, command, config, seed)` succeeded job exists on a clean tree, reuse it |
| `seed`           | int            | Recorded + injected as `$LAB_SEED` |
| `code_ref`       | str            | Git ref to pin; default `"HEAD"` |
| `cpus` / `memory`/ `gpus` | int/str/int | Resource hints (memory like `"8"` GB) |
| `accelerators`   | str            | **Required for skypilot.** e.g. `"RTX4090:1"` |
| `timeout`        | str            | Wall-clock cap; `"30m"`, `"2h"`, `"45s"`. On overrun the job is killed, the machine torn down, and the manifest reads `status: timed_out`, `end_reason: "timed out after <N>s wall-clock cap"` |
| `with_pkg`       | list[str]      | Per-job extra runtime deps (e.g. `["scipy", "scikit-learn>=1.4"]`) — layered via `uv run --with` |
| `allow_dirty`    | bool           | Default `true` (dirty tree → snapshot the diff). Set `false` to **refuse** a dirty submit (FR-B1) |
| `allow_unknown_config` | bool     | Default `false`. `true` skips the config-consumption check (§3) for legacy entrypoints |

Returns: `{"job_id": "...", "cached": bool, "status": "queued"|"succeeded"|...}`.

### `mcp__lab__sweep`
Submit a Cartesian-product grid of jobs under one `sweep_id`. Same kwargs as
`submit`, plus:

| Input | Type | Notes |
|-------|------|-------|
| `grid` | dict[str, list] | e.g. `{"seed": [1,2,3], "K": [100, 200, 500]}` → 9 jobs |
| `seeds` | str \| list[int] | **Sharded sweep (P1-2).** A seed set as a range `"0-31"`, a comma list `"0,1,2"`, or `[0,1,2]`. Declares seeds as a first-class axis evaluated for **every** grid cell. |
| `shard_size` | int | Max seeds per sub-job. Each cell's `seeds` are split into shards of at most this size (`shard_size ≥ len(seeds)` ⇒ one shard per cell = today's behavior). `timeout`/`backend`/`accelerators` apply **per shard**. |
| `results_file` | str | Per-run row-structured result file to row-concatenate per cell (default `"results.csv"`). |
| `seed_column` | str | Column in `results_file` identifying each row's seed (default `"seed"`). |
| `row_key` | str \| list | Columns identifying a row when the job writes multiple rows per seed (e.g. `"seed,alpha"` for an inner α loop); duplicates judged on the full key. |

Grid values become `key=value` overrides on the experiment's argv (string-valued
— Hydra/typer coerce). If `seed` is a grid key, it sets each job's recorded seed.
It is an error to declare seeds in both `seeds` and a grid key (`seed`/the
seed-axis key). **`grid` is optional when `seeds` is given** — a pure seed sweep
needs no grid.

Returns (plain grid): `{"sweep_id": "...", "job_ids": [...]}`.
Returns (sharded, i.e. `seeds` given): `{"sweep_id": "...", "cells": [{coords,
cell_id, shard_job_ids, aggregate_ref, seeds_expected, seeds_present, status}, ...]}`.

### `mcp__lab__sweep_aggregate`
**Sharded sweeps only.** `{sweep_id, strict?, row_key?}` → row-concatenate each
cell's shards' `results_file` into one per-cell aggregate at `aggregate_ref`, and
return the cell view. **By default it also salvages partial rows** from terminal
*non-succeeded* shards (timed-out etc.) — those rows carry a `_shard_status`
column and the cell view lists them in `seeds_partial`; `strict=true` (CLI
`--strict`) aggregates succeeded shards only. **Idempotent** — safe to re-run as
shards finish; it is also the way to view current per-cell status (`status` is
`complete` iff every expected seed is present, else `incomplete` with
`missing_seeds` named — never reports a short aggregate as complete, never
discards recovered seeds). `row_key` (e.g. `"seed,alpha"`, CLI `--row-key`)
retrofits the row identity onto a plan created before `row_key` existed (pre-v0.2.1)
and persists it for future aggregates.

### `mcp__lab__sweep_retry`
**Sharded sweeps only.** `{sweep_id}` → resubmit **only** the missing seeds of
incomplete cells (skipping seeds already covered — including partial-salvaged
ones — or with an in-flight prior retry), then re-aggregate and return the
updated cell view.

> Note: `mcp__lab__sweep_status` remains the *outcome/cost* summary
> (states, preemptions, per-point spend) — it does **not** carry the per-cell
> seed view; use `sweep_aggregate` for that.

### `mcp__lab__status`
`{job_id}` → `{state, lab_version, started_at, ended_at, exit_code, end_reason,
cost, estimated_running_usd, last_log_line, teardown_status, sweep_id,
code: {git_commit, git_dirty, diff_ref}, mirrored}`. A running remote job shows
its spend-so-far (`estimated_running_usd`) and latest log line — enough to judge
"is it alive and worth its bill" from one cheap call. `lab_version` is the lab
release that produced the run (`null` on pre-0.5.0 manifests) — see §8.
States: `queued`, `running`, `succeeded`, `failed`, `cancelled`, `timed_out`,
`preempted`. Cheap to poll. **`teardown_status: "failed"` is the FR-C2 money
alarm** — call `mcp__lab__reconcile` immediately. Scheduler-launched (deferred)
jobs are read from the mirrored manifest (`mirrored: true`; may be a tick stale).

### `mcp__lab__wait`
`{job_ids?, sweep?, timeout?=600, interval?=10}` → `{all_terminal, teardown_leaks,
teardown_unconfirmed, jobs}`. Blocks up to `timeout` seconds. Non-empty
`teardown_leaks` = a paid machine may still be billing. For long runs prefer
`lab wait` as a background task (below); this tool is for bounded waits.

### `mcp__lab__confirm`
`{run_id, metric?, rtol?=1e-3, atol?, wait?}` → the **reproducibility gate (FR-B)**:
relaunch `run_id` fresh (no cache) from its pinned commit, compare the re-run's
final metrics against the original within tolerance → verdict
`"match" | "drift" | "rerun_failed"` with per-metric deltas. Refuses a
non-succeeded or **dirty** producer (no honest result to re-derive).
`wait=false` submits the re-run and returns `{confirm_id, verdict: "pending"}`.
CLI: `uv run lab confirm <run_id>`.

### `mcp__lab__export`
`{job_or_sweep_id, to, include_logs?}` → write the **committable provenance
bundle** to a directory: manifests + result tables + resolved configs + code
diffs (+ sweep plan/aggregates), tied together by an `index.json` with commits
and spend. The part of git-ignored `runs/` that belongs in version control next
to the paper; excluded blobs are listed under `skipped`, never silently dropped.
CLI: `uv run lab export <job|sweep_id> --to DIR [--logs]`.

### `mcp__lab__reconcile`
`{}` → the dry-run leak report. Multi-pass and cloud-aware: a Vast-direct pass
(`orphans`; skipped without vastai-sdk), a cloud-agnostic `sky.status` pass
covering DO/GCP clusters (`sky_orphans`), a DO detached-volume pass, a GCP
compute-API pass (`lab-*` instances + unattached `lab-*` disks), plus `ghosts`
and `unsupervised` (running jobs whose supervisor pid is dead — their clusters
are NOT counted as healthy). Read-only: it never destroys anything — cleanup is
`lab reconcile --apply --yes` at the CLI (only `--apply` is CLI-only; the
dry-run report is right here).

### `mcp__lab__metrics`
`{job_id, names?, since_step?}` → `{"series": {name: [{step, value, wall_time}, ...]}}`.
Incremental — pass `since_step=<last_step_seen>` to fetch only new points.
Designed for live polling at ~5–15s cadence (the early-kill loop).

### `mcp__lab__logs`
`{job_id, tail=100}` → `{"lines": [...]}`. The stdout/stderr of the job.

### `mcp__lab__history`
`{limit=50, since?, action?, job?, session?, failures?, full?, stats?}` →
`{"events": [{id, ts, action, surface, status, duration_ms, refs, result,
error}, ...]}`. **The lab's record of its own calls** — every CLI invocation and
MCP tool call, with `status` one of `ok` / `error` / `usage_error` / `crash` /
`interrupted` / `running-or-died`. Use it to answer *what did I already try*
before repeating a submit, and *why did that fail* afterwards.

- `full=True` adds the params, the internal trace (provisioning attempts,
  zone skips, launch retries, teardown steps) and a cross-reference to the
  job's manifest state and `logs.txt` path.
- `stats=True` returns the aggregate instead of rows: failure rate per command,
  ranked error signatures, dollars burned on failed calls.
- `job=<id>` finds every call that touched a job — including the SkyPilot
  supervisor's own record, which is where a provisioning or teardown failure
  is explained.

**This is not `logs`.** `logs` tails one job's stdout; `history` reads the
tool's own ledger.

### `mcp__lab__report`
`{since="7d", all_projects?}` → `{"markdown": "..."}`. A digest of what failed
in the window and what it cost: a triage table ranked by frequency × dollars,
then per-finding attempted / observed / cost. Hand this to the user (or paste
into an issue) when they ask what keeps going wrong.

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

Every MCP tool has a matching CLI command (`uv run lab submit / confirm / sweep /
sweep-aggregate / sweep-retry / export / lint / status / logs / metrics / fetch /
cancel / list / history / report`).
The `lab` CLI prints JSON mirroring the MCP returns. (`lab submit
--no-dirty` is the CLI form of `allow_dirty=false` — refuse a dirty tree instead
of snapshotting it.) Sharded-sweep CLI form:
`lab sweep -c "<cmd>" --grid N=1000,1500 --seeds 0-31 --shard-size 8`, then
`lab sweep-aggregate <sweep_id>` (idempotent; also the per-cell status view) and
`lab sweep-retry <sweep_id>` (resubmit only missing seeds).

Two commands have their full form on the CLI (MCP carries a bounded `wait` and
a read-only `reconcile`; `--apply` cleanup and unbounded waits stay CLI):

### `uv run lab wait` — the push-notify primitive

Block until one or more jobs reach a terminal state:

```bash
uv run lab wait <job_id_1> <job_id_2> --done-file done.json
# or:
uv run lab wait --sweep <sweep_id> --done-file done.json
# with a deadline (duration string or bare seconds):
uv run lab wait <job_id> --timeout 30m --done-file done.json
# stop waiting the moment any job fails (exit 4; survivors keep running/billing):
uv run lab wait --sweep <sweep_id> --fail-fast --done-file done.json
```

For long runs the right pattern is still to run `lab wait` as a **Claude Code
background task**, keep working in the foreground, and let the task's
process-exit notify the harness — at which point you read `done.json` and
proceed. Use `mcp__lab__wait` only for bounded waits (its `timeout` defaults
to 600s); it returns the same summary shape as `done.json`.

**Exit codes:**
- `0` — all jobs terminal AND all teardowns clean.
- `1` — gave up on `--timeout` (some jobs still not terminal).
- `2` — bad arguments (no job ids / unknown id / empty sweep).
- `3` — all terminal BUT at least one **teardown leaked** (`teardown_status: "failed"`).
  Treat as an urgent signal — a paid GPU rental may still be running. Run
  `lab reconcile` immediately (see §6.H below).
- `4` — `--fail-fast` tripped: a job hit `failed`/`timed_out` while others still run.
  The summary's `pending` lists the survivors (still billing); nothing was cancelled.
  **Exception:** if the dead job's teardown is already a confirmed leak, exit is `3`, not
  `4` — the money alarm always outranks the fail-fast signal.

`--done-file` is atomically rewritten after **each** job reaches a terminal state (with a
`pending` list), so a watcher can react mid-wait instead of waiting for process exit.
`--timeout` accepts durations (`"10m"`, `"2h"`) as well as bare seconds.

### `uv run lab reconcile [--apply]` — leak detection & cleanup (FR-C2)

Cross-checks live cloud resources against the local job DB, across all three
clouds (Vast-direct via vastai-sdk, a cloud-agnostic `sky.status` pass for
DO/GCP clusters, DO detached volumes, GCP `lab-*` instances + unattached disks).
**Always run this after seeing `teardown_status: "failed"` or `lab wait`
exiting 3.**

```bash
uv run lab reconcile               # dry-run: print orphans + ghosts
uv run lab reconcile --apply --yes # destroy the orphans, unattended
```

**`--apply` alone prompts.** It re-runs the dry sweep, lists every resource it is
about to destroy on stderr, and asks — then destroys *only* the approved set.
With **no tty (which is every agent invocation)** it refuses rather than
prompting: exit `4`, `{"aborted": true, "reason": "no tty", "would_destroy":
[...]}` on stdout, **nothing destroyed**. So from an agent always pass
`--apply --yes`; use bare `--apply` only when a human is at the terminal.

- **Orphans** = cloud resources labelled `lab-*` but not tied to any running
  lab job (probable leaks; bill until destroyed).
- **Ghosts** = lab jobs whose manifest still says `running` but the cloud has no
  matching rental (supervisor probably died; safe to investigate via
  `lab status <id>`).
- **Unsupervised** = non-terminal jobs whose supervisor pid is dead — their
  clusters do NOT count as healthy, so a frozen `running` manifest can't hide a
  still-billing box.

Exit codes: `0` nothing to do · `2` error · `3` orphans found in dry-run mode
(action required — re-run with `--apply --yes`) · `4` the confirmation was
declined **or** there was no tty to ask at (nothing was destroyed).

### `uv run lab dashboard` — live terminal view

Live table of all jobs with state, cost, latest metric, and a **`teardown`**
column that flags `LEAK` rows loudly. Ctrl-C to exit.

## 5b. Deferred scheduling — `lab register` / `lab queue` / `lab scheduler`

For "run this **tonight** / when a GPU is **cheap** / **after** that job — and let me close
the laptop." A *registration* = a normal job spec + triggers + guardrails, written to an R2
queue; an always-on host evaluates triggers every 60s and launches via skypilot. Spec:
the [deferred-scheduling design](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/superpowers/specs/2026-06-10-deferred-scheduling-design.md); host runbook:
the [scheduler deploy README](https://github.com/spicysauce1955-stack/laboratory/blob/main/deploy/scheduler/README.md).

**Prereq:** export `LAB_R2_ENDPOINT` (+ `LAB_R2_BUCKET=lab-artifacts`) — register/queue talk
to R2, not the local store. **And you need the scheduler host:** any always-on box running
`lab scheduler tick` on a 60s timer (the host runbook linked above ships the systemd unit +
timer; a small cloud VM costs a few $/mo). It is stateless — all state lives in R2, so it is recreate-safe
and can be destroyed when idle. Without a live host, registrations just sit in the queue
(`lab queue list` → `heartbeat_age_s` > 120s).

**The canonical "run tonight" command:**

```bash
uv run lab register -c "uv run experiments/example.py" \
    --gpu RTX4090:1 --timeout 2h \
    --window 23:00-07:00 --tz Europe/Berlin \
    --max-hourly 1.50 --max-cost 3 --expires +2d
# -> {reg_id, worst_case_cost_usd}; then just close the laptop.
```

| Trigger / guardrail | Flag | Notes |
|---|---|---|
| daily window | `--window HH:MM-HH:MM --tz <IANA>` | gates *start* only; may cross midnight |
| absolute earliest | `--not-before <ISO>` | |
| price gate | `--max-hourly <$/h>` (+ `--offer-query`) | **Vast-only** (queries the Vast offer feed; rejected for `--cloud do|gcp`); see headroom gotcha below |
| dependency | `--after <reg_id>` (repeatable) | dead/typo'd dep ⇒ auto-cancel, so get the id right |
| run-by deadline | `--expires +2d` / ISO | **required**; entry expires, never fires late |
| per-job cap | `--max-cost <$>` | vs best-offer×timeout |
| no triggers | (none) | = launch ASAP under scheduler-host supervision |

**Manage:** `lab queue list` (states, skip reasons, `heartbeat_age_s` — >120s means the
scheduler is down) · `queue show <reg_id>` · `queue cancel|hold|release <reg_id>` ·
`queue pause|resume` · `queue budget --per-day 5 --max-concurrent 4 [--clear-budget]`.
`lab register-sweep` registers a whole (optionally sharded) sweep the same way; both take
`--cloud vast|do|gcp`. MCP mirrors: `register`, `register_sweep`, `queue_list/show/cancel/pause`.

**Next morning:** `lab queue list` → `lab status <job_id>` (works from the laptop via the
R2-mirrored manifest, incl. cost) → `lab fetch <job_id>` (artifacts come from R2) →
`lab reconcile` if anything looks off.

**Live-learned gotchas (these cost real money to discover):**

- **GPU names: use the sky-catalog form `RTX4090:1`** (no underscore). Vast's own API says
  `RTX_4090` — the price feed translates automatically, but sky's launcher does NOT: an
  underscored name fails with "Catalog does not contain any instances".
- **Set `--max-hourly` ~2× the cheapest live offer.** Sky picks hosts by its *stale catalog*
  price; the actual Vast rental often bills 2–6× the cheapest offer. Too-tight caps trigger
  the post-launch price-verify **rollback loop** (rent → detect over-price → teardown →
  retry), each cycle costing cents until `--expires`. The real cost bound is
  `--timeout` × actual hourly, capped by `--max-cost`.
- **`lab logs` / `lab metrics` do NOT work from the laptop for scheduler-launched jobs**
  (only the manifest is mirrored to R2; they exit 2 with a structured error). Use
  `lab status` + `lab fetch` from the laptop; for live logs, tail them on the scheduler host
  itself (`ssh <scheduler-host> sudo tail /opt/<your-project>/runs/<job_id>/logs.txt`, where
  `/opt/<your-project>` is wherever its checkout lives — see the host runbook).
- **Cancel applies on the next tick** (≤60s), including killing an already-launched job.
- **Mirror lag:** `teardown_status` may read `null` from the laptop for a tick or two after
  success; the scheduler host's manifest is authoritative, `lab reconcile` is ground truth.

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

### B′. Shard a large-seed sweep and aggregate per cell (P1-2)
When a cell has many seeds / long per-cell wall time (so one all-seeds job is too
long for a single timeout), declare the seed axis and a shard size instead of
chunking by hand. Pattern: `mcp__lab__sweep(grid=…, seeds="0-31", shard_size=8)`
→ background `lab wait --sweep <sweep_id>` → on wake, `mcp__lab__sweep_aggregate
(<sweep_id>)` to get one per-cell `results.csv` plus `seeds_present`/`status`
per cell → if a cell is `incomplete`, `mcp__lab__sweep_retry(<sweep_id>)` reruns
only the missing seeds, then aggregate again. Each shard is a normal job (own
timeout + teardown + manifest); the experiment must honor the sharded results
contract in §3. Guide: the [sharded sweeps guide](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/guides/sharded-sweeps.md).

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
The remote env is **this project's locked env**: the box syncs your checkout and
runs `uv sync --frozen --no-default-groups`, so every runtime dependency in
`pyproject.toml`/`uv.lock` is installed (including the `laboratory` dependency's
own runtime deps) — only the dev/test groups (pytest, ruff, mypy) are skipped.
A dep that isn't in the lockfile is *not* there. For a one-off, don't edit the
project: `mcp__lab__submit(command="python experiments/needs_scipy.py",
with_pkg=["scipy"])` wraps the command as
`uv run --with scipy python experiments/needs_scipy.py`. Same on the CLI:
`lab submit -c "..." --with scipy` (repeatable). If the dep is permanent,
`uv add` it and commit instead — `--with` resolves on every launch.

### F. Verify a result still reproduces
`mcp__lab__confirm(run_id)` relaunches the run from its pinned commit and
compares final metrics → `match`/`drift`/`rerun_failed`. Producers must be
succeeded and clean (commit first). Use before building on a surprising result.

### G. Export results for the paper / repo
`mcp__lab__export(job_or_sweep_id, to="artifacts/<name>")` (CLI
`lab export <id> --to DIR`) writes the committable bundle — manifests, tables,
configs, diffs, `index.json`. This is how results leave git-ignored `runs/`.

### H. Recover from a teardown leak (FR-C2)
See **`examples/04-reconcile-leak.md`**. Pattern: `lab wait` exits 3 (or you
see `teardown_status: "failed"` in `lab status`) → `lab reconcile` (dry-run)
→ inspect the orphans → `lab reconcile --apply --yes` to destroy them. The lab
already retries `sky.down` and falls back to vastai-sdk directly on failure,
so leaks are rare — but `reconcile` is the operational safety net when even
that fails.

### I. Work out what went wrong (and what keeps going wrong)
The lab records every call it runs. After a failure:
`mcp__lab__history(job=<id>, full=True)` — the failing call plus its internal
trace (which zones were skipped, which provision attempt timed out, how
teardown went) and the pointer to that job's manifest and `logs.txt`. Before
repeating an expensive submit: `mcp__lab__history(limit=20)` to see what this
session already attempted. For a pattern rather than an incident:
`mcp__lab__history(stats=True, since="30d")`, or `mcp__lab__report(since="7d")`
for a pasteable digest. Successful calls carry no trace — the detail appears
only where something failed.

## 7. Backend selection

| Backend | When to use | Required kwargs |
|---------|-------------|-----------------|
| `local` | Dev, smoke tests, CPU experiments on the local machine. Free. | none |
| `skypilot` | GPU work, parallel jobs, anything that shouldn't tie up the local machine. Costs money. | `accelerators` (e.g. `"RTX4090:1"` on Vast, `"T4:1"`/`"L4:1"` on GCP) AND `timeout` (e.g. `"30m"`) |
| `cpu` | Remote CPU work on a cheap on-demand box: **4 vCPU + 50 GB volume** default (DigitalOcean), up to 48 vCPU. | none (accelerators rejected) |

The **cloud** is orthogonal: `--cloud vast|do|gcp` (default `vast`) on
submit/sweep/register. `--backend cpu --cloud gcp` runs the cpu profile on GCP
— unlike DO, GCP allows spot there. GCP needs the `gcp` extra on the
`laboratory` dependency, gcloud ADC auth, and (for GPUs) per-family regional
quota; guide:
the [GCP backend guide](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/guides/gcp-backend.md).

The cpu-profile defaults are deliberately inside a **fresh DO account's tier**:
8-vCPU sizes and SkyPilot's default 256 GB volume both `422` on an untouched
account — bigger needs a DO tier-increase ticket (`disk_size` overrides the
volume GB).

If `accelerators` is omitted on `skypilot`, SkyPilot may land you on a non-GPU
host — pass it explicitly. `timeout` is a hard wall-clock cap; the job is killed
and marked `timed_out` if it overruns, and the machine is torn down.

## 8. Reproducibility & manifests

Every job writes `runs/<job_id>/manifest.json` (schema: `lab.models.JobManifest`)
recording: created_at, git commit (+ dirty flag + `diff_ref`), uv.lock sha256,
command, resolved config, seed, backend + machine type + region, status
timeline, exit code + end reason, cost (estimated + actual), artifact URIs,
**`lab_version`**, and **`teardown_status`** (`"succeeded" | "failed" | null`) —
the FR-C2 leak signal. A `"failed"` value means a paid rental may still be
billing; the `end_reason` field is annotated with an actionable instruction in
that case.

**`lab_version`** is the lab release that produced the run (`null` on manifests
written before v0.5.0). It matters because the lab is a *dependency* upgraded on
its own schedule, independent of this project's commits: when `lab confirm`
reports `drift`, compare the two runs' `lab_version` first — a lab upgrade
between them explains a difference that the pinned commit + config + seed
cannot.

**Fail-closed provenance (FR-B1).** The store refuses to *create* a manifest
whose `code` can't reproduce the run: `git_commit` is always a real SHA, and a
dirty tree always carries a `diff_ref`. `diff_ref` points at the captured
uncommitted changes — a local `runs/<job_id>/code_diff.tar.gz`, its durable
`r2://…` mirror when R2 is enabled, or (for deferred `register`/`register-sweep`
jobs) the code bundle key. To reconstruct a dirty run's exact tree:
`git checkout <git_commit>` then `lab.manifest.apply_diff(<diff_ref blob>, ".")`.
The guard is on *create* only, so old manifests still read; you never migrate.

`runs/` is git-ignored. For artifacts that must survive a clean clone (and
for cross-machine `mcp__lab__fetch_artifacts`), enable R2 (see §2). Manifests
record artifact **URIs**, never credentials (spec FR-J1).

## 9. Common gotchas

- **Dirty tree disables cache.** Cache lookups skip when `git_dirty: true`.
- **A succeeded job can flip to `failed` on unconsumed overrides.** If you pass
  `k=v` overrides and the entrypoint never writes them into
  `effective_config.json`, the lab fails the job at the succeeded transition
  (§3). Pre-check legacy scripts with `lab lint`; opt out with
  `--allow-unknown-config`.
- **Vast marketplace flakiness.** A single failed launch (machine vanished
  mid-provision) is not "the pipeline is broken" — resubmit; SkyPilot will
  pick a different offer. Transient *local-API* launch errors are already
  retried with backoff (`LAB_LAUNCH_RETRIES`, default 3; a give-up marks the
  job `failed` with an `end_reason` prefixed `transient:` — safe to resubmit).
  Remote sweep submits stagger by `LAB_SUBMIT_STAGGER_S` (default 1.5s) to
  avoid racing the provisioner.
- **Provisioning watchdog → `failed` with "provisioning exceeded …".** A dead
  Vast host stuck in "loading" used to hang the job forever. The lab now aborts
  any host that doesn't reach UP within the provision timeout (**per-cloud
  default: vast 8m, do 12m, gcp 20m**; override with `--provision-timeout 10m` /
  `provision_timeout="10m"`), tears it
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
  `sky.down` for ~3.5 min and falls back to a cloud-direct destroy (vastai-sdk
  on Vast, compute API on GCP); a `"failed"` value means even that failed. **Always follow up with
  `lab reconcile --apply --yes`** to stop the bleed.
- **MCP `wait` is bounded; MCP `reconcile` is read-only.** Don't block an MCP
  call for hours — use `lab wait` as a background task for long runs. The
  dry-run leak report *is* available over MCP; only destroying orphans
  (`--apply`) is CLI-only — and `--apply` without `--yes` refuses (exit 4,
  nothing destroyed) when no tty is there to confirm, which is always the case
  for an agent.
- **Skypilot jobs need explicit `accelerators` and `timeout`.** Missing
  either is the most common mistake.
- **Grid values are strings on the argv.** The experiment (Hydra/typer/argparse)
  coerces types — the lab doesn't guess.
- **`history` is the tool's ledger; `logs` is a job's stdout.** Reaching for
  `logs` to find out why a *submit* failed will not work — a submit that never
  became a job has no log. `history` covers it, including the calls that failed
  before a manifest existed.
- **A `running-or-died` row means the process never closed its call** — killed,
  OOMed, or still running. It is a finding, not a glitch: something ended
  without recording an outcome.

## 10. Pointers

In **this project**:

- **Example experiment:** `experiments/example.py` — the Experiment Contract, worked.
- **Machine-local settings:** `.env.example` → copy to `.env` (git-ignored).
- **Results:** `runs/<job_id>/` — manifest, logs, metrics, output.
- **Installed version:** `uv run lab --version`; refresh this skill and `.mcp.json`
  after upgrading with `uv run lab init`.

In the [laboratory repo](https://github.com/spicysauce1955-stack/laboratory)
(the lab's own source and docs):

- **Getting started:** [getting-started guide](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/guides/getting-started.md).
- **What a release freezes:** [COMPATIBILITY.md](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/COMPATIBILITY.md).
- **Provenance & timeouts:** [guide](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/guides/provenance-and-timeouts.md).
- **CPU backend:** [guide](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/guides/cpu-backend.md).
- **GCP backend:** [guide](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/guides/gcp-backend.md).
- **Sharded sweeps:** [guide](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/guides/sharded-sweeps.md).
- **Event ledger:** [guide](https://github.com/spicysauce1955-stack/laboratory/blob/main/docs/guides/event-logging.md).
- **Spec:** [LAB-REQUIREMENTS.md](https://github.com/spicysauce1955-stack/laboratory/blob/main/LAB-REQUIREMENTS.md) (RFC-2119, FR/AC/NFR).
