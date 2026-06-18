# Seed-sharded sweep with per-cell aggregation — design

**Date:** 2026-06-18 · **Status:** approved, pre-implementation
**Implements:** `p1-2-seed-sharded-sweep-requirements.md` (FR-SS-1…9, AC-1…5)
**Relates to:** FR-A5 (`sweep`), FR-B1 (fail-closed provenance), FR-C2 (teardown), §7 (Experiment Contract)

## 1. Problem

A sweep *cell* is one combination of the non-seed grid keys (e.g. one `(N, α)` point), evaluated
over a set of seeds. Today the lab runs **all seeds of a cell inside one job**. For large per-cell
wall time (Stage-2 A2: N=1500, 32 seeds; A1 density sweep) that single job carries one timeout over
everything and, in Stage-1, did not fit at all — seeds were chunked by hand and per-seed outputs
stitched together manually. That manual path caused a real ~$5.79 overrun (bad wall estimate), a
coarse failure blast radius (one dead host loses the whole cell), and mistake-prone hand
aggregation.

The ask: let a sweep declare its seed set and a shard size, split each cell's seeds into
independently-bounded sub-jobs, and return **one aggregated per-cell result** equivalent to a single
all-seeds job — automatically, with full provenance. This is **P1** (a manual fallback works).

## 2. Scope

**In scope.** The `sweep` interface (CLI `lab sweep` + MCP `sweep`): declaring a seed set and shard
size; per-cell row-concatenation aggregation; per-shard observability/teardown; merged provenance;
present-vs-expected accounting; honest partial-failure record; **re-running missing shards**;
**results-file override**.

**Out of scope (v1).** Merged per-cell *metrics* stream (results *table* aggregation is enough —
deferred nice-to-have); cross-cell aggregation; statistics/plotting (caller owns analysis); any
change to the single-job `submit` path or non-seed grid semantics.

## 3. How it lands on the existing architecture

- `Lab.sweep` (`core.py:328`) already expands a Cartesian `grid` into one `JobSpec` per point via
  `build_sweep_point_spec`, which appends shell-quoted `k=v` overrides to the command and records
  them in `resolved_config`. Jobs are grouped by a flat `sweep_id` field on the manifest.
- Seeds today are a single int per job (`run.seed` → `$LAB_SEED`). The "all seeds in one job"
  behavior means the **experiment script already consumes a seed-set config key and loops
  internally**, emitting one results row per seed. **Sharding = narrowing that set per job**, which
  maps directly onto the existing override-append mechanism — no contract change to how overrides
  are passed.
- There is **no cross-job artifact aggregation today**: `fetch_artifacts` is per-job and
  `sweep_summary` (`core.py:516`) reduces *outcomes* (states/cost), not *result rows*. The
  row-aggregation reducer is the genuinely new capability.
- A shard is a normal job, so FR-SS-2 (own timeout), FR-SS-3 (own seed record / reproducibility),
  FR-SS-8 (own fail-closed manifest), and FR-SS-9 (observable via `status`/`metrics`/`logs`) are
  inherited for free — each shard already gets a manifest, a per-job `timeout`, and `robust_teardown`.

## 4. Chosen approach: persist a `SweepPlan` (Approach B)

The one real decision is **where the cell→shard structure is recorded**. We persist an authoritative
`SweepPlan` under the `sweep_id` at submit time. Rejected alternatives: deriving grouping by diffing
`resolved_config` (fragile, no durable home for `aggregate_ref`, awkward retry — Approach A); making
the sweep a parent manifest (refactors the store + fail-closed invariants, overkill for P1 —
Approach C). Approach B mirrors how the codebase already persists structured plans (`Registration`,
the deferred-sweep bundle).

## 5. The Experiment-Contract addition

Row-concatenation is mechanical, but FR-SS-6 (expected vs present) and FR-SS-7 (name the missing
seeds) require the lab to read a **seed value out of each result row**. So the lab and the experiment
agree on **two names** — both with a sensible default and an override:

1. **Which file** is the row-structured result to concatenate → default `results.csv`.
2. **Which column** identifies the seed → default `seed`.

The script must emit one row per seed in `results_file`, including a `seed_column`. This is a thin
extension of §7 (alongside `$LAB_RUN_DIR` / `log_metric`), documented in the guide. Without the seed
column the lab could only count anonymous rows — it could not name missing seeds or detect
duplicates.

## 6. Components

### 6.1 Seed partitioning (pure helper)
`parse_seeds(spec) -> list[int]` accepts a range string (`"0-31"`) or an explicit list; returns a
**sorted, de-duplicated** seed set. `partition_seeds(seeds, shard_size) -> list[list[int]]` splits
into contiguous chunks of at most `shard_size` (complete, non-overlapping cover — FR-SS-1).
`shard_size >= len(seeds)` ⇒ one shard (AC-4). Pure; unit-tested in isolation.

### 6.2 `Lab.sweep` extension
New optional kwargs `seeds`, `shard_size`, `results_file`, `seed_column` (absent ⇒ today's behavior
exactly — AC-4). Steps:
1. Validate: **error** if seeds appear in both `seeds` and a `grid` key (§3, ambiguous) — clear
   message, no guessing.
2. `cells = expand_grid(grid)`; for each cell, `shards = partition_seeds(parse_seeds(seeds), shard_size)`.
3. For each `(cell, shard)`, build a `JobSpec` via `build_sweep_point_spec` with the cell coords as
   today **plus** the seed subset appended as a config override under `seed_column` (e.g.
   `seeds=0-7`), recorded in `resolved_config` (FR-SS-3). Submit under the shared `sweep_id`, tagging
   each shard manifest with `cell_id`.
4. Cost admission uses the **shard** count (more jobs than cells) — `check_sweep_admission` /
   `max_jobs` operate on total shard jobs, and `timeout`/`backend`/`accelerators` apply **per shard**.
5. Persist a `SweepPlan`; return the structured plan (§7).

### 6.3 `SweepPlan` model + store path
A Pydantic model persisted at `runs/<sweep_id>/plan.json` (new `store` read/write helpers). Holds,
per cell: `coords`, `cell_id`, `seeds_expected`, the seed-shards, `shard_job_ids`, `results_file`,
`seed_column`, and the deterministic `aggregate_ref` = `runs/<sweep_id>/cells/<cell_id>/<results_file>`
(mirrored to R2 like artifacts). The plan is the single source of truth for status/aggregate/retry;
shard manifests stay independently valid and are cross-referenced.

### 6.4 `Lab.aggregate_sweep(sweep_id)` — idempotent pull reducer
Reads the plan; for each cell, fetches each **succeeded** shard's `results_file` (reusing the
`fetch_artifacts` path), concatenates data rows (keep a single header), normalizes order by
`seed_column` (FR-SS-4: row content unaltered, ordering MAY be normalized). Computes `seeds_present`
from the seed column vs `seeds_expected`; marks the cell `complete` or `incomplete` and names the
**missing seeds** (FR-SS-6/7) — never presents a short aggregate as complete, never discards
recovered seeds. Writes the aggregate to `aggregate_ref` (+ R2 mirror), references all shard
manifests (FR-SS-8). Idempotent + resumable: safe to re-run as more shards finish. Surfaced through
`sweep_status`.

### 6.5 `Lab.retry_sweep(sweep_id)` — resubmit missing shards
For each incomplete cell, resubmit **only** the shards whose seeds are missing (failed/timed-out/never
ran) as fresh jobs under the same `sweep_id`/`cell_id`, update the plan's `shard_job_ids`, then
re-aggregate. Succeeded shards are untouched (FR-SS-7's "MAY offer re-running" convenience).

### 6.6 CLI + MCP surface (thin shells)
`lab sweep` gains `--seeds`, `--shard-size`, `--results-file`, `--seed-column`; new `lab sweep
aggregate <sweep_id>` and `lab sweep retry <sweep_id>` subcommands; `sweep_status` shows the
cell→shard grouping (FR-SS-9). MCP `sweep` gains the equivalent optional params and structured
return; add `sweep_aggregate` / `sweep_retry` tools. No logic in the shells — all in `Lab` (per
CLAUDE.md convention).

## 7. Return shape

```
{ "sweep_id": "...",
  "cells": [ { "coords": {"N": "1000"},
               "cell_id": "...",
               "shard_job_ids": ["...","...","...","..."],
               "aggregate_ref": "<path/uri to merged result>",
               "seeds_expected": 32,
               "seeds_present": 32,        # filled by aggregate; 0 at submit time
               "status": "complete" } ] }  # complete | incomplete | pending
```

At `sweep` (submit) time `aggregate_ref` is the deterministic intended path and `seeds_present` is
not yet known; `aggregate_sweep`/`sweep_status` materialize the aggregate and fill the counts.

## 8. Acceptance mapping

- **AC-1 (equivalence):** `--grid N=1000 --seeds 0-31 --shard-size 8` ⇒ four 8-seed shard jobs;
  `aggregate_sweep` yields the same per-cell `results.csv` as four manual runs concatenated; all four
  shard manifests retained and referenced in the plan.
- **AC-2 (per-shard bound):** a shard overrunning its `timeout` is killed + torn down on its own
  (inherited from per-job behavior); siblings unaffected and still aggregate.
- **AC-3 (partial-failure honesty):** one of four shards forced to fail ⇒ aggregate has 24 seeds,
  cell marked `incomplete`, missing seeds 8–15 named; nothing reports the cell complete.
- **AC-4 (no-op compatibility):** no `seeds`/`shard_size`, or `shard_size >= len(seeds)`, behaves
  exactly as today (one job per cell).
- **AC-5 (provenance):** from the aggregate + referenced shard manifests, every `(cell, seed)` row
  is reproducible — no null `git_commit`, no dangling `diff_ref` (fail-closed holds per shard).

## 9. Risks / notes

- **Seed-column convention** is the only contract surface; defaulted + overridable, documented in the
  guide. The requester's tempotron script must emit a `seed` column (it already produces one row per
  seed today).
- **Cost admission** must count shards, not cells — otherwise a 4× fan-out escapes the budget gate.
- **Plan/manifest consistency:** the plan references shard job_ids; shard manifests remain the
  fail-closed source of truth for code/seed state. The plan is convenience + grouping, never a
  provenance substitute.
- **Deferred path:** `register_sweep` is out of scope for v1 (immediate `sweep` only); a follow-up
  can thread `seeds`/`shard_size` through the deferred registration if needed.
