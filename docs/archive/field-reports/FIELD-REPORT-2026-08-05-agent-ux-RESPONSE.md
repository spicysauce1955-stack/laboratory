# Response — agent-UX field report, 2026-08-05

*Archived point-in-time record: the lab author's reply of 2026-08-06 to the
[2026-08-05 field report](FIELD-REPORT-2026-08-05-agent-ux.md), describing v0.2.0. For current
information see [`README.md`](../../../README.md),
[`docs/guides/getting-started.md`](../../guides/getting-started.md),
[`docs/COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`CHANGELOG.md`](../../../CHANGELOG.md).*

**To:** the agent (and any future operator) who filed
[`FIELD-REPORT-2026-08-05-agent-ux.md`](FIELD-REPORT-2026-08-05-agent-ux.md)
**Status:** all 7 issues addressed; shipped to `main` in **v0.2.0** (PR #7, 2026-08-05)
**Verified by:** 477-test suite (+70 new, written test-first), `ruff` + `mypy --strict` clean,
live local-backend end-to-end of #1 and #3, plus an independent 5-agent code review whose
findings were fixed before merge.

Thank you for this report. It was specific, evidenced, and correctly ranked — #1 and #2 were
treated as correctness bugs, not UX polish. Below: what changed per issue, and what you should
do differently in your next session.

---

## #1 Unconsumed `key=value` overrides — FIXED (fail-closed)

Your `patience=0` scenario can no longer produce a silent false negative. Two layers:

1. **At runtime (best case):** entrypoints adopt a one-import helper —

   ```python
   from lab.experiment import get_overrides
   ov = get_overrides(known={"patience", "optimizer", "lr_schedule", ...})
   ```

   It parses the argv overrides, writes `$LAB_RUN_DIR/effective_config.json`, and **exits
   non-zero naming any unknown key**. Your job would have died in second one with
   `unknown config override(s): ['patience', ...]` instead of returning `P_solve = 0.0`.

2. **At the verdict (backstop):** the store compares argv-passed overrides against
   `effective_config.json` on every terminal transition. A *succeeded* job with unconsumed
   keys **flips to `failed`** with `end_reason: "unconsumed config keys: [...]"`.
   Opt-out: `--allow-unknown-config` / `allow_unknown_config=true`.

**What you should do:** prefer entrypoints that call `get_overrides`. For legacy scripts
(none of `experiments/v*.py` are migrated yet), run **`lab lint -c "python experiments/v9_capacity.py" -g patience=0,1500`**
before submitting — it greps the script for each key and exits 1 on misses. Your stale-script
case (`v9` vs `v3`) is exactly what it catches.

**Manifest changes:** `config_effective` (what applied; `null` = legacy entrypoint) and
`unconsumed_config` now sit beside `run.resolved_config` (what was asked). Where they differ,
that difference is the diagnosis — your issue #6, same fix.

## #2 `sweep-aggregate` discarding paid-for partial rows — FIXED (default-on)

The `succeeded`-only filter is gone. Any **terminal** shard with a readable results file
contributes; your 108 recovered rows would have aggregated with zero hand-rolling. Exactly as
you proposed:

- every row carries a **`_shard_status`** column (`succeeded` / `timed_out` / ...),
- the cell view adds **`seeds_partial`** next to `seeds_present` / `missing_seeds`,
- **`--strict`** (CLI) / `strict=true` (MCP) restores succeeded-only.

Also beyond the report: `sweep-retry` now resubmits **only the missing seeds** of a partially
recovered shard (narrowed shard jobs), and duplicate seeds across a partial shard + its retry
resolve automatically (succeeded rows win). A duplicate across two *succeeded* shards still
fails loudly — that's a sharding-contract violation, not recovery.

**Watch out:** aggregate CSVs have one new trailing column; `seeds_present` now includes
partially-recovered seeds (use `seeds_partial` to disambiguate); running/queued shards are
still never read (their file is mid-heartbeat).

## #3 `lab wait` blackout windows — FIXED (all three proposals)

- **`--fail-fast`**: returns the moment any job hits `failed`/`timed_out` — your 20 min 42 s
  window becomes one poll interval (~10 s). Exit code **4**; the offender is listed first;
  `pending` names the survivors (still running, still billing — nothing is auto-cancelled).
  `preempted`/`cancelled` do **not** trip it. One precedence rule: a *confirmed teardown
  leak* still exits **3** even under fail-fast — the money alarm outranks everything.
- **Incremental `--done-file`**: atomically rewritten after *every* job's terminal
  transition (full summary JSON with `pending`), so your watcher can act mid-wait without
  waiting for process exit. Always ends with the final verdict.
- **Durations everywhere**: `lab wait --timeout 30m` now works (bare seconds still do).

The MCP `wait` tool mirrors all of this (`fail_fast=`, `timeout="30m"`, summary carries
`failed_fast` + `pending`), so one background `lab wait` per job is no longer necessary.

## #4 Submission stampede — FIXED (retry + stagger)

- Sweeps **stagger** remote submits: `LAB_SUBMIT_STAGGER_S` (default 1.5 s, `0` disables).
- The supervisor **retries `sky.launch`** with backoff+jitter (`LAB_LAUNCH_RETRIES`, default
  3) — but only for connection failures against the *local* API server (`127.0.0.1`);
  cloud-endpoint errors still fail loud.
- On exhaustion, `end_reason` is prefixed **`transient:`** — your requested classification.
  Anything with that prefix is safe to resubmit blindly.

Your five $0 casualties would have retried through the cold start and launched.

## #5 `runs/` unreachable from the analysis repo — FIXED (`lab export`)

```
lab export <job_id | sweep-id> --to ../research-repo/data/lab-bundle [--logs]
```

(also MCP `export`.) Produces the committable subset you reconstructed by hand: per-job
`manifest.json`, `resolved_config.json`, `code_diff.tar.gz`, figure/table artifacts under a
32 MB cap, sweep `plan.json` + per-cell aggregates, and an **`index.json`** tying every file
to commit / seed / state / spend. Blobs are excluded but **listed under `skipped`** — an
auditor can always see what was left behind. Your 50-manifest hand recovery is one command.

## #6 Manifests recording config that had no effect — FIXED

Covered by #1's `config_effective` / `unconsumed_config` split (recorded on every terminal
transition, not just success). Your `n_restarts: 10 / anchor_K: []` case surfaces as
`n_restarts` present in `resolved_config` but absent from `config_effective` — visible in
`lab status` provenance without reading output columns for `-1` sentinels.

## #7 Smaller frictions

| Report item | Status |
|---|---|
| No mid-run cost | `lab status` → `estimated_running_usd` (hourly × elapsed while running) |
| Running jobs opaque | `lab status` → `last_log_line` + `last_log_at` (progressing vs. wedged) |
| `--timeout` unit inconsistency | duration strings accepted on `wait` (bare seconds still work) |
| `sweep` requires `--grid` | optional when `--seeds` is given; no more `-g Keff_list=8` |
| DO backend unauthenticated | environment issue, not code — unchanged (same bucket as GCP auth) |

---

## Behavior changes to expect (breaking-ish, intentional)

1. Sweeping a key a handshake-enabled entrypoint doesn't consume now **fails those jobs**.
   That is the feature. Legacy entrypoints are unaffected until migrated.
2. Aggregate CSVs gain `_shard_status`; `seeds_present` includes partial seeds.
3. `lab wait` has a new exit code **4** (fail-fast), and exit 3 takes precedence over it.
4. Remote sweeps take ~1.5 s × (n−1) longer to submit (stagger). Set
   `LAB_SUBMIT_STAGGER_S=0` to disable.

## Not done (honest gaps)

- `experiments/*.py` are **not migrated** to `get_overrides` — `lab lint` is the bridge.
- `--events-file` JSONL stream, `--since` export filter, and the exit-1-vs-3 precedence
  revisit remain follow-ons (tracked in the PR #7 description).

Every fix above carries tests named after your scenarios (`tests/test_effective_config.py`,
`test_aggregate_merge.py`, `test_cli_wait.py`, `test_sky_launch_retry.py`, `test_export.py`).
If a next session finds these guarantees not holding as described, file another report like
this one — it worked.

---

## Addendum (2026-08-06): §2 unblocked in v0.2.1

Your verification report's §2 finding — the duplicate-seed guard rejecting this project's
one-row-per-(seed, α) layout — is resolved with your preferred **option 1 (composite row key)**:

```
lab sweep -c "..." --seeds 0-47 --shard-size 6 --row-key seed,alpha
```

`--row-key` (MCP: `row_key="seed,alpha"`) declares the columns that identify a result row; it is
stored on the sweep plan, and every duplicate rule — the within-one-file raise, the
succeeded+succeeded contract tripwire, and the partial/retry winner resolution — is judged on the
full key. The default remains the seed column alone, so nothing changes for one-row-per-seed
sweeps. It must include the seed column (`seeds_present`/`seeds_partial`/`missing_seeds` are
still seed-accounted). Replaying your `sweep-20260804-124022-0a188f` layout is covered by tests
mirroring your exact case, including a timed-out shard's (100, 2.72) row surviving a succeeded
retry of (100, 2.7).

Your minor finding is also fixed: `fetch_artifacts` with `LAB_R2_ENDPOINT` set but no `r2` extra
now warns and proceeds with local state instead of raising `ModuleNotFoundError`.

Shipped as **v0.2.1** (PR #8). With this, hand-aggregation of the headline data should no longer
be necessary — if the real sweep still resists mechanical aggregation, that's a new report.

**Replay proof (2026-08-06, v0.2.2):** `sweep-aggregate` also gained an aggregate-time
`--row-key` override for plans that predate v0.2.1 — and we ran it against your real sweep:

```
lab sweep-aggregate sweep-20260804-124022-0a188f --row-key seed,alpha
→ 108 rows aggregated (your exact hand-stitched count), 30/48 seeds present (all partial,
  stamped _shard_status=timed_out), missing seeds 124-141 named, status: incomplete
```

The mechanical aggregate now lives at `runs/sweep-20260804-124022-0a188f/cells/efb125e7/results.csv`
with the row key persisted on the plan. Hand-aggregation of this data is no longer necessary.
