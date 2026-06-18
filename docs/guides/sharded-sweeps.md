# Sharded sweeps (`--seeds` + `--shard-size`)

Run a parameter sweep where each grid cell needs many seeds — the lab splits the seed axis into
independently-bounded **shard jobs**, aggregates the results when they finish, and lets you retry
only the missing shards if some fail.

```bash
uv run lab sweep -c "uv run experiments/example_capacity.py" \
  --grid N=1000,1500 \
  --seeds 0-31 \
  --shard-size 8
```

This creates **8 shards per grid cell** (seeds 0–7, 8–15, 16–23, 24–31), so with 2 grid values
the sweep launches **16 shard jobs** in total.

---

## Experiment contract

Your experiment must honour two conventions so the lab can shard it and aggregate its results.

### 1. Read the seed subset from config

The lab passes each shard's seeds as a comma-delimited list under the config key `seeds`
(e.g. `seeds=0,1,2,3`). Your entrypoint reads that key and iterates over only those seeds:

```bash
# the lab calls your command like this under the hood:
uv run experiments/example_capacity.py seeds=0,1,2,3 N=1000
```

Override the key name with `--seed-column` if your config uses a different name.

### 2. Emit one row per seed into a results file

Each shard must write a row-structured CSV to `$LAB_RUN_DIR/results.csv` (default name) that
includes a column identifying the seed. Example output for a shard with `seeds=0,1,2,3`:

```csv
seed,N,accuracy
0,1000,0.923
1,1000,0.917
2,1000,0.931
3,1000,0.908
```

The `seed` column name and the file path are both overridable:

```bash
uv run lab sweep -c "uv run experiments/example_capacity.py" \
  --grid N=1000,1500 \
  --seeds 0-31 \
  --shard-size 8 \
  --results-file metrics.csv \
  --seed-column run_seed
```

---

## Per-shard resources

`--timeout`, `--backend`, and `--accelerators` apply **per shard** — every shard job gets its own
timeout and teardown. A shard that times out or fails is torn down independently; it does not
cancel other shards.

```bash
uv run lab sweep -c "uv run experiments/example_capacity.py" \
  --grid N=1000,1500 \
  --seeds 0-31 \
  --shard-size 8 \
  --backend cpu \
  --timeout 30m
```

---

## Aggregate and retry lifecycle

### Aggregate

Once shards have finished, concatenate all succeeded shard result files into one per-cell
`results.csv`:

```bash
uv run lab sweep-aggregate <sweep_id>
```

This row-concatenates only the succeeded shards (order normalised, content unaltered) and writes
one aggregate artifact per grid cell, addressable via the sweep plan.

### Retry missing shards

If some shards failed or timed out, resubmit only those:

```bash
uv run lab sweep-retry <sweep_id>
```

This resubmits every shard that has no succeeded result, using the same command, config, and
per-shard resources recorded in the sweep plan. After the new shards finish, run
`sweep-aggregate` again — already-recovered seeds are never discarded.

---

## Partial-failure reporting

A cell is reported `complete` only when **all** expected seeds are present. If any seeds are
missing the cell reports `status: incomplete` and names them:

```jsonc
{
  "cell_id": "N=1000",
  "status": "incomplete",
  "seeds_present": 24,
  "seeds_expected": 32,
  "missing_seeds": [8, 9, 10, 11, 12, 13, 14, 15]
}
```

(Example: a 4-shard cell of 32 seeds 0–31 where shard 1 = seeds 8–15 failed — 8 seeds missing.)

The per-cell status view is returned by `lab sweep-aggregate`, which is **idempotent** — safe to
re-run at any time as shards finish:

```bash
uv run lab sweep-aggregate <sweep_id>
```

This re-aggregates from current shard states and returns the full cell view (including
`seeds_present`, `seeds_expected`, `missing_seeds`, and `status` per cell). Run it whenever you
want to check which seeds have landed so far.

`lab sweep-status <sweep_id>` is a separate command that returns the outcome/cost summary
(job states, preemption count, per-point spend) — it does not show per-cell seed coverage.

Via MCP the equivalent tools are `sweep_aggregate` and `sweep_retry`.

---

## Provenance

Each shard is a normal job with its own fail-closed manifest (the same commit-SHA / `diff_ref`
guarantee described in [`docs/guides/provenance-and-timeouts.md`](provenance-and-timeouts.md)).
The **SweepPlan** stored under the `sweep_id` is the cell→shards map and records every shard job
id, so each `(cell, seed)` row in the aggregate is traceable back to the exact shard job that
produced it (AC-5).

---

## Quick reference

| Flag | Default | Description |
|---|---|---|
| `--seeds` | — | Seed range, e.g. `0-31` or `0,4,8` |
| `--shard-size` | — | Seeds per shard job |
| `--results-file` | `results.csv` | Results file name inside `$LAB_RUN_DIR` |
| `--seed-column` | `seed` | Column identifying the seed in the results file |
| `--timeout` | — | Wall-clock cap **per shard** |
| `--backend` | — | Backend **per shard** (`cpu`, `skypilot`, …) |
| `--accelerators` | — | Accelerator spec **per shard** |

| Command | What it does |
|---|---|
| `lab sweep … --seeds … --shard-size …` | Submit all shard jobs |
| `lab sweep-aggregate <sweep_id>` | Re-aggregate shards and show per-cell `seeds_present`/`seeds_expected`/`missing_seeds` (idempotent) |
| `lab sweep-retry <sweep_id>` | Resubmit missing-seed shards, then re-aggregate |
| `lab sweep-status <sweep_id>` | Show outcome/cost summary (states, preemptions, per-point spend) |
