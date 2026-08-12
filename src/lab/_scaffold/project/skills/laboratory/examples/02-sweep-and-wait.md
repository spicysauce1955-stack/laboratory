# Workflow 02 — Sweep a grid, wait, aggregate

**Goal:** run a Cartesian-product grid of jobs under one `sweep_id`, wait for
all of them as a single background task, then aggregate.

## The pattern

1. `mcp__lab__sweep` with a `grid` dict — get back `sweep_id` + a list of
   `job_ids`, one per grid point.
2. Background `uv run lab wait --sweep <sweep_id> --done-file done.json`.
3. On wake, `mcp__lab__list`, filter to the sweep, `mcp__lab__fetch_artifacts`
   on each, summarize.

## Step-by-step

### 1. Build and submit the sweep

```python
mcp__lab__sweep(
    command="python experiments/tempotron_capacity.py",
    grid={"seed": [1, 2, 3], "K": [100, 200, 500]},   # 3 × 3 = 9 jobs
    backend="skypilot",
    accelerators="RTX4090:1",
    timeout="20m",
)
# → {"sweep_id": "sweep-20260528-141900-9f00",
#    "job_ids": ["20260528-141900-...", "20260528-141900-...", ... (9 ids)]}
```

Notes on the grid:

- **Cartesian product, insertion order.** `{"seed": [1,2,3], "K": [100,200]}`
  → 6 jobs in that order.
- **Values are strings on argv.** `K=100` becomes the literal `K=100` token on
  the experiment's argv; Hydra/typer/argparse coerces to int. The lab does not
  parse types.
- **`seed` is special.** If the grid contains a `seed` key, each job's recorded
  manifest seed comes from the grid (not the `seed=` kwarg, which becomes the
  default for grids that don't include it).
- **`max_jobs=256` cap.** A grid that would expand beyond 256 jobs is rejected.

### 2. Wait for all of them, in the background

```bash
uv run lab wait --sweep sweep-20260528-141900-9f00 \
                --done-file /tmp/lab-done-sweep-9f00.json \
                --timeout 7200   # SECONDS (= 2h). `lab wait --timeout` is NOT a duration string.
```

Run this as a Claude Code background task. Sweeps with skypilot run in
parallel (subject to Vast.ai availability), so total wall-clock is roughly
the **slowest** job, not the sum.

### 3. On wake-up: aggregate

```python
import json
summary = json.loads(open("/tmp/lab-done-sweep-9f00.json").read())
# summary["jobs"] is a list of {"job_id", "state", "exit_code"} for each.

succeeded = [j for j in summary["jobs"] if j["state"] == "succeeded"]
failed    = [j for j in summary["jobs"] if j["state"] != "succeeded"]
```

To pull every succeeded job's artifacts + metrics:

```python
for j in succeeded:
    mcp__lab__fetch_artifacts(job_id=j["job_id"])
    mcp__lab__metrics(job_id=j["job_id"])
```

To find the sweep again later (or in a new session):

```python
mcp__lab__list()
# Filter jobs whose sweep_id == "sweep-20260528-141900-9f00".
```

## Notes

- **Resubmit just the failures.** If 8/9 succeeded and 1 host died on Vast,
  re-submit the failed grid point as a single `mcp__lab__submit` rather than
  re-running the whole sweep.
- **Cache + sweep.** If the underlying tree was clean and a grid point matches
  a previously-succeeded job's `(commit, command, config, seed)`, you can
  re-issue the sweep with the same command but pass each grid point through
  `mcp__lab__submit(cache=true)` instead — the cached job_id is returned for
  hits. The bulk `sweep` tool does not currently look up the cache per-point;
  use it for fresh sweeps.
- **Per-job extra deps apply to every point.** `with_pkg=["scipy"]` on
  `mcp__lab__sweep` wraps every grid point in `uv run --with scipy ...`.
