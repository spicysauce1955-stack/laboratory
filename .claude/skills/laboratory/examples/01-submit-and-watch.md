# Workflow 01 — Submit one job, keep working, get notified

**Goal:** submit an experiment, do *not* block on it, and act on its completion
without polling in a tight loop.

## The pattern

1. `mcp__lab__submit` — get a `job_id` back immediately.
2. Start `uv run lab wait <job_id> --done-file done.json` as a **Claude Code
   background task** (not a foreground `Bash` call — it would block).
3. Continue with other work in the foreground (write code, run analysis, plan
   the next sweep, …).
4. When the background task exits, the harness re-wakes you. Read
   `done.json` (the summary). If `all_terminal` is true and the state is
   `succeeded`, fetch artifacts and read final metrics. Otherwise inspect
   logs / status and decide whether to retry.

## Step-by-step

### 1. Submit

```python
# Tool call
mcp__lab__submit(
    command="python experiments/example_capacity.py",
    backend="local",       # or "skypilot" for Vast.ai
    seed=42,
    timeout="10m",         # wall-clock cap
)
# → {"job_id": "20260528-141233-a1b2c3", "cached": false, "status": "queued"}
```

For a remote GPU run:

```python
mcp__lab__submit(
    command="python experiments/tempotron_capacity.py",
    backend="skypilot",
    accelerators="RTX4090:1",
    timeout="30m",
    seed=42,
)
```

### 2. Launch `lab wait` as a background task

In Claude Code, run this as a **background** bash task (so it runs across
turns and the harness wakes you on exit):

```bash
uv run lab wait 20260528-141233-a1b2c3 --done-file /tmp/lab-done-a1b2c3.json
```

The `--done-file` is a sentinel: when the task exits, `done.json` exists with
the final summary (`{all_terminal, jobs: [{job_id, state, exit_code}, ...]}`).

### 3. Keep working

Don't poll `mcp__lab__status` in a hot loop — the background task is the
notification. Use the wait time productively (other code, other reads).

### 4. On wake-up: read the summary

```python
# Read done.json
import json
summary = json.loads(open("/tmp/lab-done-a1b2c3.json").read())
# {"all_terminal": true,
#  "jobs": [{"job_id": "20260528-...", "state": "succeeded", "exit_code": 0}]}
```

If `state == "succeeded"`:

```python
mcp__lab__fetch_artifacts(job_id="20260528-141233-a1b2c3")
# → {"local_paths": ["runs/20260528-.../output/result.json", ...], "artifacts": [...]}

mcp__lab__metrics(job_id="20260528-141233-a1b2c3")
# → {"series": {"demo_metric": [{"step": 0, "value": 0.31, "wall_time": ...}, ...]}}
```

If `state != "succeeded"`:

```python
mcp__lab__status(job_id="20260528-141233-a1b2c3")  # exit_code, end_reason
mcp__lab__logs(job_id="20260528-141233-a1b2c3", tail=200)  # stdout/stderr tail
```

`end_reason` will be `"timeout"`, `"cancelled"`, `"exit-1"`, etc.

## Notes

- `lab wait` exits non-zero if it gives up on its own `--timeout`. If
  `done.json` has `all_terminal: false`, you timed out — bump the interval or
  check the job state directly.
- For long jobs you can pass `--timeout 2h` to `lab wait` so you eventually
  get woken even if the job hangs.
- The same shape works for `--backend skypilot`; the only difference is the
  fetch may pull from R2 if the local output dir is empty.
