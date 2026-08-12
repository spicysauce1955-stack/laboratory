# Workflow 03 — Live early-kill (watch metrics, stop if off-track)

**Goal:** submit a long job, poll its metrics live, and cancel it the moment a
divergence criterion fires — saving GPU-hours on a run that's already lost.

## The pattern

1. `mcp__lab__submit` to launch.
2. In a loop (≈ every 10 s):
   - `mcp__lab__metrics(job_id, since_step=last)` — incremental, cheap.
   - Update `last = max(step)` of points seen.
   - Apply the divergence criterion to the latest values.
   - If it fires → `mcp__lab__cancel(job_id)` and break.
3. If the loop ends without a cancel, fall through to a normal
   submit-and-watch wrap-up (background `lab wait` + fetch).

## Step-by-step

### 1. Submit

```python
res = mcp__lab__submit(
    command="python experiments/tempotron_capacity.py",
    backend="skypilot",
    accelerators="RTX4090:1",
    timeout="2h",
    seed=42,
)
job_id = res["job_id"]
```

### 2. The early-kill loop

The `metrics` tool returns

```jsonc
{"series": {"loss": [{"step": 0, "value": 1.23, "wall_time": ...}, ...]}}
```

`since_step` is **exclusive** — pass the last step you've already seen and
you'll only get new points. This makes polling at ~10s essentially free.

```python
import time

last_step = -1
target = 0.05         # divergence criterion: e.g. loss > 5× a known-good baseline
baseline = 0.01

while True:
    status = mcp__lab__status(job_id=job_id)
    if status["state"] in ("succeeded", "failed", "cancelled", "timed_out"):
        break   # job is done — exit loop, fall through to wrap-up

    m = mcp__lab__metrics(job_id=job_id, names=["loss"], since_step=last_step)
    loss_points = m["series"].get("loss", [])
    if loss_points:
        last_step = max(p["step"] for p in loss_points)
        latest_loss = loss_points[-1]["value"]
        if latest_loss > 5 * baseline and last_step > 20:
            mcp__lab__cancel(job_id=job_id)
            print(f"killed {job_id} at step {last_step}: loss={latest_loss:.4f}")
            break

    time.sleep(10)
```

Notes:

- **Skip the first N steps.** Early-training values are noisy; gate the
  criterion behind `last_step > 20` (or similar) so you don't kill on a
  transient first-batch spike.
- **Use `names=[...]`** to filter to just the metrics you care about — keeps
  the payload small even for experiments that log many series.
- **Status check first.** `mcp__lab__metrics` on a terminal job still works,
  but you want to exit the loop as soon as the job is done.

### 3. After the loop (whether cancelled or not)

```python
final = mcp__lab__status(job_id=job_id)
mcp__lab__logs(job_id=job_id, tail=200)         # for debugging
if final["state"] == "succeeded":
    mcp__lab__fetch_artifacts(job_id=job_id)
```

## Notes

- **Polling budget.** `mcp__lab__metrics` reads `runs/<job_id>/metrics.jsonl`
  line-by-line and is tolerant of half-written trailing lines — so a 5–15 s
  cadence is fine even mid-write.
- **Long-running watch.** If the loop itself might span > tens of minutes,
  consider doing the wrap-up via the background `lab wait` pattern from
  `01-submit-and-watch.md` rather than blocking the agent in this loop.
- **Cancel is idempotent on terminal states.** If the job finished a beat
  before you fired the cancel, you get back `{"state": "succeeded"|"failed"|...}`
  unchanged — no error.
