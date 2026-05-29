# Workflow 04 — Recover from a teardown leak (FR-C2)

**Goal:** find and destroy a Vast.ai rental that the lab failed to tear
down. Every hour a leak runs costs money, so this is a stop-the-bleed flow.

## How you find out there's a leak

The lab now signals leaks loudly. You'll see one of:

- **`lab wait` exited 3.** Your background `lab wait` task came back with
  exit-code `3` (distinct from `1` = timeout). Its `done.json` will have a
  non-empty `teardown_leaks` list. The job's manifest is in a terminal
  state, but a paid rental may still be running.
- **`mcp__lab__status` returns `teardown_status: "failed"`.** Same
  diagnosis from the MCP side. The `end_reason` field is annotated with
  an explicit instruction (`"TEARDOWN FAILED … Run lab reconcile --apply"`).
- **Dashboard shows `LEAK` in the `teardown` column.** Visual heads-up.

## The pattern

1. **Dry-run reconcile** to list orphans (no destruction).
2. **Sanity-check** the orphan labels match the leaked job's cluster name.
3. **Apply reconcile** to actually destroy them.
4. **Verify** the next dry-run is empty.

## Step-by-step

### 1. Detect

```python
# After a background `lab wait` task wakes you:
import json
summary = json.loads(open("/tmp/lab-done-XYZ.json").read())
if summary.get("teardown_leaks"):
    print("LEAK:", summary["teardown_leaks"])  # list of job_ids that leaked
```

Or via MCP:

```python
status = mcp__lab__status(job_id="20260529-141233-a1b2c3")
# {"state": "failed", "teardown_status": "failed",
#  "end_reason": "launch error: ... | TEARDOWN FAILED for cluster 'lab-...': ...",
#  ...}
```

### 2. Dry-run reconcile

This is a **CLI-only command** (`lab reconcile` is not exposed via MCP — it
talks to the vastai-sdk directly and is operational/destructive):

```bash
uv run lab reconcile
```

Returns JSON like:

```jsonc
{
  "instances_total": 3,
  "orphans": [
    {"id": 12345678, "label": "sky-lab-20260529-141233-a1b2c3-abcdef"}
  ],
  "destroyed": [],
  "ghosts": [],
  "applied": false
}
```

- `orphans` — Vast.ai rentals labelled `lab-*` with **no matching running
  lab job**. These are the leaks.
- `ghosts` — running lab jobs whose cluster name does **not appear in any
  active Vast rental**. The supervisor probably died; the job's manifest is
  stale. Mark them done with `lab cancel <id>` to clean state.
- `instances_total` — sanity check; matches what `vastai show_instances`
  prints.

**Exit code is `3`** when orphans are found in dry-run — that's the
"action required, run with --apply" signal.

### 3. Apply

Once the orphan list looks right:

```bash
uv run lab reconcile --apply
```

Returns the same shape with `destroyed` filled in:

```jsonc
{
  "instances_total": 3,
  "orphans": [{"id": 12345678, "label": "sky-lab-..."}],
  "destroyed": [12345678],
  "ghosts": [],
  "applied": true
}
```

Each `destroy_instance` call goes directly to Vast.ai (bypasses SkyPilot's
local registry, which may have already lost track).

### 4. Verify

```bash
uv run lab reconcile     # should print "orphans: []", exit 0
```

If `vastai show_instances` is still reporting the rental after a few
seconds, destroy it by hand: `vastai destroy_instance <id>`.

## Why this is safe to run

`lab reconcile --apply` only destroys rentals whose Vast label contains
`lab-` **and** does not match any running lab job. It will not touch:

- Rentals you launched outside the lab (different label prefix).
- Rentals tied to a job whose manifest is in `running` state — those show
  up as `matched`, not `orphans`.

If you have multiple lab repos sharing one Vast account, run `lab reconcile`
from each repo before `--apply` (a rental tied to a job in repo B looks like
an orphan from repo A's perspective).

## Notes

- **The dashboard column makes leaks visible at a glance** — `LEAK` in the
  `teardown` column of `lab dashboard` flags any job whose teardown failed.
- **The lab tries hard before declaring a leak.** `sky.down` retries with
  exponential backoff (~3.5 min total), then falls back to destroying via
  the vastai-sdk directly. `teardown_status: "failed"` means even that
  fallback errored — usually a Vast.ai API outage. Re-running `lab reconcile`
  a few minutes later is the recovery.
- **Tests cover this path** — see `tests/test_skypilot.py::test_robust_teardown_*`
  and `tests/test_core.py::test_reconcile_*` for the exact retry/fallback/
  detection semantics.
