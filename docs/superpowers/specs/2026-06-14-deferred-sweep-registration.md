# Follow-up Spec: Deferred-sweep registration path

**Status:** proposed (2026-06-14) · **Size:** small follow-up to the spot-instances feature
**Depends on:** spot-instances (merged) — `Registration.sweep_id` / `sweep_max_cost`, the scheduler
sweep-ceiling guard (`_evaluate_and_launch`), `JobStore.sweep_spend`, and `Lab.sweep_summary`
already exist and are honored.

## Why

The spot-instances feature added the scheduler-side machinery to run a sweep *deferred* and stop
launching its points once finished-spend hits a ceiling (never killing a running point). But **no
user-facing command populates `Registration.sweep_id` / `sweep_max_cost`**, so that machinery is
currently unreachable:

- `Lab.sweep` submits all points *immediately* (gets the up-front worst-case admission check, but
  not the during-sweep ceiling, and ignores triggers/windows).
- `lab register` (`scheduler.register.register`) registers exactly **one** job.

The gap: a way to register a *grid* as N deferred registrations sharing a `sweep_id` and a
`sweep_max_cost`, so the always-on scheduler launches them over time (subject to triggers,
`max_concurrent`, and the sweep ceiling) — the "fire-and-forget overnight spot sweep" path.

## Goal

Add `register_sweep` (core) + `lab register-sweep` (CLI) + `register_sweep` (MCP tool) that expands a
grid into N registrations sharing one `sweep_id` and one resolved `sweep_max_cost`, with shared
triggers/guardrails, from a single code bundle. The scheduler then paces them and enforces the
ceiling already built in Task 12.

## Required changes

### 1. Propagate `sweep_id` at launch (correctness prerequisite)
`scheduler/tick.py::_launch` currently calls:
```python
job_id = lab.submit(reg.spec, code=reg.code, registration_id=reg.reg_id)
```
It must also pass the sweep id so the launched manifest is attributable to the sweep (without this,
`sweep_spend` and `sweep_summary` see `manifest.sweep_id is None` and aggregate nothing):
```python
job_id = lab.submit(reg.spec, code=reg.code, registration_id=reg.reg_id, sweep_id=reg.sweep_id)
```
(`Lab.submit` already accepts `sweep_id`.) This is the single most important change — verify with a
test that a scheduler-launched sweep point's manifest has the right `sweep_id`.

### 2. `register_sweep` core function (`scheduler/register.py`)
Signature mirrors `register` but takes a grid and a sweep cost cap:
```python
def register_sweep(
    repo: Path, queue: QueueStore, base_command: str, grid: dict[str, list[Any]],
    *, resources: ResourceRequest, triggers: Triggers, guardrails: Guardrails,
    seed: int | None = None, sweep_max_cost: float | None = None,
    daily_budget: float | None = None, committed: float = 0.0, max_jobs: int = 256,
) -> tuple[str, list[Registration]]:
```
Behavior:
- Expand the grid with `lab.core.expand_grid`; refuse if `> max_jobs` (reuse the existing guard /
  error style). Build each point's command + `resolved_config` + per-point seed exactly as
  `Lab.sweep` does (shell-quoted `key=value` overrides; a `seed` grid key sets per-point seed) —
  factor the point→JobSpec logic out of `Lab.sweep` into a shared helper so the two paths can't drift.
- Generate one `sweep_id = f"sweep-{...}"`.
- **Admission check up front:** derive `per_point_cap` (the per-point cumulative cap = explicit
  `guardrails.max_cost_usd`, or `worst_case_cost(triggers, resources)` from `max_hourly_usd × timeout`
  if set, else None) and call `check_sweep_admission(n_points, per_point_cap, daily_budget, committed)`;
  it raises `LabError` if the worst case won't fit the daily budget. The resolved ceiling is
  `sweep_max_cost` if the user set one, else the returned worst case (the default = worst case, which
  doubles as a leak alarm on the per-point cap — see the cost-safety philosophy).
- Create N `Registration`s, each with: the same `sweep_id`, `sweep_max_cost=<resolved ceiling>`, the
  same `triggers` and `guardrails` (so per-point `max_cost_usd` still applies), `code` from the
  bundle, and the point's JobSpec.
- **Bundle once, share it (see Decision A).** Write ordering keeps the integrity guarantee: bundle
  first, entries last.
- Return `(sweep_id, registrations)`.

### 3. CLI `lab register-sweep` and MCP `register_sweep`
Thin shells over `register_sweep`, mirroring the existing `register` command's trigger/guardrail
options (`--tonight`/window, `--not-before`, `--max-hourly`, `--expires`, `--max-cost`, `--spot`/
`--no-fallback`, etc.) plus `--grid` (repeatable) and `--sweep-max-cost`. Emit `{sweep_id, count,
reg_ids}`. No logic beyond building args and calling core.

## Decisions

**Decision A — bundle sharing (recommended: share by bundle key).** `_launch` currently fetches the
bundle by `reg.reg_id`. For a 48-point sweep, storing 48 identical tarballs is wasteful. Recommended:
bundle once under the `sweep_id`, set every point's `bundle_key` to that shared key, and change
`_launch` to fetch by `reg.bundle_key` (already a field on `Registration`) instead of `reg.reg_id`.
Small, localized change; makes sharing natural. **Lifecycle note:** a shared bundle must not be
deleted until *all* points of the sweep are terminal — if any bundle GC exists, scope it by
"no non-terminal reg references this bundle_key." *Fallback (zero `_launch` change):* copy the same
tarball under each point's `reg_id`; simpler but N copies. Pick the shared-key refactor unless it
proves to ripple.

**Decision B — points are independent.** No `after` dependencies between points; the scheduler's
`max_concurrent` provides natural pacing and is what makes the during-sweep ceiling meaningful.

**Decision C — shared guardrails.** Each point keeps its own per-point cumulative `max_cost_usd`
(retry budget from spot-instances) AND participates in the sweep-wide `sweep_max_cost` ceiling. Both
apply; the sweep ceiling defaults to the worst case so it only fires as a leak alarm.

## Testing
- `_launch` propagates `sweep_id` → a scheduler-launched point's manifest has the sweep id (and thus
  shows up in `sweep_spend`/`sweep_summary`).
- `register_sweep` creates N entries sharing one `sweep_id` + `sweep_max_cost`; respects `max_jobs`;
  raises `LabError` when the worst case exceeds the daily budget (admission).
- Shared point→JobSpec helper produces identical commands/configs to `Lab.sweep` (guard against drift).
- End-to-end (local backend, fake clock): register a 3-point sweep with `sweep_max_cost` below the
  worst case; let finished points accrue `actual_usd`; assert the scheduler stops launching the
  remaining point with the "sweep budget" skip reason (this exercises Task 12 through the real path).
- Bundle sharing (Decision A): all points launch from the one shared bundle.

## Out of scope
- Checkpoint-restart (still future).
- Per-point heterogeneous resources/triggers (one shared config per sweep here).
- A sweep-wide *cancel/hold* convenience (cancelling points individually via existing `queue` cmds
  still works); a `queue cancel --sweep <id>` could be a later nicety.
