# Spot Instances with Resubmit-on-Preemption — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a job opt into cheaper Vast spot/interruptible GPUs (`--spot`), recover safely when a spot box is reclaimed mid-run, and enforce cumulative cost ceilings — without ever corrupting results or abandoning the billing meter.

**Architecture:** Keep the existing unmanaged `sky.launch` model. Spot is a per-job opt-in that adds an on-demand fallback candidate to the SkyPilot `Resources`. A reclaimed instance is *inferred* as a new terminal state `preempted` (strictly below user-cancel and timeout in precedence), always routed through verified teardown. Registered jobs auto-resubmit per-point up to a retry cap, bounded by a **cumulative** per-job budget. Sweep budgeting is derived from finished-point actuals (no new stateful meter): admission-control up front, stop-launching (never kill) on ceiling.

**Tech Stack:** Python 3.12, Pydantic v2 models, Typer CLI, FastMCP, SkyPilot (Vast cloud), pytest. `ruff` (line 100) + `mypy --strict` on `src/lab`.

**Source design:** `docs/proposals/2026-06-14-spot-instances-feature-proposal.md` (rev. 2).

**Conventions for every task:** CLI and MCP are thin shells over `lab.core.Lab` — never duplicate logic. Run `uv run pytest -q && uv run mypy && uv run ruff check .` green before each commit. `use_spot` defaults `False` everywhere, so the existing suite must stay green at every step.

---

## Phase A — Spot request + launch

### Task 1: Add spot fields to the data model

**Files:**
- Modify: `src/lab/models.py:27-33` (`ResourceRequest`), `:53-56` (`BackendInfo`)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from lab.models import ResourceRequest, BackendInfo


def test_resource_request_spot_defaults_off():
    r = ResourceRequest()
    assert r.use_spot is False
    assert r.spot_fallback is True  # fallback-to-on-demand is default-on


def test_backend_info_records_launched_spot():
    b = BackendInfo(provisioner="skypilot", launched_spot=True)
    assert b.launched_spot is True
    assert BackendInfo(provisioner="local").launched_spot is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -k spot -v`
Expected: FAIL — `ResourceRequest` has no `use_spot`.

- [ ] **Step 3: Implement the fields**

In `ResourceRequest` add:

```python
    use_spot: bool = False  # opt into spot/interruptible instances (skypilot)
    spot_fallback: bool = True  # if spot capacity is unavailable, fall back to on-demand
```

In `BackendInfo` add:

```python
    launched_spot: bool | None = None  # which kind actually launched (None for local/on-demand-only)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -k spot -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/models.py tests/test_models.py
git commit -m "feat(models): spot request fields (use_spot/spot_fallback) + launched_spot"
```

---

### Task 2: Build spot-aware SkyPilot resources (with fallback)

**Files:**
- Modify: `src/lab/backends/skypilot.py:401-426` (`build_task`)
- Test: `tests/test_skypilot_build.py`

- [ ] **Step 1: Write the failing test**

`build_task` calls `import sky`. The test runs in the skypilot extra env; gate it so CI without the extra skips cleanly.

```python
# tests/test_skypilot_build.py
import pytest

sky = pytest.importorskip("sky")
from lab.backends.skypilot import build_task  # noqa: E402
from tests.helpers import make_manifest  # noqa: E402
from pathlib import Path  # noqa: E402


def _spot_flags(manifest):
    task = build_task(manifest, workdir=Path("."))
    return sorted(r.use_spot for r in task.resources)  # task.resources is a set of Resources


def test_on_demand_only_when_spot_off(tmp_path):
    m = make_manifest("j1", "echo hi", accelerators="RTX_4090:1")
    assert _spot_flags(m) == [False]


def test_spot_with_fallback_emits_both_candidates(tmp_path):
    m = make_manifest("j2", "echo hi", accelerators="RTX_4090:1")
    m.resources.use_spot = True  # spot_fallback defaults True
    assert _spot_flags(m) == [False, True]  # spot preferred (cheaper); on-demand fallback


def test_spot_only_when_fallback_disabled(tmp_path):
    m = make_manifest("j3", "echo hi", accelerators="RTX_4090:1")
    m.resources.use_spot = True
    m.resources.spot_fallback = False
    assert _spot_flags(m) == [True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_skypilot_build.py -v`
Expected: FAIL — current `build_task` always emits one on-demand `Resources`.

- [ ] **Step 3: Implement candidate resources**

Replace the `task.set_resources(sky.Resources(...))` block (`skypilot.py:418-425`) with:

```python
    base = dict(
        cloud=sky.Vast(),
        cpus=manifest.resources.cpus,
        memory=manifest.resources.memory,
        accelerators=manifest.resources.accelerators or None,
    )
    if not manifest.resources.use_spot:
        task.set_resources(sky.Resources(**base))
    elif manifest.resources.spot_fallback:
        # Prefer spot (cheaper); SkyPilot's optimizer fails over to on-demand if spot is scarce.
        task.set_resources(
            [sky.Resources(use_spot=True, **base), sky.Resources(use_spot=False, **base)]
        )
    else:
        task.set_resources(sky.Resources(use_spot=True, **base))  # spot-only, no fallback
    return task
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_skypilot_build.py -v` → PASS (or SKIP if `sky` not installed; run in the skypilot extra to actually exercise it).

- [ ] **Step 5: Commit**

```bash
git add src/lab/backends/skypilot.py tests/test_skypilot_build.py
git commit -m "feat(skypilot): spot resources with on-demand fallback / spot-only"
```

---

### Task 3: Thread `--spot` / `--no-fallback` through CLI and MCP

**Files:**
- Modify: `src/lab/cli.py` (the `submit`, `sweep`, `register` commands — build the `ResourceRequest`)
- Modify: `src/lab/mcp_server.py:50-110` (`submit`, `sweep` tools)
- Test: `tests/test_cli_spot.py`

- [ ] **Step 1: Write the failing test (CLI builds the right ResourceRequest)**

```python
# tests/test_cli_spot.py
from typer.testing import CliRunner
from unittest.mock import patch
from lab.cli import app

runner = CliRunner()


def test_submit_spot_flag_sets_resources():
    with patch("lab.cli._make_lab") as make:  # adapt to the actual lab factory in cli.py
        lab = make.return_value
        lab.find_cached.return_value = None
        lab.submit.return_value = "job-x"
        res = runner.invoke(
            app,
            ["submit", "--backend", "skypilot", "--spot", "--no-fallback", "echo hi"],
        )
        assert res.exit_code == 0
        spec = lab.submit.call_args.args[0]
        assert spec.resources.use_spot is True
        assert spec.resources.spot_fallback is False
```

> Note: open `src/lab/cli.py` and match the existing `submit` wiring (the lab-factory name, how `ResourceRequest` is currently assembled from `--cpus/--accelerators/--timeout`). Mirror that exactly; add the two options below.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_spot.py -v`
Expected: FAIL — `--spot` is an unknown option.

- [ ] **Step 3: Implement the CLI options**

In each of `submit`, `sweep`, `register` add Typer options and pass them into the `ResourceRequest`:

```python
    spot: bool = typer.Option(False, "--spot", help="use spot/interruptible instances (skypilot)"),
    no_fallback: bool = typer.Option(
        False, "--no-fallback", "--spot-only",
        help="with --spot, do NOT fall back to on-demand if spot is scarce (wait/skip instead)",
    ),
```

When constructing the `ResourceRequest`, set `use_spot=spot, spot_fallback=not no_fallback`.

In `src/lab/mcp_server.py`, add `use_spot: bool = False` and `spot_fallback: bool = True` params to the `submit` and `sweep` tools and pass them into the `ResourceRequest`/`JobSpec`. Update each docstring with one clause: `use_spot uses spot instances (skypilot); spot_fallback=False makes it spot-only.`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_spot.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/cli.py src/lab/mcp_server.py tests/test_cli_spot.py
git commit -m "feat(cli,mcp): --spot / --no-fallback (use_spot/spot_fallback) on submit/sweep/register"
```

---

## Phase B — Preemption state machine + result integrity

### Task 4: Add the `preempted` terminal state

**Files:**
- Modify: `src/lab/models.py:16-24` (`JobState`)
- Modify: `src/lab/core.py:42` and `src/lab/sky_runner.py:39` and any other `_TERMINAL*` set / `TERMINAL` constant (grep first)
- Test: `tests/test_job_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_job_state.py
from lab.models import JobState
from lab.core import _TERMINAL_STATES  # adapt name if different


def test_preempted_is_a_terminal_state():
    assert JobState.preempted.value == "preempted"
    assert JobState.preempted in _TERMINAL_STATES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_job_state.py -v`
Expected: FAIL — no `JobState.preempted`.

- [ ] **Step 3: Implement**

Add to `JobState`:

```python
    preempted = "preempted"  # spot instance reclaimed mid-run (retryable, not a failure)
```

Then grep and update every terminal set so `preempted` is included:

Run: `grep -rn "JobState.timed_out" src/lab tests` — at each terminal-set definition (`core.py`, `sky_runner.py` `_TERMINAL_NAMES` is the *sky* names map — leave it; add `preempted` to the lab-side `_TERMINAL_STATES`/`TERMINAL` constants and `tests/helpers.py:20` `TERMINAL`). Add `JobState.preempted` alongside `failed`/`cancelled`/`timed_out`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_job_state.py tests/helpers.py -v` and the full suite → PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/models.py src/lab/core.py src/lab/sky_runner.py tests/
git commit -m "feat(models): add terminal JobState.preempted"
```

---

### Task 5: Success sentinel — only mark `succeeded` on a clean exit

**Files:**
- Modify: `src/lab/backends/skypilot.py` (`build_run_script` — write `.lab_success` on exit 0; `_wall_clock_wrap`)
- Modify: `src/lab/sky_runner.py` (the post-run classification that maps the sky status to a `JobState`)
- Test: `tests/test_success_sentinel.py`

**Context:** `build_run_script` already drops a `.lab_timed_out` sentinel on a timeout and `promote_timeout` relabels `failed → timed_out` from it. We add a symmetric **positive** sentinel: `SUCCESS_SENTINEL = ".lab_success"`, written only when the entrypoint exits 0. A run that reports SUCCEEDED to sky but is missing the sentinel is downgraded to `failed` (defensive: a half-flushed success can't be cached).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_success_sentinel.py
from pathlib import Path
from lab.models import JobState
from lab.backends.skypilot import confirm_success, SUCCESS_SENTINEL


def test_success_requires_sentinel(tmp_path: Path):
    # sky said SUCCEEDED but the run dir has no sentinel -> not trusted
    assert confirm_success(JobState.succeeded, tmp_path) is JobState.failed
    (tmp_path / SUCCESS_SENTINEL).write_text("1")
    assert confirm_success(JobState.succeeded, tmp_path) is JobState.succeeded


def test_non_success_unchanged(tmp_path: Path):
    assert confirm_success(JobState.failed, tmp_path) is JobState.failed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_success_sentinel.py -v`
Expected: FAIL — `confirm_success`/`SUCCESS_SENTINEL` undefined.

- [ ] **Step 3: Implement**

In `skypilot.py` near `TIMEOUT_SENTINEL`:

```python
SUCCESS_SENTINEL = ".lab_success"  # written only on a clean exit-0; gates the `succeeded` label


def confirm_success(state: JobState, run_dir: Path) -> JobState:
    """Downgrade succeeded->failed unless the clean-exit sentinel is present (FR-B5 integrity)."""
    if state is JobState.succeeded and not (run_dir / SUCCESS_SENTINEL).exists():
        return JobState.failed
    return state
```

In `build_run_script`, after the entrypoint runs under `timeout`, write the sentinel on success. The script already captures the exit code; add (inside the run script, after the entrypoint line):

```bash
rc=$?
if [ "$rc" -eq 0 ]; then : > "$LAB_RUN_DIR/.lab_success"; fi
exit $rc
```

(Match the existing sentinel-writing style in `_wall_clock_wrap`; the `.lab_timed_out` path must NOT write success.)

In `sky_runner.py`, where `promote_timeout(final, ...)` is applied (`:242`), chain it:

```python
    final = promote_timeout(final, store.output_dir(job_id))  # failed -> timed_out if sentinel
    final = confirm_success(final, store.output_dir(job_id))  # succeeded only if .lab_success present
```

Add the import: `from lab.backends.skypilot import confirm_success`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_success_sentinel.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/backends/skypilot.py src/lab/sky_runner.py tests/test_success_sentinel.py
git commit -m "feat(integrity): .lab_success sentinel gates the succeeded label (no caching half-runs)"
```

---

### Task 6: Classify preemption — with strict precedence

**Files:**
- Create: `src/lab/preemption.py` (pure classifier — unit-testable, no sky calls)
- Modify: `src/lab/sky_runner.py` (call the classifier; record `launched_spot`)
- Test: `tests/test_preemption.py`

**Context:** On the unmanaged path, a reclaimed spot box makes `_wait_terminal` return non-terminal/`FAILED` and the cluster vanishes. We only infer `preempted` when **none** of the higher-precedence outcomes apply. Precedence (highest first): timeout sentinel → `timed_out`; user cancel/early-kill → `cancelled`; spot + cluster-gone-before-finish → `preempted`; else `failed`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preemption.py
from lab.models import JobState
from lab.preemption import classify_terminal


def base(**kw):
    d = dict(sky_state=JobState.failed, timed_out=False, cancel_requested=False,
             use_spot=False, cluster_gone=False, reached_terminal=False)
    d.update(kw)
    return d


def test_timeout_wins_over_everything():
    s = classify_terminal(**base(timed_out=True, use_spot=True, cluster_gone=True,
                                 cancel_requested=True))
    assert s is JobState.timed_out


def test_user_cancel_beats_inferred_preemption():
    s = classify_terminal(**base(cancel_requested=True, use_spot=True, cluster_gone=True))
    assert s is JobState.cancelled


def test_spot_cluster_gone_is_preempted():
    s = classify_terminal(**base(use_spot=True, cluster_gone=True))
    assert s is JobState.preempted


def test_on_demand_cluster_gone_is_failed_not_preempted():
    s = classify_terminal(**base(use_spot=False, cluster_gone=True))
    assert s is JobState.failed


def test_clean_success_passthrough():
    s = classify_terminal(**base(sky_state=JobState.succeeded, reached_terminal=True))
    assert s is JobState.succeeded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preemption.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the classifier**

```python
# src/lab/preemption.py
"""Pure terminal-state classifier for the unmanaged spot path (no cloud calls).

Inferred preemption is *strictly* the lowest-precedence outcome: an explicit timeout or a
user-initiated cancel/early-kill always wins, so a deliberately-killed run is never auto-resubmitted.
"""
from __future__ import annotations

from lab.models import JobState


def classify_terminal(
    *,
    sky_state: JobState,
    timed_out: bool,
    cancel_requested: bool,
    use_spot: bool,
    cluster_gone: bool,
    reached_terminal: bool,
) -> JobState:
    if reached_terminal and sky_state in (JobState.succeeded, JobState.failed):
        # the job actually reported a clean terminal status; trust it (subject to success sentinel)
        if not (timed_out or cancel_requested):
            return sky_state
    if timed_out:
        return JobState.timed_out
    if cancel_requested:
        return JobState.cancelled
    if use_spot and cluster_gone:
        return JobState.preempted
    return JobState.failed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_preemption.py -v` → PASS

- [ ] **Step 5: Wire it into the supervisor + record launched_spot**

In `sky_runner.py`, after `_wait_terminal` returns and before final teardown, compute the inputs and replace the bare `final`:
- `timed_out`: `(store.output_dir(job_id) / TIMEOUT_SENTINEL).exists()`
- `cancel_requested`: check the queue/store cancel marker for this job (reuse the scheduler's `cancel_requested` notion; for a plain submit, this is the `cancel()`-set flag — add a `store.cancel_requested(job_id)` reading a `.cancel` marker written by `Lab.cancel`).
- `use_spot`: `manifest.resources.use_spot`
- `cluster_gone`: probe `sky.status(cluster)` returned no UP record (wrap in try/except; treat exception/empty as gone).
- `reached_terminal`: the sky status name was in `_TERMINAL_NAMES`.

Record which kind launched: after provisioning, set `launched_spot` from the handle's resources (`getattr(handle.launched_resources, "use_spot", None)`) via `store.update_manifest(job_id, backend=BackendInfo(provisioner="skypilot", launched_spot=...))` (preserve existing region/machine_type fields).

- [ ] **Step 6: Run full suite + commit**

Run: `uv run pytest -q && uv run mypy && uv run ruff check .`

```bash
git add src/lab/preemption.py src/lab/sky_runner.py tests/test_preemption.py
git commit -m "feat(spot): infer preempted with cancel/timeout precedence; record launched_spot"
```

---

### Task 7: Verified teardown on every preemption

**Files:**
- Modify: `src/lab/backends/skypilot.py` (`tear_down_and_record` / `robust_teardown`) — add a Vast confirmation probe
- Modify: `src/lab/sky_runner.py` (preempted path always runs verified teardown; never returns "preempted" with an unconfirmed teardown)
- Test: `tests/test_teardown_confirm.py`

**Context:** Preemption is inferred from "the box vanished," so we must *confirm* against Vast that no rental for the job remains before declaring it gone. If unconfirmed, flip `teardown_status="failed"` (→ `lab wait` exit 3) and leave the job for `reconcile`; do **not** report a clean preemption that a watcher would resubmit on top of.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_teardown_confirm.py
from unittest.mock import patch
from lab.backends.skypilot import confirm_no_rental


def test_confirm_true_when_no_matching_rental():
    with patch("lab.backends.skypilot.list_vast_instances", return_value=[]):
        assert confirm_no_rental("lab-j-sky") is True


def test_confirm_false_when_rental_still_present():
    inst = [{"id": 1, "label": "lab-j-sky-abc"}]
    with patch("lab.backends.skypilot.list_vast_instances", return_value=inst), \
         patch("lab.backends.skypilot._instance_label", side_effect=lambda i: i["label"]):
        assert confirm_no_rental("lab-j-sky") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_teardown_confirm.py -v`
Expected: FAIL — `confirm_no_rental` undefined.

- [ ] **Step 3: Implement the confirmation probe**

```python
def confirm_no_rental(cluster: str) -> bool:
    """True iff no Vast rental labelled for this cluster remains (best-effort; False if any match
    or the listing fails — we never claim 'gone' on uncertainty)."""
    try:
        instances = list_vast_instances()
    except Exception:  # noqa: BLE001 — uncertainty must read as "still maybe billing"
        return False
    needle = cluster.lower()
    return not any(needle in _instance_label(inst).lower() for inst in instances)
```

In `sky_runner.py`, on the `preempted` branch: call `tear_down_and_record(...)` then `if not confirm_no_rental(cluster): store.update_manifest(job_id, teardown_status="failed", end_reason="preempted but teardown unconfirmed — see reconcile")`. The supervisor return code follows the existing convention (`2` when teardown leaked). A `preempted` job with `teardown_status="failed"` MUST NOT be eligible for auto-resubmit (enforced in Task 9).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_teardown_confirm.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/backends/skypilot.py src/lab/sky_runner.py tests/test_teardown_confirm.py
git commit -m "feat(spot): confirm teardown against Vast on preemption; flag unconfirmed leaks"
```

---

## Phase C — Cumulative budget + scheduler auto-retry

### Task 8: Cumulative per-job budget + retry-count fields

**Files:**
- Modify: `src/lab/scheduler/models.py:62-79` (`Guardrails`, `Registration`)
- Test: `tests/test_scheduler_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler_models.py
from datetime import datetime, timezone
from lab.scheduler.models import Guardrails, Registration
from lab.models import JobSpec, CodeRef


def test_guardrails_retry_default():
    g = Guardrails(expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert g.max_preempt_retries == 2  # retries PER POINT


def test_registration_tracks_cumulative_and_retries():
    g = Guardrails(expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc), max_cost_usd=5.0)
    r = Registration(reg_id="r", created_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
                     spec=JobSpec(command="x"), guardrails=g, bundle_key="b",
                     code=CodeRef(git_commit="0" * 40))
    assert r.preempt_count == 0
    assert r.cumulative_usd == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scheduler_models.py -v` → FAIL.

- [ ] **Step 3: Implement**

In `Guardrails` add (and update the `max_cost_usd` comment to say *cumulative across all preemption retries*):

```python
    max_cost_usd: float | None = None  # CUMULATIVE ceiling for the logical job (all retries)
    max_preempt_retries: int = 2  # per-point spot-preemption resubmits
```

In `Registration` add:

```python
    preempt_count: int = 0  # spot-preemption resubmits used so far
    cumulative_usd: float = 0.0  # summed actual spend across this job's attempts
```

- [ ] **Step 4: Run test to verify it passes** → `uv run pytest tests/test_scheduler_models.py -v` PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/models.py tests/test_scheduler_models.py
git commit -m "feat(scheduler): cumulative max_cost_usd + max_preempt_retries + per-reg counters"
```

---

### Task 9: Scheduler auto-resubmits preempted registered jobs (budget-bounded)

**Files:**
- Modify: `src/lab/scheduler/tick.py` (the watchdog/sync section that reads launched jobs' manifests)
- Test: `tests/test_scheduler_tick.py` (new test alongside the existing watchdog tests)

**Context:** When a launched registration's manifest is `preempted` (and teardown is confirmed), the scheduler relaunches it from the same bundle, IF `preempt_count < max_preempt_retries` AND `cumulative_usd + next_estimate <= max_cost_usd`. Otherwise it transitions the reg to a terminal state and stops.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler_tick.py  (add near the other watchdog tests)
def test_preempted_registered_job_is_resubmitted_under_budget(tmp_path):
    sched, q = _watchdog_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command="python x.py",
            expires=utc_now() + timedelta(days=1), max_cost=100.0)
    # a launched, spot job that came back preempted with confirmed teardown
    m = make_manifest("j-spot", "python x.py", timeout="1h").model_copy(update={
        "status": JobState.preempted, "ended_at": utc_now(), "registration_id": "reg-a",
        "teardown_status": "succeeded",
        "cost": CostInfo(actual_usd=0.4, hourly_usd=0.4, estimated_usd=0.4),
    })
    sched.store.create(m)
    q.put_entry(q.get_entry("reg-a").model_copy(update={
        "state": RegState.launched, "job_id": "j-spot", "launched_at": utc_now()}))
    relaunched: list[str] = []
    sched._relaunch_preempted = lambda reg: relaunched.append(reg.reg_id)  # type: ignore[method-assign]
    sched.tick()
    e = q.get_entry("reg-a")
    assert relaunched == ["reg-a"]
    assert e.preempt_count == 1
    assert e.cumulative_usd == 0.4


def test_preempted_stops_at_retry_cap(tmp_path):
    sched, q = _watchdog_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command="python x.py",
            expires=utc_now() + timedelta(days=1), max_cost=100.0)
    m = make_manifest("j-spot", "python x.py", timeout="1h").model_copy(update={
        "status": JobState.preempted, "ended_at": utc_now(), "registration_id": "reg-a",
        "teardown_status": "succeeded", "cost": CostInfo(actual_usd=0.4)})
    sched.store.create(m)
    q.put_entry(q.get_entry("reg-a").model_copy(update={
        "state": RegState.launched, "job_id": "j-spot", "launched_at": utc_now(),
        "preempt_count": 2}))  # cap reached
    sched._relaunch_preempted = lambda reg: (_ for _ in ()).throw(AssertionError("must not relaunch"))  # type: ignore[method-assign]
    rep = sched.tick()
    assert q.get_entry("reg-a").state is RegState.failed
    assert rep.synced["reg-a"] == "preempted (retry cap reached)"


def test_preempted_with_failed_teardown_is_not_resubmitted(tmp_path):
    sched, q = _watchdog_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command="python x.py", expires=utc_now() + timedelta(days=1))
    m = make_manifest("j-spot", "python x.py").model_copy(update={
        "status": JobState.preempted, "registration_id": "reg-a",
        "teardown_status": "failed"})  # ambiguous billing — never relaunch
    sched.store.create(m)
    q.put_entry(q.get_entry("reg-a").model_copy(update={
        "state": RegState.launched, "job_id": "j-spot"}))
    sched._relaunch_preempted = lambda reg: (_ for _ in ()).throw(AssertionError("must not relaunch"))  # type: ignore[method-assign]
    rep = sched.tick()
    assert q.get_entry("reg-a").state is RegState.failed
    assert "teardown" in rep.synced["reg-a"]
```

Add `CostInfo` to the test imports from `lab.models`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scheduler_tick.py -k preempt -v` → FAIL.

- [ ] **Step 3: Implement in `tick.py`**

In the sync/watchdog pass over launched regs, when `manifest.status is JobState.preempted`:

```python
        if m.status is JobState.preempted:
            if m.teardown_status == "failed":
                self._transition(reg, RegState.failed,
                                 reason="preempted; teardown unconfirmed (see reconcile)")
                rep.synced[reg.reg_id] = "preempted (teardown unconfirmed)"
                continue
            spent = reg.cumulative_usd + (m.cost.actual_usd if m.cost and m.cost.actual_usd else 0.0)
            nxt = self._estimate_cost(reg) or 0.0
            cap = reg.guardrails.max_cost_usd
            over_budget = cap is not None and spent + nxt > cap
            if reg.preempt_count >= reg.guardrails.max_preempt_retries:
                self._transition(reg, RegState.failed, reason="preempted (retry cap reached)")
                rep.synced[reg.reg_id] = "preempted (retry cap reached)"
            elif over_budget:
                self._transition(reg, RegState.failed,
                                 reason=f"preempted; ${spent:.2f}+${nxt:.2f} would exceed cap ${cap:.2f}")
                rep.synced[reg.reg_id] = "preempted (budget exhausted)"
            else:
                self.queue.put_entry(reg.model_copy(update={
                    "preempt_count": reg.preempt_count + 1, "cumulative_usd": spent,
                    "state": RegState.pending, "job_id": None}))
                self._relaunch_preempted(reg)
                rep.synced[reg.reg_id] = "resubmitted (preempted)"
            continue
```

Add a thin `_relaunch_preempted(self, reg)` that re-runs the existing bundle-launch path (the same one `_launch` uses). If `_launch` already takes a reg and handles extraction, call that; keep it a separate method only so the test can stub it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scheduler_tick.py -k preempt -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/tick.py tests/test_scheduler_tick.py
git commit -m "feat(scheduler): auto-resubmit preempted regs, bounded by retries + cumulative budget"
```

---

### Task 10: Surface preemption in the tick report / status

**Files:**
- Modify: `src/lab/scheduler/models.py:91-101` (`TickReport`) — add a `preempted` list
- Modify: `src/lab/scheduler/tick.py` (append reg_ids to it)
- Test: `tests/test_scheduler_tick.py`

- [ ] **Step 1: Write the failing test**

```python
def test_tick_report_lists_preempted(tmp_path):
    sched, q = _watchdog_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command="python x.py", expires=utc_now() + timedelta(days=1))
    m = make_manifest("j-spot", "python x.py").model_copy(update={
        "status": JobState.preempted, "registration_id": "reg-a", "teardown_status": "succeeded",
        "cost": CostInfo(actual_usd=0.1)})
    sched.store.create(m)
    q.put_entry(q.get_entry("reg-a").model_copy(update={
        "state": RegState.launched, "job_id": "j-spot"}))
    sched._relaunch_preempted = lambda reg: None  # type: ignore[method-assign]
    rep = sched.tick()
    assert "reg-a" in rep.preempted
```

- [ ] **Step 2: Run** → FAIL (`TickReport` has no `preempted`).

- [ ] **Step 3: Implement** — add to `TickReport`:

```python
    preempted: list[str] = Field(default_factory=list)  # regs that hit spot preemption this tick
```

In the Task-9 block, `rep.preempted.append(reg.reg_id)` whenever a preempted manifest is handled (all three branches).

- [ ] **Step 4: Run** → PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/models.py src/lab/scheduler/tick.py tests/test_scheduler_tick.py
git commit -m "feat(scheduler): TickReport.preempted for observability"
```

---

## Phase D — Sweep budget (derived) + visibility

### Task 11: Sweep admission check + `--sweep-max-cost`

**Files:**
- Modify: `src/lab/core.py:164-211` (`sweep`) — accept `sweep_max_cost`, compute worst case, refuse if it won't fit the daily budget
- Modify: `src/lab/cli.py` (`sweep`) and `src/lab/mcp_server.py` (`sweep` tool) — pass `--sweep-max-cost`
- Test: `tests/test_sweep_budget.py`

**Context:** Worst case = `len(points) × per_point_cap`, where `per_point_cap` is `resources` cumulative cap (from `worst_case_cost`-style estimate: `hourly × timeout`, or the explicit cap). No new meter — pure arithmetic checked once, up front.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_budget.py
import pytest
from lab.core import LabError, worst_case_sweep_cost


def test_worst_case_is_points_times_cap():
    assert worst_case_sweep_cost(n_points=48, per_point_cap=0.5) == 24.0


def test_admission_refuses_when_worst_case_exceeds_budget():
    with pytest.raises(LabError, match="worst case"):
        # helper that mirrors the guard inside Lab.sweep
        from lab.core import check_sweep_admission
        check_sweep_admission(n_points=48, per_point_cap=0.5, daily_budget=10.0, committed=0.0)


def test_admission_passes_when_it_fits():
    from lab.core import check_sweep_admission
    check_sweep_admission(n_points=10, per_point_cap=0.5, daily_budget=10.0, committed=0.0)
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** in `core.py`:

```python
def worst_case_sweep_cost(*, n_points: int, per_point_cap: float) -> float:
    return round(n_points * per_point_cap, 6)


def check_sweep_admission(*, n_points: int, per_point_cap: float | None,
                          daily_budget: float | None, committed: float) -> float | None:
    """Refuse a sweep whose worst case won't fit the daily budget. Returns the worst case (or None
    when uncosted). Pure; no state."""
    if per_point_cap is None:
        return None  # uncosted (e.g. local/CPU) — nothing to gate on
    worst = worst_case_sweep_cost(n_points=n_points, per_point_cap=per_point_cap)
    if daily_budget is not None and committed + worst > daily_budget:
        raise LabError(
            f"sweep worst case ${worst:.2f} ({n_points} x ${per_point_cap:.2f}) + "
            f"committed ${committed:.2f} exceeds daily budget ${daily_budget:.2f}; "
            "narrow the grid, lower --max-cost, or raise the budget"
        )
    return worst
```

Call `check_sweep_admission` at the top of `Lab.sweep` (after `expand_grid`), deriving `per_point_cap` from the resources/explicit cap and reading the daily budget + committed via the same source the scheduler uses (pass them in, or read the control config — keep `Lab.sweep` pure by accepting `daily_budget`/`committed` params with `None` defaults so existing callers are unaffected). Store the resolved `sweep_max_cost` (explicit or worst case) for Task 12.

Add `--sweep-max-cost` to the CLI/MCP `sweep`.

- [ ] **Step 4: Run** → PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/core.py src/lab/cli.py src/lab/mcp_server.py tests/test_sweep_budget.py
git commit -m "feat(sweep): worst-case admission check + --sweep-max-cost"
```

---

### Task 12: Stop launching sweep points when the ceiling is hit (never kill)

**Files:**
- Modify: `src/lab/scheduler/tick.py` (when launching points of a sweep, sum finished points' actuals; skip launching further points of that sweep once total ≥ ceiling)
- Test: `tests/test_scheduler_tick.py`

**Context:** The scheduler is the pacing controller for deferred sweeps. Before launching a registration that belongs to a sweep, sum `actual_usd` of that sweep's already-finished points (from their manifests — data already on disk); if it ≥ the sweep ceiling, skip launching and record the reason. Running points are never touched.

- [ ] **Step 1: Write the failing test**

```python
def test_sweep_ceiling_stops_launching_new_points(tmp_path):
    sched, q = _watchdog_sched(tmp_path)
    # two finished points cost 9.0 total; ceiling 8.0 -> the third point must not launch
    for jid, cost in (("p1", 5.0), ("p2", 4.0)):
        sched.store.create(make_manifest(jid, "x").model_copy(update={
            "status": JobState.succeeded, "sweep_id": "sw-1", "cost": CostInfo(actual_usd=cost)}))
    put_reg(q, tmp_path, "reg-c", command="python x.py", expires=utc_now() + timedelta(days=1))
    q.put_entry(q.get_entry("reg-c").model_copy(update={"sweep_id": "sw-1", "sweep_max_cost": 8.0}))
    launched: list[str] = []
    sched._launch = lambda reg, rep: launched.append(reg.reg_id)  # type: ignore[method-assign]
    rep = sched.tick()
    assert launched == []
    assert "sweep budget" in rep.skipped["reg-c"]
```

> Requires `Registration` to carry `sweep_id` and `sweep_max_cost` (add them in this task if not already present; default `None`). Add a `Store` helper `sweep_spend(sweep_id) -> float` that sums finished points' `actual_usd`.

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

Add to `Registration`: `sweep_id: str | None = None`, `sweep_max_cost: float | None = None`.
Add `Store.sweep_spend`:

```python
    def sweep_spend(self, sweep_id: str) -> float:
        total = 0.0
        for m in (self.read_manifest(j) for j in self.list_job_ids()):
            if m.sweep_id == sweep_id and m.cost and m.cost.actual_usd:
                total += m.cost.actual_usd
        return round(total, 6)
```

In the tick launch loop, before `self._launch(reg, rep)`:

```python
            if reg.sweep_id and reg.sweep_max_cost is not None:
                spent = self.store.sweep_spend(reg.sweep_id)
                if spent >= reg.sweep_max_cost:
                    self._skip(reg, rep,
                               f"sweep budget: ${spent:.2f} >= ceiling ${reg.sweep_max_cost:.2f}")
                    continue
```

- [ ] **Step 4: Run** → PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/models.py src/lab/scheduler/tick.py src/lab/store.py tests/test_scheduler_tick.py
git commit -m "feat(sweep): stop launching points once finished-spend hits the ceiling (never kill)"
```

---

### Task 13: Sweep summary with preemption + fallback + per-point spend

**Files:**
- Modify: `src/lab/core.py` (add `sweep_summary(sweep_id) -> dict`)
- Modify: `src/lab/cli.py` (a `lab sweep-status <sweep_id>` command or extend existing sweep output) and `src/lab/mcp_server.py` (expose it)
- Test: `tests/test_sweep_summary.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_summary.py  (uses the local backend store directly via Lab)
def test_sweep_summary_counts(make_lab):  # adapt to your existing Lab test fixture
    lab = make_lab()
    # seed three manifests under one sweep via the store
    from lab.models import CostInfo, JobState, BackendInfo
    for jid, st, spot, cost in [
        ("a", JobState.succeeded, True, 1.0),
        ("b", JobState.preempted, True, 0.2),
        ("c", JobState.succeeded, False, 2.0),  # fell back to on-demand
    ]:
        m = _seed_manifest(lab, jid, sweep_id="sw", status=st, launched_spot=spot, actual=cost)
        lab.store.create(m)
    s = lab.sweep_summary("sw")
    assert s["total"] == 3
    assert s["preempted"] == 1
    assert s["fell_back_to_on_demand"] == 1
    assert round(s["total_usd"], 2) == 3.20
    assert s["per_point"]["b"]["state"] == "preempted"
```

> `_seed_manifest`/`make_lab` mirror existing helpers in the test suite; reuse them.

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** in `core.py`:

```python
    def sweep_summary(self, sweep_id: str) -> dict[str, Any]:
        """Aggregate a sweep's outcomes for trustworthy reporting (preemptions, fallback, spend)."""
        ms = [m for m in self.list_jobs() if m.sweep_id == sweep_id]
        def spend(m: JobManifest) -> float:
            return m.cost.actual_usd if m.cost and m.cost.actual_usd else 0.0
        return {
            "sweep_id": sweep_id,
            "total": len(ms),
            "succeeded": sum(m.status is JobState.succeeded for m in ms),
            "preempted": sum(m.status is JobState.preempted for m in ms),
            "failed": sum(m.status is JobState.failed for m in ms),
            "fell_back_to_on_demand": sum(
                m.resources.use_spot and m.backend.launched_spot is False for m in ms
            ),
            "total_usd": round(sum(spend(m) for m in ms), 6),
            "per_point": {
                m.job_id: {"state": m.status.value, "usd": round(spend(m), 6),
                           "launched_spot": m.backend.launched_spot}
                for m in ms
            },
        }
```

Expose via CLI (`lab sweep-status <sweep_id>` → `_emit(lab.sweep_summary(sweep_id))`) and an MCP `sweep_status` tool returning `dict[str, Any]`.

- [ ] **Step 4: Run** → PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/core.py src/lab/cli.py src/lab/mcp_server.py tests/test_sweep_summary.py
git commit -m "feat(sweep): sweep_summary with preemption/fallback/per-point spend (CLI+MCP)"
```

---

## Final verification

- [ ] Run the whole gate: `uv run pytest -q && uv run mypy && uv run ruff check .` — all green.
- [ ] Update `docs/proposals/2026-06-14-spot-instances-feature-proposal.md` status to "implemented".
- [ ] Update `CLAUDE.md` "Key facts" with one line on spot (`--spot`, cumulative `--max-cost`, `--sweep-max-cost`, `preempted` state).
- [ ] Manual smoke (optional, costs money): `lab submit --backend skypilot --spot --accelerators RTX4090:1 --timeout 20m "uv run python -c 'print(1)'"` then `lab wait` and confirm teardown + `launched_spot` in the manifest.

---

## Self-review notes (author)

- **Spec coverage:** ask#1 cumulative budget → Tasks 8–9, 11–12; ask#2 verified teardown → Task 7; ask#3 atomic cache → Task 5 (success sentinel; existing cache is already `succeeded`-gated, `core.py:153`). Risks: state precedence → Task 6; fallback opt-out+visibility → Tasks 2,3,13; preemption visibility → Tasks 10,13; retry granularity (per-point) → Tasks 8–9; determinism → guaranteed by reuse of the existing bundle-relaunch (same commit+lock+config+seed) in Task 9, documented in the proposal.
- **Cross-task type consistency:** `JobState.preempted` (T4); `ResourceRequest.use_spot/spot_fallback` (T1); `BackendInfo.launched_spot` (T1); `confirm_success`/`SUCCESS_SENTINEL` (T5); `classify_terminal` (T6); `confirm_no_rental` (T7); `Guardrails.max_preempt_retries`, `Registration.preempt_count/cumulative_usd/sweep_id/sweep_max_cost` (T8,T12); `Store.sweep_spend` (T12); `Lab.sweep_summary`, `check_sweep_admission`, `worst_case_sweep_cost` (T11,T13).
- **Open integration points to confirm while implementing (not placeholders — verify against current code):** the exact lab-factory name in `cli.py`; how `Lab.cancel` records a cancel marker the supervisor can read (Task 6 needs `store.cancel_requested(job_id)` — add if absent); whether `_launch` in `tick.py` can be reused directly by `_relaunch_preempted`.
