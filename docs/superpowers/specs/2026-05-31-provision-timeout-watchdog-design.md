# Provision-timeout watchdog (SkyPilot backend)

**Date:** 2026-05-31
**Status:** approved, ready to implement
**Relates to:** FR-C2 (cost-critical teardown), FR-I1 (wall-clock limits), the
dead-host hang observed on the RTX PRO 6000 WS SKU (stuck in Vast "loading",
`sky.stream_and_get` blocks forever).

## Problem

A SkyPilot/Vast host that never finishes bring-up (stuck in "loading") hangs the
supervisor indefinitely. The blocking call is `sky.stream_and_get(request_id)`
at `src/lab/sky_runner.py:103` — it streams provisioning logs until the *remote
job is submitted*, which only happens after the host reaches `UP`. A dead host
never reaches `UP`, so the call never returns. There is no wall-clock guard
around provisioning anywhere today.

Today the only mitigation is manual: a human notices "loading with no `uv sync`
progress", cancels, and resubmits onto a fresh host. We want the lab to self-heal
that case.

## Decisions (locked during brainstorming)

1. **Feature:** a provision-timeout watchdog (not the separate cost-accuracy
   fix, which stays deferred).
2. **On timeout:** abort the stuck launch, run `robust_teardown` to kill the
   half-provisioned host, mark the job **`failed`** with a clear reason. No
   auto-retry, no new terminal state. The user (or `lab wait`'s non-zero exit)
   resubmits.
3. **Config:** default **8 minutes** (constant), overridable per job via a new
   `--provision-timeout` CLI flag / `provision_timeout` submit param. A healthy
   Vast host reaches `UP` in ~2–4m, so 8m clears slow-but-alive hosts while
   catching dead ones.

## Mechanism

Bound the blocking `stream_and_get` with a **daemon thread + `join(timeout)`**:

- Run `sky.stream_and_get(request_id)` in a `threading.Thread(daemon=True)`,
  storing its result/exception in a holder dict.
- Main thread `join`s for the provision-timeout.
- If the thread is still alive after the join → provisioning hung →
  best-effort `sky.api_cancel(request_id)`, then raise `ProvisionTimeout`.
- The dangling daemon thread dies when the supervisor process exits, so it never
  blocks teardown or process exit.

Rejected alternatives:

- **Poll `sky.status(cluster)`** instead of `stream_and_get`: cleaner
  conceptually (launch is async, returns `request_id` immediately) but loses the
  streamed provisioning logs (FR-D1) and is a bigger happy-path rewrite.
- **`ThreadPoolExecutor.future.result(timeout=…)`**: its `shutdown` would try to
  join the stuck worker and hang exit; only works with `shutdown(wait=False)`,
  i.e. a clumsier daemon thread.

The recommended approach is the smallest surgical change and reuses the
`robust_teardown` machinery already hardened for FR-C2.

## Components

### `src/lab/models.py`
Add to `ResourceRequest` (next to `timeout`):
```python
provision_timeout: str | None = None  # max time to reach UP, e.g. "10m" (default 8m)
```

### `src/lab/backends/skypilot.py`
```python
DEFAULT_PROVISION_TIMEOUT_MIN = 8

class ProvisionTimeout(Exception):
    """Raised when a SkyPilot launch does not finish provisioning in time."""

def provision_with_watchdog(sky_mod, request_id, *, timeout_s: float) -> tuple[Any, Any]:
    """Run sky_mod.stream_and_get(request_id) under a wall-clock watchdog.

    Returns (sky_job_id, handle). Raises ProvisionTimeout if provisioning does
    not complete within timeout_s (best-effort sky_mod.api_cancel first). A
    genuine launch error raised by stream_and_get before the timeout is
    re-raised unchanged.
    """
```
Implementation: daemon thread runs `stream_and_get`, holder dict captures
`value`/`error`; `thread.join(timeout_s)`; if `thread.is_alive()` →
best-effort `api_cancel` → raise `ProvisionTimeout`; if `error` present →
re-raise it; else return `value`.

### `src/lab/sky_runner.py`
Inside the existing `try` block, replace:
```python
sky_job_id, handle = sky.stream_and_get(request_id)
```
with:
```python
provision_s = parse_duration(manifest.resources.provision_timeout) \
    or DEFAULT_PROVISION_TIMEOUT_MIN * 60
sky_job_id, handle = provision_with_watchdog(sky, request_id, timeout_s=provision_s)
```
Add a dedicated `except ProvisionTimeout` (before the broad `except Exception`)
that sets `end_reason="provisioning exceeded {provision_s:.0f}s (host never "
"reached UP — likely a dead Vast offer)"`, calls `tear_down_and_record`, and
`return 1`. (Functionally the broad handler would also catch it, but an explicit
branch gives the precise message and intent.)

### `src/lab/cli.py`
Add `--provision-timeout` option to `submit` and `sweep`, threaded into
`ResourceRequest(..., provision_timeout=provision_timeout)`.

### `src/lab/mcp_server.py`
Add `provision_timeout: str | None = None` to the `submit` tool signature and
pass it into the `ResourceRequest`.

## Data flow

`submit --provision-timeout 10m` → `ResourceRequest.provision_timeout="10m"` →
manifest → supervisor resolves `600s` → watchdog joins 600s → on hang:
`api_cancel` + `robust_teardown` + `status=failed` → `lab wait` reports terminal
`failed` and exits non-zero → user's resubmit signal.

## Error handling

- Watchdog wraps **provisioning only**. A *run-time* hang is still governed by
  the existing GNU-`timeout` wall-clock in `build_run_script`. The two timeouts
  are independent.
- `sky.api_cancel` is best-effort (wrapped in try/except). Even if the request
  can't be cancelled, `robust_teardown` (sky.down retry → vast-sdk fallback)
  kills the actual rental — the cost-critical part.
- A genuine launch error raised *before* the timeout is re-raised unchanged, so
  the existing broad `except Exception` path handles it exactly as today.

## Testing

Extend `tests/test_skypilot.py` (no new file):

- `test_provision_watchdog_returns_on_fast_launch`: `_FakeSky.stream_and_get`
  returns immediately → `provision_with_watchdog` returns `(job_id, handle)` and
  `api_cancel` was **not** called.
- `test_provision_watchdog_times_out_on_hang`: `_FakeSky.stream_and_get` sleeps
  longer than a tiny `timeout_s` (e.g. 0.05s) → raises `ProvisionTimeout` and
  `api_cancel` **was** called (record the call on the fake).
- `test_provision_watchdog_reraises_real_error`: `stream_and_get` raises a
  `RuntimeError` quickly → that error propagates unchanged (not swallowed, not
  `ProvisionTimeout`).

No real sleeps of consequence; the hang test uses a ~0.05s timeout against a
fake that sleeps a bit longer.

## Out of scope (deferred)

- Cost-accuracy fix (`dph_total` from the booked offer vs SkyPilot's
  `get_cost()` estimate) — its own task.
- Auto-retry on a fresh host / retry budget.
- A distinct `provision_failed` terminal state.
- Any change to the `local` backend (no provisioning there).
