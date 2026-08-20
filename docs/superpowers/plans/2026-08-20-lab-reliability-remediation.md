# Lab Reliability Remediation — Register & Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the cost-safety and observability defects that let `lab reconcile --apply` destroy seven running jobs on 2026-08-20, and that leave a job whose machine disappears sitting `running` for its full timeout.

**Architecture:** Every defect here is the same shape — **a definitive negative swallowed into an indeterminate one**. SkyPilot says "cluster does not exist"; the supervisor treats it as "not finished yet". The job store says "I have no record"; reconcile treats it as "therefore it is a leak". A destroy raises; the report says nothing was destroyed. The fix in each case is to classify the answer and act on the definitive ones, while never letting an *unknown* masquerade as either a success or a failure.

**Tech Stack:** Python 3.12, uv, SkyPilot 0.12.3 (client) / 0.13.0 (server — see R4), pytest, ruff (line length 100), mypy --strict.

**Spec:** `FIELD-REPORT-2026-08-20-do-backend-reliability.md` (as corrected), plus the live evidence recorded inline below.

## Global Constraints

- `ruff` line length **100**; `uv run ruff check src/lab tests` must pass. **`ruff format` is not used** — decided 2026-08-20, see R12. Never reformat.
- `uv run mypy --strict src/lab` must pass.
- `uv run pytest -m "not packaging"` must pass; `uv run pytest -m packaging` is the installed-wheel guard.
- CLI and MCP are thin shells over `lab.core.Lab` — never duplicate logic between them.
- **stdout carries only JSON**; all diagnostics go to stderr.
- `lab wait` exit codes `3` (teardown leak) and `4` (fail-fast / refused) are contract. `reconcile` now also uses `5` (destroy unconfirmed). Task 4 adds `lab wait` exit **`6`** (teardown outcome unknown) — decided 2026-08-20.
- Secrets never in repo/manifest/logs (FR-J1).

---

## Part 1 — Consolidated register

Status key: **FIXED** (done, on branch `fix/reconcile-attribution-safety`) · **OPEN** · **RETRACTED** · **RECLASSIFIED**.

### Fixed this session

| # | Finding | Sev | Evidence |
|---|---|---|---|
| **R1** | `reconcile`'s orphan test joined **machine-global** cloud state (`sky.status()` → `~/.sky/state.db`) against the **current repo's** job store (`JobStore(repo_root()/"runs")`). Every other lab project's live clusters qualified as orphans; `--apply` destroyed them. | **critical** | 7 DO droplets destroyed 08:20:32–08:23:03 UTC (DO action log), all belonging to running `tempotron-capacity` jobs |
| **R2** | `reconcile --apply` exited **0** when every destroy errored. | high | live run printed `sky_destroyed: []`, `EXIT=0`, while 7 machines died |
| **R3** | Destroy failures were `print`ed to **stdout**, corrupting the JSON-only stdout contract. | medium | `[lab] reconcile sky.down … failed:` appeared before the JSON payload |
| **R4** | SkyPilot client 0.12.3 vs API server 0.13.0 makes a **successful** `sky.down` undecodable — `sky.get` unpickles the server-side entrypoint before reading the result, so a completed teardown surfaces as `AttributeError: Can't get attribute 'user_initiated_down'`. Affects `robust_teardown`, not just reconcile. | **critical** | 7 "failed" destroys vs 7 `destroy droplet … completed` in the DO action log |
| **R5** | GCP truncates cluster names to 35 chars. `lab-<job_id>` fit by *exactly zero* margin; adding the project slug pushed live instances out of match range, so `reconcile --apply` would have destroyed running GCP boxes — the incident recreated by its own fix. | high | `make_cluster_name_on_cloud` → `lab-laboratory-20260820-ef-3dd12990` |
| **R6** | After the cluster rename, a job **launched under the old name and still running** no longer matched `cluster_name_for(job_id)`. | high | found via `test_core.py::test_reconcile_finds_orphans_dry_run` flagging a live job's rental |
| **R7** | Reconcile tests silently depended on whether a real sky API server was running locally, and at which version. | medium | full suite behaviour changed when the skew guard landed |

**Fixes delivered:** ownership proved via `lab.attribution` (user-global `~/.lab/jobs/index.jsonl` + project-tagged event-ledger fallback); project slug stamped into cluster names (`lab-<slug>-<job_id>`) and GCP instance labels; `other_projects` / `unattributed` reported and **never** destroyed; `destroy_outcomes` + CLI **exit 5**; version-skew hard block on `--apply` (exit 4, structured JSON); `gcp_name_matches` for truncated names; legacy alias protection; hermetic `sky_versions` test fixture.

**Verification:** 982 passed / 7 skipped, packaging guard passes, ruff + mypy --strict clean. Live: dry-run reconcile now reports 7 running `tempotron-capacity` jobs under `other_projects` with `sky_orphans: []` and exit 0.

### Open — found this session

| # | Finding | Sev |
|---|---|---|
| **R8** | **The supervisor ignores "cluster does not exist" and waits out the whole timeout.** `_wait_terminal` (`sky_runner.py:89-97`) catches every poll exception into `print(...)` and keeps looping to `max_wait = timeout + 300`. Log evidence: **65 consecutive** `[lab] queue poll error: Cluster 'lab-…' does not exist.` A job whose machine dies sits `running` for up to 125 minutes. | **high** |
| **R9** | **`lab cancel` marks the job terminal *before* tearing down, then blocks in a multi-minute retry ladder.** If the caller times out, the manifest reads `cancelled` with `teardown_status: None` — a possibly-billing machine with no leak signal. | **high** |
| **R10** | No third teardown state. `succeeded`/`failed` cannot express "the destroy's outcome is unreadable" (R4's case), so FR-C2's money alarm is either a false positive or a false negative. | medium |
| **R11** | `_instance_label` concatenates four candidate fields with spaces, emitting trailing whitespace; Vast/DO matching is a loose `in` substring test. | low |
| **R12** | `ruff format --check` fails on **73 of 129** files; CI runs only `ruff check`. The formatter is decorative. | low |

### Field-report items, re-adjudicated

| # | Original | Verdict |
|---|---|---|
| **F1** | "DO supervisors die silently, ~40% of launches" | **RECLASSIFIED.** True rate over the full ledger is **15/52 = 29%** of `supervisor/run` calls never closing. But they do not die spontaneously — they **spin** (R8) until something SIGTERMs them, and SIGTERM's default disposition skips the `except BaseException` close. The proposed **SIGTERM handler is still correct and wanted** (→ R13); the diagnosis was not. Note `cli/cancel` is **7/8 = 88%** never-closed and `cli/wait` **2/2** — those are *caller* timeouts (R9), not supervisors, and were being counted as the same phenomenon. |
| **F2** | No DO-direct teardown fallback (Vast and GCP have one) | **OPEN, confirmed** (→ R14). Partially masked by R4, but real. |
| **F3** | Dead-supervisor self-heal is pull-based only | **NARROWED.** Wrong that reconcile misses it — dry-run reconcile already computes and reports `unsupervised`. Right that it only *reports*: remediation still requires a `lab status <job_id>` on that specific id (→ R15). |
| **F4** | `pid_alive()` is bare `os.kill(pid, 0)`, blind to PID reuse | **OPEN, confirmed** (→ R16). |
| **F5** | Supervisor `Popen` exit status never collected | **OPEN, confirmed** (→ R17). |
| **F6** | "`lab <cmd> --help` exits non-deterministically 0 or 1" | **SOLVED and RECLASSIFIED.** Fully deterministic: unpiped `lab submit --help` → **0**, twenty times out of twenty; `lab submit --help \| head -3` → **1**, five times out of five. It is an unhandled `BrokenPipeError` on a closed stdout pipe, not non-determinism. The ledger's mixed statuses record whether the caller piped (→ R18). |
| **F7** | `lab kill` isn't a command and offers no suggestion | **OPEN, confirmed.** Exits 2, prints `No such command 'kill'.` with no "did you mean" (→ R19). |
| **F8** | history/report default to the current project | **SPLIT.** Correct and intentional for `history`/`report`. For `reconcile --apply` it was the critical bug = R1, now fixed. |
| — | **"Live leak — act on this first"** | **RETRACTED.** The 8 listed resources were live `tempotron-capacity` jobs. Acting on it destroyed seven of them. |

### Process findings

- **P1 — A tool's own classification is not evidence.** The report's highest-priority action item was a false positive produced by the very defect the report was investigating, and it was acted on without cross-checking the provider. **Rule: before any destructive action, verify against the provider's own API** (`doctl compute droplet list`, `doctl compute action list`, `gcloud compute instances list`) and check `ps aux | grep lab.sky_runner` for live supervisors on the box.
- **P2 — The user's external watchdogs are the bug's shadow.** A 5-minute `lab status` poller (168 calls / 13h) and a 90-second `lab cancel` loop exist because of R8/R9. **Success metric for this plan: both can be switched off.**

---

## Part 2 — Implementation plan

### File structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/lab/sky_runner.py` | supervisor loop; cluster-loss detection, SIGTERM handling | 1, 6 |
| `src/lab/backends/skypilot.py` | `cancel()` ordering, DO-direct teardown fallback, label hygiene | 2, 3, 8 |
| `src/lab/models.py` | `teardown_status` third state | 4 |
| `src/lab/core.py` | reconcile remediation of `unsupervised` | 5 |
| `src/lab/_util.py` | process identity beyond bare PID | 5 |
| `src/lab/cli.py` | BrokenPipe handling, unknown-command suggestion | 7 |
| `tests/test_cluster_loss.py` | new — R8 | 1 |
| `tests/test_cancel_teardown.py` | new — R9 | 2 |
| `tests/test_do_teardown_fallback.py` | new — R14 | 3 |
| `tests/test_teardown_unknown.py` | new — R10 | 4 |
| `tests/test_pid_identity.py` | new — R16 | 5 |
| `tests/test_cli_papercuts.py` | new — R18, R19 | 7 |

---

### Phase 0 — Environment (no code, do first)

- [x] **Step 1: Kill the live version skew** — DONE 2026-08-20. `uv lock --upgrade-package skypilot`
      moved 0.12.3 → 0.13.0 (the `>=0.12` constraint already allowed it; only the lock pinned it).
      `sky_versions()` now reports `client=0.13.0 server=0.13.0 compatible=True`. Full suite 982
      passed on the new version, packaging guard passed. **`tempotron-capacity` was NOT touched** —
      it was already on 0.13.0, which is why its teardowns worked while this repo's did not.

The machine is running client 0.12.3 against server 0.13.0 right now, which is R4. Until this is done, `lab reconcile --apply` is hard-blocked by design and every teardown outcome is unreliable.

```bash
cd /home/user/.superset/projects/laboratory
uv add "skypilot==0.13.0"        # match the running API server
uv run python -c "import sky; print(sky.__version__)"
uv run lab reconcile             # expect: exits 0, no version-skew refusal
```

Note the API server was started from `tempotron-capacity`'s venv, so both projects must agree. Check every lab-using project on the box:

```bash
for p in /home/user/.superset/projects/*/; do
  [ -x "$p/.venv/bin/python3" ] && echo -n "$p " && \
    "$p/.venv/bin/python3" -c "import sky;print(sky.__version__)" 2>/dev/null || echo
done
```

- [x] **Step 2: Commit the work already on the branch** — DONE 2026-08-20, three commits:
      `3d296d0` deps, `df5f43c` the reconcile fix, `f1c0ed4` docs. Working tree clean.
      Live dry-run reconcile with a *working* client: every destroyable list empty, 10
      `tempotron-capacity` resources under `other_projects`, exit 0.

```bash
git add -A
git commit -m "fix(reconcile): prove ownership before destroying, never trust an unread destroy

reconcile joined machine-global cloud state against the current repo's job
store, so another project's running clusters read as orphans; --apply then
destroyed seven live tempotron-capacity jobs and reported nothing destroyed
with exit 0.

- attribute lab-* resources machine-wide (lab.attribution: ~/.lab/jobs index
  + project-tagged event ledger); never destroy what we cannot attribute
- stamp the project into cluster names and GCP labels
- record destroy_outcomes; exit 5 when a destroy does not confirm success
- hard-block --apply under SkyPilot client/server version skew
- match GCP's 35-char truncated names; protect legacy cluster-name aliases

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Phase 1 — Cost safety (highest value)

#### Task 1: The supervisor must notice its machine is gone (R8) — DONE `0822fae`

**Files:**
- Modify: `src/lab/sky_runner.py:63-106` (`_wait_terminal`)
- Test: `tests/test_cluster_loss.py` (create)

**Interfaces:**
- Consumes: `lab._skycompat.classify_sky_error(exc) -> SkyErrorVerdict` with `.outcome in {"undecodable_response", "failed", "unknown"}` — already classifies `ClusterDoesNotExist` as `failed`.
- Produces: `_wait_terminal(...) -> tuple[JobState, bool, str | None]` — a third element, `lost_reason`, `None` unless the cluster is gone.

- [ ] **Step 1: Write the failing test**

```python
"""A cluster that disappears mid-run must end the job now, not at the timeout.

Live evidence (2026-08-20, job 20260820-071913-be3c72): SkyPilot answered
`Cluster 'lab-…' does not exist.` 65 consecutive times and the supervisor kept
polling, because every poll exception was printed and swallowed. max_wait is
timeout + 300, so the job stayed `running` for a possible 125 minutes.
"""

import pytest

from lab.models import JobState
from lab.sky_runner import _wait_terminal


class _ClusterGone(Exception):
    """Stands in for sky's ClusterDoesNotExist, matched by type name."""


_ClusterGone.__name__ = "ClusterDoesNotExist"


class _FakeSky:
    def __init__(self, exc):
        self.exc = exc
        self.polls = 0

    def get(self, x):
        return x

    def queue(self, cluster, skip_finished=False):
        self.polls += 1
        raise self.exc


def test_cluster_gone_ends_the_wait_immediately():
    sky = _FakeSky(_ClusterGone("Cluster 'lab-x' does not exist."))

    state, reached, lost = _wait_terminal(sky, "lab-x", None, max_wait=7200, poll_s=0.01)

    assert sky.polls <= 3, "must not keep polling a cluster that is definitively gone"
    assert state is JobState.failed
    assert reached is False
    assert lost is not None and "does not exist" in lost


def test_a_transient_poll_error_is_still_tolerated():
    """Only a *definitive* answer ends the wait; a blip must not fail a healthy job."""
    sky = _FakeSky(TimeoutError("read timed out"))

    state, reached, lost = _wait_terminal(sky, "lab-x", None, max_wait=0.05, poll_s=0.01)

    assert sky.polls > 1
    assert lost is None
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_cluster_loss.py -q`
Expected: FAIL — `_wait_terminal` returns a 2-tuple, and the first test times out polling.

- [ ] **Step 3: Implement**

In `src/lab/sky_runner.py`, replace the body of `_wait_terminal`'s loop:

```python
    deadline = time.time() + max_wait
    name: str | None = None
    since_beat = 0.0
    reached = False
    lost_reason: str | None = None
    while time.time() < deadline:
        try:
            name = _job_status_name(sky_mod, cluster, sky_job_id)
        except Exception as e:  # noqa: BLE001
            # A poll failure has three meanings and only one of them is "keep waiting".
            # `ClusterDoesNotExist` is definitive: the machine is gone, the job can never
            # reach a terminal status, and waiting out `max_wait` (timeout + 300s) just
            # burns the budget while the manifest lies about being `running`. Observed
            # 2026-08-20: 65 consecutive such answers, ignored.
            from lab._skycompat import classify_sky_error

            verdict = classify_sky_error(e)
            if verdict.outcome == "failed":
                lost_reason = str(e)
                print(f"[lab] cluster is gone, ending wait: {e}")
                break
            print(f"[lab] queue poll error: {e}")
        if name in _TERMINAL_NAMES:
            reached = True
            break
        time.sleep(poll_s)
        if heartbeat_s and on_heartbeat is not None:
            since_beat += poll_s
            if since_beat >= heartbeat_s:
                since_beat = 0.0
                try:
                    on_heartbeat()
                except Exception as e:  # noqa: BLE001
                    print(f"[lab] heartbeat rsync skipped: {e}")
    if lost_reason is not None:
        return JobState.failed, False, lost_reason
    return map_job_status(name or "FAILED"), reached, None
```

Update the docstring's return description, and update the single call site (~`sky_runner.py:562`) to unpack three values:

```python
            raw_final, reached_terminal, lost_reason = _wait_terminal(...)
            final = raw_final
```

Then, where the manifest is finalised after the wait, honour `lost_reason`:

```python
            if lost_reason is not None:
                store.update_manifest(
                    job_id,
                    status=JobState.failed,
                    ended_at=now(),
                    end_reason=f"cluster disappeared mid-run: {lost_reason}"[:300],
                )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cluster_loss.py tests/test_sky_runner.py -q`
Expected: PASS. Then `uv run pytest -q -m "not packaging"`.

- [ ] **Step 5: Commit**

```bash
git add src/lab/sky_runner.py tests/test_cluster_loss.py
git commit -m "fix(supervisor): end the wait when the cluster is definitively gone"
```

#### Task 2: `lab cancel` must not report terminal before teardown lands (R9) — DONE `4c36c24`

**Files:**
- Modify: `src/lab/backends/skypilot.py:1379-1402` (`cancel`)
- Test: `tests/test_cancel_teardown.py` (create)

**Interfaces:**
- Consumes: `tear_down_and_record(sky, cluster, store, job_id, cloud) -> bool` (writes `teardown_status`).
- Produces: no signature change; ordering and a `cancelling` intent marker in `_runtime.json`.

- [ ] **Step 1: Write the failing test**

```python
"""A cancel killed mid-teardown must not leave a job that *looks* cleanly finished.

Live evidence (2026-08-20): seven `lab cancel` calls, 90s apart, each killed by an
external watchdog's timeout while blocked in robust_teardown's retry ladder. All
seven manifests read `cancelled` with `teardown_status: None` — a machine that may
still be billing, and no leak signal anywhere.
"""

import pytest

from lab.models import JobState


def test_cancel_interrupted_during_teardown_does_not_read_as_clean(tmp_path, monkeypatch):
    from lab.backends.skypilot import SkyPilotBackend

    backend = _backend_with_running_job(tmp_path, "jc1")

    def _hang(*a, **k):
        raise KeyboardInterrupt("watchdog timeout")

    monkeypatch.setattr("lab.backends.skypilot.tear_down_and_record", _hang)

    with pytest.raises(KeyboardInterrupt):
        backend.cancel("jc1")

    m = backend.store.read_manifest("jc1")
    assert m.teardown_status != "succeeded"
    assert m.status is not JobState.cancelled or m.teardown_status is not None, (
        "a terminal status with no teardown record is the exact shape that hid seven "
        "possibly-billing machines"
    )
```

(Write `_backend_with_running_job` alongside, following `tests/test_leak_blindspots.py`'s
`_dead_running_manifest` pattern: `make_manifest`, `status=running`,
`BackendInfo(provisioner="skypilot")`, `store.create`, `store.write_runtime`.)

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_cancel_teardown.py -q`
Expected: FAIL — the manifest is already `cancelled` with `teardown_status: None`.

- [ ] **Step 3: Implement — record intent first, status last**

```python
    def cancel(self, job_id: str) -> JobState:
        m = self.store.read_manifest(job_id)
        if m.status in _TERMINAL:
            return m.status
        # Record the *intent* before doing anything slow, but do NOT declare the job
        # terminal until teardown has been attempted. A cancel that is killed mid-ladder
        # (an impatient caller, a watchdog timeout) previously left `cancelled` with
        # `teardown_status: None` — terminal, clean-looking, possibly still billing.
        self.store.write_runtime(job_id, cancelling=True)
        rt = self.store.read_runtime(job_id)
        if rt.get("runner_pid"):
            try:
                os.kill(rt["runner_pid"], signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        cluster = rt.get("cluster") or cluster_name_for(job_id)
        import sky

        try:
            sky.get(sky.cancel(cluster, all=True))  # 0.12: RequestId
        except Exception:  # noqa: BLE001 - best-effort; teardown below is what matters
            pass
        tear_down_and_record(sky, cluster, self.store, job_id, m.resources.cloud or "vast")
        self.store.update_manifest(
            job_id, status=JobState.cancelled, ended_at=now(), end_reason="cancelled by user"
        )
        return JobState.cancelled
```

Then teach `status()` to finish an interrupted cancel: when `_runtime.json` has
`cancelling: true`, the manifest is non-terminal and the supervisor pid is dead, treat it
exactly like the existing dead-supervisor branch (attempt teardown, then mark terminal).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cancel_teardown.py tests/test_leak_blindspots.py -q`, then the full suite.

- [ ] **Step 5: Commit**

```bash
git add src/lab/backends/skypilot.py tests/test_cancel_teardown.py
git commit -m "fix(cancel): tear down before declaring the job cancelled"
```

#### Task 3: DO-direct teardown fallback (R14 / F2) — DONE `4c36c24`

**Files:**
- Modify: `src/lab/backends/skypilot.py` (`robust_teardown`)
- Test: `tests/test_do_teardown_fallback.py` (create)

**Interfaces:**
- Consumes: `_get_do_client()`, `list_do_volumes()` (existing).
- Produces: `robust_teardown` gains a DO branch mirroring the vastai/gcp ones; return shape unchanged (`{"status": ..., "error": ...}`).

- [ ] **Step 1: Write the failing test**

```python
def test_cluster_does_not_exist_on_do_triggers_a_direct_destroy(monkeypatch):
    """Vast gets a vastai-sdk fallback and GCP a compute-API one; DO had neither, so a
    lost SkyPilot registration was a guaranteed orphan (field report F2)."""
    from lab.backends import skypilot as m

    destroyed = {"droplets": [], "volumes": []}

    class _Droplets:
        def list(self, **kw):
            return {"droplets": [{"id": 1, "name": "lab-x-3dd1-head"}]}

        def destroy(self, droplet_id, **kw):
            destroyed["droplets"].append(droplet_id)

    class _Volumes:
        def list(self, **kw):
            return {"volumes": [{"id": "v1", "name": "lab-x-3dd1-head", "droplet_ids": []}]}

        def delete(self, volume_id, **kw):
            destroyed["volumes"].append(volume_id)

    class _Client:
        droplets = _Droplets()
        volumes = _Volumes()

    monkeypatch.setattr(m, "_get_do_client", lambda: _Client())

    class _Sky:
        def get(self, x):
            return x

        def down(self, cluster):
            raise RuntimeError("ClusterDoesNotExist: Cluster 'lab-x' does not exist.")

    out = m.robust_teardown(_Sky(), "lab-x", cloud="do", backoffs=(0,))

    assert destroyed["droplets"] == [1]
    assert destroyed["volumes"] == ["v1"]
    assert out["status"] == "succeeded"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_do_teardown_fallback.py -q`
Expected: FAIL — no DO branch exists; `status` is `failed` and nothing is destroyed.

- [ ] **Step 3: Implement**

Add a `_do_direct_teardown(cluster) -> bool` next to the existing vastai/gcp fallbacks and
call it from `robust_teardown` when `cloud == "do"` and the `sky.down` ladder is exhausted.
Match droplets and volumes by `name.startswith(cluster)` — DO does not truncate names
(`max_cluster_name_length()` is `None` there, unlike GCP; see R5). Destroy the droplet
first, then any volume left detached, and return `True` only if nothing matching remains.

- [ ] **Step 4: Run the tests**, then the full suite.

- [ ] **Step 5: Commit**

```bash
git add src/lab/backends/skypilot.py tests/test_do_teardown_fallback.py
git commit -m "feat(teardown): DO-direct fallback when SkyPilot loses the cluster"
```

---

> **Phase 1 note (2026-08-20).** Implementing Task 3 surfaced a hazard worth keeping: this dev box
> has `doctl` configured, so an unstubbed DO call in a test does **not** fail — it silently
> enumerates the live account. The DO fallback's first test run did exactly that. `tests/conftest.py`
> now has an autouse fixture refusing to build a real DO client unless a test patches it
> deliberately. Any future provider-direct fallback should get the same guard before its first run.

> **Task 1 correction (2026-08-20).** The plan said to gate on
> `classify_sky_error(...).outcome == "failed"`. That is **wrong on its own**: `_skycompat`'s
> `_DEFINITE_FAILURES` also holds `ApiServerConnectionError`, `PermissionDeniedError`,
> `APIVersionMismatchError` and others. Those are correct verdicts about a *destroy call* that
> never left the client, but they are not evidence that a remote machine has gone away — gating on
> them would fail a healthy job on a local API-server restart and abandon a still-billing box, R8
> inverted. The shipped gate requires **both** a definitive refusal *and* `ClusterDoesNotExist` in
> the exception chain. Anything reusing that classifier for "is the machine gone?" must do the same.

### Phase 2 — Leak-net integrity

#### Task 4: A third teardown state (R10) — DONE `2be0e3c`

**DECIDED 2026-08-20 — three states, with a distinct exit code.** `teardown_status` becomes:

| value | meaning | `lab wait` |
|---|---|---|
| `succeeded` | the machine is confirmed gone | 0 |
| `failed` | the destroy was definitively refused — a real leak | **3** (unchanged) |
| `unknown` | the outcome could not be read; verify with the provider | **6** (new) |

The rationale is alarm integrity: this morning seven teardowns recorded `failed` while all seven
machines had in fact been destroyed. Folding "cannot tell" into `failed` trains the operator to
ignore exit 3, and then a real leak walks past. Keeping them separate means exit 3 stays rare and
always worth acting on.

**Migration note:** `unknown` is a *new* value on an existing field and `6` is a new exit code.
Anything reading either — the user's watchdog scripts, `lab dashboard`, the MCP `wait` tool's
`teardown_leaks` / `teardown_unconfirmed` split — must handle it. Treat an unrecognised value as
`unknown`, never as `succeeded`.

- [ ] **Step 1:** Write `tests/test_teardown_unknown.py` asserting (a) a `sky.down` raising the
      undecodable-response signature yields `teardown_status="unknown"`, not `"failed"`; (b) a
      `ClusterDoesNotExist` still yields `"failed"`; (c) `lab wait` exits **6** when the only
      non-clean teardown is `unknown`, and still **3** when any is `failed` — the money alarm
      outranks the unknown signal, exactly as it already outranks `--fail-fast`'s exit 4.
- [ ] **Step 2:** Run it; expect FAIL (currently `"failed"`).
- [ ] **Step 3:** Add `unknown` to the `teardown_status` literal in `src/lab/models.py`; in
      `tear_down_and_record`, map `classify_sky_error(...).outcome == "undecodable_response"`
      to `"unknown"` with an `end_reason` annotation naming the provider check to run.
- [ ] **Step 4:** Update `lab wait`'s classification, the MCP `wait` tool's return shape,
      `lab dashboard`'s teardown column, `docs/guides/provenance-and-timeouts.md`, `CLAUDE.md`, and
      the `laboratory` skill's exit-code table. All six state the two-value contract today.
- [ ] **Step 5:** Commit.

#### Task 5: Process identity beyond a bare PID, and reconcile remediation (R16 / F4, R15 / F3) — DONE

**Files:** `src/lab/_util.py:107-122` (`pid_alive`), `src/lab/core.py` (reconcile), `tests/test_pid_identity.py`.

- [ ] **Step 1:** Write `tests/test_pid_identity.py`: a `runner_pid` whose recorded
      `/proc/<pid>` start-time no longer matches the live process must read as dead.
- [ ] **Step 2:** Run it; expect FAIL — bare `os.kill(pid, 0)` reports it alive.
- [ ] **Step 3:** Record `(pid, starttime)` in `_runtime.json` at spawn (field 22 of
      `/proc/<pid>/stat`); make `pid_alive` compare both, treating a missing record as
      "unknown → alive" so legacy runtimes are unaffected.
- [ ] **Step 4:** In `Lab.reconcile`, for each `unsupervised` entry, attempt the same
      teardown+`failed` transition `SkyPilotBackend.status()` already performs — so
      detection no longer depends on someone polling that specific job id (F3, narrowed).
- [ ] **Step 5:** Run the full suite; commit.

---

> **Task 5 note (2026-08-20).** Recording the identity at the spawn sites is only half the fix —
> three `pid_alive()` callers (`backends/local.py`, and two in `scheduler/tick.py`) still passed a
> bare PID and would have kept the blind spot open where nobody was looking. That class of omission
> is invisible at the call site and untestable by behaviour (PID reuse is rare and
> non-deterministic), so `tests/test_pid_identity.py` enforces it structurally instead: an AST scan
> asserting every `pid_alive` call in `src/lab` passes `start_time=`, and every `write_runtime`
> recording a `runner_pid` also records `runner_start_time`. Any new call site fails the suite.

### Phase 3 — Observability

#### Task 6: SIGTERM handler in the supervisor (R13 / F1) — DONE

> **F1 re-measured, 2026-08-20 (second correction).** The register's headline "15/52 = 29% of
> supervisor calls never close" is **stale and overstated**. Re-counting the whole live ledger
> gives **11 of 57 = 19.3%**, and the composition matters more than the rate:
>
> * **8 of 11 are `lab cancel`.** Each has a matching `cli/cancel <that job_id>` call, and
>   `cancel` stops a supervisor with `os.kill(pid, SIGTERM)`. So the dominant producer of
>   `running-or-died` supervisor records is *the tool deliberately stopping them* — not DO
>   flakiness, which is what F1 originally blamed.
> * **1 of 11 was still running when sampled.** `running-or-died` means exactly that: both. Any
>   point-in-time rate counts in-flight supervisors as deaths and therefore overstates them.
> * **2 of 11 remain undetermined** — no cancel, no reboot, no OOM entry in the journal.
>   SIGTERM is plausible but unproven. F5's capture is what would settle future cases.
>
> The fix is still worth having: a signalled supervisor should label its own death, and until now
> it also **abandoned its machine** (below). But "DO supervisors die silently ~40% of the time"
> should not survive into anyone's mental model.

> **A correction to this plan's Task 6 text.** It said to "re-raise as `SystemExit` so the
> existing `except BaseException` teardown path still runs". **There was no teardown path there** —
> the pre-change handler was `events.finish(...)` then `raise`, and every `tear_down_and_record`
> call site sat on an *expected* exit inside `_impl`. A signalled supervisor would have closed its
> ledger row neatly and still walked away from a running machine. Task 6 therefore had to add the
> teardown, not just the handler.

**Files:** `src/lab/sky_runner.py` (entrypoint), `tests/test_sky_runner_signals.py`.

- [ ] **Step 1:** Write a test that sends `SIGTERM` to a `lab.sky_runner` process and asserts
      its ledger `run` call closes with a labelled outcome rather than dangling `running-or-died`.
- [ ] **Step 2:** Run it; expect FAIL (15/52 supervisor calls currently never close).
- [ ] **Step 3:** Install `signal.signal(SIGTERM, ...)` that records
      `events.note("signal", sig="TERM")`, closes the call with `outcome="crash"`, and
      re-raises as `SystemExit` so the existing `except BaseException` teardown path runs.
- [ ] **Step 4:** Re-run; confirm the ledger closes. **Success metric: `supervisor/run`
      never-closed rate falls from 29% toward 0.**
- [ ] **Step 5:** Commit.

#### Task 7: CLI papercuts (R18 / F6, R19 / F7)

**Files:** `src/lab/cli.py`, `tests/test_cli_papercuts.py`.

- [ ] **Step 1:** Write tests: (a) `lab submit --help` piped into a reader that closes early
      exits **0**, not 1; (b) `lab kill <id>` names `cancel` in its error message.
- [ ] **Step 2:** Run; expect FAIL (currently 1, and no suggestion).
- [ ] **Step 3:** In `lab.cli:main`, catch `BrokenPipeError` and exit 0 after
      `os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())`; add a click
      `Group.resolve_command` override suggesting the nearest command via `difflib.get_close_matches`.
- [ ] **Step 4:** Re-run; verify with `uv run lab submit --help | head -3; echo ${PIPESTATUS[0]}`.
- [ ] **Step 5:** Commit.

#### Task 8: Label hygiene and formatter decision (R11, R12)

- [ ] **Step 1:** Make `_instance_label` join only non-empty fields and `.strip()` the result;
      add a test pinning that a label carries no leading/trailing whitespace.
- [x] **Step 2: R12 — DECIDED 2026-08-20: drop the expectation.** `ruff format` is not adopted;
      `CLAUDE.md` now states that `ruff check` + `mypy --strict` are the authoritative gates and that a
      `ruff format --check` failure is not a real one. No code change, nothing to reformat.
- [ ] **Step 3:** Commit.

---

## Live verification — 2026-08-20, and the one finding it produced

Two DO jobs were submitted from this repo and killed deliberately. Total cost about four cents.

**R8 confirmed in production.** With the job streaming output, its droplet was destroyed via the
DO API. The supervisor recorded `status: failed`, `end_reason: "cluster disappeared mid-run:
Cluster 'lab-laboratory-…' does not exist."` **22 seconds later** (destroyed 16:21:20Z, recorded
16:21:42Z), then tore down and exited; `teardown_status: succeeded`, no droplet residue,
`lab wait` exit 0. Before this work it would have sat `running` for the full `timeout + 300s`.

**The attribution guard confirmed on the destructive path.** `lab reconcile --apply --yes` — the
exact command that destroyed seven running jobs that morning — destroyed exactly one leaked volume
belonging to this project and left all 8 `tempotron-capacity` resources alone, naming their owner
on stderr. Four live jobs were running throughout and were unharmed.

### NEW — P4-a: `teardown_status: "succeeded"` does not mean nothing is billing on DO

`severity: medium` · `confirmed live`

The first job failed during launch. Its teardown recorded **`succeeded`** — and left a **50 GB
detached block volume** behind, which was still there and still billing 17 minutes later.
`sky.down` returned cleanly, so the DO-direct fallback (which destroys the droplet *and* its
volume) never ran; nothing else checks. The volume is created before the droplet is fully up, so a
launch that fails partway is exactly the case that strands one.

This is the same leak class F2 addressed, reached by the opposite route: F2 covers `sky.down`
*failing*, this is `sky.down` *succeeding* and being incomplete. `reconcile` does catch it
(`do_volume_orphans`, exit 3), so it is a detection-latency problem rather than an invisible leak
— but a "succeeded" teardown that leaves 50 GB billing undermines the same signal R10 was about.

**Proposed fix:** on DO, before recording `succeeded`, list volumes matching the cluster prefix;
if any remain, delete them (the `_do_destroy_matching` volume half already does exactly this) and
record `failed`/`unknown` if they cannot be removed. **Test:** a `sky.down` that succeeds while a
matching volume remains must not record `teardown_status: "succeeded"`.

## Self-review

**Coverage:** every register row with status OPEN maps to a task — R8→1, R9→2, R14/F2→3, R10→4,
R16/F4 and R15/F3→5, R13/F1→6, R18/F6 and R19/F7→7, R11 and R12→8. R1–R7 are FIXED and covered by
Phase 0 Step 2 (commit). The RETRACTED item and P1/P2 are process, not code.

**Sequencing risk:** Task 1 and Task 2 both change what a job's terminal record looks like; land
Task 1 first so Task 2's tests are written against the new fail-fast behaviour.

**Dependency:** Tasks 1 and 4 both consume `lab._skycompat.classify_sky_error`, which already
exists on this branch. Phase 0 Step 1 (version alignment) should precede any live verification,
but no task's *tests* depend on it.
