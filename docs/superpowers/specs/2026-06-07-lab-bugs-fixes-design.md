# Design — LAB-BUGS fixes (§6, §7, §8, §1)

**Date:** 2026-06-07
**Source:** `LAB-BUGS.md` (observations from a lab *user*; this work is the lab-author response)
**Scope:** all currently-open items — §6 (critical), §7 (security), §8 (usability), §1 (verify).
§2/§4/§5 are already resolved; §3 is not a bug.

## Goals

1. The wall-clock cap (`--timeout`) holds **on the instance**, independent of the local
   supervisor process — a dead supervisor must not produce a 6.5 h run on a 4 h cap.
2. Secrets (Vast API key, auth headers) never reach `logs.txt` on disk.
3. A provision failure caused by a depleted Vast balance says so, instead of a generic
   "no resources" message.
4. `lab wait --timeout` is proven to exit non-zero on timeout-without-completion.

## Non-goals

- Durable object storage / R2 work (P1, separate).
- Honoring `--accelerators` strictly to avoid SKU upgrades (§5 cost note; separate concern).
- A full `lab doctor` command (a balance readout is explicitly out of scope this round).

---

## §6 — wall-clock cap enforced instance-side (critical)

### Root cause (confirmed in code)

`build_run_script` (`backends/skypilot.py:85`) wraps the entrypoint as `timeout N <cmd>`.
GNU `timeout` signals only its **direct child**; the entrypoint is
`uv run --with torch python …`, so `uv` forks `python`, and on timeout the python worker is
orphaned and keeps running. The host therefore **never goes idle**, so
`idle_minutes_to_autostop=5` (`sky_runner.py:119`) — which is *idle*-based, not wall-clock —
never fires. The only wall-clock guard is `max_wait` inside `_wait_terminal`
(`sky_runner.py:140`), which lives in the supervisor; when the supervisor dies, the cap dies
with it. Result: the job ran 6.57 h on a 4 h cap, $146.55.

### Fix — Approach A (instance-side enforcement), all three pieces

**A1. Tree-killing timeout wrapper (root-cause fix).**
Replace the fragile `timeout N <cmd>` one-liner in `build_run_script` with a wrapper that runs
the entrypoint as its own **process group** and kills the *whole group* on timeout:

```bash
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
mkdir -p "$REMOTE_RUN_DIR"
setsid bash -c '<entrypoint_command>' &
child=$!
( sleep <wall>; touch "$REMOTE_RUN_DIR/<TIMEOUT_SENTINEL>"; \
  kill -TERM -"$child" 2>/dev/null; sleep <grace>; kill -KILL -"$child" 2>/dev/null ) &
killer=$!
wait "$child"; rc=$?
kill "$killer" 2>/dev/null || true   # job finished on its own → cancel the killer (no sentinel)
exit "$rc"
```

- `setsid` makes `$child` a process-group leader; `kill -TERM -"$child"` signals the entire
  group (negative PID), so the orphaned python dies too → host goes idle → the existing 5-min
  autostop tears it down **without the supervisor**.
- TERM then KILL after `<grace>` (default 30 s) handles a process ignoring TERM.
- The **killer writes the sentinel** right before it fires, so `promote_timeout` labels the job
  `timed_out` unambiguously — no rc-code guessing (a job that exits with a high code on its own
  must not be mislabeled). If the job finishes first we `kill "$killer"`, so the sentinel is
  never written for a normal completion. (Narrow race if completion coincides with the deadline;
  acceptable.)
- `build_run_script` stays a pure string builder → unit-testable (assert the wall value, grace,
  sentinel path, and entrypoint are present).

**A2. Self-destruct watchdog (hard backstop).**
At the *start* of the run script (before the entrypoint), launch a fully detached on-instance
watchdog that powers the box off at `wall + margin` regardless of what the entrypoint or the
supervisor do:

```bash
nohup setsid bash -c 'sleep <wall+margin>; sudo poweroff -f || sudo shutdown -h now' \
  >/dev/null 2>&1 < /dev/null &
```

- `margin` default 600 s (10 min) — past the cap + autostop slack, so it only fires if A1 and
  autostop both failed.
- Powering the instance off stops GPU billing; `robust_teardown` / `lab reconcile` still
  destroys the rental. Worst-case bill is bounded by `wall + margin + teardown lag`, not 6.5 h.
- Only armed when a timeout is set (no cap → no watchdog).

**A3. Heartbeat rsync (the §6c data-loss fix).**
The supervisor currently rsyncs once, after `_wait_terminal` returns (`sky_runner.py:162`); if
teardown beats it (host STOPPED) `results.csv` is lost (rsync status 255). Add a periodic
rsync during the wait so partial results are captured incrementally:

- Refactor the wait so that every `heartbeat_s` (default 60 s) it best-effort rsyncs
  `REMOTE_RUN_DIR` → `output_dir` (failures logged, never fatal). Cleanest implementation:
  add an optional `on_poll` callback to `_wait_terminal`, or inline a heartbeat counter in its
  loop. The final rsync after terminal still runs (last-write-wins).
- This makes the locally-streamed data path the reporter relied on a first-class behavior.

### What changes

- `backends/skypilot.py`: `build_run_script` rewritten (A1 + A2); new module constants
  `TIMEOUT_KILL_GRACE_S = 30`, `SELF_DESTRUCT_MARGIN_S = 600`.
- `sky_runner.py`: heartbeat rsync in/around `_wait_terminal` (A3).

### Tests

- `build_run_script`: with a timeout → contains `setsid`, the `kill -TERM -` group kill, the
  grace, the sentinel touch, the self-destruct `poweroff` at `wall+margin`, and the entrypoint;
  without a timeout → none of the timeout/watchdog scaffolding, just the bare entrypoint.
- Heartbeat: unit-test the loop calls the rsync hook ~every `heartbeat_s` (inject a fake clock /
  fake rsync) and that an rsync exception doesn't abort the wait.

---

## §7 — redact secrets from captured logs (security)

### Root cause

`SkyPilotBackend.submit` (`backends/skypilot.py:389`) opens `logs.txt` and wires it directly to
the detached supervisor's stdout/stderr. SkyPilot itself logs the Vast API key in a request URL
(`…/asks/<id>/?api_key=<key>`), so the cleartext key is written straight to disk (and would go
to R2 if durable artifacts were on).

### Fix

**Pure core:** `redact(text: str) -> str` in a new focused module `lab/redact.py` (the
fd-redirection helper lives there too, keeping `_util.py` free of threading/os plumbing) that
masks:
- `api_key=<value>` and any `?…_key=<value>` / `&…_key=<value>` query params,
- `Authorization: <value>` headers,
replacing the secret with `…REDACTED…` while keeping the surrounding text. Pure + unit-tested.

**Capture-time installation (supervisor-owned):** at `sky_runner` startup, before any SkyPilot
import/call, install a redirector:
1. open `logs_path` (append),
2. `os.pipe()`; `os.dup2` the write end onto fd 1 and 2,
3. a daemon thread reads the pipe line-by-line, applies `redact`, writes to the log file,
   flushing per line.

Because subprocesses inherit fds 1/2, this captures **both** our `print`s and SkyPilot's
subprocess output. `submit` keeps opening the file as the child's stdout (fallback for an
early import crash); the supervisor's `dup2` then takes over fds before any provisioning/secret
output, so the secret is only ever seen post-redactor → never hits disk.

### What changes

- New `redact` function + `install_log_redaction(log_path) -> None` helper.
- `sky_runner.run_job` (or `__main__`) calls `install_log_redaction` first thing.

### Tests

- `redact`: masks `api_key=`, `?token_key=`, `&secret_key=`, `Authorization:`; leaves
  non-secret text intact; idempotent on already-redacted text.
- Capture: write a known secret to fd 1 after installing the redirector against a temp file;
  assert the file contains `REDACTED` and not the secret. (Integration-style, fd-level.)

---

## §8 — balance-aware provision error (usability)

### Root cause

A depleted/negative Vast balance makes rent attempts return `400 Bad Request`, which SkyPilot
surfaces as `Failed to provision all possible launchable resources` — indistinguishable from
"no GPUs available". `sky_runner`'s launch-failure handlers
(`sky_runner.py:142-159`) record that generic string verbatim.

### Fix (error message only — chosen depth)

- `vast_balance(client=None) -> float | None` in `backends/skypilot.py`: read
  `users/current` via the vastai SDK, return the credit/balance, `None` on any error
  (best-effort, same test-seam pattern as `_get_vast_client` / `vast_hourly_for_cluster`).
- In both the `ProvisionTimeout` and generic `except` blocks in `run_job`, if the balance is
  known and ≤ 0 (or below the run's `estimated_usd`), set
  `end_reason = f"Vast account balance is ${bal:.2f} — top up to provision"` instead of the
  generic SkyPilot string. If balance is unknown/positive, keep the existing message.

### What changes

- `backends/skypilot.py`: `vast_balance`.
- `sky_runner.py`: balance check folded into the failure handlers (small helper to build the
  end_reason so both handlers share it).

### Tests

- `vast_balance`: parses a fake `users/current` payload; returns `None` when the SDK raises.
- Failure-handler: with a fake balance ≤ 0, the recorded `end_reason` is the balance message;
  with balance unknown/positive, the generic message is kept.

---

## §1 — verify `lab wait --timeout` exit code

### Finding

The code is correct: `cli.py:244-245` raises `typer.Exit(code=1)` when `not all_terminal`, and
`core.Lab.wait` (`core.py:323-325`) breaks on the deadline **without** marking jobs terminal.
The reporter's caveat (the background-task *wrapper* reported its own exit) is the likely
explanation — not a lab bug.

### Fix

- Add a regression test: a job whose status stays non-terminal, `wait([id], timeout=<short>)`
  returns non-terminal manifests and the CLI `wait` path exits 1 (assert via the
  `all_terminal=False → Exit(1)` logic; exercise the CLI command with a fake Lab).
- Update `LAB-BUGS.md` §1 to "verified — works as designed; regression test added".

---

## Summary of touched files

| File | Change |
|------|--------|
| `src/lab/backends/skypilot.py` | `build_run_script` rewrite (A1+A2); constants; `vast_balance` |
| `src/lab/sky_runner.py` | heartbeat rsync (A3); `install_log_redaction` call; balance in failure handlers |
| `src/lab/redact.py` (new) | `redact` + `install_log_redaction` |
| `tests/…` | unit tests for run-script, redact/capture, vast_balance, wait exit code |
| `LAB-BUGS.md` | mark §1 verified (and note §6/§7/§8 addressed) |

All changes keep `ruff` (line length 100) and `mypy --strict` on `src/lab` green, and preserve
the CLI/MCP-as-thin-shell convention (logic stays in the backend/runner).
