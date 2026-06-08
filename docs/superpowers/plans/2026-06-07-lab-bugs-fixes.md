# LAB-BUGS Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the open LAB-BUGS items — enforce the wall-clock cap on the instance (§6), redact secrets from logs (§7), give a balance-aware provision error (§8), and prove `lab wait --timeout`'s exit code (§1).

**Architecture:** All fixes live in the skypilot path. §6 makes the cap instance-side: a `setsid --wait` run-wrapper whose in-session timer kills its own process group (`-$$`), a detached self-destruct `poweroff` watchdog, and heartbeat rsync in the supervisor's wait loop. §7 adds a pure `redact()` plus a supervisor-startup fd redirector. §8 adds `vast_balance()` consulted on provision failure. §1 is a CLI exit-code regression test.

**Tech Stack:** Python 3.12, uv, pytest, typer (+`typer.testing.CliRunner`), SkyPilot, vastai-sdk. `ruff` (line length 100), `mypy --strict` on `src/lab`.

**Conventions:** New tests follow `tests/test_skypilot.py` patterns (monkeypatch `skypilot_mod.list_vast_instances`, fake sky/vast classes, `helpers.make_manifest`). Keep CLI/MCP thin — logic stays in backend/runner.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/lab/redact.py` | Pure secret-masking + capture-time fd redirector | **Create** |
| `src/lab/backends/skypilot.py` | Run-script (timeout wrapper + watchdog), `vast_balance` | Modify |
| `src/lab/sky_runner.py` | Heartbeat rsync, redaction install, balance in failure handlers | Modify |
| `tests/test_redact.py` | Tests for `redact` + `install_log_redaction` | **Create** |
| `tests/test_skypilot.py` | Update run-script tests; add watchdog + `vast_balance` tests | Modify |
| `tests/test_runner.py` | Heartbeat `_wait_terminal` test | Modify |
| `tests/test_cli_wait.py` | §1 exit-code regression test | **Create** |
| `LAB-BUGS.md` | Mark §1 verified; note §6/§7/§8 addressed | Modify |

---

## Task 1: §7 — pure `redact()` secret masking

**Files:**
- Create: `src/lab/redact.py`
- Test: `tests/test_redact.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_redact.py
from lab.redact import redact

SECRET = "0000000000000000000000000000000000000"


def test_redact_masks_api_key_query_param():
    line = f"https://console.vast.ai/api/v0/asks/33945613/?api_key={SECRET}"
    out = redact(line)
    assert SECRET not in out
    assert "api_key=" in out and "REDACTED" in out


def test_redact_masks_generic_key_params():
    assert SECRET not in redact(f"url?token_key={SECRET}&x=1")
    assert SECRET not in redact(f"url?foo=1&secret_key={SECRET}")


def test_redact_masks_authorization_header():
    assert "Bearer-xyz" not in redact("Authorization: Bearer-xyz")
    assert "REDACTED" in redact("Authorization: Bearer-xyz")


def test_redact_leaves_plain_text_untouched():
    line = "[lab] provisioning host lab-abc-123 (RTX4090:1)"
    assert redact(line) == line


def test_redact_is_idempotent():
    once = redact(f"?api_key={SECRET}")
    assert redact(once) == once
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_redact.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.redact'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lab/redact.py
"""Scrub secrets from captured subprocess output before it reaches disk (FR-J1).

SkyPilot/Vast log the Vast API key inside request URLs (``…?api_key=<key>``); that output is
streamed into ``logs.txt`` (and would go to R2). :func:`redact` masks the value at capture time
so the secret never lands on disk; :func:`install_log_redaction` wires it onto fds 1/2 in the
supervisor so even subprocess output is filtered.
"""

from __future__ import annotations

import re

_REDACTED = "…REDACTED…"
# Secret value = run of non-delimiter chars after the key marker. Delimiters: & whitespace quotes.
_PATTERNS = (
    re.compile(r"(api_key=)[^&\s\"']+", re.IGNORECASE),
    re.compile(r"([?&][\w-]*?_key=)[^&\s\"']+", re.IGNORECASE),
    re.compile(r"(Authorization:\s*)\S+", re.IGNORECASE),
)


def redact(text: str) -> str:
    """Mask ``api_key=…`` / ``?…_key=…`` query params and ``Authorization:`` headers in ``text``.

    Idempotent: re-redacting already-masked text is a no-op-equivalent (the masked value carries
    no delimiters, so it just re-masks to the same string).
    """
    for pattern in _PATTERNS:
        text = pattern.sub(rf"\1{_REDACTED}", text)
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_redact.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lab/redact.py tests/test_redact.py
git commit -m "feat(redact): pure secret-masking for captured logs (LAB-BUGS §7)"
```

---

## Task 2: §7 — capture-time fd redirector + wire into supervisor

**Files:**
- Modify: `src/lab/redact.py`
- Modify: `src/lab/sky_runner.py:101` (start of `run_job`) and `:205` (`__main__`)
- Test: `tests/test_redact.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_redact.py  (append)
import os

from lab.redact import install_log_redaction


def test_install_log_redaction_scrubs_fd_output(tmp_path, capfd):
    log = tmp_path / "logs.txt"
    # Run in a child process: install_log_redaction reassigns fds 1/2 for the whole process,
    # which would clobber the test runner's stdout if done in-process.
    import subprocess
    import sys

    secret = "0000000000000000000000000000000000000"
    code = (
        "import os,sys; from lab.redact import install_log_redaction;"
        f"install_log_redaction({str(log)!r});"
        f"os.write(1, b'GET /asks/1/?api_key={secret}\\n');"
        "sys.stdout.flush()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    content = log.read_text()
    assert secret not in content
    assert "REDACTED" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_redact.py::test_install_log_redaction_scrubs_fd_output -v`
Expected: FAIL — `ImportError: cannot import name 'install_log_redaction'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/lab/redact.py`:

```python
import os
import threading
from pathlib import Path


def install_log_redaction(log_path: str | Path) -> None:
    """Route this process's stdout+stderr (fds 1 & 2) through :func:`redact` into ``log_path``.

    Opens ``log_path`` (append), replaces fds 1/2 with the write end of a pipe, and drains the
    read end on a daemon thread that redacts each line before writing. Because child processes
    inherit fds 1/2, this also scrubs SkyPilot's subprocess output — the secret is filtered
    before it ever reaches disk. Call once, before any output that may carry a secret.
    """
    sink = open(log_path, "a", buffering=1, errors="replace")  # noqa: SIM115 — lives for process
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _drain() -> None:
        with os.fdopen(read_fd, "r", errors="replace") as pipe:
            for line in pipe:
                sink.write(redact(line))
                sink.flush()

    threading.Thread(target=_drain, name="lab-log-redactor", daemon=True).start()
```

Wire it into the supervisor — `src/lab/sky_runner.py`, first lines of `run_job`:

```python
def run_job(job_dir: Path) -> int:
    job_dir = Path(job_dir)
    store = JobStore(job_dir.parent)
    job_id = job_dir.name
    install_log_redaction(store.logs_path(job_id))  # scrub secrets before any SkyPilot output
    manifest = store.read_manifest(job_id)
```

Add the import at the top of `sky_runner.py` (with the other `from lab.…` imports):

```python
from lab.redact import install_log_redaction
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_redact.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Verify the supervisor still imports and the store has `logs_path`**

Run: `uv run python -c "import lab.sky_runner; from lab.store import JobStore; assert hasattr(JobStore, 'logs_path')"`
Expected: no output, exit 0

- [ ] **Step 6: Commit**

```bash
git add src/lab/redact.py src/lab/sky_runner.py tests/test_redact.py
git commit -m "feat(redact): install fd redirector in supervisor (LAB-BUGS §7)"
```

---

## Task 3: §6 — instance-side timeout wrapper + self-destruct watchdog

**Files:**
- Modify: `src/lab/backends/skypilot.py:37-45` (constants) and `:85-106` (`build_run_script`)
- Test: `tests/test_skypilot.py:43-61` (update) and new tests

- [ ] **Step 1: Update the existing run-script tests to the new contract**

Replace `test_build_scripts_and_timeout` and `test_run_script_timeout_sentinel` in
`tests/test_skypilot.py` with:

```python
def test_build_scripts_and_timeout():
    setup = build_setup_script()
    assert "uv sync --frozen" in setup and "astral.sh/uv/install" in setup
    assert "--no-default-groups" in setup

    m = make_manifest("j1", "python experiments/example_capacity.py", timeout="30m")
    run = build_run_script(m)
    # Entrypoint runs in its own session so the timer can group-kill the whole tree (§6).
    assert "setsid --wait bash -c" in run
    assert "python experiments/example_capacity.py" in run
    assert "source .venv/bin/activate" in run
    # No timeout scaffolding when there is no cap.
    bare = build_run_script(make_manifest("j2", "python x.py"))
    assert "setsid --wait" not in bare and "poweroff" not in bare
    assert "python x.py" in bare


def test_run_script_group_kill_and_sentinel():
    run = build_run_script(make_manifest("j", "python x.py", timeout="30m"))
    assert "sleep 1800" in run                 # 30m wall
    assert "kill -TERM -$$" in run             # TERM the whole process group
    assert "kill -KILL -$$" in run             # then KILL after the grace
    assert f"sleep {skypilot_mod.TIMEOUT_KILL_GRACE_S}" in run
    assert TIMEOUT_SENTINEL in run             # killer drops the sentinel for promote_timeout


def test_run_script_self_destruct_watchdog():
    run = build_run_script(make_manifest("j", "python x.py", timeout="30m"))
    margin = skypilot_mod.SELF_DESTRUCT_MARGIN_S
    assert f"sleep {1800 + margin}" in run     # poweroff at wall + margin
    assert "poweroff" in run
    assert "nohup setsid bash -c" in run       # detached, survives the supervisor
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_skypilot.py::test_run_script_group_kill_and_sentinel tests/test_skypilot.py::test_run_script_self_destruct_watchdog tests/test_skypilot.py::test_build_scripts_and_timeout -v`
Expected: FAIL — old `build_run_script` emits GNU `timeout`, so `setsid --wait`/`poweroff` asserts fail

- [ ] **Step 3: Add constants**

In `src/lab/backends/skypilot.py`, near the other constants (after line 38):

```python
TIMEOUT_KILL_GRACE_S = 30  # SIGTERM -> wait -> SIGKILL grace for a process that ignores TERM
SELF_DESTRUCT_MARGIN_S = 600  # instance self-poweroff backstop fires at wall + this (§6)
```

- [ ] **Step 4: Rewrite `build_run_script`**

Replace the body of `build_run_script` (`skypilot.py:85-106`) with:

```python
def build_run_script(manifest: JobManifest) -> str:
    """Activate the env, then run the entrypoint under an instance-side wall-clock cap (FR-I1, §6).

    The cap must hold even if the local supervisor dies, so enforcement is entirely on the box:

    * the entrypoint runs under ``setsid --wait`` in its OWN session/process group; an in-session
      timer ``kill``s ``-$$`` (the whole group) on timeout, so the orphaned ``uv``→``python``→
      worker tree dies too → the host goes idle → SkyPilot's autostop tears it down with no
      supervisor involved. The killer drops a sentinel so ``promote_timeout`` labels it
      ``timed_out`` (not just ``failed``) regardless of the exit code.
    * a detached ``poweroff`` watchdog at ``wall + SELF_DESTRUCT_MARGIN_S`` is a hard backstop:
      if both the wrapper and autostop somehow fail, the instance powers itself off so GPU
      billing can never run far past the cap.
    """
    timeout = parse_duration(manifest.resources.timeout)
    lines = [
        'export PATH="$HOME/.local/bin:$PATH"',
        "source .venv/bin/activate",
        f'mkdir -p "{REMOTE_RUN_DIR}"',
    ]
    if not timeout:
        lines.append(manifest.run.entrypoint_command)
        return "\n".join(lines) + "\n"

    wall = int(timeout)
    grace = TIMEOUT_KILL_GRACE_S
    sentinel = f"{REMOTE_RUN_DIR}/{TIMEOUT_SENTINEL}"
    cmd = manifest.run.entrypoint_command
    # Inner script runs inside a fresh session (setsid). $$ there is the session/group leader,
    # so `kill -<sig> -$$` signals the entire group. $! / $child expand in the inner shell too —
    # they are inside the single-quoted bash -c body, untouched by the outer shell.
    inner = (
        f"{cmd} &\n"
        "child=$!\n"
        f'( sleep {wall}; touch "{sentinel}"; '
        "kill -TERM -$$ 2>/dev/null; "
        f"sleep {grace}; kill -KILL -$$ 2>/dev/null ) &\n"
        'wait "$child"\n'
    )
    lines += [
        # Hard backstop: power the box off at wall+margin no matter what (§6 cost cap).
        f"nohup setsid bash -c 'sleep {wall + SELF_DESTRUCT_MARGIN_S}; "
        "sudo poweroff -f || poweroff -f || sudo shutdown -h now || shutdown -h now' "
        ">/dev/null 2>&1 </dev/null &",
        # Foreground, blocking, group-killable run of the entrypoint.
        f"setsid --wait bash -c '{inner}'",
        "rc=$?",
        'exit "$rc"',
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 5: Run the run-script tests to verify they pass**

Run: `uv run pytest tests/test_skypilot.py -k "build_scripts or run_script" -v`
Expected: PASS (3 passed) — `test_build_scripts_and_timeout`, `test_run_script_group_kill_and_sentinel`, `test_run_script_self_destruct_watchdog`

- [ ] **Step 6: Confirm `promote_timeout` test still passes (sentinel path unchanged)**

Run: `uv run pytest tests/test_skypilot.py::test_promote_timeout -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/lab/backends/skypilot.py tests/test_skypilot.py
git commit -m "fix(skypilot): enforce wall-clock cap instance-side via setsid group-kill + poweroff watchdog (LAB-BUGS §6)"
```

---

## Task 4: §6 — heartbeat rsync in the supervisor wait loop

**Files:**
- Modify: `src/lab/sky_runner.py:36` (constant), `:52-65` (`_wait_terminal`), `:135-141` (call site)
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py  (append)
import lab.sky_runner as sky_runner


def test_wait_terminal_fires_heartbeat(monkeypatch):
    # Fake sky_mod whose queue reports RUNNING for several polls, then SUCCEEDED.
    polls = {"n": 0}

    class _Status:
        def __init__(self, name):
            self.name = name

    class _FakeSky:
        def get(self, x):
            return x

        def queue(self, cluster, skip_finished=False):
            polls["n"] += 1
            name = "RUNNING" if polls["n"] < 7 else "SUCCEEDED"
            return [{"job_id": 1, "status": _Status(name)}]

    monkeypatch.setattr(sky_runner.time, "sleep", lambda _s: None)  # no real waiting
    beats = {"n": 0}

    final = sky_runner._wait_terminal(
        _FakeSky(), "lab-x", 1, max_wait=10_000,
        poll_s=1.0, heartbeat_s=3.0, on_heartbeat=lambda: beats.__setitem__("n", beats["n"] + 1),
    )
    from lab.models import JobState

    assert final == JobState.succeeded
    # 6 RUNNING polls before terminal, heartbeat every 3 polls -> fired at poll 3 and 6.
    assert beats["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner.py::test_wait_terminal_fires_heartbeat -v`
Expected: FAIL — `_wait_terminal()` got an unexpected keyword argument `poll_s`

- [ ] **Step 3: Add the constant**

In `src/lab/sky_runner.py`, after `_TERMINAL_NAMES` (line 36):

```python
HEARTBEAT_S = 60.0  # how often the supervisor rsyncs partial results down mid-run (§6c)
```

- [ ] **Step 4: Extend `_wait_terminal`**

Replace `_wait_terminal` (`sky_runner.py:52-65`) with:

```python
def _wait_terminal(
    sky_mod,
    cluster: str,
    sky_job_id: int | None,
    max_wait: float,
    *,
    poll_s: float = 10.0,
    heartbeat_s: float | None = None,
    on_heartbeat: Any = None,
) -> JobState:
    """Poll the remote job until terminal — sky.launch (0.12) returns at submit time, not
    completion, so we must wait before fetching artifacts and tearing down.

    If ``heartbeat_s``/``on_heartbeat`` are given, ``on_heartbeat`` is called roughly every
    ``heartbeat_s`` of polling so the supervisor can fetch partial results mid-run; a callback
    error is logged, never fatal (§6c — don't lose ``results.csv`` to a late teardown).
    """
    deadline = time.time() + max_wait
    name: str | None = None
    since_beat = 0.0
    while time.time() < deadline:
        try:
            name = _job_status_name(sky_mod, cluster, sky_job_id)
        except Exception as e:  # noqa: BLE001
            print(f"[lab] queue poll error: {e}")
        if name in _TERMINAL_NAMES:
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
    return map_job_status(name or "FAILED")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_runner.py::test_wait_terminal_fires_heartbeat -v`
Expected: PASS

- [ ] **Step 6: Wire the heartbeat into `run_job`**

Replace `sky_runner.py:140-141` (the `max_wait = …` and `final = _wait_terminal(…)` lines) with:

```python
        max_wait = (parse_duration(manifest.resources.timeout) or 3600) + 300

        def _heartbeat() -> None:
            # Best-effort: pull partial results so a late/failed teardown can't lose them (§6c).
            _rsync_down(cluster, REMOTE_RUN_DIR, store.output_dir(job_id))

        final = _wait_terminal(
            sky, cluster, sky_job_id, max_wait,
            heartbeat_s=HEARTBEAT_S, on_heartbeat=_heartbeat,
        )
```

- [ ] **Step 7: Run the full runner test file**

Run: `uv run pytest tests/test_runner.py -v`
Expected: PASS (all)

- [ ] **Step 8: Commit**

```bash
git add src/lab/sky_runner.py tests/test_runner.py
git commit -m "feat(skypilot): heartbeat rsync of partial results during run (LAB-BUGS §6c)"
```

---

## Task 5: §8 — balance-aware provision error

**Files:**
- Modify: `src/lab/backends/skypilot.py` (add `vast_balance` near `vast_hourly_for_cluster`, ~line 165)
- Modify: `src/lab/sky_runner.py:142-159` (failure handlers)
- Test: `tests/test_skypilot.py`

- [ ] **Step 1: Write the failing test for `vast_balance`**

```python
# tests/test_skypilot.py  (append)
from lab.backends.skypilot import vast_balance


def test_vast_balance_reads_credit(monkeypatch):
    class _V:
        def show_user(self):
            return {"credit": -1.46, "balance": -1.46}

    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: _V())
    assert vast_balance() == -1.46


def test_vast_balance_none_on_error(monkeypatch):
    class _V:
        def show_user(self):
            raise RuntimeError("api down")

    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: _V())
    assert vast_balance() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_skypilot.py -k vast_balance -v`
Expected: FAIL — `ImportError: cannot import name 'vast_balance'`

- [ ] **Step 3: Implement `vast_balance`**

Add to `src/lab/backends/skypilot.py` after `vast_hourly_for_cluster`:

```python
def vast_balance(client: Any | None = None) -> float | None:
    """Current Vast.ai account balance/credit (USD), or None if unavailable (best-effort).

    A depleted/negative balance makes Vast reject rentals with ``400 Bad Request``, which
    SkyPilot surfaces as a generic "Failed to provision … resources" — indistinguishable from
    "no GPUs". We consult this on a provision failure to give an actionable message (§8).
    """
    if client is None:
        client = _get_vast_client()
    try:
        info = client.show_user()
    except Exception as e:  # noqa: BLE001 — best-effort; caller falls back to the generic message
        print(f"[lab] vast balance lookup failed: {e}")
        return None
    for key in ("credit", "balance"):
        val = info.get(key) if isinstance(info, dict) else getattr(info, key, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_skypilot.py -k vast_balance -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the failing test for the balance-aware end_reason**

```python
# tests/test_skypilot.py  (append)
from lab.sky_runner import provision_failure_reason


def test_provision_failure_reason_flags_negative_balance(monkeypatch):
    import lab.sky_runner as sr

    monkeypatch.setattr(sr, "vast_balance", lambda: -1.46)
    reason = provision_failure_reason("launch error: Failed to provision all possible resources")
    assert "balance is $-1.46" in reason and "top up" in reason


def test_provision_failure_reason_keeps_generic_when_funded(monkeypatch):
    import lab.sky_runner as sr

    monkeypatch.setattr(sr, "vast_balance", lambda: 25.0)
    generic = "launch error: Failed to provision all possible resources"
    assert provision_failure_reason(generic) == generic


def test_provision_failure_reason_keeps_generic_when_balance_unknown(monkeypatch):
    import lab.sky_runner as sr

    monkeypatch.setattr(sr, "vast_balance", lambda: None)
    generic = "launch error: Failed to provision all possible resources"
    assert provision_failure_reason(generic) == generic
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_skypilot.py -k provision_failure_reason -v`
Expected: FAIL — `ImportError: cannot import name 'provision_failure_reason'`

- [ ] **Step 7: Implement `provision_failure_reason` and use it in the handlers**

In `src/lab/sky_runner.py`, add `vast_balance` to the existing `from lab.backends.skypilot import (…)` block, and add this helper above `run_job`:

```python
def provision_failure_reason(generic: str) -> str:
    """Enrich a generic provision-failure message with the Vast balance when that's the cause (§8).

    Vast returns 400 on rentals when the balance is depleted; SkyPilot reports that as a generic
    "no resources" string. If the balance is known and not positive, say so instead.
    """
    bal = vast_balance()
    if bal is not None and bal <= 0:
        return f"Vast account balance is ${bal:.2f} — top up to provision"
    return generic
```

Then update the two failure handlers in `run_job`. The `ProvisionTimeout` handler is a genuine
host-never-UP case (leave as-is). The generic `except Exception` handler (`sky_runner.py:154-159`)
becomes:

```python
    except Exception as e:  # noqa: BLE001
        reason = provision_failure_reason(f"launch error: {e}")
        store.update_manifest(
            job_id, status=JobState.failed, ended_at=now(), end_reason=reason[:300]
        )
        tear_down_and_record(sky, cluster, store, job_id)
        return 1
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_skypilot.py -k "provision_failure_reason or vast_balance" -v`
Expected: PASS (5 passed)

- [ ] **Step 9: Commit**

```bash
git add src/lab/backends/skypilot.py src/lab/sky_runner.py tests/test_skypilot.py
git commit -m "feat(skypilot): balance-aware provision-failure message (LAB-BUGS §8)"
```

---

## Task 6: §1 — `lab wait --timeout` exit-code regression test

**Files:**
- Create: `tests/test_cli_wait.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_cli_wait.py
"""Regression: `lab wait` must exit non-zero when it gives up on a timeout (LAB-BUGS §1)."""

from typer.testing import CliRunner

import lab.cli as cli_mod
from helpers import make_manifest
from lab.cli import app
from lab.models import JobState


def test_wait_exits_1_on_timeout_without_completion(monkeypatch, tmp_path):
    running = make_manifest("j1", "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running}
    )

    class _FakeLab:
        def wait(self, ids, *, interval, timeout):
            return [running]  # never reached terminal -> the timeout path

    class _FakeStore:
        def __init__(self, home):
            pass

        def manifest_path(self, job_id):
            p = tmp_path / f"{job_id}.json"
            p.touch()  # exists -> passes the "unknown job id" guard
            return p

    monkeypatch.setattr(cli_mod, "_lab_for", lambda job_id: _FakeLab())
    monkeypatch.setattr(cli_mod, "JobStore", _FakeStore)

    result = CliRunner().invoke(app, ["wait", "j1", "--timeout", "0.5"])
    assert result.exit_code == 1
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_cli_wait.py -v`
Expected: PASS — confirms the existing `if not all_terminal: raise typer.Exit(code=1)` works; the original report was the background-task wrapper's exit, not a lab bug

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli_wait.py
git commit -m "test(cli): lab wait exits 1 on timeout-without-completion (LAB-BUGS §1)"
```

---

## Task 7: Update LAB-BUGS.md and run the full quality gate

**Files:**
- Modify: `LAB-BUGS.md`

- [ ] **Step 1: Mark §1 verified**

In `LAB-BUGS.md` §1, append under the existing Update line:

```markdown
**Update (2026-06-07): VERIFIED — not a lab bug.** `cli.py::wait` does
`if not all_terminal: raise typer.Exit(code=1)`, and `core.Lab.wait` breaks on the deadline
without marking jobs terminal. Added `tests/test_cli_wait.py` proving the CLI exits 1 on a
timeout-without-completion. The original exit-0 was the background-task wrapper's own exit.
```

- [ ] **Step 2: Annotate §6, §7, §8 as addressed**

Add a one-line `**Update (2026-06-07): addressed in lab — …**` note at the top of each of §6, §7,
and §8 summarizing the fix (instance-side `setsid` group-kill + `poweroff` watchdog + heartbeat
rsync; capture-time `redact()`; `vast_balance`-aware provision error), mirroring the existing
"FIXED in lab" annotation style used in §4/§5.

- [ ] **Step 3: Run ruff**

Run: `uv run ruff check src/lab tests && uv run ruff format --check src/lab tests`
Expected: no errors (run `uv run ruff format src/lab tests` to fix formatting if needed)

- [ ] **Step 4: Run mypy strict**

Run: `uv run mypy --strict src/lab`
Expected: `Success: no issues found`
(If `_wait_terminal`'s `on_heartbeat: Any` or fake-sky test seams surface a strict error, type
`on_heartbeat` as `Callable[[], None] | None` and import `Callable` from `collections.abc`.)

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -q`
Expected: all pass (new: `test_redact.py`, `test_cli_wait.py`; updated: `test_skypilot.py`, `test_runner.py`)

- [ ] **Step 6: Commit**

```bash
git add LAB-BUGS.md
git commit -m "docs(lab-bugs): mark §1 verified and §6/§7/§8 addressed"
```

---

## Self-Review notes

- **Spec coverage:** §6 A1 (Task 3 wrapper), A2 (Task 3 watchdog), A3 (Task 4 heartbeat); §7 redact (Task 1) + capture (Task 2); §8 vast_balance + error (Task 5); §1 regression (Task 6); docs + gate (Task 7). All spec sections map to a task.
- **Pre-existing tests touched:** `test_build_scripts_and_timeout` and `test_run_script_timeout_sentinel` assert the *old* GNU-`timeout` contract — Task 3 Step 1 replaces them (the second is superseded by `test_run_script_group_kill_and_sentinel`). This is called out so the worker doesn't treat the change as a regression.
- **Type consistency:** `redact(text:str)->str`, `install_log_redaction(log_path)->None`, `vast_balance(client=None)->float|None`, `provision_failure_reason(generic:str)->str`, `_wait_terminal(..., *, poll_s, heartbeat_s, on_heartbeat)` — names identical across tasks/tests.
- **Caveat carried from spec:** the timeout wrapper inlines the entrypoint inside a single-quoted `bash -c '…'`; an entrypoint containing a single quote would break it — same assumption the original `timeout {cmd}` made. Out of scope to harden here.
