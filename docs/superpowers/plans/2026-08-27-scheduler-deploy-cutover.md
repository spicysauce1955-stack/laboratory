# Scheduler Deploy Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace playground's stale `lab_scheduler` Ansible role with a laboratory-owned,
immutable blue-green droplet swap — no SSH, no in-place mutation, never more than one live
registration-launcher ticking against the queue at once.

**Architecture:** A new free function `wait_for_queue_drain()` in `lab.scheduler.queue` (mirroring
`Lab.wait`'s interval/timeout shape) gives `deploy.sh` a tested way to know the queue is safe to
pause. `lab queue list` gains a `host` field so the script can tell the new droplet's heartbeat
apart from the old one's while both are briefly up. A cloud-init template (rendered by `deploy.sh`
via `envsubst`) bootstraps a fresh droplet from a pinned release tag with no SSH involved; the
systemd unit files are fetched fresh from that tag at boot, never duplicated into the template.

**Tech Stack:** Python 3.12 (pydantic, typer), bash (`set -euo pipefail`, matching
`scripts/release.sh`), cloud-init `#cloud-config`, `doctl`.

**Spec:** `docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md`

## Global Constraints

- No SSH to the droplet anywhere in this plan — this machine has no key for it.
- `deploy.sh` targets `tempotron-capacity` specifically (the only current deferred-scheduling
  consumer); unit files are fetched verbatim from the pinned tag, not templated.
- Nothing here is self-updating — every run of `deploy.sh` is a deliberate, human-initiated act.
- `ruff check` and `mypy --strict` are the authoritative gates on `src/lab`; do not run
  `ruff format`.
- Secrets (Vast key, R2 credentials) are read from the controller (the machine running
  `deploy.sh`) exactly as the current Ansible role does — never written into git.
- **Task 8 (live verification) is human-gated.** It creates and destroys real cloud resources and
  costs real money. Do not execute it as part of an unattended plan run — stop after Task 7 and
  hand control back for an explicit go-ahead.

---

### Task 1: Expose which host wrote the heartbeat

**Files:**
- Modify: `src/lab/cli.py:1376-1397` (`queue_list`)
- Modify: `src/lab/mcp_server.py:541-560` (`queue_list`)
- Test: `tests/test_events_history_cli.py` — no, wrong file; use `tests/test_scheduler_tick.py`
  for the heartbeat-write fixture pattern and add a new test file `tests/test_queue_list_host.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `QueueStore.read_heartbeat() -> dict[str, Any] | None` (existing; the dict already
  contains `"host"`, written by `Scheduler.tick()` at `src/lab/scheduler/tick.py:141`)
- Produces: `lab queue list` / `mcp__lab__queue_list` output gains a top-level `"host": str | None`
  key (`None` when there is no heartbeat yet). Task 5's `deploy.sh` depends on this key existing.

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_queue_list_host.py`:

```python
"""`lab queue list` must surface which host wrote the heartbeat — needed to tell two droplets
ticking against the same queue apart during a scheduler cutover (see
docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from lab.cli import app
from lab.scheduler.queue import LocalQueueStore

runner = CliRunner()


def test_queue_list_reports_which_host_wrote_the_heartbeat(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    q = LocalQueueStore(tmp_path / "queue")
    q.write_heartbeat({"at": "2026-08-27T00:00:00+00:00", "host": "lab-scheduler-old", "tick_count": 1})

    result = runner.invoke(app, ["queue", "list"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["host"] == "lab-scheduler-old"


def test_queue_list_host_is_none_with_no_heartbeat_yet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    LocalQueueStore(tmp_path / "queue")  # never writes a heartbeat

    result = runner.invoke(app, ["queue", "list"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["host"] is None
```

Check `default_queue()` (`src/lab/scheduler/queue.py`) actually honors `LAB_QUEUE_DIR` for a
`LocalQueueStore` before relying on it above — read its body; if the env var it checks has a
different name, use that name in the test instead (do not guess; this is the one thing in this
task worth reading first).

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_queue_list_host.py -v`
Expected: FAIL — `KeyError: 'host'` (or `assert None == "lab-scheduler-old"`).

- [ ] **Step 3: Add `host` to the CLI's `queue_list`**

In `src/lab/cli.py`, inside `queue_list()` (around line 1376), the `_emit({...})` call currently
starts with `"heartbeat_age_s": _heartbeat_age_s(queue),`. Add a `"host"` key read from the same
heartbeat dict `_heartbeat_age_s` already reads. Refactor `_heartbeat_age_s` isn't necessary —
just read the heartbeat once more (it's a cheap local/R2 read, already done twice elsewhere in
this file for control/heartbeat):

```python
@queue_app.command(name="list")
def queue_list() -> None:
    """Entries + state + skip reason, plus scheduler heartbeat age and which host wrote it."""
    queue = default_queue()
    entries = queue.list_entries()
    hb = queue.read_heartbeat()
    _emit(
        {
            "heartbeat_age_s": _heartbeat_age_s(queue),
            "host": (hb or {}).get("host"),
            "control": queue.read_control().model_dump(),
            "entries": [
                ...  # unchanged
            ],
        }
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_queue_list_host.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing MCP test**

In `tests/test_mcp_server.py`, add:

```python
def test_queue_list_tool_reports_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from lab.scheduler.queue import LocalQueueStore

    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    q = LocalQueueStore(tmp_path / "queue")
    q.write_heartbeat({"at": "2026-08-27T00:00:00+00:00", "host": "lab-scheduler-new", "tick_count": 1})
    _, server = _make(tmp_path)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool("queue_list", {})).data

    data = asyncio.run(go())
    assert data["host"] == "lab-scheduler-new"
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_mcp_server.py::test_queue_list_tool_reports_host -v`
Expected: FAIL — `KeyError: 'host'`.

- [ ] **Step 7: Add `host` to the MCP tool's `queue_list`**

In `src/lab/mcp_server.py`, inside the `queue_list` tool (around line 541), the return dict starts
with `"heartbeat_age_s": age,`. Add `"host": (hb or {}).get("host"),` right after it — `hb` is
already read into a local variable a few lines above for the age calculation, reuse it.

- [ ] **Step 8: Run it to verify it passes**

Run: `uv run pytest tests/test_mcp_server.py::test_queue_list_tool_reports_host -v`
Expected: PASS

- [ ] **Step 9: Run the full affected suite + lint**

Run: `uv run pytest tests/test_queue_list_host.py tests/test_mcp_server.py tests/test_cli_papercuts.py -v`
Run: `uv run ruff check src/lab/cli.py src/lab/mcp_server.py tests/test_queue_list_host.py`
Run: `uv run mypy --strict src/lab/cli.py src/lab/mcp_server.py`
Expected: all pass, no lint/type errors.

- [ ] **Step 10: Commit**

```bash
git add src/lab/cli.py src/lab/mcp_server.py tests/test_queue_list_host.py tests/test_mcp_server.py
git commit -m "feat(scheduler): expose which host wrote the queue heartbeat"
```

---

### Task 2: `wait_for_queue_drain()` — tested drain-polling logic

**Files:**
- Modify: `src/lab/scheduler/queue.py` (add `import time`, add the function)
- Test: `tests/test_wait_for_queue_drain.py`

**Interfaces:**
- Consumes: `QueueStore.list_entries() -> list[Registration]`; `Registration.state: RegState`;
  `RegState.launching`, `RegState.launched` (all existing, `src/lab/scheduler/models.py`).
- Produces: `wait_for_queue_drain(queue: QueueStore, *, interval: float = 10.0, timeout: float |
  None = None) -> list[Registration]`. Returns `[]` on a clean drain; returns the still-blocking
  registrations (non-empty) if `timeout` elapsed first. Never raises on timeout. Task 3's CLI
  wrapper imports this exact name from `lab.scheduler.queue`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_wait_for_queue_drain.py`:

```python
"""`wait_for_queue_drain` — the safety gate a scheduler cutover waits on before pausing the
queue (docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md). A registration's
`state` already reflects its mirrored job's real terminality: `Scheduler._sync` keeps them in
lock-step while the queue is unpaused, so checking `state` alone — no separate job-status lookup
— is sufficient."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import PYTHON

from lab.models import CodeRef, JobSpec
from lab.scheduler.models import Guardrails, Registration, RegState, Triggers
from lab.scheduler.queue import LocalQueueStore, wait_for_queue_drain

T0 = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)


def _reg(reg_id: str, state: RegState) -> Registration:
    return Registration(
        reg_id=reg_id,
        created_at=T0,
        spec=JobSpec(command=f"{PYTHON} -c 'print(1)'"),
        triggers=Triggers(),
        guardrails=Guardrails(expires_at=T0 + timedelta(days=1)),
        bundle_key=f"bundles/{reg_id}.tar.gz",
        code=CodeRef(git_commit="0" * 40),
        state=state,
    )


def test_drains_immediately_when_nothing_is_in_flight(tmp_path: Path) -> None:
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("r1", RegState.pending))
    q.put_entry(_reg("r2", RegState.succeeded))

    blocking = wait_for_queue_drain(q, interval=0.01, timeout=1.0)

    assert blocking == []


def test_blocks_on_launching_and_launched(tmp_path: Path) -> None:
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("r1", RegState.launching))
    q.put_entry(_reg("r2", RegState.launched))

    blocking = wait_for_queue_drain(q, interval=0.01, timeout=0.05)

    assert {r.reg_id for r in blocking} == {"r1", "r2"}


def test_returns_empty_once_a_blocking_entry_transitions_to_terminal(tmp_path: Path) -> None:
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("r1", RegState.launched))

    # Flip it to terminal shortly after the first poll, on a real background thread —
    # exercises the actual polling loop, not a mocked clock.
    import threading

    def _finish():
        import time

        time.sleep(0.03)
        q.put_entry(_reg("r1", RegState.succeeded))

    threading.Thread(target=_finish, daemon=True).start()

    blocking = wait_for_queue_drain(q, interval=0.01, timeout=2.0)

    assert blocking == []


def test_pending_with_a_future_trigger_never_blocks(tmp_path: Path) -> None:
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("r1", RegState.pending))

    blocking = wait_for_queue_drain(q, interval=0.01, timeout=0.05)

    assert blocking == []


def test_no_timeout_means_no_timeout_arg_is_required() -> None:
    import inspect

    sig = inspect.signature(wait_for_queue_drain)
    assert sig.parameters["timeout"].default is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_wait_for_queue_drain.py -v`
Expected: FAIL — `ImportError: cannot import name 'wait_for_queue_drain'`.

- [ ] **Step 3: Implement `wait_for_queue_drain`**

In `src/lab/scheduler/queue.py`, add `import time` to the existing imports at the top, then add
this function after the `LocalQueueStore` class (before `r2queue.py`'s sibling module, i.e. at
module scope in this file):

```python
from lab.scheduler.models import ControlConfig, Registration, RegState

_BLOCKING_STATES = frozenset({RegState.launching, RegState.launched})


def wait_for_queue_drain(
    queue: QueueStore, *, interval: float = 10.0, timeout: float | None = None
) -> list[Registration]:
    """Block until no registration is `launching`/`launched`, or `timeout` elapses.

    The safety gate a scheduler cutover waits on before pausing the queue: pausing stops
    `Scheduler._sync`, so a job that finishes after pausing would never be observed reaching
    terminal — this must run first, while the queue is still unpaused and genuinely draining.

    A registration's `state` already reflects its mirrored job's real terminality (`_sync` keeps
    them in lock-step while unpaused), so checking `state` alone is sufficient — no separate
    job-status lookup. A `pending` registration (not yet triggered) is never blocking, regardless
    of how far in the future its trigger is.

    Returns the still-blocking registrations: empty on a clean drain, non-empty (whatever was
    still in flight) when `timeout` was hit first. Never raises on timeout — the caller decides
    what a non-empty result means.
    """
    deadline = time.monotonic() + timeout if timeout is not None else None
    while True:
        blocking = [r for r in queue.list_entries() if r.state in _BLOCKING_STATES]
        if not blocking:
            return []
        if deadline is not None and time.monotonic() >= deadline:
            return blocking
        time.sleep(interval)
```

Note `Registration`/`RegState` may already be imported in this file for type hints used by the
`QueueStore` Protocol — check the existing import block first and merge rather than duplicate.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_wait_for_queue_drain.py -v`
Expected: PASS — all 5 tests.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/lab/scheduler/queue.py tests/test_wait_for_queue_drain.py`
Run: `uv run mypy --strict src/lab/scheduler/queue.py`
Expected: clean.

- [ ] **Step 6: Run the full scheduler test suite to check for regressions**

Run: `uv run pytest tests/test_scheduler_tick.py tests/test_scheduler_models.py -v`
Expected: all pass (unchanged — this task only adds a function, touches nothing existing).

- [ ] **Step 7: Commit**

```bash
git add src/lab/scheduler/queue.py tests/test_wait_for_queue_drain.py
git commit -m "feat(scheduler): add wait_for_queue_drain, the safe-cutover drain gate"
```

---

### Task 3: `lab queue wait-drain` CLI command

**Files:**
- Modify: `src/lab/cli.py` (add the command near `queue_pause`/`queue_resume`, ~line 1450)
- Test: `tests/test_queue_wait_drain_cli.py`

**Interfaces:**
- Consumes: `wait_for_queue_drain` from Task 2; `default_queue()`, `parse_duration()`,
  `_emit()`, `_fail()` (all existing in `lab.cli`).
- Produces: `lab queue wait-drain [--interval FLOAT] [--timeout DURATION]`. Exit `0` with
  `{"drained": true, "blocking": []}` on a clean drain; exit `1` with `{"drained": false,
  "blocking": [reg_id, ...]}` if the timeout was hit. Task 5's `deploy.sh` depends on both the
  command name and this exit-code contract.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_queue_wait_drain_cli.py`:

```python
"""`lab queue wait-drain` — the CLI surface for the scheduler-cutover drain gate."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import PYTHON
from typer.testing import CliRunner

from lab.cli import app
from lab.models import CodeRef, JobSpec
from lab.scheduler.models import Guardrails, Registration, RegState, Triggers
from lab.scheduler.queue import LocalQueueStore

runner = CliRunner()
T0 = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)


def _reg(reg_id: str, state: RegState) -> Registration:
    return Registration(
        reg_id=reg_id,
        created_at=T0,
        spec=JobSpec(command=f"{PYTHON} -c 'print(1)'"),
        triggers=Triggers(),
        guardrails=Guardrails(expires_at=T0 + timedelta(days=1)),
        bundle_key=f"bundles/{reg_id}.tar.gz",
        code=CodeRef(git_commit="0" * 40),
        state=state,
    )


def test_exits_0_when_already_drained(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    LocalQueueStore(tmp_path / "queue")

    result = runner.invoke(app, ["queue", "wait-drain", "--interval", "0.01", "--timeout", "1"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data == {"drained": True, "blocking": []}


def test_exits_1_and_names_blockers_on_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    q = LocalQueueStore(tmp_path / "queue")
    q.put_entry(_reg("reg-stuck", RegState.launched))

    result = runner.invoke(app, ["queue", "wait-drain", "--interval", "0.01", "--timeout", "0.05"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["drained"] is False
    assert data["blocking"] == ["reg-stuck"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_queue_wait_drain_cli.py -v`
Expected: FAIL — `No such command 'wait-drain'` (typer usage error, non-zero exit, no JSON on
stdout).

- [ ] **Step 3: Implement the command**

In `src/lab/cli.py`, add right after `queue_resume` (~line 1461), and add
`wait_for_queue_drain` to the existing `from lab.scheduler.queue import QueueStore,
default_queue` import line:

```python
@queue_app.command(name="wait-drain")
def queue_wait_drain(
    interval: float = typer.Option(10.0, help="seconds between polls"),
    timeout: str | None = typer.Option(
        None, help="give up after this long, e.g. '30m' (bare numbers = seconds)"
    ),
) -> None:
    """Block until no registration is launching/launched, or --timeout elapses — the safety gate
    to run before pausing the queue for a scheduler redeploy (never pause first: pausing stops
    the sync that would let this ever observe a real drain)."""
    queue = default_queue()
    timeout_s = parse_duration(timeout) if timeout else None
    blocking = wait_for_queue_drain(queue, interval=interval, timeout=timeout_s)
    if blocking:
        _emit({"drained": False, "blocking": [r.reg_id for r in blocking]})
        _fail(1, f"{len(blocking)} registration(s) still in flight after timeout")
    _emit({"drained": True, "blocking": []})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_queue_wait_drain_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + type-check + regression check**

Run: `uv run ruff check src/lab/cli.py tests/test_queue_wait_drain_cli.py`
Run: `uv run mypy --strict src/lab/cli.py`
Run: `uv run pytest tests/test_cli_papercuts.py tests/test_events_history_cli.py -v`
Expected: all clean/passing — `test_cli_papercuts.py` does not hardcode a full command list
(checked; it only exercises `["list"]` directly), so it needs no update for the new subcommand.

- [ ] **Step 6: Commit**

```bash
git add src/lab/cli.py tests/test_queue_wait_drain_cli.py
git commit -m "feat(scheduler): add lab queue wait-drain CLI command"
```

---

### Task 4: `deploy/scheduler/cloud-init.yaml.tmpl`

**Files:**
- Create: `deploy/scheduler/cloud-init.yaml.tmpl`
- Test: `deploy/scheduler/test_cloud_init_template.sh` (a small bash script, not pytest — this
  repo has no shell-script test precedent (`scripts/release.sh` has none either); the closest
  existing convention is "write it, run it, read the output," which is what this does)

**Interfaces:**
- Consumes: env vars `TAG`, `DROPLET_NAME`, `LAB_R2_ENDPOINT`, `LAB_R2_BUCKET`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `VAST_API_KEY` — substituted via `envsubst`.
- Produces: a rendered `#cloud-config` file. Task 5's `deploy.sh` is the only consumer and pipes
  this template through `envsubst` before passing it to `doctl ... --user-data-file`.

- [ ] **Step 1: Write the template**

Create `deploy/scheduler/cloud-init.yaml.tmpl`:

```yaml
#cloud-config
# Rendered by deploy/scheduler/deploy.sh via envsubst. Placeholders: $TAG, $DROPLET_NAME,
# $LAB_R2_ENDPOINT, $LAB_R2_BUCKET, $AWS_ACCESS_KEY_ID, $AWS_SECRET_ACCESS_KEY, $VAST_API_KEY.
# Runs once, as root, on first boot. No SSH involved anywhere in this file.

hostname: $DROPLET_NAME
fqdn: $DROPLET_NAME
prefer_fqdn_over_hostname: false

package_update: true
packages:
  - git
  - rsync

write_files:
  - path: /etc/lab/scheduler.env
    owner: root:root
    permissions: '0600'
    content: |
      LAB_R2_ENDPOINT=$LAB_R2_ENDPOINT
      LAB_R2_BUCKET=$LAB_R2_BUCKET
      AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
      AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
      VAST_API_KEY=$VAST_API_KEY

runcmd:
  # Swapfile: skypilot imports are memory-hungry on a 1GB droplet. fstab-persisted so it survives
  # a reboot (the kill-test in deploy/scheduler/README.md explicitly reboots the droplet).
  - fallocate -l 2048M /swapfile
  - chmod 600 /swapfile
  - mkswap /swapfile
  - swapon /swapfile
  - echo '/swapfile none swap sw 0 0' >> /etc/fstab

  # uv, installed where every user can run it — the tick runs as `lab`, not root.
  - 'curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh'

  # Service user, created before anything is installed as it.
  - useradd --create-home --system --shell /bin/bash lab

  # Vast API key for the service user (sky reads the file; the sdk reads $VAST_API_KEY from env).
  - install -d -o lab -g lab -m 0700 /home/lab/.config/vastai
  - install -o lab -g lab -m 0600 /etc/lab/scheduler.env /home/lab/.config/vastai/vast_api_key.env
  - 'bash -c "echo \"$VAST_API_KEY\" > /home/lab/.config/vastai/vast_api_key"'
  - chown lab:lab /home/lab/.config/vastai/vast_api_key
  - chmod 600 /home/lab/.config/vastai/vast_api_key

  # Install the pinned laboratory release as the `lab` user (puts the entrypoint at
  # /home/lab/.local/bin/lab — root's uv tool install would land under 0700 /root/.local/bin,
  # which User=lab in the unit file below cannot execute: 203/EXEC).
  - >-
    sudo -u lab -H uv tool install
    "laboratory[skypilot,r2] @ git+https://github.com/spicysauce1955-stack/laboratory.git@$TAG"

  # Clone the experiment project the scheduler runs jobs against.
  - install -d -o lab -g lab /opt/tempotron-capacity
  - sudo -u lab -H git clone https://github.com/spicysauce1955-stack/tempotron-capacity.git /opt/tempotron-capacity

  # Fetch the systemd units FRESH from the pinned tag — never duplicated into this template.
  # They have not needed any path substitution since v0.5.0 (WorkingDirectory/ExecStart are
  # already correct for this project in the checked-in files).
  - >-
    curl -fsSL
    https://raw.githubusercontent.com/spicysauce1955-stack/laboratory/$TAG/deploy/scheduler/lab-scheduler.service
    -o /etc/systemd/system/lab-scheduler.service
  - >-
    curl -fsSL
    https://raw.githubusercontent.com/spicysauce1955-stack/laboratory/$TAG/deploy/scheduler/lab-scheduler.timer
    -o /etc/systemd/system/lab-scheduler.timer
  - systemctl daemon-reload
  - systemctl enable --now lab-scheduler.timer
```

Remove the redundant `install -o lab -g lab -m 0600 /etc/lab/scheduler.env
/home/lab/.config/vastai/vast_api_key.env` line above before saving — it was a stray leftover
from drafting and duplicates the `bash -c "echo ..."` line that actually writes the key file;
the vast key must come from `$VAST_API_KEY` directly, not copied from `/etc/lab/scheduler.env`.

- [ ] **Step 2: Write the render-and-validate script**

Create `deploy/scheduler/test_cloud_init_template.sh`:

```bash
#!/usr/bin/env bash
# Render cloud-init.yaml.tmpl with fixture values and check every fixture value actually
# appears in the output, and the result is valid YAML.
#
# NOT a "grep for leftover ${...}" check: this template uses bare $VAR (not ${VAR}) form, and
# envsubst substitutes an unset bare $VAR with an EMPTY STRING, not the literal text — so a
# leftover-braces check would silently pass even if a variable were never exported and every
# line using it rendered blank. Only checking that each real fixture VALUE shows up in the
# output actually catches that failure mode.
set -euo pipefail
cd "$(dirname "$0")"

export TAG="v9.9.9-test"
export DROPLET_NAME="lab-scheduler-test-fixture"
export LAB_R2_ENDPOINT="https://example.r2.cloudflarestorage.com"
export LAB_R2_BUCKET="lab-artifacts-test"
export AWS_ACCESS_KEY_ID="AKIAFIXTUREFIXTURE"
export AWS_SECRET_ACCESS_KEY="fixture-secret-do-not-reuse"
export VAST_API_KEY="fixture-vast-key"

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT
envsubst < cloud-init.yaml.tmpl > "$OUT"

fail=0
for value in "$TAG" "$DROPLET_NAME" "$LAB_R2_ENDPOINT" "$LAB_R2_BUCKET" \
             "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$VAST_API_KEY"; do
  grep -qF -- "$value" "$OUT" || {
    echo "FAIL: fixture value '$value' never appears in rendered output -- envsubst silently dropped it" >&2
    fail=1
  }
done
[[ "$fail" == "0" ]] || exit 1

python3 -c "import yaml, sys; yaml.safe_load(open(sys.argv[1]))" "$OUT" || {
  echo "FAIL: rendered output is not valid YAML" >&2
  exit 1
}

echo "OK: template renders to valid YAML with every fixture value substituted"
```

```bash
chmod +x deploy/scheduler/test_cloud_init_template.sh
```

- [ ] **Step 3: Run it to verify it fails first**

Prove the check is real before trusting it: temporarily comment out the `export VAST_API_KEY=...`
line (simulating a variable that was never passed through), run the script, confirm it reports
the missing-value failure — this is the exact bug class ("a secret silently rendered blank") the
check exists to catch. Uncomment the line afterward. Do this once, by hand; it is a sanity check
on the test's own effectiveness, not a formal step to repeat.

Run: `bash deploy/scheduler/test_cloud_init_template.sh`
Expected (with `VAST_API_KEY` commented out): `FAIL: fixture value 'fixture-vast-key' never appears in rendered output`

- [ ] **Step 4: Restore the export and verify it passes**

Run: `bash deploy/scheduler/test_cloud_init_template.sh`
Expected: `OK: template renders to valid YAML with every fixture value substituted`

- [ ] **Step 5: Commit**

```bash
git add deploy/scheduler/cloud-init.yaml.tmpl deploy/scheduler/test_cloud_init_template.sh
git commit -m "feat(scheduler): add the cloud-init template for droplet bootstrap"
```

---

### Task 5: `deploy/scheduler/deploy.sh`

**Files:**
- Create: `deploy/scheduler/deploy.sh`
- Test: manual `bash -n` syntax check + `--dry-run` structural run (no pytest — same reasoning
  as Task 4; this is infrastructure orchestration, not library code)

**Interfaces:**
- Consumes: `lab queue wait-drain` / `pause` / `resume` / `list` / `register` / `queue show`
  (Task 3 + existing commands); `deploy/scheduler/cloud-init.yaml.tmpl` (Task 4); `doctl`.
- Produces: `deploy/scheduler/deploy.sh vX.Y.Z [--dry-run]` — the eight-step cutover from the
  spec. Exits non-zero and leaves the old droplet untouched/running on any failure before step 5;
  rolls back automatically on a failed smoke test (step 7).

- [ ] **Step 1: Write the script**

Create `deploy/scheduler/deploy.sh`:

```bash
#!/usr/bin/env bash
# Redeploy the scheduler droplet: build a new one from a pinned release, prove it works, retire
# the old one. Immutable blue-green — never mutates a live droplet, never needs SSH.
#
#   deploy/scheduler/deploy.sh vX.Y.Z [--dry-run]
#
# Requires: doctl (authenticated), this repo's own `uv` venv (`uv sync`), and the same
# controller-side secrets the old Ansible role read: ~/.config/vastai/vast_api_key,
# ~/.cloudflare/r2.credentials, $LAB_R2_ENDPOINT exported (and optionally $LAB_R2_BUCKET).
#
# See docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md for the full design
# and why each step is ordered the way it is (two real bugs were caught and fixed in review:
# a self-deadlock from pausing before draining, and a double-launch race from deleting the old
# droplet too early).
set -euo pipefail

TAG="${1:-}"
DRY_RUN="${2:-}"
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: $0 vX.Y.Z [--dry-run]" >&2; exit 2; }

cd "$(git rev-parse --show-toplevel)"

say() { printf '\n=== %s\n' "$1"; }
run() { if [[ "$DRY_RUN" == "--dry-run" ]]; then echo "[dry-run] $*"; else "$@"; fi; }
json_field() { python3 -c "import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ''))" "$1"; }

DROPLET_NAME="lab-scheduler-$(date -u +%Y%m%dT%H%M%SZ)"
REGION="nyc3"
SIZE="s-1vcpu-1gb"
DRAIN_TIMEOUT="${LAB_DEPLOY_DRAIN_TIMEOUT:-30m}"
VERIFY_TIMEOUT_S="${LAB_DEPLOY_VERIFY_TIMEOUT_S:-300}"
SMOKE_TIMEOUT_S="${LAB_DEPLOY_SMOKE_TIMEOUT_S:-900}"

say "preflight"
command -v doctl >/dev/null || { echo "doctl not found on PATH" >&2; exit 1; }
VAST_KEY_FILE="${HOME}/.config/vastai/vast_api_key"
R2_CRED_FILE="${HOME}/.cloudflare/r2.credentials"
[[ -f "$VAST_KEY_FILE" ]] || { echo "missing $VAST_KEY_FILE" >&2; exit 1; }
[[ -f "$R2_CRED_FILE" ]] || { echo "missing $R2_CRED_FILE" >&2; exit 1; }
[[ -n "${LAB_R2_ENDPOINT:-}" ]] || { echo "LAB_R2_ENDPOINT not set" >&2; exit 1; }
LAB_R2_BUCKET="${LAB_R2_BUCKET:-lab-artifacts}"
VAST_API_KEY="$(cat "$VAST_KEY_FILE")"
AWS_ACCESS_KEY_ID="$(awk -F' *= *' '/aws_access_key_id/{print $2; exit}' "$R2_CRED_FILE")"
AWS_SECRET_ACCESS_KEY="$(awk -F' *= *' '/aws_secret_access_key/{print $2; exit}' "$R2_CRED_FILE")"
[[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" ]] || {
  echo "could not read aws_access_key_id/aws_secret_access_key from $R2_CRED_FILE" >&2; exit 1; }

OLD_DROPLET_ID="$(doctl compute droplet list --tag-name lab-scheduler --format ID --no-header | head -1)"
[[ -n "$OLD_DROPLET_ID" ]] || {
  echo "no existing lab-scheduler droplet found (doctl compute droplet list --tag-name lab-scheduler) -- nothing to swap" >&2
  exit 1
}
echo "old droplet: $OLD_DROPLET_ID"
echo "new droplet: $DROPLET_NAME (pinned to $TAG)"

say "1. wait for drain (unpaused)"
run uv run lab queue wait-drain --timeout "$DRAIN_TIMEOUT"

say "2. pause"
run uv run lab queue pause

say "3. create new droplet"
RENDERED="$(mktemp)"
trap 'rm -f "$RENDERED"' EXIT
TAG="$TAG" DROPLET_NAME="$DROPLET_NAME" LAB_R2_ENDPOINT="$LAB_R2_ENDPOINT" \
  LAB_R2_BUCKET="$LAB_R2_BUCKET" AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" VAST_API_KEY="$VAST_API_KEY" \
  envsubst < "$(dirname "$0")/cloud-init.yaml.tmpl" > "$RENDERED"
run doctl compute droplet create "$DROPLET_NAME" \
  --region "$REGION" --size "$SIZE" --image ubuntu-24-04-x64 \
  --tag-names lab-scheduler --user-data-file "$RENDERED" --wait

say "4. verify the new droplet's heartbeat (by identity, not just recency -- the old one is still up)"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  deadline=$(( $(date +%s) + VERIFY_TIMEOUT_S ))
  host=""
  while (( $(date +%s) < deadline )); do
    host="$(uv run lab queue list | json_field host)"
    [[ "$host" == "$DROPLET_NAME" ]] && break
    sleep 10
  done
  if [[ "$host" != "$DROPLET_NAME" ]]; then
    echo "new droplet never confirmed alive as $DROPLET_NAME (last seen host: ${host:-<none>})" >&2
    doctl compute droplet delete "$DROPLET_NAME" --force
    exit 1
  fi
else
  echo "[dry-run] poll lab queue list until host == $DROPLET_NAME"
fi

say "5. power off old droplet (reversible, not deleted)"
run doctl compute droplet-action power-off "$OLD_DROPLET_ID" --wait

say "6. resume -- exactly one unpaused ticker from here on"
run uv run lab queue resume

rollback() {
  echo "rolling back: old droplet resumes service, new droplet is discarded" >&2
  uv run lab queue pause
  doctl compute droplet-action power-on "$OLD_DROPLET_ID" --wait
  uv run lab queue resume
  doctl compute droplet delete "$DROPLET_NAME" --force
}

say "7. smoke test -- one real registration, through the new scheduler"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  reg_json="$(uv run lab register -c "uv run experiments/example_capacity.py" \
    --backend cpu --cloud do --timeout 10m --max-cost 1 --expires +1h)"
  reg_id="$(echo "$reg_json" | json_field reg_id)"
  [[ -n "$reg_id" ]] || { echo "smoke registration did not return a reg_id: $reg_json" >&2; rollback; exit 1; }
  echo "smoke reg_id: $reg_id"

  smoke_deadline=$(( $(date +%s) + SMOKE_TIMEOUT_S ))
  state=""
  while (( $(date +%s) < smoke_deadline )); do
    state="$(uv run lab queue show "$reg_id" | json_field state)"
    case "$state" in
      succeeded) break ;;
      failed|expired|cancelled) break ;;
    esac
    sleep 15
  done
  if [[ "$state" != "succeeded" ]]; then
    echo "smoke registration did not succeed (last state: ${state:-unknown}) -- rolling back" >&2
    rollback
    exit 1
  fi
  echo "smoke registration succeeded"
else
  echo "[dry-run] lab register a smoke job, poll lab queue show until succeeded"
fi

say "8. delete old droplet -- only reached after a real smoke success"
run doctl compute droplet delete "$OLD_DROPLET_ID" --force

say "done -- $DROPLET_NAME is now the scheduler, pinned to $TAG"
```

```bash
chmod +x deploy/scheduler/deploy.sh
```

- [ ] **Step 2: Syntax-check it**

Run: `bash -n deploy/scheduler/deploy.sh`
Expected: no output, exit 0 (bash's own parser, catches real syntax errors even without doctl or
credentials present).

- [ ] **Step 3: Best-effort lint if shellcheck is available**

Run: `command -v shellcheck && shellcheck deploy/scheduler/deploy.sh || echo "shellcheck not installed, skipping"`
Expected: either a clean shellcheck run, or the skip message. Fix anything shellcheck flags
that isn't a deliberate, commented choice (e.g. the intentional word-splitting in `run "$@"` is
fine; an actually-unquoted variable that should be quoted is not).

- [ ] **Step 4: Structural dry-run against a fake `doctl`/`lab`**

There is no real droplet or R2 bucket in this environment, so this step proves the script's
*control flow* (argument parsing, preflight checks, step ordering) without touching anything
real. Create a throwaway `PATH` shim:

```bash
mkdir -p /tmp/deploy-dryrun-bin
cat > /tmp/deploy-dryrun-bin/doctl <<'EOF'
#!/usr/bin/env bash
echo "[fake doctl] $*" >&2
[[ "$1 $2" == "compute droplet" && "$3" == "list" ]] && echo "111111111"
EOF
chmod +x /tmp/deploy-dryrun-bin/doctl
mkdir -p /tmp/fake-secrets/.config/vastai /tmp/fake-secrets/.cloudflare
echo "fixture-vast-key" > /tmp/fake-secrets/.config/vastai/vast_api_key
printf 'aws_access_key_id = fixture\naws_secret_access_key = fixture\n' > /tmp/fake-secrets/.cloudflare/r2.credentials
HOME=/tmp/fake-secrets LAB_R2_ENDPOINT=https://example.r2.cloudflarestorage.com \
  PATH="/tmp/deploy-dryrun-bin:$PATH" \
  deploy/scheduler/deploy.sh v9.9.9 --dry-run
```

Expected: runs through all 8 `say` section headers to `"done -- ... is now the scheduler"`
without a real `lab`/network call being *required to succeed* for steps wrapped in `run` (they
just echo `[dry-run] ...`). Steps 4 and 7 are not wrapped in `run` (they need real conditional
logic even in dry-run) — confirm they print their `[dry-run] ...` placeholder lines and do not
attempt a real `uv run lab` call when `DRY_RUN == --dry-run` (re-check the `if [[ "$DRY_RUN" !=
"--dry-run" ]]` guards around steps 4 and 7 match this — if the structural dry-run instead fails
trying to reach a real `lab`/`doctl` endpoint, that guard has a bug; fix it before moving on).

- [ ] **Step 5: Commit**

```bash
git add deploy/scheduler/deploy.sh
git commit -m "feat(scheduler): add deploy.sh, the blue-green cutover orchestrator"
```

---

### Task 6: Rewrite `deploy/scheduler/README.md`

**Files:**
- Modify: `deploy/scheduler/README.md`

**Interfaces:**
- Consumes: nothing new (documentation only).
- Produces: an accurate README — `deploy.sh` as the primary path, the existing manual/SSH
  sections kept but demoted to "reference," and the stale placeholder-`ExecStart` claim removed.

- [ ] **Step 1: Fix the stale line first (small, independent, easy to verify)**

In `deploy/scheduler/README.md`, find the "Fetch the unit + timer" section (step 6 of the current
manual "Provision" runbook). It currently says something like:

```
The shipped `ExecStart` is a placeholder pointing at `/root/.local/bin/lab`; with the unit's
`User=lab` that is the 203/EXEC failure above. Replace it.
```

This is false as of the checked-in `deploy/scheduler/lab-scheduler.service` (correct since
v0.5.0). Replace that paragraph with:

```
As of v0.5.0 the checked-in `ExecStart`/`WorkingDirectory`/`Environment` already point at the
right paths for a `tempotron-capacity` deploy — no substitution needed. (This used to require a
manual patch; it doesn't anymore. If you're deploying against a *different* experiment project,
you still need to edit those three lines by hand.)
```

- [ ] **Step 2: Add the new primary section at the top of the file, above "Provision (playground repo)"**

```markdown
## Redeploy (primary path, since 2026-08)

```bash
deploy/scheduler/deploy.sh vX.Y.Z
```

Builds a new droplet from the pinned tag, verifies it can actually launch a job, then retires the
old one — an immutable blue-green swap, never an in-place mutation. No SSH involved. Safe to
re-run: every step before the final delete leaves the previous droplet as a fallback, and a failed
smoke test rolls back automatically.

Requires `doctl` (authenticated) and the same controller-side secrets the manual steps below
always needed: `~/.config/vastai/vast_api_key`, `~/.cloudflare/r2.credentials`,
`$LAB_R2_ENDPOINT` exported.

Full design + the two real bugs an adversarial review caught before this shipped:
`docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md`.

The sections below (manual provisioning via the `playground` repo, in-place SSH upgrade) are kept
as reference/fallback — `deploy.sh` is the path to actually use.
```

- [ ] **Step 3: The rest of the file needs no further changes — verified, not assumed**

The "Upgrading the host" and "Cutting over from a `/opt/laboratory` clone" sections describe the
SSH-based manual path; they stay as documented fallback/reference (per Step 2's new section
explicitly saying so) rather than being deleted — someone with SSH access to the box may still
reasonably use them. "Suspend when idle" still correctly points at `playground suspend` for
idling the droplet down between uses — `deploy.sh` replaces *build/upgrade*, not the separate
cost-management suspend workflow, and that's out of this plan's scope. No edits needed beyond
Steps 1–2. This was confirmed by reading the full file during planning, not left as a TODO.

- [ ] **Step 4: Commit**

```bash
git add deploy/scheduler/README.md
git commit -m "docs(scheduler): document deploy.sh as the primary redeploy path"
```

---

### Task 7: CLAUDE.md + CHANGELOG

**Files:**
- Modify: `CLAUDE.md`
- Modify: `CHANGELOG.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add a CLAUDE.md key-fact bullet**

In `CLAUDE.md`, under "## Key facts", add a bullet near the existing "**Deferred scheduling:**"
bullet (keep it adjacent — a future reader looking for scheduler facts will look there first):

```markdown
- **Scheduler redeploy:** `deploy/scheduler/deploy.sh vX.Y.Z` — an immutable blue-green droplet
  swap (build new, verify with a real smoke job, retire old), replacing playground's Ansible role
  (which drifted for 2.5 months undetected — see `docs/superpowers/specs/
  2026-08-27-scheduler-deploy-cutover-design.md`). No SSH involved. `lab queue wait-drain` is the
  safety gate it waits on before pausing the queue — drain **before** pause, never the reverse
  (pausing stops the sync that would let a drain-wait ever observe real progress).
```

- [ ] **Step 2: Add a CHANGELOG entry**

In `CHANGELOG.md`, add an entry under the current unreleased/"Unreleased" section (match this
file's existing heading style exactly — read the top of the file first to copy the right format;
do not invent a new heading shape):

```markdown
- `deploy/scheduler/deploy.sh`: redeploy the scheduler droplet via an immutable blue-green swap
  instead of playground's Ansible role. `lab queue list` gains a `host` field; new `lab queue
  wait-drain` command.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md CHANGELOG.md
git commit -m "docs: note the scheduler redeploy mechanism"
```

---

### Task 8: Live verification — HUMAN-GATED, NOT FOR AUTONOMOUS EXECUTION

**Do not run this task as part of an unattended plan execution.** It creates and destroys real
DigitalOcean droplets and launches a real (cheap) Vast/DO rental — it costs money and touches
production-adjacent infrastructure. Stop after Task 7, report the plan as code-complete, and wait
for the user to explicitly say to proceed with this task in real time.

**Files:** none new — this exercises Tasks 1–6 end to end.

- [ ] **Step 1 (human-gated): Isolated dry run**

Point `deploy.sh` at a throwaway queue and a throwaway droplet — never the production R2 bucket,
never the real `lab-scheduler` droplet:

```bash
export LAB_QUEUE_DIR=/tmp/lab-scheduler-test-queue   # or a separate R2 prefix, per taste
deploy/scheduler/deploy.sh v0.9.0   # or whatever the current latest tag is
```

Confirm: the new test droplet comes up, `lab queue list` (against the test queue) shows its
`host`, the smoke registration reaches `succeeded`, and the (fake, throwaway) "old" droplet
handling behaves as designed. Manually `doctl compute droplet delete` anything left over from
this run — it is not production and does not need to survive.

- [ ] **Step 2 (human-gated): The real cutover**

Only after Step 1 succeeds cleanly. Confirm with the user immediately before running:

```bash
deploy/scheduler/deploy.sh v0.9.0   # or whatever tag is current at the time
```

This is the actual production cutover — it takes the live scheduler off its stale pre-v0.5.0 pin
for the first time since 2026-06-11. Watch it through all 8 steps; do not walk away mid-run.

- [ ] **Step 3 (human-gated): Confirm the result**

```bash
uv run lab queue list        # heartbeat_age_s low, host is the new droplet name
uv run lab reconcile         # no orphans (the old droplet should be gone)
```

Report the new droplet's name/tag back to the user and update the CLAUDE.md bullet from Task 7 if
anything about the live behavior differed from what was documented.
