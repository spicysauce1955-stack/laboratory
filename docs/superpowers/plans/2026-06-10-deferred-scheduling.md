# Deferred Experiment Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `lab register` queues experiments with triggers (nightly window, Vast price threshold, dependency chaining) and guardrails (expiry, cost caps); an idempotent `lab scheduler tick` — run by a systemd timer on an always-on host — launches them via the existing backends.

**Architecture:** New `src/lab/scheduler/` package (models / queue store / bundle / price / tick / register), with CLI + MCP as thin shells, per the approved spec `docs/superpowers/specs/2026-06-10-deferred-scheduling-design.md`. State lives in a QueueStore (local dir for tests/laptop, R2 for production). All scheduling logic is in `Scheduler.tick()`, dependency-injected (clock, price feed, lab factory) for table-driven unit tests.

**Tech Stack:** Python 3.12, Pydantic v2, Typer CLI, FastMCP, boto3 (R2), vastai-sdk (price feed), stdlib `tarfile`/`zoneinfo`. No new dependencies.

**Conventions (project-wide, apply to every task):** ruff line length 100; `mypy --strict` must pass on `src/lab`; run checks with:
```bash
uv run pytest tests/<file> -q          # tests
uv run ruff check src/lab tests        # lint
uv run mypy src/lab                    # types (strict)
```
Existing test helpers live in `tests/helpers.py` (`make_manifest`, `wait_terminal`, `PYTHON`). Tests import them as `from helpers import ...` (tests dir is on the path).

---

## File structure

| File | Responsibility |
|---|---|
| `src/lab/scheduler/__init__.py` | re-exports (`Scheduler`, `Registration`, `register`, …) |
| `src/lab/scheduler/models.py` | `DailyWindow`, `Triggers`, `Guardrails`, `RegState`, `Registration`, `ControlConfig`, `TickReport` |
| `src/lab/scheduler/bundle.py` | `create_bundle` (git archive + dirty diff + untracked), `extract_bundle` |
| `src/lab/scheduler/queue.py` | `QueueStore` protocol, `LocalQueueStore`, `default_queue` |
| `src/lab/scheduler/r2queue.py` | `R2QueueStore` over extended `R2Store` |
| `src/lab/scheduler/price.py` | `PriceFeed` protocol, `offer_query`, `VastPriceFeed` |
| `src/lab/scheduler/tick.py` | `Scheduler` class — the whole tick brain |
| `src/lab/scheduler/register.py` | `register()`, `worst_case_cost()`, reg-id minting |
| `src/lab/core.py` (modify) | `Lab.submit(..., code=, registration_id=)` override |
| `src/lab/models.py` (modify) | `JobManifest.registration_id` field |
| `src/lab/storage.py` (modify) | generic object ops on `R2Store` (put/get text, list, delete, file up/down) |
| `src/lab/sky_runner.py` (modify) | `--adopt` mode (re-attach to a live cluster) |
| `src/lab/cli.py` (modify) | `lab register`, `lab queue …`, `lab scheduler tick`, status fallback |
| `src/lab/mcp_server.py` (modify) | `register`, `queue_list`, `queue_show`, `queue_cancel`, `queue_pause` tools |
| `deploy/scheduler/` | systemd units + env template + deploy README |
| `tests/test_scheduler_*.py` | one test file per module |

---

### Task 1: Scheduler data models

**Files:**
- Create: `src/lab/scheduler/__init__.py`, `src/lab/scheduler/models.py`
- Test: `tests/test_scheduler_models.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Scheduler data models: window containment (incl. midnight crossing + tz), defaults."""

from datetime import datetime, time, timezone

from lab.scheduler.models import (
    ControlConfig,
    DailyWindow,
    Guardrails,
    Registration,
    RegState,
    Triggers,
)
from lab.models import CodeRef, JobSpec


def _utc(h: int, m: int = 0) -> datetime:
    return datetime(2026, 6, 10, h, m, tzinfo=timezone.utc)


def test_window_same_day():
    w = DailyWindow(start=time(9, 0), end=time(17, 0), tz="UTC")
    assert w.contains(_utc(12))
    assert not w.contains(_utc(8, 59))
    assert not w.contains(_utc(17, 0))  # end-exclusive


def test_window_crosses_midnight():
    w = DailyWindow(start=time(23, 0), end=time(7, 0), tz="UTC")
    assert w.contains(_utc(23, 30))
    assert w.contains(_utc(2))
    assert not w.contains(_utc(12))


def test_window_respects_timezone():
    # 02:00 UTC == 22:00 previous day in New York -> outside a 23:00-07:00 NY window
    w = DailyWindow(start=time(23, 0), end=time(7, 0), tz="America/New_York")
    assert not w.contains(_utc(2))
    assert w.contains(_utc(4))  # 00:00 NY


def test_registration_roundtrip_and_defaults():
    reg = Registration(
        reg_id="reg-1",
        created_at=_utc(0),
        spec=JobSpec(command="python x.py"),
        guardrails=Guardrails(expires_at=_utc(23)),
        bundle_key="queue/bundles/reg-1.tar.gz",
        code=CodeRef(git_commit="0" * 40, git_dirty=False),
    )
    assert reg.state is RegState.pending
    assert reg.triggers.after == []
    again = Registration.model_validate_json(reg.model_dump_json())
    assert again == reg


def test_control_defaults():
    c = ControlConfig()
    assert (c.paused, c.max_concurrent, c.budget_usd_per_day, c.auto_reconcile) == (
        False, 4, None, False,
    )


def test_triggers_all_optional():
    assert Triggers() == Triggers(
        not_before=None, window=None, max_hourly_usd=None, offer_query=None, after=[]
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_models.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.scheduler'`

- [ ] **Step 3: Write the implementation**

`src/lab/scheduler/__init__.py`:
```python
"""Deferred experiment scheduling (spec: docs/superpowers/specs/2026-06-10-deferred-scheduling-design.md)."""
```

`src/lab/scheduler/models.py`:
```python
"""Scheduler data model — registrations, triggers, guardrails (spec §3).

A Registration wraps an ordinary :class:`lab.models.JobSpec` with launch *triggers* (AND
semantics; none present = eligible immediately) and *guardrails*. ``state`` is owned solely by
the scheduler tick; the laptop only writes cancel/hold markers (spec §5 single-writer rule).
"""

from __future__ import annotations

from datetime import datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from lab.models import CodeRef, JobSpec


class RegState(str, Enum):
    """Registration lifecycle (spec §3). ``held`` is derived from the laptop's hold marker."""

    pending = "pending"
    launching = "launching"
    launched = "launched"
    succeeded = "succeeded"
    failed = "failed"
    expired = "expired"
    cancelled = "cancelled"
    held = "held"


class DailyWindow(BaseModel):
    """Recurring daily eligibility window, tz-aware; may cross midnight. End-exclusive."""

    start: time
    end: time
    tz: str = "UTC"  # IANA name

    def contains(self, dt: datetime) -> bool:
        local = dt.astimezone(ZoneInfo(self.tz)).time()
        if self.start <= self.end:
            return self.start <= local < self.end
        return local >= self.start or local < self.end  # crosses midnight


class Triggers(BaseModel):
    """All present triggers must hold simultaneously (AND). No triggers = launch ASAP."""

    not_before: datetime | None = None
    window: DailyWindow | None = None
    max_hourly_usd: float | None = None  # gate on best matching Vast offer price
    offer_query: str | None = None  # extra vastai search filter
    after: list[str] = Field(default_factory=list)  # reg_ids that must reach `succeeded`


class Guardrails(BaseModel):
    expires_at: datetime  # required — past this the entry expires, never launches
    max_cost_usd: float | None = None  # per-job: best hourly x timeout must fit


class Registration(BaseModel):
    reg_id: str
    created_at: datetime
    spec: JobSpec
    triggers: Triggers = Field(default_factory=Triggers)
    guardrails: Guardrails
    bundle_key: str
    code: CodeRef  # commit + dirty captured at registration (provenance, FR-B1)
    state: RegState = RegState.pending
    job_id: str | None = None
    launched_at: datetime | None = None
    state_changed_at: datetime | None = None  # drives orphaned-`launching` repair (spec §5)
    last_skip_reason: str | None = None


class ControlConfig(BaseModel):
    """Global scheduler switchboard — ``queue/control.json`` (laptop-owned)."""

    paused: bool = False
    budget_usd_per_day: float | None = None  # trailing-24h estimated-spend cap
    max_concurrent: int = 4
    auto_reconcile: bool = False  # let the periodic sweep destroy confirmed orphans


class TickReport(BaseModel):
    """Structured outcome of one tick (returned + summarized into the heartbeat)."""

    at: datetime
    launched: list[str] = Field(default_factory=list)
    expired: list[str] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)  # reg_id -> reason
    synced: dict[str, str] = Field(default_factory=dict)  # reg_id -> new state
    errors: list[str] = Field(default_factory=list)
    reconcile: dict[str, object] | None = None
```

- [ ] **Step 4: Run tests + checks, verify pass**

Run: `uv run pytest tests/test_scheduler_models.py -q && uv run ruff check src/lab tests && uv run mypy src/lab`
Expected: all PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler tests/test_scheduler_models.py
git commit -m "feat(scheduler): registration/trigger/guardrail data models"
```

---

### Task 2: Code bundles (create/extract)

**Files:**
- Create: `src/lab/scheduler/bundle.py`
- Test: `tests/test_scheduler_bundle.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Bundle = exact runnable tree: committed files + dirty diff + untracked (non-ignored) files."""

import subprocess
from pathlib import Path

from lab.scheduler.bundle import create_bundle, extract_bundle


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "uv.lock").write_text("lockfile\n")
    (repo / "exp.py").write_text("print('v1')\n")
    (repo / ".gitignore").write_text("runs/\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def test_clean_tree_bundle_roundtrip(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tar, code = create_bundle(repo, tmp_path / "out")
    assert tar.exists() and tar.suffixes[-2:] == [".tar", ".gz"]
    assert not code.git_dirty and len(code.git_commit) == 40
    dest = extract_bundle(tar, tmp_path / "x")
    assert (dest / "exp.py").read_text() == "print('v1')\n"
    assert (dest / "uv.lock").exists()


def test_dirty_and_untracked_files_included(tmp_path: Path):
    repo = _make_repo(tmp_path)
    (repo / "exp.py").write_text("print('v2')\n")  # modified tracked
    (repo / "new_exp.py").write_text("print('new')\n")  # untracked
    (repo / "runs").mkdir()
    (repo / "runs" / "big.bin").write_text("x" * 10)  # ignored -> excluded
    tar, code = create_bundle(repo, tmp_path / "out")
    assert code.git_dirty
    dest = extract_bundle(tar, tmp_path / "x")
    assert (dest / "exp.py").read_text() == "print('v2')\n"
    assert (dest / "new_exp.py").read_text() == "print('new')\n"
    assert not (dest / "runs").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_bundle.py -q`
Expected: FAIL — `ModuleNotFoundError` (no `lab.scheduler.bundle`)

- [ ] **Step 3: Write the implementation**

`src/lab/scheduler/bundle.py`:
```python
"""Code bundles — the exact tree a deferred job will run (spec §2 'code delivery').

``create_bundle`` snapshots committed tree + dirty diff + untracked non-ignored files into a
``.tar.gz`` so the scheduler host can run unpushed/dirty work; provenance (commit + dirty flag)
rides separately in the registration's :class:`~lab.models.CodeRef` (FR-B1).
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from lab.manifest import current_commit, is_dirty
from lab.models import CodeRef


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def create_bundle(repo: Path, dest_dir: Path) -> tuple[Path, CodeRef]:
    """Snapshot ``repo`` into ``dest_dir/<commit12>[-dirty].tar.gz``; returns (path, CodeRef)."""
    repo = Path(repo)
    commit = current_commit(repo)
    dirty = is_dirty(repo)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / f"{commit[:12]}{'-dirty' if dirty else ''}.tar.gz"
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "tree"
        stage.mkdir()
        # Committed tree.
        archive = _git(repo, "archive", "--format=tar", commit)
        with tempfile.NamedTemporaryFile(suffix=".tar") as tf:
            tf.write(archive)
            tf.flush()
            with tarfile.open(tf.name) as t:
                t.extractall(stage, filter="data")
        if dirty:
            # Modified/deleted tracked files: apply the diff onto the staged tree.
            patch = _git(repo, "diff", "HEAD", "--binary")
            if patch.strip():
                subprocess.run(
                    ["git", "apply", "--whitespace=nowarn"],
                    input=patch, cwd=stage, check=True,
                )
            # Untracked, non-ignored files (e.g. a brand-new experiment script).
            names = _git(repo, "ls-files", "--others", "--exclude-standard", "-z").decode()
            for rel in filter(None, names.split("\0")):
                src = repo / rel
                dst = stage / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        with tarfile.open(tar_path, "w:gz") as out:
            out.add(stage, arcname=".")
    return tar_path, CodeRef(git_commit=commit, git_dirty=dirty)


def extract_bundle(tar_path: Path, dest: Path) -> Path:
    """Extract a bundle into ``dest`` (created if missing); returns ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(dest, filter="data")
    return dest
```

- [ ] **Step 4: Run tests + checks, verify pass**

Run: `uv run pytest tests/test_scheduler_bundle.py -q && uv run ruff check src/lab tests && uv run mypy src/lab`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/bundle.py tests/test_scheduler_bundle.py
git commit -m "feat(scheduler): code bundles (committed tree + dirty diff + untracked)"
```

---

### Task 3: LocalQueueStore

**Files:**
- Create: `src/lab/scheduler/queue.py`
- Test: `tests/test_scheduler_queue.py`

- [ ] **Step 1: Write the failing tests**

```python
"""QueueStore contract via the local-dir implementation (R2 implements the same protocol)."""

from datetime import datetime, timezone
from pathlib import Path

from helpers import make_manifest
from lab.models import CodeRef, JobSpec
from lab.scheduler.models import ControlConfig, Guardrails, Registration, RegState
from lab.scheduler.queue import LocalQueueStore


def _reg(reg_id: str) -> Registration:
    return Registration(
        reg_id=reg_id,
        created_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
        spec=JobSpec(command="python x.py"),
        guardrails=Guardrails(expires_at=datetime(2026, 6, 11, tzinfo=timezone.utc)),
        bundle_key=f"bundles/{reg_id}.tar.gz",
        code=CodeRef(git_commit="0" * 40, git_dirty=False),
    )


def test_entry_crud_and_listing(tmp_path: Path):
    q = LocalQueueStore(tmp_path)
    q.put_entry(_reg("reg-b"))
    q.put_entry(_reg("reg-a"))
    assert [r.reg_id for r in q.list_entries()] == ["reg-a", "reg-b"]  # sorted
    r = q.get_entry("reg-a")
    r = r.model_copy(update={"state": RegState.launched, "job_id": "j1"})
    q.put_entry(r)
    assert q.get_entry("reg-a").job_id == "j1"


def test_control_default_and_roundtrip(tmp_path: Path):
    q = LocalQueueStore(tmp_path)
    assert q.read_control() == ControlConfig()  # missing file -> defaults
    q.write_control(ControlConfig(paused=True, budget_usd_per_day=5.0))
    assert q.read_control().paused is True


def test_heartbeat(tmp_path: Path):
    q = LocalQueueStore(tmp_path)
    assert q.read_heartbeat() is None
    q.write_heartbeat({"at": "2026-06-10T00:00:00Z", "tick_count": 3})
    hb = q.read_heartbeat()
    assert hb is not None and hb["tick_count"] == 3


def test_markers(tmp_path: Path):
    q = LocalQueueStore(tmp_path)
    assert not q.cancel_requested("reg-a")
    q.request_cancel("reg-a")
    assert q.cancel_requested("reg-a")
    q.hold("reg-a")
    assert q.held("reg-a")
    q.release("reg-a")
    assert not q.held("reg-a")


def test_bundle_roundtrip(tmp_path: Path):
    q = LocalQueueStore(tmp_path / "q")
    src = tmp_path / "code.tar.gz"
    src.write_bytes(b"tarball-bytes")
    key = q.put_bundle("reg-a", src)
    assert key.endswith("reg-a.tar.gz")
    out = q.fetch_bundle("reg-a", tmp_path / "dl")
    assert out.read_bytes() == b"tarball-bytes"


def test_manifest_mirror(tmp_path: Path):
    q = LocalQueueStore(tmp_path)
    assert q.read_mirrored("j1") is None
    m = make_manifest("j1", "python x.py")
    q.mirror_manifest(m)
    got = q.read_mirrored("j1")
    assert got is not None and got.job_id == "j1"
    assert [x.job_id for x in q.list_mirrored()] == ["j1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_queue.py -q`
Expected: FAIL — `ModuleNotFoundError` (no `lab.scheduler.queue`)

- [ ] **Step 3: Write the implementation**

`src/lab/scheduler/queue.py`:
```python
"""Queue persistence — the bus between laptop and scheduler host (spec §2).

Layout under a root (local dir here; same keys under an R2 prefix in r2queue.py):
    entries/<reg_id>.json     full Registration incl. state (scheduler-owned mutations)
    bundles/<reg_id>.tar.gz   code snapshot
    jobs/<job_id>.json        mirrored JobManifests of scheduler-launched jobs (spec §4.3)
    cancelled/<reg_id>        laptop-owned cancel markers
    held/<reg_id>             laptop-owned hold markers
    control.json              ControlConfig
    heartbeat.json            liveness + tick counter
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lab.models import JobManifest
from lab.scheduler.models import ControlConfig, Registration


@runtime_checkable
class QueueStore(Protocol):
    def put_entry(self, reg: Registration) -> None: ...
    def get_entry(self, reg_id: str) -> Registration: ...
    def list_entries(self) -> list[Registration]: ...
    def read_control(self) -> ControlConfig: ...
    def write_control(self, control: ControlConfig) -> None: ...
    def read_heartbeat(self) -> dict[str, Any] | None: ...
    def write_heartbeat(self, data: dict[str, Any]) -> None: ...
    def request_cancel(self, reg_id: str) -> None: ...
    def cancel_requested(self, reg_id: str) -> bool: ...
    def hold(self, reg_id: str) -> None: ...
    def release(self, reg_id: str) -> None: ...
    def held(self, reg_id: str) -> bool: ...
    def put_bundle(self, reg_id: str, src: Path) -> str: ...
    def fetch_bundle(self, reg_id: str, dest_dir: Path) -> Path: ...
    def mirror_manifest(self, manifest: JobManifest) -> None: ...
    def read_mirrored(self, job_id: str) -> JobManifest | None: ...
    def list_mirrored(self) -> list[JobManifest]: ...


class LocalQueueStore:
    """Filesystem QueueStore — tests, laptop-only mode, and the layout reference."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- entries -------------------------------------------------------------
    def _entry_path(self, reg_id: str) -> Path:
        return self.root / "entries" / f"{reg_id}.json"

    def put_entry(self, reg: Registration) -> None:
        self._atomic_write(self._entry_path(reg.reg_id), reg.model_dump_json(indent=2))

    def get_entry(self, reg_id: str) -> Registration:
        return Registration.model_validate_json(self._entry_path(reg_id).read_text())

    def list_entries(self) -> list[Registration]:
        d = self.root / "entries"
        if not d.exists():
            return []
        return [
            Registration.model_validate_json(p.read_text()) for p in sorted(d.glob("*.json"))
        ]

    # -- control / heartbeat ---------------------------------------------------
    def read_control(self) -> ControlConfig:
        p = self.root / "control.json"
        return ControlConfig.model_validate_json(p.read_text()) if p.exists() else ControlConfig()

    def write_control(self, control: ControlConfig) -> None:
        self._atomic_write(self.root / "control.json", control.model_dump_json(indent=2))

    def read_heartbeat(self) -> dict[str, Any] | None:
        p = self.root / "heartbeat.json"
        if not p.exists():
            return None
        loaded: dict[str, Any] = json.loads(p.read_text())
        return loaded

    def write_heartbeat(self, data: dict[str, Any]) -> None:
        self._atomic_write(self.root / "heartbeat.json", json.dumps(data, default=str))

    # -- laptop-owned markers (spec §5 single-writer rule) ----------------------
    def _marker(self, kind: str, reg_id: str) -> Path:
        return self.root / kind / reg_id

    def request_cancel(self, reg_id: str) -> None:
        self._atomic_write(self._marker("cancelled", reg_id), "")

    def cancel_requested(self, reg_id: str) -> bool:
        return self._marker("cancelled", reg_id).exists()

    def hold(self, reg_id: str) -> None:
        self._atomic_write(self._marker("held", reg_id), "")

    def release(self, reg_id: str) -> None:
        self._marker("held", reg_id).unlink(missing_ok=True)

    def held(self, reg_id: str) -> bool:
        return self._marker("held", reg_id).exists()

    # -- bundles ----------------------------------------------------------------
    def put_bundle(self, reg_id: str, src: Path) -> str:
        dest = self.root / "bundles" / f"{reg_id}.tar.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return f"bundles/{reg_id}.tar.gz"

    def fetch_bundle(self, reg_id: str, dest_dir: Path) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / f"{reg_id}.tar.gz"
        shutil.copy2(self.root / "bundles" / f"{reg_id}.tar.gz", out)
        return out

    # -- mirrored job manifests (spec §4.3) ---------------------------------------
    def mirror_manifest(self, manifest: JobManifest) -> None:
        self._atomic_write(
            self.root / "jobs" / f"{manifest.job_id}.json", manifest.model_dump_json(indent=2)
        )

    def read_mirrored(self, job_id: str) -> JobManifest | None:
        p = self.root / "jobs" / f"{job_id}.json"
        return JobManifest.model_validate_json(p.read_text()) if p.exists() else None

    def list_mirrored(self) -> list[JobManifest]:
        d = self.root / "jobs"
        if not d.exists():
            return []
        return [JobManifest.model_validate_json(p.read_text()) for p in sorted(d.glob("*.json"))]

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
```

- [ ] **Step 4: Run tests + checks, verify pass**

Run: `uv run pytest tests/test_scheduler_queue.py -q && uv run ruff check src/lab tests && uv run mypy src/lab`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/queue.py tests/test_scheduler_queue.py
git commit -m "feat(scheduler): QueueStore protocol + local-dir implementation"
```

---

### Task 4: `Lab.submit` code override + `registration_id` on the manifest

The scheduler submits from an extracted bundle, which is **not a git repo** — `Lab.submit` must accept pre-captured provenance instead of introspecting git. `registration_id` on the manifest makes a crashed `launching` repairable (find the job a registration produced).

**Files:**
- Modify: `src/lab/models.py` (JobManifest), `src/lab/core.py:94-122` (submit)
- Test: `tests/test_core.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_core.py`)

```python
def test_submit_with_code_override_skips_git(tmp_path: Path):
    """A pre-captured CodeRef lets submit run from a non-git dir (scheduler bundles)."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "uv.lock").write_text("lock")
    home = tmp_path / "runs"
    lab = Lab(backend=LocalBackend(home=home, repo=bundle), repo=bundle, home=home)
    code = CodeRef(git_commit="a" * 40, git_dirty=True)
    job_id = lab.submit(
        JobSpec(command=f"{PYTHON} -c 'print(1)'"), code=code, registration_id="reg-7"
    )
    m = lab.manifest(job_id)
    assert m.code == code
    assert m.registration_id == "reg-7"
```

Add the needed imports at the top of the test file if missing: `from lab.backends.local import LocalBackend`, `from lab.core import Lab`, `from lab.models import CodeRef, JobSpec`, `from helpers import PYTHON`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core.py::test_submit_with_code_override_skips_git -q`
Expected: FAIL — `TypeError: Lab.submit() got an unexpected keyword argument 'code'`

- [ ] **Step 3: Implement**

In `src/lab/models.py`, add to `JobManifest` (after `sweep_id`):
```python
    registration_id: str | None = None  # set when launched by the scheduler (spec §4.5 repair)
```

In `src/lab/core.py`, change `submit`:
```python
    def submit(
        self,
        spec: JobSpec,
        *,
        allow_dirty: bool = True,
        sweep_id: str | None = None,
        code: CodeRef | None = None,
        registration_id: str | None = None,
    ) -> str:
        """Build + persist the manifest, then launch via the backend (FR-A1, FR-B).

        ``code`` overrides git introspection — used by the scheduler, which submits from an
        extracted bundle (not a git repo) with provenance captured at registration time.
        """
        if code is None:
            dirty = is_dirty(self.repo)
            if dirty and not allow_dirty:
                raise LabError("working tree is dirty; commit or pass allow_dirty=True (FR-B1)")
            code = CodeRef(git_commit=current_commit(self.repo), git_dirty=dirty)
        elif code.git_dirty and not allow_dirty:
            raise LabError("bundle captured a dirty tree but allow_dirty=False (FR-B1)")
        seed = spec.seed if spec.seed is not None else 0  # explicit + recorded (FR-B4)
        job_id = _new_job_id()
        manifest = JobManifest(
            job_id=job_id,
            sweep_id=sweep_id,
            registration_id=registration_id,
            created_at=now(),
            submitted_by=spec.submitted_by,
            code=code,
            env=EnvInfo(
                uv_lock_sha256=uv_lock_sha256(self.repo / "uv.lock"),
                python_version=platform.python_version(),
            ),
            run=RunSpec(
                entrypoint_command=spec.command,
                resolved_config=spec.config or {},
                seed=seed,
            ),
            resources=spec.resources,
            backend=BackendInfo(provisioner=self.backend.name),
            status=JobState.queued,
        )
        self.store.create(manifest)
        self.backend.submit(manifest)
        return job_id
```

- [ ] **Step 4: Run the whole core suite, verify pass**

Run: `uv run pytest tests/test_core.py -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean (no existing test relied on the old positional behavior — `code`/`registration_id` are keyword-only with defaults).

- [ ] **Step 5: Commit**

```bash
git add src/lab/models.py src/lab/core.py tests/test_core.py
git commit -m "feat(core): submit accepts pre-captured CodeRef + registration_id (scheduler launches)"
```

---

### Task 5: Scheduler skeleton — heartbeat, pause, expiry

**Files:**
- Create: `src/lab/scheduler/tick.py`
- Test: `tests/test_scheduler_tick.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tick brain — table-driven with fake clock; LocalBackend does real (tiny) launches."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import PYTHON
from lab.models import JobSpec
from lab.scheduler.bundle import create_bundle
from lab.scheduler.models import ControlConfig, Guardrails, Registration, RegState, Triggers
from lab.scheduler.queue import LocalQueueStore
from lab.scheduler.tick import Scheduler

T0 = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self, t: datetime = T0) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def make_sched(tmp_path: Path, clock: FakeClock | None = None, **kw) -> tuple[Scheduler, LocalQueueStore]:
    q = LocalQueueStore(tmp_path / "queue")
    sched = Scheduler(q, home=tmp_path / "runs", now_fn=clock or FakeClock(), **kw)
    return sched, q


def put_reg(
    q: LocalQueueStore,
    tmp_path: Path,
    reg_id: str,
    *,
    command: str | None = None,
    triggers: Triggers | None = None,
    expires: datetime | None = None,
    state: RegState = RegState.pending,
    job_id: str | None = None,
    max_cost: float | None = None,
) -> Registration:
    """A registration whose bundle is a real (tiny) git repo snapshot."""
    import subprocess

    repo = tmp_path / f"src-{reg_id}"
    if not repo.exists():
        repo.mkdir()
        for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
        (repo / "uv.lock").write_text("lock")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-qm", "x"], check=True, capture_output=True
        )
    tar, code = create_bundle(repo, tmp_path / "tars")
    q.put_bundle(reg_id, tar)
    reg = Registration(
        reg_id=reg_id,
        created_at=T0,
        spec=JobSpec(command=command or f"{PYTHON} -c 'print(42)'"),
        triggers=triggers or Triggers(),
        guardrails=Guardrails(expires_at=expires or T0 + timedelta(days=1), max_cost_usd=max_cost),
        bundle_key=f"bundles/{reg_id}.tar.gz",
        code=code,
        state=state,
        job_id=job_id,
    )
    q.put_entry(reg)
    return reg


def test_heartbeat_written_even_when_paused(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    q.write_control(ControlConfig(paused=True))
    put_reg(q, tmp_path, "reg-a")
    rep = sched.tick()
    assert rep.launched == []
    hb = q.read_heartbeat()
    assert hb is not None and hb["tick_count"] == 1
    assert q.get_entry("reg-a").state is RegState.pending  # untouched while paused


def test_tick_count_increments(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    sched.tick()
    sched.tick()
    hb = q.read_heartbeat()
    assert hb is not None and hb["tick_count"] == 2


def test_expiry(tmp_path: Path):
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    put_reg(q, tmp_path, "reg-a", expires=T0 + timedelta(hours=1))
    clock.t = T0 + timedelta(hours=2)
    rep = sched.tick()
    assert rep.expired == ["reg-a"]
    e = q.get_entry("reg-a")
    assert e.state is RegState.expired and e.last_skip_reason is not None


def test_held_entries_still_expire_but_never_launch(tmp_path: Path):
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    put_reg(q, tmp_path, "reg-a", expires=T0 + timedelta(hours=1))
    q.hold("reg-a")
    rep = sched.tick()
    assert rep.launched == [] and rep.skipped["reg-a"] == "held"
    clock.t = T0 + timedelta(hours=2)
    sched.tick()
    assert q.get_entry("reg-a").state is RegState.expired
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_tick.py -q`
Expected: FAIL — `ModuleNotFoundError` (no `lab.scheduler.tick`)

- [ ] **Step 3: Write the implementation**

`src/lab/scheduler/tick.py` (skeleton — launch/triggers/sync arrive in Tasks 6–9; keep the marked seams):
```python
"""The scheduler brain — one idempotent ``tick()`` (spec §4).

Every tick re-derives everything from the QueueStore; a crashed tick costs nothing. Dependencies
(clock, price feed, Lab construction) are injected so tests run table-driven with fakes.
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from lab._util import now
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.scheduler.models import Registration, RegState, TickReport
from lab.scheduler.queue import QueueStore
from lab.store import JobStore


class Scheduler:
    def __init__(
        self,
        queue: QueueStore,
        home: Path,
        *,
        backend: str = "local",
        now_fn: Callable[[], datetime] = now,
        host: str | None = None,
    ) -> None:
        self.queue = queue
        self.home = Path(home)
        self.backend_name = backend
        self.now_fn = now_fn
        self.host = host or platform.node()
        self.store = JobStore(self.home)

    def make_lab(self, repo: Path) -> Lab:
        """Lab over the launch backend, rooted at an extracted bundle. Test seam."""
        if self.backend_name == "skypilot":
            from lab.backends.skypilot import SkyPilotBackend

            return Lab(backend=SkyPilotBackend(home=self.home, repo=repo), repo=repo, home=self.home)
        return Lab(backend=LocalBackend(home=self.home, repo=repo), repo=repo, home=self.home)

    def tick(self) -> TickReport:
        rep = TickReport(at=self.now_fn())
        tick_count = int((self.queue.read_heartbeat() or {}).get("tick_count", 0)) + 1
        control = self.queue.read_control()
        if not control.paused:
            entries = self.queue.list_entries()
            self._sync(entries, rep)            # Task 7
            self._expire(entries, rep)
            self._evaluate_and_launch(entries, control, rep)  # Task 6 (+8, 9)
        self.queue.write_heartbeat(
            {
                "at": rep.at.isoformat(),
                "host": self.host,
                "tick_count": tick_count,
                "launched": rep.launched,
                "errors": rep.errors,
            }
        )
        return rep

    # ------------------------------------------------------------------ phases
    def _sync(self, entries: list[Registration], rep: TickReport) -> None:
        """Mirror launched jobs' state back onto registrations (Task 7)."""

    def _expire(self, entries: list[Registration], rep: TickReport) -> None:
        nowt = self.now_fn()
        for reg in entries:
            if reg.state is RegState.pending and nowt >= reg.guardrails.expires_at:
                self._transition(reg, RegState.expired, reason="expired (run-by deadline passed)")
                rep.expired.append(reg.reg_id)

    def _evaluate_and_launch(
        self, entries: list[Registration], control: object, rep: TickReport
    ) -> None:
        for reg in entries:
            if reg.state is not RegState.pending:
                continue
            if self.queue.held(reg.reg_id):
                rep.skipped[reg.reg_id] = "held"
                continue
            # Trigger/guardrail evaluation + launch land in Tasks 6, 8, 9.

    def _transition(self, reg: Registration, state: RegState, *, reason: str | None = None,
                    **extra: object) -> Registration:
        updated = reg.model_copy(
            update={
                "state": state,
                "state_changed_at": self.now_fn(),
                "last_skip_reason": reason,
                **extra,
            }
        )
        self.queue.put_entry(updated)
        return updated
```

Note for the implementer: `_expire` runs on the in-memory `entries` list **before** evaluation, and `_transition` writes through to the store — re-fetch with `self.queue.get_entry()` inside later phases if you need the fresh copy. Held entries stay `pending` in the store (hold is a laptop-owned marker), so expiry naturally applies to them.

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_scheduler_tick.py -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/tick.py tests/test_scheduler_tick.py
git commit -m "feat(scheduler): tick skeleton — heartbeat, pause, expiry, hold markers"
```

---

### Task 6: Clock & dependency triggers; launching (local backend); cancel race; `launching` repair

**Files:**
- Modify: `src/lab/scheduler/tick.py`
- Test: `tests/test_scheduler_tick.py` (append)

- [ ] **Step 1: Write the failing tests** (append; also extend the test file's imports with `from helpers import wait_terminal` and `from lab.backends.local import LocalBackend`)

```python
def test_no_triggers_launches_immediately(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a")
    rep = sched.tick()
    assert rep.launched == ["reg-a"]
    e = q.get_entry("reg-a")
    assert e.state is RegState.launched and e.job_id is not None
    # The job actually ran (LocalBackend, real subprocess).
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    assert wait_terminal(backend, e.job_id).value == "succeeded"


def test_not_before_gates(tmp_path: Path):
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    put_reg(q, tmp_path, "reg-a", triggers=Triggers(not_before=T0 + timedelta(hours=5)))
    assert sched.tick().launched == []
    assert "not_before" in q.get_entry("reg-a").last_skip_reason
    clock.t = T0 + timedelta(hours=6)
    assert sched.tick().launched == ["reg-a"]


def test_window_gates(tmp_path: Path):
    from datetime import time as dtime

    from lab.scheduler.models import DailyWindow

    clock = FakeClock()  # T0 = 12:00 UTC
    sched, q = make_sched(tmp_path, clock)
    w = DailyWindow(start=dtime(23, 0), end=dtime(7, 0), tz="UTC")
    put_reg(q, tmp_path, "reg-a", triggers=Triggers(window=w))
    assert sched.tick().launched == []
    clock.t = T0.replace(hour=23, minute=30)
    assert sched.tick().launched == ["reg-a"]


def test_dependency_waits_then_launches(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a")
    put_reg(q, tmp_path, "reg-b", triggers=Triggers(after=["reg-a"]))
    rep = sched.tick()
    assert rep.launched == ["reg-a"]  # b waits
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    wait_terminal(backend, q.get_entry("reg-a").job_id)
    sched.tick()  # syncs a -> succeeded (Task 7) ... then:
    rep3 = sched.tick()
    assert "reg-b" in rep3.launched or q.get_entry("reg-b").state is RegState.launched


def test_dependency_failure_cancels_dependent(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", state=RegState.failed)
    put_reg(q, tmp_path, "reg-b", triggers=Triggers(after=["reg-a"]))
    rep = sched.tick()
    assert rep.cancelled == ["reg-b"]
    e = q.get_entry("reg-b")
    assert e.state is RegState.cancelled and "reg-a" in (e.last_skip_reason or "")


def test_cancel_marker_blocks_launch(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a")
    q.request_cancel("reg-a")
    rep = sched.tick()
    assert rep.launched == [] and q.get_entry("reg-a").state is RegState.cancelled


def test_orphaned_launching_reverts_to_pending(tmp_path: Path):
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    reg = put_reg(q, tmp_path, "reg-a", state=RegState.launching)
    q.put_entry(reg.model_copy(update={"state_changed_at": T0 - timedelta(minutes=20)}))
    rep = sched.tick()
    # no job manifest carries registration_id reg-a -> the launch never happened
    assert q.get_entry("reg-a").state in (RegState.pending, RegState.launched)
    assert rep.launched == ["reg-a"] or q.get_entry("reg-a").state is RegState.pending


def test_orphaned_launching_with_existing_job_repairs_to_launched(tmp_path: Path):
    clock = FakeClock()
    sched, q = make_sched(tmp_path, clock)
    put_reg(q, tmp_path, "reg-a")
    sched.tick()
    job_id = q.get_entry("reg-a").job_id
    assert job_id is not None
    # Simulate the crash window: entry says launching/stale, but the job exists.
    broken = q.get_entry("reg-a").model_copy(
        update={
            "state": RegState.launching,
            "job_id": None,
            "state_changed_at": T0 - timedelta(minutes=20),
        }
    )
    q.put_entry(broken)
    sched.tick()
    repaired = q.get_entry("reg-a")
    assert repaired.state is RegState.launched and repaired.job_id == job_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_tick.py -q`
Expected: new tests FAIL (nothing launches yet); Task-5 tests still PASS.

- [ ] **Step 3: Implement** — replace `_evaluate_and_launch` and add helpers in `tick.py`:

```python
    LAUNCHING_ORPHAN_S = 600.0  # spec §5: launching older than 10 min is repaired

    def _evaluate_and_launch(
        self, entries: list[Registration], control: ControlConfig, rep: TickReport
    ) -> None:
        by_id = {r.reg_id: r for r in entries}
        for reg in entries:
            if reg.state is RegState.launching:
                self._repair_launching(reg, rep)
                continue
            if reg.state is not RegState.pending:
                continue
            reg = self.queue.get_entry(reg.reg_id)  # fresh copy (expiry may have written)
            if reg.state is not RegState.pending:
                continue
            if self.queue.cancel_requested(reg.reg_id):
                self._transition(reg, RegState.cancelled, reason="cancelled by user")
                rep.cancelled.append(reg.reg_id)
                continue
            if self.queue.held(reg.reg_id):
                rep.skipped[reg.reg_id] = "held"
                continue
            blocked = self._trigger_block(reg, by_id, rep)
            if blocked is not None:
                if blocked != "":  # "" => dependency hard-failure already handled
                    self.queue.put_entry(reg.model_copy(update={"last_skip_reason": blocked}))
                    rep.skipped[reg.reg_id] = blocked
                continue
            self._launch(reg, rep)

    def _trigger_block(
        self, reg: Registration, by_id: dict[str, Registration], rep: TickReport
    ) -> str | None:
        """None = all triggers hold; '' = entry was hard-cancelled; else the skip reason."""
        nowt = self.now_fn()
        t = reg.triggers
        if t.not_before is not None and nowt < t.not_before:
            return f"not_before {t.not_before.isoformat()}"
        if t.window is not None and not t.window.contains(nowt):
            return f"outside window {t.window.start}-{t.window.end} {t.window.tz}"
        for dep_id in t.after:
            dep = by_id.get(dep_id)
            dep_state = dep.state if dep is not None else None
            if dep_state in (RegState.failed, RegState.expired, RegState.cancelled) or dep is None:
                why = dep_state.value if dep_state else "missing"
                self._transition(reg, RegState.cancelled, reason=f"dependency {dep_id} ended {why}")
                rep.cancelled.append(reg.reg_id)
                return ""
            if dep_state is not RegState.succeeded:
                return f"waiting on dependency {dep_id} ({dep_state.value})"
        return None  # price + guardrails extend this in Tasks 8-9

    def _launch(self, reg: Registration, rep: TickReport) -> None:
        reg = self._transition(reg, RegState.launching, reason=None)
        if self.queue.cancel_requested(reg.reg_id):  # spec §5 cancel race: re-check pre-submit
            self._transition(reg, RegState.cancelled, reason="cancelled by user")
            rep.cancelled.append(reg.reg_id)
            return
        try:
            bundle_dir = self.home / "_bundles" / reg.reg_id
            tar = self.queue.fetch_bundle(reg.reg_id, self.home / "_bundles")
            from lab.scheduler.bundle import extract_bundle

            extract_bundle(tar, bundle_dir)
            lab = self.make_lab(bundle_dir)
            job_id = lab.submit(reg.spec, code=reg.code, registration_id=reg.reg_id)
        except Exception as e:  # noqa: BLE001 — a bad entry must not kill the tick (spec §5)
            self._transition(reg, RegState.pending, reason=f"launch error: {e}"[:300])
            rep.errors.append(f"{reg.reg_id}: {e}")
            return
        self._transition(
            reg, RegState.launched, reason=None, job_id=job_id, launched_at=self.now_fn()
        )
        rep.launched.append(reg.reg_id)

    def _repair_launching(self, reg: Registration, rep: TickReport) -> None:
        """A tick crashed mid-launch (spec §5): decide from evidence, after a grace period."""
        changed = reg.state_changed_at or reg.created_at
        if (self.now_fn() - changed).total_seconds() < self.LAUNCHING_ORPHAN_S:
            return
        job = next(
            (
                self.store.read_manifest(j)
                for j in self.store.list_job_ids()
                if self.store.read_manifest(j).registration_id == reg.reg_id
            ),
            None,
        )
        if job is not None:  # the submit happened -> repair forward
            self._transition(
                reg, RegState.launched, reason="repaired: launch had completed",
                job_id=job.job_id, launched_at=job.created_at,
            )
            rep.synced[reg.reg_id] = "launched (repaired)"
        else:
            self._transition(reg, RegState.pending, reason="repaired: launch never happened")
```

Add imports at the top of `tick.py`: `from lab.scheduler.models import ControlConfig` (extend the existing import line). Type the `control` parameter as `ControlConfig` (it was `object` in Task 5).

- [ ] **Step 4: Run tests, verify pass** (the two dependency tests need `_sync` from Task 7 to pass fully — implement Task 7 before expecting `test_dependency_waits_then_launches` to go green; run it as `uv run pytest tests/test_scheduler_tick.py -q --deselect tests/test_scheduler_tick.py::test_dependency_waits_then_launches` here, and without the deselect after Task 7.)

Run: `uv run pytest tests/test_scheduler_tick.py -q --deselect tests/test_scheduler_tick.py::test_dependency_waits_then_launches && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/tick.py tests/test_scheduler_tick.py
git commit -m "feat(scheduler): clock/dependency triggers, launch path, cancel race, launching repair"
```

---

### Task 7: Sync — terminal-state mirroring, manifest mirroring, cancel-of-launched

**Files:**
- Modify: `src/lab/scheduler/tick.py` (`_sync`)
- Test: `tests/test_scheduler_tick.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_sync_mirrors_terminal_state_and_manifest(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a")
    sched.tick()
    e = q.get_entry("reg-a")
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    wait_terminal(backend, e.job_id)
    rep = sched.tick()
    assert rep.synced.get("reg-a") == "succeeded"
    assert q.get_entry("reg-a").state is RegState.succeeded
    mirrored = q.read_mirrored(e.job_id)
    assert mirrored is not None and mirrored.status.value == "succeeded"


def test_sync_maps_failed_job_to_failed_reg(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command=f"{PYTHON} -c 'raise SystemExit(3)'")
    sched.tick()
    e = q.get_entry("reg-a")
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    wait_terminal(backend, e.job_id)
    sched.tick()
    assert q.get_entry("reg-a").state is RegState.failed


def test_cancel_marker_on_launched_cancels_the_job(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    put_reg(q, tmp_path, "reg-a", command=f"{PYTHON} -c 'import time; time.sleep(60)'")
    sched.tick()
    e = q.get_entry("reg-a")
    q.request_cancel("reg-a")
    sched.tick()
    assert q.get_entry("reg-a").state is RegState.cancelled
    backend = LocalBackend(home=tmp_path / "runs", repo=tmp_path)
    assert backend.status(e.job_id).value == "cancelled"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_tick.py -q`
Expected: the three new tests FAIL (`_sync` is a no-op).

- [ ] **Step 3: Implement `_sync`** in `tick.py`:

```python
    _TERMINAL_MAP = {
        JobState.succeeded: RegState.succeeded,
        JobState.failed: RegState.failed,
        JobState.timed_out: RegState.failed,
        JobState.cancelled: RegState.cancelled,
    }

    def _sync(self, entries: list[Registration], rep: TickReport) -> None:
        for reg in entries:
            if reg.state is not RegState.launched or reg.job_id is None:
                continue
            try:
                manifest = self.store.read_manifest(reg.job_id)
            except FileNotFoundError:
                rep.errors.append(f"{reg.reg_id}: manifest {reg.job_id} missing")
                continue
            if manifest.status not in self._TERMINAL_MAP and self.queue.cancel_requested(
                reg.reg_id
            ):
                lab = self.make_lab(self.home / "_bundles" / reg.reg_id)
                lab.cancel(reg.job_id)
                manifest = self.store.read_manifest(reg.job_id)
            self.queue.mirror_manifest(manifest)  # laptop visibility + stateless host (spec §4.3)
            new_state = self._TERMINAL_MAP.get(manifest.status)
            if new_state is not None:
                updated = self._transition(reg, new_state, reason=manifest.end_reason)
                entries[entries.index(reg)] = updated  # dependents see it this tick's eval
                rep.synced[reg.reg_id] = new_state.value
```

Add `from lab.models import JobState` to `tick.py` imports.

- [ ] **Step 4: Run the full tick suite (no deselect), verify pass**

Run: `uv run pytest tests/test_scheduler_tick.py -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS, including `test_dependency_waits_then_launches` from Task 6 / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/tick.py tests/test_scheduler_tick.py
git commit -m "feat(scheduler): sync launched jobs — terminal mirroring, manifest mirror, late cancel"
```

---

### Task 8: Price feed + price trigger

**Files:**
- Create: `src/lab/scheduler/price.py`
- Modify: `src/lab/scheduler/tick.py`
- Test: `tests/test_scheduler_price.py`, `tests/test_scheduler_tick.py` (append)

- [ ] **Step 1: Write the failing tests**

`tests/test_scheduler_price.py`:
```python
"""Offer-query derivation + VastPriceFeed over a faked vastai client."""

import lab.scheduler.price as price_mod
from lab.scheduler.price import VastPriceFeed, offer_query


def test_offer_query_from_accelerators():
    q = offer_query("RTX_4090:1", None)
    assert "gpu_name=RTX_4090" in q and "num_gpus>=1" in q
    assert "rentable=true" in q and "rented=false" in q and "reliability>0.95" in q


def test_offer_query_extra_filter_appended():
    assert offer_query("RTX_4090:2", "geolocation in [SE]").endswith("geolocation in [SE]")


def test_best_hourly_min_dph(monkeypatch):
    class FakeClient:
        def search_offers(self, query: str):
            return [{"dph_total": 0.31}, {"dph_total": 0.22}, {"dph_total": None}]

    monkeypatch.setattr(price_mod, "_get_vast_client", lambda: FakeClient())
    assert VastPriceFeed().best_hourly("RTX_4090:1") == 0.22


def test_best_hourly_no_offers(monkeypatch):
    class FakeClient:
        def search_offers(self, query: str):
            return []

    monkeypatch.setattr(price_mod, "_get_vast_client", lambda: FakeClient())
    assert VastPriceFeed().best_hourly("RTX_4090:1") is None
```

Append to `tests/test_scheduler_tick.py` (and add `from lab.models import ResourceRequest` to its imports):
```python
class FakePrices:
    def __init__(self, hourly: float | None) -> None:
        self.hourly = hourly
        self.calls = 0

    def best_hourly(self, accelerators, extra_query=None):
        self.calls += 1
        return self.hourly


def _gpu_reg(q, tmp_path, reg_id, max_hourly: float, **kw):
    r = put_reg(q, tmp_path, reg_id, triggers=Triggers(max_hourly_usd=max_hourly), **kw)
    q.put_entry(
        r.model_copy(
            update={"spec": r.spec.model_copy(update={
                "resources": ResourceRequest(accelerators="RTX_4090:1", timeout="1h")})}
        )
    )
    return r


def test_price_above_threshold_skips(tmp_path: Path):
    feed = FakePrices(0.40)
    sched, q = make_sched(tmp_path, price_feed=feed)
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25)
    rep = sched.tick()
    assert rep.launched == [] and "price" in rep.skipped["reg-a"]


def test_price_below_threshold_launches(tmp_path: Path):
    sched, q = make_sched(tmp_path, price_feed=FakePrices(0.20))
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25)
    assert sched.tick().launched == ["reg-a"]


def test_no_offers_skips(tmp_path: Path):
    sched, q = make_sched(tmp_path, price_feed=FakePrices(None))
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25)
    assert "no matching offer" in sched.tick().skipped["reg-a"]


def test_price_feed_deduped_per_tick(tmp_path: Path):
    feed = FakePrices(0.20)
    sched, q = make_sched(tmp_path, price_feed=feed)
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25)
    _gpu_reg(q, tmp_path, "reg-b", max_hourly=0.25)
    sched.tick()
    assert feed.calls == 1  # same accelerator spec -> one search_offers call (spec §4.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_price.py tests/test_scheduler_tick.py -q`
Expected: FAIL — no `lab.scheduler.price`; price tests in tick file fail (`Scheduler.__init__` has no `price_feed`).

- [ ] **Step 3: Implement**

`src/lab/scheduler/price.py`:
```python
"""Price trigger feed — 'is a matching Vast offer at/below $X/hr right now?' (spec §4.5).

Uses the vastai-sdk already in the stack (teardown fallback, FR-C2). ``dph_total`` is the real
billed rate (see ``vast_hourly_for_cluster``); the cheapest matching offer gates eligibility.
"""

from __future__ import annotations

from typing import Any, Protocol

_DEFAULTS = ["rentable=true", "rented=false", "reliability>0.95"]


def _get_vast_client() -> Any:
    """Function-local indirection: keeps this module importable without the skypilot extra
    (``lab.backends.skypilot`` imports ``sky`` at module top). Test seam — monkeypatch me."""
    from lab.backends.skypilot import _get_vast_client as real

    return real()


class PriceFeed(Protocol):
    def best_hourly(self, accelerators: str | None, extra_query: str | None = None) -> float | None:
        """Cheapest matching offer's $/hr, or None if no offer matches."""
        ...


def offer_query(accelerators: str | None, extra: str | None = None) -> str:
    """Derive a ``vastai search offers`` query from a SkyPilot accelerator spec."""
    parts = list(_DEFAULTS)
    if accelerators:
        name, _, count = accelerators.partition(":")
        parts.append(f"gpu_name={name}")
        parts.append(f"num_gpus>={count or 1}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


class VastPriceFeed:
    def best_hourly(self, accelerators: str | None, extra_query: str | None = None) -> float | None:
        offers: list[dict[str, Any]] = _get_vast_client().search_offers(
            query=offer_query(accelerators, extra_query)
        )
        prices = [
            float(o["dph_total"])
            for o in offers
            if isinstance(o, dict) and o.get("dph_total") is not None
        ]
        return min(prices) if prices else None
```

In `tick.py`: add `price_feed: PriceFeed | None = None` to `Scheduler.__init__` (stored as `self.price_feed`), import `from lab.scheduler.price import PriceFeed`. Add a per-tick cache and extend `_trigger_block`:

```python
    # in tick(): clear the per-tick price cache before evaluation
            self._price_cache: dict[tuple[str | None, str | None], float | None] = {}

    # in _trigger_block, after the dependency loop, before `return None`:
        if t.max_hourly_usd is not None:
            if self.price_feed is None:
                return "price trigger set but no price feed configured"
            key = (reg.spec.resources.accelerators, t.offer_query)
            if key not in self._price_cache:
                try:
                    self._price_cache[key] = self.price_feed.best_hourly(key[0], key[1])
                except Exception as e:  # noqa: BLE001 — API error skips, never crashes (spec §5)
                    rep.errors.append(f"price feed: {e}")
                    return f"price feed error: {e}"[:200]
            best = self._price_cache[key]
            if best is None:
                return "no matching offer available"
            if best > t.max_hourly_usd:
                return f"price ${best:.3f}/h above max ${t.max_hourly_usd:.3f}/h"
            self._best_hourly_seen[reg.reg_id] = best  # consumed by guardrails (Task 9)
```

Also initialize `self._best_hourly_seen: dict[str, float] = {}` alongside the price cache in `tick()`.

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_scheduler_price.py tests/test_scheduler_tick.py -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/price.py src/lab/scheduler/tick.py tests/test_scheduler_price.py tests/test_scheduler_tick.py
git commit -m "feat(scheduler): Vast price feed + price trigger (deduped per tick)"
```

---

### Task 9: Guardrails — per-job cost, daily budget, concurrency; post-launch price verify

**Files:**
- Modify: `src/lab/scheduler/tick.py`
- Test: `tests/test_scheduler_tick.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_per_job_cost_cap(tmp_path: Path):
    sched, q = make_sched(tmp_path, price_feed=FakePrices(0.50))
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=1.0)
    q.put_entry(q.get_entry("reg-a").model_copy(
        update={"guardrails": Guardrails(expires_at=T0 + timedelta(days=1), max_cost_usd=0.25)}))
    rep = sched.tick()  # 0.50/h x 1h = 0.50 > 0.25
    assert "max_cost" in rep.skipped["reg-a"]


def test_daily_budget_skips_not_cancels(tmp_path: Path):
    sched, q = make_sched(tmp_path, price_feed=FakePrices(2.0))
    q.write_control(ControlConfig(budget_usd_per_day=3.0))
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=3.0)
    _gpu_reg(q, tmp_path, "reg-b", max_hourly=3.0)
    rep = sched.tick()  # each estimated 2.0 -> only one fits the $3/day budget
    assert len(rep.launched) == 1
    other = ({"reg-a", "reg-b"} - set(rep.launched)).pop()
    assert "budget" in rep.skipped[other]
    assert q.get_entry(other).state is RegState.pending  # retried next tick


def test_max_concurrent(tmp_path: Path):
    sched, q = make_sched(tmp_path)
    q.write_control(ControlConfig(max_concurrent=1))
    slow = f"{PYTHON} -c 'import time; time.sleep(60)'"
    put_reg(q, tmp_path, "reg-a", command=slow)
    put_reg(q, tmp_path, "reg-b", command=slow)
    rep = sched.tick()
    assert len(rep.launched) == 1
    other = ({"reg-a", "reg-b"} - set(rep.launched)).pop()
    assert "concurren" in rep.skipped[other]


def test_post_launch_price_verify_reverts_to_pending(tmp_path: Path, monkeypatch):
    from lab.models import CostInfo

    sched, q = make_sched(tmp_path, price_feed=FakePrices(0.20))
    _gpu_reg(q, tmp_path, "reg-a", max_hourly=0.25,
             command=f"{PYTHON} -c 'import time; time.sleep(60)'")
    sched.tick()
    e = q.get_entry("reg-a")
    # Supervisor later records the real rental price: way above the threshold (offer raced away).
    sched.store.update_manifest(e.job_id, cost=CostInfo(hourly_usd=0.90))
    rep = sched.tick()
    again = q.get_entry("reg-a")
    assert again.state is RegState.pending and again.job_id is None
    assert "price" in (again.last_skip_reason or "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_tick.py -q`
Expected: the four new tests FAIL.

- [ ] **Step 3: Implement** in `tick.py`:

In `_evaluate_and_launch`, before the per-entry loop compute the running context, and gate launches:
```python
        running = [
            r for r in entries
            if r.state is RegState.launched and r.job_id is not None
            and self._job_status(r.job_id) not in self._TERMINAL_MAP
        ]
        committed = self._spend_last_24h(entries)
        n_running = len(running)
```
where:
```python
    def _job_status(self, job_id: str) -> JobState:
        try:
            return self.store.read_manifest(job_id).status
        except FileNotFoundError:
            return JobState.failed

    def _spend_last_24h(self, entries: list[Registration]) -> float:
        cutoff = self.now_fn() - timedelta(hours=24)
        total = 0.0
        for r in entries:
            if r.launched_at is None or r.launched_at < cutoff or r.job_id is None:
                continue
            try:
                cost = self.store.read_manifest(r.job_id).cost
            except FileNotFoundError:
                continue
            if cost and cost.estimated_usd:
                total += cost.estimated_usd
        return total
```
(import `timedelta` from `datetime`.)

After `_trigger_block` returns `None` for an entry, apply guardrails before `_launch`:
```python
            est = self._estimate_cost(reg)
            cap = reg.guardrails.max_cost_usd
            if est is not None and cap is not None and est > cap:
                reason = f"estimated ${est:.2f} exceeds max_cost ${cap:.2f}"
                self.queue.put_entry(reg.model_copy(update={"last_skip_reason": reason}))
                rep.skipped[reg.reg_id] = reason
                continue
            budget = control.budget_usd_per_day
            if budget is not None and est is not None and committed + est > budget:
                reason = f"daily budget: committed ${committed:.2f} + ${est:.2f} > ${budget:.2f}"
                self.queue.put_entry(reg.model_copy(update={"last_skip_reason": reason}))
                rep.skipped[reg.reg_id] = reason
                continue
            if n_running >= control.max_concurrent:
                reason = f"max_concurrent={control.max_concurrent} reached"
                self.queue.put_entry(reg.model_copy(update={"last_skip_reason": reason}))
                rep.skipped[reg.reg_id] = reason
                continue
            self._launch(reg, rep)
            if reg.reg_id in rep.launched:
                n_running += 1
                committed += est or 0.0
```
with:
```python
    def _estimate_cost(self, reg: Registration) -> float | None:
        """Worst-case launch cost: best offer $/h x wall-clock timeout (FR-I2 arithmetic)."""
        hourly = self._best_hourly_seen.get(reg.reg_id)
        secs = parse_duration(reg.spec.resources.timeout)
        if hourly is None or secs is None:
            return None
        return hourly * secs / 3600.0
```
(import `parse_duration` from `lab._util`.)

Post-launch price verify — add to `_sync`, right after reading a **non-terminal** `manifest` (before the cancel-marker block):
```python
            max_h = reg.triggers.max_hourly_usd
            if (
                manifest.status not in self._TERMINAL_MAP
                and max_h is not None
                and manifest.cost is not None
                and manifest.cost.hourly_usd is not None
                and manifest.cost.hourly_usd > max_h * 1.15
            ):
                lab = self.make_lab(self.home / "_bundles" / reg.reg_id)
                lab.cancel(reg.job_id)
                self.queue.mirror_manifest(self.store.read_manifest(reg.job_id))
                self._transition(
                    reg, RegState.pending, job_id=None, launched_at=None,
                    reason=(
                        f"relaunch: actual ${manifest.cost.hourly_usd:.3f}/h exceeded "
                        f"max ${max_h:.3f}/h (+15% slack)"
                    ),
                )
                rep.synced[reg.reg_id] = "price-verify rollback"
                continue
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_scheduler_tick.py -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/tick.py tests/test_scheduler_tick.py
git commit -m "feat(scheduler): cost/budget/concurrency guardrails + post-launch price verify"
```

---

### Task 10: `register()` — capture, upload (bundle first), entry

**Files:**
- Create: `src/lab/scheduler/register.py`
- Test: `tests/test_scheduler_register.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Registration capture: bundle-before-entry ordering, provenance, worst-case cost."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab.models import JobSpec, ResourceRequest
from lab.scheduler.models import Guardrails, RegState, Triggers
from lab.scheduler.queue import LocalQueueStore
from lab.scheduler.register import register, worst_case_cost
from test_scheduler_bundle import _make_repo  # reuse the tiny-repo factory


def test_register_creates_bundle_then_entry(tmp_path: Path):
    repo = _make_repo(tmp_path)
    (repo / "wip.py").write_text("print('wip')\n")  # dirty tree must be captured
    q = LocalQueueStore(tmp_path / "q")
    reg = register(
        repo, q,
        JobSpec(command="python exp.py"),
        Triggers(),
        Guardrails(expires_at=datetime.now(timezone.utc) + timedelta(days=1)),
    )
    assert reg.state is RegState.pending and reg.code.git_dirty
    assert q.get_entry(reg.reg_id) == reg
    got = q.fetch_bundle(reg.reg_id, tmp_path / "dl")
    assert got.stat().st_size > 0


def test_register_failed_bundle_leaves_no_entry(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path)
    q = LocalQueueStore(tmp_path / "q")
    import lab.scheduler.register as reg_mod

    monkeypatch.setattr(
        reg_mod, "create_bundle", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError):
        register(repo, q, JobSpec(command="x"), Triggers(),
                 Guardrails(expires_at=datetime.now(timezone.utc) + timedelta(days=1)))
    assert q.list_entries() == []  # entry is the commit point (spec §5)


def test_worst_case_cost():
    t = Triggers(max_hourly_usd=0.25)
    r = ResourceRequest(timeout="2h")
    assert worst_case_cost(t, r) == 0.5
    assert worst_case_cost(Triggers(), r) is None
    assert worst_case_cost(t, ResourceRequest()) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_register.py -q`
Expected: FAIL — no `lab.scheduler.register`.

- [ ] **Step 3: Implement**

`src/lab/scheduler/register.py`:
```python
"""``lab register`` core — capture code + spec + triggers into a queue entry (spec §6).

Write ordering is the integrity guarantee (spec §5): the bundle uploads first, the entry last —
the scheduler can never see an entry whose code is missing.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from lab._util import now, parse_duration
from lab.models import JobSpec, ResourceRequest
from lab.scheduler.bundle import create_bundle
from lab.scheduler.models import Guardrails, Registration, Triggers
from lab.scheduler.queue import QueueStore


def _new_reg_id() -> str:
    return f"reg-{now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def worst_case_cost(triggers: Triggers, resources: ResourceRequest) -> float | None:
    """What the user authorizes at registration time: max $/h x wall-clock timeout."""
    secs = parse_duration(resources.timeout)
    if triggers.max_hourly_usd is None or secs is None:
        return None
    return round(triggers.max_hourly_usd * secs / 3600.0, 6)


def register(
    repo: Path,
    queue: QueueStore,
    spec: JobSpec,
    triggers: Triggers,
    guardrails: Guardrails,
) -> Registration:
    reg_id = _new_reg_id()
    with tempfile.TemporaryDirectory() as td:
        tar, code = create_bundle(Path(repo), Path(td))
        bundle_key = queue.put_bundle(reg_id, tar)
    reg = Registration(
        reg_id=reg_id,
        created_at=now(),
        spec=spec,
        triggers=triggers,
        guardrails=guardrails,
        bundle_key=bundle_key,
        code=code,
    )
    queue.put_entry(reg)  # last write = commit point
    return reg
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_scheduler_register.py -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/register.py tests/test_scheduler_register.py
git commit -m "feat(scheduler): register() — bundle-first capture with worst-case cost"
```

---

### Task 11: R2 object ops + R2QueueStore

**Files:**
- Modify: `src/lab/storage.py` (generic object ops + injectable client)
- Create: `src/lab/scheduler/r2queue.py`
- Test: `tests/test_scheduler_r2queue.py`

- [ ] **Step 1: Write the failing tests**

```python
"""R2QueueStore satisfies the same contract as LocalQueueStore, over a dict-backed fake S3."""

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import make_manifest
from lab.models import CodeRef, JobSpec
from lab.scheduler.models import ControlConfig, Guardrails, Registration, RegState
from lab.scheduler.r2queue import R2QueueStore
from lab.storage import R2Store


class FakeS3:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:
        self.blobs[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self.blobs:
            raise self._missing()
        return {"Body": io.BytesIO(self.blobs[Key])}

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.blobs.pop(Key, None)

    def head_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self.blobs:
            raise self._missing()
        return {}

    def list_objects_v2(self, Bucket: str, Prefix: str, **kw) -> dict:
        keys = sorted(k for k in self.blobs if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        self.blobs[Key] = Path(Filename).read_bytes()

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        Path(Filename).write_bytes(self.blobs[Key])

    @staticmethod
    def _missing() -> Exception:
        import botocore.exceptions

        return botocore.exceptions.ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )


def make_q() -> tuple[R2QueueStore, FakeS3]:
    fake = FakeS3()
    store = R2Store("https://example.test", "bucket", client=fake)
    return R2QueueStore(store, prefix="queue"), fake


def _reg(reg_id: str) -> Registration:
    return Registration(
        reg_id=reg_id,
        created_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
        spec=JobSpec(command="python x.py"),
        guardrails=Guardrails(
            expires_at=datetime(2026, 6, 10, tzinfo=timezone.utc) + timedelta(days=1)
        ),
        bundle_key=f"queue/bundles/{reg_id}.tar.gz",
        code=CodeRef(git_commit="0" * 40, git_dirty=False),
    )


def test_entry_roundtrip_and_keys():
    q, fake = make_q()
    q.put_entry(_reg("reg-a"))
    assert "queue/entries/reg-a.json" in fake.blobs
    assert q.get_entry("reg-a").reg_id == "reg-a"
    assert [r.reg_id for r in q.list_entries()] == ["reg-a"]


def test_control_heartbeat_markers():
    q, _ = make_q()
    assert q.read_control() == ControlConfig()
    q.write_control(ControlConfig(paused=True))
    assert q.read_control().paused
    assert q.read_heartbeat() is None
    q.write_heartbeat({"tick_count": 1})
    assert q.read_heartbeat() == {"tick_count": 1}
    q.request_cancel("reg-a")
    assert q.cancel_requested("reg-a") and not q.held("reg-a")
    q.hold("reg-a")
    q.release("reg-a")
    assert not q.held("reg-a")


def test_bundle_and_manifest_mirror(tmp_path: Path):
    q, _ = make_q()
    src = tmp_path / "b.tar.gz"
    src.write_bytes(b"bytes")
    key = q.put_bundle("reg-a", src)
    assert key == "queue/bundles/reg-a.tar.gz"
    assert q.fetch_bundle("reg-a", tmp_path / "dl").read_bytes() == b"bytes"
    q.mirror_manifest(make_manifest("j1", "python x.py"))
    got = q.read_mirrored("j1")
    assert got is not None and got.job_id == "j1"
    assert q.read_mirrored("nope") is None
    assert [m.job_id for m in q.list_mirrored()] == ["j1"]


def test_state_update_overwrites():
    q, _ = make_q()
    q.put_entry(_reg("reg-a"))
    q.put_entry(q.get_entry("reg-a").model_copy(update={"state": RegState.launched}))
    assert q.get_entry("reg-a").state is RegState.launched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_r2queue.py -q`
Expected: FAIL — `R2Store.__init__() got an unexpected keyword argument 'client'`, then no `r2queue` module.

- [ ] **Step 3: Implement**

In `src/lab/storage.py`, change `R2Store.__init__` and add generic ops:
```python
    def __init__(self, endpoint: str, bucket: str = DEFAULT_BUCKET, client: Any | None = None) -> None:
        self.bucket = bucket
        if client is not None:
            self._s3 = client
            return
        import boto3

        if not os.environ.get("AWS_ACCESS_KEY_ID") and R2_CREDENTIALS_FILE.exists():
            os.environ.setdefault("AWS_SHARED_CREDENTIALS_FILE", str(R2_CREDENTIALS_FILE))
        self._s3 = boto3.client("s3", endpoint_url=endpoint, region_name="auto")
```
(add `from typing import Any` to imports), plus methods:
```python
    def put_text(self, key: str, text: str) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=text.encode())

    def get_text(self, key: str) -> str | None:
        """The object's text, or None if the key doesn't exist."""
        try:
            body: bytes = self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as e:  # noqa: BLE001 — boto raises dynamic ClientError subclasses
            if "NoSuchKey" in str(e) or getattr(e, "response", {}).get("Error", {}).get(
                "Code"
            ) in ("NoSuchKey", "404"):
                return None
            raise
        return body.decode()

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
        except Exception:  # noqa: BLE001
            return False
        return True

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=key)

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kwargs)
            keys += [o["Key"] for o in resp.get("Contents", [])]
            if not resp.get("IsTruncated"):
                return keys
            token = resp.get("NextContinuationToken")

    def upload_file(self, local: Path, key: str) -> None:
        self._s3.upload_file(str(local), self.bucket, key)

    def download_file(self, key: str, local: Path) -> None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self.bucket, key, str(local))
```

`src/lab/scheduler/r2queue.py`:
```python
"""R2-backed QueueStore — same contract as LocalQueueStore, keys under ``<prefix>/`` (spec §2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lab.models import JobManifest
from lab.scheduler.models import ControlConfig, Registration
from lab.storage import R2Store


class R2QueueStore:
    def __init__(self, store: R2Store, prefix: str = "queue") -> None:
        self.store = store
        self.prefix = prefix.rstrip("/")

    @classmethod
    def from_env(cls) -> R2QueueStore | None:
        store = R2Store.from_env()
        return cls(store) if store is not None else None

    def _k(self, *parts: str) -> str:
        return "/".join((self.prefix, *parts))

    # -- entries -------------------------------------------------------------
    def put_entry(self, reg: Registration) -> None:
        self.store.put_text(self._k("entries", f"{reg.reg_id}.json"), reg.model_dump_json())

    def get_entry(self, reg_id: str) -> Registration:
        text = self.store.get_text(self._k("entries", f"{reg_id}.json"))
        if text is None:
            raise FileNotFoundError(f"registration {reg_id} not found")
        return Registration.model_validate_json(text)

    def list_entries(self) -> list[Registration]:
        out: list[Registration] = []
        for key in sorted(self.store.list_keys(self._k("entries") + "/")):
            text = self.store.get_text(key)
            if text is not None:
                out.append(Registration.model_validate_json(text))
        return out

    # -- control / heartbeat ---------------------------------------------------
    def read_control(self) -> ControlConfig:
        text = self.store.get_text(self._k("control.json"))
        return ControlConfig.model_validate_json(text) if text else ControlConfig()

    def write_control(self, control: ControlConfig) -> None:
        self.store.put_text(self._k("control.json"), control.model_dump_json())

    def read_heartbeat(self) -> dict[str, Any] | None:
        import json

        text = self.store.get_text(self._k("heartbeat.json"))
        if text is None:
            return None
        loaded: dict[str, Any] = json.loads(text)
        return loaded

    def write_heartbeat(self, data: dict[str, Any]) -> None:
        import json

        self.store.put_text(self._k("heartbeat.json"), json.dumps(data, default=str))

    # -- markers ----------------------------------------------------------------
    def request_cancel(self, reg_id: str) -> None:
        self.store.put_text(self._k("cancelled", reg_id), "")

    def cancel_requested(self, reg_id: str) -> bool:
        return self.store.exists(self._k("cancelled", reg_id))

    def hold(self, reg_id: str) -> None:
        self.store.put_text(self._k("held", reg_id), "")

    def release(self, reg_id: str) -> None:
        self.store.delete(self._k("held", reg_id))

    def held(self, reg_id: str) -> bool:
        return self.store.exists(self._k("held", reg_id))

    # -- bundles ----------------------------------------------------------------
    def put_bundle(self, reg_id: str, src: Path) -> str:
        key = self._k("bundles", f"{reg_id}.tar.gz")
        self.store.upload_file(src, key)
        return key

    def fetch_bundle(self, reg_id: str, dest_dir: Path) -> Path:
        out = Path(dest_dir) / f"{reg_id}.tar.gz"
        self.store.download_file(self._k("bundles", f"{reg_id}.tar.gz"), out)
        return out

    # -- mirrored manifests -------------------------------------------------------
    def mirror_manifest(self, manifest: JobManifest) -> None:
        self.store.put_text(self._k("jobs", f"{manifest.job_id}.json"), manifest.model_dump_json())

    def read_mirrored(self, job_id: str) -> JobManifest | None:
        text = self.store.get_text(self._k("jobs", f"{job_id}.json"))
        return JobManifest.model_validate_json(text) if text else None

    def list_mirrored(self) -> list[JobManifest]:
        out: list[JobManifest] = []
        for key in sorted(self.store.list_keys(self._k("jobs") + "/")):
            text = self.store.get_text(key)
            if text is not None:
                out.append(JobManifest.model_validate_json(text))
        return out
```

- [ ] **Step 4: Run tests (incl. existing storage tests), verify pass**

Run: `uv run pytest tests/test_scheduler_r2queue.py tests/test_storage.py -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/storage.py src/lab/scheduler/r2queue.py tests/test_scheduler_r2queue.py
git commit -m "feat(scheduler): R2 queue store over generic R2Store object ops"
```

---

### Task 12: CLI — `lab register`, `lab queue …`, `lab scheduler tick`, status fallback

**Files:**
- Modify: `src/lab/scheduler/queue.py` (add `default_queue`), `src/lab/cli.py`
- Test: `tests/test_scheduler_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
"""CLI surface — CliRunner over a LocalQueueStore (LAB_QUEUE_DIR overrides queue selection)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from lab.cli import app
from lab.scheduler.models import RegState
from lab.scheduler.queue import LocalQueueStore
from test_scheduler_bundle import _make_repo

runner = CliRunner()


def _env(tmp_path: Path, repo: Path) -> dict[str, str]:
    return {"LAB_QUEUE_DIR": str(tmp_path / "queue"), "LAB_REPO_DIR": str(repo)}


def _register(tmp_path: Path, repo: Path, *extra: str) -> dict:
    res = runner.invoke(
        app,
        ["register", "--command", "python exp.py", "--timeout", "1h",
         "--expires", "+3d", *extra],
        env=_env(tmp_path, repo),
    )
    assert res.exit_code == 0, res.output
    return json.loads(res.output)


def test_register_and_list(tmp_path: Path):
    repo = _make_repo(tmp_path)
    out = _register(tmp_path, repo, "--max-hourly", "0.25")
    assert out["reg_id"].startswith("reg-")
    assert out["worst_case_cost_usd"] == 0.25
    q = LocalQueueStore(tmp_path / "queue")
    assert q.get_entry(out["reg_id"]).state is RegState.pending

    res = runner.invoke(app, ["queue", "list"], env=_env(tmp_path, repo))
    listed = json.loads(res.output)
    assert listed["entries"][0]["reg_id"] == out["reg_id"]
    assert listed["heartbeat_age_s"] is None  # no scheduler has ever ticked


def test_register_window_and_after(tmp_path: Path):
    repo = _make_repo(tmp_path)
    first = _register(tmp_path, repo)
    out = _register(
        tmp_path, repo, "--window", "23:00-07:00", "--tz", "UTC", "--after", first["reg_id"]
    )
    q = LocalQueueStore(tmp_path / "queue")
    reg = q.get_entry(out["reg_id"])
    assert reg.triggers.window is not None and reg.triggers.window.start.hour == 23
    assert reg.triggers.after == [first["reg_id"]]


def test_queue_cancel_hold_pause(tmp_path: Path):
    repo = _make_repo(tmp_path)
    out = _register(tmp_path, repo)
    env = _env(tmp_path, repo)
    q = LocalQueueStore(tmp_path / "queue")
    assert runner.invoke(app, ["queue", "hold", out["reg_id"]], env=env).exit_code == 0
    assert q.held(out["reg_id"])
    assert runner.invoke(app, ["queue", "release", out["reg_id"]], env=env).exit_code == 0
    assert runner.invoke(app, ["queue", "cancel", out["reg_id"]], env=env).exit_code == 0
    assert q.cancel_requested(out["reg_id"])
    assert runner.invoke(app, ["queue", "pause"], env=env).exit_code == 0
    assert q.read_control().paused
    assert runner.invoke(app, ["queue", "resume"], env=env).exit_code == 0
    assert not q.read_control().paused
    assert (
        runner.invoke(app, ["queue", "budget", "--per-day", "5"], env=env).exit_code == 0
    )
    assert q.read_control().budget_usd_per_day == 5.0


def test_scheduler_tick_runs_once(tmp_path: Path):
    repo = _make_repo(tmp_path)
    _register(tmp_path, repo)
    res = runner.invoke(app, ["scheduler", "tick"], env=_env(tmp_path, repo))
    assert res.exit_code == 0, res.output
    rep = json.loads(res.output)
    assert len(rep["launched"]) == 1


def test_queue_show_includes_skip_reason(tmp_path: Path):
    repo = _make_repo(tmp_path)
    out = _register(tmp_path, repo, "--not-before", "2030-01-01T00:00:00Z")
    env = _env(tmp_path, repo)
    runner.invoke(app, ["scheduler", "tick"], env=env)
    res = runner.invoke(app, ["queue", "show", out["reg_id"]], env=env)
    shown = json.loads(res.output)
    assert "not_before" in shown["last_skip_reason"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_cli.py -q`
Expected: FAIL — `register` is not a CLI command.

- [ ] **Step 3: Implement**

Append to `src/lab/scheduler/queue.py`:
```python
def default_queue() -> QueueStore:
    """Queue selection: ``LAB_QUEUE_DIR`` (tests/laptop-only) > R2 (if configured) > repo-local."""
    import os

    from lab.manifest import repo_root

    env_dir = os.environ.get("LAB_QUEUE_DIR")
    if env_dir:
        return LocalQueueStore(Path(env_dir))
    from lab.scheduler.r2queue import R2QueueStore

    r2 = R2QueueStore.from_env()
    if r2 is not None:
        return r2
    return LocalQueueStore(repo_root() / "queue")
```

In `src/lab/cli.py` add (after the existing commands; respect the existing `_emit` and import style):
```python
def _repo() -> Path:
    import os

    env = os.environ.get("LAB_REPO_DIR")
    return Path(env) if env else repo_root()


def _parse_expires(value: str) -> datetime:
    """``+3d``/``+12h`` (relative) or an ISO timestamp."""
    if value.startswith("+"):
        secs = parse_duration(value[1:])
        if secs is None:
            raise typer.BadParameter(f"bad relative expiry {value!r}")
        return now() + timedelta(seconds=secs)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_window(value: str, tz: str) -> DailyWindow:
    try:
        start_s, end_s = value.split("-", 1)
        return DailyWindow(
            start=dt_time.fromisoformat(start_s.strip()),
            end=dt_time.fromisoformat(end_s.strip()),
            tz=tz,
        )
    except ValueError as e:
        raise typer.BadParameter(f"--window expects HH:MM-HH:MM (got {value!r})") from e


@app.command()
def register(
    command: str = typer.Option(..., "--command", "-c", help="entrypoint, e.g. 'uv run experiments/x.py'"),
    expires: str = typer.Option(..., "--expires", help="run-by deadline: +3d / +12h / ISO timestamp (required guardrail)"),
    seed: int | None = typer.Option(None),
    cpus: int | None = typer.Option(None),
    memory: str | None = typer.Option(None),
    gpus: int | None = typer.Option(None),
    accelerators: str | None = typer.Option(None, "--gpu", "--accelerators", help="e.g. RTX_4090:1"),
    timeout: str | None = typer.Option(None, help="wall-clock limit per job, e.g. 2h (cost bound, FR-I1)"),
    window: str | None = typer.Option(None, "--window", help="daily launch window, e.g. 23:00-07:00"),
    tz: str = typer.Option("UTC", "--tz", help="IANA timezone for --window"),
    not_before: str | None = typer.Option(None, "--not-before", help="absolute earliest start (ISO)"),
    max_hourly: float | None = typer.Option(None, "--max-hourly", help="launch only if a matching Vast offer is at/below this $/h"),
    offer_query: str | None = typer.Option(None, "--offer-query", help="extra vastai search filter"),
    max_cost: float | None = typer.Option(None, "--max-cost", help="per-job worst-case $ cap"),
    after: list[str] = typer.Option(None, "--after", help="reg_id(s) that must succeed first (repeatable)"),
    hold: bool = typer.Option(False, "--hold", help="register held; release with `lab queue release`"),
) -> None:
    """Register a deferred job; the scheduler launches it when all triggers hold (spec §6)."""
    if accelerators and timeout is None:
        _emit({"error": "--timeout is required for GPU registrations (it is the cost bound)"})
        raise typer.Exit(code=1)
    queue = default_queue()
    triggers = Triggers(
        not_before=(
            datetime.fromisoformat(not_before.replace("Z", "+00:00")) if not_before else None
        ),
        window=_parse_window(window, tz) if window else None,
        max_hourly_usd=max_hourly,
        offer_query=offer_query,
        after=list(after or []),
    )
    guardrails = Guardrails(expires_at=_parse_expires(expires), max_cost_usd=max_cost)
    spec = JobSpec(
        command=command,
        seed=seed,
        resources=ResourceRequest(
            cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout
        ),
        submitted_by="human",
    )
    reg = sched_register(_repo(), queue, spec, triggers, guardrails)
    if hold:
        queue.hold(reg.reg_id)
    _emit(
        {
            "reg_id": reg.reg_id,
            "state": "held" if hold else reg.state.value,
            "bundle_key": reg.bundle_key,
            "expires_at": reg.guardrails.expires_at,
            "worst_case_cost_usd": worst_case_cost(triggers, spec.resources),
        }
    )


queue_app = typer.Typer(help="Manage deferred registrations (spec §6).", no_args_is_help=True)
app.add_typer(queue_app, name="queue")


def _heartbeat_age_s(queue: QueueStore) -> float | None:
    hb = queue.read_heartbeat()
    if not hb or "at" not in hb:
        return None
    at = datetime.fromisoformat(str(hb["at"]))
    return max(0.0, (now() - at).total_seconds())


@queue_app.command(name="list")
def queue_list() -> None:
    """Entries + state + skip reason, plus scheduler heartbeat age (spec §6)."""
    queue = default_queue()
    entries = queue.list_entries()
    _emit(
        {
            "heartbeat_age_s": _heartbeat_age_s(queue),
            "control": queue.read_control().model_dump(),
            "entries": [
                {
                    "reg_id": r.reg_id,
                    "state": "held" if (r.state is RegState.pending and queue.held(r.reg_id))
                    else r.state.value,
                    "cancel_requested": queue.cancel_requested(r.reg_id),
                    "job_id": r.job_id,
                    "last_skip_reason": r.last_skip_reason,
                    "expires_at": r.guardrails.expires_at,
                }
                for r in entries
            ],
        }
    )


@queue_app.command(name="show")
def queue_show(reg_id: str) -> None:
    """Full registration record."""
    _emit(json.loads(default_queue().get_entry(reg_id).model_dump_json()))


@queue_app.command(name="cancel")
def queue_cancel(reg_id: str) -> None:
    """Write the cancel marker; the scheduler applies it on its next tick (spec §5)."""
    queue = default_queue()
    queue.get_entry(reg_id)  # fail-loud on unknown id (FR-F3)
    queue.request_cancel(reg_id)
    _emit({"reg_id": reg_id, "cancel_requested": True})


@queue_app.command(name="hold")
def queue_hold(reg_id: str) -> None:
    """Hold a pending entry (skipped until released)."""
    queue = default_queue()
    queue.get_entry(reg_id)
    queue.hold(reg_id)
    _emit({"reg_id": reg_id, "held": True})


@queue_app.command(name="release")
def queue_release(reg_id: str) -> None:
    """Release a held entry."""
    default_queue().release(reg_id)
    _emit({"reg_id": reg_id, "held": False})


@queue_app.command(name="pause")
def queue_pause() -> None:
    """Globally stop the scheduler from launching (heartbeat keeps beating)."""
    queue = default_queue()
    queue.write_control(queue.read_control().model_copy(update={"paused": True}))
    _emit({"paused": True})


@queue_app.command(name="resume")
def queue_resume() -> None:
    queue = default_queue()
    queue.write_control(queue.read_control().model_copy(update={"paused": False}))
    _emit({"paused": False})


@queue_app.command(name="budget")
def queue_budget(
    per_day: float | None = typer.Option(None, "--per-day", help="trailing-24h estimated-spend cap, USD"),
    max_concurrent: int | None = typer.Option(None, "--max-concurrent"),
    auto_reconcile: bool | None = typer.Option(None, "--auto-reconcile/--no-auto-reconcile"),
) -> None:
    """Edit control.json guardrails."""
    queue = default_queue()
    control = queue.read_control()
    updates: dict[str, object] = {}
    if per_day is not None:
        updates["budget_usd_per_day"] = per_day
    if max_concurrent is not None:
        updates["max_concurrent"] = max_concurrent
    if auto_reconcile is not None:
        updates["auto_reconcile"] = auto_reconcile
    control = control.model_copy(update=updates)
    queue.write_control(control)
    _emit(control.model_dump())


scheduler_app = typer.Typer(help="Scheduler host commands (spec §4).", no_args_is_help=True)
app.add_typer(scheduler_app, name="scheduler")


@scheduler_app.command(name="tick")
def scheduler_tick(
    backend: str = typer.Option("local", "--backend", help="local | skypilot (droplet uses skypilot)"),
) -> None:
    """One idempotent scheduling pass — what the systemd timer runs every ~60s."""
    price_feed: PriceFeed | None = None
    if backend == "skypilot":
        from lab.scheduler.price import VastPriceFeed

        price_feed = VastPriceFeed()
    sched = Scheduler(
        default_queue(), home=_repo() / "runs", backend=backend, price_feed=price_feed
    )
    _emit(json.loads(sched.tick().model_dump_json()))
```

New imports at the top of `cli.py`:
```python
from datetime import datetime, timedelta
from datetime import time as dt_time

from lab._util import now, parse_duration, wrap_with_extras
from lab.scheduler.models import DailyWindow, Guardrails, RegState, Triggers
from lab.scheduler.price import PriceFeed
from lab.scheduler.queue import QueueStore, default_queue
from lab.scheduler.register import register as sched_register
from lab.scheduler.register import worst_case_cost
from lab.scheduler.tick import Scheduler
```

Status fallback — in the existing `status` command, wrap the body:
```python
@app.command()
def status(job_id: str) -> None:
    """Show a job's state + cost + teardown_status (FR-A2, FR-I2, FR-C2)."""
    try:
        lab = _lab_for(job_id)
    except FileNotFoundError:
        mirrored = default_queue().read_mirrored(job_id)  # scheduler-launched job (spec §4.3)
        if mirrored is None:
            _emit({"error": f"unknown job id {job_id!r}"})
            raise typer.Exit(code=2) from None
        _emit(
            {
                "job_id": job_id,
                "state": mirrored.status.value,
                "exit_code": mirrored.exit_code,
                "cost": mirrored.cost.model_dump() if mirrored.cost else None,
                "teardown_status": mirrored.teardown_status,
                "end_reason": mirrored.end_reason,
                "mirrored": True,  # may be up to one tick stale
            }
        )
        return
    state = lab.status(job_id)
    m = lab.manifest(job_id)
    _emit(
        {
            "job_id": job_id,
            "state": state.value,
            "exit_code": m.exit_code,
            "cost": m.cost.model_dump() if m.cost else None,
            "teardown_status": m.teardown_status,
            "end_reason": m.end_reason,
        }
    )
```

Note: `price.py` is importable without the skypilot extra (its `_get_vast_client` indirection from Task 8 defers the `lab.backends.skypilot` import), so the top-level `from lab.scheduler.price import PriceFeed` in `cli.py` is safe.

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_scheduler_cli.py tests/test_cli_wait.py -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/queue.py src/lab/cli.py tests/test_scheduler_cli.py
git commit -m "feat(cli): lab register / queue / scheduler tick + mirrored-status fallback"
```

---

### Task 13: MCP tools

**Files:**
- Modify: `src/lab/mcp_server.py`
- Test: `tests/test_mcp_server.py` (append; follow the existing in-file patterns for calling tools)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_mcp_server.py`, which already uses `_make(tmp_path)` + `fastmcp.Client` + `asyncio.run`)

**Also update the existing `test_tools_registered`** — it asserts the exact sorted tool list, so it must now expect:
```python
    assert asyncio.run(go()) == [
        "cancel",
        "fetch_artifacts",
        "list",
        "logs",
        "metrics",
        "queue_cancel",
        "queue_list",
        "queue_pause",
        "queue_show",
        "register",
        "status",
        "submit",
        "sweep",
    ]
```

Append (note: `register` bundles `lab.repo`, so build the server over a tiny temp git repo, not the real one):
```python
def _make_with_repo(tmp_path: Path):
    from test_scheduler_bundle import _make_repo

    repo = _make_repo(tmp_path)
    lab = Lab(backend=LocalBackend(home=tmp_path / "runs", repo=repo), repo=repo, home=tmp_path / "runs")
    return lab, build_server(lab)


def test_register_and_queue_tools(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    _, server = _make_with_repo(tmp_path)

    async def go():
        async with Client(server) as c:
            out = (await c.call_tool(
                "register",
                {"command": "python exp.py", "expires": "+1d",
                 "max_hourly": 0.25, "timeout": "1h"},
            )).data
            listed = (await c.call_tool("queue_list", {})).data
            shown = (await c.call_tool("queue_show", {"reg_id": out["reg_id"]})).data
            cancelled = (await c.call_tool("queue_cancel", {"reg_id": out["reg_id"]})).data
            paused = (await c.call_tool("queue_pause", {"paused": True})).data
            return out, listed, shown, cancelled, paused

    out, listed, shown, cancelled, paused = asyncio.run(go())
    assert out["reg_id"].startswith("reg-")
    assert out["worst_case_cost_usd"] == 0.25
    assert listed["entries"][0]["reg_id"] == out["reg_id"]
    assert shown["state"] == "pending"
    assert cancelled["cancel_requested"] is True
    assert paused["paused"] is True


def test_register_unknown_queue_ops_fail_loud(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    _, server = _make_with_repo(tmp_path)

    async def go():
        async with Client(server) as c:
            with pytest.raises(ToolError):
                await c.call_tool("queue_show", {"reg_id": "reg-nope"})

    asyncio.run(go())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -q`
Expected: new test FAILS — unknown tool `register`.

- [ ] **Step 3: Implement** — add inside `build_server` in `src/lab/mcp_server.py`:

```python
    from lab.scheduler.models import DailyWindow, Guardrails, RegState, Triggers
    from lab.scheduler.queue import default_queue
    from lab.scheduler.register import register as sched_register
    from lab.scheduler.register import worst_case_cost

    @mcp.tool
    def register(
        command: str,
        expires: str,
        seed: int | None = None,
        cpus: int | None = None,
        memory: str | None = None,
        accelerators: str | None = None,
        timeout: str | None = None,
        window: str | None = None,
        tz: str = "UTC",
        not_before: str | None = None,
        max_hourly: float | None = None,
        offer_query: str | None = None,
        max_cost: float | None = None,
        after: list[str] | None = None,
    ) -> dict:
        """Register a deferred job: launched by the scheduler when all triggers hold (time window HH:MM-HH:MM, Vast price <= max_hourly $/h, after=reg_ids succeeded). expires (+3d / ISO) is the required run-by guardrail; worst-case cost = max_hourly x timeout."""
        from datetime import datetime, timedelta
        from datetime import time as dt_time

        from lab._util import now, parse_duration

        if accelerators and timeout is None:
            raise ToolError("timeout is required for GPU registrations (it is the cost bound)")
        if expires.startswith("+"):
            secs = parse_duration(expires[1:])
            if secs is None:
                raise ToolError(f"bad relative expiry {expires!r}")
            expires_at = now() + timedelta(seconds=secs)
        else:
            expires_at = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        win = None
        if window:
            try:
                s, e = window.split("-", 1)
                win = DailyWindow(
                    start=dt_time.fromisoformat(s.strip()),
                    end=dt_time.fromisoformat(e.strip()),
                    tz=tz,
                )
            except ValueError as exc:
                raise ToolError(f"window expects HH:MM-HH:MM (got {window!r})") from exc
        triggers = Triggers(
            not_before=(
                datetime.fromisoformat(not_before.replace("Z", "+00:00")) if not_before else None
            ),
            window=win,
            max_hourly_usd=max_hourly,
            offer_query=offer_query,
            after=list(after or []),
        )
        spec = JobSpec(
            command=command,
            seed=seed,
            resources=ResourceRequest(
                cpus=cpus, memory=memory, accelerators=accelerators, timeout=timeout
            ),
            submitted_by="agent",
        )
        try:
            reg = sched_register(
                lab.repo, default_queue(), spec, triggers,
                Guardrails(expires_at=expires_at, max_cost_usd=max_cost),
            )
        except Exception as e:  # noqa: BLE001
            raise ToolError(str(e)) from e
        return {
            "reg_id": reg.reg_id,
            "state": reg.state.value,
            "expires_at": _iso(reg.guardrails.expires_at),
            "worst_case_cost_usd": worst_case_cost(triggers, spec.resources),
        }

    @mcp.tool
    def queue_list() -> dict:
        """List deferred registrations: state, job_id, last skip reason, scheduler heartbeat age."""
        queue = default_queue()
        hb = queue.read_heartbeat()
        age = None
        if hb and "at" in hb:
            from lab._util import now

            age = max(0.0, (now() - datetime.fromisoformat(str(hb["at"]))).total_seconds())
        return {
            "heartbeat_age_s": age,
            "control": queue.read_control().model_dump(),
            "entries": [
                {
                    "reg_id": r.reg_id,
                    "state": "held" if (r.state is RegState.pending and queue.held(r.reg_id))
                    else r.state.value,
                    "job_id": r.job_id,
                    "last_skip_reason": r.last_skip_reason,
                    "expires_at": _iso(r.guardrails.expires_at),
                }
                for r in queue.list_entries()
            ],
        }

    @mcp.tool
    def queue_show(reg_id: str) -> dict:
        """Full registration record (triggers, guardrails, provenance, state history fields)."""
        import json as _json

        try:
            reg = default_queue().get_entry(reg_id)
        except FileNotFoundError as e:
            raise ToolError(f"registration '{reg_id}' not found") from e
        loaded: dict = _json.loads(reg.model_dump_json())
        return loaded

    @mcp.tool
    def queue_cancel(reg_id: str) -> dict:
        """Request cancellation; the scheduler applies it on its next tick (also cancels a launched job)."""
        queue = default_queue()
        try:
            queue.get_entry(reg_id)
        except FileNotFoundError as e:
            raise ToolError(f"registration '{reg_id}' not found") from e
        queue.request_cancel(reg_id)
        return {"reg_id": reg_id, "cancel_requested": True}

    @mcp.tool
    def queue_pause(paused: bool = True) -> dict:
        """Pause/resume all scheduler launches (global switch; heartbeat keeps beating)."""
        queue = default_queue()
        queue.write_control(queue.read_control().model_copy(update={"paused": paused}))
        return {"paused": paused}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_mcp_server.py -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): register + queue_list/show/cancel/pause tools"
```

---

### Task 14: `sky_runner --adopt` (re-attach to a live cluster)

**Files:**
- Modify: `src/lab/sky_runner.py`
- Test: `tests/test_runner_adopt.py`

- [ ] **Step 1: Write the failing test**

```python
"""--adopt re-attaches to a live cluster: no sky.launch, waits, rsyncs, tears down."""

import sys
import types
from pathlib import Path

import lab.sky_runner as runner_mod
from helpers import make_manifest
from lab._util import now
from lab.models import CostInfo, JobState
from lab.store import JobStore


def test_adopt_skips_launch_and_finishes(tmp_path: Path, monkeypatch):
    home = tmp_path / "runs"
    store = JobStore(home)
    m = make_manifest("j1", "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running, "started_at": now(),
                "cost": CostInfo(hourly_usd=0.2, estimated_usd=0.2)}
    )
    store.create(m)
    store.write_runtime("j1", runner_pid=1, cluster="lab-j1")

    fake_sky = types.ModuleType("sky")
    monkeypatch.setitem(sys.modules, "sky", fake_sky)

    launched = []
    monkeypatch.setattr(
        fake_sky, "launch", lambda *a, **k: launched.append(1), raising=False
    )
    monkeypatch.setattr(runner_mod, "_wait_terminal",
                        lambda *a, **k: JobState.succeeded)
    monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: 0.2)

    rc = runner_mod.run_job(home / "j1", adopt=True)
    assert rc == 0
    assert launched == []  # adopt never re-launches
    final = store.read_manifest("j1")
    assert final.status is JobState.succeeded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner_adopt.py -q`
Expected: FAIL — `run_job() got an unexpected keyword argument 'adopt'`

- [ ] **Step 3: Implement** in `src/lab/sky_runner.py`:

Change the signature and guard the launch phase:
```python
def run_job(job_dir: Path, adopt: bool = False) -> int:
```
After `manifest = store.read_manifest(job_id)` / `cluster = cluster_name_for(job_id)`, replace the unconditional start block with:
```python
    if not adopt:
        started = now()
        store.update_manifest(job_id, status=JobState.running, started_at=started)
    else:
        started = manifest.started_at or now()
        print(f"[lab] adopting running cluster {cluster} (supervisor restart)")
```
Wrap the provisioning block (everything from `task = build_task(...)` through the `hourly_usd`/`estimated_usd` recording) in `if not adopt:`; in the adopt branch instead:
```python
    else:
        hourly_usd = _resolve_hourly(cluster, None)
        estimated_usd = manifest.cost.estimated_usd if manifest.cost else None
        sky_job_id = None  # match any job in the cluster queue
```
For `max_wait` in adopt mode, charge only the *remaining* budget:
```python
        total = (parse_duration(manifest.resources.timeout) or 3600) + 300
        elapsed = duration_seconds(started, now()) or 0.0
        max_wait = max(60.0, total - elapsed)
```
(in the non-adopt path keep the existing `max_wait` line). The `_wait_terminal` call, rsync, R2 upload, cost recording, and teardown tail are shared and unchanged. Finally, the entrypoint:
```python
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--adopt", action="store_true",
                    help="re-attach to an already-launched cluster (scheduler watchdog)")
    ns = ap.parse_args()
    raise SystemExit(run_job(ns.job_dir, adopt=ns.adopt))
```

- [ ] **Step 4: Run tests (incl. existing runner tests), verify pass**

Run: `uv run pytest tests/test_runner_adopt.py tests/test_runner.py tests/test_skypilot.py -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/sky_runner.py tests/test_runner_adopt.py
git commit -m "feat(sky_runner): --adopt mode re-attaches to a live cluster after supervisor death"
```

---

### Task 15: Watchdog + periodic reconcile sweep in the tick

**Files:**
- Modify: `src/lab/scheduler/tick.py`
- Test: `tests/test_scheduler_tick.py` (append)

- [ ] **Step 1: Write the failing tests** (append; first extend the test file's imports with `from helpers import make_manifest`, `from lab._util import now as utc_now`, and `from lab.models import BackendInfo, JobState`)

The watchdog compares manifest timestamps against the scheduler clock, so build these scheds on
the real clock: `Scheduler(q, home=tmp_path / "runs", now_fn=utc_now, ...)` directly (not
`make_sched`, whose FakeClock is frozen at T0).

```python
def _watchdog_sched(tmp_path: Path, **kw) -> tuple[Scheduler, LocalQueueStore]:
    q = LocalQueueStore(tmp_path / "queue")
    return Scheduler(q, home=tmp_path / "runs", now_fn=utc_now, **kw), q


def _skypilot_launched_reg(tmp_path: Path, q, sched, *, started_ago_s: float, timeout: str = "1h"):
    """A launched skypilot-backed reg whose supervisor pid is dead (impossible pid)."""
    put_reg(q, tmp_path, "reg-a", command="python x.py")
    m = make_manifest("j-sky", "python x.py", timeout=timeout).model_copy(
        update={
            "status": JobState.running,
            "started_at": utc_now() - timedelta(seconds=started_ago_s),
            "backend": BackendInfo(provisioner="skypilot"),
            "registration_id": "reg-a",
        }
    )
    sched.store.create(m)
    sched.store.write_runtime("j-sky", runner_pid=99999999, cluster="lab-j-sky")
    q.put_entry(
        q.get_entry("reg-a").model_copy(
            update={
                "state": RegState.launched,
                "job_id": "j-sky",
                "launched_at": utc_now() - timedelta(seconds=started_ago_s),
            }
        )
    )


def test_watchdog_respawns_when_cluster_alive_within_timeout(tmp_path: Path):
    sched, q = _watchdog_sched(tmp_path)
    _skypilot_launched_reg(tmp_path, q, sched, started_ago_s=60)
    events: list[str] = []
    sched._cluster_alive = lambda cluster: True  # type: ignore[method-assign]
    sched._respawn_supervisor = lambda job_id: events.append(f"respawn:{job_id}")  # type: ignore[method-assign]
    sched.tick()
    assert events == ["respawn:j-sky"]


def test_watchdog_times_out_overdue_job(tmp_path: Path):
    sched, q = _watchdog_sched(tmp_path)
    _skypilot_launched_reg(tmp_path, q, sched, started_ago_s=2 * 3600, timeout="1h")
    events: list[str] = []
    sched._cluster_alive = lambda cluster: True  # type: ignore[method-assign]
    sched._teardown = lambda cluster, job_id: events.append(f"down:{cluster}") or True  # type: ignore[method-assign]
    sched.tick()
    assert events == ["down:lab-j-sky"]
    assert sched.store.read_manifest("j-sky").status.value == "timed_out"
    assert q.get_entry("reg-a").state is RegState.failed


def test_watchdog_marks_failed_when_cluster_gone(tmp_path: Path):
    sched, q = _watchdog_sched(tmp_path)
    _skypilot_launched_reg(tmp_path, q, sched, started_ago_s=60)
    sched._cluster_alive = lambda cluster: False  # type: ignore[method-assign]
    sched.tick()
    m = sched.store.read_manifest("j-sky")
    assert m.status.value == "failed" and "supervisor died" in (m.end_reason or "")


def test_reconcile_sweep_every_n_ticks(tmp_path: Path):
    sched, q = _watchdog_sched(tmp_path, reconcile_every=2)
    calls: list[bool] = []
    sched._reconcile = lambda apply: calls.append(apply) or {"orphans": []}  # type: ignore[method-assign]
    sched.tick()
    sched.tick()  # tick_count 2 -> sweep
    sched.tick()
    sched.tick()  # tick_count 4 -> sweep
    assert calls == [False, False]
    q.write_control(ControlConfig(auto_reconcile=True))
    sched.tick()
    sched.tick()  # tick_count 6 -> sweep with apply=True
    assert calls[-1] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_tick.py -q`
Expected: new tests FAIL (no watchdog/no `reconcile_every`).

- [ ] **Step 3: Implement** in `tick.py`:

Add to `__init__`: `reconcile_every: int = 30` parameter (store as `self.reconcile_every`).

Watchdog — insert at the **top** of the `_sync` per-entry loop, right after reading `manifest` (and **before** any `backend.status()`-style logic — reading the manifest directly is deliberate: `SkyPilotBackend.status()` would mark a dead-supervisor job failed before we can adopt it):
```python
            if (
                manifest.status not in self._TERMINAL_MAP
                and manifest.backend.provisioner == "skypilot"
            ):
                rt = self.store.read_runtime(reg.job_id)
                pid = rt.get("runner_pid")
                if pid and not _pid_alive(int(pid)):
                    cluster = str(rt.get("cluster") or f"lab-{reg.job_id}")
                    deadline_s = parse_duration(manifest.resources.timeout) or 3600.0
                    started = manifest.started_at or manifest.created_at
                    overdue = (self.now_fn() - started).total_seconds() > deadline_s + 300
                    if not self._cluster_alive(cluster):
                        manifest = self.store.update_manifest(
                            reg.job_id, status=JobState.failed, ended_at=self.now_fn(),
                            end_reason="supervisor died; instance gone",
                        )
                    elif overdue:
                        ok = self._teardown(cluster, reg.job_id)
                        manifest = self.store.update_manifest(
                            reg.job_id, status=JobState.timed_out, ended_at=self.now_fn(),
                            end_reason="watchdog: past timeout with dead supervisor",
                            teardown_status="succeeded" if ok else "failed",
                        )
                    else:
                        self._respawn_supervisor(reg.job_id)
                        rep.synced[reg.reg_id] = "supervisor respawned (adopt)"
```
with the seams + helper:
```python
    def _cluster_alive(self, cluster: str) -> bool:
        """Does a Vast rental back this cluster? Test seam."""
        from lab.backends.skypilot import vast_hourly_for_cluster

        try:
            return vast_hourly_for_cluster(cluster) is not None
        except Exception:  # noqa: BLE001
            return False

    def _respawn_supervisor(self, job_id: str) -> None:
        """Re-attach a supervisor to a live cluster (sky_runner --adopt). Test seam."""
        import subprocess
        import sys

        job_dir = self.store.job_dir(job_id)
        logf = self.store.logs_path(job_id).open("a")
        proc = subprocess.Popen(
            [sys.executable, "-m", "lab.sky_runner", str(job_dir), "--adopt"],
            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
        )
        self.store.write_runtime(job_id, runner_pid=proc.pid)

    def _teardown(self, cluster: str, job_id: str) -> bool:
        """Robust teardown of an overdue orphan. Test seam."""
        import sky

        from lab.backends.skypilot import tear_down_and_record

        return tear_down_and_record(sky, cluster, self.store, job_id)

    def _reconcile(self, apply: bool) -> dict[str, object]:
        """FR-C2 sweep. Test seam."""
        lab = self.make_lab(self.home)
        return lab.reconcile(apply=apply)
```
and module-level:
```python
def _pid_alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
```
In `tick()`, after the launch phase (still inside the not-paused branch):
```python
            if tick_count % self.reconcile_every == 0:
                try:
                    rep.reconcile = self._reconcile(control.auto_reconcile)
                except Exception as e:  # noqa: BLE001
                    rep.errors.append(f"reconcile sweep: {e}")
```

- [ ] **Step 4: Run the full suite, verify pass**

Run: `uv run pytest tests -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/tick.py tests/test_scheduler_tick.py
git commit -m "feat(scheduler): supervisor watchdog (adopt/teardown/fail) + periodic reconcile sweep"
```

---

### Task 16: Deployment assets + docs + final verification

**Files:**
- Create: `deploy/scheduler/lab-scheduler.service`, `deploy/scheduler/lab-scheduler.timer`, `deploy/scheduler/scheduler.env.example`, `deploy/scheduler/README.md`
- Modify: `CLAUDE.md` (one line), `LAB-REQUIREMENTS.md` is **not** touched (spec doc already records the design)

- [ ] **Step 1: Write the assets**

`deploy/scheduler/lab-scheduler.service`:
```ini
[Unit]
Description=Laboratory deferred-scheduler tick (one idempotent pass)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=lab
WorkingDirectory=/opt/laboratory
EnvironmentFile=/etc/lab/scheduler.env
ExecStart=/usr/local/bin/uv run lab scheduler tick --backend skypilot
```

`deploy/scheduler/lab-scheduler.timer`:
```ini
[Unit]
Description=Run the lab scheduler tick every 60s (spec §4)

[Timer]
OnBootSec=60
OnUnitActiveSec=60
Persistent=true

[Install]
WantedBy=timers.target
```

`deploy/scheduler/scheduler.env.example` (placeholders are intentional here — it's a template the operator fills; never commit real values, FR-J1):
```bash
# Copy to /etc/lab/scheduler.env (mode 0600, owner lab). NEVER commit real values (FR-J1).
LAB_R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
LAB_R2_BUCKET=lab-artifacts
AWS_ACCESS_KEY_ID=<r2-access-key>
AWS_SECRET_ACCESS_KEY=<r2-secret-key>
VAST_API_KEY=<vast-api-key>
```

`deploy/scheduler/README.md`:
```markdown
# Scheduler host deployment (spec §7)

The scheduler is **stateless**: queue entries, control, bundles, and mirrored job manifests all
live in R2, so this host can be destroyed and recreated at any time.

## Provision (playground repo)

Use the playground project's `cloud-digitalocean` backend to create the smallest droplet
(the tick is tiny and I/O-bound), then run the steps below (manually or via an Ansible role in
that repo):

1. Install uv + git; `git clone <laboratory remote> /opt/laboratory && cd /opt/laboratory && uv sync --extra skypilot`.
2. Create user `lab`; `cp deploy/scheduler/scheduler.env.example /etc/lab/scheduler.env` and fill
   in real credentials (mode 0600, owner `lab`).
3. `cp deploy/scheduler/lab-scheduler.{service,timer} /etc/systemd/system/`
4. `systemctl daemon-reload && systemctl enable --now lab-scheduler.timer`

## Verify

- `systemctl list-timers lab-scheduler.timer` — next tick scheduled.
- From the laptop: `lab queue list` — `heartbeat_age_s` under ~120.

## Live smoke (run once, at night, before trusting it)

1. Laptop: `lab register -c "uv run experiments/<cheap-exp>.py" --gpu RTX_4090:1 --timeout 15m \
   --max-hourly 0.30 --max-cost 0.10 --window 23:00-07:00 --tz <your-tz> --expires +1d`
2. Next morning, laptop: `lab queue list` (state `succeeded`), `lab status <job_id>` (mirrored
   manifest, cost recorded), `lab fetch <job_id>` (artifacts from R2), and `lab reconcile`
   (no orphans).
3. Kill test: while a registered job runs, `playground` reboot the droplet — within ~2 ticks
   `lab queue list` should show `supervisor respawned (adopt)` behavior and the job must still
   tear down on completion.

## Suspend when idle

`playground suspend <lab>` destroys the droplet (it bills while up). Registrations queued while
it is down launch when it returns (`Persistent=true` catches up missed ticks).
```

Add to `CLAUDE.md` under "Key facts":
```markdown
- **Deferred scheduling:** `lab register` + `lab queue …` queue jobs (night window / price /
  dependency triggers); an always-on host runs `lab scheduler tick` every 60s (systemd timer,
  `deploy/scheduler/`). Spec: `docs/superpowers/specs/2026-06-10-deferred-scheduling-design.md`.
```

- [ ] **Step 2: Full-project verification**

Run: `uv run pytest tests -q && uv run mypy src/lab && uv run ruff check src/lab tests`
Expected: entire suite PASS, mypy/ruff clean.

- [ ] **Step 3: Spec coverage self-check**

Walk `docs/superpowers/specs/2026-06-10-deferred-scheduling-design.md` §3–§9 and confirm each maps to a merged task (models→1, bundle→2, queue→3/11, tick steps 1–8→5–9, register/CLI/MCP→10/12/13, watchdog/adopt/reconcile→14/15, deployment+smoke→16). Fix anything missed before declaring done.

- [ ] **Step 4: Commit**

```bash
git add deploy/scheduler CLAUDE.md
git commit -m "feat(scheduler): deployment assets (systemd timer, env template, ops runbook)"
```

---

## Out of scope (per spec non-goals)

Recurring registrations, priorities, dependencies on laptop-launched job_ids, auto-retry on preemption, multi-user quotas. Playground-repo Ansible automation is follow-on work in that repo; `deploy/scheduler/README.md` is its contract.
