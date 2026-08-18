# Event Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the lab an append-only ledger of its own behaviour — every CLI/MCP call, its outcome, and (on failure) the internal trace that explains it — readable through `lab history` and `lab report`.

**Architecture:** A new `lab.events` package writes two JSONL lines per call (`open` at entry, `close` at exit) to `~/.lab/events/YYYY-MM-DD.jsonl`. Two capture points feed it: a `main()` wrapper in `cli.py` and a FastMCP `on_call_tool` middleware. Internals call `events.note(...)`, buffered in memory and flushed into the record only when the call fails. Readers are pure functions over the JSONL.

**Tech Stack:** Python 3.12, dataclasses, `fcntl.flock`, typer/click 8.1, FastMCP 3.3.1, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-event-logging-design.md`

## Global Constraints

- `ruff` line length **100**; `mypy --strict` passes on `src/lab`.
- `lab.events` takes **no dependency outside base deps** — it must work from an installed wheel (`pytest -m packaging`).
- Every write path is **best-effort**: an exception inside logging must never propagate into a command. `LAB_EVENTS_DEBUG=1` prints swallowed errors to stderr.
- **stdout carries JSON only** on CLI commands; diagnostics go to stderr.
- Secrets never reach disk (FR-J1 / AC-7): `params` passes through the sanitizer, never raw.
- Tests that involve time use an **injected clock**, never the real one — real-clock tests anchored to a fixed T0 decay into failures (the `_watchdog_sched` precedent).
- Env knobs, exact names: `LAB_EVENTS` (`0` disables), `LAB_EVENTS_DIR`, `LAB_EVENTS_DEBUG`, `LAB_EVENTS_SUCCESS_TTL_DAYS` (default 14), `LAB_EVENTS_MAX_AGE_DAYS` (default 90), `LAB_EVENTS_MAX_MB` (default 50), `LAB_SESSION_ID`.

## File Structure

**Create:**

| File | Responsibility |
|---|---|
| `src/lab/events/__init__.py` | Public surface only: `record`, `begin`, `finish_current`, `current`, `note`, `read`, `stats`, `report`, `Call`, `Event` |
| `src/lab/events/models.py` | `Note`, `Event` (a *folded* row), `ActionStat`, `SignatureStat`, `StatsView` |
| `src/lab/events/sanitize.py` | `sanitize_params`, `sanitize_argv` — the only code that decides what is safe to write |
| `src/lab/events/store.py` | Paths, enable flag, locked append, tolerant line iteration, retention |
| `src/lab/events/record.py` | `begin`/`finish`/`record`/`note`, the ring buffer, session + id generation |
| `src/lab/events/annotate.py` | `refs_from`, `digest_of` — pull ids and a small result digest out of a payload |
| `src/lab/events/read.py` | `read()` — fold open/close pairs, filter, order |
| `src/lab/events/stats.py` | `signature()`, `stats()` |
| `src/lab/events/report.py` | `report()` — markdown digest |
| `docs/guides/event-logging.md` | User-facing guide |

**Modify:** `src/lab/cli.py` (capture + `history`/`report` commands), `src/lab/mcp_server.py` (middleware + tools), `pyproject.toml` (entry point), `src/lab/placement.py`, `src/lab/doctor.py`, `src/lab/core.py`, `src/lab/storage.py`, `src/lab/backends/skypilot.py`, `src/lab/scheduler/tick.py` (`note()` sites), `tests/test_packaging.py`, `CLAUDE.md`, `CHANGELOG.md`, `docs/COMPATIBILITY.md`.

---

### Task 1: Event models and the parameter sanitizer

The sanitizer is first because nothing may be written until it exists — it is the FR-J1 gate.

**Files:**
- Create: `src/lab/events/__init__.py`, `src/lab/events/models.py`, `src/lab/events/sanitize.py`
- Test: `tests/test_events_sanitize.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Note(t: int, k: str, d: dict[str, Any])` — frozen dataclass
  - `Event` — folded row, fields: `id, ts, session, seq, surface, action, params, project, lab_version, outcome, exit_code, duration_ms, refs, result, error, trace`; property `status -> str`
  - `sanitize_params(params: Mapping[str, Any]) -> dict[str, Any]`
  - `sanitize_argv(argv: Sequence[str]) -> list[str]`
  - `MASK: str = "…REDACTED…"`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events_sanitize.py`:

```python
"""The sanitizer is the FR-J1 gate: nothing reaches the ledger without passing through it."""

from __future__ import annotations

import pytest

from lab.events.sanitize import MASK, sanitize_argv, sanitize_params

SECRET = "abcd1234efgh5678ijkl9012mnop3456qrst"


@pytest.mark.parametrize(
    "key",
    ["api_key", "vast_api_key", "token", "access_token", "client_secret", "password",
     "credential", "AUTH_HEADER"],
)
def test_secret_shaped_keys_are_masked(key: str) -> None:
    assert sanitize_params({key: SECRET}) == {key: MASK}


def test_ordinary_params_survive_verbatim() -> None:
    params = {"command": "python experiments/x.py", "backend": "cpu", "cloud": "gcp", "seeds": "0-31"}
    assert sanitize_params(params) == params


def test_secret_shaped_values_are_masked_under_innocent_keys() -> None:
    out = sanitize_params({"note": "ya29.a0AfH6SM", "pem": "-----BEGIN RSA PRIVATE KEY-----x"})
    assert out == {"note": MASK, "pem": MASK}


def test_high_entropy_value_is_masked_but_ids_and_shas_are_not() -> None:
    assert sanitize_params({"x": SECRET}) == {"x": MASK}
    # hex ids (commits, cell ids, job ids) are not credential-shaped and stay readable
    assert sanitize_params({"commit": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"})["commit"].startswith("a1b2")
    assert sanitize_params({"job_id": "j-4f2a8c1e"}) == {"job_id": "j-4f2a8c1e"}


def test_long_strings_and_lists_are_truncated() -> None:
    out = sanitize_params({"blob": "a b " * 400, "many": list(range(100))})
    assert len(out["blob"]) <= 512 + 1  # + the ellipsis marker
    assert out["blob"].endswith("…")
    assert len(out["many"]) == 33  # 32 items + the "…N more" marker
    assert out["many"][-1] == "…68 more"


def test_nested_structures_are_sanitized() -> None:
    out = sanitize_params({"resources": {"cpus": 4, "api_key": SECRET}})
    assert out == {"resources": {"cpus": 4, "api_key": MASK}}


def test_argv_masks_the_value_after_a_secret_flag() -> None:
    argv = ["lab", "submit", "--vast-api-key", SECRET, "--backend", "cpu"]
    assert sanitize_argv(argv) == ["lab", "submit", "--vast-api-key", MASK, "--backend", "cpu"]


def test_argv_masks_inline_secret_flags() -> None:
    assert sanitize_argv([f"--token={SECRET}"]) == [f"--token={MASK}"]


def test_unserializable_values_degrade_to_a_type_name() -> None:
    assert sanitize_params({"obj": object()})["obj"].startswith("<object")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_sanitize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.events'`

- [ ] **Step 3: Write `src/lab/events/models.py`**

```python
"""Event records. ``Event`` is a *folded* row: one open line plus its close line."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Note:
    """One ring-buffer entry: ``t`` is milliseconds since the call opened."""

    t: int
    k: str
    d: dict[str, Any]


@dataclass
class Event:
    """A folded call. ``outcome is None`` means the close line never arrived."""

    id: str
    ts: datetime
    session: str
    seq: int
    surface: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    project: dict[str, Any] = field(default_factory=dict)
    lab_version: str = ""
    outcome: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None
    refs: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    trace: list[Note] = field(default_factory=list)

    @property
    def status(self) -> str:
        """``outcome``, or the standing finding that a call never closed."""
        return self.outcome or "running-or-died"

    @property
    def failed(self) -> bool:
        """A call that never closed counts as failed — the missing close *is* the finding."""
        return self.outcome != "ok"


@dataclass
class ActionStat:
    action: str
    calls: int
    failures: int
    failure_rate: float
    median_ms: int


@dataclass
class SignatureStat:
    signature: str
    count: int
    first_seen: datetime
    last_seen: datetime
    actions: list[str]
    usd: float


@dataclass
class StatsView:
    since: datetime | None
    total: int
    failures: int
    dangling: int
    usd_burned: float
    actions: list[ActionStat]
    signatures: list[SignatureStat]
```

- [ ] **Step 4: Write `src/lab/events/sanitize.py`**

```python
"""What may be written. Recording argv means recording whatever was typed, and
:func:`lab.redact.redact` only knows patterns that appear in *subprocess output* — it will not
catch a key passed as a flag value. Everything entering the ledger passes through here first
(FR-J1, AC-7)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from lab.redact import redact

MASK = "…REDACTED…"
MAX_STR = 512
MAX_ITEMS = 32

_SECRET_KEY = re.compile(r"key|token|secret|password|credential|auth", re.IGNORECASE)
_SECRET_VALUE = (
    re.compile(r"^ya29\."),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$"),  # bare base64 blobs
)
_HEXISH = re.compile(r"^[0-9a-f-]+$", re.IGNORECASE)  # commits, cell ids, job ids — not secrets


def _entropy(s: str) -> float:
    counts = {c: s.count(c) for c in set(s)}
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in counts.values())


def _looks_secret(value: str) -> bool:
    if any(p.search(value) for p in _SECRET_VALUE):
        return True
    if " " in value or len(value) < 32 or _HEXISH.match(value):
        return False
    return _entropy(value) > 3.5


def _scalar(value: Any) -> Any:
    if isinstance(value, str):
        if _looks_secret(value):
            return MASK
        value = redact(value)
        return value[:MAX_STR] + "…" if len(value) > MAX_STR else value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return f"<{type(value).__name__}>"


def _walk(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _SECRET_KEY.search(key):
        return MASK
    if isinstance(value, Mapping):
        return {str(k): _walk(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_walk(v) for v in list(value)[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"…{len(value) - MAX_ITEMS} more")
        return items
    return _scalar(value)


def sanitize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize a parameter mapping for the ledger. Never raises — a sanitizer that
    failed would take the whole record with it."""
    try:
        return {str(k): _walk(v, key=str(k)) for k, v in params.items()}
    except Exception:  # noqa: BLE001 — logging must never fail a command
        return {"_unsanitizable": True}


def sanitize_argv(argv: Sequence[str]) -> list[str]:
    """Sanitize a raw command line: mask ``--flag=<secret>`` and the token after ``--flag``."""
    out: list[str] = []
    mask_next = False
    for token in argv:
        if mask_next:
            out.append(MASK)
            mask_next = False
            continue
        if token.startswith("-") and "=" in token:
            flag, _, value = token.partition("=")
            out.append(f"{flag}={MASK}" if _SECRET_KEY.search(flag) else f"{flag}={_scalar(value)}")
            continue
        if token.startswith("-") and _SECRET_KEY.search(token):
            mask_next = True
        out.append(_scalar(token))
    return out
```

- [ ] **Step 5: Write `src/lab/events/__init__.py`**

```python
"""The lab's ledger of its own behaviour. See
``docs/superpowers/specs/2026-08-18-event-logging-design.md``."""

from __future__ import annotations

from lab.events.models import ActionStat, Event, Note, SignatureStat, StatsView

__all__ = ["ActionStat", "Event", "Note", "SignatureStat", "StatsView"]
```

- [ ] **Step 6: Run the tests and the type checks**

Run: `uv run pytest tests/test_events_sanitize.py -v && uv run mypy --strict src/lab/events && uv run ruff check src/lab/events`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/lab/events tests/test_events_sanitize.py
git commit -m "feat(events): event models and the parameter sanitizer"
```

---

### Task 2: The store — paths, locked append, tolerant reads

**Files:**
- Create: `src/lab/events/store.py`
- Test: `tests/test_events_store.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `enabled() -> bool`
  - `events_dir() -> Path`
  - `day_file(when: datetime) -> Path`
  - `append(record: dict[str, Any], *, when: datetime) -> None`
  - `day_files() -> list[Path]` (sorted oldest→newest)
  - `iter_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]` (skips malformed lines)
  - `debug(message: str) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events_store.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.events import store


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


T0 = datetime(2026, 8, 18, 14, 3, 11, tzinfo=timezone.utc)


def test_append_writes_one_json_line_to_the_utc_day_file(_events_dir: Path) -> None:
    store.append({"id": "a", "phase": "open"}, when=T0)
    path = _events_dir / "2026-08-18.jsonl"
    assert path.read_text().count("\n") == 1
    assert list(store.iter_records([path])) == [{"id": "a", "phase": "open"}]


def test_appends_accumulate(_events_dir: Path) -> None:
    store.append({"id": "a"}, when=T0)
    store.append({"id": "b"}, when=T0)
    assert len(list(store.iter_records(store.day_files()))) == 2


def test_a_malformed_line_is_skipped_not_raised(_events_dir: Path) -> None:
    store.append({"id": "a"}, when=T0)
    path = _events_dir / "2026-08-18.jsonl"
    with path.open("a") as f:
        f.write("{not json\n")
    store.append({"id": "b"}, when=T0)
    assert [r["id"] for r in store.iter_records([path])] == ["a", "b"]


def test_disabled_by_env(_events_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EVENTS", "0")
    assert store.enabled() is False
    store.append({"id": "a"}, when=T0)
    assert not _events_dir.exists()


def test_an_unwritable_store_never_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    monkeypatch.setenv("LAB_EVENTS_DIR", str(blocker))
    store.append({"id": "a"}, when=T0)  # must not raise


def test_day_files_are_sorted_oldest_first(_events_dir: Path) -> None:
    store.append({"id": "a"}, when=T0)
    store.append({"id": "b"}, when=T0.replace(day=19))
    assert [p.name for p in store.day_files()] == ["2026-08-18.jsonl", "2026-08-19.jsonl"]


def test_values_that_json_cannot_encode_do_not_lose_the_record(_events_dir: Path) -> None:
    store.append({"id": "a", "params": {"p": Path("/x")}}, when=T0)
    (record,) = list(store.iter_records(store.day_files()))
    assert record["params"]["p"] == "/x"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.events.store'`

- [ ] **Step 3: Write `src/lab/events/store.py`**

```python
"""On-disk ledger: one JSONL file per UTC day under ``~/.lab/events``.

User-global rather than project-local on purpose — the lab installs into other projects
(v0.5.0+), so a project-local store would scatter the history across repos exactly when the
cross-project pattern is the thing worth seeing. Every event carries its project, so
per-project filtering is a read-side concern.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DIR = "~/.lab/events"


def enabled() -> bool:
    return (os.environ.get("LAB_EVENTS") or "").strip() != "0"


def debug(message: str) -> None:
    """Surface a swallowed logging error, but only when asked (``LAB_EVENTS_DEBUG=1``)."""
    if (os.environ.get("LAB_EVENTS_DEBUG") or "").strip() == "1":
        print(f"[lab.events] {message}", file=sys.stderr)


def events_dir() -> Path:
    override = (os.environ.get("LAB_EVENTS_DIR") or "").strip()
    return Path(override).expanduser() if override else Path(DEFAULT_DIR).expanduser()


def day_file(when: datetime) -> Path:
    return events_dir() / f"{when.astimezone(tz=None).strftime('%Y-%m-%d')}.jsonl"


def day_files() -> list[Path]:
    try:
        return sorted(events_dir().glob("????-??-??.jsonl"))
    except OSError as e:
        debug(f"listing failed: {e}")
        return []


def append(record: dict[str, Any], *, when: datetime) -> None:
    """Append one record as a single locked ``O_APPEND`` write.

    A sharded sweep launches many ``lab`` processes against this one file; the lock is what
    rules out the torn or interleaved lines that would make the store untrustworthy in exactly
    the situation where it matters most. Best-effort throughout: a ledger failure must never
    fail a command.
    """
    if not enabled():
        return
    try:
        path = day_file(when)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:  # noqa: BLE001 — logging must never fail a command
        debug(f"append failed: {e}")


def iter_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield every parseable record. A ledger unreadable because one line is bad would fail at
    its only job, so malformed lines are skipped, never raised on."""
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        yield record
        except OSError as e:
            debug(f"read failed for {path}: {e}")
```

Note on `day_file`: the UTC day is what the spec calls for, and callers always pass a UTC-aware `when` (`lab._util.now`). `astimezone(tz=None)` would localise — replace that line with `when.strftime('%Y-%m-%d')` and let the caller own the timezone.

- [ ] **Step 4: Fix the timezone slip flagged above**

In `day_file`, use:

```python
    return events_dir() / f"{when.strftime('%Y-%m-%d')}.jsonl"
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_events_store.py -v && uv run mypy --strict src/lab/events`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lab/events/store.py tests/test_events_store.py
git commit -m "feat(events): locked append-only day-file store with tolerant reads"
```

---

### Task 3: Retention — compaction and caps, on an injected clock

**Files:**
- Modify: `src/lab/events/store.py`
- Test: `tests/test_events_retention.py`

**Interfaces:**
- Consumes: `store.events_dir`, `store.day_files`, `store.append`, `store.debug`.
- Produces:
  - `compact(*, now: datetime, success_ttl_days: int) -> None`
  - `enforce_caps(*, now: datetime, max_age_days: int, max_mb: float) -> None`
  - `maybe_prune(*, now: datetime) -> None` — stamp-gated, at most once per UTC day

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events_retention.py`:

```python
"""Retention runs on an injected clock. Real-clock tests anchored to a fixed T0 decay into
failures once the anchor ages — the scheduler watchdog already taught us that."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab.events import store

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


def _write(when: datetime, *records: dict) -> None:
    for record in records:
        store.append(record, when=when)


def _pair(id_: str, outcome: str) -> tuple[dict, dict]:
    return ({"id": id_, "phase": "open", "action": "submit"},
            {"id": id_, "phase": "close", "outcome": outcome})


def test_compaction_drops_old_successes_and_keeps_old_failures(_events_dir: Path) -> None:
    old = NOW - timedelta(days=30)
    _write(old, *_pair("ok1", "ok"), *_pair("bad1", "error"))
    store.compact(now=NOW, success_ttl_days=14)
    ids = {r["id"] for r in store.iter_records(store.day_files())}
    assert ids == {"bad1"}


def test_compaction_keeps_recent_successes(_events_dir: Path) -> None:
    _write(NOW - timedelta(days=3), *_pair("ok1", "ok"))
    store.compact(now=NOW, success_ttl_days=14)
    assert {r["id"] for r in store.iter_records(store.day_files())} == {"ok1"}


def test_compaction_keeps_dangling_opens_regardless_of_age(_events_dir: Path) -> None:
    _write(NOW - timedelta(days=60), {"id": "hung", "phase": "open", "action": "submit"})
    store.compact(now=NOW, success_ttl_days=14)
    assert {r["id"] for r in store.iter_records(store.day_files())} == {"hung"}


def test_age_cap_deletes_whole_day_files(_events_dir: Path) -> None:
    _write(NOW - timedelta(days=200), *_pair("ancient", "error"))
    _write(NOW - timedelta(days=2), *_pair("recent", "error"))
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=50)
    assert [p.name for p in store.day_files()] == [f"{(NOW - timedelta(days=2)).date()}.jsonl"]


def test_size_cap_deletes_oldest_first_until_under_budget(_events_dir: Path) -> None:
    blob = {"id": "x", "phase": "close", "outcome": "error", "pad": "p" * 2000}
    for age in (5, 4, 3):
        _write(NOW - timedelta(days=age), *[dict(blob, id=f"d{age}-{i}") for i in range(200)])
    store.enforce_caps(now=NOW, max_age_days=90, max_mb=0.5)
    total = sum(p.stat().st_size for p in store.day_files())
    assert total <= 0.5 * 1024 * 1024
    assert (NOW - timedelta(days=3)).strftime("%Y-%m-%d") in {p.stem for p in store.day_files()}


def test_maybe_prune_runs_once_per_day(_events_dir: Path) -> None:
    _write(NOW - timedelta(days=30), *_pair("ok1", "ok"))
    store.maybe_prune(now=NOW)
    assert {r["id"] for r in store.iter_records(store.day_files())} == set()
    # a second call the same day must not re-scan; re-add and confirm it survives
    _write(NOW - timedelta(days=30), *_pair("ok2", "ok"))
    store.maybe_prune(now=NOW)
    assert {r["id"] for r in store.iter_records(store.day_files())} == {"ok2"}
    store.maybe_prune(now=NOW + timedelta(days=1))
    assert {r["id"] for r in store.iter_records(store.day_files())} == set()


def test_pruning_failure_is_swallowed(_events_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "compact", lambda **_: (_ for _ in ()).throw(OSError("boom")))
    store.maybe_prune(now=NOW)  # must not raise
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_retention.py -v`
Expected: FAIL — `AttributeError: module 'lab.events.store' has no attribute 'compact'`

- [ ] **Step 3: Add retention to `src/lab/events/store.py`**

Append to the module:

```python
STAMP = ".pruned"


def _int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _rewrite(path: Path, records: list[dict[str, Any]]) -> None:
    """Replace a day file under the same lock appends take."""
    tmp = path.with_suffix(".jsonl.tmp")
    text = "".join(json.dumps(r, default=str, separators=(",", ":")) + "\n" for r in records)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def compact(*, now: datetime, success_ttl_days: int) -> None:
    """Drop successful calls older than the TTL. Failures — and dangling opens, which are
    themselves a finding — stay until the age cap takes them."""
    cutoff = now - timedelta(days=success_ttl_days)
    for path in day_files():
        try:
            if datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc) >= cutoff:
                continue
            records = list(iter_records([path]))
            succeeded = {r["id"] for r in records
                         if r.get("phase") == "close" and r.get("outcome") == "ok"}
            kept = [r for r in records if r.get("id") not in succeeded]
            if len(kept) != len(records):
                _rewrite(path, kept)
        except Exception as e:  # noqa: BLE001
            debug(f"compaction failed for {path}: {e}")


def enforce_caps(*, now: datetime, max_age_days: int, max_mb: float) -> None:
    """Delete whole day files past the age cap, then oldest-first until under the byte cap."""
    cutoff = now - timedelta(days=max_age_days)
    remaining: list[Path] = []
    for path in day_files():
        try:
            stamped = datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if stamped < cutoff:
            path.unlink(missing_ok=True)
        else:
            remaining.append(path)
    budget = max_mb * 1024 * 1024
    total = sum(p.stat().st_size for p in remaining if p.exists())
    for path in remaining:  # oldest first
        if total <= budget:
            break
        total -= path.stat().st_size
        path.unlink(missing_ok=True)


def maybe_prune(*, now: datetime) -> None:
    """Run retention at most once per UTC day per machine. Lazy, stamp-gated, best-effort."""
    if not enabled():
        return
    try:
        stamp = events_dir() / STAMP
        today = now.strftime("%Y-%m-%d")
        if stamp.exists() and stamp.read_text().strip() == today:
            return
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(today)
        compact(now=now, success_ttl_days=_int_env("LAB_EVENTS_SUCCESS_TTL_DAYS", 14))
        enforce_caps(
            now=now,
            max_age_days=_int_env("LAB_EVENTS_MAX_AGE_DAYS", 90),
            max_mb=_float_env("LAB_EVENTS_MAX_MB", 50),
        )
    except Exception as e:  # noqa: BLE001
        debug(f"pruning failed: {e}")
```

Add `from datetime import datetime, timedelta, timezone` to the imports at the top of the module.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_events_retention.py -v && uv run mypy --strict src/lab/events`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lab/events/store.py tests/test_events_retention.py
git commit -m "feat(events): two-stage retention — compact successes, cap age and bytes"
```

---

### Task 4: `begin`/`finish`/`record`/`note` — the writer

**Files:**
- Create: `src/lab/events/record.py`, `src/lab/events/annotate.py`
- Modify: `src/lab/events/__init__.py`
- Test: `tests/test_events_record.py`

**Interfaces:**
- Consumes: `store.append`, `store.maybe_prune`, `sanitize_params`, `lab._util.now`, `lab.__version__`.
- Produces:
  - `begin(surface: str, action: str, params: Mapping[str, Any]) -> Call`
  - `finish(call: Call, *, outcome: str, exit_code: int | None = None, error: dict | None = None) -> None`
  - `finish_current(*, outcome: str, exit_code: int | None = None, error: dict | None = None) -> None`
  - `current() -> Call | None`
  - `record(surface, action, params) -> ContextManager[Call]`
  - `note(kind: str, **fields: Any) -> None`
  - `error_dict(exc: BaseException) -> dict[str, Any]`
  - `Call` with `.ref(**ids) -> None` and `.result(**digest) -> None`
  - `annotate.refs_from(payload: Any) -> dict[str, Any]`, `annotate.digest_of(payload: Any) -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events_record.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from lab import events
from lab.events import store
from lab.events.annotate import digest_of, refs_from


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    monkeypatch.setenv("LAB_SESSION_ID", "sess_test")
    return tmp_path / "events"


def _records() -> list[dict]:
    return list(store.iter_records(store.day_files()))


def test_a_successful_call_writes_an_open_and_a_close_sharing_an_id() -> None:
    with events.record("cli", "status", {"job_id": "j-1"}):
        pass
    opened, closed = _records()
    assert opened["phase"] == "open" and closed["phase"] == "close"
    assert opened["id"] == closed["id"]
    assert opened["action"] == "status" and opened["surface"] == "cli"
    assert closed["outcome"] == "ok"
    assert closed["duration_ms"] >= 0


def test_the_open_line_is_written_before_the_body_runs() -> None:
    with events.record("cli", "submit", {}):
        assert [r["phase"] for r in _records()] == ["open"]


def test_params_are_sanitized_on_the_way_in() -> None:
    with events.record("cli", "submit", {"api_key": "x" * 40}):
        pass
    assert _records()[0]["params"]["api_key"] == "…REDACTED…"


def test_an_exception_records_a_crash_with_the_error_and_reraises() -> None:
    with pytest.raises(ValueError):
        with events.record("mcp", "submit", {}):
            raise ValueError("no capacity")
    closed = _records()[1]
    assert closed["outcome"] == "crash"
    assert closed["error"]["type"] == "ValueError"
    assert closed["error"]["message"] == "no capacity"
    assert "test_events_record.py" in closed["error"]["where"]


def test_keyboard_interrupt_records_interrupted() -> None:
    with pytest.raises(KeyboardInterrupt):
        with events.record("cli", "wait", {}):
            raise KeyboardInterrupt
    assert _records()[1]["outcome"] == "interrupted"


def test_notes_are_discarded_on_success() -> None:
    with events.record("cli", "submit", {}):
        events.note("provision.attempt", zone="europe-west1-b")
    assert _records()[1].get("trace") is None


def test_notes_are_flushed_into_the_trace_on_failure() -> None:
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            events.note("provision.attempt", zone="europe-west1-b")
            events.note("teardown.retry", attempt=2)
            raise RuntimeError("boom")
    trace = _records()[1]["trace"]
    assert [n["k"] for n in trace] == ["provision.attempt", "teardown.retry"]
    assert trace[0]["d"] == {"zone": "europe-west1-b"}
    assert all(isinstance(n["t"], int) for n in trace)


def test_note_outside_a_call_is_a_no_op() -> None:
    events.note("orphan", x=1)  # must not raise, must not write
    assert _records() == []


def test_the_ring_buffer_is_bounded() -> None:
    with pytest.raises(RuntimeError):
        with events.record("cli", "sweep", {}):
            for i in range(500):
                events.note("tick", i=i)
            raise RuntimeError("boom")
    trace = _records()[1]["trace"]
    assert len(trace) == 200
    assert trace[-1]["d"] == {"i": 499}  # the newest are the ones kept


def test_ref_and_result_land_on_the_close_record() -> None:
    with events.record("cli", "submit", {}) as call:
        call.ref(job_id="j-4f2a")
        call.result(state="failed", cost_usd=0.29)
    closed = _records()[1]
    assert closed["refs"] == {"job_id": "j-4f2a"}
    assert closed["result"] == {"state": "failed", "cost_usd": 0.29}


def test_seq_increments_within_a_process() -> None:
    with events.record("cli", "a", {}):
        pass
    with events.record("cli", "b", {}):
        pass
    assert [r["seq"] for r in _records() if r["phase"] == "open"] == [0, 1]


def test_session_id_comes_from_the_environment() -> None:
    with events.record("cli", "status", {}):
        pass
    assert _records()[0]["session"] == "sess_test"


def test_disabled_writes_nothing_and_still_yields_a_usable_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAB_EVENTS", "0")
    with events.record("cli", "status", {}) as call:
        call.ref(job_id="j-1")
        events.note("x")
    assert _records() == []


def test_refs_from_pulls_known_ids_out_of_a_payload() -> None:
    assert refs_from({"job_id": "j-1", "irrelevant": 5}) == {"job_id": "j-1"}
    assert refs_from({"sweep_id": "s-1", "jobs": [{"job_id": "j-1"}, {"job_id": "j-2"}]}) == {
        "sweep_id": "s-1", "job_ids": ["j-1", "j-2"]}
    assert refs_from("not a mapping") == {}


def test_digest_of_keeps_a_small_summary_not_the_payload() -> None:
    payload = {"state": "succeeded", "actual_cost_usd": 1.25, "series": list(range(1000))}
    assert digest_of(payload) == {"state": "succeeded", "cost_usd": 1.25, "series_n": 1000}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_record.py -v`
Expected: FAIL — `AttributeError: module 'lab.events' has no attribute 'record'`

- [ ] **Step 3: Write `src/lab/events/annotate.py`**

```python
"""Pull the join keys and a small digest out of a command's payload.

``result`` is deliberately a digest: the full payload already lives in the manifest, and a
second, staler copy on disk is how this kind of store gets fat."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ID_KEYS = ("job_id", "sweep_id", "reg_id", "run_id")
_COST_KEYS = ("actual_cost_usd", "cost_usd", "estimated_usd")


def refs_from(payload: Any) -> dict[str, Any]:
    """Join keys tying this call to manifests: ``job_id``, ``sweep_id``, ``reg_id``, ``run_id``,
    plus ``job_ids`` collected from any nested list of job-shaped dicts."""
    if not isinstance(payload, Mapping):
        return {}
    refs: dict[str, Any] = {k: payload[k] for k in _ID_KEYS if isinstance(payload.get(k), str)}
    ids: list[str] = []
    for value in payload.values():
        if isinstance(value, list):
            ids += [v["job_id"] for v in value
                    if isinstance(v, Mapping) and isinstance(v.get("job_id"), str)]
    if ids:
        refs["job_ids"] = ids
    return refs


def digest_of(payload: Any) -> dict[str, Any]:
    """A handful of scalars: state, cost, and the length of anything list-shaped."""
    if not isinstance(payload, Mapping):
        return {}
    digest: dict[str, Any] = {}
    if isinstance(payload.get("state"), str):
        digest["state"] = payload["state"]
    for key in _COST_KEYS:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            digest["cost_usd"] = float(value)
            break
    for key, value in payload.items():
        if isinstance(value, list):
            digest[f"{key}_n"] = len(value)
    return digest
```

- [ ] **Step 4: Write `src/lab/events/record.py`**

```python
"""The writer: one open/close pair per call, plus the ring buffer that explains failures.

Outcome, duration and ``error`` are derived here from how the block exits — a caller never sets
them, so a call cannot misreport its own success. ``ref()`` and ``result()`` are the only things
a caller adds.
"""

from __future__ import annotations

import os
import secrets
import traceback
import uuid
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from lab import __version__
from lab._util import now
from lab.events import store
from lab.events.sanitize import sanitize_params

RING = 200
_current: ContextVar["Call | None"] = ContextVar("lab_events_current", default=None)
_seq = 0
_session: str | None = None
_pruned = False


def _session_id() -> str:
    """``LAB_SESSION_ID`` when an agent harness sets it — exact grouping. Otherwise a
    per-process value; the session *view* does not depend on this being perfect."""
    global _session
    override = (os.environ.get("LAB_SESSION_ID") or "").strip()
    if override:
        return override
    if _session is None:
        _session = f"sess_{uuid.uuid4().hex[:8]}"
    return _session


def _new_id() -> str:
    """Time-ordered id: millisecond timestamp + randomness, sortable, needs no coordination."""
    return f"{int(now().timestamp() * 1000):013d}{secrets.token_hex(4)}"


def _project() -> dict[str, Any]:
    from lab.manifest import current_commit, is_dirty, repo_root

    try:
        root = repo_root()
        return {"name": root.name, "commit": current_commit(), "dirty": is_dirty()}
    except Exception as e:  # noqa: BLE001 — not every cwd is a repo
        store.debug(f"project probe failed: {e}")
        return {}


def error_dict(exc: BaseException) -> dict[str, Any]:
    """Type, message, and the innermost frame inside the lab — the ``where`` that matters."""
    tb = exc.__traceback__
    where = None
    while tb is not None:
        where = f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}"
        tb = tb.tb_next
    return {"type": type(exc).__name__, "message": str(exc)[:2048], "where": where}


class Call:
    """A call in flight. Annotate it with :meth:`ref` and :meth:`result`; nothing else."""

    def __init__(self, id: str, started: datetime, seq: int) -> None:
        self.id = id
        self.started = started
        self.seq = seq
        self.notes: deque[dict[str, Any]] = deque(maxlen=RING)
        self._refs: dict[str, Any] = {}
        self._result: dict[str, Any] = {}

    def ref(self, **ids: Any) -> None:
        self._refs.update({k: v for k, v in ids.items() if v is not None})

    def result(self, **digest: Any) -> None:
        self._result.update({k: v for k, v in digest.items() if v is not None})


def current() -> Call | None:
    return _current.get()


def begin(surface: str, action: str, params: Mapping[str, Any]) -> Call:
    """Open a call: write the open line immediately, so a call that never closes is visible."""
    global _seq, _pruned
    call = Call(_new_id(), now(), _seq)
    _seq += 1
    _current.set(call)
    if not _pruned:
        _pruned = True
        store.maybe_prune(now=now())
    store.append(
        {
            "id": call.id,
            "ts": call.started.isoformat(),
            "phase": "open",
            "session": _session_id(),
            "seq": call.seq,
            "surface": surface,
            "action": action,
            "params": sanitize_params(params),
            "project": _project(),
            "lab_version": __version__,
        },
        when=call.started,
    )
    return call


def finish(
    call: Call,
    *,
    outcome: str,
    exit_code: int | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    ended = now()
    record_: dict[str, Any] = {
        "id": call.id,
        "ts": ended.isoformat(),
        "phase": "close",
        "outcome": outcome,
        "exit_code": exit_code,
        "duration_ms": int((ended - call.started).total_seconds() * 1000),
        "refs": call._refs,
        "result": call._result,
        "error": error,
    }
    if outcome != "ok" and call.notes:
        record_["trace"] = list(call.notes)
    store.append(record_, when=ended)
    _current.set(None)


def finish_current(
    *, outcome: str, exit_code: int | None = None, error: dict[str, Any] | None = None
) -> None:
    """Close whatever call is open. The CLI opens in the group callback and closes in ``main``."""
    call = _current.get()
    if call is not None:
        finish(call, outcome=outcome, exit_code=exit_code, error=error)


@contextmanager
def record(surface: str, action: str, params: Mapping[str, Any]) -> Iterator[Call]:
    """Open a call, derive its outcome from how the block exits, close it. Re-raises unchanged."""
    call = begin(surface, action, params)
    try:
        yield call
    except KeyboardInterrupt:
        finish(call, outcome="interrupted")
        raise
    except BaseException as e:  # noqa: BLE001 — every exit path must be recorded
        finish(call, outcome="crash", error=error_dict(e))
        raise
    else:
        finish(call, outcome="ok")


def note(kind: str, **fields: Any) -> None:
    """Buffer an internal step. Discarded on success, flushed into ``trace`` on failure.

    Additive to the stderr diagnostic at the same site, never a replacement: the printed line is
    the live UX, this is the durable record.
    """
    call = _current.get()
    if call is None:
        return
    try:
        call.notes.append(
            {
                "t": int((now() - call.started).total_seconds() * 1000),
                "k": kind,
                "d": sanitize_params(fields),
            }
        )
    except Exception as e:  # noqa: BLE001
        store.debug(f"note failed: {e}")
```

- [ ] **Step 5: Export the writer from `src/lab/events/__init__.py`**

```python
"""The lab's ledger of its own behaviour. See
``docs/superpowers/specs/2026-08-18-event-logging-design.md``."""

from __future__ import annotations

from lab.events.models import ActionStat, Event, Note, SignatureStat, StatsView
from lab.events.record import (
    Call,
    begin,
    current,
    error_dict,
    finish,
    finish_current,
    note,
    record,
)

__all__ = [
    "ActionStat", "Call", "Event", "Note", "SignatureStat", "StatsView",
    "begin", "current", "error_dict", "finish", "finish_current", "note", "record",
]
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_events_record.py -v && uv run mypy --strict src/lab/events`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/lab/events tests/test_events_record.py
git commit -m "feat(events): open/close writer with a failure-only trace buffer"
```

---

### Task 5: CLI capture

The CLI opens the call in the group callback (so the `open` line lands before any work) and closes it in a new `main()`. `standalone_mode=False` is what makes `usage_error` and `crash` distinguishable, and what preserves the `__cause__` behind every `raise typer.Exit(code=1) from e` — the existing ~15 sites get rich errors with no edits.

**Files:**
- Modify: `src/lab/cli.py` (`_load_env` callback, `_emit`, new `main()`), `pyproject.toml:46`
- Test: `tests/test_events_cli.py`

**Interfaces:**
- Consumes: `events.begin`, `events.finish_current`, `events.current`, `events.error_dict`, `sanitize_argv`, `annotate.refs_from`, `annotate.digest_of`.
- Produces: `lab.cli.main() -> None` — the console entry point.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events_cli.py`:

```python
"""Every CLI exit path must land in the ledger, and none may change the exit code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lab.events import store


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


def _run(*args: str, env_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "from lab.cli import main; main()", *args],
        capture_output=True, text=True,
        env={**dict(__import__("os").environ), "LAB_EVENTS_DIR": str(env_dir)},
    )


def _folded(dir_: Path) -> list[dict]:
    return list(store.iter_records(sorted(dir_.glob("*.jsonl"))))


def test_a_successful_command_records_ok(_events_dir: Path) -> None:
    proc = _run("list", env_dir=_events_dir)
    assert proc.returncode == 0
    opened, closed = _folded(_events_dir)
    assert opened["action"] == "list" and opened["surface"] == "cli"
    assert closed["outcome"] == "ok" and closed["exit_code"] == 0


def test_an_unknown_job_records_an_error_and_keeps_exit_code_2(_events_dir: Path) -> None:
    proc = _run("status", "j-does-not-exist", env_dir=_events_dir)
    assert proc.returncode == 2
    closed = _folded(_events_dir)[1]
    assert closed["outcome"] == "error"
    assert closed["exit_code"] == 2


def test_a_bad_flag_records_a_usage_error(_events_dir: Path) -> None:
    proc = _run("list", "--nonexistent-flag", env_dir=_events_dir)
    assert proc.returncode == 2
    closed = _folded(_events_dir)[-1]
    assert closed["outcome"] == "usage_error"
    assert "no such option" in closed["error"]["message"].lower()


def test_an_unknown_command_records_a_usage_error_with_sanitized_argv(_events_dir: Path) -> None:
    _run("nosuchcommand", "--token", "s" * 40, env_dir=_events_dir)
    opened = _folded(_events_dir)[0]
    assert opened["action"] == "<unparsed>"
    assert "…REDACTED…" in opened["params"]["argv"]


def test_the_cause_behind_typer_exit_becomes_the_recorded_error(
    monkeypatch: pytest.MonkeyPatch, _events_dir: Path
) -> None:
    import typer

    from lab import cli

    @cli.app.command()
    def boom() -> None:  # a stand-in for the ~15 `raise typer.Exit(code=1) from e` sites
        try:
            raise RuntimeError("no capacity in europe-west1")
        except RuntimeError as e:
            raise typer.Exit(code=1) from e

    with pytest.raises(SystemExit) as exc:
        cli.main(["boom"])
    assert exc.value.code == 1
    closed = _folded(_events_dir)[1]
    assert closed["outcome"] == "error"
    assert closed["error"]["type"] == "RuntimeError"
    assert closed["error"]["message"] == "no capacity in europe-west1"


def test_emit_annotates_the_call_with_refs_and_a_digest(
    monkeypatch: pytest.MonkeyPatch, _events_dir: Path
) -> None:
    from lab import cli, events

    with events.record("cli", "submit", {}):
        cli._emit({"job_id": "j-4f2a", "state": "succeeded", "actual_cost_usd": 1.25})
    closed = _folded(_events_dir)[1]
    assert closed["refs"] == {"job_id": "j-4f2a"}
    assert closed["result"] == {"state": "succeeded", "cost_usd": 1.25}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'main' from 'lab.cli'`

- [ ] **Step 3: Open the call in the group callback**

In `src/lab/cli.py`, add to the imports:

```python
import click

from lab import events
from lab.events.annotate import digest_of, refs_from
from lab.events.sanitize import sanitize_argv
```

At the end of the `_load_env` body (after `_warn_if_repo_override_shadows_cwd()`), add:

```python
    ctx = click.get_current_context(silent=True)
    if ctx is not None and ctx.invoked_subcommand:
        events.begin("cli", ctx.invoked_subcommand, {"argv": sanitize_argv(sys.argv[1:])})
```

Why `argv` rather than parsed options: the group callback runs after the subcommand *name* is
resolved but before the subcommand's own options are parsed, so `ctx.params` here holds only the
group's flags. Recording the sanitized command line captures everything that was actually typed,
which is what a reader wants anyway — and it keeps the `open` line identical in shape whether or
not parsing later succeeds.

- [ ] **Step 4: Extend `_emit` to annotate the open call**

Replace `_emit` in `src/lab/cli.py:128-129` with:

```python
def _emit(obj: Any) -> None:
    """Print a command's JSON payload, and annotate the open ledger call with its ids/digest.

    Nearly every command funnels its result through here, which makes it the one place to learn
    what a call produced without touching each command.
    """
    call = events.current()
    if call is not None:
        call.ref(**refs_from(obj))
        call.result(**digest_of(obj))
    typer.echo(json.dumps(obj, indent=2, default=str))
```

- [ ] **Step 5: Write `main()` at the bottom of `src/lab/cli.py`**

Replace the `if __name__ == "__main__": app()` block with:

```python
def main(argv: list[str] | None = None) -> None:
    """Console entry point. Observational only — every exit code and stream behaviour below
    matches click's own standalone handling, because `lab wait`'s exit codes (3 teardown, 4
    fail-fast) are contract.

    ``standalone_mode=False`` is what makes the ledger useful: click stops swallowing
    ``UsageError`` (so a caller misusing the interface is distinguishable from a command that
    failed) and ``typer.Exit`` arrives with the ``__cause__`` that every
    ``raise typer.Exit(code=1) from e`` site already sets.
    """
    outcome, code, error = "ok", 0, None
    try:
        app(args=argv, standalone_mode=False)
    except click.exceptions.Exit as e:
        code = int(e.exit_code)
        outcome = "ok" if code == 0 else "error"
        error = events.error_dict(e.__cause__) if e.__cause__ else (
            None if code == 0 else {"type": "Exit", "message": f"exited {code}", "where": None}
        )
    except click.UsageError as e:
        e.show()
        code, outcome = int(e.exit_code), "usage_error"
        error = {"type": "UsageError", "message": str(e), "where": None}
    except click.Abort:
        typer.echo("Aborted.", err=True)
        code, outcome = 1, "interrupted"
    except KeyboardInterrupt:
        code, outcome = 130, "interrupted"
    except Exception as e:  # noqa: BLE001 — record, then behave exactly as before
        traceback.print_exc()
        code, outcome, error = 1, "crash", events.error_dict(e)
    if events.current() is None and outcome != "ok":
        # Parsing never reached the group callback (unknown command, group-level bad flag), so
        # no call is open. Synthesise one: a caller getting the interface wrong is a finding.
        events.begin("cli", "<unparsed>", {"argv": sanitize_argv(sys.argv[1:])})
    events.finish_current(outcome=outcome, exit_code=code, error=error)
    sys.exit(code)


if __name__ == "__main__":
    main()
```

Add `import traceback` to the imports.

- [ ] **Step 6: Move the console entry point**

In `pyproject.toml:46`, change:

```toml
lab = "lab.cli:main"
```

- [ ] **Step 7: Run the new tests and the whole CLI suite**

Run: `uv run pytest tests/test_events_cli.py tests/test_cli_wait.py tests/test_cli_spot.py tests/test_cli_init.py -v`
Expected: PASS. The pre-existing CLI tests are the regression gate on `standalone_mode=False` — if any exit code moved, the wrapper is wrong, not the test.

- [ ] **Step 8: Commit**

```bash
git add src/lab/cli.py pyproject.toml tests/test_events_cli.py
git commit -m "feat(events): record every CLI invocation, including usage errors and crashes"
```

---

### Task 6: MCP capture

**Files:**
- Modify: `src/lab/mcp_server.py`
- Test: `tests/test_events_mcp.py`

**Interfaces:**
- Consumes: `events.record`, `annotate.refs_from`, `annotate.digest_of`.
- Produces: `lab.mcp_server.EventMiddleware` — registered inside `build_server`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_events_mcp.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from lab.core import default_lab
from lab.events import store
from lab.mcp_server import build_server


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


def _records() -> list[dict]:
    return list(store.iter_records(store.day_files()))


@pytest.mark.anyio
async def test_a_tool_call_records_an_open_close_pair(tmp_path: Path) -> None:
    server = build_server(default_lab(home=tmp_path / "runs"))
    async with Client(server) as client:
        await client.call_tool("list", {})
    opened, closed = _records()
    assert opened["surface"] == "mcp" and opened["action"] == "list"
    assert closed["outcome"] == "ok"


@pytest.mark.anyio
async def test_a_tool_error_records_an_error(tmp_path: Path) -> None:
    server = build_server(default_lab(home=tmp_path / "runs"))
    async with Client(server) as client:
        with pytest.raises(Exception):
            await client.call_tool("status", {"job_id": "j-nope"})
    closed = _records()[1]
    assert closed["outcome"] in ("error", "crash")
    assert "j-nope" in closed["error"]["message"]
```

If `tests/test_mcp_server.py` already establishes an anyio/asyncio fixture convention, copy it rather than introducing a second one.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_events_mcp.py -v`
Expected: FAIL — no events are written.

- [ ] **Step 3: Add the middleware to `src/lab/mcp_server.py`**

Add the imports and the class above `build_server`:

```python
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware, MiddlewareContext

from lab import events
from lab.events.annotate import digest_of, refs_from


class EventMiddleware(Middleware):
    """Record every tool call in the ledger (one open/close pair), leaving results untouched."""

    async def on_call_tool(self, context: MiddlewareContext, call_next: Any) -> Any:
        name = getattr(context.message, "name", "<unknown>")
        arguments = getattr(context.message, "arguments", None) or {}
        with events.record("mcp", name, dict(arguments)) as call:
            result = await call_next(context)
            payload = getattr(result, "structured_content", None) or getattr(result, "data", None)
            call.ref(**refs_from(payload))
            call.result(**digest_of(payload))
            return result
```

A `ToolError` raised by a tool propagates through `record()` and is recorded as `crash` with
`error.type == "ToolError"`, which is the honest classification — the tool raised, the middleware
did not choose the outcome.

- [ ] **Step 4: Register it in `build_server`**

Immediately after `mcp: FastMCP = FastMCP("laboratory")`, add:

```python
    mcp.add_middleware(EventMiddleware())
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_events_mcp.py tests/test_mcp_server.py -v && uv run mypy --strict src/lab`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lab/mcp_server.py tests/test_events_mcp.py
git commit -m "feat(events): record MCP tool calls via on_call_tool middleware"
```

---

### Task 7: `read()` — fold the pairs, filter, order

**Files:**
- Create: `src/lab/events/read.py`
- Modify: `src/lab/events/__init__.py`
- Test: `tests/test_events_read.py`

**Interfaces:**
- Consumes: `store.iter_records`, `store.day_files`, `models.Event`, `models.Note`, `lab._util.parse_duration`.
- Produces:
  - `fold(records: Iterable[dict]) -> list[Event]`
  - `read(*, since: str | None = None, project: str | None = None, action: str | None = None, session: str | None = None, job: str | None = None, failures_only: bool = False, limit: int | None = None, now_: datetime | None = None) -> list[Event]` — newest first
  - `row(event: Event, *, full: bool = False) -> dict[str, Any]` — the display shape both shells emit

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events_read.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from lab.events.read import fold

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def _open(id_: str, **over) -> dict:
    base = {"id": id_, "ts": NOW.isoformat(), "phase": "open", "session": "s", "seq": 0,
            "surface": "cli", "action": "submit", "params": {}, "project": {"name": "lab"},
            "lab_version": "0.5.1"}
    return {**base, **over}


def _close(id_: str, **over) -> dict:
    base = {"id": id_, "ts": (NOW + timedelta(seconds=2)).isoformat(), "phase": "close",
            "outcome": "ok", "exit_code": 0, "duration_ms": 2000, "refs": {}, "result": {},
            "error": None}
    return {**base, **over}


def test_a_pair_folds_into_one_event() -> None:
    (event,) = fold([_open("a"), _close("a", outcome="error", exit_code=1)])
    assert event.id == "a" and event.action == "submit"
    assert event.outcome == "error" and event.exit_code == 1
    assert event.status == "error"


def test_a_dangling_open_folds_into_a_running_or_died_row() -> None:
    (event,) = fold([_open("a")])
    assert event.outcome is None
    assert event.status == "running-or-died"
    assert event.failed is True


def test_a_close_without_an_open_is_dropped() -> None:
    assert fold([_close("orphan")]) == []


def test_pairs_split_across_day_files_still_fold() -> None:
    records = [_open("a"), _open("b"), _close("a"), _close("b")]
    assert {e.id for e in fold(records)} == {"a", "b"}


def test_trace_becomes_note_objects() -> None:
    (event,) = fold([_open("a"), _close("a", outcome="crash",
                                        trace=[{"t": 5, "k": "provision.attempt", "d": {"z": "b"}}])])
    assert event.trace[0].k == "provision.attempt"
    assert event.trace[0].d == {"z": "b"}


def test_events_are_newest_first() -> None:
    older = _open("a", ts=(NOW - timedelta(hours=1)).isoformat())
    events = fold([older, _close("a"), _open("b"), _close("b")])
    assert [e.id for e in events] == ["b", "a"]
```

Add the filter tests to the same file:

```python
import pytest
from pathlib import Path
from lab.events import store
from lab.events.read import read


@pytest.fixture
def _ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    store.append(_open("a", action="submit", project={"name": "capacity"}), when=NOW)
    store.append(_close("a", outcome="error", refs={"job_id": "j-1"}), when=NOW)
    store.append(_open("b", action="doctor", project={"name": "lab"},
                       ts=(NOW - timedelta(days=5)).isoformat()), when=NOW - timedelta(days=5))
    store.append(_close("b"), when=NOW - timedelta(days=5))


def test_read_filters_by_action_project_failures_and_job(_ledger: None) -> None:
    assert [e.id for e in read(action="doctor")] == ["b"]
    assert [e.id for e in read(project="capacity")] == ["a"]
    assert [e.id for e in read(failures_only=True)] == ["a"]
    assert [e.id for e in read(job="j-1")] == ["a"]


def test_read_since_uses_the_duration_parser(_ledger: None) -> None:
    assert [e.id for e in read(since="2d", now_=NOW)] == ["a"]
    assert {e.id for e in read(since="30d", now_=NOW)} == {"a", "b"}


def test_read_limit_applies_after_filtering(_ledger: None) -> None:
    assert len(read(limit=1)) == 1


def test_row_is_brief_by_default_and_detailed_with_full() -> None:
    from lab.events.read import row

    (event,) = fold([_open("a", params={"backend": "cpu"}),
                     _close("a", outcome="error",
                            trace=[{"t": 5, "k": "provision.attempt", "d": {"z": "b"}}])])
    brief = row(event)
    assert brief["status"] == "error" and brief["action"] == "submit"
    assert "trace" not in brief and "params" not in brief
    detailed = row(event, full=True)
    assert detailed["params"] == {"backend": "cpu"}
    assert detailed["trace"][0]["k"] == "provision.attempt"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_read.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.events.read'`

- [ ] **Step 3: Write `src/lab/events/read.py`**

```python
"""Fold the ledger's open/close lines into rows, and filter them. Pure functions over JSONL."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from lab._util import now as _now
from lab._util import parse_duration
from lab.events import store
from lab.events.models import Event, Note


def _dt(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def fold(records: Iterable[dict]) -> list[Event]:
    """Pair opens with closes. A close with no open is dropped (its open aged out); an open with
    no close becomes a ``running-or-died`` row, which is itself the finding."""
    events: dict[str, Event] = {}
    closes: dict[str, dict] = {}
    for record in records:
        id_ = record.get("id")
        if not isinstance(id_, str):
            continue
        if record.get("phase") == "open":
            ts = _dt(record.get("ts"))
            if ts is None:
                continue
            events[id_] = Event(
                id=id_, ts=ts, session=str(record.get("session", "")),
                seq=int(record.get("seq") or 0), surface=str(record.get("surface", "")),
                action=str(record.get("action", "")), params=dict(record.get("params") or {}),
                project=dict(record.get("project") or {}),
                lab_version=str(record.get("lab_version", "")),
            )
        elif record.get("phase") == "close":
            closes[id_] = record
    for id_, close in closes.items():
        event = events.get(id_)
        if event is None:
            continue
        event.outcome = close.get("outcome")
        event.exit_code = close.get("exit_code")
        event.duration_ms = close.get("duration_ms")
        event.refs = dict(close.get("refs") or {})
        event.result = dict(close.get("result") or {})
        event.error = close.get("error")
        event.trace = [
            Note(t=int(n.get("t") or 0), k=str(n.get("k", "")), d=dict(n.get("d") or {}))
            for n in (close.get("trace") or [])
        ]
    return sorted(events.values(), key=lambda e: e.ts, reverse=True)


def read(
    *,
    since: str | None = None,
    project: str | None = None,
    action: str | None = None,
    session: str | None = None,
    job: str | None = None,
    failures_only: bool = False,
    limit: int | None = None,
    now_: datetime | None = None,
) -> list[Event]:
    """Folded rows, newest first. ``since`` takes a duration string (``2d``, ``30m``)."""
    events = fold(store.iter_records(store.day_files()))
    if since:
        seconds = parse_duration(since)
        if seconds is not None:
            cutoff = (now_ or _now()) - timedelta(seconds=seconds)
            events = [e for e in events if e.ts >= cutoff]
    if project:
        events = [e for e in events if e.project.get("name") == project]
    if action:
        events = [e for e in events if e.action == action]
    if session:
        events = [e for e in events if e.session == session]
    if job:
        events = [e for e in events
                  if job == e.refs.get("job_id") or job in (e.refs.get("job_ids") or [])]
    if failures_only:
        events = [e for e in events if e.failed]
    return events[:limit] if limit else events


def row(event: Event, *, full: bool = False) -> dict[str, Any]:
    """The display shape. Lives here so the CLI and the MCP server emit identical rows without
    either shell importing the other."""
    out: dict[str, Any] = {
        "id": event.id, "ts": event.ts.isoformat(), "action": event.action,
        "surface": event.surface, "status": event.status, "duration_ms": event.duration_ms,
        "project": event.project.get("name"), "refs": event.refs, "result": event.result,
        "error": (event.error or {}).get("message"),
    }
    if full:
        out |= {
            "params": event.params, "session": event.session, "exit_code": event.exit_code,
            "lab_version": event.lab_version, "error_detail": event.error,
            "trace": [vars(n) for n in event.trace],
        }
    return out
```

`read.py` needs `from typing import Any` for `row`'s return type.

- [ ] **Step 4: Export `fold`, `read` and `row` from `src/lab/events/__init__.py`**

Add `from lab.events.read import fold, read, row` and extend `__all__` with `"fold"`, `"read"`,
`"row"`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_events_read.py -v && uv run mypy --strict src/lab/events`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lab/events tests/test_events_read.py
git commit -m "feat(events): fold open/close pairs into filterable rows"
```

---

### Task 8: `stats()` and the error signature

The signature is what makes months of events actionable rather than merely voluminous: twelve occurrences of one bug must collapse into one ranked row.

**Files:**
- Create: `src/lab/events/stats.py`
- Modify: `src/lab/events/__init__.py`
- Test: `tests/test_events_stats.py`

**Interfaces:**
- Consumes: `models.Event`, `models.ActionStat`, `models.SignatureStat`, `models.StatsView`.
- Produces:
  - `signature(error: dict | None) -> str`
  - `stats(events: Sequence[Event], *, since: datetime | None = None) -> StatsView`
  - `stats_dict(view: StatsView) -> dict[str, Any]` — the JSON shape `lab history --stats` emits

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events_stats.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from lab.events.models import Event
from lab.events.stats import signature, stats

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def _event(id_: str, *, action="submit", outcome="ok", error=None, usd=None, ms=1000,
           ts=NOW) -> Event:
    return Event(id=id_, ts=ts, session="s", seq=0, surface="cli", action=action,
                 outcome=outcome, duration_ms=ms, error=error,
                 result={"cost_usd": usd} if usd else {})


def test_signature_normalizes_ids_numbers_paths_and_zones() -> None:
    a = {"type": "ProvisionTimeout", "message": "host never reached UP in 20m (europe-west1-b)"}
    b = {"type": "ProvisionTimeout", "message": "host never reached UP in 45m (us-central1-a)"}
    assert signature(a) == signature(b)


def test_signature_keeps_different_bugs_apart() -> None:
    a = {"type": "ProvisionTimeout", "message": "host never reached UP in 20m"}
    b = {"type": "TeardownFailed", "message": "sky.down exhausted 3 retries"}
    assert signature(a) != signature(b)


def test_signature_of_a_missing_error_is_stable() -> None:
    assert signature(None) == "unknown"


def test_stats_counts_calls_failures_and_failure_rate() -> None:
    view = stats([_event("1"), _event("2", outcome="error"), _event("3", action="doctor")])
    submit = next(a for a in view.actions if a.action == "submit")
    assert submit.calls == 2 and submit.failures == 1 and submit.failure_rate == 0.5
    assert view.total == 3 and view.failures == 1


def test_stats_counts_dangling_opens_as_failures() -> None:
    view = stats([_event("1", outcome=None)])
    assert view.dangling == 1 and view.failures == 1


def test_stats_sums_dollars_burned_in_failed_calls_only() -> None:
    view = stats([_event("1", outcome="error", usd=0.29), _event("2", outcome="ok", usd=5.0)])
    assert view.usd_burned == 0.29


def test_stats_ranks_signatures_by_count_and_records_the_window_seen() -> None:
    err = {"type": "ProvisionTimeout", "message": "host never reached UP in 20m"}
    other = {"type": "TeardownFailed", "message": "sky.down exhausted 3 retries"}
    events = [_event(str(i), outcome="error", error=err, ts=NOW - timedelta(hours=i))
              for i in range(3)]
    events.append(_event("x", outcome="error", error=other))
    view = stats(events)
    assert view.signatures[0].count == 3
    assert view.signatures[0].first_seen < view.signatures[0].last_seen
    assert view.signatures[0].actions == ["submit"]


def test_median_duration_is_reported_per_action() -> None:
    view = stats([_event("1", ms=1000), _event("2", ms=3000), _event("3", ms=2000)])
    assert view.actions[0].median_ms == 2000
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.events.stats'`

- [ ] **Step 3: Write `src/lab/events/stats.py`**

```python
"""Aggregate the ledger into a fix-this-first list.

Everything here hangs off the *error signature*: type plus the message with ids, numbers, paths
and zone names normalized out. Without it, twelve occurrences of one bug are twelve unique
strings and the aggregate view is voluminous rather than actionable.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from lab.events.models import ActionStat, Event, SignatureStat, StatsView

# Order matters: zones before bare numbers, paths before shas.
_NORMALIZERS = (
    (re.compile(r"\b[a-z]+-[a-z]+\d+(?:-[a-z])?\b"), "<zone>"),
    (re.compile(r"(?:/[\w.\-]+){2,}"), "<path>"),
    (re.compile(r"\b[0-9a-f]{7,40}\b"), "<sha>"),
    (re.compile(r"\bj-[0-9a-z]+\b"), "<job>"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<n>"),
)


def signature(error: dict[str, Any] | None) -> str:
    """A stable key for 'the same bug', across differing ids, zones and magnitudes."""
    if not error:
        return "unknown"
    message = str(error.get("message", ""))
    for pattern, placeholder in _NORMALIZERS:
        message = pattern.sub(placeholder, message)
    return f"{error.get('type', 'Error')}: {message[:120]}".strip()


def stats(events: Sequence[Event], *, since: datetime | None = None) -> StatsView:
    """Per-action failure rates, ranked error signatures, and the money spent on failures."""
    by_action: dict[str, list[Event]] = defaultdict(list)
    by_sig: dict[str, list[Event]] = defaultdict(list)
    usd_burned = 0.0
    dangling = 0
    for event in events:
        by_action[event.action].append(event)
        if event.outcome is None:
            dangling += 1
        if event.failed:
            by_sig[signature(event.error)].append(event)
            cost = event.result.get("cost_usd")
            if isinstance(cost, (int, float)):
                usd_burned += float(cost)

    actions = []
    for action, group in by_action.items():
        failures = sum(1 for e in group if e.failed)
        durations = [e.duration_ms for e in group if isinstance(e.duration_ms, int)]
        actions.append(
            ActionStat(
                action=action, calls=len(group), failures=failures,
                failure_rate=round(failures / len(group), 3),
                median_ms=int(statistics.median(durations)) if durations else 0,
            )
        )
    actions.sort(key=lambda a: (-a.failures, -a.calls, a.action))

    signatures = [
        SignatureStat(
            signature=sig,
            count=len(group),
            first_seen=min(e.ts for e in group),
            last_seen=max(e.ts for e in group),
            actions=sorted({e.action for e in group}),
            usd=round(sum(float(e.result.get("cost_usd") or 0) for e in group), 4),
        )
        for sig, group in by_sig.items()
    ]
    signatures.sort(key=lambda s: (-s.count, -s.usd, s.signature))

    return StatsView(
        since=since,
        total=len(events),
        failures=sum(1 for e in events if e.failed),
        dangling=dangling,
        usd_burned=round(usd_burned, 4),
        actions=actions,
        signatures=signatures,
    )


def stats_dict(view: StatsView) -> dict[str, Any]:
    """The JSON shape ``lab history --stats`` emits."""
    return {
        "since": view.since.isoformat() if view.since else None,
        "total": view.total,
        "failures": view.failures,
        "dangling": view.dangling,
        "usd_burned": view.usd_burned,
        "actions": [vars(a) for a in view.actions],
        "signatures": [
            {**vars(s), "first_seen": s.first_seen.isoformat(), "last_seen": s.last_seen.isoformat()}
            for s in view.signatures
        ],
    }
```

- [ ] **Step 4: Export from `src/lab/events/__init__.py`**

Add `from lab.events.stats import signature, stats, stats_dict` and extend `__all__`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_events_stats.py -v && uv run mypy --strict src/lab/events`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lab/events tests/test_events_stats.py
git commit -m "feat(events): aggregate stats built on a normalized error signature"
```

---

### Task 9: `report()` — the markdown digest

**Files:**
- Create: `src/lab/events/report.py`
- Modify: `src/lab/events/__init__.py`
- Test: `tests/test_events_report.py`

**Interfaces:**
- Consumes: `models.Event`, `stats.stats`, `stats.signature`.
- Produces:
  - `report(events: Sequence[Event], *, since: datetime | None = None) -> str`
  - `report_dict(events: Sequence[Event], *, since: datetime | None = None) -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events_report.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from lab.events.models import Event
from lab.events.report import report

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
ERR = {"type": "ProvisionTimeout", "message": "host never reached UP in 20m",
       "where": "lab/backends/skypilot.py:612"}


def _event(id_, **over) -> Event:
    base = dict(id=id_, ts=NOW, session="s", seq=0, surface="cli", action="submit",
                params={"backend": "cpu"}, outcome="error", duration_ms=1000, error=ERR,
                refs={"job_id": "j-1"}, result={"cost_usd": 0.29})
    return Event(**{**base, **over})


def test_report_is_markdown_with_a_triage_table() -> None:
    text = report([_event("a")], since=NOW)
    assert text.startswith("# ")
    assert "| Finding |" in text and "ProvisionTimeout" in text


def test_report_ranks_by_frequency_and_dollars() -> None:
    cheap = {"type": "Cheap", "message": "harmless"}
    events = [_event(str(i), error=cheap, result={}) for i in range(5)]
    events += [_event("x", result={"cost_usd": 12.0})]
    text = report(events)
    assert text.index("ProvisionTimeout") < text.index("Cheap")


def test_report_records_attempted_observed_and_cost_per_finding() -> None:
    text = report([_event("a")])
    assert "**Attempted:**" in text and "**Observed:**" in text and "**Cost:**" in text
    assert "j-1" in text  # the job id, so the reader can reach the manifest and logs.txt


def test_a_clean_window_says_so_rather_than_printing_an_empty_table() -> None:
    text = report([Event(id="a", ts=NOW, session="s", seq=0, surface="cli", action="list",
                         outcome="ok", duration_ms=5)])
    assert "no failures" in text.lower()


def test_dangling_opens_appear_as_their_own_finding() -> None:
    text = report([_event("a", outcome=None, error=None)])
    assert "running-or-died" in text or "never closed" in text.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.events.report'`

- [ ] **Step 3: Write `src/lab/events/report.py`**

```python
"""The markdown digest — a field report generated instead of hand-written.

Shaped after ``FIELD-REPORT-2026-08-12-capability-campaign.md``: a triage table first, then
per-finding *attempted / observed / cost*, with the job ids that reach the manifest and logs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from lab.events.models import Event
from lab.events.stats import signature, stats


def _severity(count: int, usd: float) -> float:
    """Frequency, weighted by what it burned. A cheap bug seen five times outranks an expensive
    one seen once only when the money is small — which is the call the reader wants made."""
    return count * (1.0 + usd)


def _params_line(event: Event) -> str:
    items = ", ".join(f"{k}={v}" for k, v in list(event.params.items())[:6])
    return f"`lab {event.action}` ({items})" if items else f"`lab {event.action}`"


def report(events: Sequence[Event], *, since: datetime | None = None) -> str:
    """Render failures in the window as a pasteable markdown report."""
    view = stats(events, since=since)
    window = f"since {since.isoformat()}" if since else "all recorded history"
    lines: list[str] = [
        f"# Lab event report — {window}",
        "",
        f"{view.total} calls, {view.failures} failed, {view.dangling} never closed, "
        f"${view.usd_burned:.4f} burned in failed calls.",
        "",
    ]
    failures = [e for e in events if e.failed]
    if not failures:
        lines += ["**No failures in this window.**", ""]
        return "\n".join(lines)

    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in failures:
        key = "never closed (running-or-died)" if event.outcome is None else signature(event.error)
        grouped[key].append(event)
    ranked = sorted(
        grouped.items(),
        key=lambda kv: -_severity(len(kv[1]),
                                  sum(float(e.result.get("cost_usd") or 0) for e in kv[1])),
    )

    lines += ["## Triage", "", "| # | Finding | Seen | $ burned | Actions |", "|---|---|---|---|---|"]
    for i, (key, group) in enumerate(ranked, 1):
        usd = sum(float(e.result.get("cost_usd") or 0) for e in group)
        actions = ", ".join(sorted({e.action for e in group}))
        lines.append(f"| F{i} | {key} | {len(group)} | ${usd:.4f} | {actions} |")
    lines += ["", "---", ""]

    for i, (key, group) in enumerate(ranked, 1):
        newest = max(group, key=lambda e: e.ts)
        usd = sum(float(e.result.get("cost_usd") or 0) for e in group)
        job_ids = sorted({str(e.refs.get("job_id")) for e in group if e.refs.get("job_id")})
        lines += [
            f"## F{i} — {key}",
            "",
            f"**Attempted:** {_params_line(newest)}  ",
            f"**Observed:** {newest.status}"
            + (f" — {newest.error.get('message')}" if newest.error else "")
            + (f" (at `{newest.error.get('where')}`)" if newest.error else "") + "  ",
            f"**Seen:** {len(group)}× between {min(e.ts for e in group).isoformat()} "
            f"and {max(e.ts for e in group).isoformat()}  ",
            f"**Cost:** ${usd:.4f}  ",
        ]
        if job_ids:
            lines.append(f"**Jobs:** {', '.join(job_ids)} (`runs/<job_id>/logs.txt`)  ")
        if newest.trace:
            lines += ["", "Trace of the most recent occurrence:", "", "```"]
            lines += [f"+{n.t:>7}ms  {n.k}  {n.d}" for n in newest.trace]
            lines += ["```"]
        lines.append("")
    return "\n".join(lines)


def report_dict(events: Sequence[Event], *, since: datetime | None = None) -> dict[str, Any]:
    """``{markdown}`` — the MCP tool's return shape."""
    return {"markdown": report(events, since=since)}
```

- [ ] **Step 4: Export from `src/lab/events/__init__.py`**

Add `from lab.events.report import report, report_dict` and extend `__all__`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_events_report.py -v && uv run mypy --strict src/lab/events`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lab/events tests/test_events_report.py
git commit -m "feat(events): markdown digest shaped like the hand-written field report"
```

---

### Task 10: `lab history` and `lab report` commands

**Files:**
- Modify: `src/lab/cli.py`
- Test: `tests/test_events_history_cli.py`

**Interfaces:**
- Consumes: `events.read`, `events.stats`, `events.stats_dict`, `events.report`.
- Produces: CLI commands `history` and `report`.

`lab history` emits JSON (the stdout-is-JSON convention every other command follows); `lab report` emits markdown, because it is the human-facing artifact. `logs` is left untouched — `lab logs <job_id>` still means "that job's stdout", and the collision is exactly why the new command is not called `lab log`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events_history_cli.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lab.cli import app
from lab.events import store

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
runner = CliRunner()


@pytest.fixture(autouse=True)
def _ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    store.append({"id": "a", "ts": NOW.isoformat(), "phase": "open", "session": "s", "seq": 0,
                  "surface": "cli", "action": "submit", "params": {"backend": "cpu"},
                  "project": {"name": "capacity"}, "lab_version": "0.5.1"}, when=NOW)
    store.append({"id": "a", "ts": NOW.isoformat(), "phase": "close", "outcome": "error",
                  "exit_code": 1, "duration_ms": 2000, "refs": {"job_id": "j-1"},
                  "result": {"cost_usd": 0.29},
                  "error": {"type": "ProvisionTimeout", "message": "no capacity"},
                  "trace": [{"t": 5, "k": "provision.attempt", "d": {"zone": "europe-west1-b"}}]},
                 when=NOW)


def test_history_emits_json_rows_newest_first() -> None:
    result = runner.invoke(app, ["history", "--all-projects"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["events"][0]["id"] == "a"
    assert payload["events"][0]["status"] == "error"


def test_history_omits_the_trace_unless_full_is_given() -> None:
    brief = json.loads(runner.invoke(app, ["history", "--all-projects"]).stdout)
    assert "trace" not in brief["events"][0]
    full = json.loads(runner.invoke(app, ["history", "--all-projects", "--full"]).stdout)
    assert full["events"][0]["trace"][0]["k"] == "provision.attempt"


def test_history_filters_by_job_and_failures() -> None:
    out = json.loads(runner.invoke(app, ["history", "--all-projects", "--job", "j-1"]).stdout)
    assert len(out["events"]) == 1
    empty = json.loads(runner.invoke(app, ["history", "--all-projects", "--job", "j-2"]).stdout)
    assert empty["events"] == []


def test_history_stats_emits_the_aggregate_view() -> None:
    out = json.loads(runner.invoke(app, ["history", "--all-projects", "--stats"]).stdout)
    assert out["failures"] == 1
    assert out["signatures"][0]["count"] == 1
    assert out["usd_burned"] == 0.29


def test_report_emits_markdown_on_stdout() -> None:
    result = runner.invoke(app, ["report", "--all-projects"])
    assert result.exit_code == 0
    assert result.stdout.startswith("# Lab event report")


def test_report_out_writes_a_file(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    result = runner.invoke(app, ["report", "--all-projects", "--out", str(target)])
    assert result.exit_code == 0
    assert target.read_text().startswith("# Lab event report")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_history_cli.py -v`
Expected: FAIL — `No such command 'history'`

- [ ] **Step 3: Add both commands to `src/lab/cli.py`**

Add near the other read-only commands (after `logs`):

```python
@app.command()
def history(
    limit: int = typer.Option(50, "--limit", "-n", help="most recent N calls"),
    since: str | None = typer.Option(None, "--since", help="window, e.g. 2d / 30m"),
    action: str | None = typer.Option(None, "--action", help="filter to one command/tool"),
    job: str | None = typer.Option(None, "--job", help="calls that touched this job id"),
    session: str | None = typer.Option(None, "--session", help="filter to one session id"),
    failures: bool = typer.Option(False, "--failures", help="only calls that did not succeed"),
    all_projects: bool = typer.Option(False, "--all-projects", help="across every project"),
    full: bool = typer.Option(False, "--full", help="include params and the failure trace"),
    stats: bool = typer.Option(False, "--stats", help="aggregate view instead of rows"),
) -> None:
    """Read the lab's own event ledger — what was run, what it did, why it failed.

    This is *not* `lab logs`, which tails one job's stdout.
    """
    project = None if all_projects else repo_root().name
    found = events.read(
        since=since, project=project, action=action, session=session, job=job,
        failures_only=failures, limit=None if stats else limit,
    )
    if stats:
        _emit(events.stats_dict(events.stats(found)))
        return
    _emit({"events": [events.row(e, full=full) for e in found]})


@app.command()
def report(
    since: str = typer.Option("7d", "--since", help="window, e.g. 7d"),
    all_projects: bool = typer.Option(False, "--all-projects", help="across every project"),
    out: str | None = typer.Option(None, "--out", help="write to this file instead of stdout"),
) -> None:
    """A pasteable markdown digest of what failed and what it cost (field-report shaped)."""
    project = None if all_projects else repo_root().name
    text = events.report(events.read(since=since, project=project))
    if out:
        Path(out).write_text(text)
        _emit({"written": out})
        return
    typer.echo(text)
```

Ensure `from pathlib import Path` and `from lab.manifest import repo_root` are already imported in `cli.py` (both are).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_events_history_cli.py -v && uv run mypy --strict src/lab`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lab/cli.py tests/test_events_history_cli.py
git commit -m "feat(cli): lab history and lab report over the event ledger"
```

---

### Task 11: MCP `history` and `report` tools

**Files:**
- Modify: `src/lab/mcp_server.py`
- Test: `tests/test_events_mcp.py` (extend)

**Interfaces:**
- Consumes: `events.read`, `events.row`, `events.stats`, `events.stats_dict`, `events.report_dict`. The row builder is shared through `lab.events`, so neither shell imports the other.
- Produces: MCP tools `history`, `report`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_events_mcp.py`:

```python
@pytest.mark.anyio
async def test_history_tool_returns_recent_calls(tmp_path: Path) -> None:
    server = build_server(default_lab(home=tmp_path / "runs"))
    async with Client(server) as client:
        await client.call_tool("list", {})
        result = await client.call_tool("history", {"limit": 10, "all_projects": True})
    actions = [e["action"] for e in result.data["events"]]
    assert "list" in actions


@pytest.mark.anyio
async def test_report_tool_returns_markdown(tmp_path: Path) -> None:
    server = build_server(default_lab(home=tmp_path / "runs"))
    async with Client(server) as client:
        result = await client.call_tool("report", {"since": "7d", "all_projects": True})
    assert result.data["markdown"].startswith("# Lab event report")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_mcp.py -v`
Expected: FAIL — unknown tool `history`.

- [ ] **Step 3: Add the tools inside `build_server`**

```python
    @mcp.tool
    def history(
        limit: int = 50,
        since: str | None = None,
        action: str | None = None,
        job: str | None = None,
        failures: bool = False,
        all_projects: bool = False,
        full: bool = False,
        stats: bool = False,
    ) -> dict[str, Any]:
        """Read the lab's own event ledger: which commands/tools ran, their outcome, duration,
        ids, cost and — with `full` — the internal trace behind a failure. This is what you
        already tried; `logs` is one job's stdout. `stats` returns the aggregate view instead
        (failure rates per action, ranked error signatures, dollars burned)."""
        project = None if all_projects else repo_root().name
        found = events.read(
            since=since, project=project, action=action, job=job,
            failures_only=failures, limit=None if stats else limit,
        )
        if stats:
            return events.stats_dict(events.stats(found))
        return {"events": [events.row(e, full=full) for e in found]}

    @mcp.tool
    def report(since: str = "7d", all_projects: bool = False) -> dict[str, Any]:
        """A markdown digest of what failed in the window and what it cost — triage table plus
        per-finding attempted/observed/cost. Paste into an issue or hand to a developer."""
        project = None if all_projects else repo_root().name
        return events.report_dict(events.read(since=since, project=project))
```

Add `from lab.manifest import repo_root` to `mcp_server.py`'s imports if not present.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_events_mcp.py -v && uv run mypy --strict src/lab`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lab/mcp_server.py tests/test_events_mcp.py
git commit -m "feat(mcp): history and report tools over the event ledger"
```

---

### Task 12: `note()` call sites in the internals

Each is **additive** to the existing stderr print: the printed line is the live UX, the note is the durable record. Nothing is removed.

**Files:**
- Modify: `src/lab/placement.py`, `src/lab/doctor.py`, `src/lab/core.py`, `src/lab/storage.py`, `src/lab/backends/skypilot.py`, `src/lab/scheduler/tick.py`
- Test: `tests/test_events_notes.py`

**Interfaces:**
- Consumes: `events.note`.
- Produces: no new API. Note kinds, which the report renders verbatim — keep them stable:
  `placement.zone_skipped`, `placement.zone_exhausted`, `placement.priced`, `placement.disk_override`,
  `doctor.check`, `provision.attempt`, `provision.timeout`, `launch.retry`,
  `teardown.attempt`, `teardown.retry`, `teardown.fallback`, `vast.balance_failed`,
  `core.dirty_snapshot`, `core.cache_hit`, `core.config_rejected`, `core.submit_stagger`,
  `storage.upload_failed`, `scheduler.trigger`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_events_notes.py`:

```python
"""The notes that make a failure explicable. Each asserts the note fires at the site that
already prints the same thing to stderr — the print is the live UX, the note is the record."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab import events, placement
from lab.events import store


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


def _trace_kinds() -> list[str]:
    close = [r for r in store.iter_records(store.day_files()) if r["phase"] == "close"][-1]
    return [n["k"] for n in close.get("trace", [])]


def test_placement_warn_also_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError):
        with events.record("cli", "submit", {}):
            placement._warn("zone europe-west1-b exhausted")
            raise RuntimeError("boom")
    assert "placement.warn" in _trace_kinds()


def test_a_note_never_escapes_into_a_successful_record() -> None:
    with events.record("cli", "submit", {}):
        placement._warn("harmless")
    close = [r for r in store.iter_records(store.day_files()) if r["phase"] == "close"][-1]
    assert "trace" not in close
```

Extend this file with one test per site as you add them, following the same shape: open a
`record()`, drive the site, raise, assert the kind appears in the trace.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_events_notes.py -v`
Expected: FAIL — `placement.warn` not in the trace.

- [ ] **Step 3: Route `placement._warn` through `note()`**

In `src/lab/placement.py:43-49`:

```python
def _warn(message: str) -> None:
    """Diagnostics go to **stderr**, never stdout — callers parse stdout as JSON.

    The same line is buffered into the event ledger, where it survives the terminal scrolling.
    """
    from lab import events

    print(message, file=sys.stderr)
    events.note("placement.warn", message=message)
```

The import is local to keep `lab.placement`'s import graph unchanged (it is imported by
`build_task`, which runs in the scheduler).

- [ ] **Step 4: Add the remaining notes**

Add each of these next to the code that already logs or decides. Every one is a single added line.

| File | Site | Call |
|---|---|---|
| `placement.py` | exhausted-zone memo read | `events.note("placement.zone_skipped", zone=zone, reason="exhausted_memo")` |
| `placement.py` | exhausted-zone memo write | `events.note("placement.zone_exhausted", zone=zone)` |
| `placement.py` | resolved price | `events.note("placement.priced", instance=instance, region=region, hourly_usd=price)` |
| `placement.py` | `effective_disk_gb` override | `events.note("placement.disk_override", requested=requested, applied=applied)` |
| `doctor.py` | each check result | `events.note("doctor.check", name=name, status=status, detail=detail)` |
| `backends/skypilot.py` | before `sky.launch` | `events.note("provision.attempt", cloud=cloud, zone=zone, instance=instance)` |
| `backends/skypilot.py` | provision watchdog fires | `events.note("provision.timeout", cloud=cloud, after_s=elapsed)` |
| `backends/skypilot.py` | launch retry (`LAB_LAUNCH_RETRIES`) | `events.note("launch.retry", attempt=attempt, backoff_s=delay, error=str(e))` |
| `backends/skypilot.py` | `robust_teardown` each attempt | `events.note("teardown.attempt", cluster=name, attempt=attempt)` |
| `backends/skypilot.py` | teardown retry | `events.note("teardown.retry", cluster=name, attempt=attempt, error=str(e))` |
| `backends/skypilot.py` | vastai/gcp-direct fallback | `events.note("teardown.fallback", cluster=name, via=via, ok=ok)` |
| `backends/skypilot.py:680` | vast balance lookup failure | `events.note("vast.balance_failed", error=str(e))` |
| `core.py` | dirty-diff snapshot | `events.note("core.dirty_snapshot", diff_ref=key, bytes=size)` |
| `core.py` | cache hit (`--cache`) | `events.note("core.cache_hit", job_id=hit)` |
| `core.py` | unknown-config rejection | `events.note("core.config_rejected", unknown=sorted(unknown))` |
| `core.py` | sweep submit stagger | `events.note("core.submit_stagger", seconds=stagger, shard=shard_id)` |
| `storage.py` | R2 upload/download failure | `events.note("storage.upload_failed", key=key, error=str(e))` |
| `scheduler/tick.py` | per-registration trigger verdict | `events.note("scheduler.trigger", reg_id=reg_id, fired=fired, reason=reason)` |

Each file needs `from lab import events` — use a function-local import in `placement.py` and
`scheduler/tick.py` (both are imported on hot paths), a module-level import elsewhere.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -x && uv run mypy --strict src/lab && uv run ruff check src/lab`
Expected: PASS. Notes are additive, so no existing assertion about stderr output may change — if one did, a print was replaced instead of augmented.

- [ ] **Step 6: Commit**

```bash
git add src/lab tests/test_events_notes.py
git commit -m "feat(events): note the internal steps that explain a failure"
```

---

### Task 13: Concurrency and packaging guarantees

**Files:**
- Create: `tests/test_events_concurrency.py`
- Modify: `tests/test_packaging.py`

**Interfaces:**
- Consumes: `store.append`, `store.iter_records`.
- Produces: no new API.

- [ ] **Step 1: Write the failing concurrency test**

Create `tests/test_events_concurrency.py`:

```python
"""A sharded sweep launches many `lab` processes against one ledger file. Torn or interleaved
lines would make the store untrustworthy exactly when it matters most."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WRITER = """
import os, sys
from datetime import datetime, timezone
from lab.events import store
when = datetime(2026, 8, 18, tzinfo=timezone.utc)
tag = sys.argv[1]
for i in range(200):
    store.append({"id": f"{tag}-{i}", "phase": "close", "pad": "x" * 3000}, when=when)
"""


def test_concurrent_writers_produce_only_whole_lines(tmp_path: Path) -> None:
    env = {**dict(__import__("os").environ), "LAB_EVENTS_DIR": str(tmp_path / "events")}
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, f"w{n}"], env=env) for n in range(8)]
    for p in procs:
        assert p.wait() == 0
    path = tmp_path / "events" / "2026-08-18.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 8 * 200
    for line in lines:
        json.loads(line)  # every line whole and parseable
```

- [ ] **Step 2: Run it to verify it passes with the lock and fails without**

Run: `uv run pytest tests/test_events_concurrency.py -v`
Expected: PASS. Then temporarily comment out the two `fcntl.flock` calls in `store.append` and
re-run — with 3 KB records it must fail. Restore the lock. This confirms the test has teeth
rather than passing because the writes were small enough to be atomic anyway.

- [ ] **Step 3: Extend the packaging test**

In `tests/test_packaging.py`, inside `test_installed_wheel_scaffolds_and_runs_a_job`, set
`LAB_EVENTS_DIR` in the subprocess environment to `tmp_path / "events"` and add after the job
runs:

```python
    ledger = sorted((tmp_path / "events").glob("*.jsonl"))
    assert ledger, "the installed wheel must write its ledger with no checkout on sys.path"
    records = [json.loads(line) for p in ledger for line in p.read_text().splitlines()]
    assert any(r.get("phase") == "open" and r.get("action") == "submit" for r in records)
```

This is the guard that `lab.events` took no dependency outside base deps — the same class of
failure `pytest -m packaging` exists to catch.

- [ ] **Step 4: Run the packaging test**

Run: `uv run pytest -m packaging -v`
Expected: PASS (slow — it builds a wheel and installs it into a clean venv).

- [ ] **Step 5: Commit**

```bash
git add tests/test_events_concurrency.py tests/test_packaging.py
git commit -m "test(events): concurrent-writer integrity and installed-wheel ledger"
```

---

### Task 14: Documentation

**Files:**
- Create: `docs/guides/event-logging.md`
- Modify: `CLAUDE.md`, `CHANGELOG.md`, `docs/COMPATIBILITY.md`

**Interfaces:**
- Consumes: everything shipped in Tasks 1-13.
- Produces: no code.

- [ ] **Step 1: Write `docs/guides/event-logging.md`**

Cover, in this order, with a runnable example under each heading:

1. *What is recorded* — the open/close pair, the field table from the spec, and the fact that
   `trace` appears only on failure.
2. *Where it lives* — `~/.lab/events/YYYY-MM-DD.jsonl`, user-global and project-tagged; why not
   project-local (the lab installs into other projects).
3. *Reading it* — the four views, with real invocations:
   `lab history`, `lab history --job j-4f2a --full`, `lab history --stats --since 30d`,
   `lab report --since 7d --out report.md`.
4. *`lab history` is not `lab logs`* — one is the tool's ledger, the other is a job's stdout.
5. *Retention* — 14-day success TTL, 90-day age cap, 50 MB byte cap, all env-tunable; the byte
   cap is a runaway alarm, not a routine constraint.
6. *Secrets* — the sanitizer's rules, and that env dicts and file contents are never recorded.
7. *Turning it off* — `LAB_EVENTS=0`; and `LAB_EVENTS_DEBUG=1` when the ledger itself misbehaves.
8. *Setting a session id* — `LAB_SESSION_ID` for exact grouping from an agent harness.

- [ ] **Step 2: Add the `CLAUDE.md` key fact**

Under **Key facts**, after the agent-UX bullet:

```markdown
- **Event ledger (`lab.events`):** every CLI/MCP call writes an open/close pair to
  `~/.lab/events/YYYY-MM-DD.jsonl` (user-global, project-tagged; `LAB_EVENTS_DIR` overrides,
  `LAB_EVENTS=0` disables). Internals call `events.note(...)`, buffered in memory and flushed
  into the record **only when the call fails** — successes stay tiny, failures carry the trace.
  Read it with **`lab history`** (`--job/--since/--failures/--full/--stats`) and **`lab report`**
  (markdown digest); `lab logs` is unrelated — that tails one job's stdout. Retention: successes
  compacted after 14d, files deleted past 90d or 50 MB. `params` passes the `lab.events.sanitize`
  allow-list, never raw (FR-J1). The CLI entry point is `lab.cli:main`, which runs typer with
  `standalone_mode=False` so usage errors and crashes are distinguishable — exit codes are
  unchanged and `lab wait`'s 3/4 remain contract.
  Guide: `docs/guides/event-logging.md`.
```

- [ ] **Step 3: Add the CHANGELOG entry**

Under the open (unreleased) section:

```markdown
### Added
- **Event ledger.** Every CLI and MCP call is recorded to `~/.lab/events/`, with the internal
  trace attached when the call fails. New `lab history` (session, forensic and aggregate views)
  and `lab report` (markdown digest). Guide: `docs/guides/event-logging.md`.

### Changed
- The `lab` console entry point moved from `lab.cli:app` to `lab.cli:main`. Exit codes and output
  are unchanged; the wrapper exists to record usage errors and crashes.
```

- [ ] **Step 4: Note the entry-point move in `docs/COMPATIBILITY.md`**

Record that the console script target changed, that it is invisible to anyone invoking `lab` on
the command line, and that it matters only to code importing `lab.cli:app` directly.

- [ ] **Step 5: Verify the docs match the code**

Run every command quoted in the guide and confirm the output shape matches what the guide claims:

```bash
uv run lab history --limit 5
uv run lab history --stats --since 30d
uv run lab report --since 7d
```

- [ ] **Step 6: Full verification and commit**

Run: `uv run pytest && uv run mypy --strict src/lab && uv run ruff check src/lab`
Expected: all PASS.

```bash
git add docs CLAUDE.md CHANGELOG.md
git commit -m "docs(events): guide, key fact, changelog and compatibility note"
```

---

## Self-Review Notes

**Spec coverage.** Event record → Task 1, 4. Capture points (CLI, MCP, scheduler-via-CLI,
`note()`) → Tasks 5, 6, 12; the scheduler is covered for free because its systemd timer runs
`lab scheduler tick` through the CLI. Session identity → Task 4. Storage, concurrency,
best-effort → Task 2. Redaction → Task 1. Retention → Task 3. The four read views → Tasks 7-11.
Module boundaries → the File Structure table. Testing → each task's own tests plus Task 13.
Documentation → Task 14.

**Two deviations from the spec, both deliberate:**

1. The spec sketched `lab/events.py` as a single module of ~250 lines. The plan starts as a
   package, because the surface (sanitize, store, record, annotate, read, stats, report) plainly
   exceeds that and splitting later would churn every import.
2. The spec's `Call` API is `ref()` / `result()`; the plan adds `begin`/`finish_current` as public
   surface, because the CLI must open the call in the group callback and close it in `main()` —
   a single context manager cannot span two functions. `record()` remains the context manager and
   is implemented in terms of them.

**Riskiest step: `standalone_mode=False` in Task 5.** It changes who prints usage errors and who
converts exceptions into exit codes. Task 5 Step 7 runs the pre-existing CLI suites as the
regression gate for exactly this; treat any exit-code change there as a bug in the wrapper.
