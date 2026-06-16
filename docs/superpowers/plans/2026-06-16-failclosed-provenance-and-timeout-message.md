# Fail-closed Provenance + Reliable-Timeout Residual — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the lab fail-closed on provenance — every job records a non-null git SHA and, when the tree is dirty, a resolvable `diff_ref` — and surface the wall value in a timed-out job's manifest.

**Architecture:** A pure `assert_fail_closed` guard on `CodeRef`, enforced at the store's *write* path (legacy manifests still read). A `capture_diff`/`apply_diff` helper pair snapshots dirty state (tracked patch + untracked files) into a small tarball; `Lab.submit` writes it into the job dir, mirrors it to R2 when enabled, and sets `diff_ref`. The deferred/register path sets `diff_ref` to its bundle key. Both runners' timeout `end_reason` gains the wall value.

**Tech Stack:** Python 3.12, Pydantic v2, Typer (CLI), FastMCP, pytest. Conventions: `ruff` (line length 100), `mypy --strict` on `src/lab`.

**Spec:** `docs/superpowers/specs/2026-06-16-failclosed-provenance-and-timeout-message-design.md`

---

## File Structure

- `src/lab/models.py` — add `CodeRef.assert_fail_closed()` (the invariant, pure).
- `src/lab/store.py` — `write_manifest` calls `assert_fail_closed` (write-path guard).
- `src/lab/manifest.py` — add `capture_diff()` and `apply_diff()` (dependency-free git helpers).
- `src/lab/core.py` — `Lab.submit` captures the diff + sets `diff_ref` (+ R2 mirror) when dirty.
- `src/lab/cli.py` — `submit --no-dirty`.
- `src/lab/mcp_server.py` — `submit` `allow_dirty` arg.
- `src/lab/scheduler/register.py` — set `CodeRef.diff_ref` to the bundle key when dirty.
- `src/lab/runner.py` + `src/lab/sky_runner.py` — richer `timed_out` `end_reason`.
- `tests/` — one focused test module per behavior.

---

## Task 1: `CodeRef.assert_fail_closed` invariant

**Files:**
- Modify: `src/lab/models.py:39-42` (the `CodeRef` class)
- Test: `tests/test_provenance_invariant.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provenance_invariant.py
import pytest

from lab.models import CodeRef


def test_clean_coderef_passes():
    CodeRef(git_commit="a" * 40, git_dirty=False).assert_fail_closed()


def test_dirty_with_ref_passes():
    CodeRef(git_commit="a" * 40, git_dirty=True, diff_ref="r2://b/x").assert_fail_closed()


def test_empty_commit_rejected():
    with pytest.raises(ValueError, match="git_commit"):
        CodeRef(git_commit="", git_dirty=False).assert_fail_closed()


def test_dirty_without_ref_rejected():
    with pytest.raises(ValueError, match="diff_ref"):
        CodeRef(git_commit="a" * 40, git_dirty=True, diff_ref=None).assert_fail_closed()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provenance_invariant.py -v`
Expected: FAIL with `AttributeError: 'CodeRef' object has no attribute 'assert_fail_closed'`

- [ ] **Step 3: Add the method**

In `src/lab/models.py`, add to the `CodeRef` class (after the fields):

```python
class CodeRef(BaseModel):
    git_commit: str
    git_dirty: bool = False
    diff_ref: str | None = None  # blob ref of the snapshotted diff if dirty (FR-B1)

    def assert_fail_closed(self) -> None:
        """The fail-closed provenance invariant (FR-B1): a job's code state must always be
        reconstructable from its manifest. Enforced at the store write path, NOT on load, so
        legacy Gap-B manifests still read."""
        if not self.git_commit:
            raise ValueError("CodeRef.git_commit must be a non-null commit SHA (FR-B1)")
        if self.git_dirty and self.diff_ref is None:
            raise ValueError(
                "CodeRef is dirty but diff_ref is None — a dirty run must capture its diff so "
                "the exact code state is reconstructable (FR-B1, Gap B)"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provenance_invariant.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lab/models.py tests/test_provenance_invariant.py
git commit -m "feat(provenance): CodeRef.assert_fail_closed invariant (FR-B1)"
```

---

## Task 2: Enforce the invariant at the store write path

**Files:**
- Modify: `src/lab/store.py:47-48` (`write_manifest`)
- Test: `tests/test_provenance_invariant.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_provenance_invariant.py`:

```python
from pathlib import Path

from helpers import make_manifest

from lab.store import JobStore


def test_write_manifest_rejects_gapb(tmp_path: Path):
    store = JobStore(tmp_path)
    m = make_manifest("g1", "echo hi")
    m.code.git_dirty = True  # dirty but diff_ref is None -> Gap B
    with pytest.raises(ValueError, match="diff_ref"):
        store.write_manifest(m)


def test_read_manifest_tolerates_legacy_gapb(tmp_path: Path):
    # A legacy Gap-B manifest already on disk must still LOAD (no migration break).
    store = JobStore(tmp_path)
    m = make_manifest("g2", "echo hi")
    (tmp_path / "g2").mkdir()
    m.code.git_dirty = True
    (tmp_path / "g2" / "manifest.json").write_text(m.model_dump_json(indent=2))
    loaded = store.read_manifest("g2")  # must not raise
    assert loaded.code.git_dirty is True and loaded.code.diff_ref is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provenance_invariant.py -k gapb -v`
Expected: `test_write_manifest_rejects_gapb` FAILS (no exception raised); `test_read_manifest_tolerates_legacy_gapb` PASSES already.

- [ ] **Step 3: Add the write-path guard**

In `src/lab/store.py`, change `write_manifest`:

```python
    def write_manifest(self, manifest: JobManifest) -> None:
        manifest.code.assert_fail_closed()  # fail-closed on write; reads stay tolerant (FR-B1)
        self._atomic_write(self.manifest_path(manifest.job_id), manifest.model_dump_json(indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provenance_invariant.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Run the full suite to catch any test that wrote a dirty manifest**

Run: `uv run pytest -q`
Expected: PASS. (If a pre-existing test constructs a dirty `CodeRef` without `diff_ref` and writes it, fix that test to set `diff_ref="test"`; `make_manifest` is clean so this is unlikely.)

- [ ] **Step 6: Commit**

```bash
git add src/lab/store.py tests/test_provenance_invariant.py
git commit -m "feat(provenance): enforce fail-closed invariant on manifest write"
```

---

## Task 3: `capture_diff` / `apply_diff` helpers

**Files:**
- Modify: `src/lab/manifest.py` (add two functions + imports)
- Test: `tests/test_capture_diff.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_capture_diff.py
import subprocess
from pathlib import Path

from lab.manifest import apply_diff, capture_diff, current_commit


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "tracked.txt").write_text("original\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")


def test_clean_tree_returns_none(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert capture_diff(repo, tmp_path / "dest") is None


def test_roundtrip_restores_dirty_state(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    commit = current_commit(repo)
    # Make the tree dirty: edit a tracked file, add an untracked file.
    (repo / "tracked.txt").write_text("CHANGED\n")
    (repo / "new_script.py").write_text("print('hi')\n")

    blob = capture_diff(repo, tmp_path / "dest")
    assert blob is not None and Path(blob).exists()

    # Reconstruct: fresh checkout of the commit + apply the captured diff.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    _git(repo, "worktree", "add", "-q", "--detach", str(fresh), commit)
    apply_diff(Path(blob), fresh)

    assert (fresh / "tracked.txt").read_text() == "CHANGED\n"
    assert (fresh / "new_script.py").read_text() == "print('hi')\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_capture_diff.py -v`
Expected: FAIL with `ImportError: cannot import name 'capture_diff'`

- [ ] **Step 3: Implement the helpers**

In `src/lab/manifest.py`, add imports at the top (after the existing ones):

```python
import shutil
import tarfile
import tempfile
```

Then add at the end of the file:

```python
def capture_diff(repo: Path, dest_dir: Path) -> str | None:
    """Snapshot uncommitted state into ``dest_dir/code_diff.tar.gz``; return its path, or None
    when the tree is clean (FR-B1). The tarball holds ``tracked.patch`` (``git diff HEAD --binary``)
    and ``untracked/<rel>`` for untracked, non-ignored files. The committed tree is NOT archived —
    the pinned commit already captures it; ``apply_diff`` restores onto a checkout of that commit."""
    repo = Path(repo)
    if not is_dirty(repo):
        return None
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "code_diff.tar.gz"
    patch = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--binary"])
    untracked = (
        subprocess.check_output(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"]
        )
        .decode()
        .split("\0")
    )
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "stage"
        (stage / "untracked").mkdir(parents=True)
        (stage / "tracked.patch").write_bytes(patch)
        for rel in filter(None, untracked):
            dst = stage / "untracked" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / rel, dst)
        with tarfile.open(tar_path, "w:gz") as out:
            out.add(stage, arcname=".")
    return str(tar_path)


def apply_diff(tarball: Path, tree: Path) -> None:
    """Restore captured dirty state (from :func:`capture_diff`) onto a checkout at ``tree``:
    apply ``tracked.patch`` then drop the ``untracked/`` files into place."""
    tree = Path(tree)
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        with tarfile.open(tarball) as t:
            t.extractall(stage, filter="data")
        patch = stage / "tracked.patch"
        if patch.read_bytes().strip():
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn"],
                input=patch.read_bytes(), cwd=tree, check=True,
            )
        untracked_root = stage / "untracked"
        for src in untracked_root.rglob("*"):
            if src.is_file():
                dst = tree / src.relative_to(untracked_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_capture_diff.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lab/manifest.py tests/test_capture_diff.py
git commit -m "feat(provenance): capture_diff/apply_diff for dirty-tree snapshots (FR-B1)"
```

---

## Task 4: Wire diff capture into `Lab.submit` (+ R2 mirror) and CLI/MCP flag

**Files:**
- Modify: `src/lab/core.py:196-243` (`Lab.submit`)
- Modify: `src/lab/cli.py:93-132` (`submit` command — add `--no-dirty`)
- Modify: `src/lab/mcp_server.py:48-82` (`submit` tool — add `allow_dirty`)
- Test: `tests/test_submit_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submit_provenance.py
import subprocess
from pathlib import Path

import pytest

from lab.core import Lab, LabError
from lab.backends.local import LocalBackend
from lab.models import JobSpec


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo_with_lockfile(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "uv.lock").write_text("lock\n")
    (repo / "tracked.txt").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _lab(repo: Path) -> Lab:
    home = repo / "runs"
    return Lab(backend=LocalBackend(home=home, repo=repo), repo=repo, home=home)


def test_dirty_submit_captures_diff_ref(tmp_path: Path, monkeypatch):
    repo = _repo_with_lockfile(tmp_path)
    (repo / "tracked.txt").write_text("DIRTY\n")  # make the tree dirty
    monkeypatch.chdir(repo)
    lab = _lab(repo)
    job_id = lab.submit(JobSpec(command="true"))
    m = lab.manifest(job_id)
    assert m.code.git_dirty is True
    assert m.code.diff_ref is not None
    assert Path(m.code.diff_ref).exists()  # local path resolves (no R2 in this test)


def test_clean_submit_has_no_diff_ref(tmp_path: Path, monkeypatch):
    repo = _repo_with_lockfile(tmp_path)
    monkeypatch.chdir(repo)
    lab = _lab(repo)
    m = lab.manifest(lab.submit(JobSpec(command="true")))
    assert m.code.git_dirty is False and m.code.diff_ref is None


def test_no_dirty_refuses(tmp_path: Path, monkeypatch):
    repo = _repo_with_lockfile(tmp_path)
    (repo / "tracked.txt").write_text("DIRTY\n")
    monkeypatch.chdir(repo)
    lab = _lab(repo)
    with pytest.raises(LabError, match="dirty"):
        lab.submit(JobSpec(command="true"), allow_dirty=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_submit_provenance.py -v`
Expected: `test_dirty_submit_captures_diff_ref` FAILS (`diff_ref is None`).

- [ ] **Step 3: Implement diff capture in `Lab.submit`**

In `src/lab/core.py`, add the import near the other manifest imports (line 26):

```python
from lab.manifest import (
    capture_diff,
    commit_exists,
    current_commit,
    is_dirty,
    repo_root,
    uv_lock_sha256,
)
```

Replace the `code is None` block and the `job_id`/manifest assembly (currently lines 211-241) so the job dir exists before the diff is captured:

```python
        job_id = _new_job_id()
        if code is None:
            dirty = is_dirty(self.repo)
            if dirty and not allow_dirty:
                raise LabError("working tree is dirty; commit or pass allow_dirty=True (FR-B1)")
            diff_ref: str | None = None
            if dirty:
                # Capture into the job dir, then mirror to R2 (if enabled) for durability — the
                # local runs/ dir is git-ignored and may be lost. diff_ref points at the durable
                # copy when one exists, else the local path.
                self.store.job_dir(job_id).mkdir(parents=True, exist_ok=True)
                blob = capture_diff(self.repo, self.store.job_dir(job_id))
                diff_ref = blob
                if blob is not None and r2_enabled():
                    r2 = R2Store.from_env()
                    if r2 is not None:
                        key = f"{job_id}/code_diff.tar.gz"
                        r2.upload_file(Path(blob), key)
                        diff_ref = r2.uri(job_id) + "/code_diff.tar.gz"
            code = CodeRef(
                git_commit=current_commit(self.repo), git_dirty=dirty, diff_ref=diff_ref
            )
        elif code.git_dirty and not allow_dirty:
            raise LabError("bundle captured a dirty tree but allow_dirty=False (FR-B1)")
        seed = spec.seed if spec.seed is not None else 0  # explicit + recorded (FR-B4)
        manifest = JobManifest(
            job_id=job_id,
            sweep_id=sweep_id,
```

(The rest of the `JobManifest(...)` constructor and the `self.store.create(manifest)` / `self.backend.submit(manifest)` / `return job_id` lines are unchanged.)

Confirm `R2Store`, `r2_enabled` are already imported in `core.py` (they are, line 28).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_submit_provenance.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Add the CLI `--no-dirty` flag**

In `src/lab/cli.py`, add a parameter to `submit` (after `no_fallback`, around line 110):

```python
    no_dirty: bool = typer.Option(
        False, "--no-dirty",
        help="refuse to launch from a dirty working tree (default: snapshot the diff, FR-B1)",
    ),
```

Change the `lab.submit` call (line 128) to thread the flag:

```python
        job_id = lab.submit(spec, allow_dirty=not no_dirty)
```

- [ ] **Step 6: Add the MCP `allow_dirty` arg**

In `src/lab/mcp_server.py`, add `allow_dirty: bool = True` to the `submit` signature (after `spot_fallback`, line 62) and thread it:

```python
            job_id = the_lab.submit(spec, allow_dirty=allow_dirty)
```

Append to the tool docstring: ` allow_dirty=False refuses a dirty working tree (default snapshots the diff, FR-B1).`

- [ ] **Step 7: Run the full suite + lint**

Run: `uv run pytest -q && uv run ruff check src/lab && uv run mypy --strict src/lab`
Expected: PASS / no errors.

- [ ] **Step 8: Commit**

```bash
git add src/lab/core.py src/lab/cli.py src/lab/mcp_server.py tests/test_submit_provenance.py
git commit -m "feat(provenance): submit captures diff_ref on dirty tree; --no-dirty to refuse"
```

---

## Task 5: Populate `diff_ref` on the deferred/register path

**Files:**
- Modify: `src/lab/scheduler/register.py:88-99` (`register`)
- Test: `tests/test_register_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_register_provenance.py
import subprocess
from pathlib import Path

from lab.scheduler.queue import QueueStore
from lab.scheduler.register import register
from lab.scheduler.models import Guardrails, Triggers
from lab.models import JobSpec


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_dirty_registration_sets_diff_ref(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    (repo / "a.txt").write_text("DIRTY\n")  # dirty tree

    queue = QueueStore(tmp_path / "queue")
    reg = register(repo, queue, JobSpec(command="true"), Triggers(), Guardrails())
    assert reg.code.git_dirty is True
    assert reg.code.diff_ref == reg.bundle_key  # the bundle IS the captured dirty state
    reg.code.assert_fail_closed()  # the eventual submit must pass the write-path guard
```

(If `Triggers()`/`Guardrails()` require fields, construct them as the existing
`tests/` scheduler tests do — grep `register(` in `tests/` for the exact call shape.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_register_provenance.py -v`
Expected: FAIL (`reg.code.diff_ref` is `None`, not `reg.bundle_key`).

- [ ] **Step 3: Set `diff_ref` to the bundle key when dirty**

In `src/lab/scheduler/register.py`, in `register`, after the bundle is created (line 90):

```python
    with tempfile.TemporaryDirectory() as td:
        bundle_key, code = _snapshot_bundle(repo, Path(td), reg_id, queue)
    if code.git_dirty:
        # The bundle tarball IS this run's captured dirty state — point diff_ref at it so the
        # deferred submit satisfies the fail-closed invariant (FR-B1).
        code = code.model_copy(update={"diff_ref": bundle_key})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_register_provenance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lab/scheduler/register.py tests/test_register_provenance.py
git commit -m "feat(provenance): deferred registration sets diff_ref to its bundle key (FR-B1)"
```

---

## Task 6: Richer `timed_out` `end_reason` (P0-1 residual)

**Files:**
- Modify: `src/lab/runner.py:82-87` (local runner)
- Modify: `src/lab/sky_runner.py:336-345` (remote finalize)
- Test: `tests/test_runner.py:39-46` (update existing assertion) + `tests/test_timeout_message.py` (new, remote)

- [ ] **Step 1: Update the local-runner timeout message + its test**

In `src/lab/runner.py`, change the timeout branch (line 82-83):

```python
    if timed_out:
        wall = int(timeout) if timeout else 0
        status, reason = JobState.timed_out, f"timed out after {wall}s wall-clock cap"
```

In `tests/test_runner.py`, update `test_timeout_terminates`'s assertion:

```python
    assert m.status == JobState.timed_out
    assert m.end_reason == "timed out after 1s wall-clock cap"
```

- [ ] **Step 2: Run the local-runner test**

Run: `uv run pytest tests/test_runner.py::test_timeout_terminates -v`
Expected: PASS

- [ ] **Step 3: Write the failing remote-finalize test**

```python
# tests/test_timeout_message.py
from pathlib import Path

import lab.sky_runner as sky_runner
from helpers import make_manifest
from lab.backends.skypilot import TIMEOUT_SENTINEL
from lab.models import JobState
from lab.store import JobStore


class _Status:
    def __init__(self, name): self.name = name


class _FakeSky:
    """Minimal sky stand-in: launch UP immediately, queue reports FAILED (timeout path),
    teardown succeeds."""
    def launch(self, *a, **k): return "req"
    def get(self, x): return x
    def stream_and_get(self, req): return (1, object())
    def tail_logs(self, *a, **k): return None
    def queue(self, cluster, skip_finished=False):
        return [{"job_id": 1, "status": _Status("FAILED")}]
    def status(self, cluster_names=None): return []
    def down(self, cluster): return "ok"
    def cancel(self, *a, **k): return "ok"


def test_remote_timeout_reason_carries_wall(tmp_path: Path, monkeypatch):
    store = JobStore(tmp_path)
    store.create(make_manifest("t1", "true", timeout="120s", accelerators="RTX_3070:1"))
    # The on-box `timeout` wrapper would have left this sentinel; simulate it.
    (store.output_dir("t1") / TIMEOUT_SENTINEL).touch()

    monkeypatch.setattr(sky_runner, "provision_with_watchdog", lambda *a, **k: (1, object()))
    monkeypatch.setattr(sky_runner, "_resolve_hourly", lambda *a, **k: 0.5)
    monkeypatch.setattr(sky_runner, "_rsync_down", lambda *a, **k: None)
    monkeypatch.setattr(sky_runner, "r2_enabled", lambda: False)
    monkeypatch.setattr(sky_runner, "_cluster_up", lambda *a, **k: False)
    monkeypatch.setattr(sky_runner.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sky_runner, "confirm_no_rental", lambda c: True)
    teardown_calls = {"n": 0}
    real_teardown = sky_runner.tear_down_and_record

    def _spy(sky_mod, cluster, st, jid):
        teardown_calls["n"] += 1
        st.update_manifest(jid, teardown_status="succeeded")
        return True

    monkeypatch.setattr(sky_runner, "tear_down_and_record", _spy)
    monkeypatch.setattr("sky.launch", lambda *a, **k: "req", raising=False)
    # Inject the fake sky module used via `import sky` inside run_job.
    import sys
    sys.modules["sky"] = _FakeSky()  # type: ignore[assignment]

    rc = sky_runner.run_job(store.job_dir("t1"))
    m = store.read_manifest("t1")
    assert m.status == JobState.timed_out
    assert m.end_reason == "timed out after 120s wall-clock cap"
    assert m.teardown_status == "succeeded" and teardown_calls["n"] == 1
    assert rc == 0
```

(If wiring `sys.modules["sky"]` proves brittle against the real `import sky`, follow the
established pattern in `tests/test_runner.py` / `tests/test_skypilot.py` for injecting a fake
sky module — grep those files for how they monkeypatch `sky` — and adapt this test to match.)

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_timeout_message.py -v`
Expected: FAIL (`end_reason` is `"timed_out"`, not the wall message).

- [ ] **Step 5: Implement the richer remote reason**

In `src/lab/sky_runner.py`, just before the finalize `update_manifest` (line 335 `if ...cancelled`), compute the reason:

```python
    if final is JobState.timed_out:
        wall = int(parse_duration(manifest.resources.timeout) or 0)
        end_reason = f"timed out after {wall}s wall-clock cap"
    else:
        end_reason = final.value
```

and use it in the `update_manifest` call (line 337-345), replacing `end_reason=final.value`:

```python
        store.update_manifest(
            job_id,
            status=final,
            ended_at=ended,
            exit_code=0 if final == JobState.succeeded else 1,
            end_reason=end_reason,
            artifacts_uri=artifacts_uri,
            cost=cost,
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_timeout_message.py tests/test_runner.py -v`
Expected: PASS

- [ ] **Step 7: Full suite + lint + types**

Run: `uv run pytest -q && uv run ruff check src/lab && uv run mypy --strict src/lab`
Expected: PASS / no errors.

- [ ] **Step 8: Commit**

```bash
git add src/lab/runner.py src/lab/sky_runner.py tests/test_runner.py tests/test_timeout_message.py
git commit -m "feat(timeout): timed_out end_reason carries the wall value (P0-1 residual)"
```

---

## Task 7: Manual live smoke (optional, costs ~\$0.10 of Vast time)

Not automated (real GPU spend). Run once after Task 6 lands to confirm the real path, mirroring
the scheduler GPU smoke:

- [ ] Submit a sleeper past its cap on the remote backend:
  `uv run lab submit --backend skypilot --accelerators RTX_3070:1 --timeout 60s -c "python -c 'import time; time.sleep(600)'"`
- [ ] After it ends, confirm: `lab status <id>` shows `timed_out` with the wall in `end_reason`,
  `teardown_status` is `succeeded`, and `uv run lab reconcile` reports **no orphans**.

---

## Self-Review

**Spec coverage:**
- A1 capture/apply helper → Task 3. ✓
- A2 wire into `Lab.submit` + R2 mirror → Task 4. ✓
- A3 invariant on write, legacy reads tolerant → Tasks 1 (method) + 2 (write guard + legacy-read test). ✓
- A4 default snapshot / `--no-dirty` / MCP `allow_dirty` → Task 4 steps 5-6. ✓
- A5 confirm unchanged → no task needed (explicitly out of scope; no code touched). ✓
- B1 richer timeout message (both runners) → Task 6. ✓
- B2 offline verification of kill-and-teardown → Task 6 step 3 (asserts state, wall, teardown ran). ✓
- Register path populates diff_ref → Task 5. ✓
- Testing section items → covered across Tasks 1-6.

**Placeholder scan:** No TBD/TODO; every code step shows the code. Two "adapt to the existing fake-sky / register call shape" notes (Task 5 step 1, Task 6 step 3) point at concrete existing patterns to copy rather than leaving logic unspecified — acceptable, as the core assertions and implementation are fully given.

**Type consistency:** `assert_fail_closed()` (Tasks 1, 2, 5) consistent. `capture_diff(repo, dest_dir) -> str | None` and `apply_diff(tarball, tree)` consistent between Task 3 def and Task 4 use. `diff_ref` set as local path or `r2://…/code_diff.tar.gz` (Task 4) / bundle key (Task 5) — both non-None when dirty, satisfying the invariant. `end_reason` string `"timed out after {wall}s wall-clock cap"` identical in local (Task 6 step 1) and remote (step 5) and both tests.
