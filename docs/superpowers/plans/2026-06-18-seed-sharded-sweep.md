# Seed-sharded Sweep with Per-cell Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `lab sweep` declare a seed set and a shard size, split each grid cell's seeds into independently-bounded sub-jobs, and produce one aggregated per-cell result equivalent to a single all-seeds job — with full provenance and honest partial-failure accounting.

**Architecture:** A shard is a normal job (inherits its own manifest, timeout, and teardown). At submit time `Lab.sweep` expands the grid into cells, partitions each cell's seeds into shards, submits each shard with its seed subset appended as a config override, and persists an authoritative `SweepPlan` under the `sweep_id`. Aggregation is a separate, idempotent pull reducer (`Lab.aggregate_sweep`) that row-concatenates the succeeded shards' results files, counts present-vs-expected seeds, and marks each cell complete/incomplete. `Lab.retry_sweep` resubmits only the missing shards.

**Tech Stack:** Python (uv), Pydantic models, Typer CLI, FastMCP server, pytest. Follows the existing `Lab` core library pattern (CLI + MCP are thin shells).

## Global Constraints

- `ruff` line length 100; `mypy --strict` on `src/lab` must pass.
- CLI (`src/lab/cli.py`) and MCP (`src/lab/mcp_server.py`) are **thin shells** over `lab.core.Lab` — never duplicate logic between them.
- Fail-closed provenance (FR-B1) is unchanged and holds **per shard**: each shard is a normal job whose manifest is validated at `JobStore.create`. Do not weaken `CodeRef.assert_fail_closed`.
- Secrets never in repo/manifest/logs; manifests record URIs not keys.
- NumPy `<2` pin; config via Hydra+Pydantic; outputs under `$LAB_RUN_DIR`.
- The single per-job `seed` (`run.seed` → `$LAB_SEED`, singular) is distinct from the **seed-axis config key** (`seeds`, plural) the lab appends to pass a shard's seed subset to the experiment. Never conflate them.
- New defaults (overridable): seed-axis config key = `seeds`; results file = `results.csv`; seed column = `seed`.

---

### Task 1: Seed parsing + partitioning (pure helpers)

**Files:**
- Create: `src/lab/sharding.py`
- Test: `tests/test_sharding.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `parse_seeds(spec: str | list[int]) -> list[int]` — sorted, de-duplicated seed set. Accepts a range string `"0-31"` (inclusive) or an explicit list `[0,1,2]`. Raises `ValueError` on a malformed range or non-int member.
  - `partition_seeds(seeds: list[int], shard_size: int) -> list[list[int]]` — contiguous chunks of at most `shard_size`; complete, non-overlapping cover. `shard_size >= len(seeds)` ⇒ one chunk. Raises `ValueError` if `shard_size < 1`.
  - `seeds_to_arg(seeds: list[int]) -> str` — render a shard's seed subset as the config-override value (a comma-joined list, e.g. `"0,1,2,3"`; injection-safe values only — digits and commas).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sharding.py
from __future__ import annotations

import pytest

from lab.sharding import parse_seeds, partition_seeds, seeds_to_arg


def test_parse_seeds_range_inclusive():
    assert parse_seeds("0-3") == [0, 1, 2, 3]


def test_parse_seeds_list_sorted_deduped():
    assert parse_seeds([3, 1, 1, 2, 0]) == [0, 1, 2, 3]


def test_parse_seeds_rejects_bad_range():
    with pytest.raises(ValueError):
        parse_seeds("3-1")
    with pytest.raises(ValueError):
        parse_seeds("a-b")


def test_partition_contiguous_cover():
    assert partition_seeds([0, 1, 2, 3, 4, 5, 6, 7], 3) == [[0, 1, 2], [3, 4, 5], [6, 7]]


def test_partition_one_shard_when_size_ge_len():
    assert partition_seeds([0, 1, 2, 3], 8) == [[0, 1, 2, 3]]


def test_partition_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        partition_seeds([0, 1], 0)


def test_seeds_to_arg():
    assert seeds_to_arg([0, 1, 2, 3]) == "0,1,2,3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sharding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab.sharding'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lab/sharding.py
"""Pure seed-axis helpers for sharded sweeps (P1-2): parse a declared seed set, partition it into
per-shard subsets, and render a shard's subset as an injection-safe config-override value. No I/O."""

from __future__ import annotations


def parse_seeds(spec: str | list[int]) -> list[int]:
    """Parse a seed declaration into a sorted, de-duplicated seed set.

    Accepts an inclusive range string ``"0-31"`` or an explicit list ``[0, 1, 2]``.
    """
    if isinstance(spec, str):
        if "-" not in spec:
            raise ValueError(f"seed range must be 'lo-hi', got {spec!r}")
        lo_s, _, hi_s = spec.partition("-")
        try:
            lo, hi = int(lo_s), int(hi_s)
        except ValueError as e:
            raise ValueError(f"seed range bounds must be integers, got {spec!r}") from e
        if hi < lo:
            raise ValueError(f"seed range hi < lo: {spec!r}")
        return list(range(lo, hi + 1))
    try:
        return sorted({int(s) for s in spec})
    except (TypeError, ValueError) as e:
        raise ValueError(f"seed list members must be integers, got {spec!r}") from e


def partition_seeds(seeds: list[int], shard_size: int) -> list[list[int]]:
    """Split ``seeds`` into contiguous chunks of at most ``shard_size`` (complete, non-overlapping)."""
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")
    return [seeds[i : i + shard_size] for i in range(0, len(seeds), shard_size)]


def seeds_to_arg(seeds: list[int]) -> str:
    """Render a shard's seed subset as a comma-joined config-override value (digits + commas only)."""
    return ",".join(str(s) for s in seeds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sharding.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lab/sharding.py tests/test_sharding.py
git commit -m "feat(sweep): pure seed parse/partition helpers for sharded sweeps"
```

---

### Task 2: SweepPlan model + store persistence

**Files:**
- Modify: `src/lab/models.py` (append two models after `JobManifest`)
- Modify: `src/lab/store.py` (add plan path + read/write + cell_id helper)
- Test: `tests/test_sweep_plan_store.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `lab.models.SweepCell` — Pydantic model: `coords: dict[str, Any]`, `cell_id: str`, `seeds_expected: list[int]`, `shard_seeds: list[list[int]]`, `shard_job_ids: list[str]`, `results_file: str`, `seed_column: str`, `aggregate_ref: str`, `seeds_present: list[int] = []`, `missing_seeds: list[int] = []`, `status: Literal["pending","complete","incomplete"] = "pending"`.
  - `lab.models.SweepPlan` — `sweep_id: str`, `created_at: datetime`, `command: str`, `seed_axis_key: str`, `cells: list[SweepCell]`. (`command` is the base entrypoint, needed so `retry_sweep` can rebuild a shard spec without re-parsing a shard manifest.)
  - `lab.store.cell_id_for(coords: dict[str, Any]) -> str` — deterministic 8-hex id from canonical coords.
  - `JobStore.sweep_plan_path(sweep_id) -> Path`, `JobStore.write_sweep_plan(plan)`, `JobStore.read_sweep_plan(sweep_id) -> SweepPlan`, `JobStore.has_sweep_plan(sweep_id) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_plan_store.py
from __future__ import annotations

from pathlib import Path

from lab._util import now
from lab.models import SweepCell, SweepPlan
from lab.store import JobStore, cell_id_for


def test_cell_id_stable_and_order_independent():
    a = cell_id_for({"N": "1000", "alpha": "0.5"})
    b = cell_id_for({"alpha": "0.5", "N": "1000"})
    assert a == b and len(a) == 8


def test_write_read_roundtrip(tmp_path: Path):
    store = JobStore(tmp_path)
    cell = SweepCell(
        coords={"N": "1000"},
        cell_id=cell_id_for({"N": "1000"}),
        seeds_expected=[0, 1, 2, 3],
        shard_seeds=[[0, 1], [2, 3]],
        shard_job_ids=["j1", "j2"],
        results_file="results.csv",
        seed_column="seed",
        aggregate_ref=str(tmp_path / "sw1" / "cells" / "x" / "results.csv"),
    )
    plan = SweepPlan(
        sweep_id="sw1", created_at=now(), command="true", seed_axis_key="seeds", cells=[cell]
    )
    assert not store.has_sweep_plan("sw1")
    store.write_sweep_plan(plan)
    assert store.has_sweep_plan("sw1")
    got = store.read_sweep_plan("sw1")
    assert got.cells[0].status == "pending"
    assert got.cells[0].seeds_expected == [0, 1, 2, 3]
    assert got.sweep_id == "sw1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep_plan_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'SweepCell' from 'lab.models'`.

- [ ] **Step 3a: Add models to `src/lab/models.py`**

Append after the `JobManifest` class (keep `Literal` import — it is already imported at the top):

```python
class SweepCell(BaseModel):
    """One non-seed grid point of a sharded sweep, plus its seed-shard bookkeeping (P1-2).

    The authoritative cell->shards map. ``seeds_present``/``missing_seeds``/``status`` are filled by
    aggregation (pending until then); the shard job manifests stay the fail-closed source of truth
    for code/seed state — this record is grouping + accounting, never a provenance substitute.
    """

    coords: dict[str, Any]
    cell_id: str
    seeds_expected: list[int]
    shard_seeds: list[list[int]]
    shard_job_ids: list[str]
    results_file: str
    seed_column: str
    aggregate_ref: str
    seeds_present: list[int] = Field(default_factory=list)
    missing_seeds: list[int] = Field(default_factory=list)
    status: Literal["pending", "complete", "incomplete"] = "pending"


class SweepPlan(BaseModel):
    """Persisted plan for a sharded sweep (P1-2), keyed by ``sweep_id`` under the lab home."""

    sweep_id: str
    created_at: datetime
    command: str  # base entrypoint, so retry_sweep can rebuild a shard spec without a manifest re-parse
    seed_axis_key: str  # config-override key carrying each shard's seed subset (default "seeds")
    cells: list[SweepCell]
```

- [ ] **Step 3b: Add store helpers to `src/lab/store.py`**

Add the import and a module-level helper near the top (after the existing imports):

```python
import hashlib
from lab.models import JobManifest, JobState, SweepPlan


def cell_id_for(coords: dict[str, Any]) -> str:
    """Deterministic, order-independent 8-hex id for a cell's non-seed coordinates."""
    canon = json.dumps(coords, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:8]
```

Add these methods inside `JobStore` (after `read_runtime`):

```python
    def sweep_plan_path(self, sweep_id: str) -> Path:
        return self.home / sweep_id / "plan.json"

    def has_sweep_plan(self, sweep_id: str) -> bool:
        return self.sweep_plan_path(sweep_id).exists()

    def write_sweep_plan(self, plan: SweepPlan) -> None:
        self._atomic_write(self.sweep_plan_path(plan.sweep_id), plan.model_dump_json(indent=2))

    def read_sweep_plan(self, sweep_id: str) -> SweepPlan:
        return SweepPlan.model_validate_json(self.sweep_plan_path(sweep_id).read_text())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep_plan_store.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Type-check + commit**

Run: `uv run mypy src/lab/models.py src/lab/store.py`
Expected: no errors.

```bash
git add src/lab/models.py src/lab/store.py tests/test_sweep_plan_store.py
git commit -m "feat(sweep): SweepPlan/SweepCell models + store persistence + cell_id"
```

---

### Task 3: `Lab.sweep` builds and persists the shard plan

**Files:**
- Modify: `src/lab/core.py` (extend `Lab.sweep`; add `Lab.sweep_plan`)
- Modify: `src/lab/models.py` (add `cell_id` to `JobManifest`)
- Modify: `src/lab/store.py` (set `cell_id` is already a manifest field — no extra work)
- Test: `tests/test_sweep_sharding.py`

**Interfaces:**
- Consumes: `parse_seeds`, `partition_seeds`, `seeds_to_arg` (Task 1); `SweepCell`, `SweepPlan` (Task 2); `cell_id_for` (Task 2); existing `expand_grid`, `build_sweep_point_spec`, `check_sweep_admission`.
- Produces:
  - Extended `Lab.sweep(..., seeds: str | list[int] | None = None, shard_size: int | None = None, results_file: str = "results.csv", seed_column: str = "seed", seed_axis_key: str = "seeds")` — unchanged return `tuple[str, list[str]]` (sweep_id, all shard job_ids). When `seeds` is None it behaves exactly as today and writes **no** plan.
  - `Lab.sweep_plan(sweep_id: str) -> SweepPlan` — read the persisted plan; raises `LabError` if none.
  - `JobManifest.cell_id: str | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_sharding.py
from __future__ import annotations

from pathlib import Path

import pytest

from lab.backends.local import LocalBackend
from lab.core import Lab, LabError
from lab.manifest import repo_root


def _lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    return Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)


def test_sweep_no_seeds_writes_no_plan(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, job_ids = lab.sweep("true", {"N": [1000]})
    assert len(job_ids) == 1
    assert not lab.store.has_sweep_plan(sweep_id)


def test_sharded_sweep_partitions_and_plans(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, job_ids = lab.sweep("true", {"N": [1000]}, seeds="0-7", shard_size=2)
    # one cell, 4 shards of 2 seeds each
    assert len(job_ids) == 4
    plan = lab.sweep_plan(sweep_id)
    assert len(plan.cells) == 1
    cell = plan.cells[0]
    assert cell.coords == {"N": "1000"}
    assert cell.seeds_expected == [0, 1, 2, 3, 4, 5, 6, 7]
    assert cell.shard_seeds == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert len(cell.shard_job_ids) == 4
    assert cell.status == "pending"
    # each shard's command carries its seed subset under the seed-axis key
    cmds = [lab.manifest(j).run.entrypoint_command for j in cell.shard_job_ids]
    assert any("seeds=0,1" in c for c in cmds)
    assert any("seeds=6,7" in c for c in cmds)
    # each shard records its cell_id; the per-job singular seed anchors to the shard's first seed
    assert all(lab.manifest(j).cell_id == cell.cell_id for j in cell.shard_job_ids)


def test_seeds_in_both_axis_and_grid_is_rejected(tmp_path: Path):
    lab = _lab(tmp_path)
    with pytest.raises(LabError, match="both"):
        lab.sweep("true", {"seeds": [0, 1]}, seeds="0-7", shard_size=2)


def test_shard_size_ge_len_is_one_shard_per_cell(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, job_ids = lab.sweep("true", {"N": [1000, 1500]}, seeds="0-3", shard_size=8)
    assert len(job_ids) == 2  # one shard per cell
    plan = lab.sweep_plan(sweep_id)
    assert all(len(c.shard_seeds) == 1 for c in plan.cells)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep_sharding.py -v`
Expected: FAIL — `Lab.sweep()` got an unexpected keyword argument `seeds`.

- [ ] **Step 3a: Add `cell_id` to `JobManifest` (`src/lab/models.py`)**

In `JobManifest`, add after `sweep_id`:

```python
    cell_id: str | None = None  # sharded-sweep cell grouping (P1-2); None for non-sharded jobs
```

- [ ] **Step 3b: Thread `cell_id` through `Lab.submit` (`src/lab/core.py`)**

In `Lab.submit`, add a `cell_id: str | None = None` keyword parameter (after `confirms`), and pass it into the `JobManifest(...)` construction:

```python
            sweep_id=sweep_id,
            cell_id=cell_id,
```

- [ ] **Step 3c: Extend `Lab.sweep` (`src/lab/core.py`)**

Add imports at the top of `core.py`:

```python
from lab.sharding import parse_seeds, partition_seeds, seeds_to_arg
```

Replace the body of `Lab.sweep` with the sharded version. New signature adds `seeds`, `shard_size`, `results_file`, `seed_column`, `seed_axis_key`. Keep the existing docstring's first paragraph and append a sharding note.

```python
    def sweep(
        self,
        command: str,
        grid: dict[str, list[Any]],
        *,
        resources: ResourceRequest | None = None,
        seed: int | None = None,
        code_ref: str = "HEAD",
        submitted_by: str = "agent",
        allow_dirty: bool = True,
        max_jobs: int = 256,
        sweep_max_cost: float | None = None,
        daily_budget: float | None = None,
        committed: float = 0.0,
        seeds: str | list[int] | None = None,
        shard_size: int | None = None,
        results_file: str = "results.csv",
        seed_column: str = "seed",
        seed_axis_key: str = "seeds",
    ) -> tuple[str, list[str]]:
        """Submit one job per grid point under a shared sweep_id (FR-A5).

        With ``seeds`` + ``shard_size`` (P1-2) each cell's seed set is partitioned into shards of at
        most ``shard_size`` seeds; each shard runs as its own job (own timeout + teardown) with its
        seed subset appended under ``seed_axis_key`` (e.g. ``seeds=0,1``). A ``SweepPlan`` is persisted
        for aggregation/retry. ``seeds`` absent ⇒ today's behavior, no plan written.
        """
        cells = expand_grid(grid)
        if seeds is None:
            return self._sweep_unsharded(
                command, cells, resources=resources, seed=seed, code_ref=code_ref,
                submitted_by=submitted_by, allow_dirty=allow_dirty, max_jobs=max_jobs,
                sweep_max_cost=sweep_max_cost, daily_budget=daily_budget, committed=committed,
            )
        if seed_axis_key in grid:
            raise LabError(
                f"seeds declared in both 'seeds' and grid key {seed_axis_key!r}; "
                "remove one — seeds are an aggregation axis, not a Cartesian grid key"
            )
        seed_set = parse_seeds(seeds)
        shards = partition_seeds(seed_set, shard_size if shard_size is not None else len(seed_set))
        n_jobs = len(cells) * len(shards)
        if n_jobs > max_jobs:
            raise LabError(
                f"sharded sweep would submit {n_jobs} jobs (> max_jobs={max_jobs}); "
                "narrow the grid/seeds, raise shard_size, or raise max_jobs"
            )
        per_point_cap: float | None = (
            sweep_max_cost / n_jobs if sweep_max_cost is not None and n_jobs > 0 else None
        )
        check_sweep_admission(
            n_points=n_jobs, per_point_cap=per_point_cap,
            daily_budget=daily_budget, committed=committed,
        )
        sweep_id = f"sweep-{_new_job_id()}"
        all_job_ids: list[str] = []
        plan_cells: list[SweepCell] = []
        for cell in cells:
            cid = cell_id_for(cell)
            shard_job_ids: list[str] = []
            for shard in shards:
                point = {**cell, seed_axis_key: seeds_to_arg(shard)}
                spec = build_sweep_point_spec(
                    command, point, seed=shard[0], resources=resources,
                    code_ref=code_ref, submitted_by=submitted_by,
                )
                jid = self.submit(
                    spec, allow_dirty=allow_dirty, sweep_id=sweep_id, cell_id=cid
                )
                shard_job_ids.append(jid)
                all_job_ids.append(jid)
            plan_cells.append(
                SweepCell(
                    coords=cell, cell_id=cid, seeds_expected=seed_set, shard_seeds=shards,
                    shard_job_ids=shard_job_ids, results_file=results_file, seed_column=seed_column,
                    aggregate_ref=str(self.home / sweep_id / "cells" / cid / results_file),
                )
            )
        self.store.write_sweep_plan(
            SweepPlan(
                sweep_id=sweep_id, created_at=now(), command=command,
                seed_axis_key=seed_axis_key, cells=plan_cells,
            )
        )
        return sweep_id, all_job_ids

    def _sweep_unsharded(
        self,
        command: str,
        points: list[dict[str, Any]],
        *,
        resources: ResourceRequest | None,
        seed: int | None,
        code_ref: str,
        submitted_by: str,
        allow_dirty: bool,
        max_jobs: int,
        sweep_max_cost: float | None,
        daily_budget: float | None,
        committed: float,
    ) -> tuple[str, list[str]]:
        """The pre-P1-2 one-job-per-cell path (FR-A5), extracted unchanged."""
        if len(points) > max_jobs:
            raise LabError(
                f"sweep would submit {len(points)} jobs (> max_jobs={max_jobs}); "
                "narrow the grid or raise max_jobs"
            )
        per_point_cap: float | None = (
            sweep_max_cost / len(points) if sweep_max_cost is not None and len(points) > 0 else None
        )
        check_sweep_admission(
            n_points=len(points), per_point_cap=per_point_cap,
            daily_budget=daily_budget, committed=committed,
        )
        sweep_id = f"sweep-{_new_job_id()}"
        job_ids: list[str] = []
        for point in points:
            spec = build_sweep_point_spec(
                command, point, seed=seed, resources=resources,
                code_ref=code_ref, submitted_by=submitted_by,
            )
            job_ids.append(self.submit(spec, allow_dirty=allow_dirty, sweep_id=sweep_id))
        return sweep_id, job_ids

    def sweep_plan(self, sweep_id: str) -> SweepPlan:
        """Read the persisted shard plan for a sharded sweep (P1-2)."""
        if not self.store.has_sweep_plan(sweep_id):
            raise LabError(f"no shard plan for {sweep_id!r} (not a sharded sweep?)")
        return self.store.read_sweep_plan(sweep_id)
```

Add the imports for the new model + helper near the top of `core.py` (with the other `lab.models` / `lab.store` imports):

```python
from lab.models import SweepCell, SweepPlan  # add to the existing lab.models import line
from lab.store import cell_id_for  # add to the existing lab.store import
```

> Note: `seed_column` is carried into the `SweepCell` here but only *consumed* by Task 4's aggregator. It is recorded at submit time so aggregation needs no extra inputs.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep_sharding.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Regression + type-check + commit**

Run: `uv run pytest tests/test_core.py tests/test_sweep_budget.py tests/test_sweep_summary.py -v`
Expected: PASS (existing sweep behavior unchanged).
Run: `uv run mypy src/lab/core.py src/lab/models.py`
Expected: no errors.

```bash
git add src/lab/core.py src/lab/models.py tests/test_sweep_sharding.py
git commit -m "feat(sweep): partition cells into seed shards + persist SweepPlan"
```

---

### Task 4: Pure results-merge helper (row-concatenation + present seeds)

**Files:**
- Create: `src/lab/aggregate.py`
- Test: `tests/test_aggregate_merge.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `merge_seed_rows(csv_texts: list[str], seed_column: str) -> tuple[str, list[int]]` — concatenate the CSV bodies (a single shared header, validated identical across inputs), sort the data rows by the integer value in `seed_column`, and return `(merged_csv_text, sorted_present_seeds)`. Row **content is never altered**; only ordering is normalized (FR-SS-4). Raises `ValueError` if headers differ, `seed_column` is absent, or a seed value is non-integer. An empty input list returns `("", [])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregate_merge.py
from __future__ import annotations

import pytest

from lab.aggregate import merge_seed_rows


def test_merge_concatenates_and_sorts_by_seed():
    a = "seed,acc\n2,0.9\n3,0.8\n"
    b = "seed,acc\n0,0.7\n1,0.6\n"
    merged, present = merge_seed_rows([b, a], "seed")
    assert merged == "seed,acc\n0,0.7\n1,0.6\n2,0.9\n3,0.8\n"
    assert present == [0, 1, 2, 3]


def test_merge_preserves_row_content_unaltered():
    a = "seed,note\n0,hello world\n"
    merged, present = merge_seed_rows([a], "seed")
    assert "hello world" in merged
    assert present == [0]


def test_merge_rejects_mismatched_headers():
    with pytest.raises(ValueError, match="header"):
        merge_seed_rows(["seed,acc\n0,1\n", "seed,loss\n1,2\n"], "seed")


def test_merge_rejects_missing_seed_column():
    with pytest.raises(ValueError, match="seed_column"):
        merge_seed_rows(["acc\n0.9\n"], "seed")


def test_merge_empty():
    assert merge_seed_rows([], "seed") == ("", [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregate_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab.aggregate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lab/aggregate.py
"""Pure per-cell results aggregation for sharded sweeps (P1-2): row-concatenate shard result CSVs
into one cell table and report which seeds are present. No I/O — the orchestration in lab.core does
the fetch/write; this module is the deterministic, unit-testable reduction."""

from __future__ import annotations

import csv
import io


def merge_seed_rows(csv_texts: list[str], seed_column: str) -> tuple[str, list[int]]:
    """Concatenate shard result CSVs (identical headers) into one table sorted by ``seed_column``.

    Returns ``(merged_csv_text, sorted_present_seeds)``. Row content is preserved verbatim; only the
    row order is normalized. Raises ``ValueError`` on mismatched headers, a missing seed column, or a
    non-integer seed value.
    """
    if not csv_texts:
        return "", []
    header: list[str] | None = None
    rows: list[tuple[int, dict[str, str]]] = []
    for text in csv_texts:
        reader = csv.reader(io.StringIO(text))
        try:
            this_header = next(reader)
        except StopIteration:
            continue
        if header is None:
            header = this_header
        elif this_header != header:
            raise ValueError(f"shard result header {this_header} != {header}")
        if seed_column not in header:
            raise ValueError(f"seed_column {seed_column!r} not in results header {header}")
        idx = header.index(seed_column)
        for raw in reader:
            if not raw:
                continue
            try:
                seed_val = int(raw[idx])
            except (ValueError, IndexError) as e:
                raise ValueError(f"non-integer {seed_column} in row {raw}") from e
            rows.append((seed_val, dict(zip(header, raw))))
    if header is None:
        return "", []
    rows.sort(key=lambda r: r[0])
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    for _, row in rows:
        writer.writerow([row[c] for c in header])
    return out.getvalue(), [seed for seed, _ in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_aggregate_merge.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lab/aggregate.py tests/test_aggregate_merge.py
git commit -m "feat(sweep): pure per-cell results-merge helper (concat + present seeds)"
```

---

### Task 5: `Lab.aggregate_sweep` — idempotent pull reducer

**Files:**
- Modify: `src/lab/core.py` (add `Lab.aggregate_sweep`)
- Test: `tests/test_aggregate_sweep.py`

**Interfaces:**
- Consumes: `Lab.sweep_plan`, `Lab.fetch_artifacts`, `Lab.manifest`, `Lab.store` (Tasks 2-3); `merge_seed_rows` (Task 4); `JobState`.
- Produces:
  - `Lab.aggregate_sweep(sweep_id: str) -> SweepPlan` — for each cell: fetch each **succeeded** shard's `results_file`, merge rows, write the aggregate to `aggregate_ref` (creating parent dirs; mirror to R2 if enabled), set `seeds_present`, `missing_seeds`, and `status` (`complete` iff present == expected, else `incomplete`). Persist and return the updated plan. Idempotent: re-running recomputes from current shard states. A cell with zero succeeded shards is `incomplete` with all seeds missing and no aggregate file written.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregate_sweep.py
from __future__ import annotations

from pathlib import Path

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import JobState


def _lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    return Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)


def _write_shard_result(lab: Lab, job_id: str, seeds: list[int]) -> None:
    """Mark a shard succeeded and drop a results.csv with one row per seed."""
    out = lab.store.output_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["seed,acc"] + [f"{s},0.{s}" for s in seeds]
    (out / "results.csv").write_text("\n".join(lines) + "\n")
    lab.store.update_manifest(job_id, status=JobState.succeeded)


def test_aggregate_complete_cell(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    plan = lab.sweep_plan(sweep_id)
    cell = plan.cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    _write_shard_result(lab, cell.shard_job_ids[1], [2, 3])

    updated = lab.aggregate_sweep(sweep_id)
    c = updated.cells[0]
    assert c.status == "complete"
    assert c.seeds_present == [0, 1, 2, 3]
    assert c.missing_seeds == []
    agg = Path(c.aggregate_ref).read_text()
    assert agg == "seed,acc\n0,0.0\n1,0.1\n2,0.2\n3,0.3\n"


def test_aggregate_partial_failure_is_honest(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    plan = lab.sweep_plan(sweep_id)
    cell = plan.cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.failed)  # shard 2 dies

    updated = lab.aggregate_sweep(sweep_id)
    c = updated.cells[0]
    assert c.status == "incomplete"
    assert c.seeds_present == [0, 1]
    assert c.missing_seeds == [2, 3]
    assert Path(c.aggregate_ref).read_text() == "seed,acc\n0,0.0\n1,0.1\n"


def test_aggregate_is_idempotent_and_resumable(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _write_shard_result(lab, cell.shard_job_ids[0], [0, 1])
    first = lab.aggregate_sweep(sweep_id).cells[0]
    assert first.status == "incomplete" and first.missing_seeds == [2, 3]
    _write_shard_result(lab, cell.shard_job_ids[1], [2, 3])  # second shard finishes later
    second = lab.aggregate_sweep(sweep_id).cells[0]
    assert second.status == "complete" and second.seeds_present == [0, 1, 2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregate_sweep.py -v`
Expected: FAIL — `Lab` object has no attribute `aggregate_sweep`.

- [ ] **Step 3: Implement `Lab.aggregate_sweep` (`src/lab/core.py`)**

Add the import:

```python
from lab.aggregate import merge_seed_rows
```

Add the method to the `Lab` class (after `sweep_plan`):

```python
    def aggregate_sweep(self, sweep_id: str) -> SweepPlan:
        """Row-concatenate each cell's succeeded shards into one per-cell result (P1-2, FR-SS-4..7).

        Idempotent pull reducer: recomputes from current shard states each call, so it is safe to run
        repeatedly as shards finish. A cell is ``complete`` iff every expected seed is present, else
        ``incomplete`` with the missing seeds named — never presents a short aggregate as complete and
        never discards recovered seeds (FR-SS-7).
        """
        plan = self.sweep_plan(sweep_id)
        for cell in plan.cells:
            texts: list[str] = []
            for jid in cell.shard_job_ids:
                if self.manifest(jid).status is not JobState.succeeded:
                    continue
                self.fetch_artifacts(jid)  # ensure the local copy exists (R2 fallback inside)
                rf = self.store.output_dir(jid) / cell.results_file
                if rf.exists():
                    texts.append(rf.read_text())
            merged, present = merge_seed_rows(texts, cell.seed_column)
            cell.seeds_present = present
            present_set = set(present)
            cell.missing_seeds = [s for s in cell.seeds_expected if s not in present_set]
            cell.status = "complete" if not cell.missing_seeds else "incomplete"
            if merged:
                dest = Path(cell.aggregate_ref)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(merged)
                if r2_enabled():
                    r2 = R2Store.from_env()
                    if r2 is not None:
                        try:
                            r2.upload_file(dest, f"{sweep_id}/cells/{cell.cell_id}/{cell.results_file}")
                        except Exception as e:  # noqa: BLE001 — local aggregate stays authoritative
                            print(f"[lab] aggregate R2 mirror failed, keeping local copy: {e}")
        self.store.write_sweep_plan(plan)
        return plan
```

> `r2_enabled` and `R2Store` are already imported in `core.py` (used by `submit`/`fetch_artifacts`). Confirm by grepping; if not, add `from lab.storage import R2Store, r2_enabled`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_aggregate_sweep.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Type-check + commit**

Run: `uv run mypy src/lab/core.py src/lab/aggregate.py`
Expected: no errors.

```bash
git add src/lab/core.py tests/test_aggregate_sweep.py
git commit -m "feat(sweep): aggregate_sweep — idempotent per-cell row aggregation with honest partial-failure"
```

---

### Task 6: `Lab.retry_sweep` — resubmit only missing shards

**Files:**
- Modify: `src/lab/core.py` (add `Lab.retry_sweep`)
- Test: `tests/test_retry_sweep.py`

**Interfaces:**
- Consumes: `Lab.sweep_plan`, `Lab.aggregate_sweep`, `Lab.manifest`, `Lab.submit`, `build_sweep_point_spec`, `seeds_to_arg`, `JobState`.
- Produces:
  - `Lab.retry_sweep(sweep_id: str) -> SweepPlan` — for each `incomplete` cell, find shards whose seeds are not fully present, resubmit **those shards only** as fresh jobs under the same `sweep_id`/`cell_id` (appending the shard's seed subset under `plan.seed_axis_key`), append the new job ids to the cell's `shard_job_ids`, persist the plan, then re-aggregate and return the updated plan. Succeeded shards are untouched.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retry_sweep.py
from __future__ import annotations

from pathlib import Path

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import JobState


def _lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    return Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)


def _succeed(lab: Lab, job_id: str, seeds: list[int]) -> None:
    out = lab.store.output_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.csv").write_text("seed,acc\n" + "".join(f"{s},0.{s}\n" for s in seeds))
    lab.store.update_manifest(job_id, status=JobState.succeeded)


def test_retry_resubmits_only_missing_shards(tmp_path: Path):
    lab = _lab(tmp_path)
    sweep_id, _ = lab.sweep("true", {"N": [1000]}, seeds="0-3", shard_size=2)
    cell = lab.sweep_plan(sweep_id).cells[0]
    _succeed(lab, cell.shard_job_ids[0], [0, 1])
    lab.store.update_manifest(cell.shard_job_ids[1], status=JobState.failed)
    lab.aggregate_sweep(sweep_id)

    before_ids = set(lab.sweep_plan(sweep_id).cells[0].shard_job_ids)
    updated = lab.retry_sweep(sweep_id)
    c = updated.cells[0]
    new_ids = [j for j in c.shard_job_ids if j not in before_ids]
    assert len(new_ids) == 1  # only the missing shard (seeds 2,3) was resubmitted
    assert "seeds=2,3" in lab.manifest(new_ids[0]).run.entrypoint_command
    assert lab.manifest(new_ids[0]).cell_id == c.cell_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retry_sweep.py -v`
Expected: FAIL — `Lab` object has no attribute `retry_sweep`.

- [ ] **Step 3: Implement `Lab.retry_sweep` (`src/lab/core.py`)**

Add the method after `aggregate_sweep`:

```python
    def retry_sweep(self, sweep_id: str, *, allow_dirty: bool = True) -> SweepPlan:
        """Resubmit only the missing shards of incomplete cells, then re-aggregate (P1-2, FR-SS-7).

        A shard is missing if any of its assigned seeds is absent from the current aggregate. Fresh
        shard jobs join the same ``sweep_id``/``cell_id``; succeeded shards are never touched.
        """
        plan = self.aggregate_sweep(sweep_id)  # refresh present/missing from current shard states
        for cell in plan.cells:
            if cell.status != "incomplete":
                continue
            present = set(cell.seeds_present)
            # inherit the original shard resources (timeout/backend/etc.) from an existing shard
            base_resources = self.manifest(cell.shard_job_ids[0]).resources
            for shard in cell.shard_seeds:
                if all(s in present for s in shard):
                    continue  # this shard's seeds are already covered
                point = {**cell.coords, plan.seed_axis_key: seeds_to_arg(shard)}
                spec = build_sweep_point_spec(
                    plan.command, point, seed=shard[0], resources=base_resources
                )
                jid = self.submit(
                    spec, allow_dirty=allow_dirty, sweep_id=sweep_id, cell_id=cell.cell_id
                )
                cell.shard_job_ids.append(jid)
        self.store.write_sweep_plan(plan)
        return self.aggregate_sweep(sweep_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retry_sweep.py tests/test_sweep_plan_store.py -v`
Expected: PASS (the store roundtrip test now sets `command=...`).

- [ ] **Step 5: Type-check + commit**

Run: `uv run mypy src/lab/core.py src/lab/models.py`
Expected: no errors.

```bash
git add src/lab/core.py src/lab/models.py tests/test_retry_sweep.py tests/test_sweep_plan_store.py
git commit -m "feat(sweep): retry_sweep — resubmit only missing shards + re-aggregate"
```

---

### Task 7: CLI surface — flags + `sweep-aggregate` / `sweep-retry`

**Files:**
- Modify: `src/lab/cli.py` (extend `sweep`; add two commands)
- Test: `tests/test_sweep_cli.py`

**Interfaces:**
- Consumes: `Lab.sweep`, `Lab.aggregate_sweep`, `Lab.retry_sweep`, `Lab.sweep_plan`.
- Produces: CLI flags `--seeds`, `--shard-size`, `--results-file`, `--seed-column` on `lab sweep`; new commands `lab sweep-aggregate <sweep_id>` and `lab sweep-retry <sweep_id>`. When a plan was written, `lab sweep` emits the structured `cells` view; otherwise the existing `{sweep_id, count, job_ids}` shape (AC-4).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_cli.py
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lab.cli import app

runner = CliRunner()


def test_sweep_sharded_emits_cells(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LAB_HOME", str(tmp_path))  # if cli honors LAB_HOME; else use --home pattern
    res = runner.invoke(
        app,
        ["sweep", "-c", "true", "-g", "N=1000", "--seeds", "0-3", "--shard-size", "2"],
    )
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["sweep_id"]
    assert len(out["cells"]) == 1
    assert out["cells"][0]["seeds_expected"] == 4
    assert len(out["cells"][0]["shard_job_ids"]) == 2


def test_sweep_unsharded_unchanged(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LAB_HOME", str(tmp_path))
    res = runner.invoke(app, ["sweep", "-c", "true", "-g", "N=1000,1500"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["count"] == 2
    assert "cells" not in out
```

> Before writing the implementation, **check how the CLI selects its home** (read `src/lab/cli.py` `_lab`/`_lab(provisioner)`): if it uses `default_lab(home=...)` via an env var or a global option, mirror that in the test instead of `LAB_HOME`. Match the pattern in `tests/test_cli_spot.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep_cli.py -v`
Expected: FAIL — unknown option `--seeds` (exit code 2).

- [ ] **Step 3: Extend the CLI (`src/lab/cli.py`)**

Add options to the `sweep` command signature (after `sweep_max_cost`):

```python
    seeds: str | None = typer.Option(None, "--seeds", help="seed set as a range '0-31' or comma list '0,1,2'; declares seeds as a sharded axis (P1-2)"),
    shard_size: int | None = typer.Option(None, "--shard-size", help="max seeds per sub-job; each cell's seeds are split into shards of this size"),
    results_file: str = typer.Option("results.csv", "--results-file", help="per-run row-structured result file to aggregate per cell"),
    seed_column: str = typer.Option("seed", "--seed-column", help="column in --results-file identifying each row's seed"),
```

Pass them into the `lab.sweep(...)` call:

```python
            seeds=parse_seeds_arg(seeds),
            shard_size=shard_size,
            results_file=results_file,
            seed_column=seed_column,
```

where `parse_seeds_arg` accepts the raw string and returns `None` unchanged when `seeds is None` (a comma list is passed through to `lab.sweep`, which calls `parse_seeds`). Add a tiny local helper near the top of `cli.py`:

```python
def parse_seeds_arg(raw: str | None) -> str | list[int] | None:
    """CLI seeds flag passthrough: a comma list becomes a list[int]; a range string stays a string."""
    if raw is None:
        return None
    return [int(x) for x in raw.split(",")] if "," in raw else raw
```

Replace the final `_emit` in `sweep` so the structured view is emitted when a plan exists:

```python
    if lab.store.has_sweep_plan(sweep_id):
        plan = lab.sweep_plan(sweep_id)
        _emit({
            "sweep_id": sweep_id,
            "cells": [
                {
                    "coords": c.coords,
                    "cell_id": c.cell_id,
                    "shard_job_ids": c.shard_job_ids,
                    "aggregate_ref": c.aggregate_ref,
                    "seeds_expected": len(c.seeds_expected),
                    "seeds_present": len(c.seeds_present),
                    "status": c.status,
                }
                for c in plan.cells
            ],
        })
    else:
        _emit({"sweep_id": sweep_id, "count": len(job_ids), "job_ids": job_ids})
```

Add two commands (near `sweep_status`):

```python
@app.command(name="sweep-aggregate")
def sweep_aggregate(sweep_id: str) -> None:
    """Row-concatenate each cell's succeeded shards into one per-cell result (P1-2)."""
    try:
        plan = _lab().aggregate_sweep(sweep_id)
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit(_plan_view(plan))


@app.command(name="sweep-retry")
def sweep_retry(sweep_id: str) -> None:
    """Resubmit only the missing shards of incomplete cells, then re-aggregate (P1-2)."""
    try:
        plan = _lab().retry_sweep(sweep_id)
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    _emit(_plan_view(plan))
```

Add a shared `_plan_view(plan)` helper (factor out the dict-building used in `sweep`'s emit, so the three sites stay DRY):

```python
def _plan_view(plan) -> dict:  # type: ignore[no-untyped-def]
    return {
        "sweep_id": plan.sweep_id,
        "cells": [
            {
                "coords": c.coords, "cell_id": c.cell_id, "shard_job_ids": c.shard_job_ids,
                "aggregate_ref": c.aggregate_ref, "seeds_expected": len(c.seeds_expected),
                "seeds_present": len(c.seeds_present), "missing_seeds": c.missing_seeds,
                "status": c.status,
            }
            for c in plan.cells
        ],
    }
```

(Use `_plan_view` in the `sweep` emit branch too, instead of the inline dict.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep_cli.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Regression + type-check + commit**

Run: `uv run pytest tests/test_cli_spot.py tests/test_cli_wait.py -v`
Expected: PASS.
Run: `uv run mypy src/lab/cli.py`
Expected: no errors.

```bash
git add src/lab/cli.py tests/test_sweep_cli.py
git commit -m "feat(cli): sweep --seeds/--shard-size + sweep-aggregate/sweep-retry"
```

---

### Task 8: MCP surface — params + `sweep_aggregate` / `sweep_retry` tools

**Files:**
- Modify: `src/lab/mcp_server.py` (extend `sweep`; add two tools)
- Test: `tests/test_mcp_server.py` (add cases — match the existing style in that file)

**Interfaces:**
- Consumes: `Lab.sweep`, `Lab.aggregate_sweep`, `Lab.retry_sweep`, `Lab.sweep_plan`.
- Produces: `sweep` MCP tool gains optional `seeds`, `shard_size`, `results_file`, `seed_column`; returns the structured `cells` view when sharded, else `{sweep_id, job_ids}`. New tools `sweep_aggregate(sweep_id)` and `sweep_retry(sweep_id)` returning the same plan view.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_mcp_server.py — follow the existing harness for getting tool callables
def test_mcp_sweep_sharded_returns_cells(tmp_path, monkeypatch):
    # arrange a Lab over tmp_path exactly as the other mcp tests do, then call the sweep tool with
    # seeds="0-3", shard_size=2 and assert the result has one cell with 2 shard_job_ids and status
    # "pending". Then call sweep_aggregate after writing shard results and assert status flips.
    ...
```

> Open `tests/test_mcp_server.py` first and copy its tool-invocation harness (how it constructs the server/lab and resolves a tool by name). Replace the `...` with concrete arrange/act/assert mirroring `test_sweep_cli.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_server.py -k sweep_sharded -v`
Expected: FAIL — `sweep()` got an unexpected keyword argument `seeds`.

- [ ] **Step 3: Extend the MCP server (`src/lab/mcp_server.py`)**

Add params to the `sweep` tool signature (after `sweep_max_cost`): `seeds: str | list[int] | None = None`, `shard_size: int | None = None`, `results_file: str = "results.csv"`, `seed_column: str = "seed"`. Append to the docstring: `"With seeds + shard_size each cell's seeds are split into shards of at most shard_size, run as independent jobs (own timeout + teardown) and aggregated per cell; returns {sweep_id, cells:[{coords, shard_job_ids, aggregate_ref, seeds_expected, seeds_present, status}]}. results_file/seed_column name the per-run result table and its seed column."` Pass the four kwargs into `the_lab.sweep(...)`.

Replace the return with the plan-aware view (factor a module-level `_plan_view(plan)` mirroring the CLI's):

```python
        if the_lab.store.has_sweep_plan(sweep_id):
            return _plan_view(the_lab.sweep_plan(sweep_id))
        return {"sweep_id": sweep_id, "job_ids": job_ids}
```

Add two tools after `sweep_status`:

```python
    @mcp.tool
    def sweep_aggregate(sweep_id: str) -> dict[str, Any]:
        """Row-concatenate each cell's succeeded shards into one per-cell result; returns the cell view (P1-2)."""
        try:
            return _plan_view(_lab().aggregate_sweep(sweep_id))
        except LabError as e:
            raise ToolError(str(e)) from e

    @mcp.tool
    def sweep_retry(sweep_id: str) -> dict[str, Any]:
        """Resubmit only the missing shards of incomplete cells, then re-aggregate (P1-2)."""
        try:
            return _plan_view(_lab().retry_sweep(sweep_id))
        except LabError as e:
            raise ToolError(str(e)) from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mcp_server.py -k sweep -v`
Expected: PASS.

- [ ] **Step 5: Regression + type-check + commit**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: PASS.
Run: `uv run mypy src/lab/mcp_server.py`
Expected: no errors.

```bash
git add src/lab/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): sweep seeds/shard_size + sweep_aggregate/sweep_retry tools"
```

---

### Task 9: Guide doc + CLAUDE.md key fact

**Files:**
- Create: `docs/guides/sharded-sweeps.md`
- Modify: `CLAUDE.md` (add one Key-facts bullet)

**Interfaces:**
- Consumes: nothing.
- Produces: user-facing docs.

- [ ] **Step 1: Write the guide**

Create `docs/guides/sharded-sweeps.md` covering: the seed-axis contract (experiment must read the `seeds` config key — a comma list — and emit one row per seed in `results.csv` with a `seed` column; both overridable via `--results-file`/`--seed-column`); the CLI example `lab sweep -c "<cmd>" --grid N=1000,1500 --seeds 0-31 --shard-size 8`; that `timeout`/`backend`/`accelerators` apply **per shard**; the `sweep-aggregate` / `sweep-retry` lifecycle; how partial failure is reported (`status: incomplete` + named `missing_seeds`); and the provenance guarantee (each shard keeps its own fail-closed manifest; the plan references them — AC-5).

- [ ] **Step 2: Add the CLAUDE.md Key fact**

Add under "Key facts" in `CLAUDE.md`:

```markdown
- **Sharded sweeps (FR P1-2):** `lab sweep --seeds 0-31 --shard-size 8` splits each grid cell's
  seeds into independently-bounded shard jobs (own timeout + teardown), then `lab sweep-aggregate`
  row-concatenates the succeeded shards into one per-cell `results.csv` (seed column overridable),
  reporting `seeds_present` vs expected and naming missing seeds on partial failure;
  `lab sweep-retry` resubmits only the missing shards. A `SweepPlan` under the `sweep_id` is the
  cell→shards map. Guide: `docs/guides/sharded-sweeps.md`.
```

- [ ] **Step 3: Full test suite + lint**

Run: `uv run pytest -q && uv run ruff check src/lab && uv run mypy src/lab`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add docs/guides/sharded-sweeps.md CLAUDE.md
git commit -m "docs(sweep): sharded-sweep guide + CLAUDE key fact"
```

---

## Self-Review

**Spec coverage:**
- FR-SS-1 (partition, complete non-overlapping cover) → Task 1 `partition_seeds` + Task 3.
- FR-SS-2 (per-shard timeout + teardown) → inherited; each shard is a normal job (Task 3 submits via `Lab.submit`; `resources`/`timeout` pass through per shard). Asserted by AC-2.
- FR-SS-3 (shard records its seeds; reproducible per shard) → Task 3 records the seed subset in `resolved_config` + singular `seed` anchor; fail-closed manifest per shard.
- FR-SS-4 (one aggregated artifact, row-equivalent, order may normalize, content unaltered) → Task 4 `merge_seed_rows` + Task 5.
- FR-SS-5 (aggregate addressable + fetchable) → `aggregate_ref` (Task 3/5), written under the store + R2 mirror.
- FR-SS-6 (expected vs present) → Task 5 `seeds_present`/`seeds_expected`.
- FR-SS-7 (partial-failure honesty + name missing + don't discard + MAY re-run) → Task 5 `status`/`missing_seeds`; Task 6 `retry_sweep`.
- FR-SS-8 (own manifest + aggregate references shards) → per-shard manifests retained; `SweepPlan.cells[].shard_job_ids`.
- FR-SS-9 (each shard observable; grouping discoverable) → shards are normal jobs (status/metrics/logs unchanged); plan + `cell_id` expose grouping; CLI/MCP emit cell view.
- API surface (§3: `seeds`, `shard_size`, error on double-declared, per-shard timeout, structured return) → Tasks 3/7/8.
- AC-1..AC-5 → Tasks 3/5/6 tests + Task 9 full-suite gate.
- Out-of-scope merged metrics + deferred `register_sweep` → intentionally not implemented (noted in design §2 / §9).

**Placeholder scan:** No placeholders remain. `SweepPlan.command` is defined in Task 2 (model + roundtrip test), set in Task 3 (`command=command`), and consumed in Task 6 — no cross-task amendment is left for the executor. Task 8 Step 1 carries a deliberate `...` skeleton **with explicit instructions** to copy the existing `tests/test_mcp_server.py` harness and mirror `test_sweep_cli.py`; this is intentional because the MCP test-invocation harness must be read from that file rather than guessed.

**Type consistency:** `SweepCell`/`SweepPlan` field names are consistent across Tasks 2/3/5/6/7/8. `aggregate_sweep`/`retry_sweep`/`sweep_plan` signatures match their call sites. `merge_seed_rows(csv_texts, seed_column) -> (text, list[int])` consistent in Tasks 4/5. `command` field consistent (Task 2 def → Task 3 set → Task 6 use).
