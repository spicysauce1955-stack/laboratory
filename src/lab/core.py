"""Lab core — the single library both the CLI and the MCP server are thin shells over
(NFR-3, FR-F2). See research/10-architecture.md.

``Lab.submit`` resolves a :class:`~lab.models.JobSpec` into a :class:`~lab.models.JobManifest`
(pin commit, hash uv.lock, resolve seed), persists it via the store, then dispatches to the
chosen :class:`~lab.backends.base.Backend`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import shlex
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from lab._util import (
    actual_cost,
    atomic_write_text,
    duration_seconds,
    infer_artifact_type,
    now,
    tail_last_line,
)
from lab.backends.base import Backend
from lab.backends.local import LocalBackend
from lab.manifest import (
    capture_diff,
    commit_exists,
    current_commit,
    is_dirty,
    repo_root,
    uv_lock_sha256,
)
from lab import placement
from lab.metrics import final_values, group_series
from lab.storage import R2Store, r2_enabled
from lab.models import (
    ArtifactRecord,
    BackendInfo,
    CodeRef,
    EnvInfo,
    JobManifest,
    JobSpec,
    JobState,
    ResourceRequest,
    RunSpec,
    SweepCell,
    SweepPlan,
)
from lab.aggregate import merge_seed_rows
from lab.sharding import parse_seeds, partition_seeds, seeds_to_arg
from lab.store import JobStore, cell_id_for

_TERMINAL_STATES = frozenset(
    {
        JobState.succeeded, JobState.failed, JobState.cancelled,
        JobState.timed_out, JobState.preempted,
    }
)


class LabError(RuntimeError):
    """Fail-loud lab error (FR-F3)."""


def _new_job_id() -> str:
    return f"{now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid -> one config dict per point (FR-A5)."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _normalize_config(value: Any) -> Any:
    """Canonicalise config for hashing: stringify leaf values (preserving structure) so the same
    logical job hashes equal regardless of how its values were typed — the CLI keeps grid values as
    strings while the API/MCP pass ints/floats, and the experiment coerces types anyway."""
    if isinstance(value, dict):
        return {k: _normalize_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config(v) for v in value]
    return str(value)


def cache_key(commit: str, command: str, config: dict[str, Any] | None, seed: int) -> str:
    """Stable hash identifying an 'identical job' for result caching (FR-B5).

    The spec keys on commit+config+seed; we also include the command (entrypoint), since the lab
    runs arbitrary commands and two different experiments at the same commit/config/seed are not
    the same job. Config leaves are normalised (stringified) so a value isn't type-sensitive.
    """
    payload = json.dumps(
        {"commit": commit, "command": command, "config": _normalize_config(config or {}), "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def compare_final_metrics(
    orig: dict[str, float],
    new: dict[str, float],
    *,
    names: Iterable[str] | None,
    rtol: float = 1e-3,
    atol: float = 1e-12,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Judge a re-run's final metrics against the original's snapshot (the reproducibility gate).

    Returns ``("match" | "drift", deltas)``. ``names`` restricts which baseline metrics are judged
    (``None`` = all). A baseline metric absent from the re-run can't be re-derived → counts as drift.
    Tolerance is ``math.isclose`` semantics (relative ``rtol`` + absolute ``atol``). The small default
    ``atol`` is a float noise floor so a metric of exactly 0.0 doesn't false-drift against a tiny
    re-run value — ``math.isclose``'s relative tolerance collapses to zero at zero.
    """
    selected = list(names) if names is not None else list(orig)
    deltas: dict[str, dict[str, Any]] = {}
    verdict = "match"
    for name in selected:
        ov = orig.get(name)
        nv = new.get(name)
        if ov is None or nv is None:
            within = False
        else:
            within = math.isclose(ov, nv, rel_tol=rtol, abs_tol=atol)
        deltas[name] = {
            "orig": ov,
            "new": nv,
            "abs_delta": (abs(nv - ov) if ov is not None and nv is not None else None),
            "within_tol": within,
        }
        if not within:
            verdict = "drift"
    return verdict, deltas


def worst_case_sweep_cost(*, n_points: int, per_point_cap: float) -> float:
    return round(n_points * per_point_cap, 6)


def check_sweep_admission(
    *,
    n_points: int,
    per_point_cap: float | None,
    daily_budget: float | None,
    committed: float,
) -> float | None:
    """Refuse a sweep whose worst case won't fit the daily budget. Returns the worst-case cost
    (the default ceiling), or None when uncosted. Pure; no state (cost-safety, derived not metered)."""
    if per_point_cap is None:
        return None
    worst = worst_case_sweep_cost(n_points=n_points, per_point_cap=per_point_cap)
    if daily_budget is not None and committed + worst > daily_budget:
        raise LabError(
            f"sweep worst case ${worst:.2f} ({n_points} x ${per_point_cap:.2f}) + "
            f"committed ${committed:.2f} exceeds daily budget ${daily_budget:.2f}; "
            "narrow the grid, lower --max-cost, or raise the budget"
        )
    return worst


def _parse_row_key(row_key: str | list[str], seed_column: str) -> list[str]:
    """Parse/validate a row-key spec ('seed,alpha' or a list). Must include the seed column —
    presence/retry accounting is keyed on seeds. Pure; raises :class:`LabError`."""
    cols = (
        [c.strip() for c in row_key.split(",") if c.strip()]
        if isinstance(row_key, str)
        else list(row_key)
    )
    if seed_column not in cols:
        raise LabError(
            f"--row-key {cols} must include the seed column {seed_column!r} "
            "(presence/retry accounting is keyed on seeds)"
        )
    return cols


def _submit_stagger_s() -> float:
    """Delay between remote sweep submits (``LAB_SUBMIT_STAGGER_S``, default 1.5s; 0 disables).

    Each skypilot submit spawns a supervisor that immediately hits the local SkyPilot API
    server; N back-to-back spawns can refuse connections while the server cold-starts
    (field-report #4). A short stagger lets the first submit's autostart win. Lives in the sweep
    loops only — single submits and the (already 60s-paced) scheduler don't pay it."""
    import os

    try:
        return max(0.0, float(os.environ.get("LAB_SUBMIT_STAGGER_S", "1.5")))
    except ValueError:
        return 1.5


def build_sweep_point_spec(
    command: str,
    point: dict[str, Any],
    *,
    seed: int | None,
    resources: ResourceRequest | None = None,
    code_ref: str = "HEAD",
    submitted_by: str = "agent",
    allow_unknown_config: bool = False,
) -> JobSpec:
    """One grid point -> a JobSpec, identical for immediate (``Lab.sweep``) and deferred
    (``register_sweep``) paths so they can't drift.

    Point params are appended to the command as **shell-quoted** ``key=value`` overrides
    (injection-safe) and recorded in ``config``. A ``seed`` key in the point sets the per-point
    seed (must be int); otherwise the sweep-level ``seed`` default applies.
    """
    overrides = " ".join(shlex.quote(f"{k}={v}") for k, v in point.items())
    full_command = f"{command} {overrides}".strip()
    point_seed = point.get("seed")
    if point_seed is not None:
        try:
            job_seed: int | None = int(point_seed)
        except (TypeError, ValueError) as e:
            raise LabError(f"grid 'seed' values must be integers, got {point_seed!r}") from e
    else:
        job_seed = seed
    return JobSpec(
        code_ref=code_ref,
        command=full_command,
        seed=job_seed,
        config=point,
        resources=resources or ResourceRequest(),
        submitted_by=submitted_by,  # type: ignore[arg-type]
        allow_unknown_config=allow_unknown_config,
    )


SUPPORTED_CLOUDS: tuple[str, ...] = ("vast", "do", "gcp")

# How long a non-terminal skypilot job may lack a live supervisor pid before `reconcile` stops
# treating its cluster as protected (covers the submit->runtime-write race on fresh jobs).
UNSUPERVISED_GRACE_S = 300.0

CPU_DEFAULT_CLOUD = "do"
CPU_DEFAULT_VCPUS = 4
# The disk defaults and the "which clouds bill storage separately" rule live in lab.placement,
# because build_task needs them too: the scheduler launches registrations without ever calling
# resolve_backend_profile, so the invariant has to be enforced closer to the launch than this.
# Re-exported here since this is where callers already look for the profile constants.
CPU_DEFAULT_DISK_GB = placement.CPU_DEFAULT_DISK_GB
GPU_DEFAULT_DISK_GB = placement.GPU_DEFAULT_DISK_GB
STORAGE_BILLING_CLOUDS = placement.STORAGE_BILLING_CLOUDS
default_disk_gb = placement.effective_disk_gb


def validate_cloud(cloud: str | None) -> str | None:
    """Reject cloud names outside :data:`SUPPORTED_CLOUDS` (None = the Vast default). Pure."""
    if cloud is not None and cloud not in SUPPORTED_CLOUDS:
        raise LabError(
            f"unknown cloud {cloud!r}; supported: {', '.join(SUPPORTED_CLOUDS)} (default: vast)"
        )
    return cloud


def resolve_backend_profile(
    backend: str, resources: ResourceRequest
) -> tuple[str, ResourceRequest]:
    """Resolve the ``cpu`` convenience backend into (provisioner_name, resources).

    ``cpu`` is sugar for the SkyPilot provisioner on a cheap CPU cloud (DigitalOcean by default,
    overridable via ``resources.cloud``): it clears accelerators and defaults to
    ``CPU_DEFAULT_VCPUS`` vCPUs. Spot is forced off only on DO (which has none) — GCP CPU jobs
    may use preemptible.

    Every skypilot spec — cpu profile or not — also leaves here with an explicit ``disk_size`` on
    the storage-billing clouds, so no path can inherit SkyPilot's 256 GB default. That default was
    only ever noticed on DO, where it 422s loudly; on GCP it provisions fine and quietly doubles a
    spot job's bill. Pure; no I/O.
    """
    validate_cloud(resources.cloud)
    if backend != "cpu":
        disk = default_disk_gb(resources)
        if backend == "skypilot" and disk != resources.disk_size:
            return backend, resources.model_copy(update={"disk_size": disk})
        return backend, resources
    if resources.accelerators:
        raise LabError("--backend cpu provisions a CPU-only box; drop --accelerators")
    cloud = resources.cloud or CPU_DEFAULT_CLOUD
    update: dict[str, Any] = {
        "cloud": cloud,
        "cpus": resources.cpus or CPU_DEFAULT_VCPUS,
        "disk_size": resources.disk_size or CPU_DEFAULT_DISK_GB,
    }
    if cloud == "do":
        update["use_spot"] = False
        update["spot_fallback"] = False
    return "skypilot", resources.model_copy(update=update)


class Lab:
    def __init__(self, backend: Backend, repo: Path, home: Path) -> None:
        self.backend = backend
        self.repo = Path(repo)
        self.home = Path(home)
        self.store = JobStore(self.home)

    def preflight(self, spec: JobSpec) -> None:
        """Refuse a remote launch that a cheap local check proves cannot work (FR-F3).

        Only definitive negatives block — a quota of zero, a disabled API, an absent permission.
        A check that merely *fails to answer* is skipped, because a preflight that blocked on its
        own breakage would be worse than none. See :mod:`lab.doctor`.
        """
        if self.backend.name == "local":
            return
        from lab.doctor import format_report, preflight

        blocking = preflight(spec.resources.cloud, spec.resources, home=self.home)
        if not blocking:
            return
        raise LabError(
            "preflight refused this launch — it would fail after provisioning:\n"
            + format_report(blocking)
            + "\n(pass --no-preflight / preflight=False to launch anyway)"
        )

    def submit(
        self,
        spec: JobSpec,
        *,
        allow_dirty: bool = True,
        sweep_id: str | None = None,
        cell_id: str | None = None,
        code: CodeRef | None = None,
        registration_id: str | None = None,
        confirms: str | None = None,
        preflight: bool = True,
    ) -> str:
        """Build + persist the manifest, then launch via the backend (FR-A1, FR-B).

        ``code`` overrides git introspection — used by the scheduler, which submits from an
        extracted bundle (not a git repo) with provenance captured at registration time.
        """
        if preflight:
            self.preflight(spec)
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
                if blob is None:
                    # is_dirty said dirty but capture found nothing — the tree changed under us
                    # (e.g. a concurrent stash/checkout). Fail loud rather than write a Gap-B
                    # manifest (which would surface as a raw ValueError at create) (FR-B1).
                    raise LabError(
                        "working tree changed during submit (no diff to capture); retry (FR-B1)"
                    )
                diff_ref = blob
                if r2_enabled():
                    r2 = R2Store.from_env()
                    if r2 is not None:
                        rel = f"{job_id}/code_diff.tar.gz"
                        try:
                            r2.upload_file(Path(blob), rel)
                            diff_ref = r2.uri(rel)
                        except Exception as e:  # noqa: BLE001 — local diff_ref stays fail-closed
                            print(f"[lab] diff R2 upload failed, keeping local copy: {e}")
            code = CodeRef(
                git_commit=current_commit(self.repo), git_dirty=dirty, diff_ref=diff_ref
            )
        elif code.git_dirty and not allow_dirty:
            raise LabError("bundle captured a dirty tree but allow_dirty=False (FR-B1)")
        seed = spec.seed if spec.seed is not None else 0  # explicit + recorded (FR-B4)
        manifest = JobManifest(
            job_id=job_id,
            sweep_id=sweep_id,
            cell_id=cell_id,
            registration_id=registration_id,
            confirms=confirms,
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
                allow_unknown_config=spec.allow_unknown_config,
            ),
            resources=spec.resources,
            backend=BackendInfo(provisioner=self.backend.name),
            status=JobState.queued,
        )
        self.store.create(manifest)
        self.backend.submit(manifest)
        return job_id

    def find_cached(self, spec: JobSpec, *, require_clean: bool = True) -> str | None:
        """Return a prior SUCCEEDED job with the same commit+command+config+seed, else None (FR-B5).

        With ``require_clean`` (default), a dirty working tree disables caching and only clean-tree
        jobs are eligible — a dirty commit doesn't fully capture the code, so reusing its result
        isn't safe.
        """
        if require_clean and is_dirty(self.repo):
            return None
        seed = spec.seed if spec.seed is not None else 0
        key = cache_key(current_commit(self.repo), spec.command, spec.config, seed)
        for m in self.list_jobs():
            if m.status is not JobState.succeeded or (require_clean and m.code.git_dirty):
                continue
            if (
                cache_key(
                    m.code.git_commit, m.run.entrypoint_command, m.run.resolved_config, m.run.seed
                )
                == key
            ):
                return m.job_id
        return None

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
        allow_unknown_config: bool = False,
        row_key: str | list[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Submit one job per grid point under a shared sweep_id (FR-A5).

        Each point's params are appended to the command as **shell-quoted** ``key=value`` overrides
        (injection-safe) and recorded in the job's ``resolved_config``; jobs stay independently
        monitorable by ``job_id``. A ``seed`` key in the grid sets each job's seed (varying
        ``$LAB_SEED`` per point). Refuses to fan out beyond ``max_jobs`` (cost-safety).

        ``sweep_max_cost`` caps total sweep spend; ``daily_budget`` + ``committed`` enforce an
        up-front admission check (cost-safety, derived not metered). All default to no-op.

        With ``seeds`` + ``shard_size`` (P1-2) each cell's seed set is partitioned into shards of at
        most ``shard_size`` seeds; each shard runs as its own job (own timeout + teardown) with its
        seed subset appended under ``seed_axis_key`` (e.g. ``seeds=0,1``). A ``SweepPlan`` is
        persisted for aggregation/retry. ``seeds`` absent ⇒ today's behavior, no plan written.
        """
        if not grid and seeds is None:
            # One guard for every shell (thin-shell rule): a "sweep" with no axis would silently
            # run a single empty point instead of the fan-out the caller asked for.
            raise LabError("pass --grid and/or --seeds (a sweep needs at least one axis)")
        row_key_cols = _parse_row_key(row_key, seed_column) if row_key is not None else None
        cells = expand_grid(grid)
        if seeds is None:
            return self._sweep_unsharded(
                command, cells, resources=resources, seed=seed, code_ref=code_ref,
                submitted_by=submitted_by, allow_dirty=allow_dirty, max_jobs=max_jobs,
                sweep_max_cost=sweep_max_cost, daily_budget=daily_budget, committed=committed,
                allow_unknown_config=allow_unknown_config,
            )
        if seed_axis_key in grid or "seed" in grid:
            raise LabError(
                "seeds declared in both 'seeds' and a grid key ('seed'/'" + seed_axis_key + "'); "
                "remove one — seeds are an aggregation axis, not a Cartesian grid key"
            )
        try:
            seed_set = parse_seeds(seeds)
            shards = partition_seeds(seed_set, shard_size if shard_size is not None else len(seed_set))
        except ValueError as e:
            raise LabError(str(e)) from e
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
        stagger = _submit_stagger_s() if self.backend.name != "local" else 0.0
        for cell in cells:
            coords = {k: str(v) for k, v in cell.items()}
            cid = cell_id_for(coords)
            shard_job_ids: list[str] = []
            for shard in shards:
                if all_job_ids and stagger:
                    time.sleep(stagger)  # don't stampede the local SkyPilot API server
                point = {**cell, seed_axis_key: seeds_to_arg(shard)}
                spec = build_sweep_point_spec(
                    command, point, seed=shard[0], resources=resources,
                    code_ref=code_ref, submitted_by=submitted_by,
                    allow_unknown_config=allow_unknown_config,
                )
                jid = self.submit(
                    spec, allow_dirty=allow_dirty, sweep_id=sweep_id, cell_id=cid
                )
                shard_job_ids.append(jid)
                all_job_ids.append(jid)
            plan_cells.append(
                SweepCell(
                    coords=coords,
                    cell_id=cid,
                    seeds_expected=seed_set,
                    shard_seeds=shards,
                    shard_job_ids=shard_job_ids,
                    results_file=results_file,
                    seed_column=seed_column,
                    row_key=row_key_cols,
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
        allow_unknown_config: bool = False,
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
        stagger = _submit_stagger_s() if self.backend.name != "local" else 0.0
        for point in points:
            if job_ids and stagger:
                time.sleep(stagger)  # don't stampede the local SkyPilot API server
            spec = build_sweep_point_spec(
                command, point, seed=seed, resources=resources,
                code_ref=code_ref, submitted_by=submitted_by,
                allow_unknown_config=allow_unknown_config,
            )
            job_ids.append(self.submit(spec, allow_dirty=allow_dirty, sweep_id=sweep_id))
        return sweep_id, job_ids

    def sweep_plan(self, sweep_id: str) -> SweepPlan:
        """Read the persisted shard plan for a sharded sweep (P1-2)."""
        if not self.store.has_sweep_plan(sweep_id):
            raise LabError(f"no shard plan for {sweep_id!r} (not a sharded sweep?)")
        return self.store.read_sweep_plan(sweep_id)

    def aggregate_sweep(
        self,
        sweep_id: str,
        *,
        include_partial: bool = True,
        row_key: str | list[str] | None = None,
    ) -> SweepPlan:
        """Row-concatenate each cell's shard results into one per-cell table (P1-2, FR-SS-4..7).

        Idempotent pull reducer: recomputes from current shard states each call, so it is safe to run
        repeatedly as shards finish. A cell is ``complete`` iff every expected seed is present, else
        ``incomplete`` with the missing seeds named — never presents a short aggregate as complete and
        never discards recovered seeds (FR-SS-7).

        By default any *terminal* shard with a readable results file contributes — the flush-per-seed
        design means a timed-out shard's recovered rows are valid, paid-for data (field-report #2);
        their seeds are reported under ``seeds_partial`` and their rows stamped ``_shard_status``.
        ``include_partial=False`` restores succeeded-only aggregation. Running/queued shards are
        always excluded (their file is still moving under the heartbeat rsync), as are shards whose
        config was unconsumed (wrong-config rows are what the fail-closed check exists to kill).
        """
        plan = self.sweep_plan(sweep_id)
        for cell in plan.cells:
            if row_key is not None:
                # Aggregate-time declaration for plans that predate --row-key (persisted below
                # via write_sweep_plan, so later aggregates/retries inherit it).
                cell.row_key = _parse_row_key(row_key, cell.seed_column)
            shard_texts: list[tuple[str, str]] = []
            for jid in cell.shard_job_ids:
                m = self.manifest(jid)
                if m.status not in _TERMINAL_STATES:
                    continue  # still moving under the heartbeat rsync — never read mid-flight
                if not include_partial and m.status is not JobState.succeeded:
                    continue
                if m.unconsumed_config:
                    continue  # ran different config than requested — never aggregate its rows
                self.fetch_artifacts(jid)  # ensure the local copy exists (R2 fallback inside)
                rf = self.store.output_dir(jid) / cell.results_file
                if rf.exists():
                    shard_texts.append((rf.read_text(), m.status.value))
            merged, present, partial = merge_seed_rows(
                shard_texts, cell.seed_column, row_key=cell.row_key
            )
            cell.seeds_present = present
            cell.seeds_partial = partial
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
                            r2.upload_file(
                                dest, f"{sweep_id}/cells/{cell.cell_id}/{cell.results_file}"
                            )
                        except Exception as e:  # noqa: BLE001 — local aggregate stays authoritative
                            print(f"[lab] aggregate R2 mirror failed, keeping local copy: {e}")
        self.store.write_sweep_plan(plan)
        return plan

    def retry_sweep(self, sweep_id: str, *, allow_dirty: bool = True) -> SweepPlan:
        """Resubmit only the missing shards of incomplete cells, then re-aggregate (P1-2, FR-SS-7).

        A shard is missing if any of its assigned seeds is absent from the current aggregate. Fresh
        shard jobs join the same ``sweep_id``/``cell_id``; succeeded shards are never touched.

        Safe to call repeatedly: if a prior retry's job for a given seed subset is still in a
        non-terminal state (queued/running), that shard is skipped — no duplicate in-flight jobs.
        """
        plan = self.aggregate_sweep(sweep_id)  # refresh present/missing from current shard states
        for cell in plan.cells:
            if cell.status != "incomplete":
                continue
            present = set(cell.seeds_present)
            # Collect seed SETS of all currently in-flight (non-terminal) shard jobs so we can
            # skip resubmitting seeds that already have a live retry covering them. Sets (not
            # exact strings): a full-shard in-flight retry "2,3" must suppress a narrowed "3".
            in_flight_seed_sets: list[set[int]] = []
            for jid in cell.shard_job_ids:
                m = self.manifest(jid)
                if m.status not in _TERMINAL_STATES:
                    sub = m.run.resolved_config.get(plan.seed_axis_key)
                    if sub is not None:
                        try:
                            in_flight_seed_sets.append(set(parse_seeds(str(sub))))
                        except ValueError:
                            pass  # unparseable subset — can't prove coverage, don't suppress
            # inherit the original shard resources (timeout/backend/etc.) from an existing shard
            # [0] is safe: an incomplete cell always has >=1 shard (seeds_expected is non-empty)
            base_manifest = self.manifest(cell.shard_job_ids[0])
            base_resources = base_manifest.resources
            for shard in cell.shard_seeds:
                # Partial recovery (field-report #2): resubmit only the seeds still missing from
                # this shard — a timed-out shard's recovered rows are kept, not re-bought.
                shard_missing = [s for s in shard if s not in present]
                if not shard_missing:
                    continue  # this shard's seeds are already covered
                in_flight_union: set[int] = set().union(*in_flight_seed_sets) if in_flight_seed_sets else set()
                if set(shard_missing) <= in_flight_union:
                    continue  # live retries already cover these seeds (jointly) — don't duplicate
                point = {**cell.coords, plan.seed_axis_key: seeds_to_arg(shard_missing)}
                spec = build_sweep_point_spec(
                    plan.command, point, seed=shard_missing[0], resources=base_resources,
                    allow_unknown_config=base_manifest.run.allow_unknown_config,
                )
                jid = self.submit(
                    spec, allow_dirty=allow_dirty, sweep_id=sweep_id, cell_id=cell.cell_id
                )
                cell.shard_job_ids.append(jid)
        self.store.write_sweep_plan(plan)
        return self.aggregate_sweep(sweep_id)

    def _sibling_lab(self, repo: Path) -> Lab:
        """A Lab rooted at ``repo`` (e.g. an extracted bundle) over the same backend kind, sharing
        this lab's home/store — mirrors the scheduler's ``make_lab`` for confirm relaunches."""
        return Lab(
            backend=build_backend(self.backend.name, home=self.home, repo=repo),
            repo=repo,
            home=self.home,
        )

    def confirm(
        self,
        orig_id: str,
        *,
        metrics: Iterable[str] | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-12,
        wait: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Re-derive a prior result from its pinned provenance and judge whether it still holds
        (the reproducibility gate). Relaunches ``orig_id`` *fresh* from its committed commit (no
        cache), then compares the re-run's final metric(s) against the original's snapshot within
        tolerance → ``match`` / ``drift`` / ``rerun_failed``.

        Refuses outright (``LabError``) to confirm a run that did not **succeed** or that ran from a
        dirty tree — a non-succeeded or not-fully-captured run has no honest result to re-derive.
        ``metrics`` restricts which metrics are judged (default: all in the baseline).
        """
        try:
            m = self.manifest(orig_id)
        except FileNotFoundError as e:
            raise LabError(f"cannot confirm {orig_id!r}: run not found in {self.home}") from e
        if m.status is not JobState.succeeded:
            raise LabError(
                f"cannot confirm {orig_id}: its producing run is '{m.status.value}', not "
                "'succeeded' — a non-succeeded run has no result to re-derive (FR-B)"
            )
        if m.code.git_dirty:
            raise LabError(
                f"cannot confirm {orig_id}: it ran from a dirty working tree, so its code was not "
                "fully captured and can't be honestly re-derived (FR-B1)"
            )
        # Baseline: prefer the durable manifest snapshot; fall back to the original's metrics file.
        baseline = dict(m.final_metrics) or final_values(self.backend.read_metrics(orig_id))
        if not baseline:
            raise LabError(
                f"cannot confirm {orig_id}: no baseline metrics — the manifest snapshot is empty "
                "and metrics.jsonl is unavailable; nothing to compare against"
            )
        # Relaunch fresh from the pinned commit: committed tree only, never the cache.
        from lab.scheduler.bundle import create_bundle, extract_bundle  # avoid import cycle

        if not commit_exists(self.repo, m.code.git_commit):
            raise LabError(
                f"cannot confirm {orig_id}: its pinned commit {m.code.git_commit[:12]} is not in "
                f"{self.repo} — fetch it (e.g. `git fetch --all`) then retry"
            )
        bundle_root = self.home / "_confirm"
        tar, _ = create_bundle(
            self.repo, bundle_root, commit=m.code.git_commit, include_dirty=False
        )
        bundle_dir = extract_bundle(tar, bundle_root / orig_id)
        bundle_lab = self._sibling_lab(bundle_dir)
        spec = JobSpec(
            command=m.run.entrypoint_command,
            config=m.run.resolved_config,
            seed=m.run.seed,
            resources=m.resources,
            submitted_by="agent",
        )
        confirm_id = bundle_lab.submit(
            spec,
            code=CodeRef(git_commit=m.code.git_commit, git_dirty=False),
            confirms=orig_id,
        )
        result: dict[str, Any] = {"orig_id": orig_id, "confirm_id": confirm_id}
        if not wait:
            result["verdict"] = "pending"
            return result
        (rerun,) = self.wait([confirm_id], timeout=timeout)
        if rerun.status not in _TERMINAL_STATES:
            # wait gave up before the re-run finished — it's still alive (and, on a remote backend,
            # still billing until it tears down). Don't call a running job failed.
            result["verdict"] = "timed_out_waiting"
            result["rerun_status"] = rerun.status.value
            return result
        if rerun.status is not JobState.succeeded:
            result["verdict"] = "rerun_failed"
            result["rerun_status"] = rerun.status.value
            return result
        verdict, deltas = compare_final_metrics(
            baseline, rerun.final_metrics, names=metrics, rtol=rtol, atol=atol
        )
        result["verdict"] = verdict
        result["deltas"] = deltas
        result["env_drift"] = rerun.env.uv_lock_sha256 != m.env.uv_lock_sha256
        return result

    def status(self, job_id: str) -> JobState:
        return self.backend.status(job_id)

    def logs(self, job_id: str, tail: int | None = 100) -> list[str]:
        return list(self.backend.tail_logs(job_id, tail=tail))

    def metrics(
        self, job_id: str, names: Iterable[str] | None = None, since_step: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Grouped incremental metric series for a job, queryable live (FR-D2)."""
        return group_series(self.backend.read_metrics(job_id, names=names, since_step=since_step))

    def cancel(self, job_id: str) -> JobState:
        return self.backend.cancel(job_id)

    def fetch_artifacts(self, job_id: str, dest: str | None = None) -> list[ArtifactRecord]:
        out = self.store.output_dir(job_id)
        has_local = out.exists() and any(out.iterdir())
        if not has_local and r2_enabled():  # local copy gone — pull the durable copy from R2
            try:
                manifest = self.store.read_manifest(job_id)
                r2 = R2Store.from_env()
                if manifest.artifacts_uri and r2 is not None:
                    r2.download_dir(job_id, out)
            except ImportError as e:
                # R2 env configured but the `r2` extra isn't installed — the fallback is
                # best-effort recovery, never a reason to fail a local operation.
                print(f"[lab] R2 fallback unavailable ({e}); using local state only")
        return self.backend.collect_artifacts(job_id, dest or str(self.store.job_dir(job_id)))

    def manifest(self, job_id: str) -> JobManifest:
        return self.store.read_manifest(job_id)

    def list_jobs(self) -> list[JobManifest]:
        return [self.store.read_manifest(j) for j in self.store.list_job_ids()]

    def jobs_in_sweep(self, sweep_id: str) -> list[str]:
        return [j.job_id for j in self.list_jobs() if j.sweep_id == sweep_id]

    def export(
        self,
        target_id: str,
        dest: Path,
        *,
        include_logs: bool = False,
        max_file_mb: float = 32.0,
    ) -> dict[str, Any]:
        """Export a committable provenance bundle for a job or a whole sweep (field-report #5).

        ``runs/`` is git-ignored and lives only in the lab repo — the analysis repo (where the
        paper is written) has no supported route to the manifests and result tables behind a
        figure. This writes exactly the small, durable subset that belongs in version control:
        per job ``manifest.json``, ``resolved_config.json``, ``code_diff.tar.gz`` when present,
        and the figure/table artifacts under a size cap (blobs like ``.npz``/checkpoints are
        excluded — recorded in the index under ``skipped``, never silently dropped). Sweeps also
        bundle ``plan.json`` + the per-cell aggregates. ``index.json`` ties every file to a
        commit, seed, state, and spend. Idempotent; returns the index.
        """
        dest = Path(dest)
        if target_id.startswith("sweep-"):
            kind = "sweep"
            job_ids = self.jobs_in_sweep(target_id)
            if not job_ids:
                raise LabError(f"sweep {target_id!r} matched no jobs in {self.home}")
        else:
            kind = "job"
            if not self.store.manifest_path(target_id).exists():
                raise LabError(f"unknown job or sweep id {target_id!r}")
            job_ids = [target_id]

        max_bytes = int(max_file_mb * 1024 * 1024)
        jobs_index: list[dict[str, Any]] = []
        for jid in sorted(job_ids):
            m = self.manifest(jid)
            try:
                self.fetch_artifacts(jid)  # R2 fallback for outputs that only live durably
            except Exception as e:  # noqa: BLE001 — export what exists locally
                print(f"[lab] export: fetch_artifacts({jid}) failed, exporting local state: {e}")
            jdir = dest / jid
            jdir.mkdir(parents=True, exist_ok=True)
            (jdir / "manifest.json").write_text(self.store.manifest_path(jid).read_text())
            (jdir / "resolved_config.json").write_text(
                json.dumps(m.run.resolved_config, indent=2, sort_keys=True, default=str)
            )
            files = ["manifest.json", "resolved_config.json"]
            diff = self.store.job_dir(jid) / "code_diff.tar.gz"
            if diff.exists():
                (jdir / "code_diff.tar.gz").write_bytes(diff.read_bytes())
                files.append("code_diff.tar.gz")
            if include_logs and self.store.logs_path(jid).exists():
                (jdir / "logs.txt").write_text(
                    self.store.logs_path(jid).read_text(errors="replace")
                )
                files.append("logs.txt")
            skipped: list[dict[str, str]] = []
            out = self.store.output_dir(jid)
            if out.exists():
                for f in sorted(p for p in out.rglob("*") if p.is_file()):
                    rel = f.relative_to(out).as_posix()
                    if rel.startswith("."):
                        continue
                    atype = infer_artifact_type(f.name)
                    if atype not in ("figure", "table"):
                        skipped.append({"file": rel, "reason": f"type {atype} (blob)"})
                        continue
                    size = f.stat().st_size
                    if size > max_bytes:
                        skipped.append(
                            {"file": rel, "reason": f"size {size}B > {max_file_mb}MB cap"}
                        )
                        continue
                    target = jdir / "output" / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(f.read_bytes())
                    files.append(f"output/{rel}")
            jobs_index.append(
                {
                    "job_id": jid,
                    "state": m.status.value,
                    "git_commit": m.code.git_commit,
                    "git_dirty": m.code.git_dirty,
                    "diff_ref": m.code.diff_ref,
                    "seed": m.run.seed,
                    "sweep_id": m.sweep_id,
                    "cell_id": m.cell_id,
                    "end_reason": m.end_reason,
                    "actual_usd": m.cost.actual_usd if m.cost else None,
                    "files": files,
                    "skipped": skipped,
                }
            )

        sweep_block: dict[str, Any] | None = None
        if kind == "sweep" and self.store.has_sweep_plan(target_id):
            plan = self.sweep_plan(target_id)
            (dest / "plan.json").write_text(self.store.sweep_plan_path(target_id).read_text())
            for cell in plan.cells:
                agg = Path(cell.aggregate_ref)
                if agg.exists():
                    target = dest / "cells" / cell.cell_id / agg.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(agg.read_bytes())
            sweep_block = {"sweep_id": target_id, "plan_exported": True, **plan.view()}

        index: dict[str, Any] = {
            "exported_at": now().isoformat(),
            "source_id": target_id,
            "kind": kind,
            "jobs": jobs_index,
            "sweep": sweep_block,
        }
        atomic_write_text(dest / "index.json", json.dumps(index, indent=2, default=str))
        return index

    def sweep_summary(self, sweep_id: str) -> dict[str, Any]:
        """Aggregate a sweep's outcomes for trustworthy reporting (preemptions, fallback, spend)."""
        ms = [m for m in self.list_jobs() if m.sweep_id == sweep_id]

        def spend(m: JobManifest) -> float:
            return m.cost.actual_usd if m.cost and m.cost.actual_usd else 0.0

        return {
            "sweep_id": sweep_id,
            "total": len(ms),
            "succeeded": int(sum(int(m.status is JobState.succeeded) for m in ms)),
            "preempted": int(sum(int(m.status is JobState.preempted) for m in ms)),
            "failed": int(sum(int(m.status is JobState.failed) for m in ms)),
            "fell_back_to_on_demand": int(
                sum(int(m.resources.use_spot and m.backend.launched_spot is False) for m in ms)
            ),
            "total_usd": round(sum(spend(m) for m in ms), 6),
            "per_point": {
                m.job_id: {
                    "state": m.status.value,
                    "usd": round(spend(m), 6),
                    "launched_spot": m.backend.launched_spot,
                }
                for m in ms
            },
        }

    def _sky_status_orphans(self, running_clusters: set[str]) -> list[str]:
        """Cloud-agnostic orphan pass: ``lab-*`` clusters SkyPilot still tracks/that are still up
        but are NOT tied to a running local job. Covers DO/GCP (and Vast) via SkyPilot's own state,
        complementing the Vast-direct scan. Raises :class:`LabError` if the status query fails."""
        import sky

        try:
            recs = sky.get(sky.status(refresh=sky.StatusRefreshMode.AUTO))  # 0.12: RequestId -> list
        except Exception as e:  # noqa: BLE001
            raise LabError(f"could not query SkyPilot cluster status: {e}") from e
        orphans: list[str] = []
        for rec in recs or []:
            name = rec.get("name") if isinstance(rec, dict) else getattr(rec, "name", None)
            if not name or not str(name).startswith("lab-") or name in running_clusters:
                continue
            orphans.append(name)
        return orphans

    def reconcile(self, *, apply: bool = False) -> dict[str, Any]:
        """Cross-check Vast.ai rentals against the local job DB (FR-C2 leak detection).

        Returns a structured report of:

        - ``orphans``: Vast.ai rentals whose label looks like a lab cluster (``lab-*``) but does
          NOT match any **running** local job — these are very likely leaked rentals.
        - ``ghosts``: running local jobs whose cluster name does not appear in any active Vast
          rental label — the supervisor probably died before recording terminal state.
        - ``destroyed``: Vast instance IDs we actually destroyed (only when ``apply=True``).

        With ``apply=True``, each orphan is destroyed via the vastai-sdk directly (bypassing
        SkyPilot's local registry — which may have already lost track of the rental). Without
        ``apply``, it's a dry run; no rentals are touched.

        A missing vastai-sdk (a DO/GCP-only install) skips the Vast-direct pass — the report
        carries ``vast_pass`` explaining why — while the cloud-agnostic ``sky.status`` pass and
        the DO volume pass still run. Any other listing failure raises :class:`LabError`: when
        the SDK *is* present there is no safe degraded mode for a leak-detection command.

        A non-terminal skypilot job whose supervisor pid is dead (past a short grace window) does
        NOT protect its cluster: the canonical leak is a supervisor crash that freezes the
        manifest at ``running`` — counting that cluster as healthy would hide the still-billing
        box from every pass. Such jobs are reported under ``unsupervised``.
        """
        from lab.backends.skypilot import (  # local import: skypilot is an optional extra
            _alive,
            _instance_label,
            cluster_name_for,
            list_vast_instances,
        )

        vast_pass = "ran"
        try:
            instances = list_vast_instances()
        except ImportError:
            instances = []
            vast_pass = "skipped (vastai-sdk not installed)"
        except Exception as e:  # noqa: BLE001
            raise LabError(f"could not list Vast.ai rentals: {e}") from e

        unsupervised: list[dict[str, str]] = []
        running_clusters: dict[str, str] = {}
        for j in self.list_jobs():
            if j.status in _TERMINAL_STATES:
                continue
            cluster = cluster_name_for(j.job_id)
            if j.backend.provisioner == "skypilot":
                age = (now() - (j.started_at or j.created_at)).total_seconds()
                pid = self.store.read_runtime(j.job_id).get("runner_pid")
                if age > UNSUPERVISED_GRACE_S and not _alive(pid):
                    unsupervised.append({"job_id": j.job_id, "cluster": cluster})
                    continue  # dead supervisor -> the cluster is NOT protected
            running_clusters[cluster] = j.job_id

        orphans: list[dict[str, Any]] = []
        matched_clusters: set[str] = set()
        for inst in instances:
            label = _instance_label(inst)
            if "lab-" not in label:
                continue  # not ours — leave it alone
            matched = next((c for c in running_clusters if c.lower() in label), None)
            if matched is not None:
                matched_clusters.add(matched)
                continue
            orphans.append({"id": inst.get("id"), "label": label})

        destroyed: list[int] = []
        if apply and orphans:
            from lab.backends.skypilot import _get_vast_client

            client = _get_vast_client()
            for orph in orphans:
                inst_id = orph["id"]
                if inst_id is None:
                    continue
                try:
                    client.destroy_instance(id=int(inst_id))
                    destroyed.append(int(inst_id))
                except Exception as e:  # noqa: BLE001
                    print(f"[lab] reconcile destroy {inst_id} failed: {e}")

        ghosts = sorted(running_clusters.keys() - matched_clusters)

        sky_orphans = self._sky_status_orphans(set(running_clusters))
        sky_destroyed: list[str] = []
        if apply and sky_orphans:
            import sky

            for cl in sky_orphans:
                try:
                    sky.get(sky.down(cl))
                    sky_destroyed.append(cl)
                except Exception as e:  # noqa: BLE001
                    print(f"[lab] reconcile sky.down {cl} failed: {e}")

        # DO block-volume pass (best-effort): sky.down deletes the volume with its droplet, but a
        # partial teardown can leave a detached `lab-*` volume that the instance passes above can't
        # see (its droplet is gone) — yet it keeps billing. Skipped silently when DO isn't
        # configured (no doctl), since not every account uses DigitalOcean.
        from lab.backends.skypilot import do_volume_orphans as _find_do_volume_orphans
        from lab.backends.skypilot import list_do_volumes

        do_volume_orphans: list[dict[str, Any]] = []
        do_volumes_destroyed: list[Any] = []
        try:
            volumes = list_do_volumes()
        except Exception:  # noqa: BLE001 — DO not configured/unavailable: skip the volume pass
            volumes = None
        if volumes is not None:
            do_volume_orphans = _find_do_volume_orphans(volumes, set(running_clusters))
            if apply and do_volume_orphans:
                from lab.backends.skypilot import _get_do_client

                client = _get_do_client()
                for vol in do_volume_orphans:
                    vol_id = vol.get("id")
                    if vol_id is None:
                        continue
                    try:
                        client.volumes.delete(volume_id=vol_id)
                        do_volumes_destroyed.append(vol_id)
                    except Exception as e:  # noqa: BLE001
                        print(f"[lab] reconcile delete volume {vol_id} failed: {e}")

        # GCP pass (best-effort): out-of-band instance + unattached-disk sweep via the compute
        # API — `sky.status` only sees clusters SkyPilot still tracks, and a GCP persistent disk
        # that outlives its VM keeps billing (same failure mode as the DO volume leak). Skipped
        # silently when GCP isn't configured (no ADC), since not every account uses GCP.
        from lab.backends.skypilot import GcpNotConfigured, delete_gcp_disk, delete_gcp_instance
        from lab.backends.skypilot import gcp_disk_orphans as _find_gcp_disk_orphans
        from lab.backends.skypilot import gcp_instance_orphans as _find_gcp_instance_orphans
        from lab.backends.skypilot import list_gcp_disks, list_gcp_instances

        def _gcp_list(what: str, lister: Callable[[], list[dict[str, Any]]]) -> tuple[
            list[dict[str, Any]] | None, str
        ]:
            """Run one GCP listing, distinguishing "GCP isn't set up here" (skip, and say so in
            the report) from "the API failed" (raise — a leak pass that swallows an API error
            reports clean while blind). The two passes are independent: an instance-API hiccup
            must not also hide leaked disks, which are the slow, quiet leak."""
            try:
                return lister(), "ran"
            except GcpNotConfigured as e:
                return None, f"skipped ({e})"
            except Exception as e:  # noqa: BLE001
                raise LabError(f"could not list GCP {what}: {e}") from e

        gcp_orphans: list[dict[str, Any]] = []
        gcp_destroyed: list[str] = []
        gcp_disk_orphans: list[dict[str, Any]] = []
        gcp_disks_destroyed: list[str] = []

        gcp_instances, gcp_pass = _gcp_list("instances", list_gcp_instances)
        if gcp_instances is not None:
            gcp_orphans = _find_gcp_instance_orphans(gcp_instances, set(running_clusters))
            if apply and gcp_orphans:
                for inst in gcp_orphans:
                    try:
                        delete_gcp_instance(str(inst["name"]), str(inst["zone"]))
                        gcp_destroyed.append(str(inst["name"]))
                    except Exception as e:  # noqa: BLE001
                        print(f"[lab] reconcile delete gcp instance {inst['name']} failed: {e}")

        gcp_disks, gcp_disk_pass = _gcp_list("disks", list_gcp_disks)
        if gcp_disks is not None:
            gcp_disk_orphans = _find_gcp_disk_orphans(gcp_disks, set(running_clusters))
            if apply and gcp_disk_orphans:
                for disk in gcp_disk_orphans:
                    try:
                        delete_gcp_disk(str(disk["name"]), str(disk["zone"]))
                        gcp_disks_destroyed.append(str(disk["name"]))
                    except Exception as e:  # noqa: BLE001
                        print(f"[lab] reconcile delete gcp disk {disk['name']} failed: {e}")

        return {
            "vast_pass": vast_pass,
            "gcp_pass": gcp_pass,
            "gcp_disk_pass": gcp_disk_pass,
            "instances_total": len(instances),
            "unsupervised": unsupervised,  # running manifests with a dead supervisor (FR-C2)
            "orphans": orphans,
            "destroyed": destroyed,
            "ghosts": ghosts,
            "sky_orphans": sky_orphans,
            "sky_destroyed": sky_destroyed,
            "do_volume_orphans": do_volume_orphans,
            "do_volumes_destroyed": do_volumes_destroyed,
            "gcp_orphans": gcp_orphans,
            "gcp_destroyed": gcp_destroyed,
            "gcp_disk_orphans": gcp_disk_orphans,
            "gcp_disks_destroyed": gcp_disks_destroyed,
            "applied": apply,
        }

    def _settle_teardown(
        self, manifests: list[JobManifest], *, interval: float, attempts: int = 3
    ) -> list[JobManifest]:
        """Re-read manifests briefly so a teardown_status that's merely lagging (a job reports
        terminal a tick before its teardown is recorded) settles to its real value before we
        classify clean vs. leaked vs. unconfirmed. Only re-reads while some remote job still
        shows a null teardown."""

        def _unsettled(ms: list[JobManifest]) -> bool:
            return any(
                m.status in _TERMINAL_STATES  # only terminal jobs can settle; pending never will
                and m.backend.provisioner != "local"
                and m.teardown_status is None
                for m in ms
            )

        for _ in range(attempts):
            if not _unsettled(manifests):
                break
            time.sleep(min(interval, 5.0))
            manifests = [self.manifest(m.job_id) for m in manifests]
        return manifests

    def _wait_summary_dict(
        self, manifests: list[JobManifest], *, failed_fast: bool
    ) -> dict[str, Any]:
        """The FR-C2 verdict as data (one shape for CLI + MCP + done-file snapshots)."""
        all_terminal = all(m.status in _TERMINAL_STATES for m in manifests)
        pending = [m.job_id for m in manifests if m.status not in _TERMINAL_STATES]
        if failed_fast:
            # Offender-first ordering so a watcher's first glance lands on what died.
            manifests = sorted(
                manifests,
                key=lambda m: m.status not in (JobState.failed, JobState.timed_out),
            )
        teardown_leaks = [m.job_id for m in manifests if m.teardown_status == "failed"]
        teardown_unconfirmed = [
            m.job_id
            for m in manifests
            if m.status in _TERMINAL_STATES
            and m.backend.provisioner != "local"
            and m.teardown_status is None
        ]
        return {
            "all_terminal": all_terminal,
            "failed_fast": failed_fast,
            "pending": pending,  # still running — and, for remote jobs, still billing
            "teardown_leaks": teardown_leaks,
            "teardown_unconfirmed": teardown_unconfirmed,
            "jobs": [
                {
                    "job_id": m.job_id,
                    "state": m.status.value,
                    "exit_code": m.exit_code,
                    "teardown_status": m.teardown_status,
                }
                for m in manifests
            ],
        }

    def wait_summary(
        self,
        job_ids: list[str],
        *,
        interval: float = 10.0,
        timeout: float | None = None,
        fail_fast: bool = False,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """:meth:`wait`, then classify the FR-C2 verdict as data — shared by CLI ``lab wait``
        and the MCP ``wait`` tool so both surfaces expose the same leak signal.

        ``teardown_leaks`` non-empty means a paid machine may still be running;
        ``teardown_unconfirmed`` means a remote job's teardown never recorded either way (a null
        must not masquerade as clean — run ``lab reconcile`` to be sure). ``on_update`` receives
        a fresh summary snapshot after each job's terminal transition and once more with the
        final summary — the incremental done-file feed (field-report #3). ``fail_fast`` returns
        as soon as any job is failed/timed_out; no surviving job is ever cancelled."""

        def _on_terminal(_m: JobManifest) -> None:
            if on_update is not None:
                on_update(
                    self._wait_summary_dict(
                        [self.manifest(j) for j in job_ids], failed_fast=False
                    )
                )

        manifests = self.wait(
            job_ids, interval=interval, timeout=timeout, fail_fast=fail_fast,
            on_terminal=_on_terminal,
        )
        all_terminal = all(m.status in _TERMINAL_STATES for m in manifests)
        failed_fast = fail_fast and not all_terminal and any(
            m.status in (JobState.failed, JobState.timed_out) for m in manifests
        )
        if all_terminal or failed_fast:
            # Settle on the fail-fast path too: the offender's teardown_status may be merely
            # lagging, and a null must not hide a real leak verdict behind "unconfirmed".
            manifests = self._settle_teardown(manifests, interval=interval)
        summary = self._wait_summary_dict(manifests, failed_fast=failed_fast)
        if on_update is not None:
            try:
                on_update(summary)
            except Exception as e:  # noqa: BLE001 — a watcher crash must not eat the verdict
                print(f"[lab] wait on_update callback failed: {e}")
        return summary

    def wait(
        self,
        job_ids: list[str],
        *,
        interval: float = 10.0,
        timeout: float | None = None,
        fail_fast: bool = False,
        on_terminal: Callable[[JobManifest], None] | None = None,
    ) -> list[JobManifest]:
        """Block until every job reaches a terminal state (or ``timeout``), then return manifests.

        Meant to run as a Claude Code background task: its completion is the push signal, so the
        agent need not poll (FR-G1). Uses cheap status reads (FR-G2); status reads the store, so
        this works for jobs of any backend.

        ``on_terminal`` fires once per job on its first observed terminal transition (errors are
        logged, never fatal). ``fail_fast`` returns immediately when any job reaches
        ``failed``/``timed_out`` — preempted (retryable) and cancelled (operator-initiated) do
        not trigger it. ``wait`` never mutates jobs: nothing is cancelled on the way out.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        pending = list(job_ids)
        while pending:
            still_pending: list[str] = []
            tripwire = False
            for j in pending:
                state = self.status(j)
                if state not in _TERMINAL_STATES:
                    still_pending.append(j)
                    continue
                if on_terminal is not None:
                    try:
                        on_terminal(self.manifest(j))
                    except Exception as e:  # noqa: BLE001 — callback must never abort the wait
                        print(f"[lab] wait on_terminal callback failed: {e}")
                if fail_fast and state in (JobState.failed, JobState.timed_out):
                    tripwire = True
            pending = still_pending
            if tripwire or not pending:
                break
            if deadline is None:
                time.sleep(max(0.05, interval))  # guard against a busy-loop on interval<=0
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break  # timed out before all jobs finished
                time.sleep(max(0.05, min(interval, remaining)))  # never overrun the deadline
        return [self.manifest(j) for j in job_ids]


def build_backend(name: str, *, home: Path, repo: Path) -> Backend:
    """The single name->backend mapping. Both Lab construction paths (``default_lab``,
    ``Lab._sibling_lab``) and the scheduler (``Scheduler.make_lab``) route through here, so a new
    backend is wired in one place instead of three. Unknown names fall back to ``local``.
    """
    if name in ("skypilot", "cpu"):
        from lab.backends.skypilot import SkyPilotBackend  # optional extra; import lazily

        return SkyPilotBackend(home=home, repo=repo)
    return LocalBackend(home=home, repo=repo)


def job_status_view(home: Path, repo: Path, job_id: str) -> dict[str, Any]:
    """One status shape for both shells (FR-A2/FR-I2/FR-C2), with a mirrored-manifest fallback.

    Reads the local manifest and the live backend status (which finalizes dead-supervisor jobs);
    a job absent from local ``runs/`` falls back to the scheduler queue's mirrored manifest
    (spec §4.3) so deferred jobs are observable from every surface, not write-only. Raises
    :class:`FileNotFoundError` when the job exists in neither place.
    """
    store = JobStore(home)
    try:
        m = store.read_manifest(job_id)
    except FileNotFoundError:
        from lab.scheduler.queue import default_queue  # local import: avoids a module cycle

        mirrored = default_queue().read_mirrored(job_id)
        if mirrored is None:
            raise
        return _status_fields(mirrored, state=mirrored.status.value, mirrored=True)
    lab = Lab(backend=build_backend(m.backend.provisioner, home=home, repo=repo), repo=repo, home=home)
    state = lab.status(job_id)
    m = store.read_manifest(job_id)  # re-read: status may have just finalized/torn down the job
    return _status_fields(
        m, state=state.value, mirrored=False, logs_path=store.logs_path(job_id)
    )


def _status_fields(
    m: JobManifest, *, state: str, mirrored: bool, logs_path: Path | None = None
) -> dict[str, Any]:
    last_line, last_at = tail_last_line(logs_path) if logs_path is not None else (None, None)
    return {
        "job_id": m.job_id,
        "state": state,
        "started_at": m.started_at,
        "ended_at": m.ended_at,
        "exit_code": m.exit_code,
        "end_reason": m.end_reason,
        "cost": m.cost.model_dump() if m.cost else None,
        # Burn-rate visibility mid-run (field-report #7): a derived estimate, not a meter.
        "estimated_running_usd": (
            actual_cost(m.cost.hourly_usd, duration_seconds(m.started_at, now()))
            if state == "running" and m.cost is not None and m.started_at is not None
            else None
        ),
        "teardown_status": m.teardown_status,  # FR-C2 — "failed" means a box may still bill
        # Where it actually landed. GCP prices and exhausts per zone, so "which zone" is the
        # difference between a $0.034/hr job and a $0.12/hr one.
        "placement": {
            "cloud": m.resources.cloud or "vast",
            "machine_type": m.backend.machine_type,
            "region": m.backend.region,
            "zone": m.backend.zone,
            "spot": m.backend.launched_spot,
        },
        # spot_fallback defaults on, so `--spot` can land on-demand at ~5x the price the user was
        # budgeting for. `launched_spot` recorded that all along; nothing ever surfaced it.
        "spot_downgraded": bool(m.resources.use_spot and m.backend.launched_spot is False),
        "sweep_id": m.sweep_id,
        # Progressing-vs-wedged signal for long runs that don't log metrics.
        "last_log_line": last_line,
        "last_log_at": last_at,
        "code": {
            "git_commit": m.code.git_commit,
            "git_dirty": m.code.git_dirty,
            "diff_ref": m.code.diff_ref,
        },
        "mirrored": mirrored,  # True = read from the scheduler mirror; may be a tick stale
    }


def default_lab(home: Path | None = None, backend: str = "local") -> Lab:
    """Construct a Lab rooted at the current git repo, over the named backend
    (``local`` or ``skypilot``). Shared by the CLI and MCP so both drive the identical core.
    """
    repo = repo_root()
    resolved_home = Path(home) if home else repo / "runs"
    return Lab(backend=build_backend(backend, home=resolved_home, repo=repo), repo=repo, home=resolved_home)
