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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lab._util import now
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


def build_sweep_point_spec(
    command: str,
    point: dict[str, Any],
    *,
    seed: int | None,
    resources: ResourceRequest | None = None,
    code_ref: str = "HEAD",
    submitted_by: str = "agent",
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
    )


CPU_DEFAULT_CLOUD = "do"
CPU_DEFAULT_VCPUS = 8


def resolve_backend_profile(
    backend: str, resources: ResourceRequest
) -> tuple[str, ResourceRequest]:
    """Resolve the ``cpu`` convenience backend into (provisioner_name, resources).

    ``cpu`` is sugar for the SkyPilot provisioner on a cheap CPU cloud (DigitalOcean): it clears
    accelerators, defaults to ``CPU_DEFAULT_VCPUS`` vCPUs, and disables spot (DO has none). Other
    backends pass through unchanged (identity), so the CLI and MCP stay thin shells. Pure; no I/O.
    """
    if backend != "cpu":
        return backend, resources
    if resources.accelerators:
        raise LabError("--backend cpu provisions a CPU-only box; drop --accelerators")
    return "skypilot", resources.model_copy(
        update={
            "cloud": CPU_DEFAULT_CLOUD,
            "cpus": resources.cpus or CPU_DEFAULT_VCPUS,
            "use_spot": False,
            "spot_fallback": False,
        }
    )


class Lab:
    def __init__(self, backend: Backend, repo: Path, home: Path) -> None:
        self.backend = backend
        self.repo = Path(repo)
        self.home = Path(home)
        self.store = JobStore(self.home)

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
    ) -> str:
        """Build + persist the manifest, then launch via the backend (FR-A1, FR-B).

        ``code`` overrides git introspection — used by the scheduler, which submits from an
        extracted bundle (not a git repo) with provenance captured at registration time.
        """
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
            coords = {k: str(v) for k, v in cell.items()}
            cid = cell_id_for(coords)
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
                    coords=coords,
                    cell_id=cid,
                    seeds_expected=seed_set,
                    shard_seeds=shards,
                    shard_job_ids=shard_job_ids,
                    results_file=results_file,
                    seed_column=seed_column,
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
            # Collect seed-subset strings of all currently in-flight (non-terminal) shard jobs so
            # we can skip resubmitting a shard that already has a live retry running.
            in_flight_subsets: set[str] = set()
            for jid in cell.shard_job_ids:
                m = self.manifest(jid)
                if m.status not in _TERMINAL_STATES:
                    sub = m.run.resolved_config.get(plan.seed_axis_key)
                    if sub is not None:
                        in_flight_subsets.add(str(sub))
            # inherit the original shard resources (timeout/backend/etc.) from an existing shard
            base_resources = self.manifest(cell.shard_job_ids[0]).resources
            for shard in cell.shard_seeds:
                if all(s in present for s in shard):
                    continue  # this shard's seeds are already covered
                if seeds_to_arg(shard) in in_flight_subsets:
                    continue  # a prior retry for this exact subset is still running — don't duplicate
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
            manifest = self.store.read_manifest(job_id)
            r2 = R2Store.from_env()
            if manifest.artifacts_uri and r2 is not None:
                r2.download_dir(job_id, out)
        return self.backend.collect_artifacts(job_id, dest or str(self.store.job_dir(job_id)))

    def manifest(self, job_id: str) -> JobManifest:
        return self.store.read_manifest(job_id)

    def list_jobs(self) -> list[JobManifest]:
        return [self.store.read_manifest(j) for j in self.store.list_job_ids()]

    def jobs_in_sweep(self, sweep_id: str) -> list[str]:
        return [j.job_id for j in self.list_jobs() if j.sweep_id == sweep_id]

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

        Raises :class:`LabError` if vastai-sdk is unavailable or the listing call fails — there is
        no safe degraded mode for a leak-detection command.
        """
        from lab.backends.skypilot import (  # local import: skypilot is an optional extra
            _instance_label,
            cluster_name_for,
            list_vast_instances,
        )

        try:
            instances = list_vast_instances()
        except ImportError as e:
            raise LabError(
                "vastai-sdk not installed; run `uv sync --extra skypilot` then retry"
            ) from e
        except Exception as e:  # noqa: BLE001
            raise LabError(f"could not list Vast.ai rentals: {e}") from e

        running_clusters = {
            cluster_name_for(j.job_id): j.job_id
            for j in self.list_jobs()
            if j.status not in _TERMINAL_STATES
        }

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

        return {
            "instances_total": len(instances),
            "orphans": orphans,
            "destroyed": destroyed,
            "ghosts": ghosts,
            "sky_orphans": sky_orphans,
            "sky_destroyed": sky_destroyed,
            "applied": apply,
        }

    def wait(
        self, job_ids: list[str], *, interval: float = 10.0, timeout: float | None = None
    ) -> list[JobManifest]:
        """Block until every job reaches a terminal state (or ``timeout``), then return manifests.

        Meant to run as a Claude Code background task: its completion is the push signal, so the
        agent need not poll (FR-G1). Uses cheap status reads (FR-G2); status reads the store, so
        this works for jobs of any backend.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        pending = list(job_ids)
        while pending:
            pending = [j for j in pending if self.status(j) not in _TERMINAL_STATES]
            if not pending:
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


def default_lab(home: Path | None = None, backend: str = "local") -> Lab:
    """Construct a Lab rooted at the current git repo, over the named backend
    (``local`` or ``skypilot``). Shared by the CLI and MCP so both drive the identical core.
    """
    repo = repo_root()
    resolved_home = Path(home) if home else repo / "runs"
    return Lab(backend=build_backend(backend, home=resolved_home, repo=repo), repo=repo, home=resolved_home)
