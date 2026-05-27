"""Lab core — the single library both the CLI and the MCP server are thin shells over
(NFR-3, FR-F2). See research/10-architecture.md.

``Lab.submit`` resolves a :class:`~lab.models.JobSpec` into a :class:`~lab.models.JobManifest`
(pin commit, hash uv.lock, resolve seed), persists it via the store, then dispatches to the
chosen :class:`~lab.backends.base.Backend`.
"""

from __future__ import annotations

import itertools
import platform
import shlex
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lab._util import now
from lab.backends.base import Backend
from lab.backends.local import LocalBackend
from lab.manifest import current_commit, is_dirty, repo_root, uv_lock_sha256
from lab.metrics import group_series
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
)
from lab.store import JobStore


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


class Lab:
    def __init__(self, backend: Backend, repo: Path, home: Path) -> None:
        self.backend = backend
        self.repo = Path(repo)
        self.home = Path(home)
        self.store = JobStore(self.home)

    def submit(self, spec: JobSpec, *, allow_dirty: bool = True, sweep_id: str | None = None) -> str:
        """Build + persist the manifest, then launch via the backend (FR-A1, FR-B)."""
        dirty = is_dirty(self.repo)
        if dirty and not allow_dirty:
            raise LabError("working tree is dirty; commit or pass allow_dirty=True (FR-B1)")
        seed = spec.seed if spec.seed is not None else 0  # explicit + recorded (FR-B4)
        job_id = _new_job_id()
        manifest = JobManifest(
            job_id=job_id,
            sweep_id=sweep_id,
            created_at=now(),
            submitted_by=spec.submitted_by,
            code=CodeRef(git_commit=current_commit(self.repo), git_dirty=dirty),
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
    ) -> tuple[str, list[str]]:
        """Submit one job per grid point under a shared sweep_id (FR-A5).

        Each point's params are appended to the command as **shell-quoted** ``key=value`` overrides
        (injection-safe) and recorded in the job's ``resolved_config``; jobs stay independently
        monitorable by ``job_id``. A ``seed`` key in the grid sets each job's seed (varying
        ``$LAB_SEED`` per point). Refuses to fan out beyond ``max_jobs`` (cost-safety).
        """
        points = expand_grid(grid)
        if len(points) > max_jobs:
            raise LabError(
                f"sweep would submit {len(points)} jobs (> max_jobs={max_jobs}); "
                "narrow the grid or raise max_jobs"
            )
        sweep_id = f"sweep-{_new_job_id()}"
        job_ids: list[str] = []
        for point in points:
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
            spec = JobSpec(
                code_ref=code_ref,
                command=full_command,
                seed=job_seed,
                config=point,
                resources=resources or ResourceRequest(),
                submitted_by=submitted_by,  # type: ignore[arg-type]
            )
            job_ids.append(self.submit(spec, allow_dirty=allow_dirty, sweep_id=sweep_id))
        return sweep_id, job_ids

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


def default_lab(home: Path | None = None, backend: str = "local") -> Lab:
    """Construct a Lab rooted at the current git repo, over the named backend
    (``local`` or ``skypilot``). Shared by the CLI and MCP so both drive the identical core.
    """
    repo = repo_root()
    resolved_home = Path(home) if home else repo / "runs"
    be: Backend
    if backend == "skypilot":
        from lab.backends.skypilot import SkyPilotBackend

        be = SkyPilotBackend(home=resolved_home, repo=repo)
    else:
        be = LocalBackend(home=resolved_home, repo=repo)
    return Lab(backend=be, repo=repo, home=resolved_home)
