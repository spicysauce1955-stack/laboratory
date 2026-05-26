"""Lab core — the single library both the CLI and the MCP server are thin shells over
(NFR-3, FR-F2). See research/10-architecture.md.

``Lab.submit`` resolves a :class:`~lab.models.JobSpec` into a :class:`~lab.models.JobManifest`
(pin commit, hash uv.lock, resolve seed), persists it via the store, then dispatches to the
chosen :class:`~lab.backends.base.Backend`.
"""

from __future__ import annotations

import platform
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lab._util import now
from lab.backends.base import Backend
from lab.backends.local import LocalBackend
from lab.manifest import current_commit, is_dirty, repo_root, uv_lock_sha256
from lab.metrics import group_series
from lab.models import (
    ArtifactRecord,
    BackendInfo,
    CodeRef,
    EnvInfo,
    JobManifest,
    JobSpec,
    JobState,
    RunSpec,
)
from lab.store import JobStore


class LabError(RuntimeError):
    """Fail-loud lab error (FR-F3)."""


def _new_job_id() -> str:
    return f"{now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


class Lab:
    def __init__(self, backend: Backend, repo: Path, home: Path) -> None:
        self.backend = backend
        self.repo = Path(repo)
        self.home = Path(home)
        self.store = JobStore(self.home)

    def submit(self, spec: JobSpec, *, allow_dirty: bool = True) -> str:
        """Build + persist the manifest, then launch via the backend (FR-A1, FR-B)."""
        dirty = is_dirty(self.repo)
        if dirty and not allow_dirty:
            raise LabError("working tree is dirty; commit or pass allow_dirty=True (FR-B1)")
        seed = spec.seed if spec.seed is not None else 0  # explicit + recorded (FR-B4)
        job_id = _new_job_id()
        manifest = JobManifest(
            job_id=job_id,
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
        return self.backend.collect_artifacts(job_id, dest or str(self.store.job_dir(job_id)))

    def manifest(self, job_id: str) -> JobManifest:
        return self.store.read_manifest(job_id)

    def list_jobs(self) -> list[JobManifest]:
        return [self.store.read_manifest(j) for j in self.store.list_job_ids()]


def default_lab(home: Path | None = None) -> Lab:
    """Construct a Lab over the local backend, rooted at the current git repo.

    Shared by the CLI and the MCP server so both drive the identical core.
    """
    repo = repo_root()
    resolved_home = Path(home) if home else repo / "runs"
    return Lab(backend=LocalBackend(home=resolved_home, repo=repo), repo=repo, home=resolved_home)
