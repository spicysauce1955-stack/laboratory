"""The backend interface every provisioner implements (architecture §10)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from lab.models import ArtifactRecord, JobManifest, JobState


@runtime_checkable
class Backend(Protocol):
    """One execution backend. ``status``/``tail_logs`` must be cheap (NFR-3, FR-G2)."""

    name: str

    def submit(self, manifest: JobManifest) -> str:
        """Launch the job **without blocking**; return a backend job handle (FR-A1)."""
        ...

    def status(self, job_id: str) -> JobState:
        """Current lifecycle state (FR-A2)."""
        ...

    def tail_logs(
        self, job_id: str, tail: int | None = None, follow: bool = False
    ) -> Iterable[str]:
        """Stream/tail captured stdout+stderr (FR-D1)."""
        ...

    def cancel(self, job_id: str) -> JobState:
        """Cancel a queued/running job **and** tear down its machine (FR-A3, FR-C2)."""
        ...

    def collect_artifacts(self, job_id: str, dest: str) -> list[ArtifactRecord]:
        """Pull the run's outputs into ``dest`` (``runs/<job_id>/``) (FR-E1/E2)."""
        ...

    def read_metrics(
        self, job_id: str, names: Iterable[str] | None = None, since_step: int | None = None
    ) -> list[dict[str, Any]]:
        """Read a job's incremental metric points, live (FR-D2)."""
        ...
