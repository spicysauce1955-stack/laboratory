"""Lab core — the single library both the CLI and the MCP server are thin shells over
(NFR-3, FR-F2). See research/10-architecture.md.

``Lab.submit`` resolves a :class:`~lab.models.JobSpec` into a :class:`~lab.models.JobManifest`
(pin commit, hash uv.lock, resolve config+seed), writes ``runs/<job_id>/manifest.json``, and
dispatches to the chosen :class:`~lab.backends.base.Backend`.
"""

from __future__ import annotations

from lab.backends.base import Backend
from lab.models import JobSpec


class Lab:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def submit(self, spec: JobSpec) -> str:
        # TODO(P0 build order, step 2): build the manifest (lab.manifest helpers),
        # persist it, then call self.backend.submit(manifest); return job_id.
        raise NotImplementedError("Lab.submit — P0 build order steps 1-2")
