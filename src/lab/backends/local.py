"""Local subprocess backend — the NFR-4 fallback and the fastest P0 dev loop.

TODO(P0 build order, step 1 in research/16-decisions.md): implement.
  - submit: spawn the entrypoint as a subprocess into a temp run dir; inject
    $LAB_RUN_ID / $LAB_RUN_DIR / $LAB_SEED / metrics endpoint (FR-C4); capture
    stdout+stderr to a log file; enforce the wall-clock timeout (FR-I1).
  - status: derive state from the process (running / exit code → succeeded|failed|timed_out).
  - tail_logs: read the captured log file.
  - cancel: terminate the process group (FR-A3).
  - collect_artifacts: copy $LAB_RUN_DIR into runs/<job_id>/ with checksums (FR-E).
"""

from __future__ import annotations

from collections.abc import Iterable

from lab.models import ArtifactRecord, JobManifest, JobState


class LocalBackend:
    name = "local"

    def submit(self, manifest: JobManifest) -> str:
        raise NotImplementedError("LocalBackend.submit — P0 build order step 1")

    def status(self, job_id: str) -> JobState:
        raise NotImplementedError

    def tail_logs(
        self, job_id: str, tail: int | None = None, follow: bool = False
    ) -> Iterable[str]:
        raise NotImplementedError

    def cancel(self, job_id: str) -> JobState:
        raise NotImplementedError

    def collect_artifacts(self, job_id: str, dest: str) -> list[ArtifactRecord]:
        raise NotImplementedError
