"""On-disk job store — the run-dir layout and atomic manifest/runtime persistence.

Layout (per job, under the lab ``home`` dir, default ``runs/``):
    <job_id>/manifest.json   spec §8 record (source of truth)
    <job_id>/_runtime.json   local-only {runner_pid, command_pgid}
    <job_id>/logs.txt        captured stdout+stderr (FR-D1)
    <job_id>/output/         = $LAB_RUN_DIR, experiment outputs (FR-E1)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from lab.metrics import snapshot_final_metrics
from lab.models import JobManifest, JobState


class JobStore:
    def __init__(self, home: Path) -> None:
        self.home = Path(home)

    def job_dir(self, job_id: str) -> Path:
        return self.home / job_id

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output"

    def logs_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "logs.txt"

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def runtime_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "_runtime.json"

    def create(self, manifest: JobManifest) -> Path:
        """Create the run dir (incl. output/) and persist the initial manifest.

        The fail-closed provenance guard lives here — at the single new-manifest chokepoint —
        not in ``write_manifest``: ``code`` is immutable after create, so validating once at
        creation prevents any new Gap-B manifest, while later status ``update_manifest`` writes
        (including on legacy Gap-B manifests already on disk) never re-validate and so never
        crash (FR-B1)."""
        manifest.code.assert_fail_closed()
        self.output_dir(manifest.job_id).mkdir(parents=True, exist_ok=True)
        self.logs_path(manifest.job_id).touch()
        self.write_manifest(manifest)
        return self.job_dir(manifest.job_id)

    def write_manifest(self, manifest: JobManifest) -> None:
        self._atomic_write(self.manifest_path(manifest.job_id), manifest.model_dump_json(indent=2))

    def read_manifest(self, job_id: str) -> JobManifest:
        return JobManifest.model_validate_json(self.manifest_path(job_id).read_text())

    def update_manifest(self, job_id: str, **fields: Any) -> JobManifest:
        """Read-modify-write the manifest's mutable fields (used by runner/backend).

        On any transition to ``succeeded``, snapshot the run's final metric values into the manifest
        (FR-B4 durable baseline) unless the caller supplied them — so every backend's finalize path
        captures the baseline ``lab confirm`` compares against, without having to remember to.
        """
        updated = self.read_manifest(job_id).model_copy(update=fields)
        if updated.status is JobState.succeeded and not updated.final_metrics:
            fm = snapshot_final_metrics(self.output_dir(job_id))
            if fm:
                updated = updated.model_copy(update={"final_metrics": fm})
        self.write_manifest(updated)
        return updated

    def write_runtime(self, job_id: str, **fields: Any) -> None:
        """Merge local-only runtime fields (pids) into _runtime.json."""
        data = self.read_runtime(job_id)
        data.update(fields)
        self._atomic_write(self.runtime_path(job_id), json.dumps(data))

    def read_runtime(self, job_id: str) -> dict[str, Any]:
        p = self.runtime_path(job_id)
        return json.loads(p.read_text()) if p.exists() else {}

    def list_job_ids(self) -> list[str]:
        if not self.home.exists():
            return []
        return sorted(d.name for d in self.home.iterdir() if (d / "manifest.json").exists())

    def sweep_spend(self, sweep_id: str) -> float:
        """Sum actual spend of FINISHED points of a sweep (derived ceiling input; no meter)."""
        total = 0.0
        for jid in self.list_job_ids():
            m = self.read_manifest(jid)
            if m.sweep_id == sweep_id and m.cost and m.cost.actual_usd:
                total += m.cost.actual_usd
        return round(total, 6)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
