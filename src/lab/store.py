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

from lab.models import JobManifest


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
        """Create the run dir (incl. output/) and persist the initial manifest."""
        self.output_dir(manifest.job_id).mkdir(parents=True, exist_ok=True)
        self.logs_path(manifest.job_id).touch()
        self.write_manifest(manifest)
        return self.job_dir(manifest.job_id)

    def write_manifest(self, manifest: JobManifest) -> None:
        self._atomic_write(self.manifest_path(manifest.job_id), manifest.model_dump_json(indent=2))

    def read_manifest(self, job_id: str) -> JobManifest:
        return JobManifest.model_validate_json(self.manifest_path(job_id).read_text())

    def update_manifest(self, job_id: str, **fields: Any) -> JobManifest:
        """Read-modify-write the manifest's mutable fields (used by runner/backend)."""
        updated = self.read_manifest(job_id).model_copy(update=fields)
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

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
