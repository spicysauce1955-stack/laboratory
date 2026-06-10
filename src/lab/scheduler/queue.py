"""Queue persistence — the bus between laptop and scheduler host (spec §2).

Layout under a root (local dir here; same keys under an R2 prefix in r2queue.py):
    entries/<reg_id>.json     full Registration incl. state (scheduler-owned mutations)
    bundles/<reg_id>.tar.gz   code snapshot
    jobs/<job_id>.json        mirrored JobManifests of scheduler-launched jobs (spec §4.3)
    cancelled/<reg_id>        laptop-owned cancel markers
    held/<reg_id>             laptop-owned hold markers
    control.json              ControlConfig
    heartbeat.json            liveness + tick counter
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lab.models import JobManifest
from lab.scheduler.models import ControlConfig, Registration


@runtime_checkable
class QueueStore(Protocol):
    def put_entry(self, reg: Registration) -> None: ...
    def get_entry(self, reg_id: str) -> Registration: ...
    def list_entries(self) -> list[Registration]: ...
    def read_control(self) -> ControlConfig: ...
    def write_control(self, control: ControlConfig) -> None: ...
    def read_heartbeat(self) -> dict[str, Any] | None: ...
    def write_heartbeat(self, data: dict[str, Any]) -> None: ...
    def request_cancel(self, reg_id: str) -> None: ...
    def cancel_requested(self, reg_id: str) -> bool: ...
    def hold(self, reg_id: str) -> None: ...
    def release(self, reg_id: str) -> None: ...
    def held(self, reg_id: str) -> bool: ...
    def put_bundle(self, reg_id: str, src: Path) -> str: ...
    def fetch_bundle(self, reg_id: str, dest_dir: Path) -> Path: ...
    def mirror_manifest(self, manifest: JobManifest) -> None: ...
    def read_mirrored(self, job_id: str) -> JobManifest | None: ...
    def list_mirrored(self) -> list[JobManifest]: ...


class LocalQueueStore:
    """Filesystem QueueStore — tests, laptop-only mode, and the layout reference."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- entries -------------------------------------------------------------
    def _entry_path(self, reg_id: str) -> Path:
        return self.root / "entries" / f"{reg_id}.json"

    def put_entry(self, reg: Registration) -> None:
        self._atomic_write(self._entry_path(reg.reg_id), reg.model_dump_json(indent=2))

    def get_entry(self, reg_id: str) -> Registration:
        return Registration.model_validate_json(self._entry_path(reg_id).read_text())

    def list_entries(self) -> list[Registration]:
        d = self.root / "entries"
        if not d.exists():
            return []
        return [
            Registration.model_validate_json(p.read_text()) for p in sorted(d.glob("*.json"))
        ]

    # -- control / heartbeat ---------------------------------------------------
    def read_control(self) -> ControlConfig:
        p = self.root / "control.json"
        return ControlConfig.model_validate_json(p.read_text()) if p.exists() else ControlConfig()

    def write_control(self, control: ControlConfig) -> None:
        self._atomic_write(self.root / "control.json", control.model_dump_json(indent=2))

    def read_heartbeat(self) -> dict[str, Any] | None:
        p = self.root / "heartbeat.json"
        if not p.exists():
            return None
        loaded: dict[str, Any] = json.loads(p.read_text())
        return loaded

    def write_heartbeat(self, data: dict[str, Any]) -> None:
        self._atomic_write(self.root / "heartbeat.json", json.dumps(data, default=str))

    # -- laptop-owned markers (spec §5 single-writer rule) ----------------------
    def _marker(self, kind: str, reg_id: str) -> Path:
        return self.root / kind / reg_id

    def request_cancel(self, reg_id: str) -> None:
        self._atomic_write(self._marker("cancelled", reg_id), "")

    def cancel_requested(self, reg_id: str) -> bool:
        return self._marker("cancelled", reg_id).exists()

    def hold(self, reg_id: str) -> None:
        self._atomic_write(self._marker("held", reg_id), "")

    def release(self, reg_id: str) -> None:
        self._marker("held", reg_id).unlink(missing_ok=True)

    def held(self, reg_id: str) -> bool:
        return self._marker("held", reg_id).exists()

    # -- bundles ----------------------------------------------------------------
    def put_bundle(self, reg_id: str, src: Path) -> str:
        dest = self.root / "bundles" / f"{reg_id}.tar.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return f"bundles/{reg_id}.tar.gz"

    def fetch_bundle(self, reg_id: str, dest_dir: Path) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / f"{reg_id}.tar.gz"
        shutil.copy2(self.root / "bundles" / f"{reg_id}.tar.gz", out)
        return out

    # -- mirrored job manifests (spec §4.3) ---------------------------------------
    def mirror_manifest(self, manifest: JobManifest) -> None:
        self._atomic_write(
            self.root / "jobs" / f"{manifest.job_id}.json", manifest.model_dump_json(indent=2)
        )

    def read_mirrored(self, job_id: str) -> JobManifest | None:
        p = self.root / "jobs" / f"{job_id}.json"
        return JobManifest.model_validate_json(p.read_text()) if p.exists() else None

    def list_mirrored(self) -> list[JobManifest]:
        d = self.root / "jobs"
        if not d.exists():
            return []
        return [JobManifest.model_validate_json(p.read_text()) for p in sorted(d.glob("*.json"))]

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
