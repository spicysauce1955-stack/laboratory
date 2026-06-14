"""R2-backed QueueStore — same contract as LocalQueueStore, keys under ``<prefix>/`` (spec §2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lab.models import JobManifest
from lab.scheduler.models import ControlConfig, Registration
from lab.storage import R2Store


class R2QueueStore:
    def __init__(self, store: R2Store, prefix: str = "queue") -> None:
        self.store = store
        self.prefix = prefix.rstrip("/")

    @classmethod
    def from_env(cls) -> R2QueueStore | None:
        store = R2Store.from_env()
        return cls(store) if store is not None else None

    def _k(self, *parts: str) -> str:
        return "/".join((self.prefix, *parts))

    # -- entries -------------------------------------------------------------
    def put_entry(self, reg: Registration) -> None:
        self.store.put_text(self._k("entries", f"{reg.reg_id}.json"), reg.model_dump_json())

    def get_entry(self, reg_id: str) -> Registration:
        text = self.store.get_text(self._k("entries", f"{reg_id}.json"))
        if text is None:
            raise FileNotFoundError(f"registration {reg_id} not found")
        return Registration.model_validate_json(text)

    def list_entries(self) -> list[Registration]:
        out: list[Registration] = []
        for key in sorted(self.store.list_keys(self._k("entries") + "/")):
            text = self.store.get_text(key)
            if text is not None:
                out.append(Registration.model_validate_json(text))
        return out

    # -- control / heartbeat ---------------------------------------------------
    def read_control(self) -> ControlConfig:
        text = self.store.get_text(self._k("control.json"))
        return ControlConfig.model_validate_json(text) if text else ControlConfig()

    def write_control(self, control: ControlConfig) -> None:
        self.store.put_text(self._k("control.json"), control.model_dump_json())

    def read_heartbeat(self) -> dict[str, Any] | None:
        import json

        text = self.store.get_text(self._k("heartbeat.json"))
        if text is None:
            return None
        loaded: dict[str, Any] = json.loads(text)
        return loaded

    def write_heartbeat(self, data: dict[str, Any]) -> None:
        import json

        self.store.put_text(self._k("heartbeat.json"), json.dumps(data, default=str))

    # -- markers ----------------------------------------------------------------
    def request_cancel(self, reg_id: str) -> None:
        self.store.put_text(self._k("cancelled", reg_id), "")

    def cancel_requested(self, reg_id: str) -> bool:
        return self.store.exists(self._k("cancelled", reg_id))

    def hold(self, reg_id: str) -> None:
        self.store.put_text(self._k("held", reg_id), "")

    def release(self, reg_id: str) -> None:
        self.store.delete(self._k("held", reg_id))

    def held(self, reg_id: str) -> bool:
        return self.store.exists(self._k("held", reg_id))

    # -- bundles ----------------------------------------------------------------
    def put_bundle(self, reg_id: str, src: Path) -> str:
        key = self._k("bundles", f"{reg_id}.tar.gz")
        self.store.upload_file(src, key)
        return key

    def fetch_bundle(self, bundle_key: str, dest_dir: Path) -> Path:
        out = Path(dest_dir) / Path(bundle_key).name
        self.store.download_file(bundle_key, out)  # bundle_key is the full stored key
        return out

    def list_bundle_keys(self) -> list[str]:
        return sorted(self.store.list_keys(self._k("bundles") + "/"))

    def delete_bundle(self, bundle_key: str) -> None:
        self.store.delete(bundle_key)  # bundle_key is the full stored key

    # -- mirrored manifests -------------------------------------------------------
    def mirror_manifest(self, manifest: JobManifest) -> None:
        self.store.put_text(self._k("jobs", f"{manifest.job_id}.json"), manifest.model_dump_json())

    def read_mirrored(self, job_id: str) -> JobManifest | None:
        text = self.store.get_text(self._k("jobs", f"{job_id}.json"))
        return JobManifest.model_validate_json(text) if text else None

    def list_mirrored(self) -> list[JobManifest]:
        out: list[JobManifest] = []
        for key in sorted(self.store.list_keys(self._k("jobs") + "/")):
            text = self.store.get_text(key)
            if text is not None:
                out.append(JobManifest.model_validate_json(text))
        return out
