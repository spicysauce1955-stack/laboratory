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
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lab.models import JobManifest
from lab.scheduler.models import ControlConfig, Registration, RegState


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
    def put_bundle(self, reg_id: str, src: Path) -> str:
        """Store a code bundle; return an opaque key. Pass it unchanged to fetch_bundle
        (its structure differs per store and must not be interpreted by callers)."""
        ...
    def fetch_bundle(self, bundle_key: str, dest_dir: Path) -> Path: ...
    def list_bundle_keys(self) -> list[str]: ...
    def delete_bundle(self, bundle_key: str) -> None: ...
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
        # sorted() is chronological because reg ids are timestamp-prefixed (register._new_reg_id)
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
        tmp = dest.parent / f"{reg_id}.tar.gz.tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)  # atomic — a half-copied bundle is never visible
        return f"bundles/{reg_id}.tar.gz"

    def fetch_bundle(self, bundle_key: str, dest_dir: Path) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / Path(bundle_key).name
        shutil.copy2(self.root / bundle_key, out)  # bundle_key is relative to the queue root
        return out

    def list_bundle_keys(self) -> list[str]:
        d = self.root / "bundles"
        if not d.exists():
            return []
        return sorted(f"bundles/{p.name}" for p in d.glob("*.tar.gz"))

    def delete_bundle(self, bundle_key: str) -> None:
        (self.root / bundle_key).unlink(missing_ok=True)  # bundle_key is relative to the queue root

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


def default_queue() -> QueueStore:
    """Queue selection: ``LAB_QUEUE_DIR`` (tests/laptop-only) > R2 (if configured) > repo-local."""
    import os

    from lab.manifest import repo_root

    env_dir = os.environ.get("LAB_QUEUE_DIR")
    if env_dir:
        return LocalQueueStore(Path(env_dir))
    from lab.scheduler.r2queue import R2QueueStore

    r2 = R2QueueStore.from_env()
    if r2 is not None:
        return r2
    return LocalQueueStore(repo_root() / "queue")


_BLOCKING_STATES = frozenset({RegState.launching, RegState.launched})


def wait_for_queue_drain(
    queue: QueueStore, *, interval: float = 10.0, timeout: float | None = None
) -> list[Registration]:
    """Block until no registration is `launching`/`launched`, or `timeout` elapses.

    The safety gate a scheduler cutover waits on before pausing the queue: pausing stops
    `Scheduler._sync`, so a job that finishes after pausing would never be observed reaching
    terminal — this must run first, while the queue is still unpaused and genuinely draining.

    A registration's `state` already reflects its mirrored job's real terminality (`_sync` keeps
    them in lock-step while unpaused), so checking `state` alone is sufficient — no separate
    job-status lookup. A `pending` registration (not yet triggered) is never blocking, regardless
    of how far in the future its trigger is.

    Returns the still-blocking registrations: empty on a clean drain, non-empty (whatever was
    still in flight) when `timeout` was hit first. Never raises on timeout, as long as at least
    one read succeeded along the way — the caller decides what a non-empty result means. A single
    flaky read (a real store like R2QueueStore has no internal retry) is tolerated rather than
    aborting a wait that can run for up to 30 minutes, the same tolerance used by every other poll
    this feature adds; a `list_entries()` that never once succeeds re-raises its last error at
    the deadline instead of silently reporting a clean drain with zero evidence, right before the
    caller pauses production.
    """
    deadline = time.monotonic() + timeout if timeout is not None else None
    blocking: list[Registration] = []
    ever_read = False
    last_error: Exception | None = None
    while True:
        try:
            entries = queue.list_entries()
        except Exception as e:  # noqa: BLE001 — tolerated below, re-raised only if never observed
            last_error = e
        else:
            ever_read = True
            blocking = [r for r in entries if r.state in _BLOCKING_STATES]
            if not blocking:
                return []
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not ever_read and last_error is not None:
                    raise last_error
                return blocking
            time.sleep(max(0.05, min(interval, remaining)))  # never overrun the deadline
        else:
            time.sleep(max(0.05, interval))  # guard against a busy-loop on interval<=0
