"""R2-backed QueueStore — same contract as LocalQueueStore, keys under ``<prefix>/`` (spec §2)."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from lab.models import JobManifest
from lab.scheduler.models import ControlConfig, Registration
from lab.storage import R2Store

DEFAULT_LIST_CONCURRENCY = 8
"""Simultaneous R2 GETs per listing.

A listing was one sequential GET per key: 163 registrations measured a median of 33.6s and a p90
of 50.9s per `lab queue list` (2,419 calls, 23.3 hours of wall clock, over one 5-day campaign),
and the always-on scheduler host pays it on every 60s tick. The work is pure network latency, so
8 in flight cuts it by roughly its own factor while staying far below anything a queue-sized
listing could rate-limit; the object store is untouched (no index object, no layout change).
"""


def _int_env(name: str, default: int) -> int:
    """Same defensive parse as the ``LAB_EVENTS_*`` knobs: a typo must never fail a listing."""
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _list_concurrency() -> int:
    # Clamped to >= 1: ThreadPoolExecutor(max_workers=0) raises ValueError, and a knob set to 0
    # plainly means "don't parallelize", not "crash every listing".
    return max(1, _int_env("LAB_R2_LIST_CONCURRENCY", DEFAULT_LIST_CONCURRENCY))


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

    # -- concurrent reads ------------------------------------------------------
    def _read_key(self, key: str) -> tuple[str, str | None, Exception | None]:
        """Fetch one object, returning any failure instead of raising it.

        Runs on a pool thread, where an escaping exception would discard every sibling result the
        same batch was fetching. The error travels back with its key so the caller can decide, in
        the main thread, whether it is a skippable bad blob or a real I/O failure to re-raise.
        """
        try:
            return key, self.store.get_text(key), None
        except Exception as e:  # noqa: BLE001 — classified/re-raised by the caller, never dropped
            return key, None, e

    def _read_prefix(self, prefix: str) -> list[tuple[str, str | None, Exception | None]]:
        """Every object under ``prefix``, in sorted-key order, fetched concurrently.

        Order is the *key* order, not completion order: a caller diffing successive listings must
        never see spurious reordering (``Executor.map`` preserves the input order). One boto3
        client is shared across the pool — low-level clients are thread-safe for API calls, and a
        client per thread would pay a fresh TLS handshake for every key.
        """
        keys = sorted(self.store.list_keys(prefix))
        if len(keys) <= 1:  # nothing to overlap; skip the pool entirely
            return [self._read_key(k) for k in keys]
        workers = min(_list_concurrency(), len(keys))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lab-r2-list") as pool:
            return list(pool.map(self._read_key, keys))

    def list_entries(self) -> list[Registration]:
        from lab import events

        out: list[Registration] = []
        for key, text, error in self._read_prefix(self._k("entries") + "/"):
            try:
                if error is not None:
                    raise error
                if text is None:
                    continue
                reg = Registration.model_validate_json(text)
            except (ValidationError, UnicodeDecodeError) as e:
                # Same rationale as list_mirrored below: one partial/stale registration (an
                # older/newer scheduler host writing an entry this model can't validate — real
                # and recurring here — a read racing an in-progress write, or genuinely corrupted
                # non-UTF-8 blob bytes surfacing via `get_text`'s `.decode()`) must not take down
                # the whole listing: every other queued job would go invisible for the sake of
                # one bad blob. The skip is surfaced (stderr + ledger) rather than silent, so a
                # real corruption can't hide behind a listing that just looks one entry short.
                # Anything else — a genuine, persistent I/O failure — is re-raised above and
                # propagates, including from a pool thread.
                print(f"[lab] skipping unreadable queue entry {key}: {e}", file=sys.stderr)
                events.note("queue.entry_corrupt", key=key, error=str(e))
                continue
            out.append(reg)
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
    def _marker_ids(self, kind: str) -> set[str]:
        """Every marker id under ``<prefix>/<kind>/`` in one LIST — the exact inverse of the
        ``_k(kind, reg_id)`` key the writers below use (bare reg_id, no suffix, one segment).

        Skipped: the prefix itself (a ``<prefix>/<kind>/`` directory-placeholder object, which
        several S3 tools create) and any deeper key — neither is a marker this store wrote, and
        read literally either would invent a registration id.
        """
        prefix = self._k(kind) + "/"
        ids: set[str] = set()
        for key in self.store.list_keys(prefix):
            reg_id = key[len(prefix) :] if key.startswith(prefix) else ""
            if reg_id and "/" not in reg_id:
                ids.add(reg_id)
        return ids

    def request_cancel(self, reg_id: str) -> None:
        self.store.put_text(self._k("cancelled", reg_id), "")

    def cancel_requested(self, reg_id: str) -> bool:
        return self.store.exists(self._k("cancelled", reg_id))

    def cancel_requested_ids(self) -> set[str]:
        return self._marker_ids("cancelled")

    def hold(self, reg_id: str) -> None:
        self.store.put_text(self._k("held", reg_id), "")

    def release(self, reg_id: str) -> None:
        self.store.delete(self._k("held", reg_id))

    def held(self, reg_id: str) -> bool:
        return self.store.exists(self._k("held", reg_id))

    def held_ids(self) -> set[str]:
        return self._marker_ids("held")

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
        try:
            text = self.store.get_text(self._k("jobs", f"{job_id}.json"))
            if not text:
                return None
            return JobManifest.model_validate_json(text)
        except (ValidationError, UnicodeDecodeError):
            # A partial/stub manifest (e.g. version-skewed scheduler host, or a read racing an
            # in-progress write) must read as "not yet available", never crash the caller
            # (2026-09-04 `lab status` incident: 7 required fields missing). Genuinely corrupted
            # (non-UTF-8) blob bytes hit the same path via `get_text`'s `.decode()` rather than
            # pydantic — same degrade applies, so the decode call must sit inside this try too.
            return None

    def list_mirrored(self) -> list[JobManifest]:
        from lab import events

        out: list[JobManifest] = []
        for key, text, error in self._read_prefix(self._k("jobs") + "/"):
            try:
                if error is not None:
                    raise error
                if text is None:
                    continue
                manifest = JobManifest.model_validate_json(text)
            except (ValidationError, UnicodeDecodeError) as e:
                # Same rationale as read_mirrored: one partial/stale manifest (version skew, a
                # read racing an in-progress write, or genuinely corrupted non-UTF-8 blob bytes
                # surfacing via `get_text`'s `.decode()`) must not take down the whole listing.
                # Unlike read_mirrored this has no single caller waiting on "not yet available",
                # so the skip is surfaced (stderr + ledger) rather than silent — otherwise a real
                # corruption could sit invisible behind a listing that just looks one job short.
                print(f"[lab] skipping unreadable mirrored manifest {key}: {e}", file=sys.stderr)
                events.note("queue.manifest_corrupt", key=key, error=str(e))
                continue
            out.append(manifest)
        return out
