"""The scheduler brain — one idempotent ``tick()`` (spec §4).

Every tick re-derives everything from the QueueStore; a crashed tick costs nothing. Dependencies
(clock, price feed, Lab construction) are injected so tests run table-driven with fakes.
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from lab._util import now
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.scheduler.bundle import extract_bundle
from lab.scheduler.models import ControlConfig, Registration, RegState, TickReport
from lab.scheduler.queue import QueueStore
from lab.store import JobStore


class Scheduler:
    LAUNCHING_ORPHAN_S = 600.0  # spec §5: launching older than 10 min is repaired

    def __init__(
        self,
        queue: QueueStore,
        home: Path,
        *,
        backend: str = "local",
        now_fn: Callable[[], datetime] = now,
        host: str | None = None,
    ) -> None:
        self.queue = queue
        self.home = Path(home)
        self.backend_name = backend
        self.now_fn = now_fn
        self.host = host or platform.node()
        self.store = JobStore(self.home)

    def make_lab(self, repo: Path) -> Lab:
        """Lab over the launch backend, rooted at an extracted bundle. Test seam."""
        if self.backend_name == "skypilot":
            from lab.backends.skypilot import SkyPilotBackend

            return Lab(backend=SkyPilotBackend(home=self.home, repo=repo), repo=repo, home=self.home)
        return Lab(backend=LocalBackend(home=self.home, repo=repo), repo=repo, home=self.home)

    def tick(self) -> TickReport:
        rep = TickReport(at=self.now_fn())
        tick_count = int((self.queue.read_heartbeat() or {}).get("tick_count", 0)) + 1
        control = self.queue.read_control()
        if not control.paused:
            entries = self.queue.list_entries()
            self._sync(entries, rep)            # Task 7
            self._expire(entries, rep)
            self._evaluate_and_launch(entries, control, rep)  # Task 6 (+8, 9)
        self.queue.write_heartbeat(
            {
                "at": rep.at.isoformat(),
                "host": self.host,
                "tick_count": tick_count,
                "launched": rep.launched,
                "errors": rep.errors,
            }
        )
        return rep

    # ------------------------------------------------------------------ phases
    def _sync(self, entries: list[Registration], rep: TickReport) -> None:
        """Mirror launched jobs' state back onto registrations (Task 7)."""

    def _expire(self, entries: list[Registration], rep: TickReport) -> None:
        nowt = self.now_fn()
        for reg in entries:
            if reg.state is RegState.pending and nowt >= reg.guardrails.expires_at:
                self._transition(reg, RegState.expired, reason="expired (run-by deadline passed)")
                rep.expired.append(reg.reg_id)

    def _evaluate_and_launch(
        self, entries: list[Registration], control: ControlConfig, rep: TickReport
    ) -> None:
        by_id = {r.reg_id: r for r in entries}
        for reg in entries:
            if reg.state is RegState.launching:
                self._repair_launching(reg, rep)
                continue
            if reg.state is not RegState.pending:
                continue
            reg = self.queue.get_entry(reg.reg_id)  # fresh copy (expiry may have written)
            if reg.state is not RegState.pending:
                continue
            if self.queue.cancel_requested(reg.reg_id):
                self._transition(reg, RegState.cancelled, reason="cancelled by user")
                rep.cancelled.append(reg.reg_id)
                continue
            if self.queue.held(reg.reg_id):
                rep.skipped[reg.reg_id] = "held"
                continue
            blocked = self._trigger_block(reg, by_id, rep)
            if blocked is not None:
                if blocked != "":  # "" => dependency hard-failure already handled
                    self.queue.put_entry(reg.model_copy(update={"last_skip_reason": blocked}))
                    rep.skipped[reg.reg_id] = blocked
                continue
            self._launch(reg, rep)

    def _trigger_block(
        self, reg: Registration, by_id: dict[str, Registration], rep: TickReport
    ) -> str | None:
        """None = all triggers hold; '' = entry was hard-cancelled; else the skip reason."""
        nowt = self.now_fn()
        t = reg.triggers
        if t.not_before is not None and nowt < t.not_before:
            return f"not_before {t.not_before.isoformat()}"
        if t.window is not None and not t.window.contains(nowt):
            return f"outside window {t.window.start}-{t.window.end} {t.window.tz}"
        for dep_id in t.after:  # dead deps cancel immediately, even behind waiting ones
            dep = by_id.get(dep_id)
            if dep is None or dep.state in (
                RegState.failed,
                RegState.expired,
                RegState.cancelled,
            ):
                why = dep.state.value if dep is not None else "missing"
                self._transition(reg, RegState.cancelled, reason=f"dependency {dep_id} ended {why}")
                rep.cancelled.append(reg.reg_id)
                return ""
        for dep_id in t.after:
            dep_state = by_id[dep_id].state
            if dep_state is not RegState.succeeded:
                return f"waiting on dependency {dep_id} ({dep_state.value})"
        return None  # price + guardrails extend this in Tasks 8-9

    def _launch(self, reg: Registration, rep: TickReport) -> None:
        reg = self._transition(reg, RegState.launching, reason=None)
        if self.queue.cancel_requested(reg.reg_id):  # spec §5 cancel race: re-check pre-submit
            self._transition(reg, RegState.cancelled, reason="cancelled by user")
            rep.cancelled.append(reg.reg_id)
            return
        try:
            bundle_dir = self.home / "_bundles" / reg.reg_id
            tar = self.queue.fetch_bundle(reg.reg_id, self.home / "_bundles")
            extract_bundle(tar, bundle_dir)
            lab = self.make_lab(bundle_dir)
            job_id = lab.submit(reg.spec, code=reg.code, registration_id=reg.reg_id)
        except Exception as e:  # noqa: BLE001 — a bad entry must not kill the tick (spec §5)
            self._transition(reg, RegState.pending, reason=f"launch error: {e}"[:300])
            rep.errors.append(f"{reg.reg_id}: {e}")
            return
        self._transition(
            reg, RegState.launched, reason=None, job_id=job_id, launched_at=self.now_fn()
        )
        rep.launched.append(reg.reg_id)

    def _repair_launching(self, reg: Registration, rep: TickReport) -> None:
        """A tick crashed mid-launch (spec §5): decide from evidence, after a grace period."""
        changed = reg.state_changed_at or reg.created_at
        if (self.now_fn() - changed).total_seconds() < self.LAUNCHING_ORPHAN_S:
            return
        job = next(
            (
                m
                for j in self.store.list_job_ids()
                if (m := self.store.read_manifest(j)).registration_id == reg.reg_id
            ),
            None,
        )
        if job is not None:  # the submit happened -> repair forward
            self._transition(
                reg, RegState.launched, reason="repaired: launch had completed",
                job_id=job.job_id, launched_at=job.created_at,
            )
            rep.synced[reg.reg_id] = "launched (repaired)"
        else:
            self._transition(reg, RegState.pending, reason="repaired: launch never happened")
            rep.errors.append(f"{reg.reg_id}: repaired launching -> pending (no job manifest)")

    def _transition(self, reg: Registration, state: RegState, *, reason: str | None = None,
                    **extra: object) -> Registration:
        updated = reg.model_copy(
            update={
                "state": state,
                "state_changed_at": self.now_fn(),
                "last_skip_reason": reason,
                **extra,
            }
        )
        self.queue.put_entry(updated)
        return updated
