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
from lab.scheduler.models import Registration, RegState, TickReport
from lab.scheduler.queue import QueueStore
from lab.store import JobStore


class Scheduler:
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
        self, entries: list[Registration], control: object, rep: TickReport
    ) -> None:
        for reg in entries:
            if reg.state is not RegState.pending:
                continue
            if self.queue.held(reg.reg_id):
                rep.skipped[reg.reg_id] = "held"
                continue
            # Trigger/guardrail evaluation + launch land in Tasks 6, 8, 9.

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
