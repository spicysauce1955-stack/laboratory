"""Scheduler data model — registrations, triggers, guardrails (spec §3).

A Registration wraps an ordinary :class:`lab.models.JobSpec` with launch *triggers* (AND
semantics; none present = eligible immediately) and *guardrails*. ``state`` is owned solely by
the scheduler tick; the laptop only writes cancel/hold markers (spec §5 single-writer rule).
"""

from __future__ import annotations

from datetime import datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from lab.models import CodeRef, JobSpec


class RegState(str, Enum):
    """Registration lifecycle (spec §3).

    ``held`` is a *display-only* state derived from the laptop's hold marker — the scheduler
    MUST NOT persist it into an entry; a held entry stays ``pending`` on disk.
    """

    pending = "pending"
    launching = "launching"
    launched = "launched"
    succeeded = "succeeded"
    failed = "failed"
    expired = "expired"
    cancelled = "cancelled"
    held = "held"


# A reg in one of these states will never launch (or relaunch) again, so its code bundle is no
# longer needed. Used by bundle GC to decide a tarball is safe to delete. ``held`` is omitted: it
# is display-only and a held entry is stored as ``pending`` (still live).
TERMINAL_REG_STATES = frozenset(
    {RegState.succeeded, RegState.failed, RegState.expired, RegState.cancelled}
)


class DailyWindow(BaseModel):
    """Recurring daily eligibility window, tz-aware; may cross midnight. End-exclusive."""

    start: time
    end: time
    tz: str = "UTC"  # IANA name

    def contains(self, dt: datetime) -> bool:
        if dt.tzinfo is None:
            raise ValueError("DailyWindow.contains requires a tz-aware datetime")
        local = dt.astimezone(ZoneInfo(self.tz)).time()
        if self.start <= self.end:
            return self.start <= local < self.end
        return local >= self.start or local < self.end  # crosses midnight


class Triggers(BaseModel):
    """All present triggers must hold simultaneously (AND). No triggers = launch ASAP."""

    not_before: datetime | None = None
    window: DailyWindow | None = None
    max_hourly_usd: float | None = None  # gate on best matching Vast offer price
    offer_query: str | None = None  # extra vastai search filter
    after: list[str] = Field(default_factory=list)  # reg_ids that must reach `succeeded`


class Guardrails(BaseModel):
    expires_at: datetime  # required — past this the entry expires, never launches
    max_cost_usd: float | None = None  # CUMULATIVE ceiling for the logical job (all retries)
    max_preempt_retries: int = 2  # per-point spot-preemption resubmits


class Registration(BaseModel):
    reg_id: str
    created_at: datetime
    spec: JobSpec
    triggers: Triggers = Field(default_factory=Triggers)
    guardrails: Guardrails
    bundle_key: str
    code: CodeRef  # commit + dirty captured at registration (provenance, FR-B1)
    state: RegState = RegState.pending
    job_id: str | None = None
    launched_at: datetime | None = None
    state_changed_at: datetime | None = None  # drives orphaned-`launching` repair (spec §5)
    last_skip_reason: str | None = None
    preempt_count: int = 0  # spot-preemption resubmits used so far
    cumulative_usd: float = 0.0  # summed actual spend across this job's attempts
    sweep_id: str | None = None  # set when this reg is one point of a deferred sweep
    sweep_max_cost: float | None = None  # derived sweep ceiling (cost-safety, Task 12)


class ControlConfig(BaseModel):
    """Global scheduler switchboard — ``queue/control.json`` (laptop-owned)."""

    paused: bool = False
    budget_usd_per_day: float | None = None  # trailing-24h estimated-spend cap
    max_concurrent: int = 4
    auto_reconcile: bool = False  # let the periodic sweep destroy confirmed orphans


class TickReport(BaseModel):
    """Structured outcome of one tick (returned + summarized into the heartbeat)."""

    at: datetime
    launched: list[str] = Field(default_factory=list)
    expired: list[str] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)  # reg_id -> reason
    synced: dict[str, str] = Field(default_factory=dict)  # reg_id -> new state
    errors: list[str] = Field(default_factory=list)
    reconcile: dict[str, object] | None = None
    preempted: list[str] = Field(default_factory=list)  # regs that hit spot preemption this tick
