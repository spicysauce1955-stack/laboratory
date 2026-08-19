"""Event records. ``Event`` is a *folded* row: one open line plus its close line."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Note:
    """One ring-buffer entry: ``t`` is milliseconds since the call opened."""

    t: int
    k: str
    d: dict[str, Any]


@dataclass
class Event:
    """A folded call. ``outcome is None`` means the close line never arrived."""

    id: str
    ts: datetime
    session: str
    seq: int
    surface: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    project: dict[str, Any] = field(default_factory=dict)
    lab_version: str = ""
    outcome: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None
    refs: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    trace: list[Note] = field(default_factory=list)

    @property
    def status(self) -> str:
        """``outcome``, or the standing finding that a call never closed."""
        return self.outcome or "running-or-died"

    @property
    def failed(self) -> bool:
        """A call that never closed counts as failed — the missing close *is* the finding."""
        return self.outcome != "ok"


@dataclass
class ActionStat:
    action: str
    calls: int
    failures: int
    failure_rate: float
    median_ms: int


@dataclass
class SignatureStat:
    signature: str
    count: int
    first_seen: datetime
    last_seen: datetime
    actions: list[str]
    usd: float


@dataclass
class StatsView:
    since: datetime | None
    total: int
    failures: int
    dangling: int
    usd_burned: float
    actions: list[ActionStat]
    signatures: list[SignatureStat]
