"""``lab register`` core — capture code + spec + triggers into a queue entry (spec §6).

Write ordering is the integrity guarantee (spec §5): the bundle uploads first, the entry last —
the scheduler can never see an entry whose code is missing.
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

from lab._util import now, parse_duration
from lab.core import LabError, build_sweep_point_spec, check_sweep_admission, expand_grid
from lab.models import JobSpec, ResourceRequest
from lab.scheduler.bundle import create_bundle
from lab.scheduler.models import DailyWindow, Guardrails, Registration, Triggers
from lab.scheduler.queue import QueueStore


def _new_reg_id() -> str:
    return f"reg-{now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _new_sweep_id() -> str:
    return f"sweep-{now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def worst_case_cost(triggers: Triggers, resources: ResourceRequest) -> float | None:
    """What the user authorizes at registration time: max $/h x wall-clock timeout."""
    secs = parse_duration(resources.timeout)
    if triggers.max_hourly_usd is None or secs is None:
        return None
    return round(triggers.max_hourly_usd * secs / 3600.0, 6)


def parse_expires(value: str) -> datetime:
    """``+3d``/``+12h`` (relative to now) or an ISO timestamp (trailing ``Z`` accepted)."""
    if value.startswith("+"):
        try:
            secs = parse_duration(value[1:])
        except ValueError as e:
            raise ValueError(f"bad relative expiry {value!r}") from e
        if secs is None:
            raise ValueError(f"bad relative expiry {value!r}")
        return now() + timedelta(seconds=secs)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"bad expiry timestamp {value!r}: {e}") from e


def parse_window(value: str, tz: str) -> DailyWindow:
    """``HH:MM-HH:MM`` (+ IANA tz) -> DailyWindow; may cross midnight."""
    try:
        start_s, end_s = value.split("-", 1)
        return DailyWindow(
            start=time.fromisoformat(start_s.strip()),
            end=time.fromisoformat(end_s.strip()),
            tz=tz,
        )
    except ValueError as e:
        raise ValueError(f"window expects HH:MM-HH:MM (got {value!r})") from e


def register(
    repo: Path,
    queue: QueueStore,
    spec: JobSpec,
    triggers: Triggers,
    guardrails: Guardrails,
) -> Registration:
    reg_id = _new_reg_id()
    with tempfile.TemporaryDirectory() as td:
        try:
            tar, code = create_bundle(Path(repo), Path(td))
        except subprocess.CalledProcessError as e:  # fail-loud, not a traceback (FR-F3)
            raise LabError(
                f"cannot snapshot {repo}: not a git repository (or git failed: {e})"
            ) from e
        bundle_key = queue.put_bundle(reg_id, tar)
    reg = Registration(
        reg_id=reg_id,
        created_at=now(),
        spec=spec,
        triggers=triggers,
        guardrails=guardrails,
        bundle_key=bundle_key,
        code=code,
    )
    queue.put_entry(reg)  # last write = commit point
    return reg


def register_sweep(
    repo: Path,
    queue: QueueStore,
    base_command: str,
    grid: dict[str, list[Any]],
    *,
    resources: ResourceRequest,
    triggers: Triggers,
    guardrails: Guardrails,
    seed: int | None = None,
    sweep_max_cost: float | None = None,
    daily_budget: float | None = None,
    committed: float = 0.0,
    submitted_by: str = "human",
    max_jobs: int = 256,
) -> tuple[str, list[Registration]]:
    """Register a grid as N deferred points sharing one sweep_id + ceiling + bundle (spec follow-up).

    The scheduler paces the points (triggers, ``max_concurrent``) and enforces the during-sweep
    ceiling. Each point keeps its own per-point ``max_cost_usd`` AND participates in the sweep-wide
    ``sweep_max_cost`` (default = worst case, so it only fires as a leak alarm). Bundle once, share
    by key (Decision A); write the bundle first and the entries last (integrity ordering, spec §5).
    """
    points = expand_grid(grid)
    if len(points) > max_jobs:
        raise LabError(
            f"sweep would register {len(points)} jobs (> max_jobs={max_jobs}); "
            "narrow the grid or raise max_jobs"
        )
    # Per-point cumulative cap: explicit guardrail, else worst case (max $/h x timeout). The resolved
    # sweep ceiling is the user's sweep_max_cost if set, else the worst case (doubles as leak alarm).
    per_point_cap = guardrails.max_cost_usd
    if per_point_cap is None:
        per_point_cap = worst_case_cost(triggers, resources)
    worst = check_sweep_admission(
        n_points=len(points),
        per_point_cap=per_point_cap,
        daily_budget=daily_budget,
        committed=committed,
    )
    resolved_ceiling = sweep_max_cost if sweep_max_cost is not None else worst

    sweep_id = _new_sweep_id()
    with tempfile.TemporaryDirectory() as td:
        try:
            tar, code = create_bundle(Path(repo), Path(td))
        except subprocess.CalledProcessError as e:  # fail-loud, not a traceback (FR-F3)
            raise LabError(
                f"cannot snapshot {repo}: not a git repository (or git failed: {e})"
            ) from e
        bundle_key = queue.put_bundle(sweep_id, tar)  # bundle first (integrity ordering)

    regs: list[Registration] = []
    for point in points:
        spec = build_sweep_point_spec(
            base_command, point, seed=seed, resources=resources, submitted_by=submitted_by
        )
        reg = Registration(
            reg_id=_new_reg_id(),
            created_at=now(),
            spec=spec,
            triggers=triggers,
            guardrails=guardrails,
            bundle_key=bundle_key,
            code=code,
            sweep_id=sweep_id,
            sweep_max_cost=resolved_ceiling,
        )
        queue.put_entry(reg)  # entries last (commit point)
        regs.append(reg)
    return sweep_id, regs
