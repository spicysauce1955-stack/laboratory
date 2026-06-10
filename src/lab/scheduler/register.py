"""``lab register`` core — capture code + spec + triggers into a queue entry (spec §6).

Write ordering is the integrity guarantee (spec §5): the bundle uploads first, the entry last —
the scheduler can never see an entry whose code is missing.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from lab._util import now, parse_duration
from lab.models import JobSpec, ResourceRequest
from lab.scheduler.bundle import create_bundle
from lab.scheduler.models import Guardrails, Registration, Triggers
from lab.scheduler.queue import QueueStore


def _new_reg_id() -> str:
    return f"reg-{now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def worst_case_cost(triggers: Triggers, resources: ResourceRequest) -> float | None:
    """What the user authorizes at registration time: max $/h x wall-clock timeout."""
    secs = parse_duration(resources.timeout)
    if triggers.max_hourly_usd is None or secs is None:
        return None
    return round(triggers.max_hourly_usd * secs / 3600.0, 6)


def register(
    repo: Path,
    queue: QueueStore,
    spec: JobSpec,
    triggers: Triggers,
    guardrails: Guardrails,
) -> Registration:
    reg_id = _new_reg_id()
    with tempfile.TemporaryDirectory() as td:
        tar, code = create_bundle(Path(repo), Path(td))
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
