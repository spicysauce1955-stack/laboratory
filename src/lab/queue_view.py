"""The deferred queue's listing, as one view both shells return (FR-F2: thin shells, one logic).

``lab queue list`` and the MCP ``queue_list`` tool used to build this payload separately, and the
MCP copy drifted: it never gained ``scheduler_skew`` — the diagnostic that exists because an older
scheduler host silently dropped ``--price-cap`` from deferred jobs — nor the per-entry
``cancel_requested``. Shaping lives here so that cannot happen again; a shell only decides how to
present :attr:`QueueListView.warning`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from lab._util import now
from lab.scheduler.models import RegState
from lab.scheduler.queue import QueueStore
from lab.scheduler.skew import skew_from_heartbeat


@dataclass(frozen=True)
class QueueListView:
    payload: dict[str, Any]
    # Human-facing text for stderr, or None. Never part of the payload: stdout carries only JSON.
    warning: str | None


def _heartbeat_age_s(hb: dict[str, Any] | None) -> float | None:
    if not hb or "at" not in hb:
        return None
    at = datetime.fromisoformat(str(hb["at"]))
    return max(0.0, (now() - at).total_seconds())


def queue_list_view(queue: QueueStore) -> QueueListView:
    """Entries + state + skip reason, scheduler heartbeat age/pause-ack and which host wrote it.

    ``heartbeat_paused`` is what the *last completed tick* actually observed and acted on — unlike
    ``control.paused`` (which flips the instant a write lands), it's the signal a redeploy cutover
    needs to confirm the running scheduler has genuinely stopped launching.

    Every value is JSON-ready (timestamps as ISO-8601 strings), so both shells emit the same bytes.
    """
    entries = queue.list_entries()
    hb = queue.read_heartbeat()
    # Two listings, not two round trips per entry: asking `held()`/`cancel_requested()` per row
    # cost 163-326 sequential HEADs on the campaign that motivated this (2026-09, ~65s of a ~70s
    # `queue list`). One snapshot each also means the rows can no longer self-contradict, which
    # per-entry reads taken 30s apart could.
    held_ids = queue.held_ids()
    cancelled_ids = queue.cancel_requested_ids()
    # Version skew is silent and expensive: the always-on host deserialises registrations with
    # ITS pydantic models, so an older host drops fields it doesn't know — that is how
    # `--price-cap` was quietly lost on the deferred path, and it was then misdiagnosed as
    # `lab register` not having the flag. Diagnostic only: never a gate, never a refusal.
    #
    # `may_drop_fields` is False when there is no heartbeat at all: warning about an old host on a
    # queue no scheduler has ever ticked against asserts a fact nobody has evidence for, and a
    # diagnostic that is usually wrong is ignored when it is finally right (R10). The verdict and
    # its `detail` still ride in the payload, next to `heartbeat_age_s: null`.
    skew = skew_from_heartbeat(hb)
    payload = {
        "scheduler_skew": {
            "verdict": skew.verdict,
            "host_version": skew.host_version,
            "client_version": skew.client_version,
            "detail": skew.detail,
        },
        "heartbeat_age_s": _heartbeat_age_s(hb),
        "host": (hb or {}).get("host"),
        "heartbeat_paused": (hb or {}).get("paused"),
        "tick_count": (hb or {}).get("tick_count"),
        "control": queue.read_control().model_dump(),
        "entries": [
            {
                "reg_id": r.reg_id,
                "state": "held"
                if (r.state is RegState.pending and r.reg_id in held_ids)
                else r.state.value,
                "cancel_requested": r.reg_id in cancelled_ids,
                "job_id": r.job_id,
                "last_skip_reason": r.last_skip_reason,
                "expires_at": r.guardrails.expires_at.isoformat(),
            }
            for r in entries
        ],
    }
    warning = f"[lab] warning: {skew.detail}" if skew.may_drop_fields else None
    return QueueListView(payload=payload, warning=warning)
