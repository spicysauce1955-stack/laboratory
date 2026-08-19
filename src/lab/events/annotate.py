"""Pull the join keys and a small digest out of a command's payload.

``result`` is deliberately a digest: the full payload already lives in the manifest, and a
second, staler copy on disk is how this kind of store gets fat."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ID_KEYS = ("job_id", "sweep_id", "reg_id")
# Keys whose *value* is itself a job id, not a separate id namespace — e.g. `lab confirm`'s
# {orig_id, confirm_id, verdict}: `confirm_id` is a real job submitted to re-derive `orig_id`.
# Mapped onto `job_ids` so a confirm run is findable by `--job <orig_id>`/`--job <confirm_id>`
# the same way anything else that touched a job is.
_JOB_ID_ALIASES = ("orig_id", "confirm_id")
_COST_KEYS = ("actual_cost_usd", "cost_usd", "estimated_usd")
MAX_JOB_IDS = 64


def refs_from(payload: Any) -> dict[str, Any]:
    """Join keys tying this call to manifests: ``job_id``, ``sweep_id``, ``reg_id``, plus
    ``job_ids`` collected from a top-level ``job_ids`` list of strings (e.g. ``lab sweep``'s
    ``{sweep_id, count, job_ids:[str]}``) and from ``orig_id``/``confirm_id`` (``lab confirm``'s
    result — see ``_JOB_ID_ALIASES``).

    Deliberately does **not** walk arbitrary nested lists of job-shaped dicts: ``lab list``'s
    ``{"jobs": [{"job_id": ...}, ...]}`` is every job that *exists*, not every job this call
    *touched* — harvesting it made a listing call match every job in the store, bloating records
    and producing false positives in ``lab history --job <id>``.
    """
    if not isinstance(payload, Mapping):
        return {}
    refs: dict[str, Any] = {k: payload[k] for k in _ID_KEYS if isinstance(payload.get(k), str)}

    ids: list[str] = []
    top = payload.get("job_ids")
    if isinstance(top, list):
        ids += [v for v in top if isinstance(v, str)]
    for key in _JOB_ID_ALIASES:
        value = payload.get(key)
        if isinstance(value, str):
            ids.append(value)

    if ids:
        seen: set[str] = set()
        deduped: list[str] = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        if len(deduped) > MAX_JOB_IDS:
            refs["job_ids"] = deduped[:MAX_JOB_IDS] + [f"…{len(deduped) - MAX_JOB_IDS} more"]
        else:
            refs["job_ids"] = deduped
    return refs


def digest_of(payload: Any) -> dict[str, Any]:
    """A handful of scalars: state, cost, and the length of anything list-shaped."""
    if not isinstance(payload, Mapping):
        return {}
    digest: dict[str, Any] = {}
    if isinstance(payload.get("state"), str):
        digest["state"] = payload["state"]
    for key in _COST_KEYS:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            digest["cost_usd"] = float(value)
            break
    for key, value in payload.items():
        if isinstance(value, list):
            digest[f"{key}_n"] = len(value)
    return digest
