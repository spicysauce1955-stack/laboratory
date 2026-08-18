"""Pull the join keys and a small digest out of a command's payload.

``result`` is deliberately a digest: the full payload already lives in the manifest, and a
second, staler copy on disk is how this kind of store gets fat."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ID_KEYS = ("job_id", "sweep_id", "reg_id", "run_id")
_COST_KEYS = ("actual_cost_usd", "cost_usd", "estimated_usd")


def refs_from(payload: Any) -> dict[str, Any]:
    """Join keys tying this call to manifests: ``job_id``, ``sweep_id``, ``reg_id``, ``run_id``,
    plus ``job_ids`` collected from any nested list of job-shaped dicts."""
    if not isinstance(payload, Mapping):
        return {}
    refs: dict[str, Any] = {k: payload[k] for k in _ID_KEYS if isinstance(payload.get(k), str)}
    ids: list[str] = []
    for value in payload.values():
        if isinstance(value, list):
            ids += [v["job_id"] for v in value
                    if isinstance(v, Mapping) and isinstance(v.get("job_id"), str)]
    if ids:
        refs["job_ids"] = ids
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
