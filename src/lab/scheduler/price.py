"""Price trigger feed — 'is a matching Vast offer at/below $X/hr right now?' (spec §4.5).

Uses the vastai-sdk already in the stack (teardown fallback, FR-C2). ``dph_total`` is the real
billed rate (see ``vast_hourly_for_cluster``); the cheapest matching offer gates eligibility.
"""

from __future__ import annotations

from typing import Any, Protocol

_DEFAULTS = ["rentable=true", "rented=false", "reliability>0.95"]


def _get_vast_client() -> Any:
    """Function-local indirection: keeps this module importable without the skypilot extra
    (``lab.backends.skypilot`` imports ``sky`` at module top). Test seam — monkeypatch me."""
    from lab.backends.skypilot import _get_vast_client as real

    return real()


class PriceFeed(Protocol):
    def best_hourly(self, accelerators: str | None, extra_query: str | None = None) -> float | None:
        """Cheapest matching offer's $/hr, or None if no offer matches."""
        ...


def offer_query(accelerators: str | None, extra: str | None = None) -> str:
    """Derive a ``vastai search offers`` query from a SkyPilot accelerator spec."""
    parts = list(_DEFAULTS)
    if accelerators:
        name, _, count = accelerators.partition(":")
        parts.append(f"gpu_name={name}")
        parts.append(f"num_gpus>={count or 1}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


class VastPriceFeed:
    def best_hourly(self, accelerators: str | None, extra_query: str | None = None) -> float | None:
        offers: list[dict[str, Any]] = _get_vast_client().search_offers(
            query=offer_query(accelerators, extra_query)
        )
        prices = [
            float(o["dph_total"])
            for o in offers
            if isinstance(o, dict) and o.get("dph_total") is not None
        ]
        return min(prices) if prices else None
