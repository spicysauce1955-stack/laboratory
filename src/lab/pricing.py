"""What a price cap means once the machine is real (pure; no cloud, no sky, no vastai).

``--price-cap`` maps to ``sky.Resources(max_hourly_cost=)``, which SkyPilot's optimizer honours
against **its own catalog**. On Vast that catalog under-reports ~4x — a fact this codebase already
records, in :func:`lab.backends.skypilot.vast_hourly_for_cluster`:

    SkyPilot's ``get_cost()`` reads its own catalog and under-reports Vast prices (~4x low)

So on 2026-08-23 three of nine Vast jobs billed above a $0.85 cap, two of them at 2.61x
($2.220/hr); the two that finished cost $5.50 against an expected ~$1.03. The optimizer had done
exactly what it promised, against a number roughly four times too low.

The lab already *knew*: ``resolve_cost`` reads the rental's real ``dph_total`` seconds after the
host is UP, which is the only reason those overruns are visible in the manifests at all. What was
missing is the comparison itself — the one thing that needs no cloud to decide. It lives here,
dependency-free, so it is trivially testable and importable without ``sky`` installed. Talking to
Vast, tearing machines down and writing manifests all stay with their existing owners.
"""

from __future__ import annotations

# `CostInfo.hourly_usd` folds storage in and catalogue prices round, so a cap is not a knife edge.
# 5% is wide enough that arithmetic never raises a money alarm and far below the 2.6x that actually
# happened — the band exists to keep the alarm believable, which is its whole value (R10).
OVER_CAP_TOLERANCE = 0.05


def exceeds_cap(
    actual_hourly: float | None, cap: float | None, *, tolerance: float = OVER_CAP_TOLERANCE
) -> bool:
    """Is the billed rate meaningfully above the cap the user asked for?

    ``False`` whenever the question cannot be answered — no cap set, or no price read. A price we
    could not determine is not evidence of an overrun, and inventing one would put a money alarm on
    the healthy path, which is the failure this codebase keeps having to undo.
    """
    if cap is None or actual_hourly is None:
        return False
    return actual_hourly > cap * (1 + tolerance)


def over_cap_warning(actual: float, cap: float, cluster: str, timeout_s: float | None) -> str:
    """The operator-facing line for a rental billing above its cap.

    Leads with the numbers and the ratio, because "2.6x" is the part that makes someone act, and
    ends with the one command that stops the meter. A warning with no action in it is noise.
    """
    ratio = actual / cap if cap else 0.0
    # `:.0f` hours rendered a 30-minute job as "0h" and a 90-minute one as "2h" while the dollar
    # figure stayed exact — wrong units in the one line whose job is making the size obvious.
    if timeout_s:
        span = (
            f"{timeout_s / 60:.0f}m" if timeout_s < 3600 else f"{timeout_s / 3600:.1f}h".replace(".0h", "h")
        )
        projected = f" At this job's {span} timeout that is ${actual * timeout_s / 3600:.2f}."
    else:
        projected = ""
    return (
        f"[lab] PRICE CAP EXCEEDED: {cluster} bills ${actual:.3f}/hr against --price-cap "
        f"${cap:.2f}/hr ({ratio:.1f}x).{projected} SkyPilot applies the cap to its own catalog, "
        f"which under-reports Vast ~4x, so the optimizer accepted this host. The job is still "
        f"running — `lab cancel <job_id>` to stop paying, or raise --price-cap if this rate is "
        f"acceptable."
    )


def cap_admission_error(best_offer: float, cap: float, accelerators: str | None) -> str:
    """The refusal for a cap no live offer can satisfy — raised before anything is rented."""
    return (
        f"cheapest live Vast offer for {accelerators or 'this spec'} is ${best_offer:.3f}/hr, "
        f"above --price-cap ${cap:.2f}/hr. Nothing was rented. Raise --price-cap above "
        f"${best_offer:.3f}, or use `lab register --max-hourly` to queue until prices drop."
    )
