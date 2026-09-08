"""What a job's money record *means* (pure; no cloud, no sky, no vastai).

Three decisions live here — whether a rental exceeded its cap, whether a terminal job ever ran at
all, and what this project has actually spent so far. All three are arithmetic over records that
already exist, so they are dependency-free and importable without ``sky`` installed. Talking to
Vast, tearing machines down and writing manifests all stay with their existing owners.

## The price cap

``--price-cap`` maps to ``sky.Resources(max_hourly_cost=)``, which SkyPilot's optimizer honours
against **its own catalog**. On Vast that catalog under-reports ~4x — a fact this codebase already
records, in :func:`lab.backends.skypilot.vast_hourly_for_cluster`:

    SkyPilot's ``get_cost()`` reads its own catalog and under-reports Vast prices (~4x low)

So on 2026-08-23 three of nine Vast jobs billed above a $0.85 cap, two of them at 2.61x
($2.220/hr); the two that finished cost $5.50 against an expected ~$1.03. The optimizer had done
exactly what it promised, against a number roughly four times too low.

The lab already *knew*: ``resolve_cost`` reads the rental's real ``dph_total`` seconds after the
host is UP, which is the only reason those overruns are visible in the manifests at all. What was
missing is the comparison itself — the one thing that needs no cloud to decide.

## Failed launches and running spend

Both were found hand-rolled inside a live shell watcher during the 2026-09 campaign post-mortem,
which is the evidence they belong in the tool. See :func:`failed_launch_reason` and
:func:`realized_spend` for the incidents each one carries.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from lab._util import actual_cost, duration_seconds
from lab.models import JobManifest, JobState

_TERMINAL_STATES = frozenset(
    {
        JobState.succeeded, JobState.failed, JobState.cancelled,
        JobState.timed_out, JobState.preempted,
    }
)

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


# ---------------------------------------------------------------------------------------------
# Failed launches
# ---------------------------------------------------------------------------------------------


def failed_launch_reason(m: JobManifest) -> str | None:
    """Why this job reached a terminal state without its workload ever running — or ``None``.

    A **failed launch** costs a slot and (near enough) no money. Counting one as an ordinary
    failure corrupts every success rate a campaign computes, and counting its ``null`` cost as
    ``$0`` corrupts every spend number. During the 2026-09 post-mortem this was found re-derived
    by hand inside a live shell watcher::

        term and st != 'succeeded' and (hr in (None, 0) or 'provisioning' in end_reason)

    Two of those three disjuncts are wrong here, and the errors point in opposite directions:

    * ``hr is None`` is **not** "free". ``cost: null`` means *not known* — a rule this codebase
      states explicitly, and one a monitor already broke the other way by reading ``terminal +
      cost:null`` as still-live. The honest structural signal is that there is no ``CostInfo`` at
      **all**: :func:`lab.sky_runner.resolve_cost` writes one the instant the host reaches UP, so
      its absence on a terminal remote job means the host never came up. A ``CostInfo`` that
      exists with ``hourly_usd=None`` is the opposite case — the machine ran and the *price
      lookup* failed — and returns ``None`` here, deliberately.
    * ``hr == 0`` is the **local** backend's genuine zero (own machine, really free), so that
      clause flagged every local failure as a launch that never happened.
    * substring-matching ``end_reason`` fires on a workload whose own traceback happens to say
      "provisioning", deleting a real failure from the record. Every provisioning death already
      satisfies the structural test above, so no string matching is needed for it.

    The reasons, all machine-readable and stable:

    ``"never_reached_up"``
        No ``CostInfo`` was ever written: provisioning timed out, the launch errored, or the job
        was cancelled/preempted before the host was UP. Terminal, not succeeded, never priced.
    ``"transient_launch_error"``
        ``end_reason`` carries the documented ``transient:`` prefix — the launch never reached a
        provider at all. Structurally a subset of the above; named separately because it is the
        one that is worth retrying verbatim.
    ``"price_cap_destroyed"``
        The rental billed over ``--price-cap`` with ``--price-cap-strict`` set, so
        :func:`lab.sky_runner.enforce_price_cap` destroyed it seconds after boot. It billed a few
        minutes and the workload never ran; charging that to the failure column hides a *pricing*
        problem behind a *code* one.

    Deliberately ``None`` for the **local** backend whatever the state: nothing is provisioned,
    nothing is billed, and there is no launch phase to fail — so a crashed local job is a real
    failure, not a job that never ran. Also ``None`` for any non-terminal job: a skypilot job is
    ``running`` with ``cost: null`` for its whole provisioning window, and calling that stillborn
    is how a monitor cancels a job that was about to start.
    """
    if m.status not in _TERMINAL_STATES or m.status is JobState.succeeded:
        return None
    if m.backend.provisioner == "local":
        return None
    cost = m.cost
    # `enforce_price_cap` fires exactly on this pair (strict + a definitive over-cap verdict) and
    # returns straight to teardown, so the pair on a terminal non-succeeded job *is* the destroy.
    if cost is not None and cost.over_cap is True and m.resources.price_cap_strict:
        return "price_cap_destroyed"
    if (m.end_reason or "").startswith("transient:"):
        return "transient_launch_error"
    if cost is None:
        return "never_reached_up"
    return None


def is_failed_launch(m: JobManifest) -> bool:
    """Did this job reach a terminal state without ever really running? See
    :func:`failed_launch_reason`, which also says *why*."""
    return failed_launch_reason(m) is not None


# ---------------------------------------------------------------------------------------------
# Running realized spend
# ---------------------------------------------------------------------------------------------

# Travels *with* the number, in every payload that carries it: an ambiguous money figure is worse
# than none, and this one is deliberately narrow.
SPEND_SCOPE = (
    "realized USD across this project's own runs/ only: finished jobs at their recorded "
    "actual_usd (or rate x their own elapsed time when no actual was written), still-running "
    "jobs at rate x elapsed-so-far, local-backend jobs at zero (own machine). Derived from the "
    "job manifests on every read — there is no meter. EXCLUDES: jobs whose rate was never "
    "readable (named in unknown_cost_jobs, never counted as zero), other projects' jobs, and "
    "scheduler-launched jobs with no local run dir. Not a provider bill."
)


def realized_spend(manifests: Iterable[JobManifest], *, at: datetime) -> dict[str, Any]:
    """What this project's jobs have actually burned by ``at`` — derived, never metered.

    On the 2026-09 campaign a $45 threshold crossing surfaced only in a retrospective self-audit
    about two landings later, and the two most expensive losses were exactly the ones with no live
    witness. There was no way to watch realized spend accumulate, so nobody did.

    A running job therefore contributes ``hourly_usd x elapsed-so-far``: waiting for the landing
    is precisely how a crossing goes unnoticed for two more jobs. A landed job contributes its
    recorded ``actual_usd``, and when that was never written (the ``--price-cap-strict`` destroy
    path, and any supervisor that died between pricing the box and finalizing it) it contributes
    ``hourly_usd x its own`` elapsed time, which stops growing at ``ended_at``.

    Derived on every read on purpose. This project's cost-safety rule is that a second stateful
    source of truth for money is a bug generator, so there is no counter file to keep in sync:
    the manifests already hold every term.

    The three-way split is the point:

    * a job with a readable amount is **counted**;
    * a job that never started, and any job on the **local** backend, is a **known zero** and is
      counted (nothing was provisioned, so nothing was burned) — filing queue noise and dev-loop
      runs under "unknown" would bury the real unknowns;
    * a job that started and has no readable rate is **unknown**: named in ``unknown_cost_jobs``
      and left out of the total entirely, never silently added as zero.

    Returns ``{realized_usd, jobs_counted, running_usd, running_jobs, unknown_cost_jobs, scope}``.
    ``running_usd`` is the still-accruing part of ``realized_usd``, not a separate sum.
    """
    total = 0.0
    running = 0.0
    counted = 0
    running_jobs: list[str] = []
    unknown: list[str] = []

    for m in manifests:
        cost = m.cost
        if cost is not None and cost.actual_usd is not None:
            total += cost.actual_usd
            counted += 1
            continue
        if m.backend.provisioner == "local":
            # Own machine, nothing rented — a known zero even mid-run, where the local runner has
            # not written its CostInfo yet. Listing a local job as "cost unknown" would put every
            # dev-loop run in the alarm list and teach people to ignore it (R10).
            counted += 1
            continue
        if m.started_at is None:
            counted += 1  # never provisioned: a known zero, not an unknown
            continue
        hourly = cost.hourly_usd if cost is not None else None
        if hourly is None:
            unknown.append(m.job_id)
            continue
        live = m.status not in _TERMINAL_STATES
        burned = actual_cost(hourly, duration_seconds(m.started_at, at if live else m.ended_at))
        if burned is None:  # terminal with no ended_at — nothing honest to size it against
            unknown.append(m.job_id)
            continue
        total += burned
        counted += 1
        if live:
            running += burned
            running_jobs.append(m.job_id)

    return {
        "realized_usd": round(total, 6),
        "jobs_counted": counted,
        "running_usd": round(running, 6),
        "running_jobs": running_jobs,
        "unknown_cost_jobs": unknown,
        "scope": SPEND_SCOPE,
    }


def spend_alert(summary: dict[str, Any], *, threshold_usd: float | None) -> dict[str, Any] | None:
    """The crossing verdict for a :func:`realized_spend` summary, or ``None`` if it has not
    crossed (or no threshold was asked for). Pure so both shells can render it their own way."""
    if threshold_usd is None:
        return None
    realized = float(summary["realized_usd"])
    if realized < threshold_usd:
        return None
    return {
        "threshold_usd": threshold_usd,
        "realized_usd": realized,
        "over_by_usd": round(realized - threshold_usd, 6),
        "message": spend_alert_message(
            realized_usd=realized,
            threshold_usd=threshold_usd,
            running_usd=float(summary["running_usd"]),
            running_jobs=len(summary["running_jobs"]),
            unknown_jobs=len(summary["unknown_cost_jobs"]),
        ),
    }


def spend_alert_message(
    *,
    realized_usd: float,
    threshold_usd: float,
    running_usd: float,
    running_jobs: int,
    unknown_jobs: int,
) -> str:
    """The operator-facing line for a crossed spend threshold.

    Leads with the numbers, says how much of the total is *still climbing*, and names what the
    total does not include — an alarm nobody can act on gets ignored (R10), and one whose scope is
    unstated is worse than none.
    """
    parts = [
        f"[lab] SPEND ALERT: this project's jobs have realized ${realized_usd:.2f} against "
        f"--spend-alert ${threshold_usd:.2f} (${realized_usd - threshold_usd:.2f} over)."
    ]
    if running_jobs:
        parts.append(
            f"${running_usd:.2f} of that is still accruing on {running_jobs} running job(s)."
        )
    if unknown_jobs:
        parts.append(
            f"{unknown_jobs} job(s) have no readable rate and are NOT in this total "
            f"(unknown_cost_jobs)."
        )
    parts.append("`lab ps` for what is still billing; `lab list` for the breakdown.")
    return " ".join(parts)
