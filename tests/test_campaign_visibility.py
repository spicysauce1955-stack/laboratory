"""Three numbers an operator had to hand-roll inside a shell watcher during the 2026-09 campaign.

All three were found reimplemented in inline python in a live watch loop, which is the evidence
they belong in the tool. Each one already has an incident attached:

* **failed launch** — the watcher re-derived ``FAILED-LAUNCH`` from ``terminal and status !=
  succeeded and (hourly in (None, 0) or 'provisioning' in end_reason)``. Two of those disjuncts
  are wrong in this codebase: ``cost: null`` means *not known*, never zero (a monitor already
  read ``terminal + cost:null`` as still-live), and ``hourly_usd == 0`` is the **local** backend's
  genuine free run. Conflating either with "never ran" corrupts every success-rate and cost
  number a campaign computes.
* **age** — an orchestrator compared ``started_at`` against a wrongly-believed current date,
  concluded two healthy jobs were 25-hour runaways and cancelled them: $2 and ~2h of progress
  gone. The recorded lesson was "never do date arithmetic on ``started_at`` by hand", so the
  arithmetic is exposed instead.
* **spend** — a $45 threshold crossing was noticed ~2 landings later in a retrospective audit,
  and the two most expensive losses were exactly the ones with no live witness.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from helpers import make_manifest
from typer.testing import CliRunner

import lab.cli as cli_mod
from lab._util import now
from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab, job_status_view
from lab.models import BackendInfo, CostInfo, JobManifest, JobState
from lab.pricing import (
    SPEND_SCOPE,
    failed_launch_reason,
    is_failed_launch,
    realized_spend,
    spend_alert,
    spend_alert_message,
)
from lab.store import JobStore

runner = CliRunner()


def _remote(
    job_id: str = "j1",
    *,
    status: JobState = JobState.failed,
    cost: CostInfo | None = None,
    end_reason: str | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    price_cap_strict: bool = False,
    max_hourly_usd: float | None = None,
    timeout: str | None = None,
) -> JobManifest:
    """A skypilot job manifest, in whatever lifecycle shape the test needs."""
    m = make_manifest(job_id, "python x.py")
    return m.model_copy(
        update={
            "backend": BackendInfo(provisioner="skypilot", machine_type="RTX4090"),
            "resources": m.resources.model_copy(
                update={
                    "cloud": "vast",
                    "price_cap_strict": price_cap_strict,
                    "max_hourly_usd": max_hourly_usd,
                    "timeout": timeout,
                }
            ),
            "status": status,
            "cost": cost,
            "end_reason": end_reason,
            "started_at": started_at,
            "ended_at": ended_at,
        }
    )


# ---------------------------------------------------------------------------------------------
# is_failed_launch — the predicate
# ---------------------------------------------------------------------------------------------


class TestAFailedLaunchIsAJobThatNeverRan:
    def test_a_provisioning_death_is_a_failed_launch(self) -> None:
        """The real shape: ProvisionTimeout writes `failed` with no CostInfo at all, because
        `resolve_cost` only runs once the host is UP."""
        m = _remote(
            status=JobState.failed,
            cost=None,
            end_reason="provisioning exceeded 900s (host never reached UP — likely a dead offer)",
            started_at=now() - timedelta(minutes=15),
            ended_at=now(),
        )

        assert is_failed_launch(m) is True
        assert failed_launch_reason(m) == "never_reached_up"

    def test_a_transient_launch_error_is_a_failed_launch(self) -> None:
        """`transient:` is the documented machine-readable prefix for a launch that never
        reached a provider."""
        m = _remote(
            cost=None, end_reason="transient: launch error: API server busy", ended_at=now()
        )

        assert failed_launch_reason(m) == "transient_launch_error"

    def test_a_price_cap_destroy_is_a_failed_launch(self) -> None:
        """--price-cap-strict destroys the box right after boot: it billed a few minutes and the
        workload never ran. Charging that to the failure column hides a *pricing* problem."""
        m = _remote(
            status=JobState.failed,
            price_cap_strict=True,
            max_hourly_usd=0.85,
            cost=CostInfo(
                hourly_usd=2.22, compute_hourly_usd=2.22, cap_hourly_usd=0.85, over_cap=True
            ),
            end_reason="price cap: rental bills $2.220/hr against --price-cap $0.85/hr",
            started_at=now() - timedelta(minutes=6),
            ended_at=now(),
        )

        assert failed_launch_reason(m) == "price_cap_destroyed"

    def test_a_real_post_run_failure_is_not(self) -> None:
        """The job ran on a machine we paid for and the code failed. That is a real failure and
        must keep counting as one."""
        m = _remote(
            status=JobState.failed,
            cost=CostInfo(hourly_usd=0.74, actual_usd=1.48, duration_seconds=7200),
            end_reason="exit code 1",
            started_at=now() - timedelta(hours=2),
            ended_at=now(),
        )

        assert is_failed_launch(m) is False
        assert failed_launch_reason(m) is None

    def test_a_succeeded_job_is_never_a_failed_launch(self) -> None:
        m = _remote(status=JobState.succeeded, cost=CostInfo(hourly_usd=0.74, actual_usd=1.48))

        assert is_failed_launch(m) is False

    def test_a_succeeded_job_with_no_cost_record_is_still_not_one(self) -> None:
        """Belt and braces: success is decided by the state, never by the money."""
        assert is_failed_launch(_remote(status=JobState.succeeded, cost=None)) is False


class TestTheCostUnknownBoundary:
    """`cost: null` means "not known", never "zero" — and the two are different manifests."""

    def test_a_job_that_ran_at_an_unreadable_price_is_not_a_failed_launch(self) -> None:
        """The distinction the hand-rolled `hr in (None, 0)` erased.

        `resolve_cost` writes a CostInfo the instant the host is UP; when the Vast price lookup
        fails it writes one with `hourly_usd=None` and says so in `hourly_basis`. The machine
        existed, the workload ran, and we merely failed to read the rate. Calling that a failed
        launch would delete a real failure from the record *and* hide a real bill.
        """
        m = _remote(
            status=JobState.failed,
            cost=CostInfo(hourly_usd=None, hourly_basis="vast RTX4090 compute unknown + 0GiB disk"),
            end_reason="exit code 1",
            started_at=now() - timedelta(hours=1),
            ended_at=now(),
        )

        assert is_failed_launch(m) is False
        assert failed_launch_reason(m) is None

    def test_a_local_job_that_failed_is_not_a_failed_launch(self) -> None:
        """The local backend records a genuine `hourly_usd=0.0` — own machine, really free. The
        watcher's `hourly in (None, 0)` flagged every local failure as a failed launch."""
        m = make_manifest("jl", "python x.py").model_copy(
            update={
                "status": JobState.failed,
                "cost": CostInfo(hourly_usd=0.0, actual_usd=0.0, duration_seconds=3.0),
                "end_reason": "exit code 1",
            }
        )

        assert is_failed_launch(m) is False

    def test_a_local_job_with_no_cost_record_is_not_a_failed_launch_either(self) -> None:
        """`runner exited without recording status` leaves cost null on a local job. Nothing is
        provisioned locally, so there is no launch to fail — and the crash is a real failure."""
        m = make_manifest("jl2", "python x.py").model_copy(
            update={
                "status": JobState.failed,
                "cost": None,
                "end_reason": "runner exited without recording status",
            }
        )

        assert is_failed_launch(m) is False


class TestOnlyTerminalJobsQualify:
    @pytest.mark.parametrize("state", [JobState.queued, JobState.running])
    def test_a_live_job_with_no_cost_yet_is_not_a_failed_launch(self, state: JobState) -> None:
        """A skypilot job is `running` with `cost: null` for its whole provisioning window. A
        monitor already misread `terminal + cost:null` once; the mirror-image misread — calling a
        provisioning job dead — would cancel a job that is about to start."""
        assert is_failed_launch(_remote(status=state, cost=None)) is False

    def test_a_job_cancelled_before_the_host_came_up_never_ran(self) -> None:
        """Terminal, not succeeded, never priced. It cost a slot and no money, which is exactly
        what the flag is for — the name says `failed launch`, the meaning is `never ran`."""
        assert failed_launch_reason(_remote(status=JobState.cancelled, cost=None)) == (
            "never_reached_up"
        )


class TestTheAlarmStaysBelievable:
    def test_over_cap_without_strict_is_not_a_failed_launch(self) -> None:
        """Without --price-cap-strict nothing is destroyed: the job bills over and runs to the
        end. It is expensive, not stillborn (R10 — an alarm that is usually wrong gets ignored)."""
        m = _remote(
            status=JobState.failed,
            price_cap_strict=False,
            max_hourly_usd=0.85,
            cost=CostInfo(hourly_usd=2.22, actual_usd=5.50, cap_hourly_usd=0.85, over_cap=True),
            end_reason="exit code 1",
        )

        assert is_failed_launch(m) is False

    def test_an_end_reason_merely_mentioning_provisioning_does_not_trip_it(self) -> None:
        """The watcher substring-matched `end_reason`; a workload whose own error text says
        "provisioning" would be silently deleted from the failure column."""
        m = _remote(
            status=JobState.failed,
            cost=CostInfo(hourly_usd=0.74, actual_usd=0.74),
            end_reason="exit code 1: RuntimeError while provisioning the data loader",
            started_at=now() - timedelta(hours=1),
            ended_at=now(),
        )

        assert is_failed_launch(m) is False


# ---------------------------------------------------------------------------------------------
# The status view: is_failed_launch + age_s
# ---------------------------------------------------------------------------------------------


class TestTheStatusViewCarriesTheDerivedFields:
    def test_a_provisioning_death_is_flagged_in_the_view(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        store.create(
            _remote(
                "jv1",
                status=JobState.failed,
                cost=None,
                end_reason="provisioning exceeded 900s (host never reached UP)",
                started_at=now() - timedelta(minutes=15),
                ended_at=now(),
            )
        )

        view = job_status_view(tmp_path, tmp_path, "jv1")

        assert view["is_failed_launch"] is True
        assert view["failed_launch_reason"] == "never_reached_up"

    def test_a_succeeded_job_is_not(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        store.create(
            _remote(
                "jv2",
                status=JobState.succeeded,
                cost=CostInfo(hourly_usd=0.74, actual_usd=1.48),
                started_at=now() - timedelta(hours=2),
                ended_at=now(),
            )
        )

        view = job_status_view(tmp_path, tmp_path, "jv2")

        assert view["is_failed_launch"] is False
        assert view["failed_launch_reason"] is None

    def test_age_s_of_a_running_job_counts_up_from_started_at(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        store.create(
            make_manifest("jv3", "python x.py").model_copy(
                update={"status": JobState.running, "started_at": now() - timedelta(minutes=90)}
            )
        )

        view = job_status_view(tmp_path, tmp_path, "jv3")

        assert view["state"] == "running"
        assert 5390 <= view["age_s"] <= 5410  # ~90 minutes

    def test_age_s_of_a_terminal_job_freezes_at_its_final_lifetime(self, tmp_path) -> None:
        """Elapsed-to-terminal, not "seconds since it started". A job that *ended* 25 hours ago
        is not a 25-hour runaway, and a number that keeps growing would recreate exactly the
        illusion that got two healthy jobs cancelled."""
        store = JobStore(tmp_path)
        store.create(
            _remote(
                "jv4",
                status=JobState.succeeded,
                cost=CostInfo(hourly_usd=0.5, actual_usd=0.25),
                started_at=now() - timedelta(hours=25),
                ended_at=now() - timedelta(hours=24),
            )
        )

        view = job_status_view(tmp_path, tmp_path, "jv4")

        assert 3595 <= view["age_s"] <= 3605  # one hour of run, ended a day ago

    def test_age_s_is_none_before_a_job_starts(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        store.create(make_manifest("jv5", "python x.py"))  # queued, never started

        assert job_status_view(tmp_path, tmp_path, "jv5")["age_s"] is None

    def test_age_s_is_none_when_the_end_time_was_never_recorded(self, tmp_path) -> None:
        """A supervisor that died before writing `ended_at` leaves a terminal manifest with no
        end time. There is no honest age to report, and measuring to `now` would grow forever."""
        store = JobStore(tmp_path)
        store.create(
            _remote(
                "jv6",
                status=JobState.failed,
                cost=CostInfo(hourly_usd=0.5),
                started_at=now() - timedelta(hours=3),
                ended_at=None,
            )
        )

        assert job_status_view(tmp_path, tmp_path, "jv6")["age_s"] is None

    def test_a_mirrored_job_gets_an_age_and_keeps_its_staleness_marker(
        self, tmp_path, monkeypatch
    ) -> None:
        """`started_at` is an absolute UTC instant, so a mirrored running job's age is computed
        against local `now` and is not itself stale. What can be stale is whether it is still
        running — which is what the existing `mirrored` flag already says, so `age_s` carries no
        second marker of its own."""
        from lab.scheduler.queue import LocalQueueStore

        qdir = tmp_path / "queue"
        LocalQueueStore(qdir).mirror_manifest(
            _remote(
                "jv7", status=JobState.running, started_at=now() - timedelta(minutes=10), cost=None
            )
        )
        monkeypatch.setenv("LAB_QUEUE_DIR", str(qdir))

        view = job_status_view(tmp_path, tmp_path, "jv7")

        assert view["mirrored"] is True
        assert 590 <= view["age_s"] <= 610
        assert view["is_failed_launch"] is False


# ---------------------------------------------------------------------------------------------
# Realized spend
# ---------------------------------------------------------------------------------------------


class TestRealizedSpend:
    def test_it_sums_what_finished_jobs_actually_cost(self) -> None:
        at = now()
        jobs = [
            _remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=1.25)),
            _remote("b", status=JobState.failed, cost=CostInfo(actual_usd=0.75)),
        ]

        summary = realized_spend(jobs, at=at)

        assert summary["realized_usd"] == 2.0
        assert summary["jobs_counted"] == 2
        assert summary["unknown_cost_jobs"] == []

    def test_a_running_job_contributes_what_it_has_already_burned(self) -> None:
        """The whole point of a live witness: waiting for the landing is how a $45 crossing goes
        unnoticed for two more jobs."""
        at = now()
        jobs = [
            _remote(
                "r",
                status=JobState.running,
                cost=CostInfo(hourly_usd=2.0),
                started_at=at - timedelta(minutes=30),
            )
        ]

        summary = realized_spend(jobs, at=at)

        assert summary["realized_usd"] == pytest.approx(1.0, abs=1e-3)
        assert summary["running_usd"] == pytest.approx(1.0, abs=1e-3)
        assert summary["running_jobs"] == ["r"]

    def test_a_job_whose_cost_is_unknown_is_never_counted_as_free(self) -> None:
        """`cost: null` on a job that started means nobody could read the rate — not $0. Silently
        adding a zero is how a total becomes confidently wrong."""
        at = now()
        jobs = [
            _remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=1.25)),
            _remote("u", status=JobState.running, cost=None, started_at=at - timedelta(hours=4)),
        ]

        summary = realized_spend(jobs, at=at)

        assert summary["realized_usd"] == 1.25
        assert summary["unknown_cost_jobs"] == ["u"]
        assert summary["jobs_counted"] == 1

    def test_a_job_that_ran_at_an_unreadable_rate_is_unknown_too(self) -> None:
        at = now()
        jobs = [
            _remote(
                "u2",
                status=JobState.failed,
                cost=CostInfo(hourly_usd=None, hourly_basis="vast compute unknown"),
                started_at=at - timedelta(hours=1),
                ended_at=at,
            )
        ]

        summary = realized_spend(jobs, at=at)

        assert summary["realized_usd"] == 0.0
        assert summary["unknown_cost_jobs"] == ["u2"]

    def test_a_job_that_never_started_is_a_known_zero_not_an_unknown(self) -> None:
        """Nothing was provisioned, so nothing was burned. Filing it under `unknown` would bury
        the real unknowns in queue noise."""
        summary = realized_spend([make_manifest("q", "python x.py")], at=now())

        assert summary["unknown_cost_jobs"] == []
        assert summary["realized_usd"] == 0.0
        assert summary["jobs_counted"] == 1

    def test_a_local_jobs_genuine_zero_counts_as_a_zero(self) -> None:
        m = make_manifest("l", "python x.py").model_copy(
            update={"status": JobState.succeeded, "cost": CostInfo(hourly_usd=0.0, actual_usd=0.0)}
        )

        summary = realized_spend([m], at=now())

        assert summary["realized_usd"] == 0.0
        assert summary["jobs_counted"] == 1
        assert summary["unknown_cost_jobs"] == []

    def test_a_running_local_job_is_a_known_zero_not_an_unknown(self) -> None:
        """The local runner writes its CostInfo only at the end, so mid-run there is none — but
        nothing is rented. Every dev-loop run appearing in the alarm list is how an alarm gets
        ignored."""
        at = now()
        m = make_manifest("lr", "python x.py").model_copy(
            update={"status": JobState.running, "started_at": at - timedelta(hours=3)}
        )

        summary = realized_spend([m], at=at)

        assert summary["unknown_cost_jobs"] == []
        assert summary["realized_usd"] == 0.0

    def test_a_price_cap_destroy_still_bills_for_the_minutes_it_ran(self) -> None:
        """No `actual_usd` is ever written on that path, but the box really did bill. Deriving it
        from rate x elapsed is the difference between seeing the loss and not."""
        at = now()
        m = _remote(
            "p",
            status=JobState.failed,
            price_cap_strict=True,
            cost=CostInfo(hourly_usd=2.22, cap_hourly_usd=0.85, over_cap=True),
            started_at=at - timedelta(minutes=6),
            ended_at=at,
        )

        summary = realized_spend([m], at=at)

        assert summary["realized_usd"] == pytest.approx(0.222, abs=1e-3)
        assert summary["running_jobs"] == []

    def test_the_total_stops_growing_once_a_job_lands(self) -> None:
        at = now()
        m = _remote(
            "d",
            status=JobState.succeeded,
            cost=CostInfo(hourly_usd=1.0),
            started_at=at - timedelta(hours=5),
            ended_at=at - timedelta(hours=4),
        )

        assert realized_spend([m], at=at)["realized_usd"] == pytest.approx(1.0, abs=1e-3)
        later = realized_spend([m], at=at + timedelta(hours=10))
        assert later["realized_usd"] == pytest.approx(1.0, abs=1e-3)

    def test_the_number_carries_what_it_does_and_does_not_include(self) -> None:
        """An ambiguous money number is worse than none, so the caveat travels with it."""
        summary = realized_spend([], at=now())

        assert summary["scope"] == SPEND_SCOPE
        low = SPEND_SCOPE.lower()
        assert "runs/" in low and "unknown_cost_jobs" in low and "still-running" in low


def test_estimated_running_usd_is_bounded_the_same_way_realized_spend_is() -> None:
    """`lab status`'s own live estimate had the same unbounded shape `realized_spend` just fixed.

    A dead supervisor leaves a manifest at `running` forever, so an unbounded estimate keeps
    climbing over money nobody spent — on the single-job surface an operator reads most often.
    """
    from lab.core import _estimated_running_usd
    from lab.pricing import live_ceiling_s

    m = _remote(
        "stale",
        status=JobState.running,
        timeout="2h",
        cost=CostInfo(hourly_usd=0.40),
        started_at=now() - timedelta(days=7),
    )

    est = _estimated_running_usd(m, "running")

    assert est is not None
    assert est == pytest.approx(0.40 * live_ceiling_s(m) / 3600.0, rel=1e-3)
    assert est < 1.5  # unbounded, a week at $0.40/hr reads ~$67


class TestALiveJobsContributionIsBounded:
    """A `running` manifest is not evidence that a machine is running.

    ~40% of the DigitalOcean supervisors in the 2026-08 campaign died silently, leaving the
    manifest at `running` forever — `reconcile`'s `unsupervised` category exists for exactly that
    shape. Charging such a job `rate x (now - started_at)` with no ceiling adds money nobody spent,
    grows on every read, and fires `--spend-alert` on a phantom; by this project's own R10 rule an
    alarm that is usually wrong gets ignored, which is the outcome the feature exists to prevent.
    """

    def test_a_week_old_stale_running_job_does_not_inflate_the_total(self) -> None:
        at = now()
        m = _remote(
            "stale",
            status=JobState.running,
            timeout="2h",
            cost=CostInfo(hourly_usd=0.40),
            started_at=at - timedelta(days=7),
        )

        summary = realized_spend([m], at=at)

        # Unbounded, this reads ~$67. The box kills itself at its own 2h cap (GNU timeout +
        # poweroff), so no honest reading of this manifest bills more than that cap plus the
        # provisioning/teardown slack `started_at` also charges.
        assert summary["realized_usd"] < 1.5, summary
        assert summary["realized_usd"] >= 0.40 * 2  # the cap itself is still counted

    def test_the_phantom_stops_growing_on_every_read(self) -> None:
        at = now()
        m = _remote(
            "stale",
            status=JobState.running,
            timeout="2h",
            cost=CostInfo(hourly_usd=0.40),
            started_at=at - timedelta(days=7),
        )

        first = realized_spend([m], at=at)["realized_usd"]
        later = realized_spend([m], at=at + timedelta(days=1))["realized_usd"]

        assert later == first

    def test_the_alert_does_not_fire_on_a_phantom(self) -> None:
        """The whole point: $45 must mean $45 of real money."""
        at = now()
        m = _remote(
            "stale",
            status=JobState.running,
            timeout="2h",
            cost=CostInfo(hourly_usd=0.40),
            started_at=at - timedelta(days=7),
        )

        summary = realized_spend([m], at=at)

        assert spend_alert(summary, threshold_usd=45.0) is None

    def test_a_genuinely_running_job_inside_its_cap_still_counts_and_still_moves(self) -> None:
        at = now()
        m = _remote(
            "live",
            status=JobState.running,
            timeout="4h",
            cost=CostInfo(hourly_usd=2.0),
            started_at=at - timedelta(minutes=30),
        )

        summary = realized_spend([m], at=at)

        assert summary["realized_usd"] == pytest.approx(1.0, abs=1e-3)
        assert summary["running_jobs"] == ["live"]
        assert summary["unsupervised_suspect_jobs"] == []  # nothing suspicious about it yet
        # and it is still a *live* witness: the number moves between reads.
        later = realized_spend([m], at=at + timedelta(minutes=30))
        assert later["realized_usd"] == pytest.approx(2.0, abs=1e-3)

    def test_a_job_past_its_own_cap_is_surfaced_as_a_probable_dead_supervisor(self) -> None:
        """Capping quietly would hide the far more interesting fact: a job that outlived its own
        wall-clock cap and is still `running` almost certainly has no supervisor left."""
        at = now()
        m = _remote(
            "orphan",
            status=JobState.running,
            timeout="2h",
            cost=CostInfo(hourly_usd=0.40),
            started_at=at - timedelta(days=7),
        )

        summary = realized_spend([m], at=at)

        assert summary["unsupervised_suspect_jobs"] == ["orphan"]
        assert summary["running_jobs"] == ["orphan"]  # still counted, just no longer growing

    def test_a_job_with_no_timeout_is_bounded_by_the_documented_horizon(self) -> None:
        """"No timeout" is not "bill forever": there is no self-enforced ceiling, so the module
        applies its own explicit one and says so, rather than being silently unbounded."""
        at = now()
        m = _remote(
            "uncapped",
            status=JobState.running,
            timeout=None,
            cost=CostInfo(hourly_usd=1.0),
            started_at=at - timedelta(days=30),
        )

        summary = realized_spend([m], at=at)

        assert summary["realized_usd"] == pytest.approx(24.0, abs=1e-3)  # 24 h x $1.00
        assert summary["unsupervised_suspect_jobs"] == ["uncapped"]

    def test_an_unparseable_timeout_never_raises_out_of_a_listing(self) -> None:
        """Manifests are on-disk history: a legacy or hand-edited `timeout` must degrade to the
        horizon, not blow up `lab list`."""
        at = now()
        m = _remote(
            "junk",
            status=JobState.running,
            timeout="banana",
            cost=CostInfo(hourly_usd=1.0),
            started_at=at - timedelta(days=30),
        )

        summary = realized_spend([m], at=at)

        assert summary["realized_usd"] == pytest.approx(24.0, abs=1e-3)

    def test_a_clock_that_runs_backwards_never_bills_negative(self) -> None:
        at = now()
        m = _remote(
            "skewed",
            status=JobState.running,
            timeout="2h",
            cost=CostInfo(hourly_usd=1.0),
            started_at=at + timedelta(hours=1),
        )

        assert realized_spend([m], at=at)["realized_usd"] == 0.0

    def test_a_landed_job_is_never_capped(self) -> None:
        """The bound is about *live* arithmetic. A terminal job's elapsed time is a recorded fact,
        even when it overran (a wall-clock cap has been mis-anchored before: a 7h cap allowed
        703min), and the recorded fact wins."""
        at = now()
        m = _remote(
            "over",
            status=JobState.succeeded,
            timeout="2h",
            cost=CostInfo(hourly_usd=1.0),
            started_at=at - timedelta(hours=11, minutes=43),
            ended_at=at,
        )

        summary = realized_spend([m], at=at)

        assert summary["realized_usd"] == pytest.approx(11.72, abs=1e-2)
        assert summary["unsupervised_suspect_jobs"] == []

    def test_the_scope_states_the_bound_and_its_direction(self) -> None:
        summary = realized_spend([], at=now())

        low = summary["scope"].lower()
        assert "unsupervised_suspect_jobs" in low
        assert "timeout" in low or "cap" in low


class TestTheSpendAlert:
    def test_it_fires_the_moment_the_threshold_is_crossed(self) -> None:
        summary = realized_spend(
            [_remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=47.30))], at=now()
        )

        alert = spend_alert(summary, threshold_usd=45.0)

        assert alert is not None
        assert alert["threshold_usd"] == 45.0
        assert alert["over_by_usd"] == pytest.approx(2.30, abs=1e-6)

    def test_it_stays_quiet_below_the_threshold(self) -> None:
        summary = realized_spend(
            [_remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=44.99))], at=now()
        )

        assert spend_alert(summary, threshold_usd=45.0) is None

    def test_no_threshold_means_no_alert(self) -> None:
        summary = realized_spend(
            [_remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=1000.0))], at=now()
        )

        assert spend_alert(summary, threshold_usd=None) is None

    def test_the_message_names_the_numbers_and_the_unknowns(self) -> None:
        msg = spend_alert_message(
            realized_usd=47.30,
            threshold_usd=45.0,
            running_usd=1.20,
            running_jobs=1,
            unknown_jobs=2,
        )

        assert "47.30" in msg and "45.00" in msg
        assert "1.20" in msg and "still" in msg.lower()
        assert "2" in msg and "not" in msg.lower()  # the unknowns are outside the total
        assert "lab ps" in msg or "lab list" in msg  # an alarm with no action is noise

    def test_the_message_points_a_capped_out_job_at_reconcile(self) -> None:
        """A job that stopped accruing because it passed its own cap is the tool's strongest
        cheap hint of a dead supervisor, and `reconcile` is where that gets confirmed."""
        msg = spend_alert_message(
            realized_usd=47.30,
            threshold_usd=45.0,
            running_usd=1.20,
            running_jobs=1,
            unknown_jobs=0,
            unsupervised_suspects=1,
        )

        assert "reconcile" in msg
        assert "unknown_cost_jobs" not in msg  # there were none; don't invent an unknowns clause

    def test_the_alert_carries_the_suspects_from_the_summary(self) -> None:
        at = now()
        summary = realized_spend(
            [
                _remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=47.30)),
                _remote(
                    "orphan",
                    status=JobState.running,
                    timeout="2h",
                    cost=CostInfo(hourly_usd=0.40),
                    started_at=at - timedelta(days=7),
                ),
            ],
            at=at,
        )

        alert = spend_alert(summary, threshold_usd=45.0)

        assert alert is not None and "reconcile" in alert["message"]


# ---------------------------------------------------------------------------------------------
# The shells
# ---------------------------------------------------------------------------------------------


def _seed_project(tmp_path, *jobs: JobManifest) -> Lab:
    store = JobStore(tmp_path)
    for m in jobs:
        store.create(m)
    return Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)


class TestLabListShowsRunningSpend:
    def test_the_total_rides_on_the_listing(self, tmp_path) -> None:
        lab = _seed_project(
            tmp_path,
            _remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=12.5)),
        )

        with patch.object(cli_mod, "_lab", return_value=lab):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["spend"]["realized_usd"] == 12.5
        assert payload["jobs"][0]["job_id"] == "a"  # the existing shape is untouched

    def test_a_stale_running_job_neither_inflates_the_listing_nor_raises_the_alarm(
        self, tmp_path
    ) -> None:
        """End to end on the surface an operator actually reads: the abandoned-manifest shape
        (~40% of the 2026-08 DO supervisors) must not manufacture a $45 crossing."""
        lab = _seed_project(
            tmp_path,
            _remote(
                "stale",
                status=JobState.running,
                timeout="2h",
                cost=CostInfo(hourly_usd=0.40),
                started_at=now() - timedelta(days=7),
            ),
        )

        with patch.object(cli_mod, "_lab", return_value=lab):
            result = runner.invoke(app, ["list", "--spend-alert", "45"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["spend"]["realized_usd"] < 1.5
        assert payload["spend"]["unsupervised_suspect_jobs"] == ["stale"]
        assert payload["spend"]["alert"] is None
        assert "SPEND ALERT" not in result.stderr

    def test_crossing_the_threshold_warns_on_stderr(self, tmp_path) -> None:
        lab = _seed_project(
            tmp_path,
            _remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=47.30)),
        )

        with patch.object(cli_mod, "_lab", return_value=lab):
            result = runner.invoke(app, ["list", "--spend-alert", "45"])

        assert result.exit_code == 0, result.output
        assert "47.30" in result.stderr and "45.00" in result.stderr

    def test_stdout_stays_parseable_json_while_the_alert_fires(self, tmp_path) -> None:
        """stdout carries only JSON — callers parse it. The alarm goes to stderr."""
        lab = _seed_project(
            tmp_path,
            _remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=47.30)),
        )

        with patch.object(cli_mod, "_lab", return_value=lab):
            result = runner.invoke(app, ["list", "--spend-alert", "45"])

        payload = json.loads(result.stdout)  # would raise if the warning leaked into stdout
        assert payload["spend"]["alert"]["threshold_usd"] == 45.0
        # The human-readable line is a *field* of that JSON; what must never happen is it being
        # printed alongside the payload on stdout, which is where it would break the parse.
        assert result.stdout.lstrip().startswith("{")
        assert "SPEND ALERT" in result.stderr

    def test_below_the_threshold_nothing_is_printed(self, tmp_path) -> None:
        lab = _seed_project(
            tmp_path,
            _remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=1.0)),
        )

        with patch.object(cli_mod, "_lab", return_value=lab):
            result = runner.invoke(app, ["list", "--spend-alert", "45"])

        assert "SPEND ALERT" not in result.stderr
        assert json.loads(result.stdout)["spend"]["alert"] is None


class TestTheMcpSurfaceMatches:
    def test_list_returns_the_same_spend_block(self, tmp_path) -> None:
        import asyncio

        from fastmcp import Client

        from lab.mcp_server import build_server

        lab = _seed_project(
            tmp_path,
            _remote("a", status=JobState.succeeded, cost=CostInfo(actual_usd=47.30)),
        )
        server = build_server(lab)

        async def go() -> dict:
            async with Client(server) as c:
                return (await c.call_tool("list", {"spend_alert": 45.0})).data

        data = asyncio.run(go())

        assert data["spend"]["realized_usd"] == 47.30
        # No stderr channel to an agent, so the alert has to ride in the payload.
        assert data["spend"]["alert"]["over_by_usd"] == pytest.approx(2.30, abs=1e-6)

    def test_status_carries_the_derived_fields(self, tmp_path) -> None:
        import asyncio

        from fastmcp import Client

        from lab.mcp_server import build_server

        lab = _seed_project(
            tmp_path,
            _remote(
                "s1",
                status=JobState.failed,
                cost=None,
                end_reason="provisioning exceeded 900s (host never reached UP)",
                started_at=now() - timedelta(minutes=15),
                ended_at=now() - timedelta(minutes=1),
            ),
        )
        server = build_server(lab)

        async def go() -> dict:
            async with Client(server) as c:
                return (await c.call_tool("status", {"job_id": "s1"})).data

        data = asyncio.run(go())

        assert data["is_failed_launch"] is True
        assert data["failed_launch_reason"] == "never_reached_up"
        assert 830 <= data["age_s"] <= 850  # 14 minutes of provisioning, then it died
