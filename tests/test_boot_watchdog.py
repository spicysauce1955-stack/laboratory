"""The box must bound its own lifetime from boot, not from the entrypoint (FR-I1, FR-C2).

The incident, from ``tempotron-capacity/runs/20260907-143516-62769a/status_snapshot.json``::

    "end_reason": "failed",              # not a timeout message
    "estimated_usd": 1.866667,           # = 0.6222 $/hr * 3h  ->  the cap was 10800s
    "duration_seconds": 13755.774773,    # 3h49m15s  ->  49 minutes past its own cap
    "actual_usd": 2.377541               # for an empty output/ and an empty R2 prefix

The arithmetic that explains it: ``13755.8 - (10800 + SELF_DESTRUCT_MARGIN_S)`` is 2355.8 s, so
the box died at ``run_start + wall + margin`` -- the ``poweroff`` backstop in
:func:`lab.backends.skypilot.build_run_script` -- with 2355.8 s of provisioning, workdir sync and
``uv sync`` in front of it. **Every one of the lab's wall-clock guarantees is anchored to the
moment the entrypoint starts**, so all of them slide by however long the boot phase took, and a
job that hangs *before* the entrypoint is bounded by nothing on the box at all.

The fix is a second, detached watchdog armed at the top of the SkyPilot ``setup`` script -- the
earliest point that runs on the instance -- with two deadlines:

* a **boot** deadline, cleared by the run script the moment the entrypoint phase begins, so a
  hang in provisioning/sync/``uv sync`` cannot bill indefinitely;
* a **total lifetime** deadline, which is what makes the box's own bound cover the whole rental.

The property that keeps it from ever killing a healthy job is an invariant, not a hope: the run
phase can only start if the boot deadline had not fired, so it starts within ``boot_allowance``
of the arming, therefore its GNU ``timeout`` ends it within ``boot_allowance + wall`` -- which is
``SELF_DESTRUCT_MARGIN_S`` inside the total deadline, by construction. That is
:class:`TestAHealthyJobCanNeverHitTheOuterBound`.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from helpers import make_manifest

import lab.backends.skypilot as skypilot_mod
from lab.backends.skypilot import (
    SELF_DESTRUCT_MARGIN_S,
    boot_allowance_s,
    build_boot_watchdog,
    build_run_script,
    build_setup_script,
    lifetime_bound_s,
    provision_timeout_min,
)
from lab.models import ResourceRequest


def _manifest(timeout: str | None = "3h", **res: Any):
    return make_manifest(
        "j-boot", "python x.py", timeout=timeout, resources=ResourceRequest(**res)
    )


# ---------------------------------------------------------------------------------------------
# The bound itself
# ---------------------------------------------------------------------------------------------


class TestTheOuterBoundIsGreaterThanTheInnerCap:
    def test_the_setup_script_carries_a_total_bound_above_the_wall(self) -> None:
        m = _manifest("3h", cloud="vast")
        setup = build_setup_script(m)

        assert f"LAB_WD_TOTAL={lifetime_bound_s(m)}" in setup
        assert lifetime_bound_s(m) > 10800, "the outer bound must exceed the inner cap"

    @pytest.mark.parametrize("cloud", ["vast", "do", "gcp"])
    @pytest.mark.parametrize("timeout", ["30m", "3h", "7h"])
    def test_every_shape_bounds_above_its_own_wall(self, cloud: str, timeout: str) -> None:
        from lab._util import parse_duration

        m = _manifest(timeout, cloud=cloud)
        wall = parse_duration(timeout)
        assert wall is not None
        assert lifetime_bound_s(m) > wall

    def test_no_requested_cap_means_no_invented_one(self) -> None:
        """A job submitted without ``--timeout`` asked for no bound; inventing one here would
        kill exactly the long healthy runs this must never touch."""
        m = _manifest(None, cloud="vast")

        assert lifetime_bound_s(m) is None
        assert build_boot_watchdog(m) == []
        assert "LAB_WD_TOTAL" not in build_setup_script(m)
        assert "LAB_WD_BOOT" not in build_setup_script(m)


class TestTheBootAllowanceReusesTheCalibratedProvisioningBudget:
    """The boot allowance is derived from ``provision_timeout_min``, not invented.

    The supervisor already bounds provision+setup at that budget (``provision_with_watchdog``),
    and it is the better-informed watchdog: it can cancel the request and rotate. A box-side
    backstop that fired *first* would pre-empt it, so the box gives the boot phase twice whatever
    the supervisor gives it, floored so that a slow ``uv sync`` of a multi-GB wheel is never the
    thing that kills the run.
    """

    @pytest.mark.parametrize("cloud", ["vast", "do", "gcp"])
    @pytest.mark.parametrize("pinned", [{}, {"region": "Sweden, SE, EU"}, {"zone": "us-east1-b"}])
    def test_never_below_the_supervisors_own_budget(
        self, cloud: str, pinned: dict[str, str]
    ) -> None:
        m = _manifest("1h", cloud=cloud, **pinned)
        supervisor_budget = provision_timeout_min(cloud, pinned=bool(pinned)) * 60

        assert boot_allowance_s(m) >= supervisor_budget

    def test_an_explicit_provision_timeout_widens_it(self) -> None:
        """``--provision-timeout 90m`` says this launch legitimately needs 90 minutes to come up.
        A boot allowance below that would destroy the host the supervisor is still waiting on."""
        m = _manifest("1h", cloud="vast", provision_timeout="90m")

        assert boot_allowance_s(m) >= 90 * 60
        assert boot_allowance_s(m) > skypilot_mod.BOOT_ALLOWANCE_FLOOR_S

    def test_the_floor_clears_a_slow_uv_sync_by_a_wide_margin(self) -> None:
        """The number that can kill a healthy job. ~5 min of ``uv sync`` was measured on the
        2026-08-20 boxes and the largest ``--provision-timeout`` ever asked for here is 20 min;
        the floor must sit well above both, because firing early loses a working run."""
        assert skypilot_mod.BOOT_ALLOWANCE_FLOOR_S >= 6 * 300
        assert skypilot_mod.BOOT_ALLOWANCE_FLOOR_S >= 2 * 20 * 60

    def test_gcp_gets_more_room_than_vast(self) -> None:
        """GCP's budget pays for the optimizer's zone-by-zone failover walk; vast waits on one
        host. The allowance inherits that difference rather than flattening it."""
        assert boot_allowance_s(_manifest("1h", cloud="gcp")) >= boot_allowance_s(
            _manifest("1h", cloud="vast")
        )


class TestAHealthyJobCanNeverHitTheOuterBound:
    """The regression that would matter most, pinned as arithmetic.

    Worst legitimate case: the boot phase consumes the *entire* boot allowance (any more and the
    boot deadline would have fired instead), then the entrypoint consumes its *entire* wall and
    is killed by GNU ``timeout``. The box is therefore done at ``boot_allowance + wall`` after the
    watchdog was armed, and the total deadline sits ``SELF_DESTRUCT_MARGIN_S`` beyond that.
    """

    @pytest.mark.parametrize("cloud", ["vast", "do", "gcp"])
    @pytest.mark.parametrize("timeout", ["10m", "1h", "3h", "7h", "24h"])
    def test_the_slowest_legal_boot_plus_a_full_cap_still_lands_inside(
        self, cloud: str, timeout: str
    ) -> None:
        from lab._util import parse_duration

        m = _manifest(timeout, cloud=cloud)
        wall = parse_duration(timeout)
        assert wall is not None

        headroom = lifetime_bound_s(m) - (boot_allowance_s(m) + wall)

        assert headroom == SELF_DESTRUCT_MARGIN_S
        assert headroom > 0

    def test_the_existing_run_script_backstop_still_fires_first(self) -> None:
        """It must not fight teardown: the pre-existing ``poweroff`` at ``run_start + wall +
        margin`` is always earlier than the new total deadline (the boot allowance is the whole
        difference), so nothing that used to tear down cleanly now races a new poweroff."""
        from lab._util import parse_duration

        m = _manifest("3h", cloud="vast")
        wall = parse_duration(m.resources.timeout)
        assert wall is not None
        # Both measured from when the watchdog was armed; ``boot`` is however long the boot phase
        # actually took, and the run-script backstop is armed only once that is over.
        for boot in (0, 60, 600, boot_allowance_s(m)):
            inner_backstop = boot + wall + SELF_DESTRUCT_MARGIN_S
            assert inner_backstop <= lifetime_bound_s(m)


# ---------------------------------------------------------------------------------------------
# Where it is armed
# ---------------------------------------------------------------------------------------------


class TestItIsArmedBeforeAnythingThatCanHang:
    def test_the_watchdog_precedes_the_uv_install_and_sync(self) -> None:
        """``curl | sh`` and ``uv sync --frozen`` are the boot-phase steps that hang. Arming
        after them would leave the exact window this exists for uncovered."""
        setup = build_setup_script(_manifest("3h"))

        armed = setup.index("LAB_WD_TOTAL")
        assert armed < setup.index("astral.sh/uv/install")
        assert armed < setup.index("uv sync --frozen")

    def test_arming_cannot_abort_setup(self) -> None:
        """The watchdog is best-effort scaffolding; a ``set -e`` above it would let a missing
        ``setsid`` fail the whole job. It is armed before ``set -e`` is switched on."""
        setup = build_setup_script(_manifest("3h"))

        assert setup.index("LAB_WD_TOTAL") < setup.index("set -e")

    def test_it_is_detached_so_it_outlives_the_ssh_session_and_the_supervisor(self) -> None:
        setup = build_setup_script(_manifest("3h"))

        assert "nohup setsid" in setup
        assert ">/dev/null 2>&1 </dev/null &" in setup

    def test_the_run_script_clears_the_boot_deadline(self) -> None:
        run = build_run_script(_manifest("3h"))

        started = f'touch "{skypilot_mod.BOOT_WATCHDOG_DIR}/started"'
        assert started in run
        # ...before the entrypoint is handed to GNU `timeout`, or a slow first line of the run
        # script would read as a boot-phase hang.
        assert run.index(started) < run.index("timeout --kill-after")

    def test_the_bound_reaches_the_task_that_is_actually_launched(self) -> None:
        """Not just the builder: the string SkyPilot will run on the instance."""
        pytest.importorskip("sky")
        from lab.backends.skypilot import build_task

        m = _manifest("3h", cloud="vast")
        task = build_task(m, workdir=Path("."))

        assert task.setup is not None
        assert f"LAB_WD_TOTAL={lifetime_bound_s(m)}" in task.setup
        assert task.run is not None
        assert f'touch "{skypilot_mod.BOOT_WATCHDOG_DIR}/started"' in task.run

    def test_the_inner_cap_is_untouched(self) -> None:
        """This is an outer backstop, not a replacement: GNU ``timeout`` and its sentinels are
        exactly what they were."""
        run = build_run_script(_manifest("30m"))
        grace = skypilot_mod.TIMEOUT_KILL_GRACE_S

        assert f"timeout --kill-after={grace}s 1800s bash -c" in run
        assert '"$rc" = 124' in run and '"$rc" = 137' in run
        assert f"sleep {1800 + SELF_DESTRUCT_MARGIN_S}" in run  # the old backstop, unchanged


# ---------------------------------------------------------------------------------------------
# What it actually does, run as shell
# ---------------------------------------------------------------------------------------------
#
# The generated script is executed for real, with three substitutions and nothing else:
# `BOOT_WATCHDOG_DIR` (a tmp dir), `BOOT_WATCHDOG_POLL_S` (1s) and `SELF_DESTRUCT_CMD` (touch a
# marker instead of powering the machine off). No cloud, no ssh, no privileged call -- and the
# real loop, deadlines, owner check and marker handling.


def _arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, boot: int, total: int) -> Path:
    """Arm the real generated watchdog with test-sized deadlines. Returns the ``fired`` marker."""
    wd = tmp_path / "wd"
    fired = tmp_path / "fired"
    monkeypatch.setattr(skypilot_mod, "BOOT_WATCHDOG_DIR", str(wd))
    monkeypatch.setattr(skypilot_mod, "BOOT_WATCHDOG_POLL_S", 1)
    monkeypatch.setattr(skypilot_mod, "SELF_DESTRUCT_CMD", f"touch {fired}")
    # Only the two deadlines are stubbed; the script under test is byte-for-byte the real one.
    monkeypatch.setattr(skypilot_mod, "boot_allowance_s", lambda _m: boot)
    monkeypatch.setattr(skypilot_mod, "lifetime_bound_s", lambda _m: total)

    script = "\n".join(build_boot_watchdog(_manifest("3h", cloud="vast"))) + "\n"
    subprocess.run(["bash", "-c", script], check=True, timeout=30)
    return fired


def _wait_for(path: Path, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return path.exists()


class TestTheWatchdogRunsOnTheBox:
    def test_a_job_that_hangs_before_the_entrypoint_is_stopped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The incident: nothing ever touches ``started``, so the boot deadline is the bound."""
        fired = _arm(tmp_path, monkeypatch, boot=1, total=600)

        assert _wait_for(fired, 8.0), "the boot deadline never fired"

    def test_a_run_that_started_is_bounded_by_the_total_not_the_boot_deadline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The healthy-job regression, live: once the run phase begins, passing the boot
        deadline must do nothing at all -- the entrypoint owns the box until the total bound."""
        fired = _arm(tmp_path, monkeypatch, boot=2, total=7)
        # What the run script does on its first line. (Arming *clears* this marker, so it can
        # only be dropped afterwards -- which is also the real order of events on the box.)
        (tmp_path / "wd" / "started").touch()

        time.sleep(4.0)  # well past the boot deadline
        assert not fired.exists(), "the boot deadline killed a job whose entrypoint was running"
        assert _wait_for(fired, 10.0), "the total lifetime bound never fired"

    @pytest.mark.parametrize("marker", ["disarm", "owner"])
    def test_it_retires_itself_rather_than_powering_off_a_reused_box(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str
    ) -> None:
        """A scheduled shutdown left behind on a box someone else is now using is a worse bug
        than the one being fixed. Re-arming (a relaunch onto the same cluster name rewrites
        ``owner``) and an explicit ``disarm`` both retire the previous watchdog."""
        fired = _arm(tmp_path, monkeypatch, boot=3, total=600)
        wd = tmp_path / "wd"
        if marker == "disarm":
            (wd / "disarm").touch()
        else:
            (wd / "owner").write_text("a-later-launch")

        time.sleep(4.0)
        assert not fired.exists(), f"the watchdog ignored the {marker} marker"
