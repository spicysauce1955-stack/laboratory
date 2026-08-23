"""Pinning a region makes provisioning much slower, and the timeout never knew that.

Measured on 2026-08-23 from every `runs/*/logs.txt` on this machine, timing the interval between
SkyPilot's "Launching on ..." and "Cluster launched" lines:

    vast, unpinned        n=8   min  66s   p50 102s   max 209s
    vast, region-pinned   n=1                         max 526s   <-- 2.5x the unpinned max
    do,   unpinned        n=10  min 158s   p50 208s   max 227s

The single pinned launch (`20260823-105743-e62622`, `--region "Sweden, SE, EU"`) took **526s** —
longer than the 480s `vast` default would have allowed. It only survived because the user had
passed `--provision-timeout 20m` by hand. Pinning narrows the pool to one region's offers, so the
optimizer has far fewer hosts to fall back on; the unpinned budget is simply the wrong budget.

This is deliberately *not* a general tightening. An earlier reading of only two launches suggested
cutting the vast default to ~3 min; the wider sample above shows that would kill healthy hosts
(209s unpinned, 526s pinned). The unpinned defaults measure out correctly as they are — 480s is
~2.3x the observed unpinned max — so they are left alone. The bug is the missing distinction, not
the numbers.

The second half is the loss that actually happened. Three jobs died at the **20-minute** timeout
the user supplied, when a healthy host on that cloud needs under four. Nothing told them that a
generous-looking timeout is not free: it is exactly what a *dead* offer costs, every time. So an
override far above the calibrated budget now says so, once, on stderr.
"""

from __future__ import annotations

import pytest

from lab.backends.skypilot import (
    PROVISION_TIMEOUT_MIN_BY_CLOUD,
    provision_timeout_min,
    wasteful_provision_timeout_warning,
)


class TestPinnedRegionsGetMoreRoom:
    @pytest.mark.parametrize("cloud", ["vast", "do"])
    def test_pinning_raises_the_budget(self, cloud: str) -> None:
        assert provision_timeout_min(cloud, pinned=True) > provision_timeout_min(cloud)

    def test_the_pinned_budget_covers_the_observed_pinned_launch(self) -> None:
        """526s really happened. The budget has to clear it with room, not just barely."""
        assert provision_timeout_min("vast", pinned=True) * 60 >= 526 * 1.5

    def test_the_unpinned_budget_still_clears_the_observed_unpinned_max(self) -> None:
        """The regression guard for the 3-minute idea: 209s unpinned must stay comfortable."""
        for cloud in ("vast", "do"):
            assert provision_timeout_min(cloud) * 60 >= 227 * 1.5, cloud

    def test_unpinned_defaults_are_unchanged(self) -> None:
        """This change adds a case; it does not re-tune the ones that measured out fine."""
        assert PROVISION_TIMEOUT_MIN_BY_CLOUD == {"vast": 8, "do": 12, "gcp": 20}

    def test_gcp_is_not_raised_by_pinning(self) -> None:
        """GCP's 20 min pays for the optimizer's zone-by-zone failover walk. Pinning *shortens*
        that walk, so the pinned case cannot need more than the unpinned one."""
        assert provision_timeout_min("gcp", pinned=True) <= provision_timeout_min("gcp")

    def test_an_unknown_cloud_still_answers(self) -> None:
        assert provision_timeout_min("lambda") > 0
        assert provision_timeout_min(None, pinned=True) > 0


class TestAGenerousTimeoutIsNotFree:
    def test_a_large_override_warns(self) -> None:
        """The 2026-08-23 case: 20m on vast, three dead offers, 60 minutes gone."""
        warning = wasteful_provision_timeout_warning(20 * 60, "vast", pinned=False)

        assert warning is not None
        assert "20m" in warning or "20 min" in warning
        assert "dead" in warning.lower()

    def test_a_sensible_override_is_silent(self) -> None:
        assert wasteful_provision_timeout_warning(8 * 60, "vast", pinned=False) is None

    def test_no_override_is_silent(self) -> None:
        assert wasteful_provision_timeout_warning(None, "vast", pinned=False) is None

    def test_a_large_override_is_fine_when_pinned(self) -> None:
        """A pinned launch legitimately needs the room -- warning there would be noise, and
        noise is how a warning stops being read."""
        assert wasteful_provision_timeout_warning(20 * 60, "vast", pinned=True) is None

    def test_the_warning_never_changes_the_timeout(self) -> None:
        """It advises; it must not silently shorten what the user asked for. A timeout that
        quietly disagreed with the flag would be worse than the problem."""
        assert wasteful_provision_timeout_warning(20 * 60, "vast", pinned=False) is not None
        assert provision_timeout_min("vast") == 8  # unchanged by any warning
