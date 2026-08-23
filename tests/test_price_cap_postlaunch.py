"""The lab learned the real price and never checked it against the cap.

``resolve_cost`` has called ``vast_hourly_for_cluster`` since v0.5 and writes the true
``dph_total`` into ``CostInfo.hourly_usd`` — it is the only reason the 2026-08-23 overruns are
visible in the manifests at all. Nothing compared it back to ``max_hourly_usd``, so three jobs
billed over their cap in silence and the two that finished cost $5.50 against an expected ~$1.03.

Verified before writing this: ``grep max_hourly_usd src/lab/sky_runner.py
src/lab/backends/skypilot.py`` returned exactly one hit, the ``sky.Resources`` line. There was no
post-launch price enforcement anywhere on the submit path.

What this must NOT do is kill the job. "Admission-control and stop-launching, never kill" is this
project's rule, and a job the user is watching should not vanish over price without them having
asked for that — hence the last test here, and hence ``--price-cap-strict`` being opt-in.
"""

from __future__ import annotations

import pytest
from helpers import make_manifest

import lab.sky_runner as runner_mod
from lab.models import ResourceRequest


def _manifest(cap: float | None, *, strict: bool = False):
    return make_manifest(
        "pc1",
        "python x.py",
        resources=ResourceRequest(
            accelerators="RTX4090:1", timeout="3h", max_hourly_usd=cap, price_cap_strict=strict
        ),
    )


def _cost(monkeypatch, price, cap, *, strict=False):
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: price)
    return runner_mod.resolve_cost(
        "lab-pc1", None, _manifest(cap, strict=strict), "vast", instance_type="1x-RTX_4090"
    )


class TestTheOverrunIsRecorded:
    def test_the_live_case_is_flagged(self, monkeypatch, capsys) -> None:
        """20260823-101602-f9e849: $2.220/hr against a $0.85 cap."""
        cost = _cost(monkeypatch, 2.220, 0.85)

        assert cost.over_cap is True
        assert cost.cap_hourly_usd == 0.85
        assert "PRICE CAP EXCEEDED" in capsys.readouterr().out

    def test_a_rental_inside_the_cap_is_quiet(self, monkeypatch, capsys) -> None:
        cost = _cost(monkeypatch, 0.736, 0.85)

        assert cost.over_cap is False
        assert "PRICE CAP EXCEEDED" not in capsys.readouterr().out

    def test_no_cap_means_no_verdict(self, monkeypatch) -> None:
        """`None` is "not checked"; `False` is "checked, fine". Collapsing them loses the fact
        that most jobs are never priced against anything at all."""
        cost = _cost(monkeypatch, 2.220, None)

        assert cost.over_cap is None
        assert cost.cap_hourly_usd is None

    def test_an_unreadable_price_does_not_alarm(self, monkeypatch, capsys) -> None:
        """Only definitive negatives. A price we could not read is not an overrun."""
        cost = _cost(monkeypatch, None, 0.85)

        assert cost.over_cap is not True
        assert "PRICE CAP EXCEEDED" not in capsys.readouterr().out

    def test_the_real_billed_rate_is_still_recorded(self, monkeypatch) -> None:
        """The check must not disturb what `resolve_cost` already reports."""
        cost = _cost(monkeypatch, 2.220, 0.85)

        assert cost.compute_hourly_usd == pytest.approx(2.220)


class TestDetectionNeverKills:
    def test_no_teardown_on_the_default_path(self, monkeypatch) -> None:
        """`resolve_cost` reports; it must not acquire a teardown side effect. Killing is Task 4's
        opt-in job, and this project's default rule is never to kill a running job over cost."""
        torn: list[object] = []
        monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: torn.append(a))

        _cost(monkeypatch, 2.220, 0.85)

        assert torn == []
