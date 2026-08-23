"""Refuse a cap no live offer can meet, before anything is rented.

The scheduler has gated on Vast's live offer feed since deferred scheduling shipped
(`tick.py:517-531`): a registration whose `--max-hourly` is below the cheapest matching offer
simply does not fire. `lab submit` never had that. It handed the cap to SkyPilot's optimizer,
which prices against a catalog that under-reports Vast ~4x, and discovered the real rate only
once the meter was running — three of nine jobs over cap on 2026-08-23.

This closes that gap by reusing the feed the scheduler already uses rather than growing a second
price path.

**Why this does not make the post-launch check redundant.** `best_hourly` prices the *cheapest
matching* offer, and the optimizer need not land on it. So this answers "is your cap impossible?"
— a definitive negative worth refusing on — and cannot answer "will your cap hold?". The
post-launch comparison is what answers that, and both are needed.

Everything else here is the `lab doctor` rule applied to prices: only definitive negatives block.
A feed that errors, is missing, or matches no offer says nothing at all, and a launch must
proceed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import PYTHON

import lab.core as core_mod
from lab.backends.local import LocalBackend
from lab.core import Lab, LabError
from lab.manifest import repo_root
from lab.models import JobSpec, ResourceRequest


class _Feed:
    """Stand-in for `lab.scheduler.price.VastPriceFeed`."""

    def __init__(self, price: float | None) -> None:
        self.price = price
        self.asked: list[str | None] = []

    def best_hourly(self, accelerators: str | None, extra_query: str | None = None) -> float | None:
        self.asked.append(accelerators)
        return self.price


class _BrokenFeed:
    def __init__(self) -> None:
        self.asked: list[str | None] = []

    def best_hourly(self, accelerators: str | None, extra_query: str | None = None) -> float | None:
        self.asked.append(accelerators)
        raise RuntimeError("vast API 503")


def _lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())
    return Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)


def _spec(
    cap: float | None, cloud: str | None = "vast", accel: str | None = "RTX4090:1"
) -> JobSpec:
    return JobSpec(
        code_ref="HEAD",
        command=f"{PYTHON} experiments/example_capacity.py",
        seed=0,
        resources=ResourceRequest(
            cloud=cloud, accelerators=accel, max_hourly_usd=cap, timeout="1h"
        ),
    )


class TestAnImpossibleCapIsRefused:
    def test_it_raises_before_anything_is_rented(self, tmp_path, monkeypatch) -> None:
        feed = _Feed(1.10)
        monkeypatch.setattr(core_mod, "_vast_price_feed", lambda: feed)

        with pytest.raises(LabError, match="above --price-cap"):
            _lab(tmp_path).submit(_spec(0.85), preflight=False)

        assert feed.asked == ["RTX4090:1"], "the feed must be priced for the requested GPU"

    def test_the_error_names_the_cheapest_real_offer(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(core_mod, "_vast_price_feed", lambda: _Feed(1.10))

        with pytest.raises(LabError) as ei:
            _lab(tmp_path).submit(_spec(0.85), preflight=False)

        msg = str(ei.value)
        assert "1.10" in msg and "0.85" in msg
        assert "Nothing was rented" in msg


class TestTheDefaultCloudIsVast:
    """`ResourceRequest.cloud` is None for the default cloud, and the default is Vast.

    Every other site in the codebase spells this `res.cloud or "vast"` (skypilot.py:1522, :1561,
    :1895, :1967; sky_runner.py:1134). The first cut of the gate compared `cloud != "vast"` and so
    skipped a bare `lab submit --accelerators RTX4090:1 --price-cap 0.85` entirely — the exact
    shape of the 2026-08-23 incident — while refusing the explicit `--cloud vast` spelling. Every
    other test in this file passes `cloud="vast"`, which is precisely why none of them saw it.
    """

    def test_an_unset_cloud_is_still_gated(self, tmp_path, monkeypatch) -> None:
        feed = _Feed(1.10)
        monkeypatch.setattr(core_mod, "_vast_price_feed", lambda: feed)

        with pytest.raises(LabError, match="above --price-cap"):
            _lab(tmp_path).submit(_spec(0.85, cloud=None), preflight=False)

        assert feed.asked == ["RTX4090:1"]

    def test_an_unset_cloud_with_a_reachable_cap_launches(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(core_mod, "_vast_price_feed", lambda: _Feed(0.62))
        assert _lab(tmp_path).submit(_spec(0.85, cloud=None), preflight=False)


class TestNothingElseBlocks:
    def test_a_reachable_cap_launches(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(core_mod, "_vast_price_feed", lambda: _Feed(0.62))
        assert _lab(tmp_path).submit(_spec(0.85), preflight=False)

    def test_a_feed_that_cannot_answer_never_blocks(self, tmp_path, monkeypatch) -> None:
        """Only definitive negatives block — `lab doctor`'s rule, applied to prices."""
        monkeypatch.setattr(core_mod, "_vast_price_feed", lambda: _BrokenFeed())
        assert _lab(tmp_path).submit(_spec(0.85), preflight=False)

    def test_a_missing_feed_never_blocks(self, tmp_path, monkeypatch) -> None:
        """vastai-sdk is an optional extra; its absence is not a price verdict."""
        monkeypatch.setattr(core_mod, "_vast_price_feed", lambda: None)
        assert _lab(tmp_path).submit(_spec(0.85), preflight=False)

    def test_no_matching_offer_never_blocks(self, tmp_path, monkeypatch) -> None:
        """`best_hourly` returns None when nothing matches. That is not "too expensive"."""
        monkeypatch.setattr(core_mod, "_vast_price_feed", lambda: _Feed(None))
        assert _lab(tmp_path).submit(_spec(0.85), preflight=False)

    def test_no_cap_skips_the_feed_entirely(self, tmp_path, monkeypatch) -> None:
        feed = _Feed(9.99)
        monkeypatch.setattr(core_mod, "_vast_price_feed", lambda: feed)

        assert _lab(tmp_path).submit(_spec(None), preflight=False)
        assert feed.asked == [], "no cap means there is nothing to check"

    def test_non_vast_clouds_are_untouched(self, tmp_path, monkeypatch) -> None:
        """On DO/GCP the catalog is accurate for the region actually launched into, so the
        optimizer's own enforcement is sound and a Vast-shaped gate would be wrong."""
        feed = _Feed(9.99)
        monkeypatch.setattr(core_mod, "_vast_price_feed", lambda: feed)

        assert _lab(tmp_path).submit(_spec(0.05, cloud="do", accel=None), preflight=False)
        assert feed.asked == [], "the Vast feed must not be consulted for DO"
