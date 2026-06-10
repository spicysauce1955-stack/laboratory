"""Offer-query derivation + VastPriceFeed over a faked vastai client."""

import lab.scheduler.price as price_mod
from lab.scheduler.price import VastPriceFeed, offer_query


def test_offer_query_from_accelerators():
    q = offer_query("RTX_4090:1", None)
    assert "gpu_name=RTX_4090" in q and "num_gpus>=1" in q
    assert "rentable=true" in q and "rented=false" in q and "reliability>0.95" in q


def test_offer_query_extra_filter_appended():
    assert offer_query("RTX_4090:2", "geolocation in [SE]").endswith("geolocation in [SE]")


def test_best_hourly_min_dph(monkeypatch):
    class FakeClient:
        def search_offers(self, query: str):
            return [{"dph_total": 0.31}, {"dph_total": 0.22}, {"dph_total": None}]

    monkeypatch.setattr(price_mod, "_get_vast_client", lambda: FakeClient())
    assert VastPriceFeed().best_hourly("RTX_4090:1") == 0.22


def test_best_hourly_no_offers(monkeypatch):
    class FakeClient:
        def search_offers(self, query: str):
            return []

    monkeypatch.setattr(price_mod, "_get_vast_client", lambda: FakeClient())
    assert VastPriceFeed().best_hourly("RTX_4090:1") is None
