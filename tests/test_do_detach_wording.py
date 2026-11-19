"""DigitalOcean reworded its detach-race rejection, and every retry ladder went blind (2026-11-19).

Job F was ``lab cancel``led on a DO droplet. Its teardown recorded ``failed`` (``lab wait`` exit 3)::

    volume(s) survived teardown: volume lab-smoke-…-b048d2-3dd12990-ef65-head:
    HttpResponseError: (None) failed to delete volume: cannot delete attached volume

The volume was only mid-detach, and was deletable minutes later. The 2026-08-23 fix taught three
paths to ride that race out — the success-path sweep after ``sky.down``, the DO-direct fallback,
and reconcile's volume delete — but recognised it by one literal string, DO's *old* wording
("attached volume cannot be deleted"). DO now says "cannot delete attached volume", so the
substring test missed, the "not yet" was classified as "never", and the ladder never engaged
anywhere. A false ``failed`` is the alarm R10 says the lab cannot afford.

The exceptions here are real ``azure.core`` errors built from DO's JSON body (see ``fake_do``),
because the status code is part of the evidence: 2026-08-23's error was a 409
(``ResourceExistsError``), 2026-11-19's a plain ``HttpResponseError`` — so the code alone cannot
identify the race either, and only the structured message is stable enough to key on.
"""

from __future__ import annotations

from typing import Any

import pytest
from fake_do import NEW_DETACH, OLD_DETACH, FakeDo, FakeSky, http_error

from lab.backends import skypilot as m

CLUSTER = "lab-smoke-20261119-111134-b048d2"
VOLUME = f"{CLUSTER}-3dd12990-ef65-head"
DROPLET_ID = 594518796

WORDINGS = pytest.mark.parametrize(
    "message,status",
    [
        (NEW_DETACH, 422),  # 2026-11-19, plain HttpResponseError
        (OLD_DETACH, 409),  # 2026-08-23, ResourceExistsError
        (NEW_DETACH, 409),
        (OLD_DETACH, 422),
    ],
    ids=["new-422", "old-409", "new-409", "old-422"],
)


def _account(message: str, status: int, **kw: Any) -> FakeDo:
    """One droplet with its volume attached; the volume detaches 12s after the droplet goes —
    inside the ladder (5+10 = 15s), so a recognised race always settles."""
    kw.setdefault("detach_s", 12.0)
    return FakeDo(
        droplets=[{"id": DROPLET_ID, "name": VOLUME}],
        volumes=[{"id": "vol-1", "name": VOLUME, "droplet_ids": [DROPLET_ID]}],
        detach_message=message,
        detach_status=status,
        **kw,
    )


def _install(monkeypatch: pytest.MonkeyPatch, do: FakeDo) -> FakeDo:
    monkeypatch.setattr(m, "_get_do_client", lambda: do)
    monkeypatch.setattr(m.time, "sleep", do.sleep)
    return do


class TestTheRejectionIsRecognised:
    @WORDINGS
    def test_either_wording_at_either_status_is_the_detach_race(
        self, message: str, status: int
    ) -> None:
        assert m.is_volume_detach_race(http_error(status, message))

    def test_recognised_through_skypilots_rewrap(self) -> None:
        """SkyPilot's DO helpers re-raise pydo errors as ``DigitalOceanError(f"Error: {code}
        {reason}: {message}") from err`` — the chain must still be read."""

        class DigitalOceanError(Exception):
            pass

        inner = http_error(422, NEW_DETACH)
        try:
            try:
                raise inner
            except Exception as err:
                raise DigitalOceanError("Error: 422 Unprocessable Entity: see cause") from err
        except DigitalOceanError as wrapped:
            assert m.is_volume_detach_race(wrapped)

    @pytest.mark.parametrize(
        "exc",
        [
            http_error(403, "You do not have access for the attempted action.", error_id="forbidden"),
            http_error(422, "volume size must be greater than 0"),
            http_error(401, "Unable to authenticate you", error_id="unauthorized"),
            RuntimeError("DO API unreachable"),
        ],
        ids=["403", "422-other", "401", "transport"],
    )
    def test_other_refusals_are_not_mistaken_for_it(self, exc: Exception) -> None:
        """"Never" must not be waited on: only the detach race earns the ladder."""
        assert not m.is_volume_detach_race(exc)

    def test_an_auth_failure_never_reads_as_detaching_whatever_it_says(self) -> None:
        """The status is part of the structure: a 401/403 cannot be a detach in progress."""
        assert not m.is_volume_detach_race(http_error(403, NEW_DETACH, error_id="forbidden"))


class TestTheSuccessPathSweep:
    @WORDINGS
    def test_a_detaching_volume_is_waited_out(
        self, monkeypatch: pytest.MonkeyPatch, message: str, status: int
    ) -> None:
        do = _install(monkeypatch, _account(message, status))
        do.droplets.destroy(droplet_id=DROPLET_ID)  # what sky.down just did

        deleted, failures = m._do_sweep_leftover_volumes(CLUSTER)

        assert failures == [], f"a settling detach was reported as a leak: {failures}"
        assert "vol-1" not in do.vols
        assert do.slept, "the ladder must have engaged"

    def test_job_f_after_a_clean_sky_down_is_not_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The incident end to end: sky.down returns, DO's own background delete of the volume
        does not land, and our sweep meets the reworded rejection."""
        do = _install(monkeypatch, _account(NEW_DETACH, 422, associated_delete=False))

        out = m.robust_teardown(FakeSky(do), CLUSTER, cloud="do", backoffs=(0,))

        assert out["status"] == "succeeded", out["error"]
        assert "vol-1" not in do.vols


class TestTheDoDirectFallback:
    @WORDINGS
    def test_a_detaching_volume_is_waited_out(
        self, monkeypatch: pytest.MonkeyPatch, message: str, status: int
    ) -> None:
        do = _install(monkeypatch, _account(message, status))
        sky = FakeSky(do, down_error=RuntimeError("sky.down: API server 503"))

        out = m.robust_teardown(sky, CLUSTER, cloud="do", backoffs=())

        assert out["status"] == "succeeded", out["error"]
        assert out["do_fallback_used"] is True
        assert not do.live_droplets and "vol-1" not in do.vols


class TestReconcilesVolumeDelete:
    @WORDINGS
    def test_a_detaching_volume_is_waited_out(
        self, monkeypatch: pytest.MonkeyPatch, message: str, status: int
    ) -> None:
        do = _install(monkeypatch, _account(message, status))
        do.droplets.destroy(droplet_id=DROPLET_ID)  # reconcile's sky pass just did this

        gone, failures = m.delete_do_volumes(["vol-1"], client=do)

        assert failures == {}, f"a settling detach was reported as a failure: {failures}"
        assert gone == ["vol-1"] and "vol-1" not in do.vols

    def test_a_terminal_refusal_is_still_reported_without_waiting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _install(monkeypatch, _account(NEW_DETACH, 422, detach_s=0.0))
        do.droplets.destroy(droplet_id=DROPLET_ID)
        do.refuse["vol-1"] = http_error(403, "You do not have access", error_id="forbidden")

        gone, failures = m.delete_do_volumes(["vol-1"], client=do)

        assert gone == [] and "vol-1" in failures
        assert do.slept == [], "a permission error will not fix itself in a minute"
