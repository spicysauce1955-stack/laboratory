"""A succeeded job was alarmed as a teardown leak because DO detaches volumes asynchronously.

Field report 2026-08-23, second pass. Job `20260823-093642-0fddf1` **succeeded**, and then:

    teardown_status: "failed"
    end_reason: succeeded | TEARDOWN FAILED for cluster 'lab-tempotr-5171-...':
                volume(s) survived teardown: ResourceExistsError: (None) failed to delete
                volume: attached volume cannot be deleted

`lab wait` on that job exits **3** — "a paid machine may still be billing". Nothing was billing.
Queried against DO ~22 minutes later, the volume was gone and every remaining `lab-*` volume was
attached to a live droplet.

What actually happens: `sky.down` destroys the droplet, DO begins detaching the block volume
**in the background**, and `_do_sweep_leftover_volumes` tries to delete it immediately and exactly
once. If the detach has not landed yet DO answers "attached volume cannot be deleted", and that
single "not yet" is recorded as a permanent "never".

The race was measured live rather than guessed. Job `20260823-093644-873c9d` ended at 12:07:24Z;
its volume still listed as `attached` at 12:07:37 and was gone by 12:07:58 — a settle window of
tens of seconds. That job's teardown recorded `succeeded`, because its sweep happened to run after
the detach landed. Same code, same day, opposite verdicts, decided purely by timing.

Two design points, both learned from the same day's other fixes:

* **Re-list on every pass, don't just re-issue the delete.** The volume vanishing is success just
  as much as our own delete succeeding — and for `0fddf1` something other than this function is
  what finally removed it, so a retry that only re-issued DELETE could still have alarmed on a
  volume that no longer existed.
* **Only the detach message is retryable.** "Not yet" and "never" must not collapse into each
  other in *either* direction: a permission error or a malformed request should alarm on the first
  attempt rather than after a minute of pointless waiting.

This alarm is the R10 signal the whole cost-safety design rests on — rare, and therefore believed.
Firing it on a healthy job that succeeded is how it stops being believed.
"""

from __future__ import annotations

import pytest

from lab.backends import skypilot as m

CLUSTER = "lab-tempotr-5171-20260823-093642-0fddf1"
VOLUME = f"{CLUSTER}-3dd12990-fdb2-head"

# DO's wording, verbatim from the failed job's manifest.
DETACH_MSG = "failed to delete volume: attached volume cannot be deleted"


class ResourceExistsError(Exception):
    """The type pydo raised on 2026-08-23."""


class _Volumes:
    """A DO volumes endpoint that refuses N deletes with the detach message, then accepts.

    ``vanish_after`` instead models the volume disappearing on its own (what really settled
    ``0fddf1``): the listing stops returning it and no delete ever succeeds.
    """

    def __init__(self, volumes, refuse_times=0, error=None, vanish_after=None):
        self._volumes = list(volumes)
        self._refuse_times = refuse_times
        self._error = error or ResourceExistsError(f"(None) {DETACH_MSG}")
        self._vanish_after = vanish_after
        self.deleted: list[str] = []
        self.list_calls = 0
        self.delete_calls = 0

    def list(self, **kw):
        self.list_calls += 1
        if self._vanish_after is not None and self.list_calls > self._vanish_after:
            return {"volumes": []}
        return {"volumes": self._volumes}

    def delete(self, volume_id, **kw):
        self.delete_calls += 1
        if self.delete_calls <= self._refuse_times:
            raise self._error
        self.deleted.append(volume_id)
        self._volumes = [v for v in self._volumes if str(v["id"]) != str(volume_id)]


class _Client:
    def __init__(self, volumes):
        self.volumes = volumes


def _patch(monkeypatch, volumes):
    client = _Client(volumes)
    monkeypatch.setattr(m, "_get_do_client", lambda: client)
    return client


@pytest.fixture
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(m.time, "sleep", lambda s: slept.append(s))
    return slept


def _one_volume():
    return [{"id": "vol-1", "name": VOLUME, "droplet_ids": [594518796]}]


class TestTheDetachRaceIsNotALeak:
    def test_a_volume_still_detaching_is_retried_then_deleted(self, monkeypatch, no_sleep):
        """The regression: one "not yet" must not become a permanent teardown alarm."""
        client = _patch(monkeypatch, _Volumes(_one_volume(), refuse_times=1))

        deleted, failures = m._do_sweep_leftover_volumes(CLUSTER)

        assert failures == [], f"a settling detach was reported as a leak: {failures}"
        assert client.volumes.deleted == ["vol-1"]
        assert deleted == ["vol-1"]

    def test_a_volume_that_vanishes_on_its_own_is_success(self, monkeypatch, no_sleep):
        """What really settled 0fddf1: gone is gone, whoever removed it."""
        client = _patch(
            monkeypatch, _Volumes(_one_volume(), refuse_times=99, vanish_after=1)
        )

        deleted, failures = m._do_sweep_leftover_volumes(CLUSTER)

        assert failures == [], f"an already-gone volume was reported as a leak: {failures}"
        assert client.volumes.list_calls >= 2, "must re-list, not just re-issue DELETE"

    def test_a_detach_that_never_settles_still_alarms(self, monkeypatch, no_sleep):
        """The alarm must survive for the case it exists for -- a volume that really is stuck."""
        _patch(monkeypatch, _Volumes(_one_volume(), refuse_times=99))

        deleted, failures = m._do_sweep_leftover_volumes(CLUSTER)

        assert failures, "a volume that never became deletable must still alarm (FR-C2)"
        assert "attached volume cannot be deleted" in failures[0]

    def test_the_happy_path_does_not_wait(self, monkeypatch, no_sleep):
        """A volume already detached costs no extra time -- the common case must stay fast."""
        client = _patch(monkeypatch, _Volumes(_one_volume()))

        deleted, failures = m._do_sweep_leftover_volumes(CLUSTER)

        assert failures == []
        assert deleted == ["vol-1"]
        assert no_sleep == [], f"slept {no_sleep} on a volume that deleted first try"
        assert client.volumes.delete_calls == 1

    def test_nothing_matching_is_success_without_waiting(self, monkeypatch, no_sleep):
        """Finding nothing means nothing is billing; it must not spend the retry ladder."""
        _patch(monkeypatch, _Volumes([{"id": "x", "name": "someone-elses-volume"}]))

        deleted, failures = m._do_sweep_leftover_volumes(CLUSTER)

        assert (deleted, failures) == ([], [])
        assert no_sleep == []


class TestOnlyTheDetachMessageIsRetryable:
    def test_a_different_error_alarms_immediately(self, monkeypatch, no_sleep):
        """"Never" must not be waited on: only the detach race earns the ladder."""
        boom = RuntimeError("403 forbidden: insufficient permissions")
        client = _patch(monkeypatch, _Volumes(_one_volume(), refuse_times=99, error=boom))

        deleted, failures = m._do_sweep_leftover_volumes(CLUSTER)

        assert failures, "a permission error is a real failure"
        assert no_sleep == [], "a permission error will not fix itself in 60 seconds"
        assert client.volumes.delete_calls == 1


class TestTheListingContractIsUnchanged:
    def test_a_listing_failure_still_raises(self, monkeypatch, no_sleep):
        """"We could not ask" must stay distinct from "we asked and there is nothing" -- the
        caller turns the first into `unknown` and the second into `succeeded`."""

        class _Boom:
            list_calls = 0
            deleted: list[str] = []

            def list(self, **kw):
                raise RuntimeError("DO API unreachable")

        _patch(monkeypatch, _Boom())

        with pytest.raises(RuntimeError, match="unreachable"):
            m._do_sweep_leftover_volumes(CLUSTER)


class TestAMixedBatchStillGivesTheDetachItsChance:
    """One volume's unrelated failure must not abandon another's detach.

    The ladder continued only while *every* failure in a pass carried the detach marker. With two
    volumes under one cluster prefix — a sibling erroring for any other reason — the loop returned
    at once with the still-detaching volume listed as survived, re-creating the very false
    `teardown_status: "failed"` this module exists to remove.

    Both facts have to survive: the real failure alarms, and the volume that was only mid-detach
    does not.
    """

    def test_a_sibling_error_does_not_alarm_on_the_detaching_volume(self, monkeypatch, no_sleep):
        vols = [
            {"id": "vol-1", "name": VOLUME, "droplet_ids": [1]},
            {"id": "vol-2", "name": VOLUME + "-b", "droplet_ids": []},
        ]

        class _Mixed:
            def __init__(self):
                self.list_calls = 0
                self.deleted: list[str] = []
                self.attempts: dict[str, int] = {}
                self.vols = list(vols)

            def list(self, **kw):
                self.list_calls += 1
                return {"volumes": self.vols}

            def delete(self, volume_id, **kw):
                n = self.attempts.get(volume_id, 0) + 1
                self.attempts[volume_id] = n
                if volume_id == "vol-2":
                    raise RuntimeError("403 forbidden")  # never resolves
                if n <= 1:
                    raise ResourceExistsError(f"(None) {DETACH_MSG}")  # resolves on retry
                self.deleted.append(volume_id)
                self.vols = [v for v in self.vols if v["id"] != volume_id]

        _patch(monkeypatch, _Mixed())

        deleted, failures = m._do_sweep_leftover_volumes(CLUSTER)

        assert "vol-1" in deleted, "the detaching volume must still get its retry"
        assert not any(DETACH_MSG in f for f in failures), (
            f"a volume that merely needed a retry was reported as survived: {failures}"
        )
        assert any("403" in f for f in failures), "the genuinely stuck volume must still alarm"
