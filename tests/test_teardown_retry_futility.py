"""`robust_teardown` retried errors that retrying provably cannot fix.

Field report 2026-08-23. Eight jobs failed to provision and every one of them then spent four
minutes tearing down a cluster that had never existed. The supervisor trace is the same shape
each time -- `20260823-105413-9b5435`, whose whole provision attempt lasted three seconds::

    t=1754   provision.attempt   {"cloud": "vast", "instance": "RTX_4090:1"}
    t=4743   teardown.attempt    {"attempt": 1}
    t=5560   teardown.retry      {"attempt": 1, "error": "ClusterDoesNotExist: ..."}
    ...      (attempts 2..6, backoffs 5s 15s 30s 60s 120s)
    t=240681 teardown.fallback   {"via": "vast", "ok": true}

The launch was rejected by sky's optimizer before any cluster was registered, so `sky.down` had
nothing to find and said so instantly, six times. `TEARDOWN_BACKOFFS` then slept 230 of those 240
seconds waiting for a fact that cannot change.

The knowledge to stop was already in the codebase and simply unused: `_skycompat` lists
`ClusterDoesNotExist` in `_DEFINITE_FAILURES`, and `robust_teardown` already calls
`classify_sky_error` on every failure -- but only to set `last_undecodable`, never to decide
whether another attempt is worth making.

**What must NOT change is the fallback.** `robust_teardown`'s own comment is emphatic that a lost
SkyPilot registration is exactly when a provider-side rental is most likely to outlive it, so
"sky says the cluster is gone" is never accepted as proof that nothing is billing. This narrows
the futile *retries*; the provider-direct destroy that actually settles the question still runs.

The retryable/non-retryable split is not the same question as `_DEFINITE_FAILURES`. That set
answers "did the call happen"; this one answers "could waiting change the answer".
`ApiServerConnectionError` is in both-and-neither: it is a definite failure (nothing was
destroyed) but a restarting API server is precisely what a backoff exists to ride out, so it
stays retryable.
"""

from __future__ import annotations

import pytest

from lab._skycompat import is_retryable_sky_error
from lab.backends import skypilot as m

CLUSTER = "lab-tempotr-5171-20260823-105413-9b5435"

# The real backoffs. Using them (rather than (0,)) is the point: a test that passes only because
# the delays were stubbed to zero would not have caught the four minutes users actually waited.
REAL_BACKOFFS = m.TEARDOWN_BACKOFFS


class ClusterDoesNotExist(Exception):
    """Type name matched by `_skycompat`; sky raises `sky.exceptions.ClusterDoesNotExist`."""


class ApiServerConnectionError(Exception):
    """Also a `_DEFINITE_FAILURES` member -- but a transient one."""


class _Sky:
    def __init__(self, exc):
        self.exc = exc
        self.downs = 0

    def get(self, x):
        return x

    def down(self, cluster):
        self.downs += 1
        raise self.exc


@pytest.fixture
def no_sleep(monkeypatch):
    """Record what the retry loop *would* have slept, without spending it."""
    slept: list[float] = []
    monkeypatch.setattr(m.time, "sleep", lambda s: slept.append(s))
    return slept


@pytest.fixture
def vast_finds_nothing(monkeypatch):
    """The provider-direct fallback runs and confirms no rental matches."""
    calls: list[str] = []

    def _destroy(cluster):
        calls.append(cluster)
        return [], []

    monkeypatch.setattr(m, "_vast_destroy_matching", _destroy)
    return calls


class TestFutileRetriesAreNotMade:
    def test_cluster_does_not_exist_is_attempted_once(self, no_sleep, vast_finds_nothing):
        sky = _Sky(ClusterDoesNotExist(f"Cluster {CLUSTER} does not exist."))

        out = m.robust_teardown(sky, CLUSTER, cloud="vast", backoffs=REAL_BACKOFFS)

        assert sky.downs == 1, f"retried a cluster that cannot appear ({sky.downs} attempts)"
        assert no_sleep == [], f"slept {sum(no_sleep)}s waiting on an unchangeable fact"
        assert out["attempts"] == 1, "reported attempts must be the attempts actually made"

    def test_the_provider_fallback_still_runs(self, no_sleep, vast_finds_nothing):
        """Cutting the retries must not cut the check that settles whether anything bills."""
        sky = _Sky(ClusterDoesNotExist(f"Cluster {CLUSTER} does not exist."))

        out = m.robust_teardown(sky, CLUSTER, cloud="vast", backoffs=REAL_BACKOFFS)

        assert vast_finds_nothing == [CLUSTER], "the vast-direct fallback was skipped"
        assert out["vast_fallback_used"] is True
        assert out["status"] == "succeeded"

    def test_a_transient_server_error_is_still_retried(self, no_sleep, vast_finds_nothing):
        """A restarting API server is what the backoff is *for*; it must not be short-circuited."""
        sky = _Sky(ApiServerConnectionError("connection refused"))

        m.robust_teardown(sky, CLUSTER, cloud="vast", backoffs=REAL_BACKOFFS)

        assert sky.downs == len(REAL_BACKOFFS) + 1
        assert no_sleep == list(REAL_BACKOFFS)

    def test_an_unrecognised_error_is_still_retried(self, no_sleep, vast_finds_nothing):
        """`unknown` means we cannot rule out that waiting helps -- so wait (R10)."""
        sky = _Sky(RuntimeError("read timeout on /api/get"))

        m.robust_teardown(sky, CLUSTER, cloud="vast", backoffs=REAL_BACKOFFS)

        assert sky.downs == len(REAL_BACKOFFS) + 1


class TestRetryabilityClassification:
    @pytest.mark.parametrize(
        "name",
        [
            "ClusterDoesNotExist",
            "InvalidClusterNameError",
            "ClusterOwnerIdentityMismatchError",
            "PermissionDeniedError",
            "ApiServerAuthenticationError",
            "UserRequestRejectedByPolicy",
            "APIVersionMismatchError",
            "APINotSupportedError",
        ],
    )
    def test_settled_states_are_not_retryable(self, name):
        exc = type(name, (Exception,), {})("boom")
        assert is_retryable_sky_error(exc) is False

    @pytest.mark.parametrize(
        "exc",
        [
            ApiServerConnectionError("connection refused"),
            RuntimeError("read timeout"),
            TimeoutError("deadline exceeded"),
        ],
    )
    def test_anything_that_might_settle_differently_is_retryable(self, exc):
        assert is_retryable_sky_error(exc) is True

    def test_a_wrapped_cause_is_seen_through(self):
        """sky re-wraps freely; the classifier walks the chain, so this must too."""
        inner = type("ClusterDoesNotExist", (Exception,), {})("gone")
        outer = RuntimeError("teardown failed")
        outer.__cause__ = inner
        assert is_retryable_sky_error(outer) is False
