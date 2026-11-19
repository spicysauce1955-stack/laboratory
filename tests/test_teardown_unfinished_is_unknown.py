"""An outcome nobody could read is ``unknown`` (exit 6), never ``failed`` (exit 3) (review finding).

``failed`` is kept rare on purpose (R10): it means "a destroy was definitively refused, a machine
is billing". Two paths still minted it from answers that were not definitive:

* **A GCE delete still running.** ``zoneOperations.wait`` blocks server-side for ~2 minutes and
  returns the operation whether or not it is DONE. A delete GCE accepted and had not finished
  raised and was recorded ``failed``. It is now re-polled; one still not done after that is
  ``unknown`` — the delete is in flight, and GCE alone can say how it ends.
* **A sky error ``classify_sky_error`` rates ``unknown`` (a timeout, a transport error), then a
  provider-direct fallback that could not answer either.** Only the *undecodable* reply was
  treated as uncertain; any other uncertain sky error fell through to ``failed``. Now ``failed``
  needs every sky.down attempt to have been a definitive refusal; otherwise nobody knows.
"""

from __future__ import annotations

from typing import Any

import pytest

import lab.backends.skypilot as sp

CLUSTER = "lab-x-20261119-101010-abcdef"
NODE = {
    "name": "lab-x-20261119-101010-a-46-3dd12990-head-abcd1234-compute",
    "zone": "us-central1-a",
    "status": "RUNNING",
    "labels": {"lab-job-id": "20261119-101010-abcdef"},
}


class _Exec:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def execute(self) -> dict[str, Any]:
        return self.result


class _Compute:
    """A delete GCE accepts; ``zoneOperations.wait`` answers each of ``statuses`` in turn."""

    def __init__(self, *statuses: str) -> None:
        self.statuses = list(statuses)
        self.waits = 0

    def instances(self) -> "_Compute":
        return self

    def delete(self, **kw: Any) -> _Exec:
        return _Exec({"name": "op-1", "status": "PENDING"})

    def zoneOperations(self) -> "_Compute":  # noqa: N802 — mirrors googleapiclient
        return self

    def wait(self, **kw: Any) -> _Exec:
        self.waits += 1
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return _Exec({"name": "op-1", "status": status})


class _Sky:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def down(self, cluster: str) -> None:
        raise self.exc

    def get(self, x: Any) -> Any:
        return x


class ClusterDoesNotExist(Exception):
    """Matched by type name, as lab._skycompat does: a definitive refusal."""


def _gcp(monkeypatch: pytest.MonkeyPatch, compute: _Compute) -> None:
    monkeypatch.setattr(sp, "_get_gcp_compute", lambda: (compute, "proj"))
    monkeypatch.setattr(sp, "list_gcp_instances", lambda *a, **k: [NODE])


class TestAnUnfinishedGceDelete:
    def test_still_running_after_the_wait_is_unknown_not_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reviewer's reproduction: the delete is accepted, the op is still RUNNING."""
        compute = _Compute("RUNNING")
        _gcp(monkeypatch, compute)

        out = sp.robust_teardown(_Sky(RuntimeError("503 from API server")), CLUSTER, backoffs=(), cloud="gcp")

        assert out["status"] == "unknown", out["error"]
        assert compute.waits > 1, "the operation was not re-polled before giving up"

    def test_a_delete_that_finishes_on_a_re_poll_is_succeeded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        compute = _Compute("RUNNING", "DONE")
        _gcp(monkeypatch, compute)

        out = sp.robust_teardown(_Sky(RuntimeError("503 from API server")), CLUSTER, backoffs=(), cloud="gcp")

        assert out["status"] == "succeeded" and out["gcp_destroyed"] == [NODE["name"]]

    def test_a_delete_gce_reports_as_failed_is_still_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No regression: a definitive GCE error is the alarm `failed` exists for."""

        class _Refused(_Compute):
            def wait(self, **kw: Any) -> _Exec:
                return _Exec({"name": "op-1", "status": "DONE",
                              "error": {"errors": [{"message": "RESOURCE_IN_USE"}]}})

        _gcp(monkeypatch, _Refused("DONE"))

        out = sp.robust_teardown(_Sky(RuntimeError("503 from API server")), CLUSTER, backoffs=(), cloud="gcp")

        assert out["status"] == "failed"


def _fallback_cannot_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def _down(*a: Any, **k: Any) -> Any:
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(sp, "_gcp_destroy_matching", _down)
    monkeypatch.setattr(sp, "_do_destroy_matching", _down)
    monkeypatch.setattr(sp, "_vast_destroy_matching", _down)


@pytest.mark.parametrize("cloud", ["gcp", "do", "vast", "lambda"])
def test_an_uncertain_sky_error_with_no_verification_is_unknown(
    monkeypatch: pytest.MonkeyPatch, cloud: str
) -> None:
    _fallback_cannot_answer(monkeypatch)

    out = sp.robust_teardown(_Sky(TimeoutError("read timed out")), CLUSTER, backoffs=(), cloud=cloud)

    assert out["status"] == "unknown"


@pytest.mark.parametrize("cloud", ["gcp", "do", "vast", "lambda"])
def test_a_definitive_refusal_with_no_verification_is_still_failed(
    monkeypatch: pytest.MonkeyPatch, cloud: str
) -> None:
    _fallback_cannot_answer(monkeypatch)

    out = sp.robust_teardown(_Sky(ClusterDoesNotExist("gone")), CLUSTER, backoffs=(), cloud=cloud)

    assert out["status"] == "failed"


def test_any_uncertain_attempt_makes_the_whole_outcome_uncertain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout, then a refusal: the first attempt may well have destroyed the cluster — which
    is exactly why the second one finds nothing to destroy."""
    _fallback_cannot_answer(monkeypatch)
    monkeypatch.setattr(sp.time, "sleep", lambda s: None)
    errors = iter([TimeoutError("read timed out"), ClusterDoesNotExist("gone")])

    class _Flaky:
        def down(self, cluster: str) -> None:
            raise next(errors)

        def get(self, x: Any) -> Any:
            return x

    out = sp.robust_teardown(_Flaky(), CLUSTER, backoffs=(1,), cloud="gcp")

    assert out["status"] == "unknown"
