"""`lab._skycompat` — the 2026-08-20 "destroyed seven boxes, reported none" regression.

Every test here fakes the `sky` module. Nothing in this file may reach an API server: the whole
point of the module under test is to be trustworthy *when* the local client and the server
disagree, and a test that needs a healthy server can never cover that.
"""

from __future__ import annotations

import pickle
import sys
import types

import pytest
from lab import _skycompat
from lab._skycompat import (
    SkyVersionSkewError,
    classify_sky_error,
    require_compatible_sky,
    sky_versions,
)


@pytest.fixture(autouse=True)
def _clear_version_cache() -> None:
    """`sky_versions()` caches for the process lifetime; pytest is one process."""
    _skycompat.reset_version_cache()
    yield
    _skycompat.reset_version_cache()


# ---------------------------------------------------------------------------
# fake sky plumbing
# ---------------------------------------------------------------------------


def _install_fake_sky(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client: str = "0.12.3",
    server: str | None = "0.12.3",
    status: str = "healthy",
    error: str | None = None,
    probe_raises: BaseException | None = None,
    remote_contextvar: str = "unknown",
) -> dict[str, int]:
    """Register a `sky` package whose API-server probe answers as told.

    Returns a call counter so a test can assert the probe is not repeated: an extra HTTP
    round-trip on every teardown would be a real cost of this safety check.
    """
    calls = {"probe": 0}

    sky = types.ModuleType("sky")
    sky.__version__ = client
    sky.__commit__ = "deadbee"

    def _get_api_server_status(endpoint: str | None = None) -> object:
        calls["probe"] += 1
        if probe_raises is not None:
            raise probe_raises
        return types.SimpleNamespace(
            status=types.SimpleNamespace(value=status),
            version=server,
            commit="cafebabe",
            error=error,
        )

    common = types.ModuleType("sky.server.common")
    common.get_api_server_status = _get_api_server_status

    versions_mod = types.ModuleType("sky.server.versions")
    versions_mod.get_remote_version = lambda: remote_contextvar

    server_pkg = types.ModuleType("sky.server")
    server_pkg.common = common
    server_pkg.versions = versions_mod
    sky.server = server_pkg

    for name, mod in (
        ("sky", sky),
        ("sky.server", server_pkg),
        ("sky.server.common", common),
        ("sky.server.versions", versions_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


# ---------------------------------------------------------------------------
# 1. classification of a failed sky call
# ---------------------------------------------------------------------------


LIVE_ERROR = AttributeError(
    "Can't get attribute 'user_initiated_down' on <module 'sky.core' from "
    "'/home/user/.venv/lib/python3.12/site-packages/sky/core.py'>"
)


def test_live_incident_error_is_undecodable() -> None:
    """The exact exception observed on 2026-08-20, when seven droplets really did die."""
    verdict = classify_sky_error(LIVE_ERROR)
    assert verdict.outcome == "undecodable_response"
    assert "user_initiated_down" in verdict.detail


def test_real_unpickle_of_a_vanished_symbol_is_undecodable() -> None:
    """Not the string — the *family*. Built by actually unpickling a symbol that is gone."""
    mod = types.ModuleType("_skycompat_fake_sky_core")
    mod.__file__ = "/nowhere/_skycompat_fake_sky_core.py"
    sys.modules[mod.__name__] = mod
    try:

        def ghost() -> None: ...

        ghost.__module__ = mod.__name__
        ghost.__qualname__ = "ghost"
        mod.ghost = ghost
        payload = pickle.dumps(ghost)
        del mod.ghost  # the older client does not have the symbol the newer server pickled
        with pytest.raises(AttributeError) as excinfo:
            pickle.loads(payload)
    finally:
        del sys.modules[mod.__name__]
    assert classify_sky_error(excinfo.value).outcome == "undecodable_response"


def test_unpickling_error_is_undecodable() -> None:
    assert classify_sky_error(pickle.UnpicklingError("bad pickle data")).outcome == (
        "undecodable_response"
    )


def test_missing_sky_module_during_decode_is_undecodable() -> None:
    exc = ModuleNotFoundError("No module named 'sky.batch'", name="sky.batch")
    assert classify_sky_error(exc).outcome == "undecodable_response"


def test_missing_unrelated_module_is_unknown() -> None:
    """A non-sky import failure is not evidence the destroy landed."""
    exc = ModuleNotFoundError("No module named 'boto3'", name="boto3")
    assert classify_sky_error(exc).outcome == "unknown"


def test_cluster_does_not_exist_is_failed() -> None:
    """Quoted in FIELD-REPORT-2026-08-20. Note sky derives it from ValueError."""

    class ClusterDoesNotExist(ValueError):
        pass

    verdict = classify_sky_error(ClusterDoesNotExist("Cluster lab-abc does not exist."))
    assert verdict.outcome == "failed"


def test_api_server_connection_error_is_failed() -> None:
    """sky raises this from its pre-flight health check, before the request is ever submitted."""

    class ApiServerConnectionError(RuntimeError):
        pass

    assert classify_sky_error(ApiServerConnectionError("could not connect")).outcome == "failed"


def test_api_version_mismatch_is_failed_not_undecodable() -> None:
    """Also version skew — but sky refuses *before* calling, so nothing was destroyed."""

    class APIVersionMismatchError(RuntimeError):
        pass

    assert classify_sky_error(APIVersionMismatchError("client too old")).outcome == "failed"


def test_unrelated_value_error_is_unknown() -> None:
    assert classify_sky_error(ValueError("something else entirely")).outcome == "unknown"


def test_timeout_is_unknown_not_failed() -> None:
    """A timeout while polling `/api/get` says nothing about the destroy the server is running."""
    assert classify_sky_error(TimeoutError("read timed out")).outcome == "unknown"


def test_wrapped_cause_is_classified_through_the_chain() -> None:
    outer = RuntimeError("failed to decode response")
    outer.__cause__ = LIVE_ERROR
    assert classify_sky_error(outer).outcome == "undecodable_response"


def test_every_verdict_carries_a_detail() -> None:
    for exc in (LIVE_ERROR, ValueError("x"), TimeoutError("y")):
        assert classify_sky_error(exc).detail.strip()


# ---------------------------------------------------------------------------
# 2. version skew detection
# ---------------------------------------------------------------------------


def test_matching_versions_are_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sky(monkeypatch, client="0.12.3", server="0.12.3")
    v = sky_versions()
    assert (v.client, v.server, v.compatible) == ("0.12.3", "0.12.3", True)


def test_patch_difference_is_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sky(monkeypatch, client="0.12.3", server="0.12.9")
    assert sky_versions().compatible is True


def test_client_older_than_server_is_incompatible(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live incident: client 0.12.3, server 0.13.0."""
    _install_fake_sky(monkeypatch, client="0.12.3", server="0.13.0")
    v = sky_versions()
    assert v.compatible is False
    assert v.client == "0.12.3" and v.server == "0.13.0"
    assert "0.13.0" in v.detail and "0.12.3" in v.detail
    assert "pip install" in v.detail  # actionable, not just a complaint


def test_client_newer_than_server_is_incompatible(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sky(monkeypatch, client="0.13.0", server="0.12.3")
    v = sky_versions()
    assert v.compatible is False
    assert v.detail


def test_unavailable_server_version_is_not_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    """The asymmetry: "cannot answer" is a skip for doctor, but never a pass for skew."""
    _install_fake_sky(monkeypatch, server=None, status="unhealthy")
    v = sky_versions()
    assert v.server is None
    assert v.compatible is False
    assert "could not determine" in v.detail.lower()


def test_probe_that_raises_never_escapes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sky(monkeypatch, probe_raises=RuntimeError("boom"))
    v = sky_versions()
    assert v.server is None and v.compatible is False
    assert "could not determine" in v.detail.lower()


def test_sky_version_mismatch_status_is_incompatible(monkeypatch: pytest.MonkeyPatch) -> None:
    """SkyPilot's own verdict is honoured even when the numbers would have passed."""
    _install_fake_sky(
        monkeypatch, client="0.12.3", server="0.12.3", status="version_mismatch", error="too old"
    )
    v = sky_versions()
    assert v.compatible is False
    assert "too old" in v.detail


def test_missing_skypilot_reports_honestly(monkeypatch: pytest.MonkeyPatch) -> None:
    """`None` in sys.modules is how Python spells "this import fails"."""
    monkeypatch.setitem(sys.modules, "sky", None)
    v = sky_versions()
    assert v.client is None and v.server is None and v.compatible is False
    assert "not installed" in v.detail.lower()


def test_result_is_cached_for_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_sky(monkeypatch, client="0.12.3", server="0.12.3")
    assert sky_versions() == sky_versions()
    assert calls["probe"] == 1
    assert sky_versions(refresh=True).compatible is True
    assert calls["probe"] == 2


def test_contextvar_version_avoids_the_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """sky already records the server version it last talked to; prefer it over a new request."""
    calls = _install_fake_sky(
        monkeypatch, client="0.12.3", server="0.9.9", remote_contextvar="0.13.0"
    )
    v = sky_versions()
    assert v.server == "0.13.0"
    assert calls["probe"] == 0


def test_dev_version_string_with_commit_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sky(monkeypatch, client="1.0.0-dev0", server="1.0.0-dev0 (commit: abc1234)")
    assert sky_versions().compatible is True


def test_unparseable_version_is_not_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sky(monkeypatch, client="0.12.3", server="who-knows")
    assert sky_versions().compatible is False


# ---------------------------------------------------------------------------
# 3. require_compatible_sky
# ---------------------------------------------------------------------------


def test_require_is_silent_when_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sky(monkeypatch, client="0.12.3", server="0.12.3")
    require_compatible_sky()  # must not raise


def test_require_raises_actionably_on_skew(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sky(monkeypatch, client="0.12.3", server="0.13.0")
    with pytest.raises(SkyVersionSkewError) as excinfo:
        require_compatible_sky()
    assert "0.13.0" in str(excinfo.value)
    assert excinfo.value.versions.server == "0.13.0"
