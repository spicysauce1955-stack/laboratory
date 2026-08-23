"""SkyPilot client/server compatibility — "it failed" vs "we couldn't read the reply".

On 2026-08-20 ``lab reconcile --apply --yes`` destroyed seven *running* DigitalOcean droplets and
then reported that it had destroyed nothing, exiting 0. The DO action log proves every droplet
died between 08:20:32 and 08:23:03 UTC. What actually happened is that a 0.12.3 client talked to a
0.13.0 API server: the server ran the teardown, pickled a reply referencing
``sky.core.user_initiated_down``, and the older client could not unpickle a symbol its own
``sky.core`` does not define::

    AttributeError: Can't get attribute 'user_initiated_down' on <module 'sky.core' from '...'>

``robust_teardown`` and ``Lab.reconcile`` both wrap ``sky.get(sky.down(...))`` in
``except Exception``, so a *successful* destroy was recorded as a failed one — and, worse, the
reverse is now equally possible. Once a decode error and a real teardown failure look identical,
``teardown_status="failed"`` stops being a money alarm (FR-C2) and becomes noise.

The fix cannot be "retry" or "assume success". It is to stop pretending the outcome is binary.
This module answers two questions and nothing else:

* **Are the two halves of SkyPilot the same version?** (:func:`sky_versions`) Skew is the
  precondition for the whole failure mode, and it is knowable *before* a destroy.
* **Given the exception that came out, did the call probably land?**
  (:func:`classify_sky_error`) Three answers, never two: ``failed`` when we are confident the
  operation did not happen, ``undecodable_response`` when the operation almost certainly ran and
  only the reply was unreadable, and ``unknown`` for everything else.

``unknown`` is the load-bearing one. A caller that can say "the outcome of this destroy is
unknown — verify against the provider" is honest; one that must pick success or failure will be
confidently wrong roughly half the time, which is what the incident cost.

Two rules the module holds to:

* **Detection never raises.** ``lab doctor``'s principle — "only definitive negatives block; a
  check that cannot answer is ``skip``" — applies to the machinery, not to the verdict. A crash
  here would take down teardown itself, which is worse than no check at all.
* **But "cannot determine" is not "compatible".** That asymmetry is deliberate. An unverifiable
  server is precisely the situation that produced a false all-clear, so it reports
  ``compatible=False``. Blocking on that is the caller's decision, not this module's.

No cloud calls, no credentials: the only I/O is the local ``/api/health`` request the sky client
already makes on its own, and even that is skipped when sky has already recorded the peer version.
"""

from __future__ import annotations

import pickle
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

# How the sky client reports "I have never spoken to a server" from its remote-version contextvar.
_UNKNOWN_REMOTE = "unknown"

# Leading ``<major>.<minor>`` of a version string. Deliberately tolerant: sky's readable versions
# include ``1.0.0-dev0`` and ``1.0.0-dev0 (commit: abc1234)``.
_VERSION_RE = re.compile(r"^\s*v?(\d+)\.(\d+)")

# pickle's "the payload names a symbol I do not have" errors. CPython words this as
# "Can't get attribute 'X' on <module 'm' from '...'>" and, for dotted lookups on newer
# interpreters, "Can't resolve path 'a.b' on module 'm'". Both mean the same thing here.
_MISSING_SYMBOL_RE = re.compile(
    r"can't (?:get attribute|resolve path) ['\"]?[^'\"]+['\"]? on <?module", re.IGNORECASE
)

# Exceptions sky raises *instead of* performing the call. Every one of these is thrown by the
# client's pre-flight (``check_server_healthy``) or by the server rejecting the request outright,
# so the cluster provably was not touched — matching on the type *name* rather than importing
# ``sky.exceptions`` keeps this module importable without the optional extra.
_DEFINITE_FAILURES = frozenset(
    {
        "ClusterDoesNotExist",
        "ApiServerConnectionError",
        "ApiServerAuthenticationError",
        "APIVersionMismatchError",
        "APINotSupportedError",
        "PermissionDeniedError",
        "UserRequestRejectedByPolicy",
        "ClusterOwnerIdentityMismatchError",
        "InvalidClusterNameError",
    }
)

# Of those, the ones a backoff cannot help. "Did the call happen" and "could waiting change the
# answer" are different questions, and only the second one decides whether to retry.
#
# Everything here describes a state that is settled for the lifetime of this teardown: a cluster
# absent from the registry does not appear, a malformed name does not become valid, credentials
# and policy do not change, and two mismatched sky versions do not converge. Retrying them cost
# eight jobs four minutes each on 2026-08-23 (see `tests/test_teardown_retry_futility.py`).
#
# ``ApiServerConnectionError`` is deliberately absent even though it is a definite failure: a
# restarting or briefly-unreachable API server is the exact case the backoff exists to ride out.
_UNRETRYABLE = frozenset(
    {
        "ClusterDoesNotExist",
        "ApiServerAuthenticationError",
        "APIVersionMismatchError",
        "APINotSupportedError",
        "PermissionDeniedError",
        "UserRequestRejectedByPolicy",
        "ClusterOwnerIdentityMismatchError",
        "InvalidClusterNameError",
    }
)

# Re-probing a server we could not reach, forever, would make one blip permanently degrade a
# long-lived supervisor's teardowns to "unknown". Determinate answers are cached for the process
# lifetime; indeterminate ones expire.
_INDETERMINATE_TTL_S = 60.0

SkyCallOutcome = Literal["undecodable_response", "failed", "unknown"]


@dataclass(frozen=True)
class SkyVersions:
    """What the two halves of SkyPilot are running, and whether that is safe.

    ``client``/``server`` are ``None`` when unknowable — sky not installed, or a server that
    could not be reached or would not say. ``compatible`` is ``True`` only on a positive
    determination; see the module docstring for why unknown never means yes.
    """

    client: str | None
    server: str | None
    compatible: bool
    detail: str


@dataclass(frozen=True)
class SkyErrorVerdict:
    """What an exception out of a sky call implies about whether the call actually happened."""

    outcome: SkyCallOutcome
    detail: str


class SkyVersionSkewError(RuntimeError):
    """Raised by :func:`require_compatible_sky` when the client and server cannot be trusted."""

    def __init__(self, versions: SkyVersions) -> None:
        super().__init__(
            f"SkyPilot client/server version skew: {versions.detail} Until the two match, the "
            "outcome of any destroy is unverifiable from the client alone — a successful "
            "teardown can be reported as a failure and vice versa (incident 2026-08-20)."
        )
        self.versions = versions


_cached: SkyVersions | None = None
_cached_at: float = 0.0


def reset_version_cache() -> None:
    """Drop the memoised probe. For tests, and for a caller that just restarted the API server."""
    global _cached, _cached_at
    _cached = None
    _cached_at = 0.0


def sky_versions(*, refresh: bool = False) -> SkyVersions:
    """Report the sky client version, the API server version, and whether they can be trusted.

    Cached: teardown calls this on every attempt and it must not add a round-trip to each one.
    A determinate answer is kept for the life of the process (neither half changes underneath a
    running process); an indeterminate one is retried after :data:`_INDETERMINATE_TTL_S`.

    Never raises — see the module docstring.
    """
    global _cached, _cached_at
    if not refresh and _cached is not None:
        fresh = _cached.server is not None or (time.monotonic() - _cached_at) < _INDETERMINATE_TTL_S
        if fresh:
            return _cached
    try:
        result = _probe()
    except BaseException as e:  # noqa: BLE001 — a crashing safety check is worse than none
        result = SkyVersions(
            client=None,
            server=None,
            compatible=False,
            detail=f"could not determine the SkyPilot versions ({type(e).__name__}: {e}).",
        )
    _cached, _cached_at = result, time.monotonic()
    return result


def require_compatible_sky() -> None:
    """Raise :class:`SkyVersionSkewError` unless client and server are positively compatible."""
    versions = sky_versions()
    if not versions.compatible:
        raise SkyVersionSkewError(versions)


def classify_sky_error(exc: BaseException) -> SkyErrorVerdict:
    """Decide what an exception out of ``sky.get(sky.down(...))`` implies about the call.

    Conservative by construction: only errors sky raises *instead of* doing the work are
    ``failed``, and only the unpickling family — where the client received a reply it could not
    read, which means there was a reply, which means the server ran the request — is
    ``undecodable_response``. Everything else, including timeouts and transport errors, is
    ``unknown``: a read timeout on ``/api/get`` says nothing about the destroy the server is still
    happily executing, and calling that "failed" is the same false alarm in the other direction.

    The chain (``__cause__``/``__context__``) is walked, since sky re-wraps freely.
    """
    for link in _chain(exc):
        name = type(link).__name__
        text = str(link)
        if isinstance(link, pickle.UnpicklingError) or _MISSING_SYMBOL_RE.search(text):
            return SkyErrorVerdict(
                "undecodable_response",
                f"{name}: {text} — the API server executed the request and returned a reply this "
                "client could not unpickle. The operation very likely SUCCEEDED; verify against "
                "the cloud provider rather than trusting this error.",
            )
        if isinstance(link, ImportError) and _names_sky(link):
            return SkyErrorVerdict(
                "undecodable_response",
                f"{name}: {text} — the reply references a sky module this client does not have "
                "(version skew). The operation very likely SUCCEEDED; verify with the provider.",
            )
        if name in _DEFINITE_FAILURES:
            return SkyErrorVerdict(
                "failed",
                f"{name}: {text} — SkyPilot rejected the request rather than performing it, so "
                "nothing was destroyed.",
            )
    return SkyErrorVerdict(
        "unknown",
        f"{type(exc).__name__}: {exc} — not a recognisable decode failure and not a definitive "
        "refusal, so whether the operation ran is unknown. Verify against the cloud provider.",
    )


def is_retryable_sky_error(exc: BaseException) -> bool:
    """Could trying this ``sky.down`` again plausibly give a different answer?

    ``False`` only for the settled states in :data:`_UNRETRYABLE`; everything else -- transport
    errors, timeouts, unreachable servers, anything unrecognised -- is ``True``.

    The default direction is deliberate and matches :func:`classify_sky_error`'s: an unrecognised
    error means we cannot rule out that waiting helps, and one wasted backoff is far cheaper than
    abandoning a teardown that would have succeeded on the second try. Callers must still run
    their provider-direct fallback when this says stop -- "sky has nothing to destroy" is not
    evidence that the provider has nothing to destroy (FR-C2).

    Walks ``__cause__``/``__context__``, since sky re-wraps freely.
    """
    return not any(type(link).__name__ in _UNRETRYABLE for link in _chain(exc))


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _chain(exc: BaseException, *, depth: int = 5) -> list[BaseException]:
    """``exc`` and the exceptions it was raised from, de-duplicated and depth-bounded."""
    out: list[BaseException] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and len(out) < depth and id(cur) not in seen:
        seen.add(id(cur))
        out.append(cur)
        cur = cur.__cause__ or cur.__context__
    return out


def _names_sky(exc: ImportError) -> bool:
    """Whether a failed import is of a sky module — the shape a skewed reply produces.

    A non-sky ``ImportError`` (a missing cloud SDK, say) is not evidence about the destroy, so it
    falls through to ``unknown`` rather than being read as a successful call.
    """
    name = getattr(exc, "name", None) or ""
    return name == "sky" or name.startswith("sky.") or "'sky" in str(exc)


def _probe() -> SkyVersions:
    """Ask the local sky client what it is and what it is talking to.

    Two sources, cheapest first. ``sky.server.versions`` keeps a contextvar the client sets from
    the ``X-SkyPilot-Version`` response header on every exchange — it is the same value that feeds
    the "API server is running in version X, which is newer than your client" warning
    (``sky/server/versions.py:215``), so when this process has already made a sky call the answer
    costs nothing. Otherwise fall back to ``sky.server.common.get_api_server_status()``
    (``sky/server/common.py:503``), the ``GET /api/health`` call sky itself uses; it reaches only
    the local API server, needs no cloud credentials, and returns SkyPilot's own compatibility
    verdict alongside the version.
    """
    try:
        import sky  # local import: skypilot is an optional extra
    except Exception as e:  # noqa: BLE001 — ImportError, but a broken install can raise anything
        return SkyVersions(
            client=None,
            server=None,
            compatible=False,
            detail=f"skypilot is not installed in this environment ({type(e).__name__}).",
        )

    client = str(getattr(sky, "__version__", "") or "") or None
    server, sky_verdict = _server_version()
    if server is None:
        return SkyVersions(
            client=client,
            server=None,
            compatible=False,
            detail=(
                f"could not determine the SkyPilot API server version (client {client}); "
                "treating that as incompatible, since an unverifiable server is exactly the case "
                "that produced a false all-clear."
            ),
        )
    if sky_verdict is not None:
        return SkyVersions(client, server, False, sky_verdict)
    return _compare(client, server)


def _server_version() -> tuple[str | None, str | None]:
    """``(server_version, skypilot's own incompatibility message)``. Both may be ``None``."""
    try:
        from sky.server import versions as sky_versions_mod  # local import: optional extra

        remote = str(sky_versions_mod.get_remote_version() or "")
        if remote and remote != _UNKNOWN_REMOTE:
            return remote, None
    except Exception:  # noqa: BLE001 — private-ish module; fall through to the health endpoint
        pass

    from sky.server import common as sky_server_common  # local import: optional extra

    info: Any = sky_server_common.get_api_server_status()
    version = getattr(info, "version", None)
    status = getattr(getattr(info, "status", None), "value", None)
    if status == "version_mismatch":
        # SkyPilot has already decided the two ends cannot talk; its message is better than ours.
        error = str(getattr(info, "error", "") or "SkyPilot reports a version mismatch.")
        return (str(version) if version else _UNKNOWN_REMOTE), error
    return (str(version) if version else None), None


def _compare(client: str | None, server: str) -> SkyVersions:
    """Compare on ``(major, minor)``.

    Patch releases share the symbols a pickled reply can name, so they pass; a minor bump is where
    ``sky.core`` grows and loses functions, which is the failure this module exists for. Skew in
    *either* direction is unsafe — the payload is pickled by whichever side is newer.
    """
    c, s = _major_minor(client), _major_minor(server)
    if c is None or s is None:
        return SkyVersions(
            client,
            server,
            False,
            f"could not compare SkyPilot versions (client {client!r}, server {server!r}); "
            "treating unparseable as incompatible.",
        )
    if c == s:
        detail = f"skypilot client {client} matches API server {server}."
        return SkyVersions(client, server, True, detail)
    if c < s:
        return SkyVersions(
            client,
            server,
            False,
            f"skypilot client {client} is OLDER than the API server {server}; replies the server "
            "pickles can name symbols this client lacks, which makes a successful call look like "
            f'a failure. Upgrade the client: pip install -U "skypilot=={_base(server)}".',
        )
    return SkyVersions(
        client,
        server,
        False,
        f"skypilot client {client} is NEWER than the API server {server}; restart the server on "
        "the client's version (`sky api stop && sky api start`) or downgrade the client: "
        f'pip install -U "skypilot=={_base(server)}".',
    )


def _major_minor(version: str | None) -> tuple[int, int] | None:
    m = _VERSION_RE.match(version or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def _base(version: str) -> str:
    """Strip sky's ``1.0.0-dev0 (commit: abc1234)`` decoration for use in an install command."""
    return version.split(" (commit:")[0].strip()
