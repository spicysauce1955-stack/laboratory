"""Is the always-on scheduler host running the same lab as this client? (pure; no sky, no network)

The scheduler host is deployed independently of every project's venv, so it drifts. It has been
observed running a **pre-v0.5.0** lab while clients were on v0.11.0. Registrations travel as
pydantic models serialised through R2, and pydantic's default is to **ignore unknown fields** — so
an old host reads a new client's registration, silently drops every field its model does not
declare, and writes the truncated entry back. Nothing errors; the queue looks healthy.

That is not hypothetical. ``--price-cap`` was silently killed on the deferred path exactly this
way, and a campaign retrospective went on to conclude that ``lab register`` "has no
``--price-cap-strict``" and "has no commit pin". Both are false — the flag is on ``JobSpec`` and
``Registration.code`` captures commit+dirty at registration time. The most likely truth is that an
old host stripped both on the way through. Nobody could tell, because ``write_heartbeat`` recorded
``at``/``host``/``tick_count``/``launched``/``errors``/``paused`` and **no version at all**: the one
number that would have named the cause was the one thing the host never published.

So the host now stamps its own ``lab_version`` into the heartbeat (additive — a reader that does
not know the field is unaffected), and this module turns the pair of versions into a verdict. It
is deliberately a **diagnostic, not a gate**: refusing to register against a skewed host would put
a hard block on the deferred path, and a false block there is worse than the skew it prevents.

Three answers, not two. "Older" is the dangerous direction, "newer" is a different and milder
problem, and **"cannot tell" is its own answer** — a heartbeat with no version field is not a
mystery, it is a host that predates the field, which is itself evidence of "older". Collapsing
that into "fine" is the exact silence this module exists to end.

Pure and dependency-free (the model here is :mod:`lab.pricing`): no cloud, no ``sky``, no I/O.
Reading the heartbeat and printing the warning stay with their existing owners.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

HEARTBEAT_VERSION_KEY = "lab_version"
"""Heartbeat field carrying the scheduler host's own ``lab.__version__``.

Same key name the event ledger and user notes already use for the same idea, so a reader that has
seen one recognises the other.
"""

# ``MAJOR.MINOR.PATCH``, tolerant of a ``v`` prefix and of anything trailing (``0.11.0.dev1``,
# ``0.11.0+local``). Same shape as `lab._skycompat._major_minor`, extended to the patch component:
# lab ships field-bearing changes in patch releases (v0.7.1), so a two-component compare would call
# a host that is genuinely behind a match. Anything this does not match is "cannot tell".
_VERSION_RE = re.compile(r"^\s*v?(\d+)\.(\d+)\.(\d+)")

_UNINSTALLED = "0.0.0+unknown"
"""``lab.__init__``'s fallback when the package has no metadata (a source tree, never installed).

It parses cleanly as ``0.0.0`` and would make every source-tree client report "the host is newer",
which is a fabricated verdict from a version string that means "I don't know mine". Read it as
unparseable instead.
"""

Verdict = Literal["same", "host_older", "host_newer", "unknown"]


@dataclass(frozen=True)
class SchedulerSkew:
    """How the scheduler host's lab version relates to this client's.

    ``host_version``/``client_version`` are echoed back verbatim (``None`` when absent) so a caller
    can print what it actually compared rather than a re-derived guess; on an ``unknown`` verdict
    they are how you tell the two causes apart — a **missing** field leaves ``host_version`` at
    ``None``, an **unparseable** one carries the offending string.
    """

    verdict: Verdict
    host_version: str | None
    client_version: str | None
    detail: str

    @property
    def may_drop_fields(self) -> bool:
        """Could this host be silently truncating what this client writes?

        True for ``host_older`` *and* ``unknown`` — i.e. everything that is not provably safe. A
        heartbeat with no version comes from a host older than the release that added the field, so
        the absence of evidence is here a weak form of the evidence itself; and an unparseable pair
        leaves the dangerous direction un-excluded. The cost of being wrong is asymmetric: a
        needless line on stderr against a silently dropped ``--price-cap`` that nobody can see.
        """
        return self.verdict in ("host_older", "unknown")


def classify_skew(host_version: str | None, client_version: str | None) -> SchedulerSkew:
    """Compare the scheduler host's lab version with this client's. Never raises.

    Every failure to decide — either side missing, either side unparseable — lands on ``unknown``
    with the reason spelled out, because a comparison that quietly returns "same" when it could not
    read one of its operands is how this was invisible in the first place.
    """
    host, client = _parse(host_version), _parse(client_version)
    if host is None or client is None:
        return SchedulerSkew(
            "unknown",
            host_version,
            client_version,
            _unknown_detail(host_version, client_version),
        )
    if host == client:
        return SchedulerSkew(
            "same",
            host_version,
            client_version,
            f"scheduler host runs lab {host_version}, the same version as this client.",
        )
    if host < client:
        return SchedulerSkew(
            "host_older",
            host_version,
            client_version,
            f"scheduler host runs lab {host_version}, OLDER than this client's {client_version}. "
            "Registrations cross as pydantic models through R2 and an older host silently drops "
            "fields its models do not declare — this is how --price-cap was lost on the deferred "
            f"path. Redeploy the host before trusting a queued job's settings: "
            f"deploy/scheduler/deploy.sh v{client_version}.",
        )
    return SchedulerSkew(
        "host_newer",
        host_version,
        client_version,
        f"scheduler host runs lab {host_version}, NEWER than this client's {client_version}. The "
        "host understands everything this client writes; the exposure is the reverse direction — "
        "entries and mirrored manifests the host writes back may carry fields this client cannot "
        "validate. Upgrade this project's pin to match the host.",
    )


def skew_from_heartbeat(
    heartbeat: Mapping[str, Any] | None, client_version: str | None = None
) -> SchedulerSkew:
    """:func:`classify_skew` against a heartbeat dict, defaulting the client to this process's lab.

    A missing heartbeat, a heartbeat that is not a mapping (the file is external JSON and a
    truncated or hand-edited one can be anything), a missing field or a null field all mean the
    same thing to a caller: the host's version could not be read.
    """
    if client_version is None:
        from lab import __version__

        client_version = __version__
    raw = heartbeat.get(HEARTBEAT_VERSION_KEY) if isinstance(heartbeat, Mapping) else None
    return classify_skew(None if raw is None else str(raw), client_version)


def _unknown_detail(host_version: str | None, client_version: str | None) -> str:
    if host_version is None:
        return (
            "scheduler heartbeat carries no lab_version, so this host predates the release that "
            "added it — it is running a lab older than this client's "
            f"{client_version}, and an older host silently drops registration fields it does not "
            "know (this is how --price-cap was lost on the deferred path). Redeploy the host: "
            f"deploy/scheduler/deploy.sh v{client_version}."
        )
    return (
        f"could not compare lab versions (scheduler host {host_version!r}, client "
        f"{client_version!r}); treating the host as possibly older, since an unreadable version "
        "cannot rule out the direction that silently drops registration fields."
    )


def _parse(version: str | None) -> tuple[int, int, int] | None:
    """``MAJOR.MINOR.PATCH`` as an orderable tuple, or ``None`` for anything unreadable."""
    text = (version or "").strip()
    if not text or text.startswith(_UNINSTALLED):
        return None
    m = _VERSION_RE.match(text)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
