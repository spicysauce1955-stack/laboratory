"""Supervisor for the SkyPilot backend (spawned detached by SkyPilotBackend.submit).

Performs the blocking ``sky.launch`` (provision + run), records terminal state, rsyncs outputs
back into the run dir, and tears the instance down. Its stdout/stderr are redirected to the job
log file by ``submit``, so SkyPilot's streamed logs become the job logs (FR-D1).

Entry point:  python -m lab.sky_runner <job_dir>
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

from lab import events
from lab._util import actual_cost, duration_seconds, now, parse_duration, timeout_reason
from lab.backends.skypilot import (
    DEFAULT_AUTOSTOP_MIN,
    REMOTE_RUN_DIR,
    TIMEOUT_SENTINEL,
    ProvisionTimeout,
    build_task,
    cluster_name_for,
    confirm_success,
    map_job_status,
    preempted_teardown_confirmed,
    promote_timeout,
    provision_timeout_min,
    provision_with_watchdog,
    tear_down_and_record,
    vast_balance,
    vast_hourly_for_cluster,
)
from lab.models import BackendInfo, CostInfo, JobManifest, JobState
from lab.placement import CapacityMemo
from lab.preemption import classify_terminal
from lab.redact import install_log_redaction
from lab.storage import R2Store, r2_enabled
from lab.store import JobStore


_TERMINAL_NAMES = {"SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_DRIVER", "CANCELLED"}
HEARTBEAT_S = 60.0  # how often the supervisor rsyncs partial results down mid-run (§6c)


def _rec_field(rec: Any, key: str) -> Any:
    return rec.get(key) if isinstance(rec, dict) else getattr(rec, key, None)


def _job_status_name(sky_mod: Any, cluster: str, sky_job_id: int | None) -> str | None:
    recs = sky_mod.get(sky_mod.queue(cluster, skip_finished=False))  # 0.12: RequestId
    for rec in recs:
        if sky_job_id is None or _rec_field(rec, "job_id") == sky_job_id:
            status = _rec_field(rec, "status")
            return getattr(status, "name", str(status).split(".")[-1])
    return None


# The one poll error that can never resolve itself. `lab._skycompat` calls several more
# exceptions "failed" (connection refused, auth, policy, version mismatch), but those are verdicts
# about *the call* — correct for a destroy that provably never left the client, wrong as evidence
# that a remote box has gone away. Reading a local API-server blip as a lost cluster would fail a
# healthy job and walk away from a still-billing instance: the same bug as R8, pointing the other
# way. Matched by type name so this module needs no `sky` import (see `lab._skycompat`).
_CLUSTER_GONE_ERROR = "ClusterDoesNotExist"


def _cluster_lost_reason(exc: BaseException) -> str | None:
    """sky's message iff ``exc`` definitively says the cluster is gone, else ``None``.

    Both halves must hold: :func:`~lab._skycompat.classify_sky_error` must rule the request
    definitively refused rather than merely unreadable or unknown (so retrying is pointless), and
    the refusal must be about the cluster itself. The ``__cause__``/``__context__`` chain is
    walked because sky re-wraps freely.
    """
    from lab._skycompat import classify_sky_error

    if classify_sky_error(exc).outcome != "failed":
        return None
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen and len(seen) < 5:
        seen.add(id(cur))
        if type(cur).__name__ == _CLUSTER_GONE_ERROR:
            return str(cur) or _CLUSTER_GONE_ERROR
        cur = cur.__cause__ or cur.__context__
    return None


def _wait_terminal(
    sky_mod: Any,
    cluster: str,
    sky_job_id: int | None,
    max_wait: float,
    *,
    poll_s: float = 10.0,
    heartbeat_s: float | None = None,
    on_heartbeat: Callable[[], None] | None = None,
) -> tuple[JobState, bool, str | None]:
    """Poll the remote job until terminal — sky.launch (0.12) returns at submit time, not
    completion, so we must wait before fetching artifacts and tearing down.

    If ``heartbeat_s``/``on_heartbeat`` are given, ``on_heartbeat`` is called roughly every
    ``heartbeat_s`` of polling so the supervisor can fetch partial results mid-run; a callback
    error is logged, never fatal (§6c — don't lose ``results.csv`` to a late teardown).

    Returns ``(mapped_state, reached_terminal, lost_reason)``.

    ``reached_terminal`` is True iff the loop broke because the cloud reported a terminal status
    (name in ``_TERMINAL_NAMES``); it is False when the loop exited via the deadline or because
    the cluster vanished. The spot classifier needs to distinguish "the cloud told us it ended"
    from "we gave up waiting" (the latter, on spot, can mean preemption).

    ``lost_reason`` is ``None`` unless polling produced a *definitive* answer that the cluster no
    longer exists, in which case it carries sky's message and the state is ``failed``. Only that
    one answer ends the wait early — see :func:`_cluster_lost_reason`.
    """
    deadline = time.time() + max_wait
    name: str | None = None
    since_beat = 0.0
    reached = False
    lost_reason: str | None = None
    while time.time() < deadline:
        try:
            name = _job_status_name(sky_mod, cluster, sky_job_id)
        except Exception as e:  # noqa: BLE001
            # A poll failure has three meanings and only one of them is "keep waiting".
            # `ClusterDoesNotExist` is definitive: the machine is gone, the job can never reach a
            # terminal status, and waiting out `max_wait` (timeout + 300s) only burns the budget
            # while the manifest claims `running`. Observed 2026-08-20 on job
            # 20260820-071913-be3c72: 65 consecutive such answers, each printed and ignored, on a
            # job that then sat `running` until an external watchdog cancelled it (R8).
            lost_reason = _cluster_lost_reason(e)
            if lost_reason is not None:
                print(f"[lab] cluster is gone, ending wait: {e}")
                break
            print(f"[lab] queue poll error: {e}")
        if name in _TERMINAL_NAMES:
            reached = True
            break
        time.sleep(poll_s)
        if heartbeat_s and on_heartbeat is not None:
            since_beat += poll_s
            if since_beat >= heartbeat_s:
                since_beat = 0.0
                try:
                    on_heartbeat()
                except Exception as e:  # noqa: BLE001
                    print(f"[lab] heartbeat rsync skipped: {e}")
    if lost_reason is not None:
        return JobState.failed, False, lost_reason
    return map_job_status(name or "FAILED"), reached, None


def cluster_up_or_raise(sky_mod: Any, cluster: str) -> bool:
    """Is the SkyPilot cluster UP? **Propagates** API errors instead of reading them as "gone".

    For callers where "gone" is a costly verdict — the scheduler watchdog marks the job failed —
    a status query that *failed* is not a cluster that *vanished*, and conflating the two throws
    away a healthy run. Callers that genuinely want the conservative reading use
    :func:`_cluster_up`.
    """
    recs = sky_mod.get(sky_mod.status(cluster_names=[cluster]))  # 0.12: RequestId -> list
    return _any_up(recs)


def _cluster_up(sky_mod: Any, cluster: str) -> bool:
    """Best-effort: is the SkyPilot cluster still UP?

    Used by the spot classifier to detect a vanished box (an unmanaged spot preemption tears the
    instance away, so ``sky.status`` no longer reports it UP). Deliberately conservative on
    uncertainty: ANY exception or an empty/non-UP result reads as "gone" (returns False). That is
    safe **here** because the classifier only *infers* preemption when there was ALSO no terminal
    cloud status AND the job was spot AND it wasn't a cancel/timeout — every authoritative outcome
    wins first, so a false "gone" can never mislabel a real success/failure/cancel/timeout. That
    reasoning does not transfer: see :func:`cluster_up_or_raise`.
    """
    try:
        return cluster_up_or_raise(sky_mod, cluster)
    except Exception as e:  # noqa: BLE001
        print(f"[lab] cluster status check failed (treating as gone): {e}")
        return False


def _any_up(recs: Any) -> bool:
    for rec in recs or []:
        status = _rec_field(rec, "status")
        name = getattr(status, "name", str(status).split(".")[-1])
        if name == "UP":
            return True
    return False


def _rsync_down(cluster: str, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", f"{cluster}:{remote_dir}/", f"{local_dir}/"],
        check=True,
        timeout=180,
    )


def _hourly_cost(handle: Any) -> float | None:
    """USD/hour for the launched cluster, or None if unavailable (best-effort, FR-I2)."""
    try:
        launched = getattr(handle, "launched_resources", None)
        if launched is not None:
            return float(launched.get_cost(3600))
    except Exception as e:  # noqa: BLE001
        print(f"[lab] cost estimate unavailable: {e}")
    return None


def _resolve_hourly(cluster: str, handle: Any, cloud: str) -> float | None:
    """Compute-only USD/hour for the launched cluster.

    Prefers the rental's real billed price on Vast (``dph_total``, which SkyPilot under-reports
    ~4x). On DO/GCP the catalog is accurate *for the region actually launched into*, which is what
    ``handle.launched_resources`` carries — the pre-launch estimate cannot know that, which is why
    it quotes a band and this quotes a number.
    """
    if cloud == "vast":
        try:
            actual = vast_hourly_for_cluster(cluster)
        except Exception as e:  # noqa: BLE001 — best-effort; the estimate is the fallback
            print(f"[lab] vast price lookup failed, using estimate: {e}")
            actual = None
        if actual is not None:
            return actual
    return _hourly_cost(handle)


def resolve_cost(
    cluster: str, handle: Any, manifest: JobManifest, cloud: str, *, instance_type: str | None
) -> CostInfo:
    """The launched job's billed rate, storage included (FR-I2).

    ``hourly_usd`` is compute + storage. Storage was absent from this number until 2026-08-11 and
    is not a rounding error: SkyPilot's default 256 GB disk runs $0.028-$0.035/hr depending on disk
    type, against a $0.034/hr spot n4-standard-4. Everything downstream reads ``hourly_usd``, so
    folding the disk in there fixes ``estimated_usd``, the dashboard, and the scheduler's spend
    accounting at once.
    """
    from lab.placement import storage_hourly_usd

    compute = _resolve_hourly(cluster, handle, cloud)
    storage = storage_hourly_usd(cloud, manifest.resources.disk_size, instance_type)
    total = None if compute is None else compute + storage
    estimated = actual_cost(total, parse_duration(manifest.resources.timeout))
    disk = manifest.resources.disk_size or 0
    shape = f"{cloud} {instance_type or 'unknown instance type'}"
    priced = "compute unknown" if compute is None else f"compute ${compute:.4f}/hr"
    return CostInfo(
        hourly_usd=total,
        compute_hourly_usd=compute,
        storage_hourly_usd=storage,
        hourly_basis=f"{shape} {priced} + {disk}GiB disk ${storage:.4f}/hr",
        estimated_usd=estimated,
    )


def record_capacity_exhaustion(
    home: Path, manifest: JobManifest, cloud: str, *, error_text: str, log_text: str = ""
) -> list[str]:
    """Remember zones that just reported "no capacity" so the next submit skips them.

    Best-effort and deliberately silent about failure: this runs on the error path of a job that
    has already failed, and nothing here may make that worse. Returns the zones recorded (for
    logging and tests).
    """
    try:
        from lab.placement import CapacityMemo, parse_exhausted_zones, resolve_instance_type

        zones = parse_exhausted_zones(f"{error_text}\n{log_text}")
        if not zones:
            return []
        instance_type = resolve_instance_type(manifest.resources)
        if instance_type is None:
            return []
        CapacityMemo.for_home(home).record(cloud, instance_type, zones)
        print(f"[lab] capacity memo: {cloud}/{instance_type} exhausted in {', '.join(zones)}")
        return zones
    except Exception as e:  # noqa: BLE001 — never worsen an already-failing job
        print(f"[lab] capacity memo not updated: {e}")
        return []


# How much of the job log to scan for exhaustion markers. SkyPilot prints one warning per failed
# zone, so the evidence is near the end and a bounded tail is plenty.
_CAPACITY_LOG_TAIL_BYTES = 200_000


def _remember_capacity(
    store: JobStore, job_id: str, manifest: JobManifest, cloud: str, *, error_text: str
) -> list[str]:
    """Scan this job's log tail (plus the raised error) for capacity exhaustion and memoise it.

    The zones are usually only in the log: SkyPilot logs a per-zone warning during failover and
    then raises a summary error that names none of them.
    """
    log_text = ""
    try:
        path = store.logs_path(job_id)
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - _CAPACITY_LOG_TAIL_BYTES))
            log_text = f.read().decode(errors="replace")
    except OSError:
        pass
    return record_capacity_exhaustion(
        store.home, manifest, cloud, error_text=error_text, log_text=log_text
    )


class TransientLaunchError(RuntimeError):
    """A launch that never reached the cloud (local API server overloaded/cold), exhausted
    after retries. ``end_reason`` gets a ``transient:`` prefix so supervisors can auto-retry
    without string-matching a traceback (field-report #4)."""


_LOCAL_MARKERS = ("127.0.0.1", "localhost")
_CONNECTION_MARKERS = (
    "connection refused",
    "connectionerror",
    "newconnectionerror",
    "failed to establish a new connection",
    "max retries exceeded",
    "remotedisconnected",
    "connection aborted",
)


def is_transient_launch_error(e: BaseException) -> bool:
    """True iff a launch error is a connection failure to the submitter's OWN local SkyPilot API
    server — the one case where retrying is provably safe (the request never reached a provider).

    Requires BOTH a localhost marker and a connection-failure marker: a connection error to a
    cloud endpoint stays a terminal failure (fail-toward-alarm). Pure."""
    text = f"{type(e).__name__}: {e}".lower()
    return any(m in text for m in _LOCAL_MARKERS) and any(
        m in text for m in _CONNECTION_MARKERS
    )


def _launch_with_retry(
    sky_mod: Any,
    task: Any,
    cluster: str,
    *,
    attempts: int | None = None,
    base_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """``sky.launch`` with backoff+jitter retries on transient local-API errors.

    Every attempt uses the SAME cluster name, so a retry can never double-provision: a refused
    connection never reached the server, and SkyPilot keys clusters on their name. Non-transient
    errors re-raise immediately; exhaustion raises :class:`TransientLaunchError`.
    """
    import random

    if attempts is None:
        attempts = int(os.environ.get("LAB_LAUNCH_RETRIES", "3"))
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return sky_mod.launch(
                task,
                cluster_name=cluster,
                down=True,
                idle_minutes_to_autostop=DEFAULT_AUTOSTOP_MIN,
            )
        except Exception as e:  # noqa: BLE001 — classified below
            if not is_transient_launch_error(e):
                raise
            last = e
            if attempt < attempts - 1:
                delay = base_s * (2**attempt) * (0.5 + random.random())
                print(
                    f"[lab] transient launch error (attempt {attempt + 1}/{attempts}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                events.note(
                    "launch.retry", attempt=attempt + 1, backoff_s=delay, error=str(e)
                )
                sleep(delay)
    raise TransientLaunchError(
        f"local SkyPilot API unreachable after {attempts} attempts: {last}"
    ) from last


# GCP failure signatures -> what to actually do about it. Ordered: the first match wins, so put
# the specific causes ahead of the catch-all. Every marker here is text GCP itself emits and that
# SkyPilot passes through, so matching it is diagnosing the real cause rather than guessing.
_GCP_FAILURE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("zone_resource_pool_exhausted", "does not have enough resources"),
        # The lab has no region/zone override, so re-pricing the search is the only lever.
        "the zone ran out of capacity for this machine type, not a setup problem. The lab can't "
        "pin a region, so retry with --spot (it re-prices the optimizer's search and often "
        "reaches a zone with capacity), or resubmit later",
    ),
    (
        # Checked before the generic quota marker: GCP enforces GPU quota at two levels and a
        # fresh project fails the *global* one while its regional quota looks fine, so naming the
        # regional metric here would send the user to the wrong console page. Seen live.
        ("gpus_all_regions",),
        "the GLOBAL 'GPUs (all regions)' quota is 0 — this blocks GPU launches in every region "
        "even when the per-region NVIDIA_*_GPUS quota is non-zero. Request an increase for "
        "GPUS_ALL_REGIONS (Global) in IAM & Admin > Quotas; `lab doctor --cloud gcp --gpu T4:1` "
        "checks both levels before you spend a launch on it",
    ),
    (
        ("quota_exceeded", "quota '", "quota exceeded", "exceeded quota"),
        "a GCP quota is exhausted — request a quota increase for that metric/region in the "
        "console (a fresh project has 0 GPU quota, and GPUs are per-family: NVIDIA_T4_GPUS, "
        "NVIDIA_L4_GPUS, plus the separate global GPUS_ALL_REGIONS)",
    ),
    (
        ("has not been used in project", "service_disabled", "api is disabled"),
        "a required API is off for this project — `gcloud services enable compute.googleapis.com "
        "cloudresourcemanager.googleapis.com` (needs roles/serviceusage.serviceUsageAdmin)",
    ),
    (
        ("billing account", "billing_disabled", "billing is not enabled"),
        "the project has no active billing — enable billing on it in the console, then retry",
    ),
    (
        # The optimizer rejected the spec before touching the cloud — nothing provisioned, nothing
        # billed. Observed live with `--price-cap 0.001`, where the generic setup checklist sent
        # the reader looking at credentials and quota for a problem that was in their own flag.
        ("does not contain any instances satisfying", "no resource satisfying"),
        "no instance type matches this spec, so nothing was provisioned (you were not billed). "
        "Most often a --price-cap below every available option, or a cpus/memory/accelerators "
        "combination this cloud does not offer. `lab doctor --cloud gcp` prints the real price "
        "band for the shape you are asking for",
    ),
    (
        ("could not find any head instance", "failed to set up skypilot runtime"),
        # What capacity exhaustion LOOKS like downstream; say so rather than let it read as a bug.
        "no VM came up. This is usually a *provisioning* failure surfacing as a runtime error — "
        "grep the launch log for ZONE_RESOURCE_POOL_EXHAUSTED or a quota message before "
        "suspecting the lab",
    ),
)


def _gcp_failure_hint(text: str) -> str:
    """Diagnose a GCP provision failure from its error text (pure).

    Vast gets a live diagnosis from its balance API (LAB-BUGS §8); GCP's causes are all named in
    the error GCP already returned, so reading it beats handing back a list of things that might
    be wrong. Falls back to the setup checklist when nothing matches.
    """
    low = text.lower()
    for markers, hint in _GCP_FAILURE_HINTS:
        if any(m in low for m in markers):
            return hint
    return (
        "if this is a GCP setup issue, check `sky check gcp` passes "
        "(`gcloud auth application-default login`), the Compute Engine API is enabled, and "
        "your regional quota covers the accelerator family (per-family quota for L4/T4)"
    )


def provision_failure_reason(generic: str, cloud: str) -> str:
    """Enrich a generic provision-failure message per cloud (§8).

    Vast returns 400 on a depleted balance, surfaced generically — consult the balance and say so.
    For DigitalOcean/GCP, diagnose from the error text.

    **The diagnosis goes first.** ``end_reason`` is truncated to 300 characters on the manifest,
    and SkyPilot's generic message alone exceeds that ("Failed to provision all possible launchable
    resources… To keep retrying… Reasons for provision failures…"). Appending the actionable part
    meant it was reliably cut off — observed live on a GPU launch whose real cause,
    ``Quota 'GPUS_ALL_REGIONS' exceeded``, never reached the manifest at all. Leading with it means
    the one sentence worth reading is the one that survives.
    """
    if cloud == "do":
        return (
            "if this is a DigitalOcean setup issue, check `sky check` shows DO enabled "
            f"(doctl token at ~/.config/doctl/config.yaml) and your DO vCPU quota covers the size "
            f"— {generic}"
        )
    if cloud == "gcp":
        return f"{_gcp_failure_hint(generic)} — {generic}"
    if cloud == "vast":
        bal = vast_balance()
        if bal is not None and bal <= 0:
            return f"Vast account balance is ${bal:.2f} — top up to provision"
    return generic


# ---------------------------------------------------------------------------------------------
# Termination signals (R13 / field report F1)
# ---------------------------------------------------------------------------------------------
# 15 of 52 `supervisor/run` ledger calls on this box never wrote a close line — they read as
# `running-or-died`. They were not dying spontaneously: `lab cancel` stops a supervisor with
# `os.kill(runner_pid, SIGTERM)` and the scheduler watchdog with `os.killpg(pgid, SIGTERM)`, and
# SIGTERM's *default* disposition terminates the interpreter without raising anything. So
# `run_job`'s `except BaseException` — the one place that records an outcome, and now also the
# one place that tears down after an unplanned exit — never ran, and neither did the teardown
# that was in flight. Turning the signal into an exception makes both happen.

_TERMINATION_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGHUP)
# SIGINT is deliberately absent: Python's own handler already raises `KeyboardInterrupt`, which
# `run_job` catches and now tears down on. Replacing a working mechanism could only break it.
# SIGHUP is included even though `submit` spawns the supervisor with `start_new_session=True`
# (so a closing terminal cannot hang it up): an explicit `kill -HUP`, or a service manager
# configured to stop with it, still terminates just as silently as SIGTERM does.

_terminating: str | None = None


class SupervisorTerminated(SystemExit):
    """The supervisor was asked to stop by a signal.

    A ``SystemExit`` subclass on purpose. The ``except Exception`` branches inside :func:`run_job`
    exist to classify *launch* failures, and must not be able to swallow a shutdown request; a
    dedicated type additionally lets the abort path tell "someone signalled us" from "our own code
    raised", which is the distinction the ledger was missing.
    """

    def __init__(self, signum: int) -> None:
        self.signal_name = signal.Signals(signum).name
        super().__init__(128 + signum)  # the conventional exit status for a signalled process

    def __str__(self) -> str:
        return f"supervisor terminated by {self.signal_name}"


def _emit(message: str) -> None:
    """Write a diagnostic straight to fd 2.

    Not ``print``: this is called from a signal handler, and a buffered stream's lock is not
    reentrant — a signal landing while the main thread already holds ``sys.stderr``'s lock would
    deadlock the very process we are trying to shut down cleanly. ``os.write`` takes no
    Python-level lock. By this point fd 2 is the job log (``install_log_redaction``), so the line
    still lands where an operator reads it.
    """
    try:
        os.write(2, message.encode("utf-8", "replace"))
    except OSError:
        pass  # a closed fd must not turn a shutdown into a crash


def _on_termination_signal(signum: int, _frame: FrameType | None) -> None:
    """Label the death in the ledger, then re-raise it into the main thread as an exception."""
    global _terminating
    name = signal.Signals(signum).name
    if _terminating is not None:
        # Already unwinding — and unwinding is what runs the teardown. Raising a second
        # `SystemExit` from here would abort the one thing that stops the machine billing, which
        # is the opposite of the point. Whoever is impatient still has SIGKILL.
        _emit(f"[lab] {name} while already shutting down on {_terminating} — ignored\n")
        return
    _terminating = name
    # Buffered in the call's ring and flushed into `trace` because the outcome is not "ok" — so
    # `lab history --full` can say which signal ended a supervisor, months later.
    events.note("signal", sig=name.removeprefix("SIG"))
    _emit(f"[lab] {name} received — recording the outcome and tearing down\n")
    raise SupervisorTerminated(signum)


def install_signal_handlers() -> list[str]:
    """Claim the signals that would otherwise kill the supervisor silently; return what we got.

    Called from :func:`main` — the ``python -m lab.sky_runner`` entrypoint — and never at import
    time. ``lab.sky_runner`` is imported by the SkyPilot backend, the scheduler tick and the test
    suite; changing SIGTERM's disposition as a side effect of an import would be a nasty surprise
    in any of them, and in ``lab cancel``'s process it would hijack the caller's own shutdown.

    ``signal.signal`` also only works on the main thread. A supervisor that cannot install the
    handler must still supervise, so a refusal is reported and never raised.
    """
    installed: list[str] = []
    for sig in _TERMINATION_SIGNALS:
        try:
            signal.signal(sig, _on_termination_signal)
        except (OSError, RuntimeError, ValueError) as e:
            _emit(f"[lab] could not install the {sig.name} handler: {e}\n")
        else:
            installed.append(sig.name)
    return installed


# The default ladder is 5+15+30+60+120 = 230s of `sky.down` retries before the provider-direct
# fallback even starts. A process that has been signalled cannot count on that much time — a
# system shutdown SIGKILLs 90s later by default — and the fallback is the half that actually asks
# the provider whether a machine is still billing. Spend the budget where it settles the question.
_ABORT_TEARDOWN_BACKOFFS = (5, 15, 30)


def _teardown_on_abort(
    store: JobStore, job_id: str, cluster: str, cloud: str, *, why: str, launched: bool
) -> None:
    """Stop the machine for a supervisor unwinding out of ``_impl`` on an unplanned exception.

    Every *expected* exit from ``_impl`` tears down on its own way out. This covers the ones that
    are not expected — a signal, a Ctrl-C, a bug — which previously recorded nothing and left the
    instance running: FR-C2's leak with no leak signal, since the manifest kept saying ``running``
    and no ``teardown_status`` was ever written.

    Never raises: it runs while another exception is already propagating, and masking that with a
    teardown error would destroy the evidence of why the supervisor died.
    """
    if not launched:
        return  # no launch request was ever issued, so nothing can be billing
    sky_mod = sys.modules.get("sky")
    if sky_mod is None:  # pragma: no cover — implied by `launched`; kept as a hard guard
        return
    try:
        if store.read_manifest(job_id).teardown_status is None:
            tear_down_and_record(
                sky_mod, cluster, store, job_id, cloud, backoffs=_ABORT_TEARDOWN_BACKOFFS
            )
        if store.read_manifest(job_id).status in (JobState.queued, JobState.running):
            # Nothing supervises this job any more, so `running` can only go stale — that is what
            # made `lab wait` hang for its full timeout and drove the operator's 5-minute status
            # poller. A concurrent `lab cancel` sets `cancelled` first; any terminal state stands.
            store.update_manifest(
                job_id, status=JobState.failed, ended_at=now(), end_reason=why[:300]
            )
    except Exception as e:  # noqa: BLE001 — never mask the exception we are unwinding on
        _emit(f"[lab] teardown after abort ({why}) failed: {type(e).__name__}: {e}\n")


def run_job(job_dir: Path, adopt: bool = False) -> int:
    job_dir = Path(job_dir)
    store = JobStore(job_dir.parent)
    job_id = job_dir.name
    install_log_redaction(store.logs_path(job_id))  # scrub secrets before any SkyPilot output
    manifest = store.read_manifest(job_id)
    # Prefer the name recorded at launch over recomputing it. `cluster_name_for` now stamps a
    # project slug, so recomputing would invent a *different* name for any job launched by an
    # older release or from another project directory — and on the adopt path that means
    # supervising, and tearing down, the wrong (or no) machine.
    cluster = str(store.read_runtime(job_id).get("cluster") or cluster_name_for(job_id))
    cloud = manifest.resources.cloud or "vast"

    call = events.begin(
        "supervisor",
        "run",
        {"job_id": job_id, "cloud": cloud, "cluster": cluster, "adopt": adopt},
    )
    call.ref(job_id=job_id)

    # Flipped once a machine may exist, i.e. once a launch has been *asked for* (a request that
    # errors can still have created an instance) or once we have adopted a live cluster. Read by
    # `_teardown_on_abort`, so that a supervisor signalled before it ever reached SkyPilot does
    # not raise a teardown alarm about a machine that was never provisioned.
    machine_requested = adopt

    def _impl() -> int:
        nonlocal machine_requested
        if not adopt:
            started = now()
            store.update_manifest(job_id, status=JobState.running, started_at=started)
        else:
            started = manifest.started_at or now()
            print(f"[lab] adopting running cluster {cluster} (supervisor restart)")

        import sky

        # provision_s is only set in the non-adopt branch (ProvisionTimeout can only be raised
        # from sky.launch / provision_with_watchdog, which are also non-adopt only).  Initialise
        # to 0.0 so the except-ProvisionTimeout error message below is always bound.
        provision_s: float = 0.0
        # Set only by `_wait_terminal`, which is the last statement in the try below; every
        # except-branch returns, so the post-try read is always bound. Declared here so that
        # stays true no matter how the try grows.
        lost_reason: str | None = None

        try:
            if not adopt:
                memo = CapacityMemo.for_home(store.home)
                task = build_task(manifest, workdir=Path.cwd(), memo=memo)
                events.note(
                    "provision.attempt",
                    cloud=cloud,
                    zone=manifest.resources.zone,
                    region=manifest.resources.region,
                    instance=manifest.resources.accelerators,
                )
                # Retries transient local-API failures (submit stampede) before giving up (fieldrep #4).
                machine_requested = True  # from here on, an abort must tear down
                request_id = _launch_with_retry(sky, task, cluster)
                # stream_and_get blocks until the job is submitted (0.12), i.e. until the host is UP.
                # Bound it so a dead Vast offer stuck in "loading" can't hang the supervisor forever
                # (FR-I1). The budget is per-cloud: Vast waits on one host, GCP walks a failover path.
                provision_s = (
                    parse_duration(manifest.resources.provision_timeout)
                    or provision_timeout_min(cloud) * 60
                )
                sky_job_id, handle = provision_with_watchdog(sky, request_id, timeout_s=provision_s)
                # Record cost up-front so a running job already shows it (FR-I2). The host is UP now,
                # so the Vast rental exists — bill at its real dph_total, not SkyPilot's low catalog
                # estimate.
                # Record which instance kind SkyPilot actually launched (spot vs on-demand) — with
                # spot_fallback the optimizer may pick on-demand, and the classifier must only infer
                # preemption for a genuinely-spot launch. None when unknown / on-demand-only.
                launched = getattr(handle, "launched_resources", None)
                launched_spot = getattr(launched, "use_spot", None)
                machine_type = getattr(launched, "instance_type", None)
                region = getattr(launched, "region", None)
                zone = getattr(launched, "zone", None)
                if manifest.resources.use_spot and launched_spot is False:
                    # spot_fallback let the optimizer land on-demand, which on GCP is ~5x the spot
                    # price the user was budgeting for. Say so; nothing surfaced this before.
                    print(
                        "[lab] NOTE: requested spot but launched ON-DEMAND (spot capacity was "
                        "scarce). Pass --no-fallback to refuse this."
                    )
                cost_info = resolve_cost(cluster, handle, manifest, cloud, instance_type=machine_type)
                store.update_manifest(
                    job_id,
                    cost=cost_info,
                    backend=BackendInfo(
                        provisioner="skypilot",
                        machine_type=machine_type,
                        region=region,
                        zone=zone,
                        launched_spot=launched_spot,
                    ),
                )
            else:
                # Adopting a running cluster: re-price it, but keep the estimate agreed at launch —
                # that number was the user's authorisation and must not drift under them.
                cost_info = resolve_cost(
                    cluster, None, manifest, cloud, instance_type=manifest.backend.machine_type
                )
                if manifest.cost is not None and manifest.cost.estimated_usd is not None:
                    cost_info = cost_info.model_copy(
                        update={"estimated_usd": manifest.cost.estimated_usd}
                    )
                sky_job_id = None  # match any job in the cluster queue

            # Wait for the run to actually finish before fetching artifacts / tearing down.
            try:
                sky.tail_logs(cluster, sky_job_id, follow=True)  # streams run logs; blocks till done
            except Exception as e:  # noqa: BLE001
                print(f"[lab] tail_logs issue: {e}")

            if not adopt:
                max_wait = (parse_duration(manifest.resources.timeout) or 3600) + 300
            else:
                total = (parse_duration(manifest.resources.timeout) or 3600) + 300
                elapsed = duration_seconds(started, now()) or 0.0
                max_wait = max(60.0, total - elapsed)

            def _heartbeat() -> None:
                # Best-effort: pull partial results so a late/failed teardown can't lose them (§6c).
                _rsync_down(cluster, REMOTE_RUN_DIR, store.output_dir(job_id))

            raw_final, reached_terminal, lost_reason = _wait_terminal(
                sky,
                cluster,
                sky_job_id,
                max_wait,
                heartbeat_s=HEARTBEAT_S,
                on_heartbeat=_heartbeat,
            )
            final = raw_final
        except ProvisionTimeout:
            _remember_capacity(store, job_id, manifest, cloud, error_text="")
            store.update_manifest(
                job_id,
                status=JobState.failed,
                ended_at=now(),
                end_reason=(
                    f"provisioning exceeded {provision_s:.0f}s "
                    "(host never reached UP — likely a dead Vast offer; resubmit for a fresh host)"
                )[:300],
            )
            tear_down_and_record(sky, cluster, store, job_id, cloud)
            return 1
        except TransientLaunchError as e:
            # The launch never reached a provider — safe to auto-retry; the `transient:` prefix is
            # the machine-readable hint (no traceback string-matching needed).
            reason = f"transient: {provision_failure_reason(f'launch error: {e}', cloud)}"
            store.update_manifest(
                job_id, status=JobState.failed, ended_at=now(), end_reason=reason[:300]
            )
            tear_down_and_record(sky, cluster, store, job_id, cloud)
            return 1
        except Exception as e:  # noqa: BLE001
            _remember_capacity(store, job_id, manifest, cloud, error_text=f"{type(e).__name__}: {e}")
            reason = provision_failure_reason(f"launch error: {e}", cloud)
            store.update_manifest(
                job_id, status=JobState.failed, ended_at=now(), end_reason=reason[:300]
            )
            tear_down_and_record(sky, cluster, store, job_id, cloud)
            return 1

        if lost_reason is not None:
            # The cluster vanished mid-run (R8). Nothing left to fetch over ssh, and nothing for
            # `classify_terminal` to weigh: no terminal status will ever arrive. Record the real
            # cause and go straight to teardown, the same shape as the ProvisionTimeout branch
            # above. Teardown is NOT skipped just because SkyPilot says the cluster is gone —
            # a lost SkyPilot registration is exactly when a provider-side rental is most likely
            # to outlive it, and `robust_teardown`'s fallbacks ask the provider directly (FR-C2).
            # Whatever the heartbeat rsync already pulled stays in the run dir.
            store.update_manifest(
                job_id,
                status=JobState.failed,
                ended_at=now(),
                end_reason=f"cluster disappeared mid-run: {lost_reason}"[:300],
            )
            tear_down_and_record(sky, cluster, store, job_id, cloud)
            return 1

        try:
            _rsync_down(cluster, REMOTE_RUN_DIR, store.output_dir(job_id))
        except Exception as e:  # noqa: BLE001
            print(f"[lab] artifact rsync failed: {e}")

        final = promote_timeout(final, store.output_dir(job_id))  # failed -> timed_out if sentinel
        final = confirm_success(final, store.output_dir(job_id))  # succeeded only if .lab_success present

        # Safety-critical: reconcile the observed terminal state with explicit/authoritative outcomes
        # so an unmanaged-spot preemption is *inferred* only as the lowest-precedence fallback — never
        # over a real cloud terminal, a user cancel, or a timeout (FR spot path). The classifier is a
        # pure function; we compute its six inputs from disk + a single defensive cloud status probe.
        # We pass the *confirmed* state (post promote_timeout/confirm_success) as ``sky_state`` so the
        # success-sentinel integrity downgrade (succeeded->failed without .lab_success) is preserved —
        # the classifier only ever *trusts* a succeeded/failed terminal, never invents one.
        timed_out = (store.output_dir(job_id) / TIMEOUT_SENTINEL).exists()
        fresh = store.read_manifest(job_id)
        cancel_requested = fresh.status == JobState.cancelled
        use_spot = (
            fresh.backend.launched_spot
            if fresh.backend.launched_spot is not None
            else manifest.resources.use_spot
        )
        cluster_gone = not _cluster_up(sky, cluster)

        # GCP-PREEMPT-1: on GCP we don't have to *infer* preemption — GCE states it, so ask. Only
        # consulted when we hold no authoritative terminal of our own, since replacing the inference
        # is the entire point; the probe abstains (None) on any doubt and the inference stands.
        # This adjusts the classifier's inputs; the classifier itself stays pure and unchanged.
        if cloud == "gcp" and not reached_terminal:
            from lab.backends.skypilot import gcp_preemption_state

            probed = gcp_preemption_state(cluster)
            if probed is JobState.failed:
                # GCE: the box stopped and was not preemptible. Feeding this as an authoritative
                # terminal is what stops the scheduler resubmitting — and re-paying for — a job that
                # genuinely failed.
                final, reached_terminal = JobState.failed, True
            elif probed is JobState.preempted:
                # `classify_terminal` never trusts sky_state=preempted directly — it reaches
                # `preempted` only via `use_spot and cluster_gone`. GCE reporting a *preemptible*
                # instance is itself authoritative about spot-ness (better evidence than the
                # manifest's `launched_spot`, which the adopt path never records), so set both or the
                # probe's answer is silently discarded and the job comes out `failed`.
                final, cluster_gone, use_spot = JobState.preempted, True, True

        final = classify_terminal(
            sky_state=final,
            timed_out=timed_out,
            cancel_requested=cancel_requested,
            use_spot=use_spot,
            cluster_gone=cluster_gone,
            reached_terminal=reached_terminal,
        )

        # Push the fetched outputs to durable storage (survives teardown / other machines).
        artifacts_uri = None
        if r2_enabled():
            try:
                r2 = R2Store.from_env()
                if r2 is not None:
                    n = r2.upload_dir(store.output_dir(job_id), job_id)
                    artifacts_uri = r2.uri(job_id)
                    print(f"[lab] uploaded {n} artifact(s) to {artifacts_uri}")
            except Exception as e:  # noqa: BLE001
                print(f"[lab] R2 upload failed: {e}")

        ended = now()
        dur = duration_seconds(started, ended)
        cost = cost_info.model_copy(
            update={
                "duration_seconds": dur,
                "actual_usd": actual_cost(cost_info.hourly_usd, dur),
            }
        )

        if final is JobState.timed_out:
            wall = int(parse_duration(manifest.resources.timeout) or 0)
            end_reason = timeout_reason(wall)
        else:
            end_reason = final.value

        # Respect a concurrent cancel (backend set status=cancelled before killing us).
        if store.read_manifest(job_id).status != JobState.cancelled:
            # final_metrics is snapshotted centrally by the store on the succeeded transition (FR-B4).
            store.update_manifest(
                job_id,
                status=final,
                ended_at=ended,
                exit_code=0 if final == JobState.succeeded else 1,
                end_reason=end_reason,
                artifacts_uri=artifacts_uri,
                cost=cost,
            )

        teardown_ok = tear_down_and_record(sky, cluster, store, job_id, cloud)
        if final is JobState.preempted and not preempted_teardown_confirmed(cloud, cluster):
            # The instance vanished (preemption inferred), but we can't confirm the Vast rental is
            # actually gone — flag it so `lab wait` exits 3 and the operator can run `lab reconcile`
            # before any auto-resubmitter builds on a potentially-still-billing orphan (FR-C2).
            store.update_manifest(
                job_id,
                teardown_status="failed",
                end_reason="preempted but teardown unconfirmed — see `lab reconcile`",
            )
            teardown_ok = False
        return 0 if teardown_ok else 2  # 2 = ran ok but teardown leaked — manifest has details

    def _abort(exc: BaseException, *, outcome: str, why: str) -> None:
        """Record the outcome, *then* stop the machine — in that order, deliberately.

        Teardown can take a minute, and whoever sent a signal may follow it with SIGKILL, which
        nothing can catch. A close line written now is guaranteed; one written after the teardown
        is a gamble against exactly the death this path exists to make visible. The teardown's own
        outcome is durable in the other place that matters — `tear_down_and_record` writes
        `teardown_status` on the manifest, which is what `lab wait` (exit 3/6) and `lab reconcile`
        read.
        """
        events.note("abort", why=why, cluster=cluster, teardown=machine_requested)
        exit_code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else None
        events.finish(call, outcome=outcome, exit_code=exit_code, error=events.error_dict(exc))
        _teardown_on_abort(store, job_id, cluster, cloud, why=why, launched=machine_requested)

    try:
        code = _impl()
    except SupervisorTerminated as e:
        _abort(e, outcome="interrupted", why=str(e))
        raise
    except KeyboardInterrupt as e:
        _abort(e, outcome="interrupted", why="supervisor interrupted (SIGINT)")
        raise
    except BaseException as e:  # noqa: BLE001 — every exit path must be recorded
        _abort(e, outcome="crash", why=f"supervisor crashed: {type(e).__name__}: {e}")
        raise
    events.finish(
        call, outcome=("ok" if code == 0 else "error"), exit_code=code,
    )
    return code


def main(argv: list[str] | None = None) -> int:
    """The supervisor entrypoint: ``python -m lab.sky_runner <job_dir> [--adopt]``.

    A function rather than bare code under ``if __name__ == "__main__"`` so that the signal
    handling installed here is reachable from a real subprocess in the tests — the defect it fixes
    lives in the interpreter's *default* signal disposition, which only a real signal can prove.
    """
    ap = argparse.ArgumentParser(prog="lab.sky_runner")
    ap.add_argument("job_dir", type=Path)
    ap.add_argument(
        "--adopt",
        action="store_true",
        help="re-attach to an already-launched cluster (scheduler watchdog)",
    )
    ns = ap.parse_args(argv)
    install_signal_handlers()
    return run_job(ns.job_dir, adopt=ns.adopt)


if __name__ == "__main__":
    raise SystemExit(main())
