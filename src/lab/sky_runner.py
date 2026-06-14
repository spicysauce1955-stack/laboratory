"""Supervisor for the SkyPilot backend (spawned detached by SkyPilotBackend.submit).

Performs the blocking ``sky.launch`` (provision + run), records terminal state, rsyncs outputs
back into the run dir, and tears the instance down. Its stdout/stderr are redirected to the job
log file by ``submit``, so SkyPilot's streamed logs become the job logs (FR-D1).

Entry point:  python -m lab.sky_runner <job_dir>
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lab._util import actual_cost, duration_seconds, now, parse_duration
from lab.backends.skypilot import (
    DEFAULT_AUTOSTOP_MIN,
    DEFAULT_PROVISION_TIMEOUT_MIN,
    REMOTE_RUN_DIR,
    TIMEOUT_SENTINEL,
    ProvisionTimeout,
    build_task,
    cluster_name_for,
    confirm_no_rental,
    confirm_success,
    map_job_status,
    promote_timeout,
    provision_with_watchdog,
    tear_down_and_record,
    vast_balance,
    vast_hourly_for_cluster,
)
from lab.models import BackendInfo, CostInfo, JobState
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


def _wait_terminal(
    sky_mod: Any,
    cluster: str,
    sky_job_id: int | None,
    max_wait: float,
    *,
    poll_s: float = 10.0,
    heartbeat_s: float | None = None,
    on_heartbeat: Callable[[], None] | None = None,
) -> tuple[JobState, bool]:
    """Poll the remote job until terminal — sky.launch (0.12) returns at submit time, not
    completion, so we must wait before fetching artifacts and tearing down.

    If ``heartbeat_s``/``on_heartbeat`` are given, ``on_heartbeat`` is called roughly every
    ``heartbeat_s`` of polling so the supervisor can fetch partial results mid-run; a callback
    error is logged, never fatal (§6c — don't lose ``results.csv`` to a late teardown).

    Returns ``(mapped_state, reached_terminal)`` where ``reached_terminal`` is True iff the loop
    broke because the cloud reported a terminal status (name in ``_TERMINAL_NAMES``); it is False
    when the loop exited via the deadline. The spot classifier needs to distinguish "the cloud
    told us it ended" from "we gave up waiting" (the latter, on spot, can mean preemption).
    """
    deadline = time.time() + max_wait
    name: str | None = None
    since_beat = 0.0
    reached = False
    while time.time() < deadline:
        try:
            name = _job_status_name(sky_mod, cluster, sky_job_id)
        except Exception as e:  # noqa: BLE001
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
    return map_job_status(name or "FAILED"), reached


def _cluster_up(sky_mod: Any, cluster: str) -> bool:
    """Best-effort: is the SkyPilot cluster still UP?

    Used by the spot classifier to detect a vanished box (an unmanaged spot preemption tears the
    instance away, so ``sky.status`` no longer reports it UP). Deliberately conservative on
    uncertainty: ANY exception or an empty/non-UP result reads as "gone" (returns False). That is
    safe because the classifier only *infers* preemption when there was ALSO no terminal cloud
    status AND the job was spot AND it wasn't a cancel/timeout — every authoritative outcome wins
    first, so a false "gone" can never mislabel a real success/failure/cancel/timeout.
    """
    try:
        recs = sky_mod.get(sky_mod.status(cluster_names=[cluster]))  # 0.12: RequestId -> list
    except Exception as e:  # noqa: BLE001
        print(f"[lab] cluster status check failed (treating as gone): {e}")
        return False
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


def _resolve_hourly(cluster: str, handle: Any) -> float | None:
    """Prefer the rental's real billed price (Vast ``dph_total``) over SkyPilot's catalog estimate,
    which under-reports Vast ~4x. Falls back to the estimate if the live price is unavailable."""
    try:
        actual = vast_hourly_for_cluster(cluster)
    except Exception as e:  # noqa: BLE001 — best-effort; the estimate is the fallback
        print(f"[lab] vast price lookup failed, using estimate: {e}")
        actual = None
    if actual is not None:
        return actual
    return _hourly_cost(handle)


def provision_failure_reason(generic: str) -> str:
    """Enrich a generic provision-failure message with the Vast balance when that's the cause (§8).

    Vast returns 400 on rentals when the balance is depleted; SkyPilot reports that as a generic
    "no resources" string. If the balance is known and not positive, say so instead.
    """
    bal = vast_balance()
    if bal is not None and bal <= 0:
        return f"Vast account balance is ${bal:.2f} — top up to provision"
    return generic


def run_job(job_dir: Path, adopt: bool = False) -> int:
    job_dir = Path(job_dir)
    store = JobStore(job_dir.parent)
    job_id = job_dir.name
    install_log_redaction(store.logs_path(job_id))  # scrub secrets before any SkyPilot output
    manifest = store.read_manifest(job_id)
    cluster = cluster_name_for(job_id)

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

    try:
        if not adopt:
            task = build_task(manifest, workdir=Path.cwd())
            request_id = sky.launch(
                task,
                cluster_name=cluster,
                down=True,
                idle_minutes_to_autostop=DEFAULT_AUTOSTOP_MIN,
            )
            # stream_and_get blocks until the job is submitted (0.12), i.e. until the host is UP.
            # Bound it so a dead Vast offer stuck in "loading" can't hang the supervisor forever
            # (FR-I1).
            provision_s = (
                parse_duration(manifest.resources.provision_timeout)
                or DEFAULT_PROVISION_TIMEOUT_MIN * 60
            )
            sky_job_id, handle = provision_with_watchdog(sky, request_id, timeout_s=provision_s)
            # Record cost up-front so a running job already shows it (FR-I2). The host is UP now,
            # so the Vast rental exists — bill at its real dph_total, not SkyPilot's low catalog
            # estimate.
            hourly_usd = _resolve_hourly(cluster, handle)
            estimated_usd = actual_cost(hourly_usd, parse_duration(manifest.resources.timeout))
            # Record which instance kind SkyPilot actually launched (spot vs on-demand) — with
            # spot_fallback the optimizer may pick on-demand, and the classifier must only infer
            # preemption for a genuinely-spot launch. None when unknown / on-demand-only.
            launched_spot = getattr(
                getattr(handle, "launched_resources", None), "use_spot", None
            )
            store.update_manifest(
                job_id,
                cost=CostInfo(hourly_usd=hourly_usd, estimated_usd=estimated_usd),
                backend=BackendInfo(provisioner="skypilot", launched_spot=launched_spot),
            )
        else:
            hourly_usd = _resolve_hourly(cluster, None)
            estimated_usd = manifest.cost.estimated_usd if manifest.cost else None
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

        raw_final, reached_terminal = _wait_terminal(
            sky,
            cluster,
            sky_job_id,
            max_wait,
            heartbeat_s=HEARTBEAT_S,
            on_heartbeat=_heartbeat,
        )
        final = raw_final
    except ProvisionTimeout:
        store.update_manifest(
            job_id,
            status=JobState.failed,
            ended_at=now(),
            end_reason=(
                f"provisioning exceeded {provision_s:.0f}s "
                "(host never reached UP — likely a dead Vast offer; resubmit for a fresh host)"
            )[:300],
        )
        tear_down_and_record(sky, cluster, store, job_id)
        return 1
    except Exception as e:  # noqa: BLE001
        reason = provision_failure_reason(f"launch error: {e}")
        store.update_manifest(
            job_id, status=JobState.failed, ended_at=now(), end_reason=reason[:300]
        )
        tear_down_and_record(sky, cluster, store, job_id)
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
    cost = CostInfo(
        duration_seconds=dur,
        hourly_usd=hourly_usd,
        estimated_usd=estimated_usd,
        actual_usd=actual_cost(hourly_usd, dur),
    )

    # Respect a concurrent cancel (backend set status=cancelled before killing us).
    if store.read_manifest(job_id).status != JobState.cancelled:
        # final_metrics is snapshotted centrally by the store on the succeeded transition (FR-B4).
        store.update_manifest(
            job_id,
            status=final,
            ended_at=ended,
            exit_code=0 if final == JobState.succeeded else 1,
            end_reason=final.value,
            artifacts_uri=artifacts_uri,
            cost=cost,
        )

    teardown_ok = tear_down_and_record(sky, cluster, store, job_id)
    if final is JobState.preempted and not confirm_no_rental(cluster):
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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", type=Path)
    ap.add_argument(
        "--adopt",
        action="store_true",
        help="re-attach to an already-launched cluster (scheduler watchdog)",
    )
    ns = ap.parse_args()
    raise SystemExit(run_job(ns.job_dir, adopt=ns.adopt))
