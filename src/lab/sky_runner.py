"""Supervisor for the SkyPilot backend (spawned detached by SkyPilotBackend.submit).

Performs the blocking ``sky.launch`` (provision + run), records terminal state, rsyncs outputs
back into the run dir, and tears the instance down. Its stdout/stderr are redirected to the job
log file by ``submit``, so SkyPilot's streamed logs become the job logs (FR-D1).

Entry point:  python -m lab.sky_runner <job_dir>
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from lab._util import actual_cost, duration_seconds, now, parse_duration
from lab.backends.skypilot import (
    DEFAULT_AUTOSTOP_MIN,
    REMOTE_RUN_DIR,
    build_task,
    cluster_name_for,
    map_job_status,
    promote_timeout,
)
from lab.models import CostInfo, JobState
from lab.storage import R2Store, r2_enabled
from lab.store import JobStore


_TERMINAL_NAMES = {"SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_DRIVER", "CANCELLED"}


def _rec_field(rec, key: str):
    return rec.get(key) if isinstance(rec, dict) else getattr(rec, key, None)


def _job_status_name(sky_mod, cluster: str, sky_job_id: int | None) -> str | None:
    recs = sky_mod.get(sky_mod.queue(cluster, skip_finished=False))  # 0.12: RequestId
    for rec in recs:
        if sky_job_id is None or _rec_field(rec, "job_id") == sky_job_id:
            status = _rec_field(rec, "status")
            return getattr(status, "name", str(status).split(".")[-1])
    return None


def _wait_terminal(sky_mod, cluster: str, sky_job_id: int | None, max_wait: float) -> JobState:
    """Poll the remote job until terminal — sky.launch (0.12) returns at submit time, not
    completion, so we must wait before fetching artifacts and tearing down."""
    deadline = time.time() + max_wait
    name: str | None = None
    while time.time() < deadline:
        try:
            name = _job_status_name(sky_mod, cluster, sky_job_id)
        except Exception as e:  # noqa: BLE001
            print(f"[lab] queue poll error: {e}")
        if name in _TERMINAL_NAMES:
            break
        time.sleep(10)
    return map_job_status(name or "FAILED")


def _rsync_down(cluster: str, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", f"{cluster}:{remote_dir}/", f"{local_dir}/"],
        check=True,
        timeout=180,
    )


def _safe_down(sky_mod, cluster: str) -> None:
    try:
        sky_mod.get(sky_mod.down(cluster))  # 0.12: RequestId
    except Exception as e:  # noqa: BLE001
        print(f"[lab] teardown warning for {cluster}: {e}")


def run_job(job_dir: Path) -> int:
    job_dir = Path(job_dir)
    store = JobStore(job_dir.parent)
    job_id = job_dir.name
    manifest = store.read_manifest(job_id)
    cluster = cluster_name_for(job_id)

    started = now()
    store.update_manifest(job_id, status=JobState.running, started_at=started)

    import sky

    try:
        task = build_task(manifest, workdir=Path.cwd())
        request_id = sky.launch(
            task,
            cluster_name=cluster,
            down=True,
            idle_minutes_to_autostop=DEFAULT_AUTOSTOP_MIN,
        )
        sky_job_id, handle = sky.stream_and_get(request_id)  # returns once the job is submitted (0.12)
        # Wait for the run to actually finish before fetching artifacts / tearing down.
        try:
            sky.tail_logs(cluster, sky_job_id, follow=True)  # streams run logs; blocks till done
        except Exception as e:  # noqa: BLE001
            print(f"[lab] tail_logs issue: {e}")
        max_wait = (parse_duration(manifest.resources.timeout) or 3600) + 300
        final = _wait_terminal(sky, cluster, sky_job_id, max_wait)
    except Exception as e:  # noqa: BLE001
        store.update_manifest(
            job_id, status=JobState.failed, ended_at=now(), end_reason=f"launch error: {e}"[:300]
        )
        _safe_down(sky, cluster)
        return 1

    try:
        _rsync_down(cluster, REMOTE_RUN_DIR, store.output_dir(job_id))
    except Exception as e:  # noqa: BLE001
        print(f"[lab] artifact rsync failed: {e}")

    final = promote_timeout(final, store.output_dir(job_id))  # failed -> timed_out if sentinel

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
    hourly_usd = None
    try:
        launched = getattr(handle, "launched_resources", None)
        if launched is not None:
            hourly_usd = float(launched.get_cost(3600))  # USD/hour for the chosen instance (FR-I2)
    except Exception as e:  # noqa: BLE001
        print(f"[lab] cost estimate unavailable: {e}")
    dur = duration_seconds(started, ended)
    cost = CostInfo(
        duration_seconds=dur, hourly_usd=hourly_usd, actual_usd=actual_cost(hourly_usd, dur)
    )

    # Respect a concurrent cancel (backend set status=cancelled before killing us).
    if store.read_manifest(job_id).status != JobState.cancelled:
        store.update_manifest(
            job_id,
            status=final,
            ended_at=ended,
            exit_code=0 if final == JobState.succeeded else 1,
            end_reason=final.value,
            artifacts_uri=artifacts_uri,
            cost=cost,
        )

    _safe_down(sky, cluster)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(run_job(Path(sys.argv[1])))
