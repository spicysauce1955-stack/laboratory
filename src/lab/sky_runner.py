"""Supervisor for the SkyPilot backend (spawned detached by SkyPilotBackend.submit).

Performs the blocking ``sky.launch`` (provision + run), records terminal state, rsyncs outputs
back into the run dir, and tears the instance down. Its stdout/stderr are redirected to the job
log file by ``submit``, so SkyPilot's streamed logs become the job logs (FR-D1).

Entry point:  python -m lab.sky_runner <job_dir>
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from lab._util import now
from lab.backends.skypilot import (
    DEFAULT_AUTOSTOP_MIN,
    REMOTE_RUN_DIR,
    build_task,
    cluster_name_for,
    map_job_status,
)
from lab.models import JobState
from lab.store import JobStore


def _final_job_state(sky_mod, cluster: str, sky_job_id: int | None) -> JobState:
    try:
        for rec in sky_mod.queue(cluster, skip_finished=False):
            if sky_job_id is None or rec.get("job_id") == sky_job_id:
                status = rec.get("status")
                name = getattr(status, "name", str(status).split(".")[-1])
                return map_job_status(name)
    except Exception as e:  # noqa: BLE001
        print(f"[lab] could not read final job status: {e}")
    return JobState.failed


def _rsync_down(cluster: str, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", f"{cluster}:{remote_dir}/", f"{local_dir}/"],
        check=True,
        timeout=180,
    )


def _safe_down(sky_mod, cluster: str) -> None:
    try:
        sky_mod.down(cluster)
    except Exception as e:  # noqa: BLE001
        print(f"[lab] teardown warning for {cluster}: {e}")


def run_job(job_dir: Path) -> int:
    job_dir = Path(job_dir)
    store = JobStore(job_dir.parent)
    job_id = job_dir.name
    manifest = store.read_manifest(job_id)
    cluster = cluster_name_for(job_id)

    store.update_manifest(job_id, status=JobState.running, started_at=now())

    import sky

    try:
        task = build_task(manifest, workdir=Path.cwd())
        sky_job_id, _ = sky.launch(
            task,
            cluster_name=cluster,
            down=True,
            idle_minutes_to_autostop=DEFAULT_AUTOSTOP_MIN,
            detach_run=False,
            stream_logs=True,
        )
        final = _final_job_state(sky, cluster, sky_job_id)
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

    # Respect a concurrent cancel (backend set status=cancelled before killing us).
    if store.read_manifest(job_id).status != JobState.cancelled:
        store.update_manifest(
            job_id,
            status=final,
            ended_at=now(),
            exit_code=0 if final == JobState.succeeded else 1,
            end_reason=final.value,
        )

    _safe_down(sky, cluster)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(run_job(Path(sys.argv[1])))
