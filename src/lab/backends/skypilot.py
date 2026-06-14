"""SkyPilot backend — remote execution on any SkyPilot cloud (Vast.ai for us).

``submit`` spawns a detached local supervisor (``lab.sky_runner``) that performs the *blocking*
``sky.launch`` (provision + run), so submission returns immediately (FR-A1) and the job survives
the CLI/MCP process exiting (NFR-2). The supervisor records terminal state, rsyncs outputs back,
and tears the instance down (FR-C2). ``sky.launch(down=True, idle_minutes_to_autostop=…)`` is the
cost-safety guarantee even if the supervisor dies (NFR-7).

P0 limitations (tracked): artifacts are rsynced from the live instance before teardown — durable
object storage (R2/S3) is a P1 item (research/15).
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lab._util import infer_artifact_type, now, parse_duration
from lab.manifest import sha256_file
from lab.metrics import METRICS_FILE, read_points
from lab.models import ArtifactRecord, JobManifest, JobState
from lab.store import JobStore

if TYPE_CHECKING:
    import sky

REMOTE_RUN_DIR = "/tmp/lab_run"
TIMEOUT_SENTINEL = ".lab_timed_out"  # written by the run script when `timeout` kills the job
TIMEOUT_KILL_GRACE_S = 30  # SIGTERM -> wait -> SIGKILL grace for a process that ignores TERM
SELF_DESTRUCT_MARGIN_S = 600  # instance self-poweroff backstop fires at wall + this (§6)
DEFAULT_AUTOSTOP_MIN = 5  # safety-net teardown if the supervisor process dies
# Provisioning watchdog: a healthy Vast host reaches UP in ~2-4 min, so 8 min clears
# slow-but-alive hosts while catching ones stuck in "loading" forever (a dead offer).
DEFAULT_PROVISION_TIMEOUT_MIN = 8
# Teardown retry budget: ~3.5 min total (first attempt + 5 retries spaced 5/15/30/60/120 s).
# Long enough to ride out a transient DNS/API hiccup; short enough that a cluster that's
# genuinely stuck still gets nuked via the vast-sdk fallback in well under 5 minutes.
TEARDOWN_BACKOFFS = (5, 15, 30, 60, 120)
_TERMINAL = {JobState.succeeded, JobState.failed, JobState.cancelled, JobState.timed_out, JobState.preempted}

# SkyPilot JobStatus name -> lab JobState (pure; unit-tested).
_STATUS_MAP = {
    "INIT": JobState.queued,
    "PENDING": JobState.queued,
    "SETTING_UP": JobState.queued,
    "RUNNING": JobState.running,
    "SUCCEEDED": JobState.succeeded,
    "FAILED": JobState.failed,
    "FAILED_SETUP": JobState.failed,
    "FAILED_DRIVER": JobState.failed,
    "CANCELLED": JobState.cancelled,
}


def map_job_status(status_name: str) -> JobState:
    """Map a SkyPilot JobStatus name to a lab JobState (unknown -> failed)."""
    return _STATUS_MAP.get(status_name, JobState.failed)


def cluster_name_for(job_id: str) -> str:
    """SkyPilot cluster name: starts with a letter, lowercase alnum + hyphen."""
    safe = re.sub(r"[^a-z0-9-]", "-", job_id.lower()).strip("-")
    return f"lab-{safe}"[:60]


def build_setup_script() -> str:
    """Install uv and materialise the locked env on the remote (FR-B2)."""
    return (
        "set -e\n"
        "curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        # --no-default-groups: skip the cli/dev groups (typer/fastmcp/pytest) — the remote only
        # needs experiment-runtime deps, not the lab control plane.
        "uv sync --frozen --no-default-groups\n"
    )


def _wall_clock_wrap(cmd: str, wall: int) -> list[str]:
    """Lines that run ``cmd`` under GNU ``timeout`` so the wall-clock cap holds on the instance.

    ``timeout`` is the *primary* enforcement (FR-I1, §6) and is deliberately self-contained: it is
    a single coreutils binary that needs no working ``setsid --wait``, no hand-rolled
    ``kill -$$`` process-group arithmetic, and no assumptions about the remote bash/util-linux
    version — the previous in-shell timer relied on all three lining up on an unknown Vast image
    and overran by hours in production (LAB-BUGS §1). Run in ``timeout``'s default (non-foreground)
    mode it places the entrypoint in its OWN process group and signals the *whole* group on expiry,
    so the ``uv``→``python``→worker tree dies together; ``--kill-after`` escalates TERM→KILL for a
    child that ignores TERM (the ``b=1`` online loop). All of this runs on the box, independent of
    the local supervisor — the exact failure mode (supervisor dies → nothing enforces the cap).

    ``timeout`` exits ``124`` when it had to TERM the job and ``137`` (128+SIGKILL) when it escalated
    to KILL; either is a cap hit, so we drop the sentinel for :func:`promote_timeout` to relabel the
    run ``timed_out``. A clean finish keeps the entrypoint's own exit code.
    """
    grace = TIMEOUT_KILL_GRACE_S
    sentinel = f"{REMOTE_RUN_DIR}/{TIMEOUT_SENTINEL}"
    return [
        f"timeout --kill-after={grace}s {wall}s bash -c {shlex.quote(cmd)}",
        "rc=$?",
        f'if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then touch "{sentinel}"; fi',
        'exit "$rc"',
    ]


def build_run_script(manifest: JobManifest) -> str:
    """Activate the env, then run the entrypoint under an instance-side wall-clock cap (FR-I1, §6).

    The cap must hold even if the local supervisor dies (it runs in the agent's sandbox, which can
    be suspended), so enforcement is entirely on the box, in two layers:

    * **Primary:** the entrypoint runs under GNU ``timeout`` — see :func:`_wall_clock_wrap` for the
      group-kill/sentinel mechanics. When it fires the host goes idle and SkyPilot's autostop /
      ``down=True`` tears it down with no supervisor involved.
    * **Backstop:** a detached ``poweroff`` watchdog at ``wall + SELF_DESTRUCT_MARGIN_S``. This is
      best-effort only — it is a no-op inside an unprivileged container — so it is defence in depth
      behind ``timeout``, not the mechanism we rely on.
    """
    timeout = parse_duration(manifest.resources.timeout)
    lines = [
        'export PATH="$HOME/.local/bin:$PATH"',
        "source .venv/bin/activate",
        f'mkdir -p "{REMOTE_RUN_DIR}"',
    ]
    if not timeout:
        lines.append(manifest.run.entrypoint_command)
        return "\n".join(lines) + "\n"

    wall = int(timeout)
    lines += [
        # Best-effort backstop (a no-op in unprivileged containers): power the box off at
        # wall+margin so billing can't run far past the cap if teardown is wedged. Detached in its
        # own session so `timeout`'s group-kill above never touches it (§6 cost cap).
        f"nohup setsid bash -c 'sleep {wall + SELF_DESTRUCT_MARGIN_S}; "
        "sudo poweroff -f || poweroff -f || sudo shutdown -h now || shutdown -h now' "
        ">/dev/null 2>&1 </dev/null &",
        *_wall_clock_wrap(manifest.run.entrypoint_command, wall),
    ]
    return "\n".join(lines) + "\n"


def promote_timeout(final: JobState, output_dir: Path) -> JobState:
    """Promote a failed run to timed_out if the run script left the timeout sentinel (FR-I1)."""
    if final == JobState.failed and (Path(output_dir) / TIMEOUT_SENTINEL).exists():
        return JobState.timed_out
    return final


# ---------------------------------------------------------------------------
# Teardown — robust retry + vast-sdk fallback so a transient SkyPilot error
# never leaks a paid GPU rental (FR-C2 leak prevention).
# ---------------------------------------------------------------------------


def _get_vast_client() -> Any:
    """Construct a vastai-sdk client. Test seam: monkeypatch this to inject a fake."""
    from vastai_sdk import VastAI  # type: ignore[import-untyped]

    return VastAI()


def _instance_label(inst: dict[str, Any]) -> str:
    """Concatenate the candidate name-fields a Vast.ai instance dict may carry, lower-cased.

    SkyPilot's Vast adapter tags the rental with the cluster name in ``label``; we also probe
    a few neighbouring field names so a Vast SDK change doesn't silently disable matching.
    """
    parts = [str(inst.get(k, "")) for k in ("label", "name", "instance_label", "machine_name")]
    return " ".join(parts).lower()


def list_vast_instances(client: Any | None = None) -> list[dict[str, Any]]:
    """Return every active rental on the Vast.ai account (raises if vastai-sdk unavailable)."""
    if client is None:
        client = _get_vast_client()
    return list(client.show_instances())


def vast_hourly_for_cluster(cluster: str, client: Any | None = None) -> float | None:
    """Actual billed USD/hour (``dph_total``) for the Vast rental backing ``cluster``, or None.

    SkyPilot's ``get_cost()`` reads its own catalog and under-reports Vast prices (~4x low); the
    rental's own ``dph_total`` is the real billed rate, so we prefer it for cost accuracy (FR-I2).
    Returns None if no rental matches the cluster or the price field is absent/unparseable, so the
    caller can fall back to the SkyPilot estimate.
    """
    needle = cluster.lower()
    for inst in list_vast_instances(client=client):
        if needle not in _instance_label(inst):
            continue
        dph = inst.get("dph_total")
        if dph is None:
            return None
        try:
            return float(dph)
        except (TypeError, ValueError):
            return None
    return None


def vast_balance(client: Any | None = None) -> float | None:
    """Current Vast.ai account balance/credit (USD), or None if unavailable (best-effort).

    A depleted/negative balance makes Vast reject rentals with ``400 Bad Request``, which
    SkyPilot surfaces as a generic "Failed to provision … resources" — indistinguishable from
    "no GPUs". We consult this on a provision failure to give an actionable message (§8).
    """
    if client is None:
        client = _get_vast_client()
    try:
        info = client.show_user()
    except Exception as e:  # noqa: BLE001 — best-effort; caller falls back to the generic message
        print(f"[lab] vast balance lookup failed: {e}")
        return None
    for key in ("credit", "balance"):
        val = info.get(key) if isinstance(info, dict) else getattr(info, key, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _vast_destroy_matching(cluster: str, client: Any | None = None) -> list[int]:
    """Destroy every Vast rental whose label contains ``cluster``; return their IDs."""
    if client is None:
        client = _get_vast_client()
    needle = cluster.lower()
    destroyed: list[int] = []
    for inst in list_vast_instances(client=client):
        if needle not in _instance_label(inst):
            continue
        inst_id = inst.get("id")
        if inst_id is None:
            continue
        try:
            client.destroy_instance(id=int(inst_id))
            destroyed.append(int(inst_id))
            print(f"[lab] vast-direct destroyed instance {inst_id} (cluster={cluster})")
        except Exception as e:  # noqa: BLE001 — best-effort; the next instance might still go
            print(f"[lab] vast-direct destroy {inst_id} failed: {e}")
    return destroyed


def robust_teardown(
    sky_mod: Any, cluster: str, *, backoffs: tuple[int, ...] = TEARDOWN_BACKOFFS
) -> dict[str, Any]:
    """Tear down a SkyPilot cluster with retry + vastai-sdk fallback.

    Why this exists: a single ``sky.down`` call that swallows transient errors (network hiccup,
    Vast.ai API timeout) used to leak rentals — the cluster kept billing while we marked the
    job ``failed`` and moved on. We now retry sky.down with exponential backoff, then bypass
    SkyPilot's local registry entirely and ask Vast.ai itself to destroy any rental whose
    label matches the cluster name.

    Returns a structured outcome suitable for persistence on the manifest:

    .. code-block:: python

        {
          "status": "succeeded" | "failed",
          "attempts": int,             # total sky.down attempts (1 + retries)
          "vast_fallback_used": bool,
          "vast_destroyed": list[int], # Vast instance IDs killed via fallback
          "error": str | None,         # last sky.down error if any
        }

    ``status == "succeeded"`` means: either sky.down returned, OR the vast SDK fallback ran and
    either destroyed instances or found none matching. ``"failed"`` means even the fallback
    raised — a human needs to check ``vastai show_instances`` (and ``lab reconcile``) NOW.
    """
    last_err: str | None = None
    delays = (0, *backoffs)  # first try has no delay
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        try:
            sky_mod.get(sky_mod.down(cluster))
            return {
                "status": "succeeded",
                "attempts": attempt,
                "vast_fallback_used": False,
                "vast_destroyed": [],
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            print(
                f"[lab] sky.down attempt {attempt}/{len(delays)} for {cluster} failed: {last_err}"
            )

    # SkyPilot teardown didn't take. Talk to Vast directly — it's the source of truth.
    print(f"[lab] sky.down exhausted for {cluster}; falling back to vast-sdk direct destroy")
    try:
        destroyed = _vast_destroy_matching(cluster)
        return {
            "status": "succeeded",  # destroyed-or-none-found are both safe outcomes
            "attempts": len(delays),
            "vast_fallback_used": True,
            "vast_destroyed": destroyed,
            "error": last_err,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "status": "failed",
            "attempts": len(delays),
            "vast_fallback_used": True,
            "vast_destroyed": [],
            "error": f"sky.down: {last_err}; vast-direct: {type(e).__name__}: {e}",
        }


def tear_down_and_record(sky_mod: Any, cluster: str, store: JobStore, job_id: str) -> bool:
    """Call :func:`robust_teardown` and persist its outcome on the job manifest.

    Returns ``True`` iff teardown succeeded. On failure, ``teardown_status='failed'`` is
    written and ``end_reason`` is annotated with an actionable instruction so the leak is
    visible in ``lab status`` / ``lab dashboard`` / ``lab wait``.
    """
    outcome = robust_teardown(sky_mod, cluster)
    succeeded: bool = outcome["status"] == "succeeded"
    fields: dict[str, Any] = {"teardown_status": "succeeded" if succeeded else "failed"}
    annotation: str | None = None
    if not succeeded:
        annotation = (
            f"TEARDOWN FAILED for cluster {cluster!r}: {outcome['error']} after "
            f"{outcome['attempts']} sky.down attempts AND vast-sdk fallback. "
            "Run `lab reconcile --apply` (or `vastai destroy_instance <id>`) to stop the bleed."
        )
    elif outcome["vast_fallback_used"]:
        annotation = (
            f"sky.down failed ({outcome['error']}); vast-sdk fallback destroyed "
            f"{outcome['vast_destroyed']}"
        )
    if annotation is not None:
        print(f"[lab] {annotation}")
        existing = (store.read_manifest(job_id).end_reason or "").strip()
        fields["end_reason"] = (f"{existing} | {annotation}" if existing else annotation)[:600]
    store.update_manifest(job_id, **fields)
    return succeeded


# ---------------------------------------------------------------------------
# Provisioning watchdog — bound the blocking ``stream_and_get`` so a Vast host
# that never reaches UP (stuck in "loading") can't hang the supervisor forever.
# ---------------------------------------------------------------------------


class ProvisionTimeout(Exception):
    """Raised when a SkyPilot launch does not finish provisioning within the watchdog window."""


def provision_with_watchdog(sky_mod: Any, request_id: Any, *, timeout_s: float) -> tuple[Any, Any]:
    """Run ``sky_mod.stream_and_get(request_id)`` under a wall-clock watchdog.

    ``stream_and_get`` blocks streaming provisioning logs until the remote job is *submitted*,
    which only happens after the host reaches UP. A dead Vast offer never gets there, so the call
    blocks indefinitely. We run it in a daemon thread and ``join`` for ``timeout_s``:

    - returns ``(sky_job_id, handle)`` if provisioning completes in time;
    - raises :class:`ProvisionTimeout` if it doesn't (best-effort ``sky_mod.api_cancel`` first);
    - re-raises unchanged any genuine error ``stream_and_get`` raised before the timeout.

    The thread is a daemon, so if it's still stuck after the timeout it dies with the supervisor
    process and never blocks teardown or exit.
    """
    holder: dict[str, Any] = {}

    def _run() -> None:
        try:
            holder["value"] = sky_mod.stream_and_get(request_id)
        except BaseException as e:  # noqa: BLE001 — surfaced to the caller below
            holder["error"] = e

    thread = threading.Thread(target=_run, name="lab-provision-watchdog", daemon=True)
    thread.start()
    thread.join(timeout_s)

    if thread.is_alive():
        try:
            sky_mod.api_cancel(request_id)  # best-effort abort; robust_teardown kills the host
        except Exception as e:  # noqa: BLE001
            print(f"[lab] api_cancel after provision timeout failed: {e}")
        raise ProvisionTimeout(f"provisioning did not complete within {timeout_s:.0f}s")

    if "error" in holder:
        raise holder["error"]
    value: tuple[Any, Any] = holder["value"]
    return value


def build_task(manifest: JobManifest, workdir: Path) -> sky.Task:
    """Translate a JobManifest into a SkyPilot Task (no cloud calls; unit-tested)."""
    import sky

    task = sky.Task(
        name=cluster_name_for(manifest.job_id),
        setup=build_setup_script(),
        run=build_run_script(manifest),
        envs={
            "LAB_RUN_ID": manifest.job_id,
            "LAB_RUN_DIR": REMOTE_RUN_DIR,
            "LAB_SEED": str(manifest.run.seed),
        },
        workdir=str(workdir),
    )
    # Vast is GPU-only in SkyPilot's catalog, so `accelerators` (e.g. "RTX_3070:1") is typically
    # required; cpus/memory further constrain. If accelerators is None SkyPilot cost-optimises.
    _cloud = sky.Vast()
    _cpus = manifest.resources.cpus
    _memory = manifest.resources.memory
    _accels = manifest.resources.accelerators or None

    def _res(*, use_spot: bool | None = None) -> sky.Resources:
        return sky.Resources(
            cloud=_cloud,
            cpus=_cpus,
            memory=_memory,
            accelerators=_accels,
            use_spot=use_spot,
        )

    if not manifest.resources.use_spot:
        task.set_resources(_res())
    elif manifest.resources.spot_fallback:
        # Prefer spot (cheaper); SkyPilot's optimizer fails over to on-demand if spot is scarce.
        task.set_resources([_res(use_spot=True), _res(use_spot=False)])
    else:
        task.set_resources(_res(use_spot=True))  # spot-only, no fallback
    return task


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SkyPilotBackend:
    name = "skypilot"

    def __init__(self, home: Path, repo: Path | None = None) -> None:
        self.store = JobStore(Path(home))
        self.repo = Path(repo) if repo else Path.cwd()

    def submit(self, manifest: JobManifest) -> str:
        job_dir = self.store.job_dir(manifest.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        # Supervisor's stdout/stderr (incl. sky.launch streamed logs) -> the job log file (FR-D1).
        logf = self.store.logs_path(manifest.job_id).open("w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "lab.sky_runner", str(job_dir)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(self.repo),
            start_new_session=True,
        )
        self.store.write_runtime(
            manifest.job_id, runner_pid=proc.pid, cluster=cluster_name_for(manifest.job_id)
        )
        return manifest.job_id

    def status(self, job_id: str) -> JobState:
        m = self.store.read_manifest(job_id)
        if m.status not in _TERMINAL:
            rt = self.store.read_runtime(job_id)
            if rt.get("runner_pid") and not _alive(rt["runner_pid"]):
                return self.store.update_manifest(
                    job_id,
                    status=JobState.failed,
                    ended_at=now(),
                    end_reason="supervisor exited without recording status",
                ).status
        return m.status

    def tail_logs(
        self, job_id: str, tail: int | None = None, follow: bool = False
    ) -> Iterable[str]:
        p = self.store.logs_path(job_id)
        if not p.exists():
            return []
        lines = p.read_text(errors="replace").splitlines()
        return lines[-tail:] if tail else lines

    def cancel(self, job_id: str) -> JobState:
        m = self.store.read_manifest(job_id)
        if m.status in _TERMINAL:
            return m.status
        self.store.update_manifest(
            job_id, status=JobState.cancelled, ended_at=now(), end_reason="cancelled by user"
        )
        rt = self.store.read_runtime(job_id)
        if rt.get("runner_pid"):
            try:
                os.kill(rt["runner_pid"], signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        cluster = rt.get("cluster") or cluster_name_for(job_id)
        try:
            import sky

            sky.get(sky.cancel(cluster, all=True))  # 0.12: RequestId
        except Exception:  # noqa: BLE001 - best-effort; teardown below is what matters
            pass
        import sky

        tear_down_and_record(sky, cluster, self.store, job_id)
        return JobState.cancelled

    def collect_artifacts(self, job_id: str, dest: str) -> list[ArtifactRecord]:
        # The supervisor rsyncs the remote run dir into output/ before teardown; read it locally.
        out = self.store.output_dir(job_id)
        records: list[ArtifactRecord] = []
        if out.exists():
            for f in sorted(out.rglob("*")):
                if f.is_file() and not f.name.startswith("."):  # skip sentinels/hidden files
                    rel = f.relative_to(out).as_posix()
                    records.append(
                        ArtifactRecord(
                            name=rel,
                            type=infer_artifact_type(rel),  # type: ignore[arg-type]
                            path=str(f),
                            sha256=sha256_file(f),
                            bytes=f.stat().st_size,
                        )
                    )
        self.store.update_manifest(job_id, artifacts=records)
        return records

    def read_metrics(
        self, job_id: str, names: Iterable[str] | None = None, since_step: int | None = None
    ) -> list[dict[str, Any]]:
        return read_points(
            self.store.output_dir(job_id) / METRICS_FILE, names=names, since_step=since_step
        )
