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
from lab.models import ArtifactRecord, JobManifest, JobState, ResourceRequest
from lab.preemption import gcp_terminal_state
from lab.store import JobStore

if TYPE_CHECKING:
    import sky

REMOTE_RUN_DIR = "/tmp/lab_run"
TIMEOUT_SENTINEL = ".lab_timed_out"  # written by the run script when `timeout` kills the job
SUCCESS_SENTINEL = ".lab_success"  # written only on a clean exit-0; gates the `succeeded` label
TIMEOUT_KILL_GRACE_S = 30  # SIGTERM -> wait -> SIGKILL grace for a process that ignores TERM
SELF_DESTRUCT_MARGIN_S = 600  # instance self-poweroff backstop fires at wall + this (§6)
DEFAULT_AUTOSTOP_MIN = 5  # safety-net teardown if the supervisor process dies
# Provisioning watchdog: a healthy Vast host reaches UP in ~2-4 min, so 8 min clears
# slow-but-alive hosts while catching ones stuck in "loading" forever (a dead offer).
DEFAULT_PROVISION_TIMEOUT_MIN = 8
# ...but that number is Vast-shaped, and the hyperscalers spend their time somewhere else. A GCP
# launch that meets ZONE_RESOURCE_POOL_EXHAUSTED fails over zone by zone and region by region, so
# its clock is spent in the optimizer's walk rather than on one stuck host — and killing that walk
# mid-provision is the LAB-BUGS §4 leak scenario, where autostop is not set yet. Give the clouds
# that fail over the room to finish failing over.
PROVISION_TIMEOUT_MIN_BY_CLOUD = {"vast": 8, "do": 12, "gcp": 20}


def provision_timeout_min(cloud: str | None) -> int:
    """Minutes to allow for provisioning on a cloud (pure)."""
    return PROVISION_TIMEOUT_MIN_BY_CLOUD.get(cloud or "vast", DEFAULT_PROVISION_TIMEOUT_MIN)
# Teardown retry budget: ~3.5 min total (first attempt + 5 retries spaced 5/15/30/60/120 s).
# Long enough to ride out a transient DNS/API hiccup; short enough that a cluster that's
# genuinely stuck still gets nuked via the vast-sdk fallback in well under 5 minutes.
TEARDOWN_BACKOFFS = (5, 15, 30, 60, 120)
_TERMINAL = {
    JobState.succeeded, JobState.failed, JobState.cancelled, JobState.timed_out, JobState.preempted
}

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
    timeout_sentinel = f"{REMOTE_RUN_DIR}/{TIMEOUT_SENTINEL}"
    success_sentinel = f"{REMOTE_RUN_DIR}/{SUCCESS_SENTINEL}"
    return [
        f"timeout --kill-after={grace}s {wall}s bash -c {shlex.quote(cmd)}",
        "rc=$?",
        f'if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then touch "{timeout_sentinel}"; fi',
        f'if [ "$rc" = 0 ]; then touch "{success_sentinel}"; fi',
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

    **What ``poweroff`` actually stops is per-cloud** (GCP-LEAK-9) — it is not the hard backstop
    the word suggests:

    ===== ================================= ==========================================
    Cloud ``poweroff`` does                 Still billing afterwards
    ===== ================================= ==========================================
    Vast  ends the rental                   nothing
    GCP   TERMINATEs the VM; compute stops  **the persistent disk, indefinitely**
    DO    powers the droplet off            **the whole droplet, at full price**
    ===== ================================= ==========================================

    So on GCP this converts a compute leak into a storage leak — a real improvement, but not a
    release of resources. The actual teardown path is ``down=True`` + autostop, with
    ``lab reconcile``'s unattached-disk pass as the net behind it; on DO the backstop stops
    nothing at all and teardown is the only mechanism.
    """
    timeout = parse_duration(manifest.resources.timeout)
    lines = [
        'export PATH="$HOME/.local/bin:$PATH"',
        "source .venv/bin/activate",
        f'mkdir -p "{REMOTE_RUN_DIR}"',
    ]
    if not timeout:
        success_sentinel = f"{REMOTE_RUN_DIR}/{SUCCESS_SENTINEL}"
        lines += [
            manifest.run.entrypoint_command,
            "rc=$?",
            f'if [ "$rc" = 0 ]; then touch "{success_sentinel}"; fi',
            'exit "$rc"',
        ]
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


def confirm_success(state: JobState, run_dir: Path) -> JobState:
    """Downgrade succeeded->failed unless the clean-exit sentinel is present (FR-B5 integrity)."""
    if state is JobState.succeeded and not (run_dir / SUCCESS_SENTINEL).exists():
        return JobState.failed
    return state


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


def _get_do_client() -> Any:
    """Construct a DigitalOcean (pydo) client via SkyPilot's DO provisioner config. Test seam:
    monkeypatch this to inject a fake."""
    from sky.provision.do import utils as do_utils

    return do_utils.client()  # type: ignore[no-untyped-call]


def list_do_volumes(client: Any | None = None) -> list[dict[str, Any]]:
    """Return every DO block volume on the account as plain dicts (raises if the DO client/listing
    fails — callers in best-effort paths swallow that). pydo paginates; pull a large page since a
    lab account holds only a handful of `lab-*` volumes."""
    if client is None:
        client = _get_do_client()
    resp = client.volumes.list(per_page=200)
    vols = resp.get("volumes", []) if isinstance(resp, dict) else resp
    return [dict(v) for v in (vols or [])]


def do_volume_orphans(
    volumes: list[dict[str, Any]], running_clusters: set[str]
) -> list[dict[str, Any]]:
    """`lab-*` DO block volumes not tied to any running cluster — the volume-leak analogue of the
    instance orphan pass (`reconcile`). SkyPilot's DO provisioner names the attached volume after
    its cluster, so a running cluster name is a substring of its volume's name. Pure."""
    orphans: list[dict[str, Any]] = []
    for v in volumes:
        name = str(v.get("name", ""))
        if not name.startswith("lab-"):
            continue  # not a lab volume — leave it alone
        if any(c.lower() in name.lower() for c in running_clusters):
            continue  # backs a live job
        orphans.append({"id": v.get("id"), "name": name})
    return orphans


class GcpNotConfigured(RuntimeError):
    """GCP isn't set up on this machine: the provider extra isn't installed, there are no
    application-default credentials, or no project is selected.

    Deliberately distinct from a GCP API *failure* (a revoked role, an expired key, a disabled
    API, a 5xx). "Not configured" is safe to skip — not every account uses GCP. A failure is not:
    a leak-detection pass that swallows one reports **clean while blind**, which is worse than
    having no pass at all, because the report claims coverage it doesn't have (FR-C2). Callers
    catch this narrowly and let everything else propagate.
    """


def _gcp_default_credentials() -> tuple[Any, str | None]:
    """ADC credentials + the ambient project. Test seam; raises :class:`GcpNotConfigured` when
    GCP isn't set up here."""
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError as e:  # the `gcp` extra isn't installed
        raise GcpNotConfigured(f"GCP client libraries not installed: {e}") from e
    try:
        creds, project = google.auth.default()
    except DefaultCredentialsError as e:
        raise GcpNotConfigured(f"no application-default credentials: {e}") from e
    return creds, project


def _get_gcp_compute() -> tuple[Any, str]:
    """GCP compute client + project via application-default credentials (the same creds
    SkyPilot's GCP provisioner uses). Test seam: monkeypatch to inject a fake."""
    from googleapiclient import discovery  # type: ignore[import-untyped]

    creds, project = _gcp_default_credentials()
    if not project:
        raise GcpNotConfigured(
            "no GCP project configured (run `gcloud config set project <id>`, or set "
            "GOOGLE_CLOUD_PROJECT in .env)"
        )
    compute = discovery.build("compute", "v1", credentials=creds, cache_discovery=False)
    return compute, str(project)


def _zone_name(url: str) -> str:
    """'https://…/zones/us-central1-a' (or 'zones/us-central1-a') -> 'us-central1-a'."""
    return url.rsplit("/", 1)[-1]


def list_gcp_instances(compute: Any | None = None, project: str | None = None) -> list[dict[str, Any]]:
    """Every GCE instance on the project as ``{name, zone, status, preemptible}`` dicts, via
    ``instances.aggregatedList`` (all zones, paginated). Raises if GCP isn't configured —
    best-effort callers (the reconcile pass) swallow that.

    ``preemptible`` mirrors ``scheduling.preemptible``, which is how GCE states a Spot VM
    outright — see :func:`lab.preemption.gcp_terminal_state` (GCP-PREEMPT-1).
    """
    if compute is None or project is None:
        compute, project = _get_gcp_compute()
    out: list[dict[str, Any]] = []
    req = compute.instances().aggregatedList(project=project)
    while req is not None:
        resp = req.execute()
        for _scope, data in (resp.get("items") or {}).items():
            for inst in data.get("instances") or []:
                out.append(
                    {
                        "name": str(inst.get("name", "")),
                        "zone": _zone_name(str(inst.get("zone", ""))),
                        "status": inst.get("status"),
                        "preemptible": bool((inst.get("scheduling") or {}).get("preemptible")),
                    }
                )
        req = compute.instances().aggregatedList_next(previous_request=req, previous_response=resp)
    return out


def list_gcp_disks(compute: Any | None = None, project: str | None = None) -> list[dict[str, Any]]:
    """Every persistent disk on the project as ``{name, zone, users}`` dicts (``users`` = URLs of
    instances the disk is attached to; empty = unattached and still billing)."""
    if compute is None or project is None:
        compute, project = _get_gcp_compute()
    out: list[dict[str, Any]] = []
    req = compute.disks().aggregatedList(project=project)
    while req is not None:
        resp = req.execute()
        for _scope, data in (resp.get("items") or {}).items():
            for disk in data.get("disks") or []:
                out.append(
                    {
                        "name": str(disk.get("name", "")),
                        "zone": _zone_name(str(disk.get("zone", ""))),
                        "users": list(disk.get("users") or []),
                    }
                )
        req = compute.disks().aggregatedList_next(previous_request=req, previous_response=resp)
    return out


def gcp_project() -> str | None:
    """The project the GCP reconcile passes actually sweep — ambient, from ADC — or ``None`` when
    GCP isn't configured here.

    Reported so a sweep of the *wrong* project is visible (GCP-LEAK-7): SkyPilot can be pinned to
    a different project in ``~/.sky/config.yaml``, in which case reconcile scans a project the lab
    never launches into and truthfully reports it clean. Without the project in the report the
    reader cannot tell that all-clear from a real one.

    Swallows *every* failure, not just :class:`GcpNotConfigured`. This is read into the reconcile
    report after the destroy passes have already run, and it is a label rather than a leak signal
    — letting a transport hiccup here propagate would discard the report of what was just
    destroyed, on the one command whose purpose is an auditable account of what it did.
    """
    try:
        _creds, project = _gcp_default_credentials()
    except Exception:  # noqa: BLE001 — a missing label must never cost us the destroy report
        return None
    return str(project) if project else None


# A GCE node SkyPilot built for one of our clusters, e.g. (real, from the live run 2026-08-11):
#
#   lab-20260811-144501-c5b340-3dd12990-head-c0h9pkx0-compute
#   \_/ \___________________/ \______/ \__/ \______/ \_____/
#    |    cluster_name_for()   user     node  uuid8   node type
#    |                         hash                   (compute|tpu|mig)
#    `-- our prefix
#
# Two parts, and only two are safe to rely on: our `lab-` prefix, and the suffix
# `sky.provision.gcp.instance_utils._generate_node_name` appends to every instance it creates.
#
# The middle is deliberately unconstrained. It is tempting to anchor on `lab-<job_id>`, but
# `make_cluster_name_on_cloud` appends an 8-char user hash *and* truncates past GCP's 35-char
# limit — and `lab-<job_id>` is 26 chars against a budget of 35-9=26, i.e. it fits by exactly
# zero characters. One more character in a job id and the name becomes
# `lab-<trunc>-<2ch>-<userhash>-head-…`, with no recoverable job id in it. An anchored pattern
# would then match nothing and the leak pass would report clean forever, which is strictly worse
# than the over-broad matching this replaced.
#
# Matching the node suffix rather than a bare `lab-` prefix is what keeps `reconcile --apply` off
# a shared project's `lab-notebook` (GCP-LEAK-7); the CLI's confirmation prompt is the second
# layer, for when this predicate is wrong anyway. A GCE boot disk
# inherits its instance's name (the boot disk's `initializeParams` sets no `diskName`), so the
# same shape identifies both. Pinned by `test_the_predicate_accepts_names_skypilot_itself_
# generates`, which builds its input with SkyPilot's own naming functions rather than ours.
# The uuid is exactly INSTANCE_NAME_UUID_LEN=8 base36 chars and the node type is one of
# GCPNodeType's three values. Both are pinned rather than left loose so a hand-named
# `lab-ml-worker-2-gpu` in a shared project cannot match; the round-trip test enumerates the real
# enum, so a new SkyPilot node type fails there instead of silently going unmatched here.
_GCP_NODE_RE = re.compile(
    r"^lab-.+-(?:head|worker)-[0-9a-z]{8}-(?:compute|tpu|mig)$", re.IGNORECASE
)


def is_lab_cluster_node(name: str) -> bool:
    """True iff ``name`` is a GCE instance (or its boot disk) that SkyPilot created for a lab
    cluster. Deliberately narrow: everything this returns True for is something
    ``reconcile --apply`` may destroy unprompted."""
    return bool(_GCP_NODE_RE.match(name))


def gcp_instance_orphans(
    instances: list[dict[str, Any]], running_clusters: set[str]
) -> list[dict[str, Any]]:
    """Lab-cluster GCE instances not tied to any running cluster — the out-of-band GCP analogue of
    the Vast rental pass (SkyPilot names instances after their cluster, so a running cluster name
    is a substring of its instance names). Pure."""
    orphans: list[dict[str, Any]] = []
    for inst in instances:
        name = str(inst.get("name", ""))
        if not is_lab_cluster_node(name):
            continue  # not ours — leave it alone
        if any(c.lower() in name.lower() for c in running_clusters):
            continue  # backs a live job
        orphans.append(inst)
    return orphans


def gcp_disk_orphans(
    disks: list[dict[str, Any]], running_clusters: set[str]
) -> list[dict[str, Any]]:
    """Unattached lab-cluster persistent disks — a disk that outlived its VM keeps billing (the
    GCP analogue of the DO volume-leak pass). Attached disks are skipped: they die with the VM.
    Pure."""
    orphans: list[dict[str, Any]] = []
    for disk in disks:
        name = str(disk.get("name", ""))
        if not is_lab_cluster_node(name):
            continue
        if disk.get("users"):
            continue  # attached — deleted together with its instance
        if any(c.lower() in name.lower() for c in running_clusters):
            continue
        orphans.append(disk)
    return orphans


def gcp_unmatched_lab_names(*resource_lists: list[dict[str, Any]]) -> list[str]:
    """``lab-*`` GCE resources that do **not** match our node shape — the narrowing's safety net.

    Reported, never destroyed. Narrowing the delete predicate trades a destructive false positive
    for the chance of a silent false negative; surfacing what we chose not to claim keeps a leak
    in an unexpected shape visible instead of dropping it on the floor. Advisory only, so it does
    not trip `lab wait`'s exit 3. Pure.
    """
    return sorted(
        {
            name
            for resources in resource_lists
            for r in resources
            if (name := str(r.get("name", ""))).startswith("lab-") and not is_lab_cluster_node(name)
        }
    )


def _await_zone_operation(compute: Any, project: str, zone: str, operation: dict[str, Any]) -> None:
    """Block until a zonal operation finishes and raise if it failed.

    ``instances().delete()`` returns an **Operation**, not a completed delete: GCE deletes take
    30-60s and can fail *after* acceptance (``RESOURCE_IN_USE_BY_ANOTHER_RESOURCE``, a quota error
    on the disk detach, a stuck zonal op). Treating the accepted request as a destroyed VM writes
    ``teardown_status="succeeded"`` while the instance still bills — so we wait for the real
    outcome (FR-C2). ``zoneOperations.wait`` blocks server-side (~2 min max) and returns the
    operation whether or not it is DONE.
    """
    name = operation.get("name")
    if not name:  # nothing to poll (a fake, or an API that already returned a completed op)
        return
    done = compute.zoneOperations().wait(project=project, zone=zone, operation=name).execute()
    error = (done or {}).get("error")
    if error:
        errors = "; ".join(str(e.get("message", e)) for e in error.get("errors", [error]))
        raise RuntimeError(f"operation {name} failed: {errors}")
    if (done or {}).get("status") != "DONE":
        raise RuntimeError(f"operation {name} did not complete (status={done.get('status')})")


def delete_gcp_instance(
    name: str, zone: str, compute: Any | None = None, project: str | None = None
) -> None:
    """Provider-direct instance delete (bypasses SkyPilot's registry). Waits for the delete to
    actually complete — see :func:`_await_zone_operation`."""
    if compute is None or project is None:
        compute, project = _get_gcp_compute()
    op = compute.instances().delete(project=project, zone=zone, instance=name).execute()
    _await_zone_operation(compute, project, zone, op or {})


def delete_gcp_disk(
    name: str, zone: str, compute: Any | None = None, project: str | None = None
) -> None:
    """Provider-direct disk delete (unattached leftovers only — attached deletes 400). Waits for
    completion, same reasoning as :func:`delete_gcp_instance`."""
    if compute is None or project is None:
        compute, project = _get_gcp_compute()
    op = compute.disks().delete(project=project, zone=zone, disk=name).execute()
    _await_zone_operation(compute, project, zone, op or {})


def _gcp_destroy_matching(cluster: str) -> tuple[list[str], list[str]]:
    """Destroy every GCE instance whose name contains ``cluster``.

    Returns ``(destroyed, failures)``. We keep going after a failure — the next instance might
    still die — but the failures are **returned, not just printed**: a destroy we attempted and
    could not complete is a live, billing box, and the caller must not report that as a clean
    teardown (FR-C2).
    """
    compute, project = _get_gcp_compute()
    needle = cluster.lower()
    destroyed: list[str] = []
    failures: list[str] = []
    for inst in list_gcp_instances(compute, project):
        if needle not in inst["name"].lower():
            continue
        try:
            delete_gcp_instance(inst["name"], inst["zone"], compute, project)
            destroyed.append(inst["name"])
            print(f"[lab] gcp-direct destroyed instance {inst['name']} (cluster={cluster})")
        except Exception as e:  # noqa: BLE001 — best-effort; the next instance might still go
            failures.append(f"{inst['name']}: {type(e).__name__}: {e}")
            print(f"[lab] gcp-direct destroy {inst['name']} failed: {e}")
    return destroyed, failures


def confirm_no_rental(cluster: str) -> bool:
    """True iff no Vast rental labelled for this cluster remains. Best-effort: returns False on any
    match OR if the listing fails — we never claim 'gone' under uncertainty (FR-C2)."""
    try:
        instances = list_vast_instances()
    except Exception:  # noqa: BLE001 — uncertainty must read as "still maybe billing"
        return False
    needle = cluster.lower()
    return not any(needle in _instance_label(inst) for inst in instances)  # _instance_label is lower


def confirm_no_instance(cluster: str) -> bool:
    """True iff no GCE instance named for this cluster remains — the GCP twin of
    :func:`confirm_no_rental`, with the same fail-toward-alarm contract: any match OR a failed
    listing returns False, because we never claim "gone" under uncertainty (FR-C2).

    Status is deliberately ignored. This runs *after* teardown, so a surviving instance record in
    any state (a TERMINATED preemptible still holds its boot disk) means the teardown didn't take.
    """
    try:
        instances = list_gcp_instances()
    except Exception:  # noqa: BLE001 — uncertainty must read as "still maybe billing"
        return False
    needle = cluster.lower()
    return not any(needle in str(inst.get("name", "")).lower() for inst in instances)


def gcp_preemption_state(cluster: str) -> JobState | None:
    """Ask GCE directly why ``cluster``'s VM stopped — ``preempted``, ``failed``, or no answer.

    The provider-direct second opinion for *classification*, mirroring what
    :func:`confirm_no_instance` already does for teardown. ``None`` means GCE could not answer
    (GCP not configured, the listing failed, nothing terminal to read) and the caller keeps its
    inference: this probe only ever *refines* the classifier's inputs, never removes them, so a
    revoked role leaves today's behaviour exactly as it was (GCP-PREEMPT-1).
    """
    try:
        instances = list_gcp_instances()
    except Exception:  # noqa: BLE001 — no answer is a valid answer here; inference still applies
        return None
    needle = cluster.lower()
    return gcp_terminal_state(
        [i for i in instances if needle in str(i.get("name", "")).lower()]
    )


def preempted_teardown_confirmed(cloud: str, cluster: str) -> bool:
    """Whether a preempted job's instance is confirmably gone.

    Vast and GCP both expose a provider-direct listing to double-check against, and an unmanaged
    spot preemption is the likeliest way a box outlives its job — so both get a second opinion
    rather than trusting the teardown call's own return. DigitalOcean has no such channel, so
    there the :func:`tear_down_and_record` outcome is already the authoritative answer.
    """
    if cloud == "vast":
        return confirm_no_rental(cluster)
    if cloud == "gcp":
        return confirm_no_instance(cluster)
    return True


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


def catalog_hourly(res: ResourceRequest) -> float | None:
    """Worst-case USD/hour for a spec from SkyPilot's catalog — a **local** lookup, no provisioning.

    Vast has a live offer feed, so its registrations price themselves from real offers. Every
    other cloud has none, which left the scheduler's cost guardrails with no number to check and
    therefore silently unenforced (a `--max-cost` the user set and the CLI accepted). The catalog
    is the pre-launch estimate for those clouds. Returns None if the catalog can't price the spec,
    so a missing price never blocks a launch — the guardrail degrades to its old behaviour rather
    than becoming a new failure mode.

    **This deliberately returns the top of the price band, not a point estimate.** It used to call
    ``sky.Resources(...).get_cost()`` with no region pinned, which returns the *minimum* across
    every region offering the resource — so the guardrail that was meant to cap a job's spend was
    checking its best case, under-estimating by up to 3.6x on GCP spot. Its callers are admission
    controls; an admission control that under-estimates admits jobs it should refuse. See
    :func:`lab.placement.estimate` for how the ceiling accounts for spot fallback and storage.
    """
    from lab import placement

    est = placement.estimate(res)
    if est is None:
        placement._note(f"[lab] catalog price unavailable for {res.cloud or 'vast'}")
        return None
    return est.worst_hourly_usd


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


def _vast_destroy_matching(
    cluster: str, client: Any | None = None
) -> tuple[list[int], list[str]]:
    """Destroy every Vast rental whose label contains ``cluster``.

    Returns ``(destroyed, failures)``. We keep going after a failure — the next rental might still
    go — but the failures are **returned, not just printed**: a rental we found and could not
    destroy is a live, billing box, and the caller must not report that as a clean teardown
    (FR-C2). Vast bills the most per hour of any backend, so optimism is most expensive here.
    """
    if client is None:
        client = _get_vast_client()
    needle = cluster.lower()
    destroyed: list[int] = []
    failures: list[str] = []
    for inst in list_vast_instances(client=client):
        if needle not in _instance_label(inst):
            continue
        inst_id = inst.get("id")
        if inst_id is None:
            failures.append(f"rental with no id (label={_instance_label(inst)!r})")
            continue
        try:
            client.destroy_instance(id=int(inst_id))
            destroyed.append(int(inst_id))
            print(f"[lab] vast-direct destroyed instance {inst_id} (cluster={cluster})")
        except Exception as e:  # noqa: BLE001 — best-effort; the next instance might still go
            failures.append(f"{inst_id}: {type(e).__name__}: {e}")
            print(f"[lab] vast-direct destroy {inst_id} failed: {e}")
    return destroyed, failures


def robust_teardown(
    sky_mod: Any, cluster: str, *, backoffs: tuple[int, ...] = TEARDOWN_BACKOFFS, cloud: str = "vast"
) -> dict[str, Any]:
    """Tear down a SkyPilot cluster with retry + a provider-direct fallback (Vast/GCP).

    Why this exists: a single ``sky.down`` call that swallows transient errors (network hiccup,
    provider API timeout) used to leak rentals — the cluster kept billing while we marked the
    job ``failed`` and moved on. We now retry sky.down with exponential backoff, then bypass
    SkyPilot's local registry entirely and ask the provider itself to destroy any instance
    matching the cluster name (vastai-sdk on Vast, the compute API on GCP; DO has no direct
    channel and reports the failure).

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

    # SkyPilot teardown didn't take.
    if cloud == "gcp":
        # Talk to the GCP compute API directly — bypasses SkyPilot's registry.
        print(f"[lab] sky.down exhausted for {cluster}; falling back to gcp-direct destroy")
        try:
            gcp_destroyed, gcp_failures = _gcp_destroy_matching(cluster)
            return {
                # Destroyed-or-none-found are both safe. Found-and-failed-to-destroy is not: that
                # is a live box we know about and could not kill, so it must alarm (FR-C2).
                "status": "failed" if gcp_failures else "succeeded",
                "attempts": len(delays),
                "vast_fallback_used": False,
                "vast_destroyed": [],
                "gcp_fallback_used": True,
                "gcp_destroyed": gcp_destroyed,  # report what DID die even when we alarm
                "error": (
                    f"sky.down: {last_err}; gcp-direct: {'; '.join(gcp_failures)}"
                    if gcp_failures
                    else last_err
                ),
            }
        except Exception as e:  # noqa: BLE001
            return {
                "status": "failed",
                "attempts": len(delays),
                "vast_fallback_used": False,
                "vast_destroyed": [],
                "gcp_fallback_used": True,
                "gcp_destroyed": [],
                "error": f"sky.down: {last_err}; gcp-direct: {type(e).__name__}: {e}",
            }
    if cloud != "vast":
        # No provider-direct fallback for this cloud (DO); sky.down + autostop + the poweroff
        # backstop + `lab reconcile` (sky.status pass) are the safety net. Report the failure.
        return {
            "status": "failed",
            "attempts": len(delays),
            "vast_fallback_used": False,
            "vast_destroyed": [],
            "error": last_err,
        }
    # Talk to Vast directly — it's the source of truth.
    print(f"[lab] sky.down exhausted for {cluster}; falling back to vast-sdk direct destroy")
    try:
        destroyed, failures = _vast_destroy_matching(cluster)
        return {
            # Destroyed-or-none-found are both safe. Found-and-failed-to-destroy is not: that is
            # a live rental we know about and could not kill, so it must alarm (FR-C2).
            "status": "failed" if failures else "succeeded",
            "attempts": len(delays),
            "vast_fallback_used": True,
            "vast_destroyed": destroyed,  # report what DID die even when we alarm
            "error": (
                f"sky.down: {last_err}; vast-direct: {'; '.join(failures)}"
                if failures
                else last_err
            ),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "status": "failed",
            "attempts": len(delays),
            "vast_fallback_used": True,
            "vast_destroyed": [],
            "error": f"sky.down: {last_err}; vast-direct: {type(e).__name__}: {e}",
        }


def tear_down_and_record(
    sky_mod: Any,
    cluster: str,
    store: JobStore,
    job_id: str,
    cloud: str = "vast",
    *,
    backoffs: tuple[int, ...] | None = None,
) -> bool:
    """Call :func:`robust_teardown` and persist its outcome on the job manifest.

    Returns ``True`` iff teardown succeeded. On failure, ``teardown_status='failed'`` is
    written and ``end_reason`` is annotated with an actionable instruction so the leak is
    visible in ``lab status`` / ``lab dashboard`` / ``lab wait``. ``backoffs`` overrides the
    retry ladder for callers that must stay quick (e.g. a status poll); None keeps the default.
    """
    if backoffs is None:
        outcome = robust_teardown(sky_mod, cluster, cloud=cloud)
    else:
        outcome = robust_teardown(sky_mod, cluster, cloud=cloud, backoffs=backoffs)
    succeeded: bool = outcome["status"] == "succeeded"
    fields: dict[str, Any] = {"teardown_status": "succeeded" if succeeded else "failed"}
    annotation: str | None = None
    if not succeeded:
        if cloud == "vast":
            remedy = (
                "AND vast-sdk fallback. Run `lab reconcile --apply` "
                "(or `vastai destroy_instance <id>`) to stop the bleed."
            )
        elif cloud == "gcp":
            remedy = (
                "AND gcp-direct fallback. Run `lab reconcile --apply` and check "
                "`gcloud compute instances list --filter=\"name~'^lab-'\"` to stop the bleed."
            )
        else:
            remedy = (
                f"(no provider-direct fallback for {cloud}). Run `lab reconcile --apply` and "
                f"check `sky status` / the {cloud} console to stop the bleed."
            )
        annotation = (
            f"TEARDOWN FAILED for cluster {cluster!r}: {outcome['error']} after "
            f"{outcome['attempts']} sky.down attempts {remedy}"
        )
    elif outcome["vast_fallback_used"]:
        annotation = (
            f"sky.down failed ({outcome['error']}); vast-sdk fallback destroyed "
            f"{outcome['vast_destroyed']}"
        )
    elif outcome.get("gcp_fallback_used"):
        annotation = (
            f"sky.down failed ({outcome['error']}); gcp-direct fallback destroyed "
            f"{outcome.get('gcp_destroyed')}"
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


def _cloud_for(name: str | None) -> "sky.clouds.Cloud":
    """Map a lab cloud name to a SkyPilot cloud object. None -> Vast (the default); an unknown
    name raises rather than silently landing on Vast (defense in depth for hand-edited
    manifests — the CLI/MCP already validate via ``validate_cloud``)."""
    import sky

    from lab.core import LabError

    clouds = {
        "vast": sky.clouds.Vast,
        "do": sky.clouds.DO,
        "gcp": sky.clouds.GCP,
    }
    key = name or "vast"
    if key not in clouds:
        raise LabError(f"unknown cloud {key!r} on manifest; supported: {', '.join(clouds)}")
    return clouds[key]()


def narrowed_regions(res: ResourceRequest, memo: Any | None) -> list[str | None]:
    """Regions to offer SkyPilot, or ``[None]`` meaning "don't narrow — use the full search space".

    We narrow for exactly one reason: the capacity memo knows some zones just ran out, and without
    that knowledge every shard of a sweep independently walks into them. Narrowing is skipped
    whenever it would be guesswork or harm:

    * nothing excluded -> ``[None]``, byte-identical to the pre-memo behaviour;
    * an explicit ``--region``/``--zone`` -> ``[None]``, because the pin is already the constraint
      and the memo must not silently override what the user asked for;
    * every region excluded -> ``[None]``, because a memo that would block the launch entirely is
      worse than a memo that is ignored.

    Ordering stays SkyPilot's: it prices the set we hand it and picks the cheapest available.
    """
    if memo is None or res.region is not None or res.zone is not None:
        return [None]
    from lab import placement

    instance_type = placement.resolve_instance_type(res)
    if instance_type is None:
        return [None]
    cloud = res.cloud or "vast"
    if not memo.exhausted_zones(cloud, instance_type):
        return [None]
    surviving = placement.candidates(res, instance_type=instance_type, memo=memo)
    if not surviving:
        placement._note(
            "[lab] capacity memo would exclude every region; ignoring it for this launch"
        )
        return [None]
    # Narrow only if the memo actually cost us a region. A single dead zone in a multi-zone region
    # leaves that region usable, so the memo can be non-empty while changing nothing — and
    # narrowing anyway would silently trade 40 regions of failover for 10, buying nothing.
    if len(surviving) == len(placement.candidates(res, instance_type=instance_type, memo=None)):
        return [None]
    names = [c.region for c in surviving[: placement.MAX_NARROWED_REGIONS]]
    placement._note(
        f"[lab] capacity memo: narrowed to {len(names)} region(s), cheapest {names[0]}"
    )
    return list(names)


def build_task(manifest: JobManifest, workdir: Path, *, memo: Any | None = None) -> sky.Task:
    """Translate a JobManifest into a SkyPilot Task (no cloud calls; unit-tested)."""
    import sky

    from lab import placement

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
    res = manifest.resources
    cloud_name = res.cloud or "vast"
    if cloud_name == "do" and res.use_spot:
        from lab.core import LabError  # lazy: avoid import cycle

        raise LabError("DigitalOcean has no spot instances; drop --spot")
    placement.validate_placement(res)  # a bad region name fails here, before anything bills
    _cloud = _cloud_for(cloud_name)
    # Last line of defence for the disk invariant. `resolve_backend_profile` also applies it, but
    # only on the CLI/MCP submit path — the scheduler launches registrations straight through
    # `Lab.submit`, so without this a deferred GCP job still inherits SkyPilot's 256 GB default.
    disk_gb = placement.effective_disk_gb(res)

    def _res(*, use_spot: bool | None = None, region: str | None = None) -> sky.Resources:
        return sky.Resources(
            cloud=_cloud,
            cpus=res.cpus,
            memory=res.memory,
            accelerators=res.accelerators or None,
            disk_size=disk_gb,
            use_spot=use_spot,
            region=region or res.region,
            zone=res.zone,
            # A ceiling the optimizer itself honours: it will not select an option above this, so
            # the worst case we quote the user is enforced rather than merely predicted.
            max_hourly_cost=res.max_hourly_usd,
        )

    if not res.use_spot:
        spots: list[bool | None] = [None]
    elif res.spot_fallback:
        # Prefer spot (cheaper); SkyPilot's optimizer fails over to on-demand if spot is scarce.
        spots = [True, False]
    else:
        spots = [True]  # spot-only, no fallback

    options = [_res(use_spot=s, region=r) for r in narrowed_regions(res, memo) for s in spots]
    task.set_resources(options[0] if len(options) == 1 else options)
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
                # The supervisor died before recording terminal state, so its teardown very
                # likely never ran — attempt it here (idempotent) rather than flip to `failed`
                # on a possibly-still-billing box (FR-C2). Quick backoffs: this is a status
                # poll; `lab reconcile` is the heavier net. Once terminal, this branch can't
                # re-enter, so the attempt runs at most once.
                cluster = rt.get("cluster") or cluster_name_for(job_id)
                try:
                    import sky

                    tear_down_and_record(
                        sky, cluster, self.store, job_id, m.resources.cloud or "vast",
                        backoffs=(5,),
                    )
                except Exception as e:  # noqa: BLE001 — the alarm must survive a crashed attempt
                    print(f"[lab] teardown attempt for dead-supervisor job {job_id} crashed: {e}")
                    self.store.update_manifest(job_id, teardown_status="failed")
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

        tear_down_and_record(sky, cluster, self.store, job_id, m.resources.cloud or "vast")
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
