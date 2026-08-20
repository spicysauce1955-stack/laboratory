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

import hashlib
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lab import events
from lab._util import (
    infer_artifact_type,
    now,
    parse_duration,
    pid_alive,
    process_start_time,
)
from lab.manifest import repo_root, sha256_file
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


# ----------------------------------------------------------------------------
# Project identity on provisioned resources
#
# A job id is a timestamp plus randomness, so `lab-<job_id>` says *when* a box was launched and
# nothing about *who* launched it — and the `lab-` prefix is shared by every project on the
# machine and in the cloud account. That is why `lab reconcile --apply --yes` was able to destroy
# seven running clusters belonging to a different project on 2026-08-20: from the cloud side the
# question "is this mine?" had no answer, not even a wrong one.
#
# So we stamp the project onto the resource at launch, by the two carriers that survive:
#
# * the **cluster name** — the one string that reaches every cloud, since SkyPilot derives the
#   instance/rental name from it on Vast, DO and GCP alike; and
# * real **instance labels**, where the cloud has them (see :func:`project_labels`).
#
# The name is the load-bearing one: it is what `sky.status`, the Vast rental label and the GCP
# instance name all show, and it is the only carrier that works on a cloud SkyPilot cannot label.
# ----------------------------------------------------------------------------

CLUSTER_NAME_MAX = 60  # SkyPilot cluster name budget (pre-existing; unchanged by the slug)
# Keep the slug short: it is spent from the same 60 characters as the job id, and — worse — GCP
# truncates the whole name to 35 characters on the cloud, where every character we add pushes
# more of the tail off the instance name.
PROJECT_SLUG_MAX = 12
_SLUG_DIGEST_LEN = 4  # collision breaker appended when a slug is truncated or unrepresentable

# The real job-id shape, from `lab.core._new_job_id`: `%Y%m%d-%H%M%S` + 6 hex. Pinning the shape
# is what makes the name parseable at all — the slug and the job id are both hyphenated lowercase
# text, so only a fixed-width anchor at the end can tell where one ends and the other begins.
_JOB_ID_PATTERN = r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}"
# Both shapes in one pattern: `lab-<slug>-<job_id>` (new) and `lab-<job_id>` (legacy, still
# running in the wild). The slug group is optional, never greedy past the anchored tail.
_CLUSTER_NAME_RE = re.compile(rf"^lab-(?:(?P<project>[a-z0-9-]+)-)?(?P<job_id>{_JOB_ID_PATTERN})$")


@dataclass(frozen=True)
class ClusterIdentity:
    """Who and what a cluster name refers to.

    ``project`` is the *slug* (:func:`project_slug`), not the directory name — the name only ever
    carried the slug. ``None`` means the name predates project stamping, i.e. **ownership
    unknown**, which is emphatically not the same as "not ours": a legacy cluster is one we may
    well own, and treating unknown as foreign would leak it, while treating it as ours is what
    caused the incident. Callers must decide explicitly.
    """

    job_id: str
    project: str | None


def project_slug(project: str) -> str:
    """A short, stable, cluster-name-safe slug for a project name.

    Directory names are unconstrained (spaces, underscores, capitals, CJK, leading digits);
    cluster names are lowercase alphanumerics and hyphens. Anything unrepresentable collapses to
    hyphens, and a name that leaves *nothing* behind still gets a slug — a digest — because an
    empty slug would silently emit the legacy shape and lose the attribution we came here for.

    Truncation appends the same digest rather than cutting bare, because a bare prefix is exactly
    how two neighbouring projects (`machine-learning-alpha`, `machine-learning-beta`) would map
    onto one name and hand `reconcile` back the false match this whole change removes.
    """
    safe = re.sub(r"[^a-z0-9]+", "-", project.lower()).strip("-")
    digest = hashlib.sha256(project.encode("utf-8")).hexdigest()[:_SLUG_DIGEST_LEN]
    if not safe:
        return f"p{digest}"  # leading letter: a slug may head the name after the `lab-` prefix
    if len(safe) <= PROJECT_SLUG_MAX:
        return safe
    head = safe[: PROJECT_SLUG_MAX - _SLUG_DIGEST_LEN - 1].rstrip("-")
    return f"{head}-{digest}"


@lru_cache(maxsize=32)
def _project_name(_cwd: str, _repo_dir: str) -> str | None:
    """Cached repo-directory name. The arguments are cache keys only.

    ``repo_root`` shells out to ``git rev-parse``, and `Lab.reconcile` asks for a cluster name
    once per job — so an uncached lookup turns a leak sweep into a few hundred subprocesses. The
    two things that can change the answer (the cwd and the ``LAB_REPO_DIR`` override) are passed
    in so they key the cache instead of being invisible to it.
    """
    try:
        return repo_root().name or None
    except Exception as e:  # noqa: BLE001 — not every cwd is a repo, and a name is never critical
        events.note("project probe failed", error=str(e))
        return None


def current_project() -> str | None:
    """The project this process is operating on — the repo directory name, ``None`` if unknowable.

    Deliberately the *same* derivation the event ledger uses (``lab.events.record._project``):
    the ledger and the cluster names are read side by side when a run is being traced, and two
    disagreeing notions of "project" would make that join silently wrong. Going through
    :func:`lab.manifest.repo_root` also means ``LAB_REPO_DIR`` is honoured, which is what keeps
    the scheduler host — whose cwd is not the repo — from stamping its own directory on every
    job it launches.
    """
    return _project_name(os.getcwd(), (os.environ.get("LAB_REPO_DIR") or "").strip())


def cluster_name_for(job_id: str, *, project: str | None = None) -> str:
    """SkyPilot cluster name for a job: ``lab-<project-slug>-<job_id>``.

    Starts with a letter, lowercase alphanumerics and hyphens, at most
    :data:`CLUSTER_NAME_MAX` characters — e.g. job ``20260820-071905-771110`` in the
    ``laboratory`` repo becomes ``lab-laboratory-20260820-071905-771110`` (37 chars).

    ``project`` defaults to :func:`current_project`; pass it explicitly when launching on behalf
    of a project that is not the caller's own working directory (the scheduler's case).

    **The job id never gives way.** It is the key reconcile and teardown match on, so when the
    budget is tight the *slug* is dropped — a job id long enough to consume the whole budget
    simply yields the legacy ``lab-<job_id>`` shape rather than a truncated, unrecoverable id.
    """
    safe_job = _safe_segment(job_id)
    name = current_project() if project is None else project
    slug = project_slug(name) if name else ""
    budget = CLUSTER_NAME_MAX - len("lab-") - len(safe_job) - 1
    if slug and budget >= len(slug):
        return f"lab-{slug}-{safe_job}"
    return f"lab-{safe_job}"[:CLUSTER_NAME_MAX]


def parse_cluster_name(name: str) -> ClusterIdentity | None:
    """Recover the job id (and project slug, if stamped) from a cluster name, or ``None``.

    ``None`` means "not a cluster this tool launched" — a hand-named `lab-notebook` in a shared
    account parses to nothing, which is the answer that keeps a leak sweep off it (GCP-LEAK-7).
    A legacy ``lab-<job_id>`` parses with ``project=None``; see :class:`ClusterIdentity` for why
    that must not be read as "foreign".
    """
    m = _CLUSTER_NAME_RE.match(name.strip().lower())
    if m is None:
        return None
    return ClusterIdentity(job_id=m.group("job_id"), project=m.group("project"))


def project_labels(job_id: str, *, project: str | None = None) -> dict[str, str]:
    """Cloud instance labels stamping the owning project (and job) onto the resource itself.

    A second carrier alongside the name, and on GCP a strictly better one: SkyPilot truncates the
    cluster name to GCP's 35-character instance-name limit, so `lab-laboratory-20260820-071905-
    771110` reaches the console as `lab-laboratory-20260820-ef-<userhash>` — project intact, job
    id shorn. The label keeps the full id, queryable via ``--filter labels.lab-job-id=…``.

    **Not every cloud stores these.** SkyPilot 0.12.3 renders `labels` into `gcp-ray.yml.j2` (and
    AWS/Kubernetes) only; `do-ray.yml.j2` and `vast-ray.yml.j2` have no labels block, and the DO
    provisioner tags droplets from `ProvisionConfig.tags`, which `provisioner.py` hard-codes to
    `{}`. Their base `Cloud.is_label_valid` accepts anything, so the labels are *dropped without
    error* there — which is why the cluster name, not this, is the load-bearing carrier. Setting
    them unconditionally anyway costs nothing and starts working the day SkyPilot wires a cloud up.

    Keys and values are constrained by GCP's validator (`^[a-z]([a-z0-9_-]{0,62})?$` /
    `^[a-z0-9_-]{0,63}$`); the slug and the sanitised job id are inside it by construction, and
    ``test_gcp_accepts_our_labels`` asks GCP's own validator rather than trusting this comment.
    """
    labels = {"lab-job-id": _safe_segment(job_id)[:63]}
    name = current_project() if project is None else project
    if name:
        labels["lab-project"] = project_slug(name)
    return labels


def _safe_segment(value: str) -> str:
    """Lowercase, hyphenate anything a cluster name cannot hold. Unchanged from the original."""
    return re.sub(r"[^a-z0-9-]", "-", value.lower()).strip("-")


def build_setup_script() -> str:
    """Install uv and materialise the locked env on the remote (FR-B2)."""
    return (
        "set -e\n"
        "curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        # --no-default-groups: skip dev/test tooling (pytest, ruff, mypy) — a provisioned box
        # runs the experiment, it does not lint or test the project. Note this no longer keeps
        # the lab's control plane off the remote: the project depends on `laboratory`, so its
        # runtime deps install here as a matter of course.
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

    Only the fields an instance actually carries are joined. Padding the result with the
    separators of the absent ones rendered a rental holding just ``label`` as
    ``"lab-…-head   "`` — harmless to the ``in`` test callers run, but this string is printed
    verbatim into reconcile reports and teardown failure lines that people read mid-incident.
    """
    parts = [
        str(inst.get(k) or "").strip()
        for k in ("label", "name", "instance_label", "machine_name")
    ]
    return " ".join(p for p in parts if p).lower()


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


def list_do_droplets(client: Any | None = None) -> list[dict[str, Any]]:
    """Every DO droplet on the account as plain dicts. Raises if the client or listing fails —
    a leak-detection caller must never read an API error as "nothing is running"."""
    if client is None:
        client = _get_do_client()
    resp = client.droplets.list(per_page=200)
    droplets = resp.get("droplets", []) if isinstance(resp, dict) else resp
    return [dict(d) for d in (droplets or [])]


def _do_destroy_matching(cluster: str) -> tuple[list[Any], list[str]]:
    """Destroy the droplet(s) and block volume(s) belonging to ``cluster``, via the DO API.

    The provider-direct fallback DigitalOcean was missing while Vast and GCP both had one (F2).
    It matters most in exactly the case SkyPilot cannot help with: its registry has lost the
    cluster (``ClusterDoesNotExist``) but the droplet is alive and billing.

    Two resources, not one. SkyPilot's DO provisioner attaches a block volume named after the
    cluster, and destroying the droplet alone leaves it behind — detached, invisible to every
    instance-level pass, and still charged for. That residue is what the 2026-08-20 sweep found.

    Matching is ``name.startswith(cluster)``, which is exact enough to be safe here: DO does not
    truncate instance names (unlike GCP — see :func:`gcp_name_matches`), and the cluster name now
    carries the owning project, so another project's droplet cannot share this prefix. Returns
    ``(destroyed_ids, failures)``; finding nothing is success, since nothing is billing.
    """
    client = _get_do_client()
    destroyed: list[Any] = []
    failures: list[str] = []
    prefix = cluster.lower()

    for d in list_do_droplets(client):
        if not str(d.get("name", "")).lower().startswith(prefix):
            continue
        try:
            client.droplets.destroy(droplet_id=int(d["id"]))
            destroyed.append(int(d["id"]))
        except Exception as e:  # noqa: BLE001 — a box we found and could not kill must alarm
            failures.append(f"droplet {d.get('name')}: {type(e).__name__}: {e}")

    for v in list_do_volumes(client):
        if not str(v.get("name", "")).lower().startswith(prefix):
            continue
        try:
            client.volumes.delete(volume_id=str(v["id"]))
            destroyed.append(v["id"])
        except Exception as e:  # noqa: BLE001
            failures.append(f"volume {v.get('name')}: {type(e).__name__}: {e}")

    return destroyed, failures


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
    """Every GCE instance on the project as ``{name, zone, status, preemptible, labels}`` dicts, via
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
                        # GCE's own copy of what `project_labels` stamped at launch. Unlike the
                        # name, a label is not truncated to 35 chars, so this is the *only* place
                        # a GCP instance states its full owning project and job id — which is
                        # what lets a leak sweep tell one project's box from another's rather
                        # than guessing from a shared `lab-` prefix.
                        "labels": dict(inst.get("labels") or {}),
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


@lru_cache(maxsize=256)
def gcp_name_fragment(cluster: str) -> str:
    """The part of ``cluster``'s GCE resource names that is stable enough to match on.

    SkyPilot does not name a GCE instance after the cluster verbatim: it runs the display name
    through ``make_cluster_name_on_cloud`` at GCP's 35-character limit, which **truncates to 23
    characters and appends a 2-char digest plus the launching user's hash** before
    ``_generate_node_name`` adds ``-<head|worker>-<uuid8>-<compute|tpu|mig>``.

    This never mattered while clusters were called ``lab-<job_id>``, because that is 26 characters
    against a 26-character budget — it fit by exactly zero, so the display name survived intact
    and a plain substring test worked. Adding a project slug spends that zero. Without this
    helper, `lab-laboratory-20260820-071905-771110` stops being a substring of its own instance
    `lab-laboratory-20260820-ef-<userhash>-head-…`, every *live* GCP box stops matching its
    running job, and `reconcile --apply` proceeds to destroy it — the 2026-08-20 incident,
    recreated by the fix for it.

    We drop the trailing user hash so the match does not silently depend on *which* user runs
    `reconcile`, and derive the rest with SkyPilot's own function rather than reimplementing its
    truncation. When nothing was truncated this returns the display name unchanged, so legacy
    ``lab-<job_id>`` clusters match exactly as before. Cached: reconcile asks per instance per job.
    """
    from sky.clouds.gcp import GCP
    from sky.utils.common_utils import make_cluster_name_on_cloud

    on_cloud = make_cluster_name_on_cloud(cluster, max_length=GCP.max_cluster_name_length())
    return on_cloud.rsplit("-", 1)[0].lower()  # strip `-<user_hash>`


def gcp_name_matches(cluster: str, name: str) -> bool:
    """Whether GCE resource ``name`` belongs to cluster ``cluster`` (instances and their disks).

    Falls back to the raw display name if SkyPilot cannot be imported or its naming helpers
    raise: a degraded match may miss a leak, whereas raising here would abort the leak sweep
    entirely, and half a report beats none.
    """
    lowered = name.lower()
    if cluster.lower() in lowered:
        return True
    try:
        return gcp_name_fragment(cluster) in lowered
    except Exception as e:  # noqa: BLE001 — never let a naming helper abort a leak sweep
        events.note("gcp name fragment failed", cluster=cluster, error=str(e))
        return False


def gcp_instance_orphans(
    instances: list[dict[str, Any]], running_clusters: set[str]
) -> list[dict[str, Any]]:
    """Lab-cluster GCE instances not tied to any running cluster — the out-of-band GCP analogue of
    the Vast rental pass (SkyPilot names instances after their cluster — see
    :func:`gcp_name_matches` for why that is not a plain substring test)."""
    orphans: list[dict[str, Any]] = []
    for inst in instances:
        name = str(inst.get("name", ""))
        if not is_lab_cluster_node(name):
            continue  # not ours — leave it alone
        if any(gcp_name_matches(c, name) for c in running_clusters):
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
        if any(gcp_name_matches(c, name) for c in running_clusters):
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
    """Destroy every GCE instance belonging to ``cluster`` (see :func:`gcp_name_matches`).

    Returns ``(destroyed, failures)``. We keep going after a failure — the next instance might
    still die — but the failures are **returned, not just printed**: a destroy we attempted and
    could not complete is a live, billing box, and the caller must not report that as a clean
    teardown (FR-C2).
    """
    compute, project = _get_gcp_compute()
    destroyed: list[str] = []
    failures: list[str] = []
    for inst in list_gcp_instances(compute, project):
        if not gcp_name_matches(cluster, str(inst["name"])):
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
    return not any(gcp_name_matches(cluster, str(inst.get("name", ""))) for inst in instances)


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
    return gcp_terminal_state(
        [i for i in instances if gcp_name_matches(cluster, str(i.get("name", "")))]
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
    try:
        # Constructing the client must be inside the guard, not before it: vastai-sdk raises when
        # no API key exists, so on a GCP-only box (or CI) this best-effort *diagnostic* threw and
        # replaced the real provision failure with "No API key found" — the enrichment destroying
        # the thing it was enriching.
        if client is None:
            client = _get_vast_client()
        info = client.show_user()
    except Exception as e:  # noqa: BLE001 — best-effort; caller falls back to the generic message
        # stderr: this runs on the CLI's JSON path, where a stray stdout line is a corrupted
        # payload rather than a log line.
        print(f"[lab] vast balance lookup failed: {e}", file=sys.stderr)
        events.note("vast.balance_failed", error=str(e))
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
    matching the cluster name: vastai-sdk on Vast, the compute API on GCP, and the DO API on
    DigitalOcean (droplet plus the block volume SkyPilot attaches to it).

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
    last_undecodable = False
    delays = (0, *backoffs)  # first try has no delay
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        events.note("teardown.attempt", cluster=cluster, attempt=attempt)
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
            # Was the *reply* unreadable rather than the request refused? Under client/server
            # version skew `sky.get` cannot unpickle a SUCCESS, so this error very likely sits on
            # top of a teardown that actually happened (incident 2026-08-20). Remembered, not
            # acted on here: a provider-direct fallback below can still settle it outright.
            from lab._skycompat import classify_sky_error

            last_undecodable = classify_sky_error(e).outcome == "undecodable_response"
            print(
                f"[lab] sky.down attempt {attempt}/{len(delays)} for {cluster} failed: {last_err}"
            )
            events.note("teardown.retry", cluster=cluster, attempt=attempt, error=last_err)

    # SkyPilot teardown didn't take.
    if cloud == "gcp":
        # Talk to the GCP compute API directly — bypasses SkyPilot's registry.
        print(f"[lab] sky.down exhausted for {cluster}; falling back to gcp-direct destroy")
        try:
            gcp_destroyed, gcp_failures = _gcp_destroy_matching(cluster)
            events.note("teardown.fallback", cluster=cluster, via="gcp", ok=not gcp_failures)
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
            events.note("teardown.fallback", cluster=cluster, via="gcp", ok=False)
            return {
                # sky could not tell us and GCP could not either: genuinely unknown (R10).
                "status": "unknown" if last_undecodable else "failed",
                "attempts": len(delays),
                "vast_fallback_used": False,
                "vast_destroyed": [],
                "gcp_fallback_used": True,
                "gcp_destroyed": [],
                "error": f"sky.down: {last_err}; gcp-direct: {type(e).__name__}: {e}",
            }
    if cloud == "do":
        # Talk to DigitalOcean directly — droplet first, then the block volume SkyPilot attaches
        # alongside it, which outlives the droplet and keeps billing if left (F2).
        print(f"[lab] sky.down exhausted for {cluster}; falling back to do-direct destroy")
        try:
            do_destroyed, do_failures = _do_destroy_matching(cluster)
            events.note("teardown.fallback", cluster=cluster, via="do", ok=not do_failures)
            return {
                # Destroyed-or-none-found are both safe. Found-and-failed-to-destroy is not.
                "status": "failed" if do_failures else "succeeded",
                "attempts": len(delays),
                "vast_fallback_used": False,
                "vast_destroyed": [],
                "do_fallback_used": True,
                "do_destroyed": do_destroyed,  # report what DID die even when we alarm
                "error": (
                    f"sky.down: {last_err}; do-direct: {'; '.join(do_failures)}"
                    if do_failures
                    else last_err
                ),
            }
        except Exception as e:  # noqa: BLE001 — an unreachable DO API is not "nothing is running"
            events.note("teardown.fallback", cluster=cluster, via="do", ok=False)
            return {
                # sky could not tell us and DO could not either: genuinely unknown (R10).
                "status": "unknown" if last_undecodable else "failed",
                "attempts": len(delays),
                "vast_fallback_used": False,
                "vast_destroyed": [],
                "do_fallback_used": True,
                "do_destroyed": [],
                "error": f"sky.down: {last_err}; do-direct: {type(e).__name__}: {e}",
            }
    if cloud != "vast":
        # No provider-direct fallback for this cloud; sky.down + autostop + the poweroff
        # backstop + `lab reconcile` (sky.status pass) are the safety net. Nothing here can
        # settle an unreadable reply, so say "unknown" rather than raising a leak alarm we
        # cannot stand behind — a `failed` that is usually wrong teaches operators to ignore
        # the one signal that matters (R10).
        return {
            "status": "unknown" if last_undecodable else "failed",
            "attempts": len(delays),
            "vast_fallback_used": False,
            "vast_destroyed": [],
            "error": last_err,
        }
    # Talk to Vast directly — it's the source of truth.
    print(f"[lab] sky.down exhausted for {cluster}; falling back to vast-sdk direct destroy")
    try:
        destroyed, failures = _vast_destroy_matching(cluster)
        events.note("teardown.fallback", cluster=cluster, via="vast", ok=not failures)
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
        events.note("teardown.fallback", cluster=cluster, via="vast", ok=False)
        return {
            # sky could not tell us and Vast could not either: genuinely unknown (R10).
            "status": "unknown" if last_undecodable else "failed",
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

    Returns ``True`` iff teardown **succeeded**. Three states are written, not two (R10):

    * ``"succeeded"`` -- the machine is confirmed gone.
    * ``"failed"`` -- the destroy was definitively refused. A real leak; ``lab wait`` exits 3.
    * ``"unknown"`` -- the reply was unreadable and nothing could verify it. ``lab wait`` exits
      6. This exists because on 2026-08-20 seven teardowns recorded ``failed`` while all seven
      machines had in fact been destroyed; an alarm that is usually wrong stops being an alarm.

    ``unknown`` is deliberately narrow. A provider-direct fallback that looks and finds nothing
    has *verified* the outcome, so that stays ``succeeded`` -- manufacturing doubt we do not have
    would be the same error as raising an alarm we cannot support. ``end_reason`` is annotated
    either way so the state is visible in ``lab status`` / ``lab dashboard`` / ``lab wait``.
    ``backoffs`` overrides the retry ladder for callers that must stay quick (e.g. a status
    poll); None keeps the default.
    """
    if backoffs is None:
        outcome = robust_teardown(sky_mod, cluster, cloud=cloud)
    else:
        outcome = robust_teardown(sky_mod, cluster, cloud=cloud, backoffs=backoffs)
    succeeded: bool = outcome["status"] == "succeeded"
    unknown: bool = outcome["status"] == "unknown"
    fields: dict[str, Any] = {"teardown_status": outcome["status"]}
    annotation: str | None = None
    if unknown:
        # Not an alarm and not an all-clear. Say plainly that the answer is unreadable and name
        # the one place that can settle it, because the operator's instinct after 2026-08-20 is
        # to disbelieve whichever way this reads.
        annotation = (
            f"TEARDOWN OUTCOME UNKNOWN for cluster {cluster!r}: {outcome['error']} after "
            f"{outcome['attempts']} sky.down attempts. The machine may already be gone OR may "
            f"still be billing — this cannot be told from the client. Verify against the "
            f"provider itself (`doctl compute droplet list`, `gcloud compute instances list "
            f"--filter=\"name~'^lab-'\"`, `vastai show_instances`), then `lab reconcile "
            f"--apply --yes` if anything remains."
        )
    elif not succeeded:
        if cloud == "vast":
            remedy = (
                "AND vast-sdk fallback. Run `lab reconcile --apply` to stop the bleed "
                "(add `--yes` when unattended: --apply asks first, and with no tty it "
                "refuses and destroys nothing). Or `vastai destroy_instance <id>`."
            )
        elif cloud == "gcp":
            remedy = (
                "AND gcp-direct fallback. Run `lab reconcile --apply` to stop the bleed "
                "(add `--yes` when unattended: with no tty --apply refuses and destroys "
                "nothing), and check "
                "`gcloud compute instances list --filter=\"name~'^lab-'\"`."
            )
        else:
            remedy = (
                f"(no provider-direct fallback for {cloud}). Run `lab reconcile --apply` "
                f"(add `--yes` when unattended: with no tty --apply refuses and destroys "
                f"nothing), and check `sky status` / the {cloud} console."
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
        # No `cloud` field: this function's signature carries no cloud (only its caller in
        # sky_runner.py does), and threading one through would be a real signature change rather
        # than a note added at an existing decision point. Deliberate omission, not an oversight
        # — the paired `provision.attempt` note (sky_runner.py) does carry it.
        events.note("provision.timeout", after_s=timeout_s)
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
    # Same reasoning as the disk invariant above, and the same reason it lives *here*: every
    # launch passes through `build_task`, including the scheduler's, which never calls
    # `resolve_backend_profile`. A resource stamped only on the CLI path is a resource that is
    # unattributable exactly when a deferred job leaks. See `project_labels` for which clouds
    # actually store these.
    labels = project_labels(manifest.job_id)

    def _res(*, use_spot: bool | None = None, region: str | None = None) -> sky.Resources:
        return sky.Resources(
            cloud=_cloud,
            labels=labels,
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




# ---------------------------------------------------------------------------
# Supervisor exit capture (F5) — *how* the detached supervisor died.
# ---------------------------------------------------------------------------
#
# The exit status of a process is a one-shot value: only its parent can collect it, and only
# until it is reaped. `submit` spawns the supervisor detached and does not wait on it — it must
# outlive the CLI — so after a bare `lab submit` the parent is gone in milliseconds, the
# supervisor reparents to init, and init reaps it the moment it dies. **In that case the exit
# status is unrecoverable, full stop.** No later poll from any process gets it back; `os.waitid`
# only works on your own children.
#
# What *is* recoverable, and what this module records:
#
#   waitpid      the spawning process is still alive — the MCP server, `lab sweep`,
#                `lab submit --wait`, the scheduler. Its Popen handle is kept here, so a poll
#                yields the exact status, signal included. This is the agent-facing path.
#   proc_zombie  a live parent elsewhere holds the corpse unreaped. `/proc/<pid>/stat` field 52
#                is the waitpid-form status and is readable by any same-uid process.
#   recycled     the PID is held by a different process now (its start-time moved, F4).
#   disappeared  nothing holds it. Exit status unrecoverable — recorded *as* unrecoverable.
#
# The last two carry no signal, and that is the point of writing them down anyway: "we looked and
# the kernel no longer knows" and "nobody ever looked" are different facts, and F5 is the
# complaint that they currently read identically.

# job_id -> (handle, store) for supervisors *this* process spawned. Holding the handle is what
# makes the status collectable at all; it also keeps CPython's opportunistic `subprocess._cleanup`
# from reaping (and discarding) the status behind our back the next time anything spawns.
_SUPERVISORS: dict[str, tuple[subprocess.Popen[bytes], JobStore]] = {}
_SUPERVISORS_LOCK = threading.Lock()

# Why a signal is worth naming, in the words an operator needs. SIGKILL is the one F5 was raised
# for: it cannot be caught, so the supervisor's own SIGTERM handler and `except BaseException`
# teardown never run, and nothing on disk explains the silence.
_SIGNAL_HINT = {
    "SIGKILL": "uncatchable, so most likely the OOM killer or kill -9",
    "SIGTERM": "a polite kill — lab cancel, a systemd stop, or a closing shell",
}


def _register_supervisor(job_id: str, proc: subprocess.Popen[bytes], store: JobStore) -> None:
    with _SUPERVISORS_LOCK:
        _SUPERVISORS[job_id] = (proc, store)


def _forget_supervisor(job_id: str) -> None:
    with _SUPERVISORS_LOCK:
        _SUPERVISORS.pop(job_id, None)


def _signal_name(num: int) -> str:
    try:
        return signal.Signals(num).name
    except ValueError:
        return f"signal {num}"


def _status_detail(returncode: int | None, signal_name: str | None) -> str:
    """One human sentence for a *known* exit status — this ends up in the manifest's
    ``end_reason``, which is what ``lab status`` prints."""
    if signal_name is not None:
        hint = _SIGNAL_HINT.get(signal_name)
        return f"killed by {signal_name}" + (f" — {hint}" if hint else "")
    return f"exited with code {returncode}"


def _exit_record(
    source: str, *, returncode: int | None = None, signal_name: str | None = None, detail: str
) -> dict[str, Any]:
    return {
        "source": source,
        "returncode": returncode,
        "signal": signal_name,
        "detail": detail,
        "observed_at": now().isoformat(),
    }


def _record_from_returncode(rc: int) -> dict[str, Any]:
    """``Popen.returncode`` form: negative means "killed by that signal"."""
    if rc < 0:
        name = _signal_name(-rc)
        return _exit_record("waitpid", signal_name=name, detail=_status_detail(None, name))
    return _exit_record("waitpid", returncode=rc, detail=_status_detail(rc, None))


def _proc_stat_tail(pid: int) -> list[str] | None:
    """``/proc/<pid>/stat`` from field 3 onwards, so ``tail[n - 3]`` is field ``n``.

    Split on the **last** ``)``: field 2 (``comm``) is parenthesised and may itself contain
    spaces and parens. Same parse as :func:`lab._util.process_start_time`, kept local because
    this needs three fields out of it rather than one.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rindex(")") + 1 :].split()
    except (OSError, ValueError):
        return None


def _stat_field(tail: list[str], number: int) -> str | None:
    idx = number - 3
    return tail[idx] if 0 <= idx < len(tail) else None


def _owned_by_us(pid: int) -> bool:
    """The kernel prints ``exit_code`` as ``0`` — indistinguishable from a clean exit — when the
    reader lacks ``PTRACE_MODE_READ`` on the task. Our own supervisors are always our own uid, so
    refusing to read anyone else's turns that silent lie into an honest "unknown"."""
    try:
        return os.stat(f"/proc/{pid}").st_uid == os.getuid()
    except OSError:
        return False


def _collect_from_proc(pid: int, start_time: int | None) -> dict[str, Any] | None:
    """Read the corpse out of ``/proc``, or say why it cannot be read. ``None`` = still alive."""
    tail = _proc_stat_tail(pid)
    if tail is None:
        # No `/proc` entry is not by itself a death: on any platform without procfs there is
        # never one, and reading that as "gone" would flip every healthy job to failed and tear
        # its machine down. Signal 0 is the portable question, and it is the one that decides.
        if pid_alive(pid, start_time=start_time):
            return None
        return _exit_record(
            "disappeared",
            detail=(
                f"exit status unrecoverable: pid {pid} was already reaped — only the process "
                "that spawned it could have collected it, and lab submit exits at once"
            ),
        )
    if start_time is not None and _stat_field(tail, 22) != str(start_time):
        return _exit_record(
            "recycled",
            detail=f"exit status unrecoverable: pid {pid} now belongs to a different process",
        )
    if _stat_field(tail, 3) != "Z":
        return None  # running, sleeping, stopped — all still alive
    raw = _stat_field(tail, 52)
    if raw is None or not _owned_by_us(pid):
        return _exit_record(
            "disappeared",
            detail=f"exit status unreadable: pid {pid} is an unreaped corpse we may not read",
        )
    try:
        status = int(raw)
    except ValueError:
        return _exit_record(
            "disappeared", detail=f"exit status unreadable: pid {pid} reported {raw!r}"
        )
    # waitpid form: low 7 bits are the terminating signal, the next byte up the exit code.
    sig = status & 0x7F
    if sig:
        name = _signal_name(sig)
        return _exit_record("proc_zombie", signal_name=name, detail=_status_detail(None, name))
    code = (status >> 8) & 0xFF
    return _exit_record("proc_zombie", returncode=code, detail=_status_detail(code, None))


def observe_supervisor_exit(store: JobStore, job_id: str) -> dict[str, Any] | None:
    """How this job's supervisor died, or ``None`` while it is still running.

    Written once into ``_runtime.json`` under ``runner_exit`` — durable by construction, since the
    process that observes the death is usually not the one that will report it. Never overwritten:
    the first observation is the most informative one (a later reader has only ``/proc``, and by
    then often not even that), so a re-observation must not degrade an exact answer to a guess.

    Also the *only* honest liveness answer for an unreaped corpse. A zombie answers
    ``os.kill(pid, 0)`` and keeps its start-time, so :func:`lab._util.pid_alive` calls it alive —
    for as long as its parent lives, which under a long-running MCP server is forever. That kept
    :meth:`SkyPilotBackend.status`'s dead-supervisor teardown from ever firing while the box bills.
    """
    rt = store.read_runtime(job_id)
    recorded = rt.get("runner_exit")
    if isinstance(recorded, dict):
        _forget_supervisor(job_id)
        return recorded
    pid = rt.get("runner_pid")
    if not pid:
        return None  # queued, or a runtime file older than this field: unknown stays unknown
    with _SUPERVISORS_LOCK:
        entry = _SUPERVISORS.get(job_id)
    if entry is not None:
        # Our own child: authoritative, and the poll reaps it so nothing accumulates.
        rc = entry[0].poll()
        rec = None if rc is None else _record_from_returncode(rc)
    else:
        rec = _collect_from_proc(int(pid), rt.get("runner_start_time"))
    if rec is None:
        return None
    store.write_runtime(job_id, runner_exit=rec)
    _forget_supervisor(job_id)
    events.note("supervisor.exit", job_id=job_id, pid=int(pid), **rec)
    return rec


def reap_supervisors() -> None:
    """Collect any finished supervisor this process spawned. Hygiene, called on each submit.

    Without it a long-lived MCP server holds one unreaped corpse per job it launched until
    somebody happens to poll that job's status — and the corpses are exactly what makes
    ``pid_alive`` lie about them. Never raises: this is bookkeeping on the launch path.
    """
    with _SUPERVISORS_LOCK:
        items = list(_SUPERVISORS.items())
    for job_id, (proc, store) in items:
        if proc.poll() is None:
            continue
        try:
            observe_supervisor_exit(store, job_id)
        except Exception as e:  # noqa: BLE001 — bookkeeping must never break a launch
            print(f"[lab] could not record supervisor exit for {job_id}: {e}", file=sys.stderr)
            _forget_supervisor(job_id)


def dead_supervisor_reason(exit_record: dict[str, Any] | None) -> str:
    """The manifest ``end_reason`` for a job whose supervisor vanished, naming the cause when one
    could be collected. ``lab status`` prints this; ``_runtime.json`` keeps the structured form."""
    base = "supervisor exited without recording status"
    detail = (exit_record or {}).get("detail")
    return f"{base} ({detail})" if detail else base


class SkyPilotBackend:
    name = "skypilot"

    def __init__(self, home: Path, repo: Path | None = None) -> None:
        self.store = JobStore(Path(home))
        self.repo = Path(repo) if repo else Path.cwd()

    def submit(self, manifest: JobManifest) -> str:
        # Before adding another child, collect the ones that have already finished (F5).
        reap_supervisors()
        job_dir = self.store.job_dir(manifest.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        # Supervisor's stdout/stderr (incl. sky.launch streamed logs) -> the job log file (FR-D1).
        logf = self.store.logs_path(manifest.job_id).open("w")
        # Group the supervisor's own ledger call into this submit's session, even when
        # LAB_SESSION_ID was never a real env var (the common case: this process generated one
        # in memory) — plain env inheritance would miss that. job_id stays the join key either
        # way (see run_job's events.begin call).
        child_env = {**os.environ, "LAB_SESSION_ID": events.session_id()}
        proc = subprocess.Popen(
            [sys.executable, "-m", "lab.sky_runner", str(job_dir)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(self.repo),
            start_new_session=True,
            env=child_env,
        )
        self.store.write_runtime(
            manifest.job_id,
            runner_pid=proc.pid,
            # Identity, not just the number: a recycled PID would otherwise report this
            # supervisor alive forever and disable every self-heal that depends on it (F4).
            runner_start_time=process_start_time(proc.pid),
            cluster=cluster_name_for(manifest.job_id),
        )
        # Keep the handle: it is the only thing that can ever collect this process's exit signal,
        # and it stays valid for exactly as long as *this* process lives (F5).
        _register_supervisor(manifest.job_id, proc, self.store)
        return manifest.job_id

    def status(self, job_id: str) -> JobState:
        m = self.store.read_manifest(job_id)
        if m.status not in _TERMINAL:
            rt = self.store.read_runtime(job_id)
            # Ask *how* it died before asking whether it did: an unreaped corpse passes
            # `pid_alive` (it answers signal 0 and keeps its start-time), so on the paths where
            # the spawner outlives the supervisor this is the only branch that ever notices.
            exit_record = observe_supervisor_exit(self.store, job_id)
            if rt.get("runner_pid") and (
                exit_record is not None
                or not pid_alive(rt["runner_pid"], start_time=rt.get("runner_start_time"))
            ):
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
                # Attribute the outcome honestly. A job whose cancel was interrupted is
                # `cancelled` -- someone asked for it; calling that a crash would misreport a
                # deliberate act and, worse, make a cancel indistinguishable from the supervisor
                # failure this branch exists to catch.
                cancelling = bool(rt.get("cancelling"))
                return self.store.update_manifest(
                    job_id,
                    status=JobState.cancelled if cancelling else JobState.failed,
                    ended_at=now(),
                    end_reason=(
                        "cancelled by user (teardown completed by recovery after the cancel "
                        "was interrupted)"
                        if cancelling
                        # A cancel sends the SIGTERM itself, so naming it here would dress a
                        # deliberate act up as a supervisor failure. Only the crash path reports
                        # the cause of death.
                        else dead_supervisor_reason(exit_record)
                    ),
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
        """Stop the job and release its machine, recording the terminal status **last**.

        Ordering is the whole point (R9, incident 2026-08-20). This used to write
        ``status=cancelled`` first and then do the slow part; seven cancels were killed by an
        impatient caller mid-``robust_teardown`` and each left a manifest reading ``cancelled``
        with ``teardown_status: None`` -- terminal, so ``lab wait`` was satisfied and every
        dashboard went quiet, while the machine may well have still been billing.

        Now the *intent* is recorded in ``_runtime.json`` up front and the manifest stays
        non-terminal until teardown has actually been attempted. An interrupted cancel therefore
        leaves a job that still looks unfinished -- which is true, and which lets
        :meth:`status` finish it (its supervisor is already gone, having been SIGTERMed below).
        """
        m = self.store.read_manifest(job_id)
        if m.status in _TERMINAL:
            return m.status
        # Durable before anything slow or killable: this is what tells a later `status()` that
        # the job was on its way out deliberately, so recovery reports `cancelled` rather than
        # inventing a crash that never happened.
        self.store.write_runtime(job_id, cancelling=True)
        rt = self.store.read_runtime(job_id)
        if rt.get("runner_pid"):
            try:
                os.kill(rt["runner_pid"], signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        # The name recorded at launch, not one recomputed today: `cluster_name_for` gained a
        # project slug, so recomputing can address a different machine than the one we started.
        cluster = rt.get("cluster") or cluster_name_for(job_id)
        import sky

        try:
            sky.get(sky.cancel(cluster, all=True))  # 0.12: RequestId
        except Exception:  # noqa: BLE001 - best-effort; teardown below is what matters
            pass
        tear_down_and_record(sky, cluster, self.store, job_id, m.resources.cloud or "vast")
        return self.store.update_manifest(
            job_id, status=JobState.cancelled, ended_at=now(), end_reason="cancelled by user"
        ).status

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
