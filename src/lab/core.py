"""Lab core — the single library both the CLI and the MCP server are thin shells over
(NFR-3, FR-F2). See research/10-architecture.md.

``Lab.submit`` resolves a :class:`~lab.models.JobSpec` into a :class:`~lab.models.JobManifest`
(pin commit, hash uv.lock, resolve seed), persists it via the store, then dispatches to the
chosen :class:`~lab.backends.base.Backend`.
"""

from __future__ import annotations

import subprocess
import sys
import hashlib
import itertools
import json
import math
import platform
import posixpath
import re
import shlex
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lab._util import (
    actual_cost,
    atomic_write_text,
    duration_seconds,
    infer_artifact_type,
    now,
    pid_alive,
    tail_last_line,
)
from lab.backends.base import Backend
from lab.backends.local import LocalBackend
from lab.manifest import (
    capture_diff,
    commit_exists,
    current_commit,
    git_work_tree,
    is_dirty,
    repo_root,
    resolve_commit,
    uv_lock_sha256,
)
from lab import events, placement, pricing
from lab import notes as lab_notes
from lab.metrics import final_values, group_series
from lab.storage import R2Store, r2_enabled
from lab.attribution import attribute_jobs, local_project
from lab.models import (
    ArtifactRecord,
    BackendInfo,
    CodeRef,
    EnvInfo,
    JobManifest,
    JobSpec,
    JobState,
    ResourceRequest,
    RunSpec,
    SweepCell,
    SweepPlan,
)
from lab.aggregate import merge_hits, merge_seed_rows
from lab.partials import PARTIAL_DIR
from lab.redact import redact
from lab.sharding import parse_seeds, partition_seeds, seeds_to_arg
from lab.store import JobStore, cell_id_for, injected_override_keys

if TYPE_CHECKING:  # import-time only: `lab.scheduler.queue` is reached lazily (module cycle)
    from lab.scheduler.queue import QueueStore

_TERMINAL_STATES = frozenset(
    {
        JobState.succeeded, JobState.failed, JobState.cancelled,
        JobState.timed_out, JobState.preempted,
    }
)

# Floor on the poll interval of a wait whose every job is read from the scheduler's mirror
# (:meth:`Lab._resolve_manifest`). The mirror is only as fresh as the scheduler's tick (60s by
# default), so polling it faster than this spends one object-store GET per job per poll to learn
# nothing. Local jobs keep the caller's interval: their manifests are a free filesystem read, and
# their poll is also what finalizes a dead supervisor (`backend.status`).
_MIRROR_MIN_INTERVAL_S = 30.0

# How long :meth:`Lab._settle_teardown` keeps re-reading a *mirrored* job whose teardown_status is
# still null. The scheduler writes the mirror in two steps: `tick.py::_sync` mirrors the terminal
# manifest, and the supervisor's later `teardown_status` only reaches the mirror on a following
# tick (the `RE_MIRROR_TERMINAL_S` grace window). So a laptop that observes terminal at T sees the
# teardown at the next tick — up to one full tick period later, plus however long that tick itself
# takes to run. The local settle's 15s cannot span that, which would have made *every* clean
# deferred wait print the "teardown not confirmed — run `lab reconcile`" warning: the money alarm,
# wrong almost every time, i.e. the R10 failure mode that gets an alarm ignored. 75s is one full
# 60s tick plus 25% for the tick's own runtime and clock skew — the smallest window that can
# contain a whole tick, and deliberately under two ticks so a clean deferred wait does not pay two
# minutes on the way out. Still null after that is a verdict, not lag: the warning is then earned.
_MIRROR_SETTLE_S = 75.0


class LabError(RuntimeError):
    """Fail-loud lab error (FR-F3)."""


class SweepSubmitError(LabError):
    """A sweep (or sweep-retry) refused partway, after some of its jobs had already launched.

    Those jobs are running — and, on a remote backend, billing — so the caller must learn which
    they are and which sweep they belong to; a bare error used to leave them orphaned from any
    record the caller held. ``details`` is the structured form a shell emits next to ``error``;
    the message names the same ids for surfaces that only carry text (MCP's ``ToolError``).
    """

    def __init__(self, message: str, *, sweep_id: str, job_ids: list[str]) -> None:
        super().__init__(message)
        self.sweep_id = sweep_id
        self.job_ids = list(job_ids)

    @property
    def details(self) -> dict[str, Any]:
        return {"sweep_id": self.sweep_id, "job_ids": self.job_ids}


class MirrorReadError(LabError):
    """The scheduler queue's mirror could not be read for a job.

    Distinct from :class:`FileNotFoundError`, which means "this job exists nowhere": this means
    "the mirror could not answer *this time*" — a transient object-store 5xx/reset, or a queue
    store that could not even be built. The distinction is load-bearing in :meth:`Lab.wait`, where
    the first is a caller error and the second is just no news yet.
    """


def _new_job_id() -> str:
    return f"{now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


# Draws before a submit gives up on finding an unclaimed id. A collision is ~1e-7 per draw, so
# reaching this means the run dir cannot be created at all (or `_new_job_id` is stuck).
_JOB_ID_ATTEMPTS = 16


def _gcp_identity(resource: dict[str, Any]) -> str:
    """A GCE resource's identity for attribution — from its labels when they are there.

    GCP truncates the cluster name to 35 characters on the cloud, which shears the job id off the
    end of a slugged name (``lab-laboratory-20260820-ef-…``). Attributing from the name alone
    therefore failed for *every* GCP resource, so each one landed in ``unattributed`` and could
    never be destroyed — the leak sweep for that cloud was inert. The ``lab-project``/``lab-job-id``
    labels are written at launch precisely because they survive truncation; prefer them and fall
    back to the name for resources that predate them.
    """
    labels = resource.get("labels") or {}
    project, job_id = labels.get("lab-project"), labels.get("lab-job-id")
    if project and job_id:
        return f"lab-{project}-{job_id}"
    return str(resource.get("name", ""))


def _lab_identity(identifier: str) -> tuple[str | None, list[str]]:
    """Read a provider's resource identifier as ``(stamped project slug, candidate job ids)``.

    Cloud identifiers are not bare cluster names. SkyPilot appends node suffixes
    (``lab-…-3dd12990-f5bf-head``), and the Vast pass matches against several concatenated
    fields at once, so neither :func:`parse_cluster_name` (anchored, full-match) nor a naive
    ``removeprefix("lab-")`` can be used on its own.

    Two signals come back, because they fail in different places:

    * the **stamped slug**, when some run of the identifier parses as a canonical cluster name.
      Authoritative and free — it travels on the resource itself, so it still answers after a
      job store or ledger is gone. Only present on names minted after project stamping.
    * **candidate job ids**, longest first, for :func:`lab.attribution.attribute_jobs`. Runs are
      taken from both offsets so ``lab-<slug>-<job_id>`` and legacy ``lab-<job_id>`` both yield
      the real id, and so ids that predate the canonical ``%Y%m%d-%H%M%S``-plus-hex shape (which
      the anchored regex cannot match) still resolve.
    """
    from lab.backends.skypilot import parse_cluster_name

    slug: str | None = None
    candidates: list[str] = []
    # `lab-…` is *embedded*, not anchored: SkyPilot's Vast adapter labels a rental
    # `sky-lab-<cluster>-<suffix>`, and the existing Vast pass matches by substring for exactly
    # that reason. Anchoring here would have made every Vast rental unattributable.
    for match in re.finditer(r"lab-[a-z0-9-]+", identifier.lower()):
        parts = match.group(0)[len("lab-") :].split("-")
        for end in range(len(parts), 0, -1):
            parsed = parse_cluster_name("lab-" + "-".join(parts[:end]))
            if parsed is not None:
                slug = slug or parsed.project
                if parsed.job_id not in candidates:
                    candidates.append(parsed.job_id)
                break
        for start in (0, 1):
            for end in range(len(parts), start, -1):
                cand = "-".join(parts[start:end])
                if cand and cand not in candidates:
                    candidates.append(cand)
    return slug, candidates


def orphan_key(field: str, item: Any) -> str:
    """Stable identity for one destroyable resource, as ``<pass>:<identifier>``.

    Shared by the CLI's confirmation preview and :meth:`Lab.reconcile`'s ``only`` filter so that
    an approval means exactly the set that was displayed. Each pass shapes its entries
    differently — Vast emits ``{id, label}``, DO volumes ``{id, name}``, the GCP passes
    ``{name, zone, …}``, and the sky pass bare cluster-name strings — so identity is resolved by
    preference rather than by a single key.
    """
    if isinstance(item, dict):
        # `unsupervised` entries are `{job_id, cluster}` — neither `name` nor `id`. Without
        # `job_id` here every entry keys as `unsupervised:None`, so approving one would approve
        # the lot.
        ident = item.get("name") or item.get("id") or item.get("label") or item.get("job_id")
        return f"{field}:{ident}"
    return f"{field}:{item}"


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid -> one config dict per point (FR-A5)."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _normalize_config(value: Any) -> Any:
    """Canonicalise config for hashing: stringify leaf values (preserving structure) so the same
    logical job hashes equal regardless of how its values were typed — the CLI keeps grid values as
    strings while the API/MCP pass ints/floats, and the experiment coerces types anyway."""
    if isinstance(value, dict):
        return {k: _normalize_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config(v) for v in value]
    return str(value)


def cache_key(commit: str, command: str, config: dict[str, Any] | None, seed: int) -> str:
    """Stable hash identifying an 'identical job' for result caching (FR-B5).

    The spec keys on commit+config+seed; we also include the command (entrypoint), since the lab
    runs arbitrary commands and two different experiments at the same commit/config/seed are not
    the same job. Config leaves are normalised (stringified) so a value isn't type-sensitive.
    """
    payload = json.dumps(
        {"commit": commit, "command": command, "config": _normalize_config(config or {}), "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def compare_final_metrics(
    orig: dict[str, float],
    new: dict[str, float],
    *,
    names: Iterable[str] | None,
    rtol: float = 1e-3,
    atol: float = 1e-12,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Judge a re-run's final metrics against the original's snapshot (the reproducibility gate).

    Returns ``("match" | "drift", deltas)``. ``names`` restricts which baseline metrics are judged
    (``None`` = all). A baseline metric absent from the re-run can't be re-derived → counts as drift.
    Tolerance is ``math.isclose`` semantics (relative ``rtol`` + absolute ``atol``). The small default
    ``atol`` is a float noise floor so a metric of exactly 0.0 doesn't false-drift against a tiny
    re-run value — ``math.isclose``'s relative tolerance collapses to zero at zero.
    """
    selected = list(names) if names is not None else list(orig)
    deltas: dict[str, dict[str, Any]] = {}
    verdict = "match"
    for name in selected:
        ov = orig.get(name)
        nv = new.get(name)
        if ov is None or nv is None:
            within = False
        else:
            within = math.isclose(ov, nv, rel_tol=rtol, abs_tol=atol)
        deltas[name] = {
            "orig": ov,
            "new": nv,
            "abs_delta": (abs(nv - ov) if ov is not None and nv is not None else None),
            "within_tol": within,
        }
        if not within:
            verdict = "drift"
    return verdict, deltas


def worst_case_sweep_cost(*, n_points: int, per_point_cap: float) -> float:
    return round(n_points * per_point_cap, 6)


def check_sweep_admission(
    *,
    n_points: int,
    per_point_cap: float | None,
    daily_budget: float | None,
    committed: float,
) -> float | None:
    """Refuse a sweep whose worst case won't fit the daily budget. Returns the worst-case cost
    (the default ceiling), or None when uncosted. Pure; no state (cost-safety, derived not metered)."""
    if per_point_cap is None:
        return None
    worst = worst_case_sweep_cost(n_points=n_points, per_point_cap=per_point_cap)
    if daily_budget is not None and committed + worst > daily_budget:
        raise LabError(
            f"sweep worst case ${worst:.2f} ({n_points} x ${per_point_cap:.2f}) + "
            f"committed ${committed:.2f} exceeds daily budget ${daily_budget:.2f}; "
            "narrow the grid, lower --max-cost, or raise the budget"
        )
    return worst


def _mask_command_secrets(command: str) -> str:
    """``command`` with inline secrets masked, for copies that leave the store (``lab export``).

    The manifest records the command verbatim, so ``HF_TOKEN=hf_… python x.py`` or
    ``--api-key …`` would land in a bundle that exists to be committed (FR-J1). Every judgement
    is the event ledger's own (:mod:`lab.events.sanitize`), applied per shell token:

    * ``NAME=value`` in env-variable form (upper-case identifier, i.e. an env prefix): masked if
      the name is secret-shaped (``*TOKEN``, ``*_KEY``, …) or the value is.
    * any other ``k=v`` (a config override): masked only if the value itself is secret-shaped —
      a hyperparameter named ``max_tokens`` is not a credential.
    * ``--flag=value`` / ``--flag value``: masked as the ledger masks argv.

    A command with nothing to mask comes back byte-identical (no re-quoting); one that cannot be
    tokenized falls back to the ledger's free-text masking.
    """
    from lab.events.sanitize import mask_text, sanitize_argv, sanitize_params

    try:
        tokens = shlex.split(command)
    except ValueError:
        return mask_text(command)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        name, eq, value = tok.partition("=")
        if tok.startswith("-"):
            if eq or i + 1 == len(tokens) or tokens[i + 1].startswith("-"):
                out.append(sanitize_argv([tok])[0])
            else:  # `--flag value`: judge the value together with the flag that names it
                out.extend([tok, str(sanitize_argv([tok, tokens[i + 1]])[1])])
                i += 1
        elif eq and name.isidentifier():
            judged_as = name if name.isupper() else "value"
            out.append(f"{name}={sanitize_params({judged_as: value})[judged_as]}")
        else:
            out.append(tok)
        i += 1
    return command if out == tokens else redact(shlex.join(out))


def _parse_row_key(row_key: str | list[str], seed_column: str) -> list[str]:
    """Parse/validate a row-key spec ('seed,alpha' or a list). Must include the seed column —
    presence/retry accounting is keyed on seeds. Pure; raises :class:`LabError`."""
    cols = (
        [c.strip() for c in row_key.split(",") if c.strip()]
        if isinstance(row_key, str)
        else list(row_key)
    )
    if seed_column not in cols:
        raise LabError(
            f"--row-key {cols} must include the seed column {seed_column!r} "
            "(presence/retry accounting is keyed on seeds)"
        )
    return cols


def _mirror_sweep_aggregate(path: Path, key: str) -> None:
    """Best-effort R2 copy of a sweep's per-cell aggregate; the local file stays authoritative.

    One behaviour for both cell files (result table and search hits): a failed upload is
    printed and noted, never raised, because the aggregate can always be recomputed from the
    shards while the run that produced them cannot."""
    if not r2_enabled():
        return
    r2 = R2Store.from_env()
    if r2 is None:
        return
    try:
        r2.upload_file(path, key)
    except Exception as e:  # noqa: BLE001 — local aggregate stays authoritative
        print(f"[lab] aggregate R2 mirror failed, keeping local copy: {e}", file=sys.stderr)
        events.note("storage.upload_failed", key=key, error=str(e))


# Bump when a cell's aggregate outputs would change for the same inputs (a merge or format
# change), so fingerprints recorded by an older lab stop matching and the cell is re-merged.
_AGGREGATE_FORMAT = 2  # 2: hits for seeds missing from results.csv are dropped


# (job_id, terminal state, results file, hits file) of one shard an aggregate includes.
_IncludedShard = tuple[str, str, Path | None, Path | None]


def _cell_fingerprint(cell: SweepCell, included: list[_IncludedShard]) -> str:
    """What a cell's aggregate is a function of: the shards it includes (id, terminal state, and
    each input file's size + mtime), plus every plan field that shapes the output.

    Re-merging one cell of 866k hits cost 7.8 s and 1.4 GB on *every* ``sweep-aggregate`` — and
    ``sweep-retry`` runs two (stress run 2026-11-19) — while a finished sweep's inputs never
    change. A stat is cheap and a terminal shard's files only change if something rewrites them,
    which changes their size or mtime. A file that cannot be stat'ed counts as absent.
    """

    def stat(p: Path | None) -> list[int] | None:
        if p is None:
            return None
        try:
            st = p.stat()
        except OSError:
            return None
        return [st.st_size, st.st_mtime_ns]

    payload = [
        _AGGREGATE_FORMAT, cell.aggregate_ref, cell.results_file, cell.seed_column, cell.row_key,
        [[jid, state, stat(rf), stat(hf)] for jid, state, rf, hf in included],
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _cell_outputs_intact(cell: SweepCell, hits_dest: Path) -> bool:
    """Do the cell files still match what the plan says the last aggregate wrote? A deleted
    output (or a hits file reappearing where none should be) forces a re-merge."""
    if cell.seeds_present and not Path(cell.aggregate_ref).exists():
        return False
    if cell.hits_ref is not None:
        return Path(cell.hits_ref).exists()
    return not hits_dest.exists()


def _bundle_relative_plan(plan: SweepPlan) -> SweepPlan:
    """A copy of ``plan`` whose cell outputs point where ``Lab.export`` puts them in the bundle
    (``cells/<cell_id>/<file>``, relative to the bundle root, where plan.json sits)."""
    cells = [
        cell.model_copy(
            update={
                "aggregate_ref": f"cells/{cell.cell_id}/{Path(cell.aggregate_ref).name}",
                "hits_ref": (
                    None if cell.hits_ref is None
                    else f"cells/{cell.cell_id}/{Path(cell.hits_ref).name}"
                ),
            }
        )
        for cell in plan.cells
    ]
    return plan.model_copy(update={"cells": cells})


def _submit_stagger_s() -> float:
    """Delay between remote sweep submits (``LAB_SUBMIT_STAGGER_S``, default 1.5s; 0 disables).

    Each skypilot submit spawns a supervisor that immediately hits the local SkyPilot API
    server; N back-to-back spawns can refuse connections while the server cold-starts
    (field-report #4). A short stagger lets the first submit's autostart win. Lives in the sweep
    loops only — single submits and the (already 60s-paced) scheduler don't pay it."""
    import os

    try:
        return max(0.0, float(os.environ.get("LAB_SUBMIT_STAGGER_S", "1.5")))
    except ValueError:
        return 1.5


def _partial_sweep_error(
    sweep_id: str,
    job_ids: list[str],
    planned: int | None,
    cause: BaseException,
    *,
    next_step: str,
) -> SweepSubmitError:
    """The error for a sweep that stopped after launching ``job_ids`` (see SweepSubmitError)."""
    of = f" of {planned}" if planned is not None else ""
    return SweepSubmitError(
        f"sweep {sweep_id} stopped after launching {len(job_ids)}{of} jobs: {cause}. "
        f"Already launched (running, and billing if remote — `lab cancel <id>` to stop): "
        f"{', '.join(job_ids)}. {next_step}.",
        sweep_id=sweep_id,
        job_ids=job_ids,
    )


def build_sweep_point_spec(
    command: str,
    point: dict[str, Any],
    *,
    seed: int | None,
    resources: ResourceRequest | None = None,
    code_ref: str = "HEAD",
    submitted_by: str = "agent",
    allow_unknown_config: bool = False,
) -> JobSpec:
    """One grid point -> a JobSpec, identical for immediate (``Lab.sweep``) and deferred
    (``register_sweep``) paths so they can't drift.

    Point params are appended to the command as **shell-quoted** ``key=value`` overrides
    (injection-safe) and recorded in ``config``. A ``seed`` key in the point sets the per-point
    seed (must be int); otherwise the sweep-level ``seed`` default applies.
    """
    overrides = " ".join(shlex.quote(f"{k}={v}") for k, v in point.items())
    full_command = f"{command} {overrides}".strip()
    point_seed = point.get("seed")
    if point_seed is not None:
        try:
            job_seed: int | None = int(point_seed)
        except (TypeError, ValueError) as e:
            raise LabError(f"grid 'seed' values must be integers, got {point_seed!r}") from e
    else:
        job_seed = seed
    return JobSpec(
        code_ref=code_ref,
        command=full_command,
        seed=job_seed,
        config=point,
        resources=resources or ResourceRequest(),
        submitted_by=submitted_by,  # type: ignore[arg-type]
        allow_unknown_config=allow_unknown_config,
    )


SUPPORTED_CLOUDS: tuple[str, ...] = ("vast", "do", "gcp")

# How long a non-terminal skypilot job may lack a live supervisor pid before `reconcile` stops
# treating its cluster as protected (covers the submit->runtime-write race on fresh jobs).
UNSUPERVISED_GRACE_S = 300.0

CPU_DEFAULT_CLOUD = "do"
CPU_DEFAULT_VCPUS = 4
# The disk defaults and the "which clouds bill storage separately" rule live in lab.placement,
# because build_task needs them too: the scheduler launches registrations without ever calling
# resolve_backend_profile, so the invariant has to be enforced closer to the launch than this.
# Re-exported here since this is where callers already look for the profile constants.
CPU_DEFAULT_DISK_GB = placement.CPU_DEFAULT_DISK_GB
GPU_DEFAULT_DISK_GB = placement.GPU_DEFAULT_DISK_GB
STORAGE_BILLING_CLOUDS = placement.STORAGE_BILLING_CLOUDS
default_disk_gb = placement.effective_disk_gb


# Every name `build_backend` maps; `cpu` is resolve_backend_profile's sugar over skypilot. Exact,
# case-sensitive names: `Local` is a typo, not a spelling of `local`.
SUPPORTED_BACKENDS: tuple[str, ...] = ("local", "skypilot", "cpu")


def validate_backend(name: str) -> str:
    """Reject a backend name outside :data:`SUPPORTED_BACKENDS`. Pure.

    ``build_backend`` used to map any name it did not know to the local backend, so
    ``--backend skypilto``, ``--backend Local`` and MCP ``backend="nonsense"`` all exited 0 and
    ran the job on this machine — a workload sized for a remote box, with none of its remote
    timeout or teardown, reported as a success (stress run 2026-11-19). Cloud names have been
    validated all along (:func:`validate_cloud`); this is the same guard for backends.
    """
    if name not in SUPPORTED_BACKENDS:
        raise LabError(f"unknown backend {name!r}; supported: {', '.join(SUPPORTED_BACKENDS)}")
    return name


def validate_cloud(cloud: str | None) -> str | None:
    """Reject cloud names outside :data:`SUPPORTED_CLOUDS` (None = the Vast default). Pure."""
    if cloud is not None and cloud not in SUPPORTED_CLOUDS:
        raise LabError(
            f"unknown cloud {cloud!r}; supported: {', '.join(SUPPORTED_CLOUDS)} (default: vast)"
        )
    return cloud


def validate_durations(resources: ResourceRequest) -> None:
    """Refuse an unparsable or non-positive ``timeout``/``provision_timeout`` when a job is asked
    for, not when it launches. Pure.

    Nothing parsed them before launch, so ``3h30`` (documented as refused) was accepted and a
    negative timeout paid for a provision that died on arrival. Called on every create path
    rather than as a model validator, so a legacy manifest carrying such a value still loads.
    """
    from lab._util import parse_duration

    for field in ("timeout", "provision_timeout"):
        try:
            parse_duration(getattr(resources, field))
        except ValueError as e:
            raise LabError(f"{field}: {e}") from e


def resolve_backend_profile(
    backend: str, resources: ResourceRequest
) -> tuple[str, ResourceRequest]:
    """Resolve the ``cpu`` convenience backend into (provisioner_name, resources).

    ``cpu`` is sugar for the SkyPilot provisioner on a cheap CPU cloud (DigitalOcean by default,
    overridable via ``resources.cloud``): it clears accelerators and defaults to
    ``CPU_DEFAULT_VCPUS`` vCPUs. Spot is forced off only on DO (which has none) — GCP CPU jobs
    may use preemptible.

    Every skypilot spec — cpu profile or not — also leaves here with an explicit ``disk_size`` on
    the storage-billing clouds, so no path can inherit SkyPilot's 256 GB default. That default was
    only ever noticed on DO, where it 422s loudly; on GCP it provisions fine and quietly doubles a
    spot job's bill. Pure; no I/O. An unknown ``backend`` is refused (:func:`validate_backend`).
    """
    validate_backend(backend)
    validate_cloud(resources.cloud)
    validate_durations(resources)
    if backend != "cpu":
        disk = default_disk_gb(resources)
        if backend == "skypilot" and disk != resources.disk_size:
            return backend, resources.model_copy(update={"disk_size": disk})
        return backend, resources
    if resources.accelerators:
        raise LabError("--backend cpu provisions a CPU-only box; drop --accelerators")
    if resources.accelerator_pool:
        raise LabError("--backend cpu provisions a CPU-only box; drop --accelerator-pool")
    cloud = resources.cloud or CPU_DEFAULT_CLOUD
    update: dict[str, Any] = {
        "cloud": cloud,
        "cpus": resources.cpus or CPU_DEFAULT_VCPUS,
        "disk_size": resources.disk_size or CPU_DEFAULT_DISK_GB,
    }
    if cloud == "do":
        update["use_spot"] = False
        update["spot_fallback"] = False
    return "skypilot", resources.model_copy(update=update)


def _vast_price_feed() -> Any | None:
    """The scheduler's live Vast offer feed, or ``None`` if it cannot be constructed.

    A module-level function rather than an inline import so tests have a seam.

    Note that a missing vastai-sdk does **not** come back here as ``None``:
    ``lab.scheduler.price`` imports nothing from the SDK at module scope (``_get_vast_client`` is
    function-local), so ``VastPriceFeed()`` always constructs and the missing extra surfaces from
    ``best_hourly`` instead, where the caller catches it. Both routes fail open; this one is kept
    for a genuinely unimportable module rather than for the SDK.
    """
    try:
        from lab.scheduler.price import VastPriceFeed
    except ImportError:
        return None
    return VastPriceFeed()


def _check_price_cap_admission(
    cloud: str | None, accelerators: str | None, price_cap: float | None
) -> None:
    """Refuse a cap the live offer feed says nothing can meet. Raises :class:`LabError`, else None.

    Vast-only. On DO/GCP SkyPilot's catalog is accurate *for the region actually launched into*,
    so the optimizer's own enforcement is sound there and a Vast-shaped gate would be wrong.

    Never blocks on doubt. The feed prices the **cheapest matching** offer, so "cheapest > cap" is
    a definitive negative — nothing on offer can satisfy this cap, and renting anything would be
    renting something we already know is too expensive. A feed that errors, is absent, or matches
    no offer says nothing at all and the launch proceeds; that is ``lab doctor``'s rule ("only
    definitive negatives block") applied to prices.

    This cannot promise the cap will *hold* — the optimizer need not land on the cheapest offer —
    which is why ``resolve_cost`` still checks the billed rate afterwards. The two are complements,
    not duplicates.
    """
    # `res.cloud` is None for the default cloud, and the default IS Vast — spelled `cloud or
    # "vast"` at every other site (skypilot.py:1522/:1561/:1895/:1967, sky_runner.py:1134).
    # Comparing the raw value skipped a bare `lab submit --price-cap ...`, i.e. the exact shape of
    # the incident this gate exists for.
    if price_cap is None or (cloud or "vast") != "vast":
        return
    feed = _vast_price_feed()
    if feed is None:
        return
    try:
        best = feed.best_hourly(accelerators)
    except Exception as e:  # noqa: BLE001 — a feed that cannot answer must never block a launch
        # Common and benign: vastai-sdk is an optional extra, so this fires on every capped
        # submit on a machine without it. Say what it means for the cap, not just that it failed.
        print(
            f"[lab] no live Vast price feed ({type(e).__name__}), so --price-cap is not "
            f"pre-checked; SkyPilot's catalog estimate still applies and the billed rate is "
            f"still checked after boot",
            file=sys.stderr,
        )
        return
    if best is not None and best > price_cap:
        from lab.pricing import cap_admission_error

        raise LabError(cap_admission_error(best, price_cap, accelerators))


class Lab:
    def __init__(self, backend: Backend, repo: Path, home: Path) -> None:
        self.backend = backend
        self.repo = Path(repo)
        self.home = Path(home)
        self.store = JobStore(self.home)
        # Lazily built, then reused for this Lab's lifetime — see `_queue` / `_backend_for`.
        self._queue_store: QueueStore | None = None
        self._backends: dict[str, Backend] = {}
        # Manifests the most recent `list_jobs` could not read: [{job_id, error}] (see there).
        self.unreadable_jobs: list[dict[str, str]] = []
        self._warned_unreadable: set[str] = set()

    def preflight(self, spec: JobSpec) -> None:
        """Refuse a remote launch that a cheap local check proves cannot work (FR-F3).

        Only definitive negatives block — a quota of zero, a disabled API, an absent permission.
        A check that merely *fails to answer* is skipped, because a preflight that blocked on its
        own breakage would be worse than none. See :mod:`lab.doctor`.
        """
        if self.backend.name == "local":
            return
        from lab.doctor import format_report, preflight

        blocking = preflight(spec.resources.cloud, spec.resources, home=self.home)
        if not blocking:
            return
        raise LabError(
            "preflight refused this launch — it would fail after provisioning:\n"
            + format_report(blocking)
            + "\n(pass --no-preflight / preflight=False to launch anyway)"
        )

    def submit(
        self,
        spec: JobSpec,
        *,
        allow_dirty: bool = True,
        sweep_id: str | None = None,
        cell_id: str | None = None,
        code: CodeRef | None = None,
        registration_id: str | None = None,
        confirms: str | None = None,
        preflight: bool = True,
    ) -> str:
        """Build + persist the manifest, then launch via the backend (FR-A1, FR-B).

        ``code`` overrides git introspection — used by the scheduler, which submits from an
        extracted bundle (not a git repo) with provenance captured at registration time.

        ``spec.code_ref`` other than the current commit runs *that* commit's committed tree: it
        is snapshotted (``_code_ref_tree``) and launched from there with ``code`` pinned to it —
        the mechanism ``confirm`` uses. It used to be accepted and silently ignored, so the
        working tree ran while the caller believed another commit had.
        """
        validate_durations(spec.resources)
        if code is None:
            pinned = self._code_ref_commit(spec.code_ref)
            if pinned is not None:
                return self._sibling_lab(self._code_ref_tree(pinned)).submit(
                    spec,
                    allow_dirty=allow_dirty,
                    sweep_id=sweep_id,
                    cell_id=cell_id,
                    code=CodeRef(git_commit=pinned, git_dirty=False),
                    registration_id=registration_id,
                    confirms=confirms,
                    preflight=preflight,
                )
        if preflight:
            self.preflight(spec)
        # Before a job id exists, let alone a manifest or a rental: a cap nothing on offer can
        # meet is knowable for free, and refusing here costs $0 instead of a provisioned box.
        _check_price_cap_admission(
            spec.resources.cloud, spec.resources.accelerators, spec.resources.max_hourly_usd
        )
        dirty = False
        if code is None:
            try:
                dirty = is_dirty(self.repo)
            except subprocess.CalledProcessError as e:
                # git exits 128 outside a work tree. Provenance is fail-closed (FR-B1) — without a
                # repo there is no commit to pin, so refusing is right — but this surfaced as a
                # raw traceback. Reachable now that an installed lab is pointed at whatever
                # directory you stand in, which need not be a repo yet (FR-F3).
                if git_work_tree(self.repo) is None:
                    raise LabError(
                        f"{self.repo} is not a git repository — the lab pins the commit a run was "
                        "produced from, so it cannot submit without one. Run `git init` and "
                        "commit, or point the lab at your project with LAB_REPO_DIR."
                    ) from e
                raise
            if dirty and not allow_dirty:
                raise LabError("working tree is dirty; commit or pass allow_dirty=True (FR-B1)")
        elif code.git_dirty and not allow_dirty:
            raise LabError("bundle captured a dirty tree but allow_dirty=False (FR-B1)")
        seed = spec.seed if spec.seed is not None else 0  # explicit + recorded (FR-B4)
        lock = self.repo / "uv.lock"
        try:
            lock_sha = uv_lock_sha256(lock)
        except FileNotFoundError as e:
            # The lab pins the environment by hashing the lockfile (FR-B2), so there is nothing
            # to record without one. This became a reachable state when the lab stopped being the
            # repo you run from: an installed lab is pointed at whatever project you stand in,
            # and that project may not be uv-managed yet. Say so, rather than letting the hash
            # helper's FileNotFoundError reach the user as a traceback (FR-F3).
            raise LabError(
                f"no uv.lock at {lock} — the lab records its sha256 so the remote environment "
                "provably matches yours (FR-B2). Run `uv lock` in this project first "
                "(`uv init` first if it is not uv-managed yet)."
            ) from e
        # Every refusal that needs no job id has run: only now is an id claimed, so a refused
        # submit leaves no run dir behind.
        job_id = self._claim_job_id()
        try:
            if code is None:
                code = self._snapshot_code(job_id, dirty=dirty)
            manifest = JobManifest(
                job_id=job_id,
                sweep_id=sweep_id,
                cell_id=cell_id,
                registration_id=registration_id,
                confirms=confirms,
                created_at=now(),
                submitted_by=spec.submitted_by,
                code=code,
                env=EnvInfo(
                    uv_lock_sha256=lock_sha,
                    python_version=platform.python_version(),
                ),
                run=RunSpec(
                    entrypoint_command=spec.command,
                    resolved_config=spec.config or {},
                    seed=seed,
                    allow_unknown_config=spec.allow_unknown_config,
                    # What the unconsumed-config audit may judge: config keys that ride the argv.
                    config_overrides=injected_override_keys(spec.command, spec.config),
                ),
                resources=spec.resources,
                backend=BackendInfo(provisioner=self.backend.name),
                status=JobState.queued,
            )
            self.store.create(manifest)
        except BaseException:
            self.store.release_dir(job_id)  # ours, and holding no manifest: nothing ran
            raise
        self.backend.submit(manifest)
        return job_id

    def _claim_job_id(self) -> str:
        """A fresh job id whose run dir this call created — exclusively.

        Ids are a one-second timestamp plus a 24-bit random suffix, so two submits in the same
        second can draw the same one (~0.2% somewhere in a 1000-shard sweep), and the run dir used
        to be made with ``exist_ok=True``: the second job silently took over the first one's dir
        and its manifest replaced the first (stress run 2026-11-19). The id is widened nowhere —
        it is stamped into cluster names that GCP truncates — so a collision draws again instead.
        """
        return self._claim_id(lambda: _new_job_id(), "job")

    def _claim_sweep_id(self) -> str:
        """A sharded sweep's id, claimed like a job id (:meth:`_claim_job_id`): its dir holds
        plan.json, and a second sweep drawing the same id would have replaced the first's plan."""
        return self._claim_id(lambda: f"sweep-{_new_job_id()}", "sweep")

    def _claim_id(self, draw: Callable[[], str], kind: str) -> str:
        for _ in range(_JOB_ID_ATTEMPTS):
            new_id = draw()
            if self.store.claim_dir(new_id):
                return new_id
            events.note("core.id_collision", id_kind=kind, id=new_id)
        raise LabError(
            f"could not claim a fresh {kind} id in {self.home} after {_JOB_ID_ATTEMPTS} attempts"
        )

    def _snapshot_code(self, job_id: str, *, dirty: bool) -> CodeRef:
        """The submit's :class:`CodeRef`: the current commit, plus — for a dirty tree — the diff
        captured into the (already claimed) job dir and mirrored to R2 when enabled (FR-B1)."""
        diff_ref: str | None = None
        if dirty:
            # Capture into the job dir, then mirror to R2 (if enabled) for durability — the
            # local runs/ dir is git-ignored and may be lost. diff_ref points at the durable
            # copy when one exists, else the local path.
            blob = capture_diff(self.repo, self.store.job_dir(job_id))
            if blob is None:
                # is_dirty said dirty but capture found nothing — the tree changed under us
                # (e.g. a concurrent stash/checkout). Fail loud rather than write a Gap-B
                # manifest (which would surface as a raw ValueError at create) (FR-B1).
                raise LabError(
                    "working tree changed during submit (no diff to capture); retry (FR-B1)"
                )
            diff_ref = blob
            try:
                diff_bytes: int | None = Path(blob).stat().st_size
            except OSError:
                diff_bytes = None
            events.note("core.dirty_snapshot", diff_ref=diff_ref, bytes=diff_bytes)
            if r2_enabled():
                r2 = R2Store.from_env()
                if r2 is not None:
                    rel = f"{job_id}/code_diff.tar.gz"
                    try:
                        r2.upload_file(Path(blob), rel)
                        diff_ref = r2.uri(rel)
                    except Exception as e:  # noqa: BLE001 — local diff_ref stays fail-closed
                        print(
                            f"[lab] diff R2 upload failed, keeping local copy: {e}",
                            file=sys.stderr,
                        )
                        events.note("storage.upload_failed", key=rel, error=str(e))
        return CodeRef(git_commit=current_commit(self.repo), git_dirty=dirty, diff_ref=diff_ref)

    def _code_ref_commit(self, code_ref: str) -> str | None:
        """The commit a non-HEAD ``code_ref`` pins, or ``None`` when the ordinary working-tree
        path applies (``HEAD``, or a ref that names the current commit). Raises
        :class:`LabError` for a ref that names no commit — never falls back to running HEAD."""
        if code_ref.strip() in ("", "HEAD"):
            return None
        pinned = resolve_commit(self.repo, code_ref.strip())
        if pinned is None:
            raise LabError(
                f"code_ref {code_ref!r} is not a commit in {self.repo} — pass a commit SHA, "
                "branch or tag that exists there (e.g. after `git fetch`), or omit it for HEAD"
            )
        return None if pinned == current_commit(self.repo) else pinned

    def _code_ref_tree(self, commit: str) -> Path:
        """The committed tree of ``commit``, extracted once under ``<home>/_code_ref/<sha>``.

        Built with the same bundle mechanism ``confirm`` uses (committed tree only, never the
        working tree's dirt), into a temp dir then renamed into place, so a tree a running job
        may be executing from is never rewritten under it and concurrent submits of one commit
        (a sweep's points) share a single snapshot.
        """
        from lab.scheduler.bundle import create_bundle, extract_bundle  # avoid import cycle

        root = self.home / "_code_ref"
        dest = root / commit
        if dest.is_dir():
            return dest
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix=".build-") as td:
            tar, _ = create_bundle(self.repo, Path(td), commit=commit, include_dirty=False)
            tree = extract_bundle(tar, Path(td) / "tree")
            try:
                tree.rename(dest)
            except OSError:
                if not dest.is_dir():  # not a lost race with another submit: a real failure
                    raise
        return dest

    def find_cached(self, spec: JobSpec, *, require_clean: bool = True) -> str | None:
        """Return a prior SUCCEEDED job with the same commit+command+config+seed, else None (FR-B5).

        With ``require_clean`` (default), a dirty working tree disables caching and only clean-tree
        jobs are eligible — a dirty commit doesn't fully capture the code, so reusing its result
        isn't safe. A ``code_ref`` pinning another commit is keyed on that commit (whose committed
        tree is what would run, so the working tree's state is irrelevant).
        """
        pinned = self._code_ref_commit(spec.code_ref)
        if pinned is None and require_clean and is_dirty(self.repo):
            return None
        seed = spec.seed if spec.seed is not None else 0
        commit = pinned if pinned is not None else current_commit(self.repo)
        key = cache_key(commit, spec.command, spec.config, seed)
        for m in self.list_jobs():
            if m.status is not JobState.succeeded or (require_clean and m.code.git_dirty):
                continue
            if (
                cache_key(
                    m.code.git_commit, m.run.entrypoint_command, m.run.resolved_config, m.run.seed
                )
                == key
            ):
                events.note("core.cache_hit", job_id=m.job_id)
                return m.job_id
        return None

    def sweep(
        self,
        command: str,
        grid: dict[str, list[Any]],
        *,
        resources: ResourceRequest | None = None,
        seed: int | None = None,
        code_ref: str = "HEAD",
        submitted_by: str = "agent",
        allow_dirty: bool = True,
        max_jobs: int = 256,
        sweep_max_cost: float | None = None,
        daily_budget: float | None = None,
        committed: float = 0.0,
        seeds: str | list[int] | None = None,
        shard_size: int | None = None,
        results_file: str = "results.csv",
        seed_column: str = "seed",
        seed_axis_key: str = "seeds",
        allow_unknown_config: bool = False,
        row_key: str | list[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Submit one job per grid point under a shared sweep_id (FR-A5).

        Each point's params are appended to the command as **shell-quoted** ``key=value`` overrides
        (injection-safe) and recorded in the job's ``resolved_config``; jobs stay independently
        monitorable by ``job_id``. A ``seed`` key in the grid sets each job's seed (varying
        ``$LAB_SEED`` per point). Refuses to fan out beyond ``max_jobs`` (cost-safety).

        ``sweep_max_cost`` caps total sweep spend; ``daily_budget`` + ``committed`` enforce an
        up-front admission check (cost-safety, derived not metered). All default to no-op.

        With ``seeds`` + ``shard_size`` (P1-2) each cell's seed set is partitioned into shards of at
        most ``shard_size`` seeds; each shard runs as its own job (own timeout + teardown) with its
        seed subset appended under ``seed_axis_key`` (e.g. ``seeds=0,1``). A ``SweepPlan`` is
        persisted for aggregation/retry. ``seeds`` absent ⇒ today's behavior, no plan written.
        """
        if not grid and seeds is None:
            # One guard for every shell (thin-shell rule): a "sweep" with no axis would silently
            # run a single empty point instead of the fan-out the caller asked for.
            raise LabError("pass --grid and/or --seeds (a sweep needs at least one axis)")
        row_key_cols = _parse_row_key(row_key, seed_column) if row_key is not None else None
        cells = expand_grid(grid)
        if seeds is None:
            return self._sweep_unsharded(
                command, cells, resources=resources, seed=seed, code_ref=code_ref,
                submitted_by=submitted_by, allow_dirty=allow_dirty, max_jobs=max_jobs,
                sweep_max_cost=sweep_max_cost, daily_budget=daily_budget, committed=committed,
                allow_unknown_config=allow_unknown_config,
            )
        if seed_axis_key in grid or "seed" in grid:
            raise LabError(
                "seeds declared in both 'seeds' and a grid key ('seed'/'" + seed_axis_key + "'); "
                "remove one — seeds are an aggregation axis, not a Cartesian grid key"
            )
        try:
            seed_set = parse_seeds(seeds)
            shards = partition_seeds(seed_set, shard_size if shard_size is not None else len(seed_set))
        except ValueError as e:
            raise LabError(str(e)) from e
        n_jobs = len(cells) * len(shards)
        if n_jobs > max_jobs:
            raise LabError(
                f"sharded sweep would submit {n_jobs} jobs (> max_jobs={max_jobs}); "
                "narrow the grid/seeds, raise shard_size, or raise max_jobs"
            )
        per_point_cap: float | None = (
            sweep_max_cost / n_jobs if sweep_max_cost is not None and n_jobs > 0 else None
        )
        check_sweep_admission(
            n_points=n_jobs, per_point_cap=per_point_cap,
            daily_budget=daily_budget, committed=committed,
        )
        sweep_id = self._claim_sweep_id()  # its dir will hold plan.json: never share one
        all_job_ids: list[str] = []
        plan_cells: list[SweepCell] = []
        for cell in cells:
            coords = {k: str(v) for k, v in cell.items()}
            cid = cell_id_for(coords)
            plan_cells.append(
                SweepCell(
                    coords=coords,
                    cell_id=cid,
                    seeds_expected=seed_set,
                    shard_seeds=shards,
                    shard_job_ids=[],  # filled as each shard launches
                    results_file=results_file,
                    seed_column=seed_column,
                    row_key=row_key_cols,
                    aggregate_ref=str(self.home / sweep_id / "cells" / cid / results_file),
                )
            )
        plan = SweepPlan(
            sweep_id=sweep_id, created_at=now(), command=command,
            seed_axis_key=seed_axis_key, cells=plan_cells,
        )
        stagger = _submit_stagger_s() if self.backend.name != "local" else 0.0
        try:
            for cell, plan_cell in zip(cells, plan_cells):
                for shard in shards:
                    if all_job_ids and stagger:
                        time.sleep(stagger)  # don't stampede the local SkyPilot API server
                        events.note("core.submit_stagger", seconds=stagger, shard=shard[0])
                    point = {**cell, seed_axis_key: seeds_to_arg(shard)}
                    spec = build_sweep_point_spec(
                        command, point, seed=shard[0], resources=resources,
                        code_ref=code_ref, submitted_by=submitted_by,
                        allow_unknown_config=allow_unknown_config,
                    )
                    jid = self.submit(
                        spec, allow_dirty=allow_dirty, sweep_id=sweep_id, cell_id=plan_cell.cell_id
                    )
                    plan_cell.shard_job_ids.append(jid)
                    all_job_ids.append(jid)
        except BaseException as e:
            if not all_job_ids:
                self.store.release_dir(sweep_id)  # claimed above, still empty
                raise  # nothing launched: there is no sweep to report, just the refusal
            # Persist what launched, so aggregate/retry/export see these jobs and `sweep-retry`
            # can submit the rest once the cause is fixed.
            self.store.write_sweep_plan(plan)
            if not isinstance(e, Exception):
                raise  # Ctrl-C/SystemExit: the plan is saved; don't dress the signal up
            raise _partial_sweep_error(
                sweep_id, all_job_ids, n_jobs, e,
                next_step=f"`lab sweep-retry {sweep_id}` submits the rest once that is fixed",
            ) from e
        self.store.write_sweep_plan(plan)
        return sweep_id, all_job_ids

    def _sweep_unsharded(
        self,
        command: str,
        points: list[dict[str, Any]],
        *,
        resources: ResourceRequest | None,
        seed: int | None,
        code_ref: str,
        submitted_by: str,
        allow_dirty: bool,
        max_jobs: int,
        sweep_max_cost: float | None,
        daily_budget: float | None,
        committed: float,
        allow_unknown_config: bool = False,
    ) -> tuple[str, list[str]]:
        """The pre-P1-2 one-job-per-cell path (FR-A5), extracted unchanged."""
        if len(points) > max_jobs:
            raise LabError(
                f"sweep would submit {len(points)} jobs (> max_jobs={max_jobs}); "
                "narrow the grid or raise max_jobs"
            )
        per_point_cap: float | None = (
            sweep_max_cost / len(points) if sweep_max_cost is not None and len(points) > 0 else None
        )
        check_sweep_admission(
            n_points=len(points), per_point_cap=per_point_cap,
            daily_budget=daily_budget, committed=committed,
        )
        sweep_id = f"sweep-{_new_job_id()}"
        job_ids: list[str] = []
        stagger = _submit_stagger_s() if self.backend.name != "local" else 0.0
        try:
            for point in points:
                if job_ids and stagger:
                    time.sleep(stagger)  # don't stampede the local SkyPilot API server
                    events.note("core.submit_stagger", seconds=stagger, shard=None)
                spec = build_sweep_point_spec(
                    command, point, seed=seed, resources=resources,
                    code_ref=code_ref, submitted_by=submitted_by,
                    allow_unknown_config=allow_unknown_config,
                )
                job_ids.append(self.submit(spec, allow_dirty=allow_dirty, sweep_id=sweep_id))
        except Exception as e:
            if not job_ids:
                raise
            raise _partial_sweep_error(
                sweep_id, job_ids, len(points), e,
                next_step="the launched points carry this sweep_id (`lab export`, `lab list`)",
            ) from e
        return sweep_id, job_ids

    def sweep_plan(self, sweep_id: str) -> SweepPlan:
        """Read the persisted shard plan for a sharded sweep (P1-2)."""
        if not self.store.has_sweep_plan(sweep_id):
            raise LabError(f"no shard plan for {sweep_id!r} (not a sharded sweep?)")
        return self.store.read_sweep_plan(sweep_id)

    def aggregate_sweep(
        self,
        sweep_id: str,
        *,
        include_partial: bool = True,
        row_key: str | list[str] | None = None,
    ) -> SweepPlan:
        """Row-concatenate each cell's shard results into one per-cell table (P1-2, FR-SS-4..7).

        Idempotent pull reducer: recomputes from current shard states each call, so it is safe to run
        repeatedly as shards finish. A cell is ``complete`` iff every expected seed is present, else
        ``incomplete`` with the missing seeds named — never presents a short aggregate as complete and
        never discards recovered seeds (FR-SS-7).

        By default any *terminal* shard with a readable results file contributes — the flush-per-seed
        design means a timed-out shard's recovered rows are valid, paid-for data (field-report #2);
        their seeds are reported under ``seeds_partial`` and their rows stamped ``_shard_status``.
        ``include_partial=False`` restores succeeded-only aggregation. Running/queued shards are
        always excluded (their file is still moving under the heartbeat rsync), as are shards whose
        config was unconsumed (wrong-config rows are what the fail-closed check exists to kill).

        Search hits: when any included shard wrote a ``lab.search`` ``hits.jsonl``, the cell also
        gets one concatenated ``hits.jsonl`` beside the result aggregate (:func:`merge_hits`;
        same inclusion rules, every hit stamped ``_shard_status``), recorded as ``hits_ref`` /
        ``hits`` on the cell. Seed presence is judged by the result table alone — a hit never
        makes a seed present — and a hit naming a seed the table lists as missing is left out,
        so the two cell files agree on which seeds the cell covers. When no included shard has one, a cell hits file left by an earlier
        aggregate is removed, so the plan, the file and an export bundle cannot disagree.

        Runs under the sweep's plan lock (:meth:`JobStore.sweep_lock`), so it can neither write
        back a plan read before a concurrent ``sweep-retry`` recorded its shards nor interleave
        with another aggregate.
        """
        self.sweep_plan(sweep_id)  # unknown sweep: LabError, before a lock file can be created
        with self.store.sweep_lock(sweep_id):
            return self._aggregate_sweep_locked(
                sweep_id, include_partial=include_partial, row_key=row_key
            )

    def _aggregate_sweep_locked(
        self,
        sweep_id: str,
        *,
        include_partial: bool = True,
        row_key: str | list[str] | None = None,
    ) -> SweepPlan:
        """:meth:`aggregate_sweep`'s body; the caller holds the sweep's plan lock."""
        from lab.search import HITS_FILE  # local: lab.search imports numpy; keep it off CLI startup

        plan = self.sweep_plan(sweep_id)
        for cell in plan.cells:
            if row_key is not None:
                # Aggregate-time declaration for plans that predate --row-key (persisted below
                # via write_sweep_plan, so later aggregates/retries inherit it).
                cell.row_key = _parse_row_key(row_key, cell.seed_column)
            included: list[_IncludedShard] = []
            for jid in cell.shard_job_ids:
                m = self._fresh_manifest(jid)  # a dead supervisor must not pin a shard `running`
                if m.status not in _TERMINAL_STATES:
                    continue  # still moving under the heartbeat rsync — never read mid-flight
                if not include_partial and m.status is not JobState.succeeded:
                    continue
                if m.unconsumed_config:
                    continue  # ran different config than requested — never aggregate its rows
                self.fetch_artifacts(jid)  # ensure the local copy exists (R2 fallback inside)
                out = self.store.output_dir(jid)
                rf, hf = out / cell.results_file, out / HITS_FILE
                included.append(
                    (jid, m.status.value, rf if rf.exists() else None, hf if hf.exists() else None)
                )
            hits_dest = Path(cell.aggregate_ref).with_name(HITS_FILE)
            # Taken before anything is read: a file that changes after this is re-read next time.
            fingerprint = _cell_fingerprint(cell, included)
            if fingerprint == cell.inputs_fingerprint and _cell_outputs_intact(cell, hits_dest):
                continue  # same inputs as the last aggregate: its outputs and counts still stand
            shard_texts = [(rf.read_text(), st) for _, st, rf, _ in included if rf is not None]
            hit_files = [(hf, st) for _, st, _, hf in included if hf is not None]
            merged, present, partial = merge_seed_rows(
                shard_texts, cell.seed_column, row_key=cell.row_key
            )
            # Both merges run before anything is written: a fatal line in either leaves this
            # cell's files exactly as they were. Hits files are read one at a time, as merged.
            merged_hits, n_hits = merge_hits(
                ((hf.read_text(), st) for hf, st in hit_files), row_key=cell.row_key,
                # Hits only for seeds the table lists as present, so the two files agree.
                seed_column=cell.seed_column, present_seeds=present,
            )
            cell.seeds_present = present
            cell.seeds_partial = partial
            present_set = set(present)
            cell.missing_seeds = [s for s in cell.seeds_expected if s not in present_set]
            cell.status = "complete" if not cell.missing_seeds else "incomplete"
            csv_key = f"{sweep_id}/cells/{cell.cell_id}/{cell.results_file}"
            if merged:
                dest = Path(cell.aggregate_ref)
                atomic_write_text(dest, merged)  # an export reading it never sees it torn
                _mirror_sweep_aggregate(dest, csv_key)
            if hit_files:
                # Written even when empty: an empty file is "searched, nothing passed".
                atomic_write_text(hits_dest, merged_hits)
                cell.hits_ref, cell.hits = str(hits_dest), n_hits
                _mirror_sweep_aggregate(
                    hits_dest, posixpath.join(posixpath.dirname(csv_key), HITS_FILE)
                )
            else:
                hits_dest.unlink(missing_ok=True)
                cell.hits_ref, cell.hits = None, None
            cell.inputs_fingerprint = fingerprint
        self.store.write_sweep_plan(plan)
        return plan

    def retry_sweep(self, sweep_id: str, *, allow_dirty: bool = True) -> SweepPlan:
        """Resubmit only the missing shards of incomplete cells, then re-aggregate (P1-2, FR-SS-7).

        A shard is missing if any of its assigned seeds is absent from the current aggregate. Fresh
        shard jobs join the same ``sweep_id``/``cell_id``; succeeded shards are never touched.

        Safe to call repeatedly: if a prior retry's job for a given seed subset is still in a
        non-terminal state (queued/running), that shard is skipped — no duplicate in-flight jobs.
        That holds for *concurrent* calls too: the whole read -> submit -> write runs under the
        sweep's plan lock (:meth:`JobStore.sweep_lock`), so a second retry waits for the first
        and then sees its shards in the plan, in flight.
        """
        self.sweep_plan(sweep_id)  # unknown sweep: LabError, before a lock file can be created
        with self.store.sweep_lock(sweep_id):
            # Refresh present/missing from current shard states.
            plan = self._aggregate_sweep_locked(sweep_id)
            launched: list[str] = []
            try:
                self._retry_cells(plan, launched, allow_dirty=allow_dirty)
            except BaseException as e:
                if not launched:
                    raise
                # Same rule as `sweep`: what launched is persisted, so a re-run of retry sees it
                # as in flight instead of buying it a second time, and the caller learns the ids.
                self.store.write_sweep_plan(plan)
                if not isinstance(e, Exception):
                    raise
                raise _partial_sweep_error(
                    sweep_id, launched, None, e,
                    next_step=f"`lab sweep-retry {sweep_id}` resumes once that is fixed",
                ) from e
            self.store.write_sweep_plan(plan)
            return self._aggregate_sweep_locked(sweep_id)

    def _retry_cells(self, plan: SweepPlan, launched: list[str], *, allow_dirty: bool) -> None:
        """:meth:`retry_sweep`'s submit loop (the caller holds the sweep's plan lock). Appends
        each launched id to its cell *and* to ``launched`` as it goes, and persists the plan after
        every launch — so a refusal partway leaves both accurate, and so does a retry killed
        outright (SIGKILL runs no ``except``), which used to leave its launched shards in no
        plan at all."""
        sweep_id = plan.sweep_id
        any_shard = next((j for c in plan.cells for j in c.shard_job_ids), None)
        for cell in plan.cells:
            if cell.status != "incomplete":
                continue
            present = set(cell.seeds_present)
            # Collect seed SETS of all currently in-flight (non-terminal) shard jobs so we can
            # skip resubmitting seeds that already have a live retry covering them. Sets (not
            # exact strings): a full-shard in-flight retry "2,3" must suppress a narrowed "3".
            in_flight_seed_sets: list[set[int]] = []
            for jid in cell.shard_job_ids:
                m = self._fresh_manifest(jid)  # a dead supervisor is not an in-flight retry
                if m.status not in _TERMINAL_STATES:
                    sub = m.run.resolved_config.get(plan.seed_axis_key)
                    if sub is not None:
                        try:
                            in_flight_seed_sets.append(set(parse_seeds(str(sub))))
                        except ValueError:
                            pass  # unparseable subset — can't prove coverage, don't suppress
            # Inherit the original shard resources (timeout/backend/etc.) from an existing shard:
            # this cell's own, or — for a cell a refused sweep never reached — any of the sweep's
            # (every shard of a sweep is submitted with the same resources).
            base_id = cell.shard_job_ids[0] if cell.shard_job_ids else any_shard
            if base_id is None:
                raise LabError(f"sweep {sweep_id} has no launched shard to inherit resources from")
            base_manifest = self.manifest(base_id)
            base_resources = base_manifest.resources
            # Launch where the sweep ran. Both shells build a local-backend Lab for sweep-retry
            # whatever the sweep used, so `self.submit` turned a retried cpu/skypilot shard into a
            # local subprocess that still carried the remote resources (`provisioner=local
            # cloud=do`) — work silently moved off the box it was sized for.
            launcher = self._lab_on(base_manifest.backend.provisioner)
            stagger = _submit_stagger_s() if launcher.backend.name != "local" else 0.0
            for shard in cell.shard_seeds:
                # Partial recovery (field-report #2): resubmit only the seeds still missing from
                # this shard — a timed-out shard's recovered rows are kept, not re-bought.
                shard_missing = [s for s in shard if s not in present]
                if not shard_missing:
                    continue  # this shard's seeds are already covered
                in_flight_union: set[int] = set().union(*in_flight_seed_sets) if in_flight_seed_sets else set()
                if set(shard_missing) <= in_flight_union:
                    continue  # live retries already cover these seeds (jointly) — don't duplicate
                point = {**cell.coords, plan.seed_axis_key: seeds_to_arg(shard_missing)}
                spec = build_sweep_point_spec(
                    plan.command, point, seed=shard_missing[0], resources=base_resources,
                    allow_unknown_config=base_manifest.run.allow_unknown_config,
                )
                if launched and stagger:
                    time.sleep(stagger)  # same API-server cold-start guard as `sweep`
                jid = launcher.submit(
                    spec, allow_dirty=allow_dirty, sweep_id=sweep_id, cell_id=cell.cell_id
                )
                cell.shard_job_ids.append(jid)
                launched.append(jid)
                self.store.write_sweep_plan(plan)

    def _lab_on(self, provisioner: str) -> Lab:
        """This Lab, or one sharing its repo/home/store that launches through ``provisioner``.

        The launch-side twin of :meth:`_backend_for`: for re-launching work (``retry_sweep``)
        under the backend that ran it, rather than whichever backend the calling shell built.
        """
        if provisioner == self.backend.name:
            return self
        try:
            backend = self._backend_for(provisioner)
        except ImportError as e:  # e.g. the skypilot extra is not installed on this machine
            raise LabError(
                f"this work ran on the {provisioner!r} backend, which cannot be loaded here "
                f"({e}); install that extra to relaunch it"
            ) from e
        return Lab(backend=backend, repo=self.repo, home=self.home)

    def _sibling_lab(self, repo: Path) -> Lab:
        """A Lab rooted at ``repo`` (e.g. an extracted bundle) over the same backend kind, sharing
        this lab's home/store — mirrors the scheduler's ``make_lab`` for confirm relaunches."""
        return Lab(
            backend=build_backend(self.backend.name, home=self.home, repo=repo),
            repo=repo,
            home=self.home,
        )

    def confirm(
        self,
        orig_id: str,
        *,
        metrics: Iterable[str] | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-12,
        wait: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Re-derive a prior result from its pinned provenance and judge whether it still holds
        (the reproducibility gate). Relaunches ``orig_id`` *fresh* from its committed commit (no
        cache), then compares the re-run's final metric(s) against the original's snapshot within
        tolerance → ``match`` / ``drift`` / ``rerun_failed``.

        Refuses outright (``LabError``) to confirm a run that did not **succeed** or that ran from a
        dirty tree — a non-succeeded or not-fully-captured run has no honest result to re-derive.
        ``metrics`` restricts which metrics are judged (default: all in the baseline).
        """
        try:
            m = self.manifest(orig_id)
        except FileNotFoundError as e:
            raise LabError(f"cannot confirm {orig_id!r}: run not found in {self.home}") from e
        if m.status is not JobState.succeeded:
            raise LabError(
                f"cannot confirm {orig_id}: its producing run is '{m.status.value}', not "
                "'succeeded' — a non-succeeded run has no result to re-derive (FR-B)"
            )
        if m.code.git_dirty:
            raise LabError(
                f"cannot confirm {orig_id}: it ran from a dirty working tree, so its code was not "
                "fully captured and can't be honestly re-derived (FR-B1)"
            )
        # Baseline: prefer the durable manifest snapshot; fall back to the original's metrics file.
        baseline = dict(m.final_metrics) or final_values(self.backend.read_metrics(orig_id))
        if not baseline:
            raise LabError(
                f"cannot confirm {orig_id}: no baseline metrics — the manifest snapshot is empty "
                "and metrics.jsonl is unavailable; nothing to compare against"
            )
        # Relaunch fresh from the pinned commit: committed tree only, never the cache.
        from lab.scheduler.bundle import create_bundle, extract_bundle  # avoid import cycle

        if not commit_exists(self.repo, m.code.git_commit):
            raise LabError(
                f"cannot confirm {orig_id}: its pinned commit {m.code.git_commit[:12]} is not in "
                f"{self.repo} — fetch it (e.g. `git fetch --all`) then retry"
            )
        bundle_root = self.home / "_confirm"
        tar, _ = create_bundle(
            self.repo, bundle_root, commit=m.code.git_commit, include_dirty=False
        )
        bundle_dir = extract_bundle(tar, bundle_root / orig_id)
        bundle_lab = self._sibling_lab(bundle_dir)
        spec = JobSpec(
            command=m.run.entrypoint_command,
            config=m.run.resolved_config,
            seed=m.run.seed,
            resources=m.resources,
            submitted_by="agent",
            # The original's own opt-out: without it a run that legitimately succeeded with
            # unconsumed keys re-runs under the fail-closed audit and always "fails" to confirm.
            allow_unknown_config=m.run.allow_unknown_config,
        )
        confirm_id = bundle_lab.submit(
            spec,
            code=CodeRef(git_commit=m.code.git_commit, git_dirty=False),
            confirms=orig_id,
        )
        result: dict[str, Any] = {"orig_id": orig_id, "confirm_id": confirm_id}
        if not wait:
            result["verdict"] = "pending"
            return result
        # A local re-run's manifest is a free filesystem read and usually lands within a second,
        # so wait's 10s default would leave every local confirm idle for a whole interval. Remote
        # re-runs keep the 10s cadence: provisioning alone takes minutes.
        interval = 0.5 if bundle_lab.backend.name == "local" else 10.0
        (rerun,) = self.wait([confirm_id], timeout=timeout, interval=interval)
        if rerun.status not in _TERMINAL_STATES:
            # wait gave up before the re-run finished — it's still alive (and, on a remote backend,
            # still billing until it tears down). Don't call a running job failed.
            result["verdict"] = "timed_out_waiting"
            result["rerun_status"] = rerun.status.value
            return result
        if rerun.status is not JobState.succeeded:
            result["verdict"] = "rerun_failed"
            result["rerun_status"] = rerun.status.value
            return result
        verdict, deltas = compare_final_metrics(
            baseline, rerun.final_metrics, names=metrics, rtol=rtol, atol=atol
        )
        result["verdict"] = verdict
        result["deltas"] = deltas
        result["env_drift"] = rerun.env.uv_lock_sha256 != m.env.uv_lock_sha256
        return result

    def status(self, job_id: str) -> JobState:
        return self.backend.status(job_id)

    def logs(self, job_id: str, tail: int | None = 100) -> list[str]:
        return list(self.backend.tail_logs(job_id, tail=tail))

    def metrics(
        self, job_id: str, names: Iterable[str] | None = None, since_step: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Grouped incremental metric series for a job, queryable live (FR-D2)."""
        return group_series(self.backend.read_metrics(job_id, names=names, since_step=since_step))

    def cancel(self, job_id: str) -> JobState:
        return self.backend.cancel(job_id)

    def fetch_artifacts(self, job_id: str, dest: str | None = None) -> list[ArtifactRecord]:
        out = self.store.output_dir(job_id)
        has_local = out.exists() and any(out.iterdir())
        if not has_local and r2_enabled():  # local copy gone — pull the durable copy from R2
            self._r2_fallback(job_id, out)
        return self.backend.collect_artifacts(job_id, dest or str(self.store.job_dir(job_id)))

    def _r2_fallback(self, job_id: str, out: Path) -> None:
        """Best-effort recovery of a job's outputs from R2 when the local copy is missing.

        Never raises for an R2 problem: ``sweep-aggregate``/``sweep-retry`` call this per shard,
        and one shard's network or credential error used to abort the whole sweep with a
        traceback and empty stdout. A failure is a stderr line plus a ledger note, and the
        caller carries on with local state (an absent results file reads as missing seeds).

        Skipped for a local-backend job: nothing on that path uploads to R2 or mirrors partials,
        so there is nothing to recover, and asking would only spend a network round trip (or
        fail). The one exception is a manifest that itself records a durable ``artifacts_uri`` —
        then there *is* something to fetch, whoever launched it.
        """
        manifest = self.store.read_manifest(job_id)
        local = manifest.backend.provisioner == "local"
        if local and not manifest.artifacts_uri:
            return
        try:
            r2 = R2Store.from_env()
            if manifest.artifacts_uri and r2 is not None:
                r2.download_dir(job_id, out)
            if r2 is not None and not local and not (out.exists() and any(out.iterdir())):
                # Nothing durable from the end-of-run upload — which is the *normal* state of
                # a job whose supervisor died, since `artifacts_uri` is written on the same
                # line as that upload. The mid-run mirror (`lab.partials`) is then the only
                # copy of rows that were paid for, so fall back to it. It lands the
                # `_PARTIAL.json` marker alongside the tables, so what arrives is
                # self-describing: `sweep-aggregate` still stamps these rows from the
                # producing job's own (non-succeeded) status, exactly as for a partial shard.
                r2.download_dir(f"{job_id}/{PARTIAL_DIR}", out)
        except ImportError as e:
            # R2 env configured but the `r2` extra isn't installed — the fallback is
            # best-effort recovery, never a reason to fail a local operation.
            print(f"[lab] R2 fallback unavailable ({e}); using local state only", file=sys.stderr)
            events.note("storage.fallback_unavailable", job_id=job_id, error=str(e))
        except Exception as e:  # noqa: BLE001 — network/credential errors: same best-effort rule
            print(
                f"[lab] R2 fallback for {job_id} failed ({type(e).__name__}: {e}); "
                "using local state only",
                file=sys.stderr,
            )
            events.note("storage.fallback_failed", job_id=job_id, error=str(e))

    def manifest(self, job_id: str) -> JobManifest:
        return self.store.read_manifest(job_id)

    def _fresh_manifest(self, job_id: str) -> JobManifest:
        """A local job's manifest, refreshed through *its own* backend while it is non-terminal.

        A raw read leaves a shard whose supervisor died frozen at ``running`` forever: the sweep
        readers then never aggregate it, and ``sweep-retry`` treats it as in flight and never
        resubmits it. ``backend.status`` is what notices the dead runner and writes the terminal
        manifest (and, on skypilot, attempts the teardown that never ran) — the same refresh
        :meth:`_resolve_manifest` does for ``wait``. Routed through :meth:`_backend_for` because
        ``sweep-aggregate``/``sweep-retry`` build a local-backend Lab whatever ran the shards.

        Terminal manifests are returned as read (nothing left to notice). A refresh that cannot
        run — e.g. the skypilot extra is not installed here — falls back to the recorded state
        with a stderr line: a stale ``running`` is the old behaviour, never a crash.
        """
        m = self.manifest(job_id)
        if m.status in _TERMINAL_STATES:
            return m
        try:
            self._backend_for(m.backend.provisioner).status(job_id)
        except Exception as e:  # noqa: BLE001 — a refresh is best-effort; the read still answers
            print(
                f"[lab] could not refresh {job_id} through its {m.backend.provisioner} backend "
                f"({type(e).__name__}: {e}); using its recorded state",
                file=sys.stderr,
            )
            events.note("core.refresh_failed", job_id=job_id, error=str(e))
            return m
        return self.manifest(job_id)

    def _queue(self) -> QueueStore:
        """The scheduler's queue store, built once per :class:`Lab` and then reused.

        Calling ``default_queue()`` per lookup rebuilds an :class:`~lab.storage.R2Store`, and with
        it a fresh ``boto3.client("s3", ...)``: botocore's service models are re-loaded and a new
        TLS connection is opened with nothing to keep alive. A poll loop pays that per mirrored
        job per poll — an 8h wait at the 30s mirror floor is ~960 clients for one job — in the
        very code path whose point is cutting the queue's read cost.

        Cached on the instance, never process-globally: ``default_queue()`` reads
        ``LAB_QUEUE_DIR``/R2 env at call time, and a global cache would leak one test's queue into
        the next. Every surface builds its Lab per invocation (the CLI is one-shot; the MCP server
        holds one for the process, which is exactly the lifetime this cache wants).
        """
        q = self._queue_store
        if q is None:
            from lab.scheduler.queue import default_queue  # local import: avoids a module cycle

            q = self._queue_store = default_queue()
        return q

    def _backend_for(self, provisioner: str) -> Backend:
        """The backend that actually ran a job — which is not necessarily this Lab's own.

        ``lab wait`` builds one Lab for a whole id list, from the *first* id's manifest. Now that
        a mirror-only id is accepted, ``lab wait <deferred_skypilot_job> <local_job>`` builds a
        SkyPilot-backed Lab and would refresh the *local* job through it: if that job is
        ``running`` with a dead runner, ``SkyPilotBackend.status``'s dead-supervisor branch tears
        down ``cluster_name_for(<local job id>)`` — a cluster that never existed — records
        ``teardown_status="failed"`` and makes the wait exit 3. That is the FR-C2 money alarm
        firing for a job that has no machine at all.

        So a refresh is routed off each manifest's own ``backend.provisioner``. (:meth:`
        _sibling_lab` is the per-*repo* version of this dispatch and does not fit: it keeps
        ``self.backend.name`` and varies the repo, which is the opposite axis.) Backends are cheap
        handles over the store, and are cached per provisioner for the poll loop.
        """
        if provisioner == self.backend.name:
            return self.backend
        cached = self._backends.get(provisioner)
        if cached is None:
            cached = self._backends[provisioner] = build_backend(
                provisioner, home=self.home, repo=self.repo
            )
        return cached

    def _resolve_manifest(self, job_id: str, *, refresh: bool = False) -> tuple[JobManifest, bool]:
        """A job's manifest from the local store, or failing that from the scheduler queue's
        mirror. Returns ``(manifest, mirrored)``.

        The same fallback :func:`job_status_view` has always given ``lab status`` (spec §4.3),
        extended to the wait path so it can watch a **scheduler-launched (deferred)** job. Those
        never get a record in this project's ``runs/`` — the scheduler launches them from its own
        host and only mirrors their manifests — so ``lab wait`` rejected their ids outright while
        ``lab status`` on the same id answered fine. That gap is what made a loop around
        ``lab status`` the only way to watch a deferred job: 98,654 ``lab status`` calls over one
        five-day campaign, which also blew the event ledger past its size cap.

        ``refresh=True`` additionally asks the backend for the job's live state and re-reads the
        manifest (the read/status/re-read shape :func:`job_status_view` uses). That call is not a
        cosmetic freshness detail: ``LocalBackend.status`` / ``SkyPilotBackend.status`` are what
        notice a supervisor that died without recording a terminal state and *write* the terminal
        manifest (and, on skypilot, attempt the teardown that never ran). A poll loop reduced to a
        plain manifest read would leave such a job ``running`` forever and burn every wait's full
        timeout — ~40% of the DO supervisors in the 2026-08 campaign died silently.

        The local read comes first precisely so a mirrored job can never reach a live backend:
        this machine is not its supervisor, and acting on a job it never supervised is the
        cross-machine risk ``cancel`` refuses (SkyPilot client/server skew, ``lab._skycompat``).
        Ordering it the other way happens to work today only because both backends' ``status``
        reads the manifest on their first line; that is not something ``Backend`` promises.

        The refresh goes through :meth:`_backend_for`, i.e. the provisioner named by the manifest
        that was just read — never blindly ``self.backend``, which for a multi-id wait is whatever
        the *first* id happened to use.

        Raises :class:`FileNotFoundError` — the store's own, unchanged — when neither place has
        the job. ``read_mirrored`` answering ``None`` (a partial/version-skewed mirrored manifest,
        or a read racing an in-progress write) is "not found" too, never a crash. A mirror that
        *raises* is neither: it is :class:`MirrorReadError`, so a caller can tell "this job does
        not exist" from "the mirror could not answer this time" (see :meth:`wait`).
        """
        try:
            m = self.manifest(job_id)
        except FileNotFoundError:
            try:
                mirrored = self._queue().read_mirrored(job_id)
            except Exception as e:  # noqa: BLE001 — every failure mode is "the mirror is down"
                # `R2Store.get_text` re-raises every non-NoSuchKey boto error, and building the
                # queue store can fail on its own. Tagged rather than propagated raw so the poll
                # loop can treat it as "no news yet" instead of dying six hours into a wait.
                raise MirrorReadError(
                    f"could not read job {job_id!r} from the scheduler's queue mirror: {e}"
                ) from e
            if mirrored is None:
                raise
            return mirrored, True
        if refresh:
            self._backend_for(m.backend.provisioner).status(job_id)
            m = self.manifest(job_id)  # re-read: status may have just finalized/torn down the job
        return m, False

    def _resolve_manifests(self, job_ids: Iterable[str]) -> tuple[list[JobManifest], list[str]]:
        """:meth:`_resolve_manifest` over several jobs, without refreshing: ``(manifests, ids
        that were read from the mirror)``."""
        manifests: list[JobManifest] = []
        mirrored: list[str] = []
        for job_id in job_ids:
            m, from_mirror = self._resolve_manifest(job_id)
            manifests.append(m)
            if from_mirror:
                mirrored.append(job_id)
        return manifests, mirrored

    def list_jobs(self) -> list[JobManifest]:
        """Every readable manifest in this project's store.

        One unreadable manifest (corrupt JSON, or a schema an older/newer lab wrote) used to
        raise out of here and take down ``list``, ``status``/``wait`` and reconcile for the whole
        project. It is now skipped with a stderr warning (once per Lab) and named in
        :attr:`unreadable_jobs`, which :meth:`spend` and :meth:`reconcile` carry into their
        output — skipped is never the same as absent: reconcile keeps such a job's machine
        protected, because for all it knows that job is running.
        """
        manifests: list[JobManifest] = []
        unreadable: list[dict[str, str]] = []
        for job_id in self.store.list_job_ids():
            try:
                manifests.append(self.store.read_manifest(job_id))
            except (OSError, ValueError) as e:  # ValueError covers pydantic's ValidationError
                reason = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"[:300]
                unreadable.append({"job_id": job_id, "error": reason})
                if job_id not in self._warned_unreadable:
                    self._warned_unreadable.add(job_id)
                    print(
                        f"[lab] skipping unreadable manifest {job_id} ({reason}); see "
                        f"{self.store.manifest_path(job_id)}",
                        file=sys.stderr,
                    )
                    events.note("core.manifest_unreadable", job_id=job_id, error=reason)
        self.unreadable_jobs = unreadable
        return manifests

    def spend(
        self,
        manifests: list[JobManifest] | None = None,
        *,
        alert_usd: float | None = None,
    ) -> dict[str, Any]:
        """Realized spend across this project's jobs, plus the threshold verdict (FR-I2).

        Thin wrapper over :func:`lab.pricing.realized_spend` so the CLI and MCP shells share one
        implementation; ``manifests`` lets a caller that has already listed the jobs (both shells
        do) avoid re-reading every manifest. ``alert_usd`` adds an ``alert`` key: ``None`` until
        the total crosses it, then the verdict a shell renders — stderr in the CLI, in-payload for
        MCP, which has no second channel.

        Read :data:`lab.pricing.SPEND_SCOPE` (carried in the return) before quoting the number:
        it covers this project's ``runs/`` only, and jobs whose rate was never readable are named
        rather than counted as free.
        """
        summary = pricing.realized_spend(
            self.list_jobs() if manifests is None else manifests, at=now()
        )
        summary["alert"] = pricing.spend_alert(summary, threshold_usd=alert_usd)
        # Jobs `list_jobs` had to skip: their spend is unknown, so name them like
        # `unknown_cost_jobs` rather than let the total silently read as complete.
        summary["unreadable_jobs"] = [u["job_id"] for u in self.unreadable_jobs]
        return summary

    def ps(self) -> dict[str, Any]:
        """Every non-terminal job this machine knows about, across every project (FR-C2 gap fix).

        ``list_jobs`` only sees *this* project's own ``runs/``; ``reconcile`` cross-references
        cloud state but reports leaks, not a plain "what's running" list; the scheduler's queue
        covers only deferred jobs. None of them answers "is anything running right now, anywhere
        on this machine" on their own — found live 2026-08-27, when both were checked and neither
        surfaced real running jobs in a different project. This walks the machine-wide job
        registry (:mod:`lab.attribution`) instead, opening each owning project's own store to
        read its current status.

        Best-effort per job: a registry entry whose manifest is missing or unreadable (a wiped
        ``runs/``, a project moved or deleted) is silently skipped rather than raising — the
        registry is a hint of where to look, not a promise the job still exists.
        """
        from lab.attribution import known_jobs

        jobs: list[dict[str, Any]] = []
        stores: dict[Path, JobStore] = {}
        for kj in known_jobs():
            if kj.runs_dir is None:
                continue
            store = stores.get(kj.runs_dir)
            if store is None:
                store = JobStore(kj.runs_dir)
                stores[kj.runs_dir] = store
            try:
                manifest = store.read_manifest(kj.job_id)
            except Exception:  # noqa: BLE001 — a stale registry entry, not a failure
                continue
            if manifest.status in _TERMINAL_STATES:
                continue
            supervised = "n/a"
            # A local job's runner is its supervisor just as a skypilot job's is: when it dies,
            # nothing enforces the wall-clock cap and nothing will record a verdict. `ps` used to
            # check skypilot only, so such a job read `running`/`n/a`. Read-only here, as for
            # skypilot: the next `status`/`wait` on the job finalizes it and stops its group.
            if manifest.backend.provisioner in ("skypilot", "local"):
                runtime = store.read_runtime(kj.job_id)
                alive = pid_alive(
                    runtime.get("runner_pid"), start_time=runtime.get("runner_start_time")
                )
                if alive:
                    supervised = "local"
                else:
                    # Same grace `reconcile()`'s `unsupervised` pass gives a dead-pid read: a
                    # just-launched supervisor hasn't written `runner_pid` yet, and that gap must
                    # not read as "orphaned" on the command whose whole point is a trustworthy
                    # answer before touching a live job.
                    age = (now() - (manifest.started_at or manifest.created_at)).total_seconds()
                    supervised = "starting" if age <= UNSUPERVISED_GRACE_S else "unsupervised"
            jobs.append(
                {
                    "job_id": kj.job_id,
                    "project": kj.project or "(unknown)",
                    "status": manifest.status.value,
                    "backend": manifest.backend.provisioner,
                    "cloud": manifest.resources.cloud,
                    "supervised": supervised,
                    "started_at": manifest.started_at.isoformat() if manifest.started_at else None,
                }
            )
        jobs.sort(key=lambda j: j["started_at"] or "")
        return {"jobs": jobs, "count": len(jobs)}

    def jobs_in_sweep(self, sweep_id: str) -> list[str]:
        """The job ids of a sweep, sorted.

        A sharded sweep's plan.json records every shard id — retries included, since they hold
        the plan lock and record each launch — so those are read off the plan (dropping any whose
        run dir is gone, as the scan never saw them). Only a sweep without a plan (unsharded, or
        one still submitting its first shards) falls back to parsing every manifest in the
        project, which is what `sweep-status`, `wait --sweep` and `export <sweep>` all used to
        do: 0.56 s at 10k jobs, and 1.77 s p50 under 20x concurrency (stress run 2026-11-19).
        """
        if self.store.has_sweep_plan(sweep_id):
            plan = self.store.read_sweep_plan(sweep_id)
            ids = {j for cell in plan.cells for j in cell.shard_job_ids}
            return sorted(j for j in ids if self.store.manifest_path(j).exists())
        return [j.job_id for j in self.list_jobs() if j.sweep_id == sweep_id]

    def _sweep_manifests(self, sweep_id: str) -> list[JobManifest]:
        """Every readable manifest of a sweep (:meth:`jobs_in_sweep`'s ids), in id order. One
        that cannot be read is skipped with a stderr line, as :meth:`list_jobs` skips it."""
        if not self.store.has_sweep_plan(sweep_id):
            return [m for m in self.list_jobs() if m.sweep_id == sweep_id]
        manifests: list[JobManifest] = []
        for job_id in self.jobs_in_sweep(sweep_id):
            try:
                manifests.append(self.store.read_manifest(job_id))
            except (OSError, ValueError) as e:  # ValueError covers pydantic's ValidationError
                print(f"[lab] skipping unreadable manifest {job_id} ({e})", file=sys.stderr)
                events.note("core.manifest_unreadable", job_id=job_id, error=str(e)[:300])
        return manifests

    def export(
        self,
        target_id: str,
        dest: Path,
        *,
        include_logs: bool = False,
        max_file_mb: float = 32.0,
    ) -> dict[str, Any]:
        """Export a committable provenance bundle for a job or a whole sweep (field-report #5).

        ``runs/`` is git-ignored and lives only in the lab repo — the analysis repo (where the
        paper is written) has no supported route to the manifests and result tables behind a
        figure. This writes exactly the small, durable subset that belongs in version control:
        per job ``manifest.json``, ``resolved_config.json``, ``code_diff.tar.gz`` when present,
        and the figure/table artifacts under a size cap (blobs like ``.npz``/checkpoints are
        excluded — recorded in the index under ``skipped``, never silently dropped). Sweeps also
        bundle ``plan.json`` + the per-cell aggregates. ``index.json`` ties every file to a
        commit, seed, state, and spend. Idempotent; returns the index.
        """
        dest = Path(dest)
        if target_id.startswith("sweep-"):
            kind = "sweep"
            job_ids = self.jobs_in_sweep(target_id)
            if not job_ids:
                raise LabError(f"sweep {target_id!r} matched no jobs in {self.home}")
        else:
            kind = "job"
            if not self.store.manifest_path(target_id).exists():
                raise LabError(f"unknown job or sweep id {target_id!r}")
            job_ids = [target_id]

        max_bytes = int(max_file_mb * 1024 * 1024)
        jobs_index: list[dict[str, Any]] = []
        for jid in sorted(job_ids):
            m = self.manifest(jid)
            try:
                self.fetch_artifacts(jid)  # R2 fallback for outputs that only live durably
            except Exception as e:  # noqa: BLE001 — export what exists locally
                print(
                    f"[lab] export: fetch_artifacts({jid}) failed, exporting local state: {e}",
                    file=sys.stderr,
                )
            jdir = dest / jid
            jdir.mkdir(parents=True, exist_ok=True)
            (jdir / "manifest.json").write_text(self._exportable_manifest_json(jid))
            (jdir / "resolved_config.json").write_text(
                json.dumps(m.run.resolved_config, indent=2, sort_keys=True, default=str)
            )
            files = ["manifest.json", "resolved_config.json"]
            diff = self.store.job_dir(jid) / "code_diff.tar.gz"
            if diff.exists():
                (jdir / "code_diff.tar.gz").write_bytes(diff.read_bytes())
                files.append("code_diff.tar.gz")
            if include_logs and self.store.logs_path(jid).exists():
                (jdir / "logs.txt").write_text(  # FR-J1: the bundle is meant to be committed
                    redact(self.store.logs_path(jid).read_text(errors="replace"))
                )
                files.append("logs.txt")
            # Notes travel with the run. `runs/` is git-ignored, so the bundle is the only route
            # a note has into the repo where the result gets written up — which is exactly where
            # "two near-threshold cells flipped across attempts" needs to be legible.
            src_notes = lab_notes.job_notes_path(jid, home=self.home)
            if src_notes.is_file():
                (jdir / "notes.jsonl").write_text(src_notes.read_text(errors="replace"))
                files.append("notes.jsonl")
            skipped: list[dict[str, str]] = []
            out = self.store.output_dir(jid)
            if out.exists():
                for f in sorted(p for p in out.rglob("*") if p.is_file()):
                    rel = f.relative_to(out).as_posix()
                    if rel.startswith("."):
                        continue
                    atype = infer_artifact_type(f.name)
                    if atype not in ("figure", "table"):
                        skipped.append({"file": rel, "reason": f"type {atype} (blob)"})
                        continue
                    size = f.stat().st_size
                    if size > max_bytes:
                        skipped.append(
                            {"file": rel, "reason": f"size {size}B > {max_file_mb}MB cap"}
                        )
                        continue
                    target = jdir / "output" / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(f.read_bytes())
                    files.append(f"output/{rel}")
            jobs_index.append(
                {
                    "job_id": jid,
                    "state": m.status.value,
                    "git_commit": m.code.git_commit,
                    "git_dirty": m.code.git_dirty,
                    # Bundle-relative when it is this job's own local diff (exported beside it).
                    "diff_ref": (
                        f"{jid}/{rel_diff}"
                        if (rel_diff := self._job_relative(jid, m.code.diff_ref)) is not None
                        else m.code.diff_ref
                    ),
                    "seed": m.run.seed,
                    "sweep_id": m.sweep_id,
                    "cell_id": m.cell_id,
                    "end_reason": m.end_reason,
                    "actual_usd": m.cost.actual_usd if m.cost else None,
                    "files": files,
                    "skipped": skipped,
                }
            )

        sweep_block: dict[str, Any] | None = None
        if kind == "sweep" and self.store.has_sweep_plan(target_id):
            plan = self.sweep_plan(target_id)
            cell_skipped: list[dict[str, str]] = []
            for cell in plan.cells:
                # The result aggregate, plus the search-hits aggregate when the plan records one
                # (keyed on the plan, not on the file, so a stale file is never bundled). Under
                # the same per-file cap as job outputs: a 91.5 MB cell hits.jsonl used to go in
                # whole, into a 150 MB bundle meant to be committed (stress run 2026-11-19).
                aggs = [Path(cell.aggregate_ref)]
                if cell.hits_ref is not None:
                    aggs.append(Path(cell.hits_ref))
                for agg in aggs:
                    if not agg.exists():
                        continue
                    rel = f"cells/{cell.cell_id}/{agg.name}"
                    target = dest / rel
                    size = agg.stat().st_size
                    if size > max_bytes:
                        cell_skipped.append(
                            {"file": rel, "reason": f"size {size}B > {max_file_mb}MB cap"}
                        )
                        target.unlink(missing_ok=True)  # never leave a stale copy behind
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(agg.read_bytes())
            # The exported plan points into the bundle, not at this machine's runs/ dir: an
            # absolute path leaks the home directory and resolves nowhere else. The stored plan
            # is untouched.
            portable = _bundle_relative_plan(plan)
            (dest / "plan.json").write_text(portable.model_dump_json(indent=2))
            sweep_block = {
                "sweep_id": target_id, "plan_exported": True, **portable.view(),
                "skipped": cell_skipped,
            }

        index: dict[str, Any] = {
            "exported_at": now().isoformat(),
            "source_id": target_id,
            "kind": kind,
            "jobs": jobs_index,
            "sweep": sweep_block,
        }
        atomic_write_text(dest / "index.json", json.dumps(index, indent=2, default=str))
        return index

    def _exportable_manifest_json(self, job_id: str) -> str:
        """The stored manifest as ``export`` writes it: verbatim, except that an inline secret in
        ``entrypoint_command`` is masked (see :func:`_mask_command_secrets`) and local paths are
        made relative to the job's own bundle dir — ``artifacts[].path`` becomes
        ``output/<name>`` and a local ``code.diff_ref`` its path beside the manifest — since an
        absolute path leaks the home directory and resolves on no other machine. The stored
        manifest is never touched — it stays the reproducibility record."""
        raw = self.store.manifest_path(job_id).read_text()
        m = self.manifest(job_id)
        update: dict[str, Any] = {}
        masked = _mask_command_secrets(m.run.entrypoint_command)
        if masked != m.run.entrypoint_command:
            update["run"] = m.run.model_copy(update={"entrypoint_command": masked})
        artifacts = [a.model_copy(update={"path": f"output/{a.name}"}) for a in m.artifacts]
        if artifacts != m.artifacts:
            update["artifacts"] = artifacts
        rel_diff = self._job_relative(job_id, m.code.diff_ref)
        if rel_diff is not None:
            update["code"] = m.code.model_copy(update={"diff_ref": rel_diff})
        if not update:
            return raw  # nothing to mask or relativize: byte-identical copy, as before
        return m.model_copy(update=update).model_dump_json(indent=2)

    def _job_relative(self, job_id: str, path: str | None) -> str | None:
        """``path`` relative to the job's run dir when it is an absolute path inside it (where
        export mirrors the job's files), else ``None`` — a URI or bundle key is left alone."""
        if not path or not Path(path).is_absolute():
            return None
        try:
            return Path(path).relative_to(self.store.job_dir(job_id)).as_posix()
        except ValueError:
            return None

    def sweep_summary(self, sweep_id: str) -> dict[str, Any]:
        """Aggregate a sweep's outcomes for trustworthy reporting (preemptions, fallback, spend)."""
        ms = [
            m if m.status in _TERMINAL_STATES else self._fresh_manifest(m.job_id)
            for m in self._sweep_manifests(sweep_id)
        ]

        def spend(m: JobManifest) -> float:
            return m.cost.actual_usd if m.cost and m.cost.actual_usd else 0.0

        # One count per state, every state, so the parts always sum to `total`: counting only
        # succeeded/failed/preempted left cancelled and timed_out shards unaccounted for
        # (`32 = 28 + 1 + 0`), which reads as jobs that went missing.
        by_state = {st.value: 0 for st in JobState}
        for m in ms:
            by_state[m.status.value] += 1
        return {
            "sweep_id": sweep_id,
            "total": len(ms),
            **by_state,
            "fell_back_to_on_demand": int(
                sum(int(m.resources.use_spot and m.backend.launched_spot is False) for m in ms)
            ),
            "total_usd": round(sum(spend(m) for m in ms), 6),
            "per_point": {
                m.job_id: {
                    "state": m.status.value,
                    "usd": round(spend(m), 6),
                    "launched_spot": m.backend.launched_spot,
                }
                for m in ms
            },
        }

    def _sky_status_records(self) -> list[Any]:
        """Raw ``sky.status()`` records — the ground truth :meth:`_sky_status_orphans` filters
        one way (extra clusters with no local job) and the ghost pass in :meth:`reconcile` filters
        the other way (a protected DO/GCP job's cluster confirmed still live). Raises on a genuine
        query failure; callers gate on ``sky_ok`` (see :mod:`lab._skycompat`) to skip this
        entirely under version skew, where the result cannot be trusted either way."""
        import sky

        try:
            return sky.get(sky.status(refresh=sky.StatusRefreshMode.AUTO))  # 0.12: RequestId -> list
        except Exception as e:  # noqa: BLE001
            raise LabError(f"could not query SkyPilot cluster status: {e}") from e

    def _sky_status_orphans(
        self, running_clusters: set[str], records: list[Any] | None = None
    ) -> list[str]:
        """Cloud-agnostic orphan pass: ``lab-*`` clusters SkyPilot still tracks/that are still up
        but are NOT tied to a running local job. Covers DO/GCP (and Vast) via SkyPilot's own state,
        complementing the Vast-direct scan. Raises :class:`LabError` if the status query fails.

        ``records`` lets a caller that already fetched ``sky.status()`` this call (``reconcile``'s
        ghost pass does) hand the result in instead of paying for a second live round-trip;
        ``None`` (the default, and what direct callers/tests use) fetches fresh.
        """
        orphans: list[str] = []
        for rec in (records if records is not None else self._sky_status_records()) or []:
            name = rec.get("name") if isinstance(rec, dict) else getattr(rec, "name", None)
            if not name or not str(name).startswith("lab-") or name in running_clusters:
                continue
            orphans.append(name)
        return orphans

    def reconcile(self, *, apply: bool = False, only: set[str] | None = None) -> dict[str, Any]:
        """Cross-check Vast.ai rentals against the local job DB (FR-C2 leak detection).

        ``only`` restricts destruction to resources whose :func:`orphan_key` is in the set --
        everything is still *listed*, only the destroy is filtered. The CLI passes the set the
        operator actually approved, so a resource that becomes an orphan between the preview and
        the confirmed pass (a job whose supervisor dies in that window, dropping its live cluster
        out of ``running_clusters``) is reported but not destroyed. ``None`` means no approval
        was solicited (``--yes``, or the scheduler's unattended sweep) and nothing is filtered.

        Returns a structured report of:

        - ``orphans``: Vast.ai rentals whose label looks like a lab cluster (``lab-*``) but does
          NOT match any **running** local job — these are very likely leaked rentals.
        - ``ghosts``: running local jobs with no confirmed live cloud resource — a Vast job with
          no matching rental label, or a DO/GCP job SkyPilot no longer tracks (unverifiable under
          client/server skew is never reported as a ghost). ``ghost_reasons`` names why, per
          cluster. Advisory only; nothing here is ever destroyed automatically.
        - ``destroyed``: Vast instance IDs we actually destroyed (only when ``apply=True``).

        With ``apply=True``, each orphan is destroyed via the vastai-sdk directly (bypassing
        SkyPilot's local registry — which may have already lost track of the rental). Without
        ``apply``, it's a dry run; no rentals are touched.

        A missing vastai-sdk (a DO/GCP-only install) skips the Vast-direct pass — the report
        carries ``vast_pass`` explaining why — while the cloud-agnostic ``sky.status`` pass and
        the DO volume pass still run. Any other listing failure is a *blind* pass, never a clean
        one. On a dry run it raises :class:`LabError` (exit 2): nothing was destroyed, and a
        report that read clean while blind would be a lie. Under ``apply`` it does not raise —
        raising there discarded the report of destroys other passes had already made, including
        unknown-outcome ones, and lost their exit 5 — so the failed pass is skipped, named in
        ``pass_errors`` (and its ``*_pass`` field), and listed in ``destroy_outcomes`` as
        ``unknown``: whatever it covers may still be billing, and the CLI exits 5.

        A non-terminal skypilot job whose supervisor pid is dead (past a short grace window) does
        NOT protect its cluster: the canonical leak is a supervisor crash that freezes the
        manifest at ``running`` — counting that cluster as healthy would hide the still-billing
        box from every pass. Such jobs are reported under ``unsupervised``.
        """
        from lab.backends.skypilot import (  # local import: skypilot is an optional extra
            _instance_label,
            cluster_name_for,
            list_vast_instances,
        )

        # Refuse to *destroy through SkyPilot* when the client provably cannot read the result:
        # under version skew that report is anti-correlated with reality (incident 2026-08-20 —
        # seven destroys reported as failures, all seven droplets actually gone, exit 0).
        #
        # Scoped to the sky pass alone, not the whole sweep. The Vast, DO-volume and GCP passes
        # talk straight to their providers and are unaffected by a skewed sky client; aborting
        # them too meant the emergency leak-stopping command refused to destroy anything at all
        # exactly when the local API server is unhealthy, which is when leaks are likeliest.
        from lab._skycompat import sky_versions

        sky_ok = sky_versions().compatible
        sky_pass = "ran" if sky_ok else "skipped (client/server version skew)"

        from lab.backends.skypilot import GcpNotConfigured, list_do_volumes, list_gcp_disks
        from lab.backends.skypilot import list_gcp_instances

        # Per-pass listing failures, keyed by the orphan field the pass fills. See the docstring:
        # a dry run raises on the first one; `apply` records them and carries on.
        pass_errors: dict[str, str] = {}

        def _blind(pass_name: str, message: str, exc: BaseException) -> str:
            """A pass whose listing failed. Raises on a dry run; under `apply`, records it and
            returns the ``*_pass`` value naming why the pass did not run."""
            if not apply:
                raise LabError(message) from exc
            pass_errors[pass_name] = message
            print(f"[lab] reconcile {pass_name}: {message} — this pass did not run", file=sys.stderr)
            return f"failed ({message})"

        # Every provider snapshot is taken BEFORE the protected set is read below, never after.
        # A job's manifest is created before its launch is requested, so any machine visible in
        # a snapshot belongs to a manifest that already existed when the snapshot was taken — and
        # is therefore protected by a `list_jobs()` read afterwards. The other order let a shard a
        # sweep submitted mid-reconcile appear in the sky/DO/GCP listing with no protection, and
        # `--apply --yes` (or the scheduler's auto_reconcile) destroyed it.
        vast_pass = "ran"
        try:
            instances = list_vast_instances()
        except ImportError:
            instances = []
            vast_pass = "skipped (vastai-sdk not installed)"
        except Exception as e:  # noqa: BLE001
            vast_pass = _blind("orphans", f"could not list Vast.ai rentals: {e}", e)
            instances = []

        # DO/GCP ground truth for the ghost pass (and the sky orphan pass): Vast has the
        # provider-direct rental listing above; for DO/GCP SkyPilot's own tracked state is the
        # signal. Fetched once and shared — a second live `sky.status()` round-trip per reconcile
        # is pure waste, and `sky` talks to a local API server this call is not otherwise gated
        # on. Under skew it is not fetched at all (see `sky_live` below).
        sky_status_recs: list[Any] = []
        if sky_ok:
            try:
                sky_status_recs = self._sky_status_records()
            except LabError as e:
                sky_pass = _blind("sky_orphans", str(e), e)
                sky_ok = False  # no answer: no sky orphans, and no ghost verdicts from it

        # DO block volumes (best-effort): skipped silently when DO isn't configured (no doctl),
        # since not every account uses DigitalOcean.
        volumes: list[dict[str, Any]] | None
        try:
            volumes = list_do_volumes()
        except Exception:  # noqa: BLE001 — DO not configured/unavailable: skip the volume pass
            volumes = None

        def _gcp_list(what: str, lister: Callable[[], list[dict[str, Any]]]) -> tuple[
            list[dict[str, Any]] | None, str
        ]:
            """Run one GCP listing, distinguishing "GCP isn't set up here" (skip, and say so in
            the report) from "the API failed" (a blind pass — a leak pass that swallows an API
            error reports clean while blind; see :func:`_blind`). The two passes are independent:
            an instance-API hiccup must not also hide leaked disks, the slow, quiet leak."""
            try:
                return lister(), "ran"
            except GcpNotConfigured as e:
                return None, f"skipped ({e})"
            except Exception as e:  # noqa: BLE001
                field = "gcp_orphans" if what == "instances" else "gcp_disk_orphans"
                return None, _blind(field, f"could not list GCP {what}: {e}", e)

        gcp_instances, gcp_pass = _gcp_list("instances", list_gcp_instances)
        gcp_disks, gcp_disk_pass = _gcp_list("disks", list_gcp_disks)

        unsupervised: list[dict[str, str]] = []
        running_clusters: dict[str, str] = {}
        # Every name a live job's machine might be wearing, not just the one we would mint for it
        # today. `cluster_name_for` gained a project slug, so a job launched before that change is
        # still running under `lab-<job_id>`; matching only the current shape would classify its
        # live machine as an orphan and destroy it — the very failure this work exists to prevent.
        # The name recorded at launch is authoritative when present; the two computed shapes cover
        # a runtime file that was lost. Kept separate from `running_clusters` because ghosts are
        # reported per job, and an unmatched *alias* is not a ghost.
        protected: dict[str, str] = {}
        # Which cloud each protected cluster's job actually asked for — the ghost pass below must
        # check a DO/GCP job against DO/GCP's own state, not against Vast's rental list, which it
        # can never appear in regardless of health.
        cluster_cloud: dict[str, str] = {}
        for j in self.list_jobs():
            if j.status in _TERMINAL_STATES:
                continue
            cluster = cluster_name_for(j.job_id)
            runtime = self.store.read_runtime(j.job_id)
            if j.backend.provisioner == "skypilot":
                age = (now() - (j.started_at or j.created_at)).total_seconds()
                # Identity, not just the PID: a recycled number would report a long-dead
                # supervisor alive forever and silently disable this whole branch (F4).
                alive = pid_alive(
                    runtime.get("runner_pid"), start_time=runtime.get("runner_start_time")
                )
                if age > UNSUPERVISED_GRACE_S and not alive:
                    unsupervised.append({"job_id": j.job_id, "cluster": cluster})
                    continue  # dead supervisor -> the cluster is NOT protected
            running_clusters[cluster] = j.job_id
            cluster_cloud[cluster] = j.resources.cloud or "vast"
            aliases = {cluster, f"lab-{j.job_id}"}
            if recorded := runtime.get("cluster"):
                aliases.add(str(recorded))
            for alias in aliases:
                protected[alias] = j.job_id
        # A manifest `list_jobs` could not read is not evidence its job ended: it may be a running
        # job whose file another lab version wrote. Skipping it must not unprotect its machine —
        # that would make it an "orphan" and destroy a live job's box (the 2026-08-20 shape).
        unreadable_ids = [u["job_id"] for u in self.unreadable_jobs]
        for job_id in unreadable_ids:
            aliases = {cluster_name_for(job_id), f"lab-{job_id}"}
            if recorded := self.store.read_runtime(job_id).get("cluster"):
                aliases.add(str(recorded))
            for alias in aliases:
                protected.setdefault(alias, job_id)

        me = local_project()
        other_projects: list[dict[str, Any]] = []
        unattributed: list[str] = []

        def _resolved_dir(path: Path) -> Path:
            """``path`` made comparable with what the registry stores (``record_job`` resolves)."""
            try:
                return Path(path).expanduser().resolve()
            except (OSError, RuntimeError):
                return Path(path)

        def _ours_only(
            pass_name: str, items: list[Any], identify: Callable[[Any], str]
        ) -> list[Any]:
            """Drop every resource this project cannot *prove* it owns.

            This inverts the predicate that caused the 2026-08-20 incident. The old test was
            "named ``lab-*`` and not in **this repo's** job store => orphan => destroy", which
            treats absence of evidence as evidence of orphanhood on a destructive path — and the
            evidence it consults is project-scoped while the cloud account is machine-global, so
            every other lab project's *running* clusters qualified.

            Ownership now comes from :mod:`lab.attribution`, which reads the user-global,
            project-tagged sources. Three outcomes, only one of which may be destroyed:

            * owned by us          -> kept, and destroyed if `--apply`
            * owned by someone else-> ``other_projects``, never destroyed, even if it looks leaked
              (a genuinely leaked foreign cluster is cleaned up by running reconcile from *that*
              project, which is the only place that can tell leaked from live)
            * unattributable       -> ``unattributed``, never destroyed. Same treatment
              ``gcp_unmatched`` already had, now applied to every pass.

            "Us" is this **job store** (``self.home``), not this project *name*. Two stores share a
            name all the time — two clones of one project, or the scheduler host
            (``LAB_REPO_DIR=/opt/<project>``) beside the laptop checkout on one cloud account — so
            a stamped slug that matches ours proves only which project, never which store. Such a
            resource is ours if its job is in this ``runs/``, or if the machine-wide registry
            names *this* runs_dir as its creator; otherwise it is someone else's (registry names
            another runs_dir) or unprovable, and never destroyed. The registry is per-machine,
            so a job the scheduler host launched is unknown here and stays unattributed — which
            is the point: only the store that launched it can tell leaked from live.
            """
            from lab.backends.skypilot import project_slug

            kept: list[Any] = []
            my_slug = project_slug(me) if me else None
            my_runs = _resolved_dir(self.home)
            # This project's own job store is the most authoritative source there is for its own
            # jobs, and it was not consulted: attribution used only the machine-wide index (written
            # from 2026-08-20 onward) and the event ledger (retention-bounded, and switched off
            # entirely by LAB_EVENTS=0). A cluster leaked by a job submitted before this release
            # was therefore permanently unattributable, and `--apply` silently refused to clean it
            # up. A job in our own runs/ is ours by definition.
            ours_locally = {j.job_id for j in self.list_jobs()}
            ours_locally |= {u["job_id"] for u in self.unreadable_jobs}  # ours, if unreadable
            read = {id(item): _lab_identity(identify(item)) for item in items}
            candidates = {c for _, cands in read.values() for c in cands}
            attributions = attribute_jobs(candidates) if candidates else {}
            for item in items:
                ident = identify(item)
                slug, cands = read[id(item)]
                # The registry's record of which job store created this job, when it has one.
                # Only the registry knows a runs_dir; a ledger-only attribution carries None.
                registered = next(
                    (
                        attributions[c]
                        for c in cands
                        if c in attributions and attributions[c].runs_dir is not None
                    ),
                    None,
                )
                owner: str | None
                owner_runs: str | None = None
                if slug is not None and slug != my_slug:
                    # Stamped on the resource itself: no lookup can make it ours.
                    owner = slug
                elif any(c in ours_locally for c in cands):
                    owner = me
                elif registered is not None and registered.runs_dir is not None:
                    if _resolved_dir(registered.runs_dir) == my_runs:
                        owner = me  # the registry names *this* store as its creator
                    else:
                        # Another job store — possibly of the same project name, in which case
                        # this is exactly the case a slug comparison used to destroy.
                        owner = registered.project or slug or "unknown"
                        owner_runs = str(_resolved_dir(registered.runs_dir))
                elif slug is not None:
                    # Our slug, and nothing that says which store: a sibling clone's job, or one
                    # the scheduler host launched (its registry is on another machine). Unproven.
                    owner = None
                else:
                    # A legacy, unstamped name with no registry record — a job that predates the
                    # registry. The ledger's project name is the only evidence that exists for
                    # those (documented fallback); it cannot tell two same-named stores apart,
                    # and is bounded by the ledger's retention.
                    owner = next(
                        (
                            attributions[c].project
                            for c in cands
                            if c in attributions and attributions[c].known
                        ),
                        None,
                    )
                if owner is None:
                    unattributed.append(ident)
                    events.note("reconcile.unattributed", pass_=pass_name, resource=ident)
                elif owner_runs is not None or owner != me:
                    entry: dict[str, Any] = {"pass": pass_name, "resource": ident, "project": owner}
                    if owner_runs is not None:
                        entry["runs_dir"] = owner_runs  # additive: which store, not just project
                    other_projects.append(entry)
                    events.note(
                        "reconcile.other_project", pass_=pass_name, resource=ident, owner=owner
                    )
                else:
                    kept.append(item)
                    events.note("reconcile.orphan", pass_=pass_name, resource=ident)
            return kept

        orphans: list[dict[str, Any]] = []
        matched_clusters: set[str] = set()
        for inst in instances:
            label = _instance_label(inst)
            if "lab-" not in label:
                continue  # not ours — leave it alone
            matched = next((c for c in protected if c.lower() in label), None)
            if matched is not None:
                matched_clusters.add(matched)
                continue
            orphans.append({"id": inst.get("id"), "label": label})
        orphans = _ours_only("orphans", orphans, lambda o: str(o["label"]))

        def _approved(field: str, item: Any) -> bool:
            """Whether this resource is inside the approved set (all of them when unfiltered)."""
            return only is None or orphan_key(field, item) in only

        destroy_outcomes: list[dict[str, Any]] = []

        def _unconfirmed(pass_name: str, resource: Any, exc: BaseException) -> None:
            """Record a destroy whose success could not be confirmed, and say so on stderr.

            A destroy has *three* outcomes, not two, and the third is the one that must reach the
            operator. On 2026-08-20 seven ``sky.down`` calls raised while the server had in fact
            destroyed all seven machines: the exceptions were swallowed by a bare ``print`` and
            the report read ``sky_destroyed: []`` with exit 0, so the operator was told nothing
            had happened. Classification lives in :mod:`lab._skycompat` because the decisive
            signal — a response the client cannot decode — is a SkyPilot version-skew symptom.

            stderr, not stdout: stdout carries only JSON, which callers parse (CLAUDE.md §9).
            The handlers this replaces printed to stdout, so a failed destroy also made the
            report unparseable.
            """
            # local import: keeps the sky-version machinery out of the module import path.
            from lab._skycompat import classify_sky_error

            name = resource if isinstance(resource, str) else (
                resource.get("name") or resource.get("id") or resource.get("label")
            )
            verdict = classify_sky_error(exc)
            destroy_outcomes.append(
                {
                    "pass": pass_name,
                    "resource": name,
                    # Three values, and the distinction matters: `undecodable_response` means the
                    # server almost certainly DID destroy the resource (sky unpickles the
                    # server-side entrypoint before it reads the result, so a reply that cannot be
                    # decoded is still a reply), `failed` means sky refused before acting, and
                    # `unknown` means genuinely undetermined.
                    "outcome": verdict.outcome,
                    "detail": verdict.detail,
                    "error": str(exc),
                }
            )
            print(
                f"[lab] reconcile {pass_name} {name}: {verdict.outcome} — {verdict.detail}",
                file=sys.stderr,
            )

        destroyed: list[int] = []
        if apply and orphans:
            from lab.backends.skypilot import _get_vast_client

            vast_client: Any = None
            for orph in orphans:
                inst_id = orph["id"]
                if inst_id is None or not _approved("orphans", orph):
                    continue
                try:
                    # Built inside the guard: a client that cannot be constructed is a destroy
                    # that did not happen, not a crash that discards every earlier pass's report.
                    vast_client = vast_client or _get_vast_client()
                    vast_client.destroy_instance(id=int(inst_id))
                    destroyed.append(int(inst_id))
                except Exception as e:  # noqa: BLE001
                    _unconfirmed("orphans", orph, e)

        matched_jobs = {protected[c] for c in matched_clusters if c in protected}

        # `None` means unverifiable (skew) rather than "confirmed gone" — a job on a cloud we
        # cannot check must not become a false ghost just because it was never going to appear in
        # Vast's rental list (found live 2026-08-27: every healthy DO job was unconditionally
        # reported as a ghost for exactly this reason).
        sky_live: set[str] | None = None
        if sky_ok:
            sky_live = {
                str(name)
                for rec in sky_status_recs
                if (
                    name := (
                        rec.get("name") if isinstance(rec, dict) else getattr(rec, "name", None)
                    )
                )
            }

        # `ghosts` stays `list[str]` — the CLI/MCP JSON shape is frozen at a release
        # (docs/COMPATIBILITY.md), and this project is unreleased since v0.9.0. The explanation
        # goes in the additive `ghost_reasons` field below instead of restructuring `ghosts`
        # itself, and both `events.note` (for `lab history --full`) and the note still carry it.
        ghosts: list[str] = []
        ghost_reasons: dict[str, str] = {}
        for cluster, jid in sorted(running_clusters.items()):
            if jid in matched_jobs:
                continue
            cloud = cluster_cloud.get(cluster, "vast")
            if cloud != "vast":
                if sky_live is None:
                    continue  # unverifiable under sky skew -- don't manufacture a false ghost
                if cluster in sky_live:
                    continue  # SkyPilot itself confirms this cluster is still live
                reason = f"no matching SkyPilot-tracked cluster (cloud={cloud})"
            else:
                if "orphans" in pass_errors:
                    continue  # the rental listing failed: absence from it proves nothing
                reason = "no matching Vast rental label"
            ghosts.append(cluster)
            ghost_reasons[cluster] = reason
            events.note("reconcile.ghost", cluster=cluster, job_id=jid, cloud=cloud, reason=reason)

        sky_orphans = (
            _ours_only(
                "sky_orphans",
                self._sky_status_orphans(set(protected), records=sky_status_recs),
                str,
            )
            if sky_ok
            else []
        )
        sky_destroyed: list[str] = []
        if apply and sky_orphans:
            import sky

            for cl in sky_orphans:
                if not _approved("sky_orphans", cl):
                    continue
                try:
                    sky.get(sky.down(cl))
                    sky_destroyed.append(cl)
                except Exception as e:  # noqa: BLE001
                    _unconfirmed("sky_orphans", cl, e)

        # DO block-volume pass (best-effort): sky.down deletes the volume with its droplet, but a
        # partial teardown can leave a detached `lab-*` volume that the instance passes above can't
        # see (its droplet is gone) — yet it keeps billing. `volumes` was listed up front.
        from lab.backends.skypilot import do_volume_orphans as _find_do_volume_orphans

        do_volume_orphans: list[dict[str, Any]] = []
        do_volumes_destroyed: list[Any] = []
        if volumes is not None:
            do_volume_orphans = _ours_only(
                "do_volume_orphans",
                _find_do_volume_orphans(volumes, set(protected)),
                lambda v: str(v.get("name") or v.get("id")),
            )
            if apply and do_volume_orphans:
                from lab.backends.skypilot import delete_do_volumes

                doomed_vols = [
                    vol
                    for vol in do_volume_orphans
                    if vol.get("id") is not None and _approved("do_volume_orphans", vol)
                ]
                if doomed_vols:
                    # Through the detach ladder: the sky pass above has often just destroyed this
                    # volume's droplet, and a delete in that instant is refused for a volume DO
                    # releases seconds later — a false exit 5 (the 2026-08-23 race).
                    try:
                        gone, refused = delete_do_volumes([vol["id"] for vol in doomed_vols])
                    except Exception as e:  # noqa: BLE001 — no client: none of them were deleted
                        gone, refused = [], {vol["id"]: e for vol in doomed_vols}
                    do_volumes_destroyed.extend(gone)
                    for vol in doomed_vols:
                        if vol["id"] in refused:
                            _unconfirmed("do_volume_orphans", vol, refused[vol["id"]])

        # GCP pass (best-effort): out-of-band instance + unattached-disk sweep via the compute
        # API — `sky.status` only sees clusters SkyPilot still tracks, and a GCP persistent disk
        # that outlives its VM keeps billing (same failure mode as the DO volume leak). Skipped
        # when GCP isn't configured (no ADC); both listings were taken up front.
        from lab.backends.skypilot import delete_gcp_disk, delete_gcp_instance
        from lab.backends.skypilot import gcp_disk_orphans as _find_gcp_disk_orphans
        from lab.backends.skypilot import gcp_instance_orphans as _find_gcp_instance_orphans
        from lab.backends.skypilot import gcp_project, gcp_unmatched_lab_names

        gcp_orphans: list[dict[str, Any]] = []
        gcp_destroyed: list[str] = []
        gcp_disk_orphans: list[dict[str, Any]] = []
        gcp_disks_destroyed: list[str] = []

        if gcp_instances is not None:
            gcp_orphans = _ours_only(
                "gcp_orphans",
                _find_gcp_instance_orphans(gcp_instances, set(protected)),
                _gcp_identity,
            )
            if apply and gcp_orphans:
                for inst in gcp_orphans:
                    if not _approved("gcp_orphans", inst):
                        continue
                    try:
                        delete_gcp_instance(str(inst["name"]), str(inst["zone"]))
                        gcp_destroyed.append(str(inst["name"]))
                    except Exception as e:  # noqa: BLE001
                        _unconfirmed("gcp_orphans", inst, e)

        if gcp_disks is not None:
            gcp_disk_orphans = _ours_only(
                "gcp_disk_orphans",
                _find_gcp_disk_orphans(gcp_disks, set(protected)),
                _gcp_identity,
            )
            if apply and gcp_disk_orphans:
                for disk in gcp_disk_orphans:
                    if not _approved("gcp_disk_orphans", disk):
                        continue
                    try:
                        delete_gcp_disk(str(disk["name"]), str(disk["zone"]))
                        gcp_disks_destroyed.append(str(disk["name"]))
                    except Exception as e:  # noqa: BLE001
                        _unconfirmed("gcp_disk_orphans", disk, e)

        if apply and unsupervised:
            # Detection without remediation is the actual gap (F3). The dry run already *reports*
            # these; flipping one to terminal and attempting its teardown used to require someone
            # running `lab status` on that exact job id, which on an unattended box nobody does.
            # `--apply` is already the destructive, opted-in path, so finish the job here the same
            # way `SkyPilotBackend.status()` would. Best-effort: a crash must not abort the sweep,
            # because the report it produces is the thing the operator acts on.
            from lab.backends.skypilot import SkyPilotBackend

            for entry in unsupervised:
                if not _approved("unsupervised", entry):
                    continue  # not in the set the operator confirmed
                try:
                    SkyPilotBackend(home=self.home, repo=self.repo).status(entry["job_id"])
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[lab] reconcile could not finalise unsupervised job "
                        f"{entry['job_id']}: {e}",
                        file=sys.stderr,
                    )

        # A pass that could not list under `apply` destroyed nothing, and nothing it covers was
        # looked at: that is a cleanup this run cannot vouch for, the same verdict as a destroy
        # whose outcome is unreadable. Listed with them so the CLI's exit 5 covers it.
        for field, message in pass_errors.items():
            destroy_outcomes.append(
                {
                    "pass": field,
                    "resource": f"(entire {field} pass)",
                    "outcome": "unknown",
                    "detail": "the listing failed, so this pass examined and destroyed nothing; "
                    "anything it covers may still be billing — verify against the provider",
                    "error": message,
                }
            )

        return {
            "vast_pass": vast_pass,
            # "ran" or why it did not — a skewed client cannot read a destroy's result, so the
            # sky pass stands down while the provider-direct passes carry on.
            "sky_pass": sky_pass,
            "gcp_pass": gcp_pass,
            "gcp_disk_pass": gcp_disk_pass,
            # Which project the two passes above actually swept — ambient from ADC, and possibly
            # not the one SkyPilot launches into (GCP-LEAK-7).
            "gcp_project": gcp_project(),
            "gcp_unmatched": gcp_unmatched_lab_names(gcp_instances or [], gcp_disks or []),
            "instances_total": len(instances),
            "unsupervised": unsupervised,  # running manifests with a dead supervisor (FR-C2)
            "orphans": orphans,
            "destroyed": destroyed,
            "ghosts": ghosts,
            # Additive, not a shape change to `ghosts` (frozen MCP/CLI contract, see above):
            # cluster -> why it was flagged, e.g. "no matching SkyPilot-tracked cluster (cloud=do)".
            "ghost_reasons": ghost_reasons,
            "sky_orphans": sky_orphans,
            "sky_destroyed": sky_destroyed,
            "do_volume_orphans": do_volume_orphans,
            "do_volumes_destroyed": do_volumes_destroyed,
            "gcp_orphans": gcp_orphans,
            "gcp_destroyed": gcp_destroyed,
            "gcp_disk_orphans": gcp_disk_orphans,
            "gcp_disks_destroyed": gcp_disks_destroyed,
            # `lab-*` resources this project could not prove it owns. Never destroyed, and
            # deliberately NOT orphan fields: `--apply` cannot act on them, so exit 3's "re-run
            # with --apply" would be wrong advice (the reasoning `gcp_unmatched` already used).
            "other_projects": other_projects,
            "unattributed": unattributed,
            # Every destroy that did NOT confirm success, across all passes. Empty on a dry run
            # (nothing was attempted) and on a fully-confirmed cleanup. Non-empty means the cloud
            # is in a state this tool cannot vouch for — the CLI exits 5 and says to verify.
            "destroy_outcomes": destroy_outcomes,
            # Jobs whose manifest could not be read. Their machines were kept protected (never
            # orphaned), so a real leak behind one needs a human: fix or remove the manifest.
            "unreadable_manifests": unreadable_ids,
            # Additive: orphan field -> why its listing failed. Only ever non-empty under
            # `apply` (a dry run raises instead); each entry is also in `destroy_outcomes`.
            "pass_errors": pass_errors,
            "applied": apply,
        }

    def _settle_teardown(
        self,
        manifests: list[JobManifest],
        *,
        interval: float,
        attempts: int = 3,
        mirrored: Iterable[str] = (),
    ) -> list[JobManifest]:
        """Re-read manifests briefly so a teardown_status that's merely lagging (a job reports
        terminal a tick before its teardown is recorded) settles to its real value before we
        classify clean vs. leaked vs. unconfirmed. Only re-reads while some remote job still
        shows a null teardown.

        Goes through :meth:`_resolve_manifest`, so a mirror-only job settles too — a plain
        ``self.manifest`` read raises for one, which would have made this the second place in the
        wait path that a deferred job could not get through.

        The window depends on where the unsettled job's state comes from, because the two sources
        lag by wildly different amounts:

        * **local** (``attempts`` × ``min(interval, 5)`` = 15s by default): the supervisor writes
          teardown_status to the very file being re-read, seconds after terminal. Unchanged, and
          deliberately not subject to ``_MIRROR_MIN_INTERVAL_S`` — a 30s-a-go settle would add
          half a minute of latency to the exit of every ordinary remote wait.
        * **mirrored** (``_MIRROR_SETTLE_S``, sampled at the mirror floor): the value has to cross
          a scheduler tick first, which 15s cannot span. See ``_MIRROR_SETTLE_S`` for why this is
          the difference between a warning that means something and one that fires on every clean
          deferred run.

        ``mirrored`` names the ids known to have been read from the mirror (``wait`` already
        collected them, so choosing the window costs no extra object-store GET).

        A re-read that cannot resolve keeps the manifest it already has. This runs *after* the
        verdict is already computable, so a failed re-read can only leave it where it was —
        ``teardown_status: None`` stays "unconfirmed", which warns and never masquerades as clean.
        Raising instead would turn an object-store blip into an exit-1 "gave up on --timeout"
        *after* a completed wait: the same disguise :meth:`wait` refuses, and worse here because
        the answer is already in hand."""
        mirrored_ids = set(mirrored)

        def _unsettled(ms: list[JobManifest]) -> list[JobManifest]:
            return [
                m
                for m in ms
                if m.status in _TERMINAL_STATES  # only terminal jobs can settle; pending never will
                and m.backend.provisioner != "local"
                and m.teardown_status is None
            ]

        def _reread(m: JobManifest) -> JobManifest:
            try:
                return self._resolve_manifest(m.job_id)[0]
            except (MirrorReadError, FileNotFoundError) as e:
                events.note("wait.settle_unreadable", job_id=m.job_id, error=str(e))
                return m

        step, budget = min(interval, 5.0), attempts * min(interval, 5.0)
        if any(m.job_id in mirrored_ids for m in _unsettled(manifests)):
            step, budget = min(_MIRROR_MIN_INTERVAL_S, _MIRROR_SETTLE_S), _MIRROR_SETTLE_S
        spent = 0.0
        # Budgeted by intended sleep, not wall clock: the settle's cost is the sleeping.
        while _unsettled(manifests) and spent < budget:
            nap = min(step, budget - spent)
            time.sleep(nap)
            spent += nap
            manifests = [_reread(m) for m in manifests]
        return manifests

    def _wait_summary_dict(
        self, manifests: list[JobManifest], *, failed_fast: bool, mirrored: Iterable[str] = ()
    ) -> dict[str, Any]:
        """The FR-C2 verdict as data (one shape for CLI + MCP + done-file snapshots)."""
        all_terminal = all(m.status in _TERMINAL_STATES for m in manifests)
        pending = [m.job_id for m in manifests if m.status not in _TERMINAL_STATES]
        if failed_fast:
            # Offender-first ordering so a watcher's first glance lands on what died.
            manifests = sorted(
                manifests,
                key=lambda m: m.status not in (JobState.failed, JobState.timed_out),
            )
        teardown_leaks = [m.job_id for m in manifests if m.teardown_status == "failed"]
        # Distinct from both: the destroy ran and its outcome could not be read (R10). Anything
        # unrecognised lands here too rather than in the all-clear — a value this version does
        # not know about must never be optimistically read as "succeeded".
        teardown_unknown = [
            m.job_id
            for m in manifests
            if m.teardown_status is not None
            and m.teardown_status not in ("succeeded", "failed")
        ]
        teardown_unconfirmed = [
            m.job_id
            for m in manifests
            if m.status in _TERMINAL_STATES
            and m.backend.provisioner != "local"
            and m.teardown_status is None
        ]
        return {
            "all_terminal": all_terminal,
            "failed_fast": failed_fast,
            "pending": pending,  # still running — and, for remote jobs, still billing
            # Which of these jobs was read from the scheduler's queue mirror rather than this
            # project's own runs/ (a scheduler-launched job has no local record at all). Named
            # ids, not a bare flag, so a mixed wait says *which* rows can be a scheduler tick
            # stale — the same caveat `lab status`'s `mirrored` field carries.
            "mirrored": list(mirrored),
            "teardown_leaks": teardown_leaks,
            "teardown_unknown": teardown_unknown,
            "teardown_unconfirmed": teardown_unconfirmed,
            "jobs": [
                {
                    "job_id": m.job_id,
                    "state": m.status.value,
                    "exit_code": m.exit_code,
                    "teardown_status": m.teardown_status,
                }
                for m in manifests
            ],
        }

    def wait_summary(
        self,
        job_ids: list[str],
        *,
        interval: float = 10.0,
        timeout: float | None = None,
        fail_fast: bool = False,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """:meth:`wait`, then classify the FR-C2 verdict as data — shared by CLI ``lab wait``
        and the MCP ``wait`` tool so both surfaces expose the same leak signal.

        ``teardown_leaks`` non-empty means a paid machine may still be running;
        ``teardown_unconfirmed`` means a remote job's teardown never recorded either way (a null
        must not masquerade as clean — run ``lab reconcile`` to be sure). ``on_update`` receives
        a fresh summary snapshot after each job's terminal transition and once more with the
        final summary — the incremental done-file feed (field-report #3). ``fail_fast`` returns
        as soon as any job is failed/timed_out; no surviving job is ever cancelled.

        Scheduler-launched jobs are resolved from the queue mirror (:meth:`_resolve_manifest`)
        and named in the summary's ``mirrored`` list."""

        def _on_terminal(_m: JobManifest) -> None:
            if on_update is not None:
                ms, snapshot_mirrored = self._resolve_manifests(job_ids)
                on_update(
                    self._wait_summary_dict(ms, failed_fast=False, mirrored=snapshot_mirrored)
                )

        # Filled by `wait` from the polls it already makes, rather than re-resolved here: an
        # up-front pass of its own would spend an extra object-store GET per mirrored job (and
        # this method must not be the place that first learns whether an id exists — `wait`
        # already raises for an id in neither place).
        mirrored_seen: set[str] = set()
        manifests = self.wait(
            job_ids, interval=interval, timeout=timeout, fail_fast=fail_fast,
            on_terminal=_on_terminal, mirrored_out=mirrored_seen,
        )
        all_terminal = all(m.status in _TERMINAL_STATES for m in manifests)
        # "Tripped" = asked for, and some job failed/timed out — NOT "the wait was cut short".
        # Requiring a survivor made a lone failing job exit 0, and made a multi-job verdict depend
        # on whether the failures landed in the same poll. Survivors are still named in `pending`.
        failed_fast = fail_fast and any(
            m.status in (JobState.failed, JobState.timed_out) for m in manifests
        )
        if all_terminal or failed_fast:
            # Settle on the fail-fast path too: the offender's teardown_status may be merely
            # lagging, and a null must not hide a real leak verdict behind "unconfirmed". The
            # mirrored ids decide how long that lag is worth waiting out (a scheduler tick, not
            # a filesystem write) — see `_settle_teardown`.
            manifests = self._settle_teardown(
                manifests, interval=interval, mirrored=mirrored_seen
            )
        summary = self._wait_summary_dict(
            manifests,
            failed_fast=failed_fast,
            mirrored=[j for j in job_ids if j in mirrored_seen],  # in the caller's id order
        )
        if on_update is not None:
            try:
                on_update(summary)
            except Exception as e:  # noqa: BLE001 — a watcher crash must not eat the verdict
                # stderr: stdout carries only the summary JSON, which callers pipe into `jq`.
                print(f"[lab] wait on_update callback failed: {e}", file=sys.stderr)
        return summary

    def wait(
        self,
        job_ids: list[str],
        *,
        interval: float = 10.0,
        timeout: float | None = None,
        fail_fast: bool = False,
        on_terminal: Callable[[JobManifest], None] | None = None,
        mirrored_out: set[str] | None = None,
    ) -> list[JobManifest]:
        """Block until every job reaches a terminal state (or ``timeout``), then return manifests.

        Meant to run as a Claude Code background task: its completion is the push signal, so the
        agent need not poll (FR-G1). Uses cheap status reads (FR-G2); one
        :meth:`_resolve_manifest` per job per poll, which reads the local store for a job this
        machine supervises and the scheduler's queue mirror for a **scheduler-launched
        (deferred)** one — so this works for jobs of any backend, launched from anywhere.

        ``on_terminal`` fires once per job on its first observed terminal transition (errors are
        logged, never fatal). ``fail_fast`` returns immediately when any job reaches
        ``failed``/``timed_out`` — preempted (retryable) and cancelled (operator-initiated) do
        not trigger it. ``wait`` never mutates jobs: nothing is cancelled on the way out.

        When *every* job resolves from the mirror the poll interval is floored at
        ``_MIRROR_MIN_INTERVAL_S`` — the mirror cannot be fresher than the scheduler's tick, so
        polling it faster only spends object-store GETs. A mixed local+mirrored wait keeps the
        caller's interval.

        ``mirrored_out``, when given, collects the ids that resolved from the mirror. It is an
        out-parameter rather than part of the return value because the return type is contract
        (``confirm`` and several callers unpack manifests from it), and because re-deriving the
        answer afterwards would cost one more object-store GET per mirrored job for something
        the polls above already know.

        **A mirror that cannot answer mid-wait means "no news yet", not a failed wait.** The
        object store re-raises every non-``NoSuchKey`` error, so one 5xx or reset on one poll used
        to unwind the whole call — and ``lab wait``'s generic handler turns that into exit 1, the
        *documented code for "gave up on --timeout"*. Six hours of waiting would end
        indistinguishable from a timeout because of a network hiccup. The only bound such a poll
        needs is the caller's own ``timeout``, which is already exactly right; there is no retry
        budget here on purpose. The **first** poll stays strict about a job that resolves nowhere
        (``FileNotFoundError``), so an id neither the store nor the mirror knows still fails loudly
        instead of hanging until the deadline.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        pending = list(job_ids)
        effective_interval = interval
        cadence_decided = False
        first_poll = True
        last_seen: dict[str, JobManifest] = {}  # freshest manifest ever resolved, per job
        unreadable: set[str] = set()  # jobs currently degraded (warned about once each)
        while pending:
            still_pending: list[str] = []
            tripwire = False
            resolved = from_mirror = 0
            for j in pending:
                # One resolve per job per poll: `.status` off the manifest this returns is the
                # same state `self.status(j)` reports (it re-reads after the backend's own
                # finalizing write), so the extra manifest read the callback used to make is gone.
                try:
                    m, mirrored = self._resolve_manifest(j, refresh=True)
                except (MirrorReadError, FileNotFoundError) as e:
                    if isinstance(e, FileNotFoundError) and first_poll:
                        raise  # an id in neither place is a caller error, not a hiccup
                    # Either way this job is not in local runs/, so it counts as mirrored for the
                    # cadence decision below (`mirrored_out` is left alone: it names the ids whose
                    # *reported* data came from the mirror, and this poll reported none).
                    resolved += 1
                    from_mirror += 1
                    events.note("wait.mirror_unreadable", job_id=j, error=str(e))
                    if j not in unreadable:
                        unreadable.add(j)
                        print(
                            f"[lab] {j}: the scheduler's queue mirror could not be read "
                            f"({e}) — treating it as 'no news yet' and polling on. This wait's "
                            "own --timeout is the bound.",
                            file=sys.stderr,  # stdout carries only JSON, which callers parse
                        )
                    still_pending.append(j)
                    continue
                unreadable.discard(j)
                last_seen[j] = m
                resolved += 1
                from_mirror += int(mirrored)
                if mirrored and mirrored_out is not None:
                    mirrored_out.add(j)
                if m.status not in _TERMINAL_STATES:
                    still_pending.append(j)
                    continue
                if on_terminal is not None:
                    try:
                        on_terminal(m)
                    except Exception as e:  # noqa: BLE001 — callback must never abort the wait
                        # stderr: stdout carries only JSON, which callers parse.
                        print(f"[lab] wait on_terminal callback failed: {e}", file=sys.stderr)
                if fail_fast and m.status in (JobState.failed, JobState.timed_out):
                    tripwire = True
            first_poll = False
            if not cadence_decided:
                # Decided on the first poll, which sees every id in the wait (`pending` starts as
                # all of them) — not re-decided later, when a mixed wait's local jobs finishing
                # first would silently slow the survivors down.
                cadence_decided = True
                if resolved and from_mirror == resolved and interval < _MIRROR_MIN_INTERVAL_S:
                    effective_interval = _MIRROR_MIN_INTERVAL_S
                    print(
                        f"[lab] every job in this wait is read from the scheduler's mirror, "
                        f"which refreshes on the scheduler's tick — polling every "
                        f"{effective_interval:g}s instead of {interval:g}s. State may be a tick "
                        f"stale (`mirrored` in the summary names these jobs).",
                        file=sys.stderr,  # stdout carries only JSON, which callers parse
                    )
            pending = still_pending
            if tripwire or not pending:
                break
            if deadline is None:
                # guard against a busy-loop on interval<=0
                time.sleep(max(0.05, effective_interval))
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break  # timed out before all jobs finished
                # never overrun the deadline
                time.sleep(max(0.05, min(effective_interval, remaining)))
        out: list[JobManifest] = []
        for j in job_ids:
            try:
                m, mirrored = self._resolve_manifest(j)
            except (MirrorReadError, FileNotFoundError) as e:
                # Same rule as the loop: a blip on the closing read must not discard a completed
                # wait. Fall back to the freshest state this wait ever saw. Only a mirror that was
                # unreadable for the *entire* wait leaves nothing to fall back on — and the CLI
                # gate (`_mirror_knows`) already read it successfully moments earlier, so that is
                # a genuinely broken mirror and belongs loud.
                cached = last_seen.get(j)
                if cached is None:
                    raise
                events.note("wait.final_read_unreadable", job_id=j, error=str(e))
                out.append(cached)
                continue
            if mirrored and mirrored_out is not None:
                mirrored_out.add(j)
            out.append(m)
        return out


def build_backend(name: str, *, home: Path, repo: Path) -> Backend:
    """The single name->backend mapping. Both Lab construction paths (``default_lab``,
    ``Lab._sibling_lab``) and the scheduler (``Scheduler.make_lab``) route through here, so a new
    backend is wired in one place instead of three. An unknown name raises :class:`LabError`
    (:func:`validate_backend`) — it used to fall back to ``local``, which ran a mistyped remote
    job on this machine.
    """
    validate_backend(name)
    if name in ("skypilot", "cpu"):
        from lab.backends.skypilot import SkyPilotBackend  # optional extra; import lazily

        return SkyPilotBackend(home=home, repo=repo)
    return LocalBackend(home=home, repo=repo)


def job_status_view(home: Path, repo: Path, job_id: str) -> dict[str, Any]:
    """One status shape for both shells (FR-A2/FR-I2/FR-C2), with a mirrored-manifest fallback.

    Reads the local manifest and the live backend status (which finalizes dead-supervisor jobs);
    a job absent from local ``runs/`` falls back to the scheduler queue's mirrored manifest
    (spec §4.3) so deferred jobs are observable from every surface, not write-only. Raises
    :class:`FileNotFoundError` when the job exists in neither place.
    """
    store = JobStore(home)
    try:
        m = store.read_manifest(job_id)
    except FileNotFoundError:
        from lab.scheduler.queue import default_queue  # local import: avoids a module cycle

        mirrored = default_queue().read_mirrored(job_id)
        if mirrored is None:
            raise
        return _status_fields(
            mirrored,
            state=mirrored.status.value,
            mirrored=True,
            notes=lab_notes.count_for_job(job_id, home=home),
        )
    lab = Lab(backend=build_backend(m.backend.provisioner, home=home, repo=repo), repo=repo, home=home)
    state = lab.status(job_id)
    m = store.read_manifest(job_id)  # re-read: status may have just finalized/torn down the job
    return _status_fields(
        m,
        state=state.value,
        mirrored=False,
        logs_path=store.logs_path(job_id),
        runner_exit=store.read_runtime(job_id).get("runner_exit"),
        partials=store.read_runtime(job_id).get("partials"),
        notes=lab_notes.count_for_job(job_id, home=home),
    )


_TERMINAL_VALUES = frozenset(s.value for s in _TERMINAL_STATES)


def job_age_s(m: JobManifest, state: str) -> float | None:
    """Seconds this job has been alive — the arithmetic nobody should be doing by hand.

    On 2026-09 an orchestrator compared ``started_at`` against a *wrongly believed* current date,
    concluded two healthy jobs were 25-hour runaways and cancelled them: $2 and about two hours of
    good progress thrown away. The recorded lesson was "never do date arithmetic on ``started_at``
    by hand", so the tool does it: a caller reads a number instead of parsing a timestamp and
    subtracting it from a date it may believe wrongly.

    For a **terminal** job this is elapsed-to-terminal — its final lifetime, frozen. Not "seconds
    since it started": a job that *ended* 25 hours ago is not a 25-hour runaway, and a number that
    kept climbing after the job landed would rebuild the exact illusion this field exists to
    prevent. "Is this a runaway?" is ``state == "running" and age_s > X``, and ``state`` is right
    there in the same view.

    ``None`` when there is no honest answer: before the job starts (nothing has begun), and for a
    terminal job whose ``ended_at`` was never recorded (a supervisor that died before writing one)
    — the manifest genuinely does not say when it stopped, and measuring that against ``now``
    would grow forever.

    Needs no marker of its own on a **mirrored** (scheduler-launched) manifest. ``started_at`` is
    an absolute UTC instant, so a mirrored running job's age is computed against local ``now`` and
    is not itself a tick stale; what can be stale is whether it is still running, which is what
    the view's existing ``mirrored`` flag already qualifies.
    """
    if m.started_at is None:
        return None
    end = m.ended_at if state in _TERMINAL_VALUES else now()
    seconds = duration_seconds(m.started_at, end)
    return None if seconds is None else round(seconds, 3)


def _estimated_running_usd(m: JobManifest, state: str) -> float | None:
    """Burn-so-far for a live job (field-report #7): a derived estimate, never a meter.

    Bounded by the same ceiling :func:`pricing.realized_spend` uses, for the same reason. Roughly
    40% of DigitalOcean supervisors die silently, leaving a manifest at ``running`` forever; an
    unbounded estimate then climbs on every read over money nobody spent, which is precisely the
    wrong-most-of-the-time alarm this codebase keeps having to undo (R10). The box enforces its own
    ``--timeout``, so a job cannot bill past it and stay honest.
    """
    if state != "running" or m.cost is None or m.started_at is None:
        return None
    elapsed = duration_seconds(m.started_at, now())
    if elapsed is None:
        return None
    return actual_cost(m.cost.hourly_usd, min(elapsed, pricing.live_ceiling_s(m)))


def _status_fields(
    m: JobManifest,
    *,
    state: str,
    mirrored: bool,
    logs_path: Path | None = None,
    runner_exit: dict[str, Any] | None = None,
    partials: dict[str, Any] | None = None,
    notes: int = 0,
) -> dict[str, Any]:
    last_line, last_at = tail_last_line(logs_path) if logs_path is not None else (None, None)
    return {
        "job_id": m.job_id,
        "state": state,
        "lab_version": m.lab_version,  # which lab produced the run (None on pre-0.5.0 manifests)
        "started_at": m.started_at,
        "ended_at": m.ended_at,
        "exit_code": m.exit_code,
        "end_reason": m.end_reason,
        "cost": m.cost.model_dump() if m.cost else None,
        # Burn-rate visibility mid-run (field-report #7): a derived estimate, not a meter. Bounded
        # by the same ceiling `pricing.realized_spend` uses, and for the same reason: ~40% of DO
        # supervisors die silently, leaving a manifest at `running` forever, and an unbounded
        # estimate then climbs on every read over money nobody spent. The box enforces its own
        # `--timeout`, so a job cannot bill past it and stay honest.
        "estimated_running_usd": _estimated_running_usd(m, state),
        # Terminal without ever really running (never priced / transient launch error / destroyed
        # over its price cap). A failed launch costs a slot and no money; counting it as an
        # ordinary failure corrupts every success-rate and cost number a campaign computes, which
        # is why an operator was re-deriving it by hand inside a shell watcher. `cost: null` alone
        # is NOT the test — see `pricing.failed_launch_reason` for the cost-unknown boundary.
        "is_failed_launch": pricing.is_failed_launch(m),
        "failed_launch_reason": pricing.failed_launch_reason(m),
        # Seconds alive, computed here so nobody subtracts `started_at` from a date they believe
        # wrongly (that mistake cancelled two healthy jobs). Frozen at its final lifetime once
        # terminal; None when there is no honest answer.
        "age_s": job_age_s(m, state),
        "teardown_status": m.teardown_status,  # FR-C2 — "failed" means a box may still bill
        # How the supervisor actually died, when that could be observed (F5): `{source,
        # returncode, signal, detail}`. `source: "disappeared"` is a real answer — the kernel no
        # longer knows — and deliberately distinct from this field being absent, which means
        # nobody looked. A SIGKILL here is the difference between "the OOM killer took it" and
        # "the code crashed", which is otherwise unrecoverable after the fact.
        "runner_exit": runner_exit,
        # Whether partial results are actually being retrieved mid-run (2026-08-20: four jobs
        # finished with empty output dirs and nothing said so). Distinguishes delivering
        # (`delivered`, growing `files_total`) from reachable-but-empty (`ok` climbing,
        # `files_total: 0`) from unreachable (`consecutive_failures` climbing with a reason) —
        # three states that used to look identical, namely silent.
        "partials": partials,
        # Surfaced here because `status` is where people actually look: a note filed on a job and
        # visible only in a digest nobody runs is a note nobody reads (2026-08-26 ledger review).
        "notes": notes,
        # Where it actually landed. GCP prices and exhausts per zone, so "which zone" is the
        # difference between a $0.034/hr job and a $0.12/hr one.
        # A local job ran on this machine, on no cloud: the "vast" default is SkyPilot's, and only
        # means something for a job a cloud provisioner launched (every local job used to report
        # `cloud: "vast"`, as did a retried remote shard that ran locally carrying `cloud=do`).
        "placement": {
            "provisioner": m.backend.provisioner,
            "cloud": None if m.backend.provisioner == "local" else (m.resources.cloud or "vast"),
            "machine_type": m.backend.machine_type,
            "region": m.backend.region,
            "zone": m.backend.zone,
            "spot": m.backend.launched_spot,
        },
        # spot_fallback defaults on, so `--spot` can land on-demand at ~5x the price the user was
        # budgeting for. `launched_spot` recorded that all along; nothing ever surfaced it.
        "spot_downgraded": bool(m.resources.use_spot and m.backend.launched_spot is False),
        "sweep_id": m.sweep_id,
        # Progressing-vs-wedged signal for long runs that don't log metrics.
        "last_log_line": last_line,
        "last_log_at": last_at,
        "code": {
            "git_commit": m.code.git_commit,
            "git_dirty": m.code.git_dirty,
            "diff_ref": m.code.diff_ref,
        },
        "mirrored": mirrored,  # True = read from the scheduler mirror; may be a tick stale
    }


def default_lab(home: Path | None = None, backend: str = "local") -> Lab:
    """Construct a Lab rooted at the current git repo, over the named backend
    (``local`` or ``skypilot``). Shared by the CLI and MCP so both drive the identical core.
    """
    repo = repo_root()
    resolved_home = Path(home) if home else repo / "runs"
    return Lab(backend=build_backend(backend, home=resolved_home, repo=repo), repo=repo, home=resolved_home)
