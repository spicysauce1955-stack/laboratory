"""Structured data model (spec §8). These Pydantic models are the typed returns the MCP tools
emit (FR-F1) and the on-disk manifest format (FR-B3)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

ArtifactType = Literal["figure", "table", "checkpoint", "log", "other"]
Submitter = Literal["human", "agent"]


class JobState(str, Enum):
    """Observable lifecycle states (FR-A2)."""

    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    timed_out = "timed_out"
    preempted = "preempted"  # spot instance reclaimed mid-run (retryable, not a failure)


class ResourceRequest(BaseModel):
    cpus: int | None = None
    gpus: int | None = None
    memory: str | None = None  # e.g. "32GB"
    disk_size: int | None = None  # boot/attached volume size in GB (skypilot; DO volume size)
    accelerators: str | None = None  # SkyPilot accelerator spec, sky-catalog name e.g. "RTX4090:1" (remote)
    cloud: str | None = None  # SkyPilot cloud: "vast" (default) | "do" | "gcp"; None -> "vast"
    region: str | None = None  # pin the cloud region, e.g. "europe-west1"; None -> optimizer picks
    zone: str | None = None  # pin the zone, e.g. "europe-west1-b"; implies its region
    # Ceiling on the compute $/hr SkyPilot may accept (sky.Resources(max_hourly_cost=...)). Unlike
    # a price *trigger* (Vast-only, a wait-until condition), this is enforced by the optimizer at
    # launch, so the worst case it implies is provable rather than estimated.
    max_hourly_usd: float | None = None
    # Opt-in: destroy the machine rather than let it bill above `max_hourly_usd`. Off by default
    # because this project's rule is admission-control and stop-launching, never kill — a running
    # job vanishing over price has to be something the user asked for.
    price_cap_strict: bool = False
    timeout: str | None = None  # wall-clock limit, e.g. "2h" (FR-I1)
    provision_timeout: str | None = None  # max time to reach UP, e.g. "10m" (per-cloud default)
    use_spot: bool = False  # opt into spot/interruptible instances (skypilot)
    spot_fallback: bool = True  # if spot capacity is unavailable, fall back to on-demand


class CodeRef(BaseModel):
    git_commit: str
    git_dirty: bool = False
    diff_ref: str | None = None  # blob ref of the snapshotted diff if dirty (FR-B1)

    def assert_fail_closed(self) -> None:
        """The fail-closed provenance invariant (FR-B1): a job's code state must always be
        reconstructable from its manifest. Enforced at the store write path, NOT on load, so
        legacy Gap-B manifests still read."""
        if not self.git_commit:
            raise ValueError("CodeRef.git_commit must be a non-null commit SHA (FR-B1)")
        if self.git_dirty and self.diff_ref is None:
            raise ValueError(
                "CodeRef is dirty but diff_ref is None — a dirty run must capture its diff so "
                "the exact code state is reconstructable (FR-B1, Gap B)"
            )


class EnvInfo(BaseModel):
    uv_lock_sha256: str  # FR-B2
    python_version: str


class RunSpec(BaseModel):
    entrypoint_command: str
    resolved_config: dict[str, Any] = Field(default_factory=dict)
    seed: int  # explicit + recorded (FR-B4)
    # Persisted on the manifest so the store's succeeded-transition audit can read it (the
    # unconsumed-config fail-closed check; see JobStore.update_manifest).
    allow_unknown_config: bool = False


class BackendInfo(BaseModel):
    provisioner: str  # "local" | "skypilot" | ...
    machine_type: str | None = None
    region: str | None = None
    zone: str | None = None  # the zone actually launched into (GCP prices/exhausts per zone)
    launched_spot: bool | None = None  # actual kind launched (None = local/on-demand-only)


class CostInfo(BaseModel):
    """Per-job cost/compute (FR-I2): an up-front estimate plus the actual. hourly/estimated/actual
    are 0 for the local backend (own machine). For remote jobs ``duration_seconds`` is billed
    wall-clock (includes provisioning/setup, which clouds charge for).

    ``hourly_usd`` is the **total** billed rate — compute plus attached storage. It was
    compute-only until 2026-08-11, which understated GCP jobs materially: a default 256 GB disk is
    $0.028-$0.035/hr depending on disk type, against a $0.034/hr spot n4-standard-4. Keeping the
    total in the field everything already reads (``estimated_usd``, admission control, the
    dashboard) means those consumers got the storage line for free; the breakdown is for humans.
    """

    duration_seconds: float | None = None
    hourly_usd: float | None = None  # total: compute + storage
    compute_hourly_usd: float | None = None  # instance (+ accelerators)
    # The cap the user asked for, and whether the rental honoured it. Optional so manifests from
    # older releases still read. `None` means "not checked" — no cap set, or no price available —
    # and is deliberately distinct from `False`, "we looked and it was fine".
    cap_hourly_usd: float | None = None
    over_cap: bool | None = None
    storage_hourly_usd: float | None = None  # boot/attached disk
    hourly_basis: str | None = None  # provenance, e.g. "gcp catalog n4-standard-4 spot ..."
    estimated_usd: float | None = None  # hourly x wall-clock budget, known at launch
    actual_usd: float | None = None


class ArtifactRecord(BaseModel):
    name: str
    type: ArtifactType = "other"
    path: str
    sha256: str  # FR-E3
    bytes: int


class JobSpec(BaseModel):
    """Input to ``submit`` (FR-A1 / MCP §9). ``code_ref`` is resolved to a commit."""

    code_ref: str = "HEAD"
    command: str
    config: dict[str, Any] | None = None
    seed: int | None = None
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    submitted_by: Submitter = "agent"
    allow_unknown_config: bool = False  # opt out of the unconsumed-config fail-closed check


class JobManifest(BaseModel):
    """The reproducibility contract — one JSON per job (spec §8). Regenerates the run from
    commit + lock + config + seed (NFR-1)."""

    job_id: str
    sweep_id: str | None = None
    cell_id: str | None = None  # sharded-sweep cell grouping (P1-2); None for non-sharded jobs
    registration_id: str | None = None  # set when launched by the scheduler (spec §4.5 repair)
    confirms: str | None = None  # the run-id this job was launched to re-derive (lab confirm)
    # Which lab produced this run. Stamped at JobStore.create (the single creation chokepoint),
    # never defaulted on read: a manifest written by v0.4.0 has no such key and must stay None
    # rather than claim the version that happens to be reading it.
    lab_version: str | None = None
    created_at: datetime
    submitted_by: Submitter
    code: CodeRef
    env: EnvInfo
    run: RunSpec
    resources: ResourceRequest
    backend: BackendInfo
    status: JobState
    started_at: datetime | None = None
    ended_at: datetime | None = None
    exit_code: int | None = None
    end_reason: str | None = None
    cost: CostInfo | None = None  # FR-I2
    # FR-C2 leak detection. "succeeded" = the machine is confirmed gone. "failed" = the
    # destroy was definitively refused, a real leak, `lab wait` exits 3. "unknown" = the
    # outcome could not be read and nothing could verify it — exits 6, go check the
    # provider. None = no teardown was ever recorded. Treat an unrecognised value as
    # "unknown", never as "succeeded".
    teardown_status: str | None = None
    metrics_uri: str | None = None
    logs_uri: str | None = None
    artifacts_uri: str | None = None  # durable object-store prefix, e.g. r2://lab-artifacts/<id>
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    final_metrics: dict[str, float] = Field(default_factory=dict)  # last value per series (FR-B4)
    # Config-consumption handshake (field-report #1/#6): what the entrypoint reported it actually
    # consumed (None = never reported, ≠ {} = reported empty), and the submitted keys it ignored.
    config_effective: dict[str, Any] | None = None
    unconsumed_config: list[str] = Field(default_factory=list)


class SweepCell(BaseModel):
    """One non-seed grid point of a sharded sweep, plus its seed-shard bookkeeping (P1-2).

    The authoritative cell->shards map. ``seeds_present``/``missing_seeds``/``status`` are filled by
    aggregation (pending until then); the shard job manifests stay the fail-closed source of truth
    for code/seed state — this record is grouping + accounting, never a provenance substitute.
    """

    coords: dict[str, Any]
    cell_id: str
    seeds_expected: list[int]
    shard_seeds: list[list[int]]
    shard_job_ids: list[str]
    results_file: str
    seed_column: str
    aggregate_ref: str
    # Columns identifying a result row; None = [seed_column] (one row per seed). Experiments
    # sweeping an axis inside the job (one row per (seed, alpha)) set e.g. ["seed", "alpha"].
    row_key: list[str] | None = None
    seeds_present: list[int] = Field(default_factory=list)
    # Seeds whose rows came only from non-succeeded shards (partial recovery, field-report #2).
    seeds_partial: list[int] = Field(default_factory=list)
    missing_seeds: list[int] = Field(default_factory=list)
    status: Literal["pending", "complete", "incomplete"] = "pending"


class SweepPlan(BaseModel):
    """Persisted plan for a sharded sweep (P1-2), keyed by ``sweep_id`` under the lab home."""

    sweep_id: str
    created_at: datetime
    command: str  # base entrypoint, so retry_sweep can rebuild a shard spec without a manifest re-parse
    seed_axis_key: str  # config-override key carrying each shard's seed subset (default "seeds")
    cells: list[SweepCell]

    def view(self) -> dict[str, Any]:
        """Structured cell view for sharded sweeps (used by CLI + MCP shells — single source of truth)."""
        return {
            "sweep_id": self.sweep_id,
            "cells": [
                {
                    "coords": c.coords,
                    "cell_id": c.cell_id,
                    "shard_job_ids": c.shard_job_ids,
                    "aggregate_ref": c.aggregate_ref,
                    "seeds_expected": len(c.seeds_expected),
                    "seeds_present": len(c.seeds_present),
                    "seeds_partial": c.seeds_partial,  # full list — actionable, like missing
                    "missing_seeds": c.missing_seeds,
                    "status": c.status,
                }
                for c in self.cells
            ],
        }
