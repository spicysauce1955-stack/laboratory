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


class ResourceRequest(BaseModel):
    cpus: int | None = None
    gpus: int | None = None
    memory: str | None = None  # e.g. "32GB"
    accelerators: str | None = None  # SkyPilot accelerator spec, e.g. "RTX_3070:1" (remote)
    timeout: str | None = None  # wall-clock limit, e.g. "2h" (FR-I1)
    provision_timeout: str | None = None  # max time to reach UP, e.g. "10m" (default 8m; skypilot)
    use_spot: bool = False  # opt into spot/interruptible instances (skypilot)
    spot_fallback: bool = True  # if spot capacity is unavailable, fall back to on-demand


class CodeRef(BaseModel):
    git_commit: str
    git_dirty: bool = False
    diff_ref: str | None = None  # blob ref of the snapshotted diff if dirty (FR-B1)


class EnvInfo(BaseModel):
    uv_lock_sha256: str  # FR-B2
    python_version: str


class RunSpec(BaseModel):
    entrypoint_command: str
    resolved_config: dict[str, Any] = Field(default_factory=dict)
    seed: int  # explicit + recorded (FR-B4)


class BackendInfo(BaseModel):
    provisioner: str  # "local" | "skypilot" | ...
    machine_type: str | None = None
    region: str | None = None
    launched_spot: bool | None = None  # which kind actually launched (None for local/on-demand-only)


class CostInfo(BaseModel):
    """Per-job cost/compute (FR-I2): an up-front estimate plus the actual. hourly/estimated/actual
    are 0 for the local backend (own machine). For remote jobs ``duration_seconds`` is billed
    wall-clock (includes provisioning/setup, which clouds charge for)."""

    duration_seconds: float | None = None
    hourly_usd: float | None = None
    estimated_usd: float | None = None  # hourly x wall-clock budget, known at launch
    actual_usd: float | None = None


class ArtifactRecord(BaseModel):
    name: str
    type: ArtifactType = "other"
    path: str
    sha256: str  # FR-E3
    bytes: int


class MetricRecord(BaseModel):
    """A single point of a named metric series (FR-D2)."""

    run_id: str
    name: str
    value: float
    step: int
    wall_time: float


class JobSpec(BaseModel):
    """Input to ``submit`` (FR-A1 / MCP §9). ``code_ref`` is resolved to a commit."""

    code_ref: str = "HEAD"
    command: str
    config: dict[str, Any] | None = None
    seed: int | None = None
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    submitted_by: Submitter = "agent"


class JobManifest(BaseModel):
    """The reproducibility contract — one JSON per job (spec §8). Regenerates the run from
    commit + lock + config + seed (NFR-1)."""

    job_id: str
    sweep_id: str | None = None
    registration_id: str | None = None  # set when launched by the scheduler (spec §4.5 repair)
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
    teardown_status: str | None = None  # "succeeded" | "failed" | None — FR-C2 leak detection
    metrics_uri: str | None = None
    logs_uri: str | None = None
    artifacts_uri: str | None = None  # durable object-store prefix, e.g. r2://lab-artifacts/<id>
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
