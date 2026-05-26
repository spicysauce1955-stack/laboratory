# Laboratory — Remote Experiment Runner: Requirements

**Status:** draft v0.1 · **Owner:** research workspace · **Date:** 2026-05-26

A spec for a general system that runs computational experiments on remote machines,
decoupled from the local Claude Code session, so we can (a) submit heavy jobs and keep
working, (b) watch results *while they run* and kill early if we're off track, and (c) in the
worst case just wait — with results coming back reproducibly. It must serve **all**
experiments — the current tempotron-capacity work and whatever comes next — so the core is
deliberately experiment-agnostic.

Requirement keywords follow RFC 2119 (**MUST** / **SHOULD** / **MAY**). Each requirement has an
ID and a phase tag (**[P0]** MVP, **[P1]** next, **[P2]** future).

---

## 1. Purpose & scope

The lab turns "run this experiment" into a **reproducible remote job** with a **structured,
agent-usable interface** and **live observability**. It is *not* a general workflow
orchestrator, a data lake, or a model-serving system (see §13 Non-goals).

## 2. Users

Two first-class users; the interface MUST serve both:
- **The researcher (human)** — submits/monitors via CLI and/or a web dashboard.
- **The AI agent (Claude Code)** — submits/monitors via an **MCP server** with structured
  returns. Agent ergonomics are a design driver, not an afterthought (§5F, §6).

## 3. Definitions

- **Experiment** — a committed, runnable entrypoint determined entirely by config + seed.
- **Job** — one execution of an experiment at a pinned code ref with a resolved config/seed/resources.
- **Run** — synonym for a single job execution; produces metrics + artifacts.
- **Sweep** — a set of jobs over a parameter grid (e.g. one job per `K`).
- **Manifest** — the machine-readable record that makes a job regenerable (§8).
- **Artifact** — any output file (figure, table, checkpoint, log).
- **Metric** — a named scalar logged over a step/time axis during a run.

## 4. Design principles

1. **Experiment-agnostic.** The lab knows about *jobs*, not about tempotrons. Any script
   honoring the Experiment Contract (§7) runs unchanged.
2. **Reproducible by construction.** A result is meaningless without a manifest that
   regenerates it (commit + lock + config + seed). No "it ran once on the remote."
3. **Agent-first interface.** Structured tool calls over text parsing; cheap status/metrics
   queries; fail-loud errors.
4. **Observable live.** Progress (logs + incremental metrics) is queryable *during* the run.
5. **Cost-bounded.** Every job has a timeout and auto-teardown; nothing runs unbounded.
6. **Extensible.** Provisioner, tracker, and storage are pluggable backends (§13 Open).

## 5. Functional requirements

### 5A. Job lifecycle
- **FR-A1 [P0]** The system MUST accept a job submission specifying: code ref, entrypoint
  command, (optional) config, (optional) seed, and resource request; and return a `job_id`
  **without blocking** (submission returns in seconds).
- **FR-A2 [P0]** A job MUST move through observable states: `queued → running → {succeeded,
  failed, cancelled, timed_out}`.
- **FR-A3 [P0]** The system MUST let a caller `cancel` a running or queued job.
- **FR-A4 [P0]** A job's terminal state MUST record the process exit code and end reason.
- **FR-A5 [P1]** The system SHOULD support a `sweep`: submit N jobs over a parameter grid,
  tracked under one `sweep_id`, each independently monitorable.

### 5B. Reproducibility & provenance
- **FR-B1 [P0]** Each job MUST run a **pinned git commit** (not the live working tree). If the
  working tree is dirty, the system MUST either refuse or snapshot+record the diff.
- **FR-B2 [P0]** The remote environment MUST be built from the committed lockfile (`uv.lock`)
  so dependency versions match local (incl. the NumPy `<2` pin).
- **FR-B3 [P0]** Each job MUST write a **manifest** (§8) capturing everything needed to replay
  it: commit, lock hash, resolved config, seed, command, resources, timestamps, backend.
- **FR-B4 [P0]** The seed MUST be explicit and recorded; given the same manifest, a re-run MUST
  reproduce results within a documented tolerance.
- **FR-B5 [P1]** The system SHOULD detect an identical prior job (same commit+config+seed) and
  offer its cached result instead of recomputing.

### 5C. Execution & provisioning
- **FR-C1 [P0]** The system MUST provision a remote machine matching a resource request
  (cores, memory, optionally GPU, wall-clock limit), set up the env (FR-B2), run the
  entrypoint, and capture stdout/stderr.
- **FR-C2 [P0]** On completion, timeout, failure, or cancel, the system MUST **tear down** the
  remote machine automatically (no orphaned cost).
- **FR-C3 [P1]** The system SHOULD support GPU machines and (P2) multi-node/distributed runs.
- **FR-C4 [P0]** The system MUST expose to the running job a standard set of environment values
  (run id, output dir, seed, metrics endpoint) per the Experiment Contract (§7).

### 5D. Live observability
- **FR-D1 [P0]** A caller MUST be able to stream/tail a running job's **logs**.
- **FR-D2 [P1]** A caller MUST be able to query a running job's **incremental metrics** (named
  series over steps), updated within a small bounded delay of emission. *(This is the
  "are we on the right track?" capability — highest research value.)*
- **FR-D3 [P1]** The system SHOULD provide a human dashboard (or integrate one, e.g. the
  tracker's UI) showing live metrics and job status.

### 5E. Artifacts
- **FR-E1 [P0]** All job outputs MUST be collected from the run output directory and made
  retrievable.
- **FR-E2 [P0]** A caller MUST be able to **fetch artifacts back into the local repo** under
  `runs/<job_id>/`, alongside the manifest. `runs/` stays git-ignored.
- **FR-E3 [P0]** Artifacts MUST be in agent-readable formats: figures as PNG/PDF, tabular data
  as CSV/JSON/Parquet, plus raw logs. Each artifact carries a checksum.

### 5F. Interfaces
- **FR-F1 [P0]** The system MUST expose an **MCP server** with structured tools (§9):
  `submit`, `status`, `logs`, `fetch_artifacts`, `cancel`, `list`; and **[P1]** `metrics`,
  `sweep`. Returns MUST be structured (JSON), not free text.
- **FR-F2 [P0]** The system SHOULD also expose an equivalent **CLI** for the human.
- **FR-F3 [P0]** Errors MUST be explicit and actionable (fail-loud): a failed job surfaces the
  error and the tail of the log, never a silent partial success.

### 5G. Notifications
- **FR-G1 [P1]** On a job reaching a terminal state, the system SHOULD **push** a signal the
  Claude Code session can act on (e.g. a sentinel that triggers a background-task
  notification), so the agent need not poll.
- **FR-G2 [P0]** Absent push, `status` MUST be cheap enough to poll on an interval matched to
  the run length.

### 5H. Concurrency & scheduling
- **FR-H1 [P0]** The system MUST support multiple concurrent jobs and list them.
- **FR-H2 [P1]** The system SHOULD queue jobs when capacity is limited and report queue position.
- **FR-H3 [P2]** The system MAY support priorities and per-user/project quotas.

### 5I. Cost & lifecycle control
- **FR-I1 [P0]** Every job MUST carry a wall-clock timeout after which it is terminated and
  torn down (FR-C2).
- **FR-I2 [P1]** The system SHOULD report (estimated and actual) cost/compute per job.
- **FR-I3 [P1]** The system SHOULD support spot/preemptible instances with auto-resume or
  clean failure.

### 5J. Security
- **FR-J1 [P0]** Credentials/secrets (cloud, tracker API keys) MUST NOT be stored in the repo,
  manifests, logs, or artifacts.
- **FR-J2 [P0]** Each job MUST run with least-privilege access to only what it needs.
- **FR-J3 [P1]** Remote machines SHOULD be isolated per job and destroyed after use.

## 6. Non-functional requirements

- **NFR-1 Reproducibility:** any reported number is regenerable from its manifest (cf. FR-B*).
- **NFR-2 Reliability:** a submitted job survives local session disconnect/restart; status and
  artifacts remain retrievable afterward.
- **NFR-3 Agent-usability:** status/metrics queries are low-latency and low-token (structured,
  paginated/summarizable); the agent can run its whole loop — submit, monitor, decide, fetch —
  via tool calls only.
- **NFR-4 Portability:** cloud-agnostic; a local backend (run on this machine) MUST exist as a
  fallback for small jobs and offline dev.
- **NFR-5 Scalability:** from a single multicore node up to GPU/multi-node, without changing the
  experiment or the interface.
- **NFR-6 Auditability:** logs, manifests, and run history are retained and queryable.
- **NFR-7 Cost-safety:** no path leaves a machine running unbounded.

## 7. The Experiment Contract

The generality hinge: a script is "lab-compatible" iff it obeys this thin contract (no lab
coupling required).

- **EC-1** It is a committed, runnable entrypoint (e.g. `uv run python <script>` or a Hydra app)
  fully determined by an explicit **config + seed** — no hidden/global state.
- **EC-2** It reads its config (CLI/Hydra/env) and a seed provided by the lab.
- **EC-3** It writes **all** outputs under the directory given in `$LAB_RUN_DIR`.
- **EC-4** It logs **incremental metrics** through the standard tracker interface
  (`log_metric(name, value, step)`), so progress is observable live (FR-D2).
- **EC-5** It exits **non-zero on failure** (fail-loud).
- **EC-6** It MAY declare resource hints (cores/GPU/mem/time) in its config.

The lab provides, per run, at least: `$LAB_RUN_ID`, `$LAB_RUN_DIR`, the seed, and the metrics
endpoint. Our current stack already fits: Python via `uv`, config via Hydra+Pydantic, metrics
via MLflow/W&B, outputs to `runs/`.

## 8. Data model

**Job manifest** (one JSON per job; the reproducibility contract):
```
job_id, sweep_id?, created_at, submitted_by (human|agent),
code: { git_commit, git_dirty, diff_ref? },
env:  { uv_lock_sha256, python_version },
run:  { entrypoint_command, resolved_config, seed },
resources: { cpus, gpus, memory, timeout },
backend: { provisioner, machine_type, region? },
status, started_at, ended_at, exit_code, end_reason,
metrics_uri, logs_uri, artifacts: [ {name, type, path, sha256, bytes} ]
```
**Metric record:** `{ run_id, name, value, step, wall_time }`.
**Artifact types:** `figure | table | checkpoint | log | other`.

## 9. Interface contract (MCP tools)

Inputs/outputs are JSON. Names indicative.
- `submit(code_ref, command, config?, seed?, resources?) → { job_id, status }`
- `status(job_id) → { state, progress?, started_at?, eta?, resource_usage? }`
- `logs(job_id, tail?, since?) → { lines[] }`
- `metrics(job_id, names?, since_step?) → { series: {name: [{step, value, wall_time}]} }`  **[P1]**
- `fetch_artifacts(job_id, dest?) → { local_paths[] }`  (writes under `runs/<job_id>/`)
- `cancel(job_id) → { state }`
- `list(filter?) → { jobs: [{job_id, state, created_at, ...}] }`
- `sweep(base_config, grid, resources?) → { sweep_id, job_ids[] }`  **[P1]**

## 10. Use-case scenarios

1. **Submit & continue (agent).** Agent `submit`s a capacity sweep, gets `sweep_id`, keeps
   working on the writeup; on push-notify (FR-G1) it `fetch_artifacts` and reports.
2. **Live early-kill (agent).** Mid-run, agent polls `metrics`; the curve clearly isn't tracking
   theory → `cancel`, saving hours. (Directly the capability the user asked for.)
3. **Reproduce a past result.** Re-`submit` from an old manifest → same numbers (NFR-1).
4. **Human inspection.** Researcher opens the dashboard, watches live metrics, cancels/launches.

## 11. Acceptance criteria

- **AC-1** A committed, seeded experiment run remotely reproduces the local result within
  tolerance, and `runs/<id>/` ends up with the manifest + artifacts.
- **AC-2** `submit` returns a `job_id` in < ~5 s and does not block the session.
- **AC-3 [P1]** While running, `metrics` returns partial series updated within a bounded delay.
- **AC-4** On terminal state the agent learns of it (push or cheap poll) and can fetch artifacts.
- **AC-5** `cancel` and timeout both terminate the job **and** tear down the machine.
- **AC-6** Every finished job has a manifest from which the run is regenerable.
- **AC-7** No secret appears in repo, manifest, logs, or artifacts.

## 12. Phasing / roadmap

- **P0 — MVP (unblocks us):** reproducible `submit`/run (commit + `uv.lock`), `status`, `logs`,
  `fetch_artifacts` → `runs/`, `cancel`, timeout + auto-teardown, manifest, MCP + CLI, single
  multicore node, local fallback backend.
- **P1 — Observe & scale:** live `metrics` + early-kill, push notifications, `sweep`, GPU,
  cost reporting, result caching, dashboard.
- **P2 — Future:** multi-node/distributed, spot + auto-resume, artifact registry/versioning,
  priorities/quotas.

## 13. Open decisions (need a call before building)

1. **Provisioner:** SkyPilot (multi-cloud, auto-teardown, spot) · Modal (serverless Python,
   great live logs) · managed cluster (Slurm/cloud batch).
2. **Tracker / live metrics:** W&B (hosted, best live/queryable API) · MLflow (self-host,
   already declared in our stack) · Aim.
3. **Completion signaling:** push/sentinel integrated with Claude Code background-tasks vs poll.
4. **Artifact storage:** object store (S3/GCS) vs direct rsync into `runs/`.
5. **Compute shape:** single fat multicore node (our jobs are embarrassingly parallel over
   seeds/α/K) vs cluster — recommend single node first, sweeps as job-per-point.
6. **Auth/secrets model:** where keys live and how the remote gets them.

## 14. Non-goals & assumptions

- **Non-goals:** general DAG orchestration (Airflow/Dagster), data lake/warehouse, model
  serving, tying the core to any one experiment domain.
- **Assumptions:** Python via `uv` with `uv.lock` pinning; git-based code pinning; config via
  Hydra+Pydantic; results in git-ignored `runs/`; compute is available on request; remote env
  may have its own network-egress limits (cf. the blocked-host issues we already hit locally).
