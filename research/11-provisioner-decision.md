# 11 — Provisioner Decision (spec §13.1)

The provisioner must: accept a **non-blocking** submit (FR-A1), expose **states/logs/cancel**
(FR-A2/3, D1), **auto-teardown** (FR-C2/I1), run an **arbitrary committed entrypoint** unchanged
(Contract §7), build the env from **`uv.lock`** (FR-B2), and ideally support **spot** (FR-I3).

## SkyPilot ⭐ (recommended primary backend)

**Why it fits the FRs almost exactly:**
- **Async SDK** — SDK calls send async requests to the SkyPilot **API server** and return a
  **request ID**; you then `stream_and_get` / `sky.get` / `sky.api_cancel`. CLI mirrors this with
  `--async`. → FR-A1 (submit returns in seconds, non-blocking); NFR-2 (jobs live on the API
  server/controller, surviving laptop disconnect).
- **Jobs surface** (all in `sky` / `sky.jobs`): `launch`, `queue`, `job_status`, `tail_logs`
  (returns exit code: 0 success / 100 fail → FR-A4), `download_logs`, `cancel`. → FR-A2/A3/D1.
- **Auto-teardown** — managed-jobs controller autostops after 10 min idle; per-job
  `autostop: {idle_minutes: 10, down: true}` terminates the VM. → FR-C2, FR-I1.
  (Controller ~$0.25/hr running, <$0.004/hr stopped.)
- **Managed spot** — preemptible VMs with auto-recovery from preemption. → FR-I3.
- **Experiment-agnostic** — the YAML `run:` block runs *any* shell command, e.g.
  `uv run python experiments/capacity.py --config-name=... seed=$LAB_SEED`. No decorators, no SDK
  imports in the experiment. **This is the decisive fit** vs Modal. `setup:` does `uv sync`
  against the committed `uv.lock`. → Contract §7, FR-B2.
- **Provider-agnostic** — supports AWS, GCP, Azure, RunPod, Lambda, **Vast.ai**, Nebius,
  CoreWeave, Paperspace, Cloudflare, OCI, K8s, Slurm, on-prem (20+). → portability (NFR-4 cloud side).
- **Agent-first bonus** — official **SkyPilot Agent Skill** already teaches Claude Code to launch/
  manage jobs (a reference for our MCP design; one demo fanned out ~910 experiments across 16 GPUs).

**Caveats / gaps:**
- **No local backend** — SkyPilot always provisions remote/K8s. We add a separate `local`
  subprocess backend for NFR-4. (Two impls, one interface.)
- Controller autostop unsupported on **Kubernetes/RunPod**; remote API servers use "consolidation
  mode." Fine for our AWS/GCP/Azure/Vast CPU-spot path.
- **No Hetzner** in the provider list → cheapest CPU node (Hetzner) must run via the `local`/manual
  backend, not SkyPilot (see `15-compute-shape.md`).

## Modal (strong, but more coupling — keep as alternative backend)

- `Function.spawn()` → returns a `FunctionCall` (non-blocking); poll `.get(timeout=…)`. Good for
  background long jobs (FR-A1).
- **Sandboxes** (`Sandbox.create` + `Sandbox.exec` → `ContainerProcess` with stdout/stderr
  streams, `timeout`) *can* "check out a git repo and run a command," so arbitrary entrypoints are
  possible. Live shells + logs during execution; serverless ⇒ auto spin-down (FR-C2 for free).
- **Friction:** you must define images via **Modal's Python SDK** — *cannot bring an arbitrary OCI
  image*; Python-centric; ties the lab to Modal's platform. Guaranteed-execution carries a ~3×
  cost multiplier. This works against "arbitrary committed entrypoint + `uv.lock` + provider-
  agnostic," so it's a **second backend**, not the default.

## Managed cluster (Slurm / cloud batch) — rejected for now

High admin overhead, single-cluster, no native multi-cloud/bursting, CLI-only. Wrong fit for solo.

## Decision

- **P0:** implement two backends behind one interface — **`local`** (subprocess; the NFR-4
  fallback, also the fastest dev loop) and **`skypilot`** (managed jobs; remote + teardown + spot).
- Keep a **`modal`** backend as a documented P1/P2 option if serverless bursty inference becomes a
  need.

Sources: `sources/lab-system-raw.md` (SkyPilot SDK/async/agent-skill, Modal sandboxes/spawn).
