# 10 — Proposed Architecture

A thin **control plane** (MCP + CLI) over **pluggable backends**, a **tracking server** for live
metrics, and an **object store** for durable artifacts. The experiment stays decoupled (Contract §7).

```
                 ┌───────────────────────── local repo ─────────────────────────┐
  researcher ──▶ │  CLI  ─┐                                                       │
                 │        ├──▶  Lab core (Python lib)                             │
  Claude Code ─▶ │  MCP ──┘     • submit/status/logs/metrics/fetch/cancel/list   │
   (agent)       │              • manifest writer (commit + uv.lock + config+seed)│
                 │              • backend interface ─────────┐                    │
                 │              runs/<job_id>/ (gitignored) ◀─┼── fetch_artifacts  │
                 └────────────────────────────────────────────┼───────────────────┘
                                                               ▼
                              ┌───────────── Backend (pluggable) ─────────────┐
                              │  local: subprocess on this machine (NFR-4)     │
                              │  skypilot: async SDK → SkyPilot API server     │
                              │            → ephemeral VM (spot/on-demand)      │
                              └───────────────┬───────────────┬────────────────┘
                                              │ run logs+metrics│ push artifacts
                                              ▼                 ▼
                              MLflow tracking server      Object store (R2/S3)
                              (always-on small box)       • artifacts • manifests
                              get_metric_history → live   • no egress (R2)
```

## Component responsibilities

- **Lab core** — a small Python package: resolves a submission into a *job* (pins commit, hashes
  `uv.lock`, resolves config+seed, builds the manifest §8), hands it to a backend, and exposes the
  query surface. Both MCP and CLI are thin shells over this one library (NFR-3, FR-F2).
- **Backend interface** — one Protocol: `submit(spec) → job_id`, `status`, `tail_logs`,
  `cancel`, `collect_artifacts`. Two impls for P0: **`local`** (subprocess + temp dir, no cloud)
  and **`skypilot`** (managed jobs). Tracker/storage are also pluggable (Design principle 6).
- **Tracking server (MLflow)** — reachable from *both* the remote run (writes metrics/params) and
  the control plane (reads `get_metric_history` for the `metrics` tool). Artifact backend = R2.
- **Object store** — canonical durable home for artifacts + manifests; survives VM teardown.
  `fetch_artifacts` pulls into `runs/<job_id>/`.
- **The experiment** — any script obeying Contract §7. The lab injects `$LAB_RUN_ID`,
  `$LAB_RUN_DIR`, seed, and the metrics endpoint as env (FR-C4); the script writes outputs to
  `$LAB_RUN_DIR` and logs metrics via the tracker — no lab imports required.

## Requirement → component traceability

| FR | Satisfied by | Mechanism (from research) |
|---|---|---|
| FR-A1 non-blocking submit | SkyPilot async SDK | SDK returns a **request ID**; CLI `--async` |
| FR-A2 states | backend + manifest | `sky.jobs.queue` / job_status → mapped to lab states |
| FR-A3 cancel | SkyPilot | `sky.jobs.cancel` / `sky.api_cancel` |
| FR-A4 exit code/reason | SkyPilot | `tail_logs` returns exit code (0 ok / 100 fail) |
| FR-B1 commit pin | Lab core | refuse-or-snapshot dirty tree; record commit |
| FR-B2 env from lockfile | run command | `uv sync`/`uv run` against committed `uv.lock` on remote |
| FR-B3 manifest | Lab core | one JSON per job (§8) |
| FR-C1/C2 provision + teardown | SkyPilot | managed job + `autostop {idle_minutes, down:true}` |
| FR-D1 logs | SkyPilot | `sky.jobs.tail_logs(follow=…)` |
| FR-D2 live metrics | MLflow | `MlflowClient.get_metric_history(run_id, key)` |
| FR-E1/E2 artifacts → runs/ | object store + core | push `$LAB_RUN_DIR`→bucket, pull→`runs/<id>/` |
| FR-F1 MCP structured | FastMCP | `outputSchema`/`structuredContent` from type hints |
| FR-F3 fail-loud | FastMCP + core | actionable errors + log tail; exit-code surfaced |
| FR-G2 cheap poll | core | `status` reads controller/state, not the VM |
| FR-I1 timeout | SkyPilot/Sandbox | wall-clock timeout → terminate + teardown |
| FR-I3 spot | SkyPilot | managed spot + auto-recovery |
| FR-J secrets | control plane | creds in env/API-server only; manifest stores URIs not keys |
| NFR-4 local fallback | `local` backend | subprocess on this machine |
| NFR-2 survives disconnect | SkyPilot API server | jobs run on the controller, not the laptop |

The single weak spot: **NFR-4 local fallback** is *not* SkyPilot — hence the separate `local`
backend behind the same interface.
