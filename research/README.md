# ML Experiment Laboratory — Research

Research backing the **Remote Experiment Runner** spec (`../LAB-REQUIREMENTS.md`).
Compiled 2026-05-26, refocused after the requirements draft.

The lab turns "run this experiment" into a **reproducible remote job** with a **structured,
agent-usable (MCP) interface** and **live observability**, decoupled from the local Claude Code
session. First workload: tempotron-capacity (CPU-bound, embarrassingly parallel over seeds/α/K);
GPU is P1. Env is already fixed: Python via **uv** (+ `uv.lock`, NumPy `<2` pin), config via
**Hydra+Pydantic**, metrics via **MLflow/W&B**, outputs to git-ignored `runs/`.

## Recommended design (one-screen answer)

| Concern | Recommendation | Maps to |
|---|---|---|
| Provisioner | **SkyPilot** (async SDK, managed jobs, autostop, runs arbitrary `uv run` entrypoints) + a **local subprocess backend** for the fallback | FR-A/C/I, NFR-4 |
| Live metrics / tracker | **MLflow self-hosted** (`get_metric_history` = live series) on an always-on small box; W&B if you want zero-ops dashboard | FR-D2/D3 |
| Interface | **FastMCP** server (structured outputs, `task` mode) + thin **CLI** mirror | FR-F |
| Reproducibility | git commit pin (+ dirty snapshot), remote env from `uv.lock`, JSON **manifest** per job | FR-B, §8 |
| Artifacts | **object store (Cloudflare R2 / S3)** canonical + `fetch_artifacts` → `runs/<job_id>/` | FR-E |
| Compute shape | **single fat multicore node first**; sweeps as job-per-point on **spot** in P1 | §13.5 |
| Completion signal | **poll** (cheap `status`) in P0; **push sentinel** → background-task notify in P1 | FR-G |

See **`16-decisions.md`** for the per-decision rationale answering spec §13.

## Files

**Lab-system design (requirement-driven — start here):**
| File | Contents |
|---|---|
| `10-architecture.md` | Proposed architecture + **FR → component traceability** |
| `11-provisioner-decision.md` | SkyPilot SDK vs Modal vs local, against the FRs |
| `12-tracking-live-metrics.md` | Live-metrics query API: MLflow vs W&B vs Aim |
| `13-mcp-server.md` | FastMCP, structured outputs, tool mapping to spec §9, security |
| `14-reproducibility-manifest.md` | git pinning, `uv.lock` remote env, manifest, caching |
| `15-compute-shape.md` | CPU spot vs fat node; provider table; storage |
| `16-decisions.md` | **Recommendation per spec §13 open decision + P0 build order** |

**Background landscape (broad survey from the earlier passes — context, not the active plan):**
`01-compute-providers.md`, `02-orchestration.md`, `03-experiment-tracking.md`,
`04-reproducibility-and-environment.md`, `05-hpo-and-sweeps.md`,
`06-gpu-tabular-acceleration.md` *(trimmed — tabular-GPU is not this workload)*.

`sources.md` = annotated bibliography · `sources/` = cached raw extracts.
