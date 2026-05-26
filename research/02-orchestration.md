# 02 — Orchestration / Provisioning

The "heart of the lab": tools that take *"give me a GPU, run this, then shut down"* and
abstract the provider. This is the build-around decision.

## Candidates

### SkyPilot ⭐ (recommended for batch + cost)
- OSS framework (UC Berkeley Sky Computing Lab). Define a job once in **YAML/Python**, run on
  **any** infra: 12+ clouds, Kubernetes, Slurm, on-prem (backends incl. CoreWeave, AMD ROCm).
- **Cost-aware scheduling**: auto-selects cheapest region/GPU, including spot/preemptible.
- **Managed spot jobs**: 3–6× savings, auto-recovery from preemption (with checkpointing).
- **Autostop**: hands-free teardown of idle clusters → no paying for forgotten boxes.
- Fault-tolerant retries; job portability (move between clouds without rewriting).
- Best for: batch experiments, training, maximum cost control, multi-provider portability.

### dstack ⭐ (recommended for interactive dev)
- OSS "control plane" for AI infra; **ML-native, higher-level** than SkyPilot. Maintainer's own
  comparison (HN, Nov 2024) — what dstack adds over SkyPilot:
  1. Authorization built into services
  2. **Dev environments with IDE integration** (attach VS Code to a GPU box)
  3. HTTPS out of the box + custom domains
  4. Projects for team management / resource isolation
  5. Hardware metrics tracking
- Primitives: **dev environments, tasks, services, fleets**. Own orchestrator that natively
  integrates with cloud providers; **distances from Kubernetes** (but has a K8s backend).
  Supports NVIDIA / AMD / TPU / Tenstorrent. Also does production inference (vLLM/SGLang/TRT-LLM).
- Best for: interactive exploration, dev-environment-on-GPU, marketplace clouds (e.g. Vast.ai).

### Modal
- Serverless, **Python decorators**, focus on fast cold starts (~2–4s). $30/mo free credits.
- Most **lock-in** (you write to their platform); ~3× multiplier for guaranteed execution.
- Best for: bursty/inference workloads, not steady cheap training.

### Slurm (for context — not recommended here)
- HPC legacy: single-cluster, no native multi-cloud/bursting, CLI-only (no built-in GUI/monitoring),
  high admin overhead (`slurm.conf`). Modern ML teams layer SkyPilot/dstack on top instead.

## Decision

| If you want… | Pick |
|---|---|
| Batch jobs, cheapest GPU across providers, set-and-forget cost control | **SkyPilot** |
| Interactive notebook / VS-Code-on-GPU, nicer out-of-box UX, Vast.ai | **dstack** |
| Serverless bursty inference, Python-decorator DX | Modal |

Both SkyPilot and dstack are OSS and provider-agnostic, matching the "assemble OSS" +
"provider-agnostic" goals. Recommendation: **SkyPilot** as the lab's backbone, evaluate
**dstack** if interactive dev sessions become the primary workflow.

Sources: `sources/orchestration-raw.md` · HN dstack-vs-SkyPilot · SkyPilot blog · dstack.ai.
