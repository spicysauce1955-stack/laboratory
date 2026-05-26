# 01 — Compute Providers

GPU cloud landscape for a solo, cost-sensitive, GPU-needing classical-ML lab.
Pricing as of Q1–Q2 2026 (on-demand, single GPU). **Prices move fast — verify before use.**

## Consolidated pricing (USD/hr, on-demand)

| GPU | Vast.ai | Thunder | Spheron | RunPod | Lambda | Paperspace | CoreWeave | AWS | GCP |
|---|---|---|---|---|---|---|---|---|---|
| RTX 4090 24GB | $0.25–0.50 | $0.42 | $0.55 | **$0.34** | $0.99 | $1.08 | — | — | — |
| L40S 48GB | $0.56 | — | — | $0.67 | — | — | — | limited | — |
| A100 40GB | ~$0.70 | — | — | $1.19 | $1.29 | — | — | $3.67 | — |
| A100 80GB | $0.78 | $0.66 | $0.76 | $1.19–1.29 | $1.48 | $1.45 | $2.06–2.70 | $4.10 | $3.67 |
| H100 80GB | $1.77–1.87 | $1.15 | $1.33 | $1.99–2.34 | $2.49–3.32 | $2.45 | $4.76–6.16 | $12.29 | $6.98 |

Other RunPod rates: RTX 5090 $0.69, H200 $3.59, B200 $5.98. Lambda B200 $6.08.

## Provider tiers (reliability vs price)

- **Decentralized marketplaces** (Vast.ai, TensorDock): cheapest (50–70% below hyperscalers),
  but **70–85% availability**, possible interruptions, variable networking, no enterprise SLA.
  → use checkpointing; great for non-sensitive experimentation.
- **Specialized / neoclouds** (Lambda, CoreWeave, Spheron, Nebius): **95–98% availability**,
  better networking, transparent hourly/per-second billing, reserved discounts 30–40%.
- **Hyperscale** (AWS/GCP/Azure): **99.9%+** SLA, global, but **2–4× the price** and complex
  (storage/egress/networking fees). Skip unless you already hold credits.

## Provider notes

- **RunPod** — best all-round default. Per-second billing, **no ingress/egress fees**,
  30–90s boot, ~99.8% SLA, pre-built templates (PyTorch/TF/HF), Community (cheap) + Secure
  (compliant) clouds, instant multi-node clusters w/ InfiniBand. Downside: pricier than
  marketplaces; community instances can occasionally drop.
- **Vast.ai** — cheapest. P2P marketplace; filter by GPU, PCIe bandwidth, location, host
  reliability score. Best for hobbyists/indie researchers on non-sensitive work. Variable uptime.
- **Lambda** — specialized, reliable, transparent pricing, reserved 30–40% off. Account
  qualification (~24h), no spot, 1-hour minimum billing. Good for multi-week training.
- **Spheron** — SLA-backed, cheap H100 ($1.33), positioned for production reliability.
- **Thunder Compute** — budget "AWS alternative," very cheap CPU/GPU.
- **Paperspace (DigitalOcean Gradient)** — notebook-friendly, clean UI, per-second.
- **Nebius** — EU data residency / compliance.

## Pricing/billing dimensions that matter (RunPod guide)

- Per-second vs hourly billing (idle cost).
- Marketplace/bidding vs fixed on-demand.
- Hidden costs: storage, **egress**, support fees — can exceed GPU rental for data-heavy work.
  → This is why a **no-egress object store (R2/B2)** matters for an ephemeral-box workflow.

## Recommendation for this lab

- **Start on RunPod** (DX + no egress + per-second + templates), keep the orchestrator able to
  also reach **Vast.ai** for cheap bursts and **Lambda** for steadier long runs.
- For classical ML + cuML/XGBoost-GPU you rarely need H100 — target the
  **RTX 4090 / L40S / A100-40GB** tier (~$0.34–1.30/hr).
- Always pair cheap/marketplace GPUs with **checkpointing** + **autostop**.

Sources: `sources/compute-providers-raw.md` · Lyceum · RunPod Top-12 · altstreet · Spheron.
