# Cached source extracts — Compute Providers

Retrieved 2026-05-26 via Tavily extract. Excerpts trimmed to relevant content.

---

## Lyceum — Lambda Labs vs RunPod vs Vast.ai
URL: https://lyceum.technology/magazine/lambda-labs-vs-runpod-vs-vast-ai

On-Demand pricing (March 2026):

| Provider | A100 80GB | H100 80GB | B200 192GB |
|---|---|---|---|
| RunPod | $1.29/hr | $2.34/hr | $4.99/hr |
| Lambda Labs | $1.48/hr | $3.32/hr | $6.08/hr |
| CoreWeave | $2.70/hr | $6.16/hr | $8.60/hr |
| AWS | $4.10/hr | $12.29/hr | — |
| GCP | $3.67/hr | $6.98/hr | — |

> Vast.ai operates on a fundamentally different model… a decentralized marketplace where
> individuals and data centers list idle GPU capacity. This peer-to-peer approach results in some
> of the lowest prices in the industry… popular for hobbyists, independent researchers, and
> startups working on non-sensitive projects where cost is the primary constraint. The platform
> provides a powerful search interface to filter by GPU model, PCIe bandwidth, geographic
> location, and host reliability scores.

> Specialized providers offer better availability of high-demand chips like the H100 and A100…
> billing models are typically more transparent (hourly/per-second)… a provider might offer a low
> hourly rate but lack the InfiniBand interconnects necessary for efficient multi-node training.

---

## RunPod — Top 12 Cloud GPU Providers for AI/ML in 2026
URL: https://www.runpod.io/articles/guides/top-cloud-gpu-providers

Four dimensions to compare (2026):
- **Pricing models & billing granularity**: per-second vs hourly (idle cost); marketplace/bidding
  (Vast.ai, TensorDock) vs fixed on-demand (AWS/Azure); hidden costs (storage, egress, support).
- RunPod pricing snapshot: RTX 4090 from $0.34/hr, RTX 5090 $0.69, A100 $1.19, H100 $1.99,
  H200 $3.59, B200 $5.98. Per-second billing, no hourly minimums.

Provider notes:
- **RunPod** — per-second billing, community/spot tiers among most affordable; RTX 4090→H100;
  ideal for developers/startups/hobbyists. Multi-node "instant clusters" w/ InfiniBand.
- **Vast.ai** — marketplace; often 50–70% cheaper than hyperscalers; consumer + datacenter GPUs;
  best for researchers/indie devs who can handle reliability variability.
- **Thunder Compute** — (listed #3, budget option).

---

## altstreet — GPU Cloud Pricing Comparison 2026
URL: https://altstreet.investments/tools/gpu/gpu-price-comparison

> Decentralized providers aggregate compute from consumer hardware… significantly lower hourly
> rates. Trade-offs: variable availability (commonly 70–85%), potential interruptions, limited
> enterprise support.
> Specialized providers: institutional-grade availability (95–98%), better networking.
> Hyperscale: enterprise SLAs (99.9%+), global, premium pricing.

> Decentralized GPU marketplaces often offer 50%+ savings vs hyperscale… work well for
> development, batch workloads, and cost-sensitive applications; production services requiring
> high reliability generally favor hyperscale or specialized providers.

Workload→GPU framework: Dev/experimentation → RTX 4090/L40S, decentralized for cost;
Fine-tune 7–13B → RTX 4090/L40S/A100-40GB, $10–100/run; Production inference → H100/A100/L40S on
hyperscale/specialized for SLAs.

---

## Spheron — 10 Best Vast.ai Alternatives 2026
URL: https://www.spheron.network/blog/vastai-alternatives

| Provider | H100/hr | A100/hr | RTX 4090/hr | Best for | Reliability |
|---|---|---|---|---|---|
| Vast.ai | $1.87 (DC) | $0.78 (verified) | $0.25 (unverified) | Cheap experimentation | Variable |
| Spheron | $1.33 | $0.76 | $0.55 | Production workloads | SLA-backed |
| RunPod | $1.99 | $1.19 | $0.34 | Community templates | Good |
| Lambda | $2.49 | $1.29 | $0.99 | Reserved capacity | Excellent |
| CoreWeave | $4.76 (on-demand) | $2.06 | N/A | Enterprise scale | Good |
| Paperspace | $2.45 | $1.45 | $1.08 | Jupyter notebooks | Good |
| Thunder Compute | $1.15 | $0.66 | $0.42 | Budget AWS alternative | Good |
| TensorDock | $1.50 | $0.80 | $0.50 | Decentralized model | Variable |
| Nebius | €1.75 | €0.95 | €0.60 | European compliance | Excellent |

> RunPod built its reputation on developer experience… pre-built templates. Reserved instances
> 30–40% off vs hourly. Where they fall short: pricing higher than budget alternatives; no
> marketplace pricing flexibility.
