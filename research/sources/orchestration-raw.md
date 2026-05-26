# Cached source extracts — Orchestration

Retrieved 2026-05-26 via Tavily extract.

---

## Hacker News — "Dstack: An alternative to k8s for AI/ML tasks"
URL: https://news.ycombinator.com/item?id=42053180
(comment by dstack maintainer `cheptsov`, Nov 5 2024)

> **What are the differences in opinions between dstack and SkyPilot?**
> SkyPilot is great. I think there are many tiny details though. At dstack, we try to provide
> out-of-the-box and more high-level experience. Examples:
> 1. Authorization built-into services
> 2. Dev environments with IDE integration
> 3. HTTPS out of the box with an ability to set up own domains
> 4. Projects for team management and resource isolation
> 5. Hardware metrics tracking
> Also, we try to distance from Kubernetes and improve our own orchestrator that natively
> integrates with cloud providers.
>
> **Same question could be posed to Modal.**
> Modal is great too. Modal's strengths is Python decorators and their focus on
> coldstarts/serverless kind of experience.
>
> [re K8s ecosystem — GPU Operator, device plugins, CNI] "both a strength and a weakness";
> dstack aims to support any accelerators out of the box.

---

## SkyPilot Blog — AI Job Orchestration Pt.1: GPU Neoclouds
URL: https://blog.skypilot.co/ai-job-orchestration-pt1-gpu-neoclouds (Jul 8 2025)

On why Slurm/K8s fall short for ML teams:
> It's 2025, and we're still asking AI Researchers and ML Engineers to become Kubernetes experts
> just to run a training job.

Slurm limitations cited:
- No native multi-cluster / cloud-bursting; designed for single-cluster; adding clusters is complex.
- Lack of native GUI/monitoring (CLI-only; top providers bolt on custom UI + Grafana).
- High administrative overhead (manage both K8s and Slurm; complex `slurm.conf`).

---

## dstack.ai — product overview
URL: https://dstack.ai

> dstack is an open-source control plane for agents and engineers to provision compute and run
> training, inference, and sandboxes across NVIDIA, AMD, TPU, and Tenstorrent GPUs — on clouds,
> Kubernetes, and bare-metal clusters.

> You declare **dev environments, tasks, services, and fleets** with simple configuration. dstack
> provisions GPUs, manages clusters via fleets with fine-grained controls, and optimizes cost and
> utilization, while keeping a simple UI and CLI. If you already use Kubernetes, you can run
> dstack on it via the Kubernetes backend.

> [Inference] deploy models as secure, auto-scaling, OpenAI-compatible endpoints integrating with
> SGLang, vLLM, TensorRT-LLM… Disaggregated Prefill/Decode and cache-aware routing.

> [vs Slurm] dstack is built for modern ML/AI workloads with cloud-native provisioning and a
> container-first architecture… also natively supports development and production-grade inference.
