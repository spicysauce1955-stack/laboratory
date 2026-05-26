# Sources — Annotated Bibliography

Retrieved 2026-05-26 via Tavily. ⭐ = cached in full under `sources/`.

## Lab system — provisioner / tracker / MCP / CPU compute (requirement-driven pass)
- ⭐ [SkyPilot Python SDK reference](https://docs.skypilot.co/en/latest/reference/api.html) — `sky.jobs.*`, clusters, tasks.
- ⭐ [SkyPilot async execution](https://docs.skypilot.co/en/latest/reference/async.html) — request IDs, non-blocking submit, `--async`.
- [SkyPilot managed jobs / spot](https://docs.skypilot.co/en/latest/examples/spot-jobs.html) — autostop/`down:true`, controller lifecycle.
- [SkyPilot Agent Skill](https://docs.skypilot.co/en/latest/getting-started/skill.html) + [GPU Job Mgmt for Agents](https://docs.skypilot.co/en/latest/examples/agents/gpu-job-management.html) — Claude Code plugin for SkyPilot.
- [Deploy ML jobs on Lambda with SkyPilot (uv example)](https://lambda.ai/blog/how-to-deploy-ml-jobs-on-lambda-cloud-with-skypilot) — `uv run` entrypoint + autostop YAML.
- ⭐ [Modal Sandboxes](https://modal.com/docs/guide/sandboxes) + [running commands](https://modal.com/docs/guide/sandbox-spawn) — arbitrary code, `exec`, streams.
- [Modal deep dive: spawn/poll](https://ehsanmkermani.com/posts/2023-12-08-modal-labs-deep-dive) — `Function.spawn` → `FunctionCall.get`.
- [Northflank: Modal Sandboxes alternatives](https://northflank.com/blog/top-modal-sandboxes-alternatives-for-secure-ai-code-execution) — Modal image/SDK lock-in caveat.
- ⭐ [MLflow REST API: get-history](https://mlflow.org/docs/latest/rest-api.html) + [mlflow.client](https://mlflow.org/docs/latest/python_api/mlflow.client.html) — `get_metric_history` full series.
- [Azure ML: query MLflow runs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments-mlflow) — get_metric_history vs last-value gotcha; async logging.
- ⭐ [FastMCP tools / structured output](https://gofastmcp.com/servers/tools) — `outputSchema`/`structuredContent`, `task=True`.
- [PrefectHQ/fastmcp](https://github.com/PrefectHQ/fastmcp) — the standard MCP framework.
- ⭐ [15 Best Practices for MCP Servers in Production](https://thenewstack.io/15-best-practices-for-building-mcp-servers-in-production) — structured content, fail-loud, transports, OAuth.
- ⭐ [Cloud VM benchmarks 2026: perf/price](https://dev.to/dkechag/cloud-vm-benchmarks-2026-performance-price-1i1m) — Hetzner/AWS/GCP/Azure CPU + spot pricing.
- [Vultr pricing](https://www.vultr.com/pricing) — simple hourly CPU rates.
- [Cloud vCPU pricing (Stefan Mai/LinkedIn)](https://www.linkedin.com/posts/stefanmai_cloud-vcpu-is-criminally-overpriced-and-the-activity-7429986452416958465-qAzj) — spot for async work; R2 no-egress.

## Compute providers
- ⭐ [Lyceum: Lambda vs RunPod vs Vast.ai](https://lyceum.technology/magazine/lambda-labs-vs-runpod-vs-vast-ai) — on-demand pricing table (Mar 2026), marketplace vs specialized framing.
- ⭐ [RunPod: Top 12 Cloud GPU Providers 2026](https://www.runpod.io/articles/guides/top-cloud-gpu-providers) — billing dimensions, per-provider notes, RunPod rates.
- ⭐ [altstreet: GPU Pricing Comparison 2026](https://altstreet.investments/tools/gpu/gpu-price-comparison) — price-performance tiers, availability %, workload→GPU framework.
- ⭐ [Spheron: 10 Best Vast.ai Alternatives 2026](https://www.spheron.network/blog/vastai-alternatives) — full provider pricing/reliability matrix.
- [computeprices.com: RunPod vs Vast (live)](https://computeprices.com/compare/runpod-vs-vast) — live per-GPU price tracker.
- [getdeploying: Cloud GPU Pricing (62 providers)](https://getdeploying.com/gpus) — broad live comparison.
- [DeployBase: Best GPU Cloud for Small Team](https://deploybase.ai/articles/best-gpu-cloud-for-small-team-provider-pricing-comparison) — small-team workload recommendations.
- [Introl: Lambda/Paperspace/Vast](https://introl.com/blog/lambda-paperspace-vast-gpu-cloud-comparison-2025) — Dec-2025 rate table.

## Orchestration
- ⭐ [HN: Dstack — alternative to k8s for AI/ML](https://news.ycombinator.com/item?id=42053180) — maintainer's candid dstack-vs-SkyPilot-vs-Modal comparison.
- ⭐ [SkyPilot blog: AI Job Orchestration Pt.1 (Neoclouds)](https://blog.skypilot.co/ai-job-orchestration-pt1-gpu-neoclouds) — why Slurm/K8s fall short; the orchestration gap.
- ⭐ [dstack.ai](https://dstack.ai) — primitives (dev environments/tasks/services/fleets), multi-vendor.
- [SkyPilot GitHub](https://github.com/skypilot-org/skypilot) — features, backends.
- [SkyPilot managed jobs docs](https://docs.skypilot.co/en/v0.6.0/examples/managed-jobs.html) — managed spot + autostop mechanics.
- [Shopify Engineering: SkyPilot multi-cloud](https://shopify.engineering/skypilot) — production case study.
- [CoreWeave + SkyPilot](https://www.coreweave.com/blog/coreweave-adds-skypilot-support-for-effortless-multi-cloud-ai-orchestration) — backend support.
- [HN: dstack vs SkyPilot (thread 2)](https://news.ycombinator.com/item?id=42055036) — more opinions.

## Experiment tracking
- ⭐ [Aim: foundations / TensorBoard alternative](https://aimstack.io/blog/new-releases/aims-foundations-why-were-building-a-tensorboard-alternative) — positioning vs MLflow/W&B, scale.
- ⭐ [ZenML: MLflow vs W&B vs ZenML](https://www.zenml.io/blog/mlflow-vs-weights-and-biases) — feature/pricing comparison.
- ⭐ [MLtraq: tracking framework benchmark](https://mltraq.com/benchmarks/speed) — speed comparison (W&B/MLflow slowest).
- [Aim GitHub](https://github.com/aimhubio/aim) — self-hosted, 10K-run scale, aimlflow.
- [DevOpsSchool: Top 10 trackers](https://www.devopsschool.com/blog/top-10-experiment-tracking-tools-features-pros-cons-comparison) — incl. DVC Experiments, Sacred.

## Reproducibility / environment / HPO
- ⭐ [Tobias Klein: Hydra + Optuna + MLflow + DVC pipeline](https://deep-learning-mastery.com/projects/a-comprehensive-look-at-hyperparameter-tuning-with-hydra-and-optuna-in-an-mlops-pipeline) — reference MLOps pipeline + best practices.
- ⭐ [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) — popular template; DVC + Optuna sweeps; caveats.
- [Optuna FAQ](https://optuna.readthedocs.io/en/stable/faq.html) — reproducibility, parallel storage, artifacts.
- [Medium: Docker + DVC + uv + CUDA pipeline](https://medium.com/@arnaldog12/building-a-reproducible-ml-pipeline-with-docker-dvc-uv-cuda-ac91ec232218) — the modern reproducible combo.
- [DVC docs: pipelines / repro](https://dvc.org/doc/user-guide) — stages, repro, metrics diff.
- [Carpentries: Reproducible ML Workflows for Scientists](https://carpentries-incubator.github.io/reproducible-ml-workflows/) — workshop curriculum (also covers Pixi).
- [RunPod: Reproducible AI w/ DVC + MLflow](https://www.runpod.io/articles/guides/reproducible-ai-made-easy-versioning-data-and-tracking-experiments) — end-to-end remote workflow.

## GPU acceleration for tabular
- ⭐ [NVIDIA: RAPIDS zero-code accel + out-of-core XGBoost](https://developer.nvidia.com/blog/rapids-brings-zero-code-change-acceleration-io-performance-gains-and-out-of-core-xgboost) — cuML 5–175×, Colab integration.
- ⭐ [rapids.ai: cudf.pandas](https://rapids.ai/cudf-pandas) — 150× zero-code, execution/fallback model.
- [cuML GitHub](https://github.com/rapidsai/cuml) — sklearn-compatible, 10–50× large data.
- [NVIDIA AI (LinkedIn): cudf.pandas caveats](https://www.linkedin.com/posts/nvidia-ai_pandas-getting-slow-on-large-datasets-we-activity-7352818215128875008-fjpg) — real "not zero-code" gotchas (regex/hashing).
- [Kaggle: cuDF pandas accelerator](https://www.kaggle.com/questions-and-answers/561085) — usage + benchmarks.
