# 03 — Experiment Tracking

Logging params/metrics/artifacts so runs are comparable and reproducible.

## Options

| Tool | Model | Strengths | Watch-outs |
|---|---|---|---|
| **DVC Experiments** | Git-native, **no server** | Zero extra infra; reuses the DVC you run for data; experiments tied to commits | Comparison UX is CLI/Studio, less rich |
| **Aim** | OSS, self-hosted, **local-first** | "Best for individual developers"; fast; **params are first-class** → query/filter/group/aggregate/subplot by hyperparam; handles 1000s metrics / 5K+ runs; remote server option | Focused on training tracking (not full lifecycle/registry) |
| **MLflow** | OSS, self-host | Language-agnostic (Py/R/Java/REST); **model registry**; projects/packaging; de-facto standard | UI slows past a few hundred runs; needs a server for remote/shared; ~30-min setup |
| **W&B** | Hosted, closed-source | Best-in-class UI/viz; auto-logs code, hyperparams, system metrics, checkpoints, sample predictions; ~5-min setup; **free tier for individuals** | Hosted/SaaS; not OSS |

Honorable mentions: **FastTrackML** (MLflow-API-compatible but much faster), Neptune, Comet,
ClearML, ZenML, Sacred, TensorBoard.

## Performance (MLtraq benchmark, 2024)

Run-creation/throughput cost, fastest→slowest tail:
- **W&B and MLflow are the worst performing** (threading + event management) — up to ~400× the
  cost of minimal methods.
- **Aim** is next (embedded key-value store overhead) but far cheaper than W&B/MLflow.
- Neptune, Comet, FastTrackML, MLtraq are fastest. FastTrackML is notable: MLflow API
  compatibility with dramatically faster run creation.

Takeaway: for *many small runs* (typical of tabular sweeps), the heavyweight trackers add real
overhead — favor lighter options (DVC Exp / Aim / FastTrackML).

## Recommendation for this lab

- **Default: DVC Experiments or Aim.** Both are OSS, self-hosted/local, low-friction, and Aim's
  hyperparam-first comparison fits tabular sweeps well.
- **Add MLflow** only when you want a **model registry** / lifecycle management (or use
  **FastTrackML** for an MLflow-compatible-but-fast server).
- **W&B free tier** is the pragmatic choice if you'd rather have zero-ops + the nicest UI and
  don't mind hosted/closed-source.

Sources: `sources/experiment-tracking-raw.md` · Aim foundations · ZenML MLflow-vs-W&B · MLtraq benchmark.
