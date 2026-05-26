# 05 — Hyperparameter Optimization & Sweeps

For tabular/classical ML, HPO is often where most of the GPU/CPU time goes.

## Optuna (recommended HPO engine)
- Best-in-class OSS hyperparameter optimization.
- Samplers: TPE (default), Random, Grid, CMA-ES; **pruning** of unpromising trials.
- **Reproducible**: fix the sampler `seed` (and make the objective deterministic — set seeds in
  the ML lib; non-deterministic objectives can't be reproduced).
- **ArtifactStore**: store/list/download best-trial models (`get_all_artifact_meta`,
  `download_artifact`).
- **Parallelism**: share a storage backend (e.g., `postgresql://...` or SQLite) across workers;
  multiple processes/terminals optimize the same `study` concurrently.

## Hydra Optuna Sweeper (recommended glue)
- Run a sweep straight from config, **no boilerplate**:
  `python train.py -m hparams_search=<name>` — just return the optimized metric from the
  `@hydra.main` entrypoint.
- Search space + sampler defined in one YAML; results land in `logs/<task>/multirun/`
  (`optimization_results.yaml`).
- Other Hydra sweeper/launcher plugins: **Ax**, **Nevergrad** (sweepers); **Ray Launcher**
  (run sweep across a cluster).
- **Caveat:** Hydra multirun sweeper does **not** support resuming an interrupted search or
  advanced pruning — for those, write a dedicated Optuna task (call `study.optimize` directly).

## Tracking integration
- **DVCLive** has an Optuna callback (`DVCLiveCallback`) to log trials into DVC Experiments.
- Or log each trial to MLflow/Aim.

## CPU/GPU parallelism for tabular
- **joblib** (`n_jobs`) for sklearn-style parallelism; **Ray** for distributed sklearn/XGBoost.
- On GPU, a single XGBoost/cuML fit is already parallel — see `06`.

## Recommendation for this lab
- **Optuna** as the engine, driven by the **Hydra Optuna Sweeper** for quick config-only sweeps.
- For long/resumable sweeps with pruning, drop to a **dedicated Optuna script** backed by a
  shared SQLite/Postgres study (also enables parallel workers across spot boxes).
- Log trials via DVCLive (if using DVC Exp) or Aim.

Sources: `sources/reproducibility-hpo-raw.md` · Optuna FAQ · Hydra Optuna Sweeper · DVC docs.
