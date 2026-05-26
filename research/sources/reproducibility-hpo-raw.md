# Cached source extracts — Reproducibility, Environment & HPO

Retrieved 2026-05-26 via Tavily extract.

---

## Tobias Klein — Hydra + Optuna + MLflow + DVC pipeline
URL: https://deep-learning-mastery.com/projects/a-comprehensive-look-at-hyperparameter-tuning-with-hydra-and-optuna-in-an-mlops-pipeline

Best practices for HPO in a modular MLOps pipeline:
1. Bound parameter ranges (search spaces defined in YAML, fed to Optuna).
2. Robust validation — cross-validation / well-defined splits (`cv_splits: 5`) to avoid
   overfitting a single hold-out.
3. Reproducibility — Hydra merges tuning configs at runtime; **DVC tracks code & data lineage**;
   changing code or YAML params causes DVC to rerun only affected pipeline stages.
4. Logging/versioning — MLflow logs all trial data so any run is re-creatable.

> This pipeline exemplifies how Hydra configs, Optuna searches, MLflow tracking, and DVC-based
> reproducibility combine to create a well-structured, scalable hyperparameter tuning process.

---

## ashleve/lightning-hydra-template
URL: https://github.com/ashleve/lightning-hydra-template

- PyTorch Lightning + Hydra; user-friendly ML experimentation template.
- **DVC** to version control big files (data/models): `dvc init`, `dvc add data/MNIST` → small
  `.dvc` pointer file versioned in Git.
- **Hyperparameter search**: add a config to `configs/hparams_search`, run
  `python train.py -m hparams_search=mnist_optuna`. No boilerplate — just return the optimized
  metric from the launch file. Supports Optuna, Ax, Nevergrad via Hydra.
  - Results in `logs/<task>/multirun/optimization_results.yaml`.
  - ⚠ Does **not** support resuming interrupted search or advanced pruning — for that, write a
    dedicated optimization task.
- Caveats: "Things break from time to time" (evolving libs); "Not adjusted for data engineering"
  (not for dependent data pipelines — better for model prototyping on ready data); "Overfitted to
  simple use case" (built for simple Lightning training).

---

## Optuna FAQ (referenced)
URL: https://optuna.readthedocs.io/en/stable/faq.html

- Reproducible results: fix sampler `seed`; also make the objective deterministic (set seeds in
  the ML library) since non-deterministic objectives can't be reproduced.
- Parallelism: open multiple workers against a shared storage (e.g.
  `postgresql://...`) running the same `study`.
- Artifacts: `ArtifactStore` with `get_all_artifact_meta()` / `download_artifact()` to retrieve
  best-trial models. Exceptions in trials → `FAIL` status.
