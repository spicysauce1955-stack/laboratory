# 04 — Reproducibility, Environment & Data

The reproducibility discipline that the whole lab leans on: **pinned env + versioned data +
logged runs**, packaged so an ephemeral remote box produces identical results.

## Components

- **uv** — fast Python dependency management + lockfile. The lockfile is what makes the env
  reproducible across your laptop and the remote box.
- **Docker** — CUDA base image is the unit shipped to the GPU box. The 2025 reference combo is
  literally **Docker + uv + DVC + CUDA**.
- **DVC (Data Version Control)** — versions data & models alongside Git:
  - `dvc add` creates a small `.dvc` pointer file (human-readable) committed to Git; the large
    data lives in a remote (S3 / **Cloudflare R2** / **Backblaze B2** / GCS / SFTP).
  - **Pipelines** (`dvc.yaml`): declare stages with deps/params/outs. `dvc repro` reruns **only
    the stages whose inputs changed** (e.g., change a hyperparam → only `train` reruns).
  - `dvc metrics diff` / `dvc params diff` to compare across commits; run-cache restores prior outputs.
- **Hydra** — composable YAML configs, merged at runtime; powers multirun/sweeps (see `05`).

## Storage choice (cost-critical)

Use a **no-egress-fee object store** as the DVC remote (and tracking artifact store):
**Cloudflare R2** or **Backblaze B2**. Because the workflow re-pulls data to fresh ephemeral
boxes repeatedly, hyperscaler egress fees would otherwise dominate cost.

## Reference pattern (Tobias Klein MLOps pipeline)

`Hydra configs → Optuna search → MLflow tracking → DVC lineage`. Best practices called out:
1. Bound hyperparameter ranges in YAML.
2. Robust validation (e.g., `cv_splits: 5`) — don't overfit a single hold-out.
3. Log every trial; maintain data lineage so any run is re-creatable.
4. DVC reruns only affected stages on code/param change.

## Project templates to borrow from

- **`ashleve/lightning-hydra-template`** — PyTorch Lightning + Hydra, DVC for big files,
  `hparams_search` configs with Optuna. **Caveats:** DL/Lightning-oriented (not tuned for data
  pipelines or classical-ML/tabular), things break as Lightning/Hydra evolve, "overfitted to
  simple training." → borrow the *config/structure ideas*, not wholesale for tabular work.
- **cookiecutter-data-science** — better base layout for classical-ML/data projects.

## Recommendation for this lab

`uv` + `Docker (CUDA base)` + `DVC (→ R2/B2)` + `Hydra`, with a cookiecutter-data-science-style
layout adapted for tabular work. Keep `dvc.yaml` stages small so reruns are cheap.

Sources: `sources/reproducibility-hpo-raw.md` · lightning-hydra-template · Tobias Klein pipeline.
