# 14 — Reproducibility, Env Build & Manifest (spec §5B, §8)

"Reproducible by construction": a result is only meaningful if its **manifest** regenerates it
(commit + lock + config + seed). No "it ran once on the remote."

## Code pinning (FR-B1)

- A job runs a **pinned git commit**, not the live working tree. Lab core resolves `code_ref` to a
  commit SHA.
- **Dirty tree policy:** either *refuse* (default, safest) or *snapshot+record* — capture
  `git diff` to a blob, store its ref in the manifest (`code.diff_ref`), and apply it on the
  remote. Record `git_dirty: true`.
- The remote obtains code by cloning at the SHA (or SkyPilot `workdir` sync of the clean tree).

## Environment from the lockfile (FR-B2)

- Remote env is built from the committed **`uv.lock`** so versions match local (incl. NumPy `<2`).
  In the SkyPilot `setup:` block: `uv sync --frozen` (or `uv run --frozen …` in `run:`), which
  installs exactly the locked versions. Record `env.uv_lock_sha256` + `python_version`.
- Containerize later (P1) with a CUDA/uv base image if GPU jobs need it; for CPU P0, `uv sync` on a
  stock image is enough.

## Run-time contract injection (FR-C4 / EC-1..6)

The lab exports to the process: `LAB_RUN_ID`, `LAB_RUN_DIR`, the seed, and the metrics endpoint
(`MLFLOW_TRACKING_URI` or equivalent). The experiment reads config (Hydra) + seed, writes all
outputs under `$LAB_RUN_DIR`, logs metrics via `log_metric`, and exits non-zero on failure.

## Manifest (FR-B3, §8)

One JSON per job, written by lab core, stored alongside artifacts in the object store and pulled to
`runs/<job_id>/manifest.json`:

```jsonc
{
  "job_id": "...", "sweep_id": null, "created_at": "...", "submitted_by": "agent",
  "code":  { "git_commit": "…", "git_dirty": false, "diff_ref": null },
  "env":   { "uv_lock_sha256": "…", "python_version": "3.12" },
  "run":   { "entrypoint_command": "uv run python …", "resolved_config": { … }, "seed": 1234 },
  "resources": { "cpus": 16, "gpus": 0, "memory": "32GB", "timeout": "2h" },
  "backend": { "provisioner": "skypilot", "machine_type": "…", "region": "…" },
  "status": "succeeded", "started_at": "…", "ended_at": "…",
  "exit_code": 0, "end_reason": "completed",
  "metrics_uri": "…", "logs_uri": "…",
  "artifacts": [ { "name": "fig.png", "type": "figure", "path": "…", "sha256": "…", "bytes": 1234 } ]
}
```

- **Reproduction (FR-B4, NFR-1):** re-`submit` from a manifest → same commit+lock+config+seed →
  same numbers within a documented tolerance. The seed is explicit and recorded.
- **Caching (FR-B5, P1):** key = hash(commit + resolved_config + seed). If a prior succeeded job
  has the same key, offer its cached result instead of recomputing.

## Artifacts (FR-E)

- The run writes everything to `$LAB_RUN_DIR`; on completion (before teardown) the backend **pushes**
  that dir to the object store (canonical, survives VM teardown). `fetch_artifacts` pulls into
  `runs/<job_id>/` (gitignored). Each artifact carries a `sha256`; agent-readable formats
  (PNG/PDF, CSV/JSON/Parquet, raw logs) per FR-E3.
- For the **local backend**, artifacts are already on disk → just copy/symlink into `runs/<id>/`.

## Provenance note

Tools like DVC could version data/models, but the spec's reproducibility hinge is the **manifest +
git + uv.lock**, not a data lake (it's an explicit non-goal §14). Keep DVC optional, for large
input datasets only.

Sources: `sources/lab-system-raw.md` · `sources/reproducibility-hpo-raw.md` (uv/lockfile patterns).
