# 06 — GPU Acceleration (trimmed — not this workload)

> **De-scoped after the requirements draft.** The first workload (tempotron-capacity) is a
> **CPU-bound spiking-neuron simulation**, embarrassingly parallel over seeds/α/K — *not* tabular
> data science. So the RAPIDS / cuML / `cudf.pandas` tabular-GPU stack from the earlier survey is
> **not relevant** here. GPU in the lab is just a generic **resource request** (P1), handled by the
> same `skypilot` backend (`15-compute-shape.md`), not a RAPIDS adoption.

## If/when GPU does help this kind of work

- The relevant acceleration path for vectorizable neuron simulations is **array libraries**
  (NumPy → **JAX** or **PyTorch** on GPU), where the inner loop is rewritten as batched tensor ops —
  *not* pandas/DataFrame acceleration.
- Decision rule unchanged from the survey: GPU pays off on **large** vectorized workloads; on small
  problems CPU wins (kernel-launch / transfer overhead). Profile before porting.
- Hardware: an **RTX 4090 / L40S / A100-40GB** tier (~$0.34–1.30/hr) is plenty; reserve H100s for
  actual deep learning. Provider details in `01-compute-providers.md`.

## Original tabular-GPU notes
Preserved (for reference only, in case a future experiment *is* tabular/data-heavy) in the cached
source extract: `sources/gpu-tabular-raw.md`. Summary: XGBoost `device="cuda"` ~20×, cuML 5–175×,
`cudf.pandas` 10–150× **but not truly zero-code** (regex/hashing gaps, small-data overhead);
Polars-GPU 10–100× vs pandas.
