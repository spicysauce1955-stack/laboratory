# Cached source extracts — GPU Acceleration for Tabular ML

Retrieved 2026-05-26 via Tavily extract.

---

## NVIDIA Technical Blog — RAPIDS zero-code-change accel + out-of-core XGBoost
URL: https://developer.nvidia.com/blog/rapids-brings-zero-code-change-acceleration-io-performance-gains-and-out-of-core-xgboost

> Announced as open beta, NVIDIA **cuML** now brings zero-code-change acceleration to workflows
> using scikit-learn, UMAP, and hdbscan… speedups ranging from **5–175×** depending on the
> algorithm and dataset. To use it, load the IPython extension before importing standard CPU ML
> libraries.

> The 25.02 and 25.04 RAPIDS releases introduce zero-code-change acceleration for Python ML, major
> IO performance gains, and expanded XGBoost training support (incl. out-of-core XGBoost).

> cuML and GPU-accelerated **Polars** are now built into **Google Colab**, alongside cuDF, pandas
> and NetworkX… the Colab Gemini assistant is now "RAPIDS-aware."

XGBoost (from related RAPIDS deck): one-line change `XGBClassifier(device="cuda")` for up to
**20× speedups**; scalable to largest datasets with Dask/PySpark; built-in SHAP; deployable with
Triton. cuML: A100 vs 96-core EPYC, 10–50× on large data.

---

## rapids.ai — cudf.pandas (pandas Accelerator Mode)
URL: https://rapids.ai/cudf-pandas

> **150× Faster, Zero Code Change.** Write code with the full flexibility of pandas; just load
> `cudf.pandas` to accelerate on the GPU, with **automatic CPU fallback** if needed.

Enable:
- Notebook: `%load_ext cudf.pandas` then `import pandas as pd`
- Script: `python -m cudf.pandas script.py`
- Explicit: `import cudf.pandas; cudf.pandas.install(); import pandas as pd`

> All `cudf.pandas` objects are a proxy to either a GPU (cuDF) or CPU (pandas) object. Operations
> are first attempted on the GPU (copying from CPU if necessary); if that fails, attempted on CPU.
> Compatible with most third-party libraries operating on pandas objects.

---

## Community caveat — cudf.pandas is NOT truly "zero code change"
URL: https://www.linkedin.com/posts/nvidia-ai_pandas-getting-slow-on-large-datasets-we-activity-7352818215128875008-fjpg

> (Practitioner) "Respectfully, 'Zero code change' is incorrect here… porting a legacy ML pipeline
> to GPUs, several functionalities in cudf.pandas are broken/don't match pandas out of the box:
> SQL-related functions don't directly work in cudf; differences in the hashing algorithm can
> impact precision-critical operations in production. Recommend thorough research before porting."

Also noted in thread: NVIDIA's own 18M-row benchmark showed ~20–40× speedups zero-code; commenters
ask how it compares to **Polars** (10–100× vs pandas), which some now prefer outright.

Additional (projectguru / Kaggle): small datasets and non-vectorized ops incur CPU-fallback
overhead — `cudf.pandas` shines on large datasets, not small ones.
