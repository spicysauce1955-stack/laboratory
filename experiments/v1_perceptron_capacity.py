"""V1 lab job: perceptron storage capacity (Cover 1965 / Gardner 1988), alpha_c -> 2.

Self-contained Experiment-Contract entrypoint (spec LAB-REQUIREMENTS.md §7): **numpy + scipy
only**, no `lab` or `tempotron` imports, fully determined by config + seed. It is the *compute*
half of V1; the figure half (Cover overlay + finite-size collapse) lives in the research repo at
``figures/v1_perceptron_capacity.py`` and reads this job's ``results.csv``.

What it measures
----------------
The fraction ``p_sep(N, alpha)`` of random instances that are *homogeneously* linearly separable
(threshold unit, no bias): draw ``P = round(alpha*N)`` Gaussian patterns ``xi ~ N(0, I_N)`` with
iid labels ``y in {-1,+1}`` and test feasibility of ``y_mu (w . xi_mu) >= 1`` by an LP (HiGHS).

Why Gaussian (the sharp test). Cover's count is exact for points *in general position*:
``P_sep(P,N) = 2^{1-P} sum_{k=0}^{N-1} C(P-1,k)``. Gaussian patterns are in general position a.s.,
so the empirical fraction must match Cover's **exact** curve at *every* finite (N, alpha) within
binomial error -- not merely in the N->inf limit (which is all the +-1 ensemble can promise, since
xi and -xi are collinear). The exact curve crosses 1/2 at ``P = 2N`` (alpha = 2) for *all* N, and
near the crossing ``p_sep ~ Phi((2-alpha) sqrt(N/2))`` -- a 1/sqrt(N) transition width that should
collapse every N onto one Gaussian-CDF curve. Recovering all of this validates the
crossing-of-1/2 + finite-size methodology before we apply it to the tempotron
(ENGINEERING.md, "Validate before scaling", V1).

Outputs (under ``$LAB_RUN_DIR``)
--------------------------------
- ``results.csv`` : one row per (N, alpha): N, alpha, P, n_trials, n_sep, p_sep, p_sep_se.
- ``results.json``: params + rows (provenance for the figure script).
- ``metrics.jsonl``: live series, one per N (``name="psep_N{N}"``, ``step=round(1000*alpha)``,
  ``value=p_sep``) plus ``name="progress"`` -- tail these to watch the run and kill early.

Reproducibility: each (N, alpha) cell seeds a ``SeedSequence([LAB_SEED, N, round(1000*alpha)])``,
so the result is independent of worker scheduling and regenerable from the manifest's seed.

Run standalone (smoke):
    LAB_RUN_DIR=/tmp/v1 LAB_SEED=0 uv run --with scipy python studies/v1_perceptron_capacity.py \
        N_list=10,20 alpha_max=2.5 n_trials=20
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog


def is_separable(patterns: NDArray[np.float64], labels: NDArray[np.float64]) -> bool:
    """True iff a homogeneous separating hyperplane exists (no bias).

    Feasibility of ``y_mu (w . xi_mu) >= 1`` for all mu, written as ``-(y_mu xi_mu).w <= -1``
    with a zero objective and free ``w``. Mirrors the tested ``tempotron.capacity.is_separable``.
    """
    xi = np.asarray(patterns, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    p, n = xi.shape
    a_ub = -y[:, None] * xi
    b_ub = -np.ones(p, dtype=np.float64)
    result = linprog(
        c=np.zeros(n, dtype=np.float64),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * n,
        method="highs",
    )
    return bool(result.success)


def cell_count(n: int, alpha: float, n_trials: int, seed_entropy: list[int]) -> tuple[int, float, int, int]:
    """Run ``n_trials`` Gaussian instances at (N, alpha); return (N, alpha, P, n_separable).

    Top-level / picklable so it runs in a ``ProcessPoolExecutor`` worker.
    """
    p = round(alpha * n)
    if p == 0:
        return n, alpha, 0, n_trials  # no constraints -> trivially separable
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    n_sep = 0
    for _ in range(n_trials):
        xi = rng.standard_normal((p, n))  # general position a.s. -> Cover is exact
        y = 2.0 * rng.integers(0, 2, size=p) - 1.0
        if is_separable(xi, y):
            n_sep += 1
    return n, alpha, p, n_sep


def build_alpha_grid(a_min: float, a_max: float, coarse: float, fine: float,
                     fine_lo: float, fine_hi: float) -> NDArray[np.float64]:
    """Coarse global grid unioned with a fine grid across the transition window."""
    coarse_grid = np.arange(a_min, a_max + 1e-9, coarse)
    fine_grid = np.arange(fine_lo, fine_hi + 1e-9, fine)
    return np.unique(np.round(np.concatenate([coarse_grid, fine_grid]), 4)).astype(np.float64)


def _overrides(argv: list[str]) -> dict[str, str]:
    return dict(tok.split("=", 1) for tok in argv if "=" in tok)


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))

    ov = _overrides(sys.argv[1:])
    n_list = [int(x) for x in ov.get("N_list", "25,50,100,200,400").split(",")]
    a_min = float(ov.get("alpha_min", "1.0"))
    a_max = float(ov.get("alpha_max", "3.0"))
    coarse = float(ov.get("alpha_coarse", "0.1"))
    fine = float(ov.get("alpha_fine", "0.025"))
    fine_lo = float(ov.get("alpha_fine_lo", "1.6"))
    fine_hi = float(ov.get("alpha_fine_hi", "2.4"))
    n_trials = int(ov.get("n_trials", "500"))
    n_workers = int(ov.get("n_workers", str(os.cpu_count() or 1)))

    alphas = build_alpha_grid(a_min, a_max, coarse, fine, fine_lo, fine_hi)
    params = {
        "master_seed": master_seed, "N_list": n_list, "n_trials": n_trials,
        "alpha_min": a_min, "alpha_max": a_max, "alpha_coarse": coarse, "alpha_fine": fine,
        "alpha_fine_lo": fine_lo, "alpha_fine_hi": fine_hi, "n_alphas": int(alphas.size),
        "n_workers": n_workers, "ensemble": "gaussian", "scipy_method": "highs",
    }
    print(f"V1 perceptron capacity | seed={master_seed} workers={n_workers} "
          f"N={n_list} alphas={alphas.size} ({alphas.min():.2f}..{alphas.max():.2f}) "
          f"n_trials={n_trials} cells={len(n_list) * alphas.size}", flush=True)

    metrics_path = run_dir / "metrics.jsonl"

    def emit(name: str, value: float, step: int) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps({"name": name, "value": float(value),
                                "step": int(step), "wall_time": time.time()}) + "\n")

    tasks = [(n, float(a)) for n in n_list for a in alphas]
    total = len(tasks)
    rows: list[dict[str, float | int]] = []
    started = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(cell_count, n, a, n_trials, [master_seed, n, int(round(1000 * a))]): (n, a)
            for (n, a) in tasks
        }
        for fut in as_completed(futures):
            n, alpha, p, n_sep = fut.result()
            p_sep = n_sep / n_trials
            se = float(np.sqrt(max(p_sep * (1.0 - p_sep), 0.0) / n_trials))
            rows.append({"N": n, "alpha": alpha, "P": p, "n_trials": n_trials,
                         "n_sep": n_sep, "p_sep": p_sep, "p_sep_se": se})
            emit(f"psep_N{n}", p_sep, round(1000 * alpha))
            done += 1
            emit("progress", done / total, done)
            print(f"[{done}/{total}] N={n:>4} alpha={alpha:5.3f} P={p:>5} "
                  f"p_sep={p_sep:.3f}+-{se:.3f}", flush=True)

    rows.sort(key=lambda r: (r["N"], r["alpha"]))
    fieldnames = ["N", "alpha", "P", "n_trials", "n_sep", "p_sep", "p_sep_se"]
    with (run_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "results.json").write_text(
        json.dumps({"params": params, "elapsed_seconds": time.time() - started, "rows": rows},
                   indent=2)
    )
    print(f"done: {len(rows)} cells in {time.time() - started:.1f}s -> {run_dir}/results.csv",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
