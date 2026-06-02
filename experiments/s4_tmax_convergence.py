"""S4 -- T_max convergence study for the iterative (min-over) separability oracle.

Direct demonstration of the claim in the V1 validation section that the small downward
offset in the iterative oracle's half-crossing is a finite-optimisation-budget artefact,
not a property of the perceptron. We hold (N, alpha) fixed at near-critical cells -- where
the maximum-margin solver's iteration requirement (R/kappa*)^2 ~ N/(2-alpha)^2 diverges --
and increase the iteration budget T_max geometrically, measuring the separable fraction
p_sep at each budget. The prediction: p_sep rises monotonically and plateaus at the exact
(linear-programming) value as T_max grows; the budget at which it plateaus grows with N and
with proximity to alpha = 2.

Self-contained Experiment-Contract entrypoint (numpy + torch + scipy). No tempotron imports
(the lab ships this file unchanged). The min-over kernel is duplicated here verbatim from
``v1_perceptron_capacity_minover.py`` so the file stands alone; the scipy linear-feasibility
oracle provides the budget-independent reference value.

Outputs (to $LAB_RUN_DIR)
-------------------------
- ``results.csv``: one row per (N, alpha, T_max): p_sep_minover, p_sep_lp_reference,
  n_reps, mean_iters, frac_hit_cap.
- ``results.json``: params + rows.
- ``metrics.jsonl``: incremental series per (N, alpha) curve.

Run standalone (smoke):
    LAB_RUN_DIR=/tmp/s4 LAB_SEED=0 \\
        uv run --with torch --with scipy python studies/sanity/s4_tmax_convergence.py \\
        cells=50:1.95,100:1.95 t_max_list=10000,100000 n_reps=30
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog


# --- exact reference oracle (budget-independent) -------------------------------------------

def lp_separable(xi: np.ndarray, y: np.ndarray) -> bool:
    """True iff the homogeneous margin LP y_mu (w . xi_mu) >= 1 is feasible (no bias)."""
    p, n = xi.shape
    a_ub = -(y[:, None] * xi)
    b_ub = -np.ones(p)
    res = linprog(
        c=np.zeros(n), A_ub=a_ub, b_ub=b_ub, bounds=[(None, None)] * n, method="highs"
    )
    return bool(res.success)


# --- iterative oracle: batched Krauth-Mezard min-over (verbatim core) ----------------------

def minover_separable_batch(
    patterns: torch.Tensor, labels: torch.Tensor, *, t_max: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (is_separable[B], iters_used[B], hit_cap[B]) for a batch of B instances.

    is_separable[b] = running-best normalised margin > 0 within t_max steps. Hebb warm-start;
    chunked pure-GPU inner loop; separability is decided on the running-best margin (a positive
    margin found at any step proves separability).
    """
    b, p, n = patterns.shape
    yx = labels.unsqueeze(-1) * patterns
    w = yx.sum(dim=1)  # (B, N) Hebb init
    kappa_best = torch.full((b,), -float("inf"), device=patterns.device, dtype=patterns.dtype)
    iter_first_pos = torch.full((b,), t_max, dtype=torch.long, device=patterns.device)
    chunk = max(500, t_max // 50)
    step = 0
    while step < t_max:
        end = min(step + chunk, t_max)
        for _ in range(end - step):
            margins = torch.einsum("bpn,bn->bp", yx, w)
            _, idx = margins.min(dim=1)
            w = w + yx.gather(1, idx.view(b, 1, 1).expand(b, 1, n)).squeeze(1)
        step = end
        with torch.no_grad():
            m = torch.einsum("bpn,bn->bp", yx, w)
            wn = w.norm(dim=1).clamp_min(1e-30)
            kap = m.min(dim=1).values / wn
            newpos = (kap > 0.0) & (kappa_best <= 0.0)
            iter_first_pos = torch.where(
                newpos, torch.full_like(iter_first_pos, step), iter_first_pos
            )
            kappa_best = torch.maximum(kappa_best, kap)
    is_sep = kappa_best > 0.0
    # "hit the cap" = exhausted the budget without ever certifying a positive margin. This is
    # the set of instances that are either genuinely non-separable OR separable-but-under-
    # converged; the gap to the LP reference is exactly the under-converged fraction.
    hit_cap = ~is_sep
    return is_sep.detach(), iter_first_pos.detach(), hit_cap.detach()


def _parse_cells(spec: str) -> list[tuple[int, float]]:
    """'50:1.95,100:1.95' -> [(50,1.95),(100,1.95)]."""
    out = []
    for tok in spec.split(","):
        n_s, a_s = tok.split(":")
        out.append((int(n_s), float(a_s)))
    return out


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))
    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    smoke = os.environ.get("LAB_RUN_DIR", "runs/local-dev") in ("", "runs/local-dev")

    cells = _parse_cells(ov.get(
        "cells",
        "50:1.95,100:1.95" if smoke else "100:1.95,200:1.95,500:1.95,500:1.90",
    ))
    t_max_list = [int(float(x)) for x in ov.get(
        "t_max_list", "10000,100000" if smoke else "30000,100000,300000,1000000,3000000"
    ).split(",")]
    n_reps = int(ov.get("n_reps", "30" if smoke else "100"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"S4 T_max convergence | seed={master_seed} device={device} cells={cells} "
        f"t_max_list={t_max_list} n_reps={n_reps} smoke={smoke}",
        flush=True,
    )

    metrics_path = run_dir / "metrics.jsonl"

    def emit(name: str, value: float, step: int) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps({"name": name, "value": float(value), "step": int(step),
                                "wall_time": time.time()}) + "\n")

    rows: list[dict[str, float | int]] = []
    started = time.time()
    for n, alpha in cells:
        p = round(alpha * n)
        # One fixed instance set per cell (same patterns across all T_max), so the only
        # variable is the optimisation budget. Seeded per (N, alpha).
        rng = np.random.default_rng(np.random.SeedSequence([master_seed, n, int(1000 * alpha)]))
        xis = rng.standard_normal((n_reps, p, n)).astype(np.float32)
        ys = (2 * rng.integers(0, 2, size=(n_reps, p)) - 1).astype(np.float32)

        # Budget-independent reference: exact LP feasibility per instance.
        n_sep_lp = sum(lp_separable(xis[b], ys[b]) for b in range(n_reps))
        p_sep_lp = n_sep_lp / n_reps

        pats = torch.from_numpy(xis).to(device)
        labs = torch.from_numpy(ys).to(device)
        for t_max in t_max_list:
            t0 = time.time()
            is_sep, iters, hit = minover_separable_batch(pats, labs, t_max=t_max)
            n_sep = int(is_sep.sum().item())
            p_sep = n_sep / n_reps
            mean_iters = float(iters.float().mean().item())
            frac_hit = float(hit.float().mean().item())
            rows.append({
                "N": n, "alpha": alpha, "P": p, "T_max": t_max, "n_reps": n_reps,
                "p_sep_minover": p_sep, "p_sep_lp_reference": p_sep_lp,
                "mean_iters_to_first_positive": mean_iters, "frac_hit_cap": frac_hit,
                "wall_seconds": time.time() - t0,
            })
            emit(f"psep_N{n}_a{int(1000*alpha)}", p_sep, t_max)
            print(
                f"  N={n:>4} a={alpha:.3f} T_max={t_max:>8}  p_sep(minover)={p_sep:.3f}  "
                f"p_sep(LP)={p_sep_lp:.3f}  frac_hit_cap={frac_hit:.2f}  "
                f"t={time.time()-t0:.1f}s",
                flush=True,
            )

    fieldnames = list(rows[0].keys())
    with (run_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (run_dir / "results.json").write_text(json.dumps(
        {"experiment": "S4 -- T_max convergence of the min-over separability oracle",
         "params": {"cells": cells, "t_max_list": t_max_list, "n_reps": n_reps,
                    "master_seed": master_seed},
         "elapsed_seconds": time.time() - started, "rows": rows}, indent=2))
    print(f"done in {time.time()-started:.1f}s -> {run_dir}/results.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
