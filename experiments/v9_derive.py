"""V9 offline derivation: raw per-seed/per-epoch capture -> all numerics.

The GPU run (``studies/v3_capacity_sweep.py capture=1``) writes ONE ``.npz`` per cell holding the
full per-seed per-epoch trajectories + final per-seed state + final weights, plus a ``manifest.json``.
This module turns that raw store into every derived numeric we care about -- WITHOUT touching the GPU.
Re-run it freely to add metrics or change definitions; the raw store is the single source of truth.

Derived here (per cell, then aggregated per N):
  * ``p_solve``                     -- fraction of seeds driven to zero training error
  * ``p_solve_at(budget)``          -- ANYTIME p_solve: solved within ``budget`` epochs (any budget,
                                       no rerun -- this is the whole point of keeping the trajectory)
  * ``median/quantile epochs``      -- epochs-to-converge among solved (the GS Fig-4a divergence)
  * ``resid``                       -- residual error among UNCONVERGED (budget- vs capacity-limited)
  * ``alpha_c``                     -- the p_solve=1/2 crossing, with a seed-bootstrap CI
  * ``alpha_c_divergence``          -- from a fit ``epochs ~ A/(alpha_c - alpha)``
  * growth laws ``<||w||>``, ``kappa_+/-`` vs alpha

CLI:  uv run python analysis/v9_derive.py <run_dir> [--out <run_dir>/derived.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_run(run_dir: str | Path) -> tuple[dict, list[dict]]:
    """Load ``manifest.json`` and every cell ``.npz`` under ``run_dir/cells/``.

    Each returned cell is a plain dict of numpy arrays/scalars (npz contents). Cells are sorted by
    (N, alpha) so downstream per-N sweeps are in alpha order.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    cells: list[dict] = []
    for npz in sorted((run_dir / "cells").glob("*.npz")):
        with np.load(npz, allow_pickle=False) as z:
            cell = {k: z[k] for k in z.files}
        cell["_file"] = npz.name
        cells.append(cell)
    cells.sort(key=lambda c: (int(c["N"]), float(c["alpha"])))
    return manifest, cells


def first_zero_epoch(traj_err: np.ndarray) -> np.ndarray:
    """First epoch index at which each seed's training error hits 0 (``-1`` if never).

    ``traj_err`` is ``(n_seeds, n_epochs)`` and NaN-padded past each seed's stop epoch.
    """
    solved = traj_err == 0.0  # NaN != 0 -> padded entries are False, as intended
    has = solved.any(axis=1)
    first = np.where(has, solved.argmax(axis=1), -1)
    return first


def crossing(alphas: np.ndarray, psolve: np.ndarray, level: float = 0.5) -> float:
    """Linear-interpolated alpha at which ``psolve`` first falls through ``level`` (NaN if none).

    Robust to a non-monotone tail: we take the LAST upward bracket ``p[i] >= level > p[i+1]`` so a
    single noisy dip above capacity does not move the estimate.
    """
    a = np.asarray(alphas, float)
    p = np.asarray(psolve, float)
    order = np.argsort(a)
    a, p = a[order], p[order]
    cross = np.nan
    for i in range(len(a) - 1):
        if p[i] >= level > p[i + 1]:
            # interpolate between (a[i], p[i]) and (a[i+1], p[i+1])
            frac = (p[i] - level) / (p[i] - p[i + 1]) if p[i] != p[i + 1] else 0.0
            cross = float(a[i] + frac * (a[i + 1] - a[i]))
    return cross


def bootstrap_crossing(alphas: np.ndarray, conv_by_alpha: np.ndarray,
                       n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Seed-bootstrap CI for the p_solve crossing.

    ``conv_by_alpha`` is a boolean ``(n_alpha, n_seeds)`` matrix (converged per seed per alpha, same
    seed columns across alpha). Returns ``(median, lo2.5, hi97.5)`` of the bootstrap crossing.
    """
    rng = np.random.default_rng(seed)
    n_seeds = conv_by_alpha.shape[1]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, n_seeds)
        ps = conv_by_alpha[:, idx].mean(axis=1)
        out[b] = crossing(alphas, ps)
    out = out[~np.isnan(out)]
    if out.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.median(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def divergence_fit(alphas: np.ndarray, med_epochs: np.ndarray) -> dict[str, float]:
    """Fit ``epochs(alpha) ~ A/(alpha_c - alpha)`` to the rising branch (below the apparent crossing).

    Reparametrise ``1/epochs = (alpha_c - alpha)/A = alpha_c/A - alpha/A``: a straight line in alpha
    whose x-intercept (1/epochs -> 0) is ``alpha_c``. Linear least squares on the finite points.
    """
    a = np.asarray(alphas, float)
    e = np.asarray(med_epochs, float)
    m = np.isfinite(a) & np.isfinite(e) & (e > 0)
    if m.sum() < 2:
        return {"alpha_c_div": float("nan"), "A": float("nan"), "n_points": int(m.sum())}
    a, y = a[m], 1.0 / e[m]
    # y = c0 + c1*a ; root at a = -c0/c1 ; A = -1/c1
    c1, c0 = np.polyfit(a, y, 1)
    alpha_c = -c0 / c1 if c1 != 0 else float("nan")
    return {"alpha_c_div": float(alpha_c), "A": float(-1.0 / c1) if c1 != 0 else float("nan"),
            "n_points": int(m.sum())}


def derive(cells: list[dict], budgets: tuple[int, ...] = (50, 100, 250, 500, 1000, 2500, 5000)
           ) -> dict[str, Any]:
    """Compute per-cell numerics and per-N aggregates (p_solve sweep, crossing, divergence)."""
    per_cell: list[dict] = []
    for c in cells:
        conv = c["converged"].astype(bool)
        n_seeds = int(c["n_seeds"])
        fze = first_zero_epoch(c["traj_err"])
        solved_epoch = np.where(fze >= 0, fze, np.nan)
        cell = {
            "N": int(c["N"]), "alpha": float(c["alpha"]), "K": float(c["K"]), "P": int(c["P"]),
            "n_seeds": n_seeds,
            "p_solve": float(conv.mean()),
            "p_solve_anytime": {int(b): float(np.nanmean((fze >= 0) & (fze <= b))) for b in budgets},
            "median_epochs_conv": float(np.nanmedian(solved_epoch)) if np.isfinite(solved_epoch).any()
            else float("nan"),
            "q25_epochs_conv": float(np.nanpercentile(solved_epoch, 25))
            if np.isfinite(solved_epoch).any() else float("nan"),
            "q75_epochs_conv": float(np.nanpercentile(solved_epoch, 75))
            if np.isfinite(solved_epoch).any() else float("nan"),
            "resid_unconv": float(c["final_errfrac"][~conv].mean()) if (~conv).any() else 0.0,
            "mean_wnorm": float(c["final_wnorm"].mean()),
            "kappa_plus_conv": float(c["final_kappa_plus"][conv].mean()) if conv.any() else float("nan"),
            "kappa_minus_conv": float(c["final_kappa_minus"][conv].mean()) if conv.any() else float("nan"),
            "mean_init_fire_rate": float(c["init_fire_rate"].mean()),
            "robustness_radius": float(np.nanmean(c["robustness_radius"]))
            if np.isfinite(c["robustness_radius"]).any() else float("nan"),
            # lossless-reconstruction check vs the in-run reference scalars stored in the npz
            "ref_p_solve": float(c["ref_p_solve"]),
            "p_solve_matches_ref": bool(abs(float(conv.mean()) - float(c["ref_p_solve"])) < 1e-9),
        }
        per_cell.append(cell)

    # per-N aggregates
    per_n: dict[str, Any] = {}
    ns = sorted({pc["N"] for pc in per_cell})
    for n in ns:
        rows = sorted([pc for pc in per_cell if pc["N"] == n], key=lambda r: r["alpha"])
        alphas = np.array([r["alpha"] for r in rows])
        psolve = np.array([r["p_solve"] for r in rows])
        med_ep = np.array([r["median_epochs_conv"] for r in rows])
        conv_by_alpha = np.stack([cells_for(cells, n, r["alpha"]) for r in rows])  # (n_alpha,n_seeds)
        med, lo, hi = bootstrap_crossing(alphas, conv_by_alpha)
        per_n[str(n)] = {
            "alphas": alphas.tolist(),
            "p_solve": psolve.tolist(),
            "median_epochs_conv": med_ep.tolist(),
            "alpha_c_crossing": crossing(alphas, psolve),
            "alpha_c_crossing_ci": [med, lo, hi],
            "alpha_c_divergence": divergence_fit(alphas, med_ep),
            "mean_wnorm": [r["mean_wnorm"] for r in rows],
            "kappa_plus_conv": [r["kappa_plus_conv"] for r in rows],
            "kappa_minus_conv": [r["kappa_minus_conv"] for r in rows],
        }
    return {"per_cell": per_cell, "per_n": per_n, "N_list": ns}


def cells_for(cells: list[dict], n: int, alpha: float) -> np.ndarray:
    """The boolean converged vector ``(n_seeds,)`` for cell (N=n, alpha)."""
    for c in cells:
        if int(c["N"]) == n and abs(float(c["alpha"]) - alpha) < 1e-9:
            return c["converged"].astype(bool)
    raise KeyError(f"cell N={n} alpha={alpha} not found")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    manifest, cells = load_run(args.run_dir)
    if not cells:
        print(f"no cells found under {args.run_dir}/cells/")
        return 1
    derived = derive(cells)
    derived["manifest"] = manifest
    out = Path(args.out) if args.out else Path(args.run_dir) / "derived.json"
    out.write_text(json.dumps(derived, indent=2, default=float))
    n_mismatch = sum(not pc["p_solve_matches_ref"] for pc in derived["per_cell"])
    print(f"derived {len(cells)} cells -> {out}")
    for n, d in derived["per_n"].items():
        ci = d["alpha_c_crossing_ci"]
        print(f"  N={n}: alpha_c={d['alpha_c_crossing']:.3f} "
              f"(CI [{ci[1]:.2f},{ci[2]:.2f}]) | divergence-fit "
              f"alpha_c={d['alpha_c_divergence']['alpha_c_div']:.3f}")
    print(f"  lossless check: {len(cells) - n_mismatch}/{len(cells)} cells match in-run p_solve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
