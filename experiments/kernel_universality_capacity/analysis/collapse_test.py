"""Cross-kernel capacity COLLAPSE analysis (Step 9 of the kernel-universality experiment).

Reads the per-cell results.csv files produced by ``v13_kernel_capacity.py`` (one row per
(seed, kernel, K_eff cell, alpha) with `solved` 0/1 and the certified axes `Keff_rice`, `D_PR`,
`kappa_S`, `Q`) and adjudicates the pre-registered targets (derivations/07-kernel-universality-prediction.md):

  T1  slope parallelism among BROAD kernels {gauss, sinc, alpha}  (overlapping bootstrap CIs);
  T2  Gabor sign: alpha_hat_c(gabor) vs gauss at matched K_eff_rice; deficit vs Q (non-monotone allowed);
  T3  which axis COLLAPSES the smooth kernels: fit a SHARED alpha_c(ln ln axis) vs a per-kernel-offset
      model for axis in {Keff_rice, D_PR, Keff/Q}; the axis minimizing AIC / offset spread is the invariant;
  T4  sinusoid alpha_hat_c flat in K_eff (degenerate radial limit);
  T5  rect alpha_hat_c flat vs dt on the D_PR axis (saturation; rises only on the Rice axis).

Estimator/stats are byte-faithful to sinusoidal_capacity/analysis/fit_capacity_slope.py (half-crossing +
bootstrap-over-seeds CI; WLS slope; AIC). Mean over seeds, never best-of. Pure post-processing; seeded.

Usage:
    python collapse_test.py results_glob='runs/**/output/results.csv' [out=figs/collapse.png] [boot=2000]
No training, no GPU. Runs anywhere numpy(+matplotlib for the figure) is present.
"""
from __future__ import annotations

import csv as _csv
import glob
import math
import sys
from collections import defaultdict

import numpy as np

SLOPE_THEORY = 1.0 / (2.0 * math.log(2.0))  # 0.7213 — EVT existence slope
BROAD = ("gauss", "sinc", "alpha")          # the broad mixing kernels for the T1 parallelism test


# ---- estimators (faithful copies of fit_capacity_slope.py) -------------------------------------
def _half_crossing(alphas, psolve):
    order = np.argsort(alphas)
    a, p = alphas[order], psolve[order]
    for i in range(a.size - 1):
        if p[i] >= 0.5 > p[i + 1]:
            t = (p[i] - 0.5) / (p[i] - p[i + 1])
            return float(a[i] + t * (a[i + 1] - a[i]))
    return None


def _cell_crossing_bootstrap(rows, rng, n_boot=2000):
    alphas = np.array(sorted(rows))
    psolve = np.array([rows[a].mean() for a in alphas])
    point = _half_crossing(alphas, psolve)
    boots = []
    for _ in range(n_boot):
        ps = [rng.choice(rows[a], size=rows[a].size, replace=True).mean() for a in alphas]
        c = _half_crossing(alphas, np.array(ps))
        if c is not None:
            boots.append(c)
    if not boots:
        return point, None, None
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _grid_err(rows):
    """Honest grid-resolution error = half the local alpha-step at the crossing (fallback 0.075)."""
    a = np.array(sorted(rows))
    return float(np.median(np.diff(a)) / 2.0) if a.size > 1 else 0.075


def _wls(x, y, w):
    W = np.diag(w)
    X = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    resid = y - X @ beta
    rmse = float(np.sqrt((w * resid**2).sum() / w.sum()))
    return float(beta[0]), float(beta[1]), rmse


def _aic(y, yhat, k):
    n = y.size
    rss = float(((y - yhat) ** 2).sum())
    return n * math.log(rss / n + 1e-300) + 2 * k


# ---- load: group by (kernel, cell) -------------------------------------------------------------
def load(csv_paths):
    """-> {kernel: {cell_key: {'alpha':{a:[solved]}, 'Keff_rice':[], 'D_PR':[], 'Q':[], 'dt':[]}}}."""
    data = defaultdict(lambda: defaultdict(
        lambda: {"alpha": defaultdict(list), "Keff_rice": [], "D_PR": [], "Q": [], "dt": []}))
    for path in csv_paths:
        with open(path) as f:
            for r in _csv.DictReader(f):
                kernel = r.get("kernel", "sinc")
                # cell key: the sweep target (Keff_target), else rounded Keff_rice
                ck = r.get("Keff_target") or r.get("Keff") or r.get("Keff_rice")
                qv = r.get("Q", "")
                try:
                    qb = round(float(qv)) if qv not in (None, "", "nan") else -1
                except ValueError:
                    qb = -1
                cell = (qb, round(float(ck), 3))   # gabor cells distinguished by Q
                celld = data[kernel][cell]
                celld["alpha"][float(r["alpha"])].append(float(r["solved"]))
                for col, key in (("Keff_rice", "Keff_rice"), ("D_PR", "D_PR"), ("Q", "Q"), ("dt", "dt")):
                    if r.get(col) not in (None, "", "nan"):
                        try:
                            celld[key].append(float(r[col]))
                        except ValueError:
                            pass
    return data


def cell_table(data, boot=2000, seed=0):
    """-> {kernel: [ {cell, keff_rice, d_pr, Q, ac, lo, hi, gerr, n_alpha} ... sorted by keff_rice ]}."""
    rng = np.random.default_rng(seed)
    out = {}
    for kernel, cells in data.items():
        rows = []
        for cell, d in cells.items():
            alpha_rows = {a: np.array(v) for a, v in d["alpha"].items()}
            ac, lo, hi = _cell_crossing_bootstrap(alpha_rows, rng, boot)
            rows.append({
                "cell": cell,
                "keff_rice": float(np.mean(d["Keff_rice"])) if d["Keff_rice"] else cell,
                "d_pr": float(np.mean(d["D_PR"])) if d["D_PR"] else float("nan"),
                "Q": float(np.nanmean(d["Q"])) if d["Q"] else float("nan"),
                "dt": float(np.mean(d["dt"])) if d["dt"] else float("nan"),
                "ac": ac, "lo": lo, "hi": hi, "gerr": _grid_err(alpha_rows),
                "n_alpha": len(alpha_rows),
            })
        rows = [r for r in rows if r["ac"] is not None]
        out[kernel] = sorted(rows, key=lambda r: r["keff_rice"])
    return out


def _slope(rows, axis_key):
    """WLS slope of ac vs ln ln(axis) with inverse-variance weights; needs >=2 bracketed cells."""
    pts = [(r[axis_key], r["ac"], max((r["hi"] - r["lo"]) / 3.92 if r["lo"] is not None else r["gerr"],
                                      1e-3)) for r in rows if r[axis_key] and r[axis_key] > math.e]
    if len(pts) < 2:
        return None
    x = np.array([math.log(math.log(a)) for a, _, _ in pts])
    y = np.array([c for _, c, _ in pts])
    w = np.array([1.0 / e**2 for _, _, e in pts])
    a0, b, rmse = _wls(x, y, w)
    return {"intercept": a0, "slope": b, "rmse": rmse, "n": len(pts)}


def collapse_axis(table, axis_key):
    """Shared-curve vs per-kernel-offset model selection (smooth kernels) on ln ln(axis).
    Returns AIC_shared, AIC_offset, offset_spread (max-min kernel intercept), slope_shared."""
    pts, groups = [], []
    for kernel, rows in table.items():
        if kernel == "rect":
            continue
        for r in rows:
            ax = r[axis_key] if axis_key != "keff_over_Q" else (
                r["keff_rice"] / r["Q"] if r["Q"] and r["Q"] > 0 else r["keff_rice"])
            if ax and ax > math.e and r["ac"] is not None:
                pts.append((math.log(math.log(ax)), r["ac"], kernel))
                groups.append(kernel)
    if len(pts) < 4:
        return None
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    ks = [p[2] for p in pts]
    # shared: single line (2 params)
    a0, b, _ = _wls(x, y, np.ones_like(x))
    aic_shared = _aic(y, a0 + b * x, 2)
    # per-kernel offset, common slope: solve LS with kernel dummies + shared slope
    uk = sorted(set(ks))
    D = np.zeros((len(x), len(uk) + 1))
    for i, kk in enumerate(ks):
        D[i, uk.index(kk)] = 1.0
    D[:, -1] = x
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    yhat = D @ beta
    aic_offset = _aic(y, yhat, len(uk) + 1)
    offsets = beta[:-1]
    return {"axis": axis_key, "aic_shared": aic_shared, "aic_offset": aic_offset,
            "offset_spread": float(offsets.max() - offsets.min()), "slope_shared": float(b),
            "slope_common": float(beta[-1]), "n": len(x), "kernels": uk}


def main():
    ov = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    paths = sorted(glob.glob(ov.get("results_glob", "runs/**/output/results.csv"), recursive=True))
    if not paths:
        print("no results.csv matched", ov.get("results_glob")); return 1
    boot = int(ov.get("boot", "2000"))
    print(f"loaded {len(paths)} results.csv files")
    data = load(paths)
    table = cell_table(data, boot=boot)

    print("\n=== per-(kernel, cell) findable alpha_hat_c ===")
    for kernel in sorted(table):
        print(f"\n  {kernel}:")
        print(f"    {'K_rice':>9}{'D_PR':>9}{'Q':>7}{'alpha_c':>9}{'CI':>20}{'gerr':>7}{'nA':>4}")
        for r in table[kernel]:
            ci = f"[{r['lo']:.3f},{r['hi']:.3f}]" if r["lo"] is not None else "(unbracketed)"
            q = f"{r['Q']:.2f}" if not math.isnan(r["Q"]) else "-"
            print(f"    {r['keff_rice']:>9.2f}{r['d_pr']:>9.2f}{q:>7}{r['ac']:>9.3f}{ci:>20}"
                  f"{r['gerr']:>7.3f}{r['n_alpha']:>4}")

    print("\n=== T1: slope on ln ln K_eff_rice (parallelism among broad kernels) ===")
    for kernel in sorted(table):
        s = _slope(table[kernel], "keff_rice")
        if s:
            print(f"  {kernel:<9} slope={s['slope']:+.3f}  intercept={s['intercept']:+.3f}  "
                  f"rmse={s['rmse']:.3f}  (n={s['n']})  [theory existence {SLOPE_THEORY:.3f}]")
    broad_slopes = {k: _slope(table[k], "keff_rice") for k in BROAD if k in table and _slope(table[k], "keff_rice")}
    if len(broad_slopes) >= 2:
        vals = [s["slope"] for s in broad_slopes.values()]
        print(f"  broad-kernel slope spread = {max(vals)-min(vals):+.3f} "
              f"(T1 CONFIRM if small vs per-slope rmse)")

    print("\n=== T2: Gabor sign vs gauss at matched K_eff cell (deficit, monotone in Q?) ===")
    gauss = {round(r["cell"][1]): r for r in table.get("gauss", [])}
    gabor = sorted(table.get("gabor", []), key=lambda r: (round(r["cell"][1]), r["Q"]))
    for r in gabor:
        g = gauss.get(round(r["cell"][1]))
        if g:
            d = r["ac"] - g["ac"]
            print(f"  K_cell={r['cell'][1]:.0f} Q={r['Q']:.1f}: gabor {r['ac']:.3f} - gauss {g['ac']:.3f} "
                  f"= {d:+.3f}  (T2: expect <=0 growing with Q at high Q)")

    print("\n=== DECISIVE two-axis test: slope dα_c/d(ln ln K_eff) vs Q ===")
    print("  (two-axis/radial-dominance: slope DECREASES toward 0 as Q grows — gabor goes K-flat like")
    print("   the sinusoid; one-axis universality: slope is ~FLAT across Q, only the offset moves.)")
    gs = _slope(table.get("gauss", []), "keff_rice")
    if gs:
        print(f"  Q=0  (gauss, low-pass): slope={gs['slope']:+.3f}  (n={gs['n']})")
    gabor_by_q = defaultdict(list)
    for r in table.get("gabor", []):
        gabor_by_q[r["Q"]].append(r)
    for q in sorted(gabor_by_q):
        s = _slope(gabor_by_q[q], "keff_rice")
        if s:
            acs = [f"{r['keff_rice']:.0f}:{r['ac']:.2f}" for r in sorted(gabor_by_q[q], key=lambda r: r['keff_rice'])]
            print(f"  Q={q:<4.0f}(gabor): slope={s['slope']:+.3f}  (n={s['n']})  ac(K)= {' '.join(acs)}")
        else:
            print(f"  Q={q:<4.0f}(gabor): <2 bracketed K cells (cannot fit slope)")
    ss = _slope(table.get("sinusoid", []), "keff_rice")
    if ss:
        print(f"  Q=inf(sinusoid): slope={ss['slope']:+.3f}  (n={ss['n']})  [radial limit; expect ~0]")

    print("\n=== T3: which axis collapses the smooth kernels (lower AIC_shared & small offset_spread) ===")
    for axis in ("keff_rice", "d_pr", "keff_over_Q"):
        c = collapse_axis(table, axis)
        if c:
            better = "SHARED" if c["aic_shared"] <= c["aic_offset"] else "per-kernel-offset"
            print(f"  axis={axis:<12} AIC_shared={c['aic_shared']:.1f} AIC_offset={c['aic_offset']:.1f} "
                  f"-> {better};  offset_spread={c['offset_spread']:.3f}  common_slope={c['slope_common']:+.3f}")
    print("  (T3: the axis with the SMALLEST offset_spread / SHARED preferred is the collapse invariant.)")

    print("\n=== T4: sinusoid flatness in K_eff (degenerate radial limit) ===")
    s = _slope(table.get("sinusoid", []), "keff_rice")
    if s:
        print(f"  sinusoid slope={s['slope']:+.3f} (T4 CONFIRM if ~0); values: "
              f"{[round(r['ac'],3) for r in table['sinusoid']]}")

    print("\n=== T5: rect flatness on D_PR vs Rice axis (saturation) ===")
    for r in table.get("rect", []):
        print(f"  dt={r['dt']:.3f} K_rice={r['keff_rice']:.1f} D_PR={r['d_pr']:.2f} alpha_c={r['ac']:.3f}")
    print("  (T5 CONFIRM if alpha_c flat across dt/D_PR while K_rice varies.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
