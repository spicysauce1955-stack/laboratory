"""V7 -- Sign-constrained (excitatory, w>=0) perceptron capacity: test alpha_c^+(kappa) = 1/2 alpha_c^unc(kappa).

Self-contained Experiment-Contract entrypoint (numpy + scipy only; NO torch, NO GPU). The lab ships
this file unchanged, so there are NO ``tempotron`` / ``lab`` imports -- everything is inlined here.

The science (see ``lit/tempotron-reductions/derivations/02-spherical-capacity-prediction.md``)
----------------------------------------------------------------------------------------------
Reduction #2 of the tempotron capacity program: the biologically-faithful Dale's-law null control.
Under the orthant constraint w >= 0 the existence capacity is predicted to be a *uniform halving*
of the unconstrained Gardner curve at every margin,

    alpha_c^+(kappa) = 1/2 alpha_c^unc(kappa)   for all kappa >= 0       [RS + CGMT-rigorous]

with alpha_c^unc(kappa) = 1 / I_kappa, I_kappa = (1+kappa^2) Phi(kappa) + kappa phi(kappa). At
kappa=0: alpha_c^unc = 2 (Cover 1965) -> alpha_c^+ = 1 (Brunel et al. 2004, excitatory). The second
prediction is a flat silent-synapse fraction p_silent = 1/2 at every kappa (NOT rising with margin;
the kappa-dependent conjecture was refuted -- CGMT fixes the cavity bias b=0 forall kappa).

This is an EXISTENCE measurement by CONVEX feasibility -- no learner, no epochs, no HPO, no
findable<existence gap. The feasible set {w >= 0, y(w.xi) >= kappa||w||} is a convex cone, so an
LP/QP DECIDES existence exactly. Playbook Phases 4.5 (learner validation) and 7 (diagnose-the-learner)
are explicitly N/A.

Method (per (seed, kappa, alpha) is read off a single QP per instance per constraint)
-------------------------------------------------------------------------------------
For each instance (seed) at load alpha we draw P = round(alpha N) Gaussian patterns xi ~ N(0, I_N)
and balanced labels y in {-1,+1}, then solve TWO max-margin QPs on the SAME (xi, y) (PAIRED, so
finite-N bias is shared and cancels in the ratio):

    min 1/2 ||w||^2   s.t.   y_mu (w.xi_mu) >= 1   [+ w >= 0 for the sign-constrained branch]

The achieved normalized margin is kappa* = 1 / ||w*||. An instance "achieves margin kappa at load
alpha" iff kappa* >= kappa -- so ONE QP per (instance, constraint) yields the WHOLE kappa-axis.
Record kappa*_unc and kappa*_pos per instance (kappa* = -1 marks an INFEASIBLE instance, so the
test kappa* >= kappa is correctly False for it at every kappa >= 0, including kappa = 0). The capacity crossing alpha_c^+(kappa) is derived
OFFLINE: it is the alpha where the fraction of instances with kappa* >= kappa crosses 1/2.

p_silent (second observable) is the zero-weight fraction of the SIGN-CONSTRAINED max-margin solution
w* >= 0:  p_silent = #{i: w*_i <= tol * max_j w*_j} / N  (predicted 1/2 at capacity). NB it must be
read from the max-margin QP, NOT a feasibility LP (whose vertex on the cone is arbitrary).

Cross-check (robust, independent of the QP route): at kappa=0 we ALSO emit the direct LP feasibility
``sep_unc``/``sep_pos`` (the margin-1 feasibility form, w free / w>=0), which must agree with the QP
verdict kappa* >= 0 (a QP that found any w* is feasible <=> the LP is feasible).

QP solver
---------
Primary: ``quadprog`` (Goldfarb-Idnani dual active-set; exact for this strictly-convex G=I QP),
available via the lab's ``--with quadprog``. Fallback (pure-scipy, no extra dep): ``scipy.optimize``
SLSQP, used automatically when quadprog is not importable. The solver actually used is recorded per
run in results.json. Both are CPU-only.

Run standalone (TINY smoke -- seconds, NOT a sweep):
    LAB_RUN_DIR=/tmp/v7_smoke LAB_SEED=0 \\
        uv run --with scipy --with quadprog python studies/v7_signconstrained_capacity.py \\
        kappa_list=0 N_list=50 alphas=0.6,0.8,1.0,1.2,1.6,2.0,2.4 seeds=0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import linprog, minimize
from scipy.stats import norm

# --- optional fast convex-QP solver (Goldfarb-Idnani). Pure-scipy SLSQP fallback if absent. ---
try:
    import quadprog  # type: ignore

    _HAVE_QUADPROG = True
except Exception:  # pragma: no cover - depends on the runtime env
    _HAVE_QUADPROG = False


# --------------------------------------------------------------------------------------------
# Theory references (closed form; the pre-registered prediction table)
# --------------------------------------------------------------------------------------------
def gardner_integral(kappa: float) -> float:
    """I_kappa = integral_{-kappa}^{inf} (t+kappa)^2 Dt = (1+kappa^2) Phi(kappa) + kappa phi(kappa)."""
    return float((1.0 + kappa**2) * norm.cdf(kappa) + kappa * norm.pdf(kappa))


def gardner_alpha_unc(kappa: float) -> float:
    """Unconstrained Gardner critical load alpha_c^unc(kappa) = 1 / I_kappa (Gardner 1988)."""
    integral, _ = quad(
        lambda t: (t + kappa) ** 2 * norm.pdf(t), -kappa, np.inf, epsabs=1e-12, epsrel=1e-10
    )
    return float(1.0 / integral)


def gardner_alpha_pos(kappa: float) -> float:
    """Sign-constrained predicted critical load alpha_c^+(kappa) = 1/(2 I_kappa) = 1/2 alpha_c^unc."""
    return 0.5 * gardner_alpha_unc(kappa)


# --------------------------------------------------------------------------------------------
# Reproducible per-cell seeding (scheduling-independent, matches the studies' SeedSequence recipe)
# --------------------------------------------------------------------------------------------
def cell_seed(master: int, seed: int, n: int, alpha: float, tag: int = 0) -> int:
    """A 63-bit per-cell seed; depends only on (master, seed, N, alpha, tag), NOT on schedule order."""
    ss = np.random.SeedSequence([master, seed, n, round(1000 * alpha), tag])
    return int(ss.generate_state(1, dtype=np.uint64)[0]) >> 1


# --------------------------------------------------------------------------------------------
# Pattern generation: Gaussian patterns + balanced labels (matches the Gaussian Gardner closed form)
# --------------------------------------------------------------------------------------------
def make_instance(
    p: int, n: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw one instance: xi ~ N(0, I_N) of shape (P, N) and BALANCED labels y in {-1,+1}.

    Balanced = exactly floor(P/2) of one sign and the rest the other (reduces label-imbalance noise,
    per the spec). The label vector is shuffled so its order is decorrelated from the pattern draw.
    """
    xi = rng.standard_normal((p, n))
    y = np.ones(p, dtype=np.float64)
    y[: p // 2] = -1.0
    rng.shuffle(y)
    return xi, y


# --------------------------------------------------------------------------------------------
# Direct LP feasibility (kappa=0 cross-check): the margin-1 homogeneous form
# --------------------------------------------------------------------------------------------
def lp_separable(xi: NDArray[np.float64], y: NDArray[np.float64], nonneg: bool) -> bool:
    """True iff exists w (free, or w>=0 if ``nonneg``) with y_mu (w.xi_mu) >= 1 for all mu.

    Pure feasibility LP (zero objective) in scipy's A_ub w <= b_ub form: -(y_mu xi_mu).w <= -1. The
    only change between the unconstrained and sign-constrained branch is the lower bound on w.
    """
    p, n = xi.shape
    a_ub = -y[:, None] * xi
    b_ub = -np.ones(p, dtype=np.float64)
    bounds = [(0.0, None)] * n if nonneg else [(None, None)] * n
    res = linprog(c=np.zeros(n), A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return bool(res.success)


# --------------------------------------------------------------------------------------------
# Max-margin QP: min 1/2 ||w||^2 s.t. y(w.xi) >= 1 [+ w >= 0]. Returns (w*, status_ok).
# --------------------------------------------------------------------------------------------
def _maxmargin_quadprog(
    xi: NDArray[np.float64], y: NDArray[np.float64], nonneg: bool
) -> tuple[NDArray[np.float64] | None, bool]:
    """Goldfarb-Idnani dual active-set solve of the hard-margin QP.

    quadprog solves  min 1/2 x^T G x - a^T x  s.t.  C^T x >= b. Here G = I_N (strictly convex),
    a = 0, and the constraint rows are the margin constraints diag(y) xi (>= 1) plus, when
    ``nonneg``, the orthant rows I_N (>= 0). Infeasibility raises ValueError -> (None, False).
    """
    p, n = xi.shape
    g_mat = np.eye(n, dtype=np.float64)
    a_vec = np.zeros(n, dtype=np.float64)
    margin_rows = y[:, None] * xi  # (P, N): row mu is y_mu xi_mu, constraint (y_mu xi_mu).w >= 1
    if nonneg:
        c_t = np.vstack([margin_rows, np.eye(n)])  # (P+N, N)
        b_vec = np.concatenate([np.ones(p), np.zeros(n)])
    else:
        c_t = margin_rows  # (P, N)
        b_vec = np.ones(p)
    c_mat = c_t.T  # quadprog wants C with C^T x >= b -> pass C = c_t.T (shape N x n_constraints)
    try:
        sol = quadprog.solve_qp(g_mat, a_vec, c_mat, b_vec, 0)
        w = np.asarray(sol[0], dtype=np.float64)
        return w, True
    except ValueError:
        # quadprog raises "constraints are inconsistent, no solution" when infeasible.
        return None, False


def _maxmargin_slsqp(
    xi: NDArray[np.float64], y: NDArray[np.float64], nonneg: bool
) -> tuple[NDArray[np.float64] | None, bool]:
    """Pure-scipy fallback: SLSQP on min 1/2 ||w||^2 s.t. y(w.xi) >= 1 [+ w >= 0].

    Feasibility is decided first via :func:`lp_separable` (a single highs LP); if infeasible we
    return (None, False) without an SLSQP solve (SLSQP cannot certify infeasibility cleanly). If
    feasible, SLSQP minimizes ||w||^2 from an LP-feasible warm start to recover w* (hence kappa*).
    """
    if not lp_separable(xi, y, nonneg):
        return None, False
    p, n = xi.shape
    a = y[:, None] * xi  # (P, N)
    cons = [{"type": "ineq", "fun": lambda w, a=a: a @ w - 1.0}]  # y(w.xi) - 1 >= 0
    bounds = [(0.0, None)] * n if nonneg else [(None, None)] * n
    # Warm start from the LP feasible point (re-solve cheaply to get an interior-ish start).
    a_ub, b_ub = -a, -np.ones(p)
    lp = linprog(c=np.zeros(n), A_ub=a_ub, b_ub=b_ub,
                 bounds=([(0.0, None)] * n if nonneg else [(None, None)] * n), method="highs")
    w0 = np.asarray(lp.x, dtype=np.float64) if lp.success else np.zeros(n)
    res = minimize(
        lambda w: 0.5 * float(w @ w),
        w0,
        jac=lambda w: w,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        return None, False
    return np.asarray(res.x, dtype=np.float64), True


def maxmargin(
    xi: NDArray[np.float64], y: NDArray[np.float64], nonneg: bool
) -> tuple[float, float]:
    """Solve the max-margin QP; return (kappa_star, p_silent_or_nan).

    kappa_star = 1/||w*|| for a feasible instance (always strictly > 0 since w* is finite, nonzero),
    and the sentinel -1.0 when the QP is INFEASIBLE (no separating w exists -- not even at margin 0).
    The sentinel is negative so the offline test ``kappa_star >= kappa`` correctly returns False for
    every kappa >= 0 on an infeasible instance, including kappa = 0 (where it must match the LP's
    "not separable" verdict). QP infeasibility coincides with LP infeasibility.

    p_silent (the zero-weight fraction) is only meaningful for the SIGN-CONSTRAINED solution and is
    returned as NaN for the unconstrained branch (its sign-free w* has no silent-synapse semantics).
    """
    solver = _maxmargin_quadprog if _HAVE_QUADPROG else _maxmargin_slsqp
    w, ok = solver(xi, y, nonneg)
    if not ok or w is None:
        return -1.0, float("nan")
    wnorm = float(np.linalg.norm(w))
    if wnorm <= 0.0:
        return -1.0, float("nan")
    kappa_star = 1.0 / wnorm
    if nonneg:
        wmax = float(np.max(w)) if w.size else 0.0
        tol = 1e-6 * wmax if wmax > 0 else 1e-12
        p_silent = float(np.mean(w <= tol))
    else:
        p_silent = float("nan")
    return kappa_star, p_silent


# --------------------------------------------------------------------------------------------
# Driver (Experiment-Contract: key=value argv + env LAB_RUN_DIR/LAB_SEED, per-seed rows)
# --------------------------------------------------------------------------------------------
def _alpha_grid(ov: dict[str, str]) -> list[float]:
    if "alphas" in ov:
        return [float(x) for x in ov["alphas"].split(",") if x != ""]
    a0 = float(ov.get("alpha_min", "0.4"))
    a1 = float(ov.get("alpha_max", "2.4"))
    step = float(ov.get("alpha_step", "0.2"))
    n = int(round((a1 - a0) / step)) + 1
    return [round(a0 + i * step, 4) for i in range(n)]


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))

    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    kappa_list = [float(x) for x in ov.get("kappa_list", "0,0.25,0.5,1.0,1.5,2.0").split(",") if x != ""]
    n_list = [int(x) for x in ov.get("N_list", "400").split(",") if x != ""]
    alphas = _alpha_grid(ov)
    # seeds=0,1,2,.. enumerated manifest (lab-shardable, one shard per seed); n_seeds is the legacy
    # contiguous fallback used only when seeds= is absent.
    if "seeds" in ov:
        seeds = [int(x) for x in ov["seeds"].split(",") if x != ""]
    else:
        seeds = list(range(int(ov.get("n_seeds", "24"))))

    solver_name = "quadprog" if _HAVE_QUADPROG else "scipy-slsqp"
    print(
        f"V7 sign-constrained capacity | seed={master_seed} solver={solver_name} "
        f"kappa_list={kappa_list} N_list={n_list} alphas={alphas} seeds={seeds}",
        flush=True,
    )
    kappa_max = max(kappa_list)

    started = time.time()
    rows: list[dict[str, object]] = []
    # Crossing axis: per (N, kappa) accumulate the separable fraction over alpha (offline crossings).
    n_cells = len(n_list) * len(alphas) * len(seeds)
    cell_i = 0

    # Incremental, durable results.csv: write the header once (lazily, when the first instance's rows
    # fix the column order) and flush+fsync after EACH (N, alpha, seed) instance -- the natural unit
    # that appends one row per kappa -- so a timeout/OOM kill leaves a valid, readable partial CSV with
    # all completed instances. The in-memory ``rows`` list is still kept for the end-of-run
    # results.json. ``_csv_written`` tracks how many rows are already on disk.
    csv_path = run_dir / "results.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer: csv.DictWriter | None = None
    _csv_written = 0

    def _flush_new_rows() -> None:
        nonlocal csv_writer, _csv_written
        if len(rows) == _csv_written:
            return
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            csv_writer.writeheader()
        csv_writer.writerows(rows[_csv_written:])
        _csv_written = len(rows)
        csv_file.flush()
        os.fsync(csv_file.fileno())

    for n_aff in n_list:
        for alpha in alphas:
            p = round(alpha * n_aff)
            if p < 2:
                continue
            for sd in seeds:
                cell_i += 1
                rng = np.random.default_rng(cell_seed(master_seed, sd, n_aff, alpha))
                xi, y = make_instance(p, n_aff, rng)

                # ONE max-margin QP per constraint on the SAME (xi, y) -> the whole kappa-axis.
                kstar_unc, _ = maxmargin(xi, y, nonneg=False)
                kstar_pos, p_silent = maxmargin(xi, y, nonneg=True)

                # Direct LP feasibility at kappa=0 (margin-1 form) -- robust cross-check of the QP route.
                sep_unc = int(lp_separable(xi, y, nonneg=False))
                sep_pos = int(lp_separable(xi, y, nonneg=True))

                # One row per kappa on the grid: "separable at margin kappa" iff kappa* >= kappa.
                for kappa in kappa_list:
                    rows.append({
                        "seed": sd,
                        "N": n_aff,
                        "kappa": kappa,
                        "alpha": alpha,
                        "P": p,
                        "sep_unc": sep_unc,            # LP feasibility at kappa=0 (margin-1), w free
                        "sep_pos": sep_pos,            # LP feasibility at kappa=0 (margin-1), w>=0
                        "kstar_unc": kstar_unc,        # achieved max margin, unconstrained
                        "kstar_pos": kstar_pos,        # achieved max margin, sign-constrained
                        "qp_sep_unc": int(kstar_unc >= kappa),  # QP verdict at this kappa, unconstrained
                        "qp_sep_pos": int(kstar_pos >= kappa),  # QP verdict at this kappa, sign-constrained
                        "p_silent": p_silent,          # zero-weight fraction of the w>=0 max-margin sol
                        "solver": solver_name,
                    })
                _flush_new_rows()  # durable per-instance write (timeout/OOM-safe partial CSV)
                if cell_i % max(1, n_cells // 20) == 0 or cell_i == n_cells:
                    print(
                        f"[{cell_i}/{n_cells}] N={n_aff} a={alpha:.3f} seed={sd} P={p} "
                        f"k*_unc={kstar_unc:.3f} k*_pos={kstar_pos:.3f} "
                        f"sep_unc={sep_unc} sep_pos={sep_pos} p_silent={p_silent:.3f}",
                        flush=True,
                    )

    elapsed = time.time() - started
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = None

    # Pre-registered prediction table (for the analysis to compare against, no extra compute).
    pred_table = [
        {
            "kappa": k,
            "I_kappa": gardner_integral(k),
            "alpha_c_unc": gardner_alpha_unc(k),
            "alpha_c_pos": gardner_alpha_pos(k),
            "ratio": 0.5,
            "p_silent": 0.5,
        }
        for k in kappa_list
    ]

    results = {
        "experiment": "v7_signconstrained_capacity",
        "git_sha": git_sha,
        "versions": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "python": sys.version.split()[0],
        },
        "solver": solver_name,
        "have_quadprog": _HAVE_QUADPROG,
        "device": "cpu",
        "params": {
            "master_seed": master_seed,
            "kappa_list": kappa_list,
            "N_list": n_list,
            "alphas": alphas,
            "seeds": seeds,
            "n_seeds": len(seeds),
        },
        "prediction_table": pred_table,
        "rows": rows,
        "elapsed_seconds": elapsed,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    _flush_new_rows()  # write any trailing rows; CSV was written incrementally per instance
    csv_file.close()
    print(
        f"done in {elapsed:.1f}s -> {run_dir}/results.csv ({len(rows)} rows, "
        f"solver={solver_name}, CPU-only)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
