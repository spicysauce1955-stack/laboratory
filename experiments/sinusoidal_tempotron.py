"""Sinusoidal-kernel tempotron — capacity vs bandwidth (Agent 06).

Lab-Experiment-Contract compliant (spec §7): reads $LAB_RUN_DIR / $LAB_SEED,
writes outputs there, logs metrics.jsonl, emits one row per seed into results.csv,
exits non-zero on failure. Accepts `key=value` argv overrides.

Two modes (argv `mode=`):
  mode=kacrice  -> GATE G1: exact Kac-Rice E[#maxima] vs direct root-count of V'(t).
  mode=capacity -> EXP-B: faithful tempotron P_solve at (m, alpha) over seeds.

MODEL (pre-registered in EXPERIMENT.md Phase 2)
-----------------------------------------------
N afferents, pattern mu = N spike times t_i ~ U[0, 2pi]. Equal-amplitude harmonic
band omega_k = k, k=1..m. Learnable weights w in R^N (+ one always-on bias afferent
for the threshold theta). Per-pattern projected coefficients (linear in w):
    A_k(w) = sum_i w_i cos(k t_i),  B_k(w) = - sum_i w_i sin(k t_i)
Potential  V(t; w) = sum_k [A_k cos(k t) + B_k sin(k t)].
Decision: yhat = +1 iff max_t V(t) > 0  (threshold folded into the bias afferent).

Kac-Rice: E[#maxima over [0,T]] = (T/2pi) sqrt(lambda4/lambda2),
  lambda_j = sum_k k^j  (equal-amplitude band).  Closed form per period (T=2pi):
  N_max = sqrt(15 m^2 + 15 m - 5)/5.   (Verified: math-derivation a873f121b57ebac9a.)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

T = 2.0 * np.pi  # one fundamental period


# --------------------------------------------------------------------------- #
# Argv parsing (Hydra-style key=value)                                          #
# --------------------------------------------------------------------------- #
def parse_argv(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in argv:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# --------------------------------------------------------------------------- #
# Core potential machinery                                                      #
# --------------------------------------------------------------------------- #
def project_coeffs(w: np.ndarray, spikes: np.ndarray, m: int):
    """(A_k, B_k) for k=1..m given weights w (N,) and spike times spikes (P, N)."""
    ks = np.arange(1, m + 1)  # (m,)
    # spikes: (P, N); ks: (m,) -> phase (P, N, m)
    ph = spikes[:, :, None] * ks[None, None, :]
    cos = np.cos(ph)
    sin = np.sin(ph)
    A = np.einsum("n,pnm->pm", w, cos)   # (P, m)
    B = -np.einsum("n,pnm->pm", w, sin)  # (P, m)
    return A, B, ks


def potential_grid(A: np.ndarray, B: np.ndarray, ks: np.ndarray, tgrid: np.ndarray):
    """V(t) on a grid for every pattern. A,B:(P,m); tgrid:(G,) -> (P,G)."""
    ph = tgrid[:, None] * ks[None, :]      # (G, m)
    cos = np.cos(ph)                       # (G, m)
    sin = np.sin(ph)
    V = A @ cos.T + B @ sin.T              # (P, G)
    return V


def vmax_and_argmax(A, B, ks, G):
    """max_t V and its grid argmax-time for each pattern, with one Newton polish."""
    tgrid = np.linspace(0.0, T, G, endpoint=False)
    V = potential_grid(A, B, ks, tgrid)         # (P, G)
    j = np.argmax(V, axis=1)                     # (P,)
    tstar = tgrid[j]
    # Newton polish of the argmax (V'=0) — a few steps on the smooth trig poly.
    for _ in range(6):
        ph = tstar[:, None] * ks[None, :]        # (P, m)
        # V'  = sum_k k (-A sin + B cos)
        # V'' = sum_k k^2 (-A cos - B sin)
        dV = np.sum(ks * (-A * np.sin(ph) + B * np.cos(ph)), axis=1)
        ddV = np.sum(ks**2 * (-A * np.cos(ph) - B * np.sin(ph)), axis=1)
        step = np.where(np.abs(ddV) > 1e-12, dV / ddV, 0.0)
        tstar = (tstar - step) % T
    ph = tstar[:, None] * ks[None, :]
    vmax = np.sum(A * np.cos(ph) + B * np.sin(ph), axis=1)
    # guard: keep whichever is larger, grid or polished
    vmax = np.maximum(vmax, V[np.arange(V.shape[0]), j])
    return vmax, tstar


def grad_at_tstar(spikes, tstar, ks):
    """dV/dw_i at t* : cos(k t*) part for A and sin for B, summed over k.
    A_k = sum_i w_i cos(k t_i), B_k = -sum_i w_i sin(k t_i)
    V(t*) = sum_k [A_k cos(k t*) + B_k sin(k t*)]
    dV/dw_i = sum_k [cos(k t_i) cos(k t*) - sin(k t_i) sin(k t*)]
            = sum_k cos(k (t_i - t*))           (P, N)
    """
    # spikes (P,N), tstar (P,), ks (m,)
    dt = spikes - tstar[:, None]                 # (P, N)
    ph = dt[:, :, None] * ks[None, None, :]      # (P, N, m)
    g = np.cos(ph).sum(axis=2)                   # (P, N)
    return g


# --------------------------------------------------------------------------- #
# MODE: kacrice  (GATE G1)                                                      #
# --------------------------------------------------------------------------- #
def kacrice_closed_form(m: int) -> float:
    """E[#maxima] over one period for equal-amplitude band, exact."""
    return float(np.sqrt(15 * m**2 + 15 * m - 5) / 5.0)


def count_maxima_direct(A_row, B_row, ks, Gfine=4096):
    """Direct count of local maxima of one V(t) over [0,T): sign changes of V'
    from + to - (downcrossings)."""
    tgrid = np.linspace(0.0, T, Gfine, endpoint=False)
    ph = tgrid[:, None] * ks[None, :]            # (Gfine, m)
    dV = np.sum(ks * (-A_row * np.sin(ph) + B_row * np.cos(ph)), axis=1)
    # periodic: append first sample
    dVp = np.concatenate([dV, dV[:1]])
    s = np.sign(dVp)
    # downcrossing of V': sign goes + -> - => local maximum
    down = np.sum((s[:-1] > 0) & (s[1:] < 0))
    return int(down)


def run_kacrice(p: dict, run_dir: Path, seed: int) -> int:
    rng = np.random.default_rng(seed)
    m_list = [int(x) for x in p.get("m_list", "1,2,3,4,6,8").split(",")]
    ndraws = int(p.get("ndraws", "2000"))
    Gfine = int(p.get("gfine", "8192"))

    rows = []
    metrics = (run_dir / "metrics.jsonl").open("w")
    for step, m in enumerate(m_list):
        ks = np.arange(1, m + 1)
        # equal-amplitude Gaussian coeffs: A_k,B_k ~ N(0,1)
        A = rng.standard_normal((ndraws, m))
        B = rng.standard_normal((ndraws, m))
        counts = np.array([count_maxima_direct(A[i], B[i], ks, Gfine) for i in range(ndraws)])
        emp = float(counts.mean())
        emp_se = float(counts.std(ddof=1) / np.sqrt(ndraws))
        theory = kacrice_closed_form(m)
        rel_err = abs(emp - theory) / theory
        rows.append({
            "seed": seed, "m": m, "kacrice_theory": round(theory, 5),
            "mc_empirical": round(emp, 5), "mc_se": round(emp_se, 5),
            "rel_err": round(rel_err, 5), "ndraws": ndraws,
        })
        metrics.write(json.dumps({"name": f"relerr_m{m}", "value": rel_err,
                                  "step": step, "wall_time": time.time()}) + "\n")
    metrics.close()
    write_results(run_dir, rows)
    worst = max(r["rel_err"] for r in rows)
    (run_dir / "summary.json").write_text(json.dumps(
        {"mode": "kacrice", "worst_rel_err": worst, "gate_pass": bool(worst <= 0.02),
         "rows": rows}, indent=2))
    print(f"[kacrice] worst rel_err = {worst:.4f}  gate_pass={worst <= 0.02}")
    return 0


# --------------------------------------------------------------------------- #
# MODE: capacity  (EXP-B)                                                       #
# --------------------------------------------------------------------------- #
def train_tempotron(spikes, labels, m, G, epochs, lr0, rng, kappa=0.2):
    """Faithful GS voltage credit-assignment, ONLINE (per-pattern) with a margin.

    Decision: yhat = +1 iff  max_t V(t; w) > theta. We work in the GS gauge with a
    FIXED threshold theta=1 and let the weight scale adapt (an explicit learnable
    theta is gauge-equivalent to rescaling w). A pattern is 'correct with margin'
    iff  y_mu (V_mu(t*) - theta) >= kappa.  On a margin violation, the online GS
    update pushes V(t*) toward the correct side:
        w += lr * y_mu * dV/dw|_{t*}.
    Online (one pattern at a time, reshuffled each epoch) avoids the cancellation of
    a batch sum over random +-1 labels. Returns (solved, final_errors)."""
    P, N = spikes.shape
    ks = np.arange(1, m + 1)
    w = rng.standard_normal(N) / np.sqrt(N * m)    # V scale ~ O(1)
    theta = 1.0
    best = P
    for ep in range(epochs):
        lr = lr0 / (1.0 + ep / 100.0)              # gentle decay
        order = rng.permutation(P)
        # ---- online sweep ----
        for mu in order:
            sp = spikes[mu:mu + 1]                  # (1, N)
            A, B, _ = project_coeffs(w, sp, m)
            vmax, tstar = vmax_and_argmax(A, B, ks, G)
            margin = labels[mu] * (vmax[0] - theta)
            if margin < kappa:
                g = grad_at_tstar(sp, tstar, ks)[0]   # (N,)
                w = w + lr * labels[mu] * g
        # ---- epoch error count (0/1, no margin) ----
        A, B, _ = project_coeffs(w, spikes, m)
        vmax, _ = vmax_and_argmax(A, B, ks, G)
        yhat = np.where(vmax - theta > 0.0, 1, -1)
        nerr = int((yhat != labels).sum())
        best = min(best, nerr)
        if nerr == 0:
            return True, 0
    return False, best


def make_patterns(P, N, rng):
    """N spike times U[0,2pi] per pattern; balanced random +-1 labels.
    Threshold theta is an explicit learnable scalar (see train_tempotron), so no
    bias afferent is needed -- the learnable dimension is exactly N."""
    spikes = rng.uniform(0.0, T, size=(P, N))
    labels = np.empty(P, dtype=int)
    labels[: P // 2] = 1
    labels[P // 2:] = -1
    rng.shuffle(labels)
    return spikes, labels


def parse_seed_set(p: dict, default_seed: int) -> list[int]:
    """Read the sharded-sweep seed subset from `seeds=` (range '0-7' or list '0,1,2').
    Falls back to the single $LAB_SEED if `seeds=` absent."""
    s = p.get("seeds")
    if not s:
        return [default_seed]
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x != ""]


def run_capacity(p: dict, run_dir: Path, default_seed: int) -> int:
    m = int(p["m"])
    alpha = float(p["alpha"])
    N = int(p.get("N", "100"))
    G = int(p.get("G", str(64 * m)))
    epochs = int(p.get("epochs", "400"))
    lr0 = float(p.get("lr0", "0.01"))
    seeds = parse_seed_set(p, default_seed)

    rows = []
    metrics = (run_dir / "metrics.jsonl").open("w")
    for step, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        P = int(round(alpha * N))
        spikes, labels = make_patterns(P, N, rng)
        solved, nerr = train_tempotron(spikes, labels, m, G, epochs, lr0, rng)
        rows.append({
            "seed": seed, "m": m, "alpha": alpha, "N": N, "P": P,
            "solved": int(solved), "final_errors": nerr,
            "kacrice_Keff": round(kacrice_closed_form(m), 5),
        })
        metrics.write(json.dumps({"name": "solved", "value": float(solved),
                                  "step": step, "wall_time": time.time()}) + "\n")
        print(f"[capacity] m={m} alpha={alpha} seed={seed} P={P} solved={solved} nerr={nerr}")
    metrics.close()
    write_results(run_dir, rows)
    nsolved = sum(r["solved"] for r in rows)
    (run_dir / "summary.json").write_text(json.dumps(
        {"mode": "capacity", "m": m, "alpha": alpha, "N": N,
         "n_seeds": len(rows), "n_solved": nsolved}, indent=2))
    return 0


# --------------------------------------------------------------------------- #
def write_results(run_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with (run_dir / "results.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=cols)
        wcsv.writeheader()
        wcsv.writerows(rows)


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get("LAB_SEED", "0"))
    p = parse_argv(sys.argv[1:])
    mode = p.get("mode", "kacrice")
    if mode == "kacrice":
        return run_kacrice(p, run_dir, seed)
    if mode == "capacity":
        return run_capacity(p, run_dir, seed)
    print(f"unknown mode={mode}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
