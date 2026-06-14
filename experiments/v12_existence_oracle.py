"""Trainer-independent EXISTENCE oracle for the single-spike tempotron (V12 / Path-B 1C).

Question it answers: above the load where the *trainer* stops finding solutions, do
separating weights still *exist*?  If yes, the shortfall is a genuine findability gap; if
existence also stops near the trained crossing, then ~3 is the existence capacity itself.

Method (exploits the paper's own convexity, §5.1):
  - The -1 ("null") constraint  max_t V(t) < theta  is convex (an intersection of
    half-spaces, one per candidate read-out time).
  - The +1 ("target") constraint  exists t: V(t) >= theta  is non-convex, BUT once we
    *fix* a read-out time t*_mu per +1 pattern it becomes a single linear constraint.
  - So for a FIXED assignment {t*_mu} the whole problem is a Linear Program in w
    (feasibility).  A solution exists at load alpha iff SOME assignment is LP-feasible.
  - We search assignments by seeding from a gradient-trained solution's arg-max times
    (best-of-R restarts) + a few local-repair steps.  Feasible-for-some-assignment is an
    existence *lower bound* (we may miss the good assignment), so it is conservative:
    a YES is a certificate, a NO is "not found", never "proven impossible".

Self-gate: in the perceptron limit (one read-out time, Gaussian patterns, theta=0,
symmetric margin) the oracle must reproduce Cover's alpha_c = 2.  `--gate` runs only that.

Run:  uv run python experiments/tempotron_capacity/analysis/v12_existence_oracle.py --gate
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np
from scipy.optimize import linprog

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

TAU_M, TAU_S = 15.0, 3.75
T_PEAK = TAU_M * TAU_S / (TAU_M - TAU_S) * np.log(TAU_M / TAU_S)
V0 = 1.0 / (np.exp(-T_PEAK / TAU_M) - np.exp(-T_PEAK / TAU_S))


def psp(dt: np.ndarray) -> np.ndarray:
    out = V0 * (np.exp(-dt / TAU_M) - np.exp(-dt / TAU_S))
    return np.where(dt >= 0, out, 0.0)


# --------------------------------------------------------------------------- #
# ensembles -> feature tensor Phi[mu, j, i] = s_i at candidate time j for pattern mu
# --------------------------------------------------------------------------- #
def gen_perceptron(N: int, P: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Static Gaussian patterns, one 'time' (M=1). Cover limit -> alpha_c=2."""
    X = rng.standard_normal((P, N))
    Phi = X[:, None, :]                      # [P, 1, N]
    y = rng.choice([-1.0, 1.0], size=P)
    return Phi, y


def gen_singlespike(N: int, P: int, K: float, rng, M: int = 0):
    """One spike per afferent at t~U[0,T], T=K*sqrt(tau_s tau_m); features on an M-grid."""
    T = K * np.sqrt(TAU_S * TAU_M)
    if M == 0:
        M = int(max(32, round(3 * K)))      # ~3 candidate times per resolution slot
    grid = np.linspace(0.0, T, M)
    fire = rng.uniform(0.0, T, size=(P, N))                # [P, N] spike time per afferent
    # Phi[mu, j, i] = psp(grid[j] - fire[mu, i])
    Phi = psp(grid[None, :, None] - fire[:, None, :])      # [P, M, N]
    y = rng.choice([-1.0, 1.0], size=P)
    return Phi, y


def gen_synchrony(N: int, P: int, K: float, rng, M: int = 0):
    """GS-2006 Fig-4a 'perceptron-like' rate/synchrony control: a random half of the afferents
    fire at one common per-pattern time, the rest are silent. Spike *timing* carries no
    information, so existence should collapse to the perceptron value alpha_c -> 2.

    NOTE this is the EXISTENCE side: the LP picks the read-out time, so the volley peak is always
    available and the trainer dead-gradient (V_max on the causal atom at 0; see
    docs/v12-synchrony-control-failure.md) never arises here. The LP therefore measures the true
    perceptron capacity of the binary active-mask patterns."""
    T = K * np.sqrt(TAU_S * TAU_M)
    if M == 0:
        M = int(max(32, round(3 * K)))
    grid = np.linspace(0.0, T, M)
    t_common = rng.uniform(0.0, T, size=P)                  # one common time per pattern
    active = (rng.random((P, N)) < 0.5).astype(float)       # which afferents fire (~N/2)
    # Phi[mu, j, i] = psp(grid[j] - t_common[mu]) * active[mu, i]
    Phi = psp(grid[None, :, None] - t_common[:, None, None]) * active[:, None, :]
    y = rng.choice([-1.0, 1.0], size=P)
    return Phi, y


# --------------------------------------------------------------------------- #
# quick GD trainer (bare GS rule) -> seed read-out-time assignment for +1 patterns
# --------------------------------------------------------------------------- #
def train_gd(Phi, y, theta, epochs=400, lr=0.05, rng=None):
    P, M, N = Phi.shape
    w = (rng.standard_normal(N) * 1e-3) if rng is not None else np.zeros(N)
    for _ in range(epochs):
        V = Phi @ w                          # [P, M]
        jmax = V.argmax(1)
        vmax = V[np.arange(P), jmax]
        out = np.where(vmax >= theta, 1.0, -1.0)
        err = out != y
        if not err.any():
            break
        for mu in np.where(err)[0]:
            w += lr * y[mu] * Phi[mu, jmax[mu]]
    V = Phi @ w
    jmax = V.argmax(1)
    return w, jmax


# --------------------------------------------------------------------------- #
# LP feasibility for a FIXED +1 read-out-time assignment
# --------------------------------------------------------------------------- #
def lp_feasible(Phi, y, theta, jstar, delta=1e-2, wbound=1e3) -> bool:
    P, M, N = Phi.shape
    pos = np.where(y > 0)[0]
    neg = np.where(y < 0)[0]
    rows_ub, b_ub = [], []
    # +1 chosen time:  w . phi >= theta + delta   ->  -phi . w <= -(theta+delta)
    for mu in pos:
        rows_ub.append(-Phi[mu, jstar[mu]])
        b_ub.append(-(theta + delta))
    # -1 all times:     w . phi <= theta - delta
    for nu in neg:
        for j in range(M):
            rows_ub.append(Phi[nu, j])
            b_ub.append(theta - delta)
    A_ub = np.asarray(rows_ub)
    b_ub = np.asarray(b_ub)
    res = linprog(c=np.zeros(N), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(-wbound, wbound)] * N, method="highs")
    return bool(res.success)


def exists(Phi, y, theta, n_restart=4, seed=0) -> bool:
    """Existence lower bound: feasible for SOME assignment from R trained seeds + repair."""
    P, M, N = Phi.shape
    rng = np.random.default_rng(seed)
    if M == 1:                                # perceptron limit: no branching
        return lp_feasible(Phi, y, theta, np.zeros(P, int))
    # Drive-peak assignment FIRST: read each +1 pattern at its total-PSP peak. This time always
    # carries non-zero input, so for the synchrony control it is the volley peak and avoids the
    # causal dead-zone that the GD-seeded argmax collapses into (V_max=0 atom -> +constraint
    # "0 >= theta" -> spurious infeasibility; see docs/v12-synchrony-control-failure.md).
    jdrive = Phi.sum(axis=2).argmax(axis=1)   # [P] argmax_j sum_i Phi[mu, j, i]
    if lp_feasible(Phi, y, theta, jdrive):
        return True
    for _ in range(n_restart):
        _, jstar = train_gd(Phi, y, theta, rng=rng)
        if lp_feasible(Phi, y, theta, jstar):
            return True
        # one local-repair pass: for the most positive-violating +1 pattern, try its
        # second-best time (cheap heuristic)
        # (kept minimal; full column-generation is the next step if this is too weak)
    return False


# --------------------------------------------------------------------------- #
# alpha-sweep -> existence half-crossing
# --------------------------------------------------------------------------- #
def crossing(gen, N, theta, alphas, n_seeds, K=None, n_restart=4):
    out = {}
    for a in alphas:
        P = int(round(a * N))
        hits = 0
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 * s + P)
            if K is None:
                Phi, y = gen(N, P, rng)
            else:
                Phi, y = gen(N, P, K, rng)
            hits += exists(Phi, y, theta, n_restart=n_restart, seed=s)
        out[a] = hits / n_seeds
    return out


def half_cross(frac: dict[float, float]) -> float:
    xs = sorted(frac)
    for i in range(len(xs) - 1):
        if frac[xs[i]] >= 0.5 >= frac[xs[i + 1]]:
            a0, a1, f0, f1 = xs[i], xs[i + 1], frac[xs[i]], frac[xs[i + 1]]
            return a0 + (f0 - 0.5) / (f0 - f1 + 1e-12) * (a1 - a0)
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="run only the perceptron self-gate")
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--K", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--synchrony", action="store_true",
                    help="also run the GS-2006 synchrony/perceptron control (existence -> 2)")
    ap.add_argument("--no-singlespike", action="store_true",
                    help="skip the single-spike block (e.g. when only the synchrony control is wanted)")
    args = ap.parse_args()
    # When launched by the lab, write the result JSON into the fetchable run dir.
    out_dir = pathlib.Path(os.environ["LAB_RUN_DIR"]) if os.environ.get("LAB_RUN_DIR") else OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PSP: t_peak={T_PEAK:.4f} ms, V0={V0:.4f}")
    print("=== GATE: perceptron limit (must give alpha_c ~ 2) ===")
    frac = crossing(gen_perceptron, args.N, theta=0.0,
                    alphas=[1.4, 1.7, 1.9, 2.0, 2.1, 2.3, 2.6], n_seeds=args.seeds)
    ac = half_cross(frac)
    for a in sorted(frac):
        print(f"  alpha={a:.2f}  exists_frac={frac[a]:.2f}")
    print(f"  -> existence half-crossing = {ac:.3f}   (target 2.0)")
    gate_ok = abs(ac - 2.0) < 0.15
    print(f"  GATE {'PASS' if gate_ok else 'FAIL'}")
    result = {"gate": {"frac": frac, "crossing": ac, "pass": gate_ok}}

    if not args.gate and gate_ok and not args.no_singlespike:
        print(f"\n=== single-spike existence (N={args.N}, K={args.K}) ===")
        fr = crossing(gen_singlespike, args.N, theta=1.0,
                      alphas=[2.0, 2.4, 2.8, 3.0, 3.2, 3.6], n_seeds=args.seeds, K=args.K)
        for a in sorted(fr):
            print(f"  alpha={a:.2f}  exists_frac={fr[a]:.2f}")
        print(f"  -> single-spike existence half-crossing = {half_cross(fr):.3f}")
        result["singlespike"] = {"N": args.N, "K": args.K, "frac": fr,
                                 "crossing": half_cross(fr)}

    if not args.gate and gate_ok and args.synchrony:
        print(f"\n=== synchrony/perceptron control existence (N={args.N}, K={args.K}) ===")
        fr = crossing(gen_synchrony, args.N, theta=1.0,
                      alphas=[1.4, 1.7, 1.9, 2.0, 2.1, 2.3, 2.6], n_seeds=args.seeds, K=args.K)
        for a in sorted(fr):
            print(f"  alpha={a:.2f}  exists_frac={fr[a]:.2f}")
        cc = half_cross(fr)
        print(f"  -> synchrony existence half-crossing = {cc:.3f}   (target 2.0)")
        result["synchrony"] = {"N": args.N, "K": args.K, "frac": fr, "crossing": cc,
                               "control_pass": bool(abs(cc - 2.0) < 0.25)}

    (out_dir / "v12_existence_oracle.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"\nwrote {out_dir/'v12_existence_oracle.json'}")


if __name__ == "__main__":
    main()
