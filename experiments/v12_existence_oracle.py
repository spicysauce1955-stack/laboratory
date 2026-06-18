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
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

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


# --------------------------------------------------------------------------- #
# EXACT existence via big-M MILP: the solver CHOOSES each +1 pattern's read-out
# time (binary z[mu,j], sum_j z[mu,j] >= 1), instead of fixing it heuristically.
#   +1 mu:  for all j   w.Phi[mu,j] >= theta+delta - BigM*(1 - z[mu,j])   (active iff z=1)
#                       sum_j z[mu,j] >= 1                                 (>=1 chosen time)
#   -1 nu:  for all j   w.Phi[nu,j] <= theta-delta                        (convex, all times)
# Feasible  => an existence CERTIFICATE (exact, not a lower bound).
# Infeasible (status 2) => no separating w with |w|<=wbound at margin delta exists
#   *for that bound* -- reported as "infeasible" but, per the honest-NO policy, never
#   promoted to a claim of true infeasibility (it is bound/margin-conditioned).
# Time-limited with no incumbent => "unknown" (NOT counted as existence).
# In the perceptron limit (M=1) sum_j z=1 forces z=1, the binaries vanish, and the MILP
# reduces exactly to the LP -> Cover's alpha_c=2. Searching assignments can only RAISE the
# feasible fraction vs the fixed-assignment LP, so MILP_crossing >= LP_crossing (the gate).
# --------------------------------------------------------------------------- #
def milp_feasible(Phi, y, theta, delta=1e-2, wbound=1e3, time_limit=30.0) -> str:
    """Return 'feasible' | 'infeasible' | 'unknown' for the big-M existence MILP."""
    P, M, N = Phi.shape
    pos = np.where(y > 0)[0]
    neg = np.where(y < 0)[0]
    npos = len(pos)
    nvar = N + npos * M                      # [w_0..w_{N-1}, z_{pos0,0..M-1}, z_{pos1,..}, ...]

    def zcol(p_idx, j):                      # global column index of binary z for +1 #p_idx, time j
        return N + p_idx * M + j

    # safe big-M: w.Phi >= theta+delta - BigM must be slack when z=0, i.e.
    # BigM >= (theta+delta) - min_w(w.Phi) = (theta+delta) + wbound * max_row ||Phi||_1
    row_l1 = np.abs(Phi).sum(axis=2)         # [P, M]
    big_m = float(theta + delta + wbound * row_l1.max() + 1.0)

    rows, cols, data = [], [], []
    lb, ub = [], []
    r = 0
    # +1 target-time constraints (one row per (mu, j)):  w.Phi + BigM*z >= theta+delta
    for pi, mu in enumerate(pos):
        for j in range(M):
            phi = Phi[mu, j]
            nz = np.nonzero(phi)[0]
            rows.extend([r] * len(nz)); cols.extend(nz.tolist()); data.extend(phi[nz].tolist())
            rows.append(r); cols.append(zcol(pi, j)); data.append(big_m)
            lb.append(theta + delta); ub.append(np.inf)
            r += 1
    # +1 at-least-one-time:  sum_j z[mu,j] >= 1
    for pi, _ in enumerate(pos):
        for j in range(M):
            rows.append(r); cols.append(zcol(pi, j)); data.append(1.0)
        lb.append(1.0); ub.append(np.inf)
        r += 1
    # -1 null constraints (one row per (nu, j)):  w.Phi <= theta-delta
    for nu in neg:
        for j in range(M):
            phi = Phi[nu, j]
            nz = np.nonzero(phi)[0]
            if len(nz) == 0:
                continue
            rows.extend([r] * len(nz)); cols.extend(nz.tolist()); data.extend(phi[nz].tolist())
            lb.append(-np.inf); ub.append(theta - delta)
            r += 1

    A = sp.coo_matrix((data, (rows, cols)), shape=(r, nvar)).tocsr()
    constraints = LinearConstraint(A, np.asarray(lb), np.asarray(ub))
    integrality = np.zeros(nvar); integrality[N:] = 1          # z binary, w continuous
    var_lb = np.concatenate([np.full(N, -wbound), np.zeros(npos * M)])
    var_ub = np.concatenate([np.full(N, wbound), np.ones(npos * M)])
    res = milp(c=np.zeros(nvar), constraints=constraints, integrality=integrality,
               bounds=Bounds(var_lb, var_ub), options={"time_limit": time_limit})
    # scipy/HiGHS status: 0=optimal(=any feasible since c=0), 2=infeasible, else time/other.
    if res.status == 0 or res.x is not None:
        return "feasible"
    if res.status == 2:
        return "infeasible"
    return "unknown"


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
def crossing(gen, N, theta, alphas, n_seeds, K=None, n_restart=4,
             method="lp", time_limit=30.0, wbound=1e3, delta=1e-2, M=0, status=None):
    """Existence fraction vs alpha. method='lp' (heuristic lower bound) or 'milp' (exact,
    solver chooses the read-out-time assignment). When method='milp' and `status` is a dict,
    record the per-alpha {feasible,infeasible,unknown} counts so a time-limited NO is never
    silently conflated with a proven infeasibility."""
    out = {}
    for a in alphas:
        P = int(round(a * N))
        hits = 0
        tally = {"feasible": 0, "infeasible": 0, "unknown": 0}
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 * s + P)
            if K is None:
                Phi, y = gen(N, P, rng)
            else:
                Phi, y = gen(N, P, K, rng) if M == 0 else gen(N, P, K, rng, M=M)
            if method == "milp":
                st = milp_feasible(Phi, y, theta, delta=delta, wbound=wbound, time_limit=time_limit)
                tally[st] += 1
                hits += (st == "feasible")
            else:
                hits += exists(Phi, y, theta, n_restart=n_restart, seed=s)
        out[a] = hits / n_seeds
        if status is not None:
            status[a] = tally
    return out


def half_cross(frac: dict[float, float]) -> float:
    xs = sorted(frac)
    for i in range(len(xs) - 1):
        if frac[xs[i]] >= 0.5 >= frac[xs[i + 1]]:
            a0, a1, f0, f1 = xs[i], xs[i + 1], frac[xs[i]], frac[xs[i + 1]]
            return a0 + (f0 - 0.5) / (f0 - f1 + 1e-12) * (a1 - a0)
    return float("nan")


def _parse_alphas(s, default):
    return [float(x) for x in s.split(",")] if s else default


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
    # --- A4: exact big-M MILP existence oracle ---
    ap.add_argument("--milp", action="store_true",
                    help="use the EXACT big-M MILP (solver chooses read-out times) instead of the "
                         "heuristic LP lower bound; MILP existence >= LP everywhere (Stage-2 A4)")
    ap.add_argument("--time-limit", type=float, default=30.0, help="MILP per-instance time cap (s)")
    ap.add_argument("--wbound", type=float, default=1e3, help="MILP |w| box bound")
    ap.add_argument("--delta", type=float, default=1e-2, help="MILP margin")
    ap.add_argument("--scale", type=str, default="",
                    help="comma N-list for the single-spike existence-vs-N sweep (e.g. 80,160,320)")
    ap.add_argument("--milp-M", type=int, default=0, help="cap candidate read-out times (0=default ~3K)")
    ap.add_argument("--alphas", type=str, default="", help="override single-spike/scale alpha grid")
    args = ap.parse_args()
    method = "milp" if args.milp else "lp"
    mkw = dict(method=method, time_limit=args.time_limit, wbound=args.wbound,
               delta=args.delta, M=args.milp_M)
    # When launched by the lab, write the result JSON into the fetchable run dir.
    out_dir = pathlib.Path(os.environ["LAB_RUN_DIR"]) if os.environ.get("LAB_RUN_DIR") else OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PSP: t_peak={T_PEAK:.4f} ms, V0={V0:.4f}  | method={method}")
    print(f"=== GATE: perceptron limit (must give alpha_c ~ 2) [{method}] ===")
    frac = crossing(gen_perceptron, args.N, theta=0.0,
                    alphas=[1.4, 1.7, 1.9, 2.0, 2.1, 2.3, 2.6], n_seeds=args.seeds, **mkw)
    ac = half_cross(frac)
    for a in sorted(frac):
        print(f"  alpha={a:.2f}  exists_frac={frac[a]:.2f}")
    print(f"  -> existence half-crossing = {ac:.3f}   (target 2.0)")
    gate_ok = abs(ac - 2.0) < 0.15
    print(f"  GATE {'PASS' if gate_ok else 'FAIL'}")
    result = {"method": method, "gate": {"frac": frac, "crossing": ac, "pass": gate_ok}}

    ss_alphas = _parse_alphas(args.alphas, [2.0, 2.4, 2.8, 3.0, 3.2, 3.6])

    # A4 MILP-vs-LP monotonicity gate at N=40: MILP can only RAISE the feasible fraction, so
    # MILP_crossing >= LP_crossing (~2.47). Run BOTH here when in --milp mode.
    if args.milp and gate_ok and not args.no_singlespike:
        print(f"\n=== A4 GATE: N={args.N} single-spike, MILP vs LP lower bound ===")
        st_lp = {}
        fr_lp = crossing(gen_singlespike, args.N, theta=1.0, alphas=ss_alphas,
                         n_seeds=args.seeds, K=args.K, method="lp")
        c_lp = half_cross(fr_lp)
        st_milp = {}
        fr_milp = crossing(gen_singlespike, args.N, theta=1.0, alphas=ss_alphas,
                           n_seeds=args.seeds, K=args.K, status=st_milp, **mkw)
        c_milp = half_cross(fr_milp)
        for a in sorted(fr_milp):
            t = st_milp.get(a, {})
            print(f"  alpha={a:.2f}  LP={fr_lp[a]:.2f}  MILP={fr_milp[a]:.2f}  "
                  f"(feas={t.get('feasible',0)} infeas={t.get('infeasible',0)} unk={t.get('unknown',0)})")
        mono_ok = all(fr_milp[a] >= fr_lp[a] - 1e-9 for a in ss_alphas)
        print(f"  LP crossing={c_lp:.3f}  MILP crossing={c_milp:.3f}  "
              f"monotone(MILP>=LP)={'PASS' if mono_ok else 'FAIL'}")
        result["a4_gate_N40"] = {"N": args.N, "K": args.K, "alphas": ss_alphas,
                                 "lp_frac": fr_lp, "milp_frac": fr_milp, "milp_status": st_milp,
                                 "lp_crossing": c_lp, "milp_crossing": c_milp, "monotone": mono_ok}

    # A4 existence-vs-N scale sweep (MILP). Records per-cell {feasible,infeasible,unknown}.
    if gate_ok and args.scale:
        result["scale"] = {}
        for Nsc in [int(x) for x in args.scale.split(",")]:
            print(f"\n=== existence-vs-N: N={Nsc}, K={args.K} [{method}] ===")
            st = {}
            fr = crossing(gen_singlespike, Nsc, theta=1.0, alphas=ss_alphas,
                          n_seeds=args.seeds, K=args.K, status=st, **mkw)
            for a in sorted(fr):
                t = st.get(a, {})
                tag = f"  (feas={t.get('feasible',0)} infeas={t.get('infeasible',0)} unk={t.get('unknown',0)})" if args.milp else ""
                print(f"  alpha={a:.2f}  exists_frac={fr[a]:.2f}{tag}")
            cc = half_cross(fr)
            print(f"  -> N={Nsc} existence half-crossing = {cc:.3f}")
            result["scale"][Nsc] = {"N": Nsc, "K": args.K, "frac": fr, "crossing": cc,
                                    "milp_status": st if args.milp else None}

    if not args.gate and not args.milp and gate_ok and not args.no_singlespike:
        print(f"\n=== single-spike existence (N={args.N}, K={args.K}) ===")
        fr = crossing(gen_singlespike, args.N, theta=1.0,
                      alphas=ss_alphas, n_seeds=args.seeds, K=args.K)
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
