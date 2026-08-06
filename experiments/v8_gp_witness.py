"""V8 GP FREE WITNESS -- E[max V] plateaus while zero up-crossings diverge as tau_s -> 0.

Self-contained Experiment-Contract entrypoint (numpy + scipy only; CPU; **NO torch**). This is the
PRIMARY, near-free, decisive test of reduction #4 (the GP / frozen-kernel rise-time saturation fork);
see ``lit/tempotron-reductions/derivations/04-gp-capacity-prediction.md`` section (a) and the
witness-gate G1-G7 in section 4.

The physics
-----------
The membrane potential of a tempotron driven by a dense random spike train is, by the CLT, a
stationary zero-mean Gaussian process ``V(t)`` whose autocovariance is the PSP-kernel autocorrelation

    C(t) = rho * integral K(u) K(u + |t|) du ,    K(s) = V0 (e^{-s/tau} - e^{-s/tau_s})  (s >= 0),

the double-exponential PSP. For the *capacity* problem only the SHAPE of C matters (rho and V0 are
absorbed into the threshold gauge), so we synthesize ``V`` at **unit variance**: C_norm(t) = C(t)/C(0).
A closed form (advance note 04, symbolically verified) is the membrane-difference covariance

    C_norm(t) = ( tau e^{-|t|/tau} - tau_s e^{-|t|/tau_s} ) / (tau - tau_s) ,

which is exactly the witness-gate covariance of section 4. (The two routes -- numeric kernel
autocorrelation vs the closed form -- agree to grid error; the closed form is the default because it
is exact and free of the K-integration cutoff. Pass ``cov=numeric`` to use the kernel-autocorr route
as an independent cross-check.)

What we measure, sweeping tau_s down at fixed T, tau
----------------------------------------------------
Over ``[0, T]`` on a grid ``dt = tau_s / oversample`` (so ``n_grid = oversample * T / tau_s ~ 1/tau_s``
-- the SAME 1/tau_s cost driver as the capacity test, but paid here once on a 1-D FFT):

(a) ``E[max_t V(t)]`` -- predicted to PLATEAU toward the OU extreme ``sqrt(2 ln(T/tau)) = 2.648`` and
    track ``sqrt(2 ln K_PR)`` (the participation-ratio count), NOT the diverging ``sqrt(2 ln K_Rice)``.
(b) the zero up-crossing count -- predicted to DIVERGE as ``1/sqrt(tau_s)``, tracking Rice's
    one-direction count ``T/(2 pi sqrt(tau tau_s))``.

These two observables move in OPPOSITE directions under the same sweep -- that is the whole fork, made
visible for free (no learning, no afferents).

Method: circulant-embedding / spectral GP synthesis
---------------------------------------------------
A stationary GP on a uniform grid is synthesized exactly by the spectral (circulant-embedding) method:
take the periodogram of the covariance over a padded ring of length ``M >= 2 n_grid``, form the
non-negative spectral weights ``sqrt(M * lambda_k)`` with ``lambda_k = FFT(c_ring)/M``, multiply by
complex white noise, inverse-FFT, and keep the first ``n_grid`` real samples. If any embedding
eigenvalue is negative (the covariance is too long for the ring), the ring is doubled until the
embedding is non-negative (Wood-Chan / Dietrich-Newsam). For the double-exp covariance, which decays
on ~tau, ``M = 2 * next_pow2(n_grid)`` is already non-negative for T/tau >> 1.

Comparators logged per row (from the spec, all closed-form):
- ``sqrt_2ln_KPR``   = sqrt(2 ln K_PR),   K_PR   = T (tau + tau_s) / (tau^2 + 3 tau tau_s + tau_s^2)
- ``sqrt_2ln_KRice`` = sqrt(2 ln K_Rice), K_Rice = T / (pi sqrt(tau tau_s))   (two-direction count)
- ``rice_upcross``   = T / (2 pi sqrt(tau tau_s))   (the one-direction up-crossing rate the count tracks)

Run standalone (tiny smoke; CPU, seconds):
    LAB_RUN_DIR=/tmp/v8w uv run python studies/v8_gp_witness.py \\
        T=500 tau_m=15 tau_s_list=15,1.5 n_samples=200 seeds=0

Full witness run (CPU, the spec's grid; minutes):
    LAB_RUN_DIR=/tmp/v8w uv run python studies/v8_gp_witness.py \\
        T=500 tau_m=15 tau_s_list=15,4.5,1.5,0.45,0.15 n_samples=2000 seeds=0 oversample=8
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------------
# PSP kernel + covariance
# --------------------------------------------------------------------------------------------
def psp_v0(tau_m: float, tau_s: float) -> float:
    """Peak-normalisation constant V0 of the double-exp PSP (peak value 1)."""
    t_peak = tau_m * tau_s / (tau_m - tau_s) * math.log(tau_m / tau_s)
    return 1.0 / (math.exp(-t_peak / tau_m) - math.exp(-t_peak / tau_s))


def cov_closed(t: np.ndarray, tau_m: float, tau_s: float) -> np.ndarray:
    """Unit-variance double-exp PSP autocovariance (advance-note closed form).

    ``C_norm(t) = (tau e^{-|t|/tau} - tau_s e^{-|t|/tau_s}) / (tau - tau_s)``, with ``C_norm(0)=1``.
    This is the membrane-difference covariance of section 4 of the prediction note (the
    Ornstein-Uhlenbeck difference whose tau_s->0 limit is the OU process e^{-|t|/tau}).
    """
    a = np.abs(t)
    if abs(tau_m - tau_s) < 1e-9 * tau_m:
        # alpha-function limit tau_s -> tau (L'Hopital): C_norm(t) = (1 + |t|/tau) e^{-|t|/tau}.
        return (1.0 + a / tau_m) * np.exp(-a / tau_m)
    return (tau_m * np.exp(-a / tau_m) - tau_s * np.exp(-a / tau_s)) / (tau_m - tau_s)


def cov_numeric(t: np.ndarray, tau_m: float, tau_s: float, dt: float) -> np.ndarray:
    """Unit-variance autocovariance from the kernel autocorrelation ``int K(u)K(u+|t|)du``.

    Independent cross-check of :func:`cov_closed`: build the causal PSP on a fine grid out to
    ~20 tau, autocorrelate, normalise to 1 at lag 0, and sample at the requested lags ``t`` (which
    are multiples of ``dt``). Used only when ``cov=numeric`` is passed.
    """
    if abs(tau_m - tau_s) < 1e-9 * tau_m:
        # PSP V0 is singular at tau_s=tau (alpha-function); the closed form is exact there.
        return cov_closed(t, tau_m, tau_s)
    v0 = psp_v0(tau_m, tau_s)
    u_max = 20.0 * tau_m
    n = int(round(u_max / dt)) + 1
    u = np.arange(n) * dt
    k = v0 * (np.exp(-u / tau_m) - np.exp(-u / tau_s))  # causal PSP, K(u>=0)
    ac = np.correlate(k, k, mode="full")               # symmetric autocorrelation
    ac = ac[ac.size // 2:]                              # lags 0, dt, 2dt, ...
    ac = ac / ac[0]                                     # unit variance
    idx = np.round(np.abs(t) / dt).astype(int)
    idx = np.clip(idx, 0, ac.size - 1)
    return ac[idx]


# --------------------------------------------------------------------------------------------
# Stationary-GP synthesis by circulant embedding (spectral method)
# --------------------------------------------------------------------------------------------
def synth_gp(
    c0: np.ndarray,     # covariance at non-negative lags 0, dt, ..., (n_grid-1) dt
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize ``n_samples`` paths of the stationary unit-variance GP with autocovariance ``c0``.

    Circulant embedding (Wood-Chan 1994 / Dietrich-Newsam 1997): embed ``c0`` in a circulant of ring
    length ``M = 2 (n_grid-1)`` (mirrored), take the real FFT eigenvalues, and -- if all are >= 0 --
    filter complex white noise by ``sqrt(lambda)``. The first ``n_grid`` samples of the inverse FFT
    are an exact draw of the GP on the grid. If an eigenvalue is negative (ring too short for the
    covariance support) the ring is doubled and ``c0`` zero-padded until non-negative. Returns an
    ``(n_samples, n_grid)`` real array.
    """
    n_grid = c0.shape[0]
    # Wood-Chan minimal embedding: circulant of length m = 2(n_grid-1) whose first column is the
    # covariance reflected, [c0, c1, ..., c_{n-1}, c_{n-2}, ..., c1]. If any FFT eigenvalue is
    # negative the covariance support exceeds the ring; pad the ring with extra (covariance ~ 0)
    # lags and reflect, doubling until the embedding is non-negative (Dietrich-Newsam).
    pad = 0
    for _ in range(10):
        n_ext = n_grid + pad
        # Extend the covariance with zeros for lags beyond what we have (C ~ 0 there for T >> tau).
        c_ext = np.zeros(n_ext)
        c_ext[:n_grid] = c0
        ring = np.concatenate([c_ext, c_ext[-2:0:-1]])  # length 2(n_ext-1), symmetric
        lam = np.fft.fft(ring).real      # eigenvalues of the circulant (covariance even -> real)
        if lam.min() >= -1e-10:
            lam = np.clip(lam, 0.0, None)
            m = ring.shape[0]
            break
        pad = max(1, n_ext - n_grid) * 2 + 64
    else:
        raise RuntimeError("circulant embedding did not become non-negative; covariance too long")

    # Filter complex white noise: V_k = sqrt(lam_k) (a_k + i b_k), a,b ~ N(0,1); ifft -> two
    # independent real GP rings per draw (real & imag parts). We keep the real part; to get
    # n_samples paths we synthesize ceil(n_samples/2) rings and take both real & imag.
    n_rings = (n_samples + 1) // 2
    z = rng.standard_normal((n_rings, m)) + 1j * rng.standard_normal((n_rings, m))
    spec = z * np.sqrt(lam)[None, :]
    rings = np.fft.fft(spec, axis=1) / math.sqrt(m)     # (n_rings, m) complex, var(real)=var(imag)=C(0)
    paths = np.empty((2 * n_rings, n_grid))
    paths[0::2] = rings[:, :n_grid].real
    paths[1::2] = rings[:, :n_grid].imag
    return paths[:n_samples]


# --------------------------------------------------------------------------------------------
# Observables
# --------------------------------------------------------------------------------------------
def e_max(paths: np.ndarray) -> tuple[float, float]:
    """``E[max_t V]`` and its standard error over the sample paths."""
    mx = paths.max(axis=1)
    return float(mx.mean()), float(mx.std(ddof=1) / math.sqrt(mx.shape[0]))


def upcross_count(paths: np.ndarray) -> tuple[float, float]:
    """Mean zero UP-crossing count per path (V<0 -> V>=0 transitions) and its standard error."""
    neg = paths[:, :-1] < 0.0
    pos = paths[:, 1:] >= 0.0
    counts = (neg & pos).sum(axis=1).astype(float)
    return float(counts.mean()), float(counts.std(ddof=1) / math.sqrt(counts.shape[0]))


# --------------------------------------------------------------------------------------------
# Closed-form comparators (the spec's predicted axes)
# --------------------------------------------------------------------------------------------
def k_pr(T: float, tau_m: float, tau_s: float) -> float:
    """Participation-ratio effective count ``K_PR = T(tau+tau_s)/(tau^2+3 tau tau_s+tau_s^2)``."""
    return T * (tau_m + tau_s) / (tau_m**2 + 3 * tau_m * tau_s + tau_s**2)


def k_rice_twodir(T: float, tau_m: float, tau_s: float) -> float:
    """Two-direction Rice crossing count ``K_Rice = T / (pi sqrt(tau tau_s))``."""
    return T / (math.pi * math.sqrt(tau_m * tau_s))


def rice_upcross(T: float, tau_m: float, tau_s: float) -> float:
    """One-direction Rice UP-crossing rate ``T / (2 pi sqrt(tau tau_s))`` (the measured-count target)."""
    return T / (2 * math.pi * math.sqrt(tau_m * tau_s))


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------
def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))

    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    T = float(ov.get("T", "500"))
    tau_m = float(ov.get("tau_m", "15"))
    tau_s_list = [float(x) for x in ov.get("tau_s_list", "15,4.5,1.5,0.45,0.15").split(",")]
    oversample = float(ov.get("oversample", "8"))
    n_samples = int(ov.get("n_samples", "2000"))
    cov_mode = ov.get("cov", "closed")  # 'closed' (exact) | 'numeric' (kernel-autocorr cross-check)
    if "seeds" in ov:
        seeds = [int(x) for x in ov["seeds"].split(",") if x != ""]
    else:
        seeds = list(range(int(ov.get("n_seeds", "1"))))

    ou_plateau = math.sqrt(2.0 * math.log(T / tau_m))
    print(
        f"V8 GP witness | T={T} tau_m={tau_m} tau_s_list={tau_s_list} oversample={oversample} "
        f"n_samples={n_samples} cov={cov_mode} seeds={seeds} | OU plateau sqrt(2 ln(T/tau))={ou_plateau:.3f}",
        flush=True,
    )

    started = time.time()
    rows: list[dict[str, float]] = []
    for tau_s in tau_s_list:
        dt = tau_s / oversample
        n_grid = int(round(T / dt)) + 1
        lags = np.arange(n_grid) * dt
        if cov_mode == "numeric":
            c0 = cov_numeric(lags, tau_m, tau_s, dt)
        else:
            c0 = cov_closed(lags, tau_m, tau_s)
        sq_pr = math.sqrt(2.0 * math.log(k_pr(T, tau_m, tau_s)))
        sq_rice = math.sqrt(2.0 * math.log(k_rice_twodir(T, tau_m, tau_s)))
        rice_up = rice_upcross(T, tau_m, tau_s)
        for seed in seeds:
            # Per-cell seeded RNG: SeedSequence(master, seed, round(1000 tau_s)) -> reproducible and
            # scheduling-independent across the tau_s grid and shards.
            ss = np.random.SeedSequence([master_seed, seed, round(1000 * tau_s)])
            rng = np.random.default_rng(ss)
            paths = synth_gp(c0, n_samples, rng)
            emax, emax_se = e_max(paths)
            upc, upc_se = upcross_count(paths)
            rows.append({
                "seed": seed,
                "tau_m": tau_m,
                "tau_s": tau_s,
                "T": T,
                "n_grid": n_grid,
                "dt": dt,
                "n_samples": n_samples,
                "e_max_V": emax,
                "e_max_V_se": emax_se,
                "upcross_count": upc,
                "upcross_count_se": upc_se,
                "sqrt_2ln_KPR": sq_pr,
                "sqrt_2ln_KRice": sq_rice,
                "K_PR": k_pr(T, tau_m, tau_s),
                "K_Rice": k_rice_twodir(T, tau_m, tau_s),
                "rice_upcross": rice_up,
                "ou_plateau": ou_plateau,
                "cov": cov_mode,
            })
            print(
                f"  tau_s={tau_s:7.3f} r={tau_s/tau_m:6.4f} n_grid={n_grid:6d} seed={seed} | "
                f"E[max V]={emax:.3f}+/-{emax_se:.3f} (PR {sq_pr:.3f}, Rice {sq_rice:.3f}) | "
                f"upcross={upc:.2f}+/-{upc_se:.2f} (Rice up {rice_up:.2f})",
                flush=True,
            )

    elapsed = time.time() - started
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = None
    results = {
        "experiment": "v8_gp_witness",
        "git_sha": git_sha,
        "versions": {"numpy": np.__version__, "python": sys.version.split()[0]},
        "params": {
            "master_seed": master_seed, "T": T, "tau_m": tau_m, "tau_s_list": tau_s_list,
            "oversample": oversample, "n_samples": n_samples, "cov": cov_mode, "seeds": seeds,
            "ou_plateau": ou_plateau,
        },
        "rows": rows,
        "elapsed_seconds": elapsed,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    if rows:
        with (run_dir / "results.csv").open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
    print(f"done in {elapsed:.1f}s -> {run_dir}/results.json ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
