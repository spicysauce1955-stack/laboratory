"""KERNEL-GENERIC free witness ($0, in-memory) for the cross-kernel capacity-universality study.

Self-contained (numpy only; CPU; **NO torch**). For any temporal PSP kernel (the same family the
capacity study ``v13_kernel_capacity.py`` sweeps) this synthesizes the stationary Gaussian readout
``V(t)`` it induces and checks, from realized paths, that the spectral counting layer behaves as the
closed-form predicts -- BEFORE spending a cent of GPU on the capacity sweep. It mirrors the structure of
``../../sinusoidal_capacity/witness_gate/witness_gate.py:run_gate`` and reuses the GP machinery of
``../../tempotron_capacity/studies/v8_gp_witness.py`` (``cov_numeric``/``synth_gp``/``e_max``/
``upcross_count``) and the counters of ``witness_gate.py`` (``count_crossings``/
``count_turning_and_maxima``).

THE MODEL
---------
A tempotron driven by a dense random spike train has, by the CLT, a stationary zero-mean Gaussian
membrane potential ``V(t)`` whose autocovariance is the PSP-kernel autocorrelation

    C(s) = integral K(u) K(u + |s|) du ,    normalised to C(0)=1 (unit variance; the gauge the
                                            capacity problem absorbs into the threshold).

So the kernel SHAPE alone fixes V's law. We synthesize V by circulant embedding (Wood-Chan) of C and
measure the spectral observables that drive capacity.

WHAT IS CHECKED (per smooth kernel, TOL 5%)
-------------------------------------------
- zero-crossing K_eff   : measured pi*E[#crossings] vs ``Keff_rice = T sqrt(lambda_2/lambda_0)``;
- local-maxima count    : measured E[#maxima] vs ``N_max = (T/2pi) sqrt(lambda_4/lambda_2)``;
- extreme value E[max V]: measured vs the Gumbel scale ``sqrt(2 ln Keff_rice)`` (the EVT capacity driver).

THE rect SATURATION DIAGNOSTIC (``--kernel rect --dt-sweep``)
------------------------------------------------------------
The boxcar's spectrum is a sinc^2, so lambda_2 (and hence ``Keff_rice``) DIVERGES as the grid is refined
(dt -> 0): the discretization is the regularization. The witness runs rect at several dt and reports

    (i) the scaling exponent of measured K_eff vs dt -- expected ~ -1/2, and
    (ii) that E[max V] PLATEAUS as dt -> 0 (the participation-ratio count stays finite while the Rice
         count diverges) -- the #4-style "expressiveness saturates" evidence, lifted to the rect kernel.

Run standalone (seconds; CPU):
    uv run python kernel_witness.py                       # all smooth kernels, PASS/FAIL table
    uv run python kernel_witness.py --kernel gauss        # one kernel
    uv run python kernel_witness.py --kernel rect --dt-sweep   # the saturation diagnostic
"""

from __future__ import annotations

import argparse
import math

import numpy as np

# ============================================================================================
# Kernel family -- numpy mirror of v13_kernel_capacity.eval_kernel (peak 1 at s=0; rect=1 inside).
# Kept here (not imported) so the witness is a self-contained $0 tool the lab can ship alone.
# ============================================================================================
def eval_kernel(s: np.ndarray, kernel: str, params: dict) -> np.ndarray:
    """``K(s)`` over time lags ``s`` [ms]; peak-normalised to 1 at s=0 (even, or causal for alpha)."""
    s = np.asarray(s, dtype=float)
    if kernel == "sinc":
        omega_b = params["omega_b"]
        n_lobes = params.get("n_lobes", 6)
        half = n_lobes * math.pi / omega_b
        x = omega_b * s
        sinc = np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))
        xw = math.pi * s / half
        win = np.where(np.abs(xw) < 1e-12, 1.0, np.sin(xw) / np.where(xw == 0, 1.0, xw))
        k = sinc * win
        return np.where(np.abs(s) <= half, k, 0.0)
    if kernel == "gauss":
        sigma = params["sigma"]
        return np.exp(-(s * s) / (2.0 * sigma * sigma))
    if kernel == "gabor":
        sigma = params["sigma"]
        omega0 = params["omega0"]
        return np.exp(-(s * s) / (2.0 * sigma * sigma)) * np.cos(omega0 * s)
    if kernel == "alpha":
        tau = params["tau"]
        sc = np.clip(s, 0.0, None)
        raw = (sc / tau) * np.exp(1.0 - sc / tau)
        return np.where(s >= 0.0, raw, 0.0)
    if kernel == "rect":
        tau = params["tau"]
        return np.where(np.abs(s) <= 0.5 * tau, 1.0, 0.0)
    if kernel == "sinusoid":
        omega0 = params["omega0"]
        n_lobes_win = params.get("n_lobes_win", 6)
        half = n_lobes_win * math.pi / max(omega0, 1e-9)
        xw = math.pi * s / half
        win = np.where(np.abs(xw) < 1e-12, 1.0, np.sin(xw) / np.where(xw == 0, 1.0, xw))
        k = np.cos(omega0 * s) * win
        return np.where(np.abs(s) <= half, k, 0.0)
    raise ValueError(f"unknown kernel {kernel!r}")


# ============================================================================================
# Covariance from kernel autocorrelation (cov_numeric spirit of v8_gp_witness, generic kernel).
# ============================================================================================
def cov_from_kernel(kernel: str, params: dict, dt: float, n_grid: int,
                    support_ms: float) -> np.ndarray:
    """Unit-variance autocovariance ``C(s)=int K(u)K(u+|s|)du / C(0)`` at lags 0,dt,...,(n_grid-1)dt.

    Build the kernel on a fine symmetric grid out to +/- ``support_ms``, autocorrelate (FFT-equivalent
    np.correlate), normalise to 1 at lag 0, and sample at the requested lags. This is exactly the
    ``v8_gp_witness.cov_numeric`` construction generalised to an arbitrary kernel.
    """
    n_u = int(round(support_ms / dt)) + 1
    u = np.arange(-n_u, n_u + 1) * dt
    k = eval_kernel(u, kernel, params)
    ac = np.correlate(k, k, mode="full")
    ac = ac[ac.size // 2:]          # lags 0, dt, 2dt, ...
    if ac[0] <= 0:
        raise RuntimeError("zero-power kernel; cannot normalise covariance")
    ac = ac / ac[0]
    lags = np.arange(n_grid)
    return ac[np.clip(lags, 0, ac.size - 1)]


# ============================================================================================
# Spectral moments / predicted axes -- numpy mirror of v13.kernel_spectral_axes.
# ============================================================================================
def spectral_axes(kernel: str, params: dict, T: float, dt: float, n_grid: int) -> dict:
    """``lambda_{0,2,4}``, ``Keff_rice``, ``N_max``, ``D_PR``, ``kappa_S`` from |K_hat|^2 on the grid."""
    n_half = n_grid - 1
    lags = np.arange(-n_half, n_half + 1) * dt
    k = eval_kernel(lags, kernel, params)
    n = k.size
    Khat = np.fft.rfft(k) * dt
    f = np.fft.rfftfreq(n, d=dt)
    omega = 2.0 * math.pi * f
    S = np.abs(Khat) ** 2
    lambda_0 = float(np.trapz(S, omega))
    lambda_2 = float(np.trapz(omega**2 * S, omega))
    lambda_4 = float(np.trapz(omega**4 * S, omega))
    bandwidth = math.sqrt(lambda_2 / lambda_0)
    Keff_rice = T * bandwidth
    N_max = (T / (2.0 * math.pi)) * math.sqrt(lambda_4 / lambda_2)
    mean_omega = float(np.trapz(omega * S, omega) / lambda_0)
    D_PR = (T / math.pi) * mean_omega
    kappa_S = lambda_4 * lambda_0 / (lambda_2 ** 2)
    return {"lambda_0": lambda_0, "lambda_2": lambda_2, "lambda_4": lambda_4,
            "Keff_rice": Keff_rice, "N_max": N_max, "D_PR": D_PR, "kappa_S": kappa_S,
            "bandwidth": bandwidth}


# ============================================================================================
# GP synthesis (circulant embedding) -- byte-faithful port of v8_gp_witness.synth_gp.
# ============================================================================================
def synth_gp(c0: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Synthesize ``n_samples`` paths of the stationary unit-variance GP with autocovariance ``c0``."""
    n_grid = c0.shape[0]
    pad = 0
    for _ in range(12):
        n_ext = n_grid + pad
        c_ext = np.zeros(n_ext)
        c_ext[:n_grid] = c0
        ring = np.concatenate([c_ext, c_ext[-2:0:-1]])
        lam = np.fft.fft(ring).real
        if lam.min() >= -1e-10:
            lam = np.clip(lam, 0.0, None)
            m = ring.shape[0]
            break
        pad = max(1, n_ext - n_grid) * 2 + 64
    else:
        raise RuntimeError("circulant embedding did not become non-negative; covariance too long")
    n_rings = (n_samples + 1) // 2
    z = rng.standard_normal((n_rings, m)) + 1j * rng.standard_normal((n_rings, m))
    spec = z * np.sqrt(lam)[None, :]
    rings = np.fft.fft(spec, axis=1) / math.sqrt(m)
    paths = np.empty((2 * n_rings, n_grid))
    paths[0::2] = rings[:, :n_grid].real
    paths[1::2] = rings[:, :n_grid].imag
    return paths[:n_samples]


# ============================================================================================
# Observables -- ports of v8_gp_witness (e_max) and witness_gate (counters).
# ============================================================================================
def e_max(paths: np.ndarray) -> tuple[float, float]:
    mx = paths.max(axis=1)
    return float(mx.mean()), float(mx.std(ddof=1) / math.sqrt(mx.shape[0]))


def count_crossings(V: np.ndarray) -> np.ndarray:
    signs = np.signbit(V)
    return np.sum(signs[:, 1:] != signs[:, :-1], axis=1)


def count_turning_and_maxima(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(V, axis=1)
    left, right = d[:, :-1], d[:, 1:]
    turning = (left * right) < 0.0
    maxima = (left > 0.0) & (right < 0.0)
    return turning.sum(axis=1), maxima.sum(axis=1)


# ============================================================================================
# Default kernel knobs (a single representative width per kernel; the witness probes the SHAPE,
# not a target K_eff -- we just need a band that resolves cleanly on the grid).
# ============================================================================================
def default_params(kernel: str) -> dict:
    return {
        "sinc": {"omega_b": 0.3, "n_lobes": 6, "window": "lanczos"},
        "gauss": {"sigma": 8.0},
        "gabor": {"sigma": 12.0, "omega0": 0.5},          # Q = omega0*sigma = 6 (band-pass regime)
        "alpha": {"tau": 8.0},
        "rect": {"tau": 20.0},
        "sinusoid": {"omega0": 0.2, "n_lobes_win": 8},
    }[kernel]


def _bandwidth_estimate(kernel: str, params: dict) -> float:
    """Coarse rms bandwidth (rad/ms) to size the Nyquist grid -- mirrors v13._coarse_bandwidth."""
    if kernel == "sinc":
        return params["omega_b"]
    if kernel == "sinusoid":
        return max(params["omega0"], 1e-6)
    if kernel == "gauss":
        return 1.0 / params["sigma"]
    if kernel == "gabor":
        return params["omega0"] + 1.0 / params["sigma"]
    if kernel == "alpha":
        return 1.0 / params["tau"]
    if kernel == "rect":
        return 2.0 * math.pi / params["tau"]
    return 1.0


def _rel_err(meas: float, pred: float) -> float:
    return abs(meas - pred) / abs(pred) if pred != 0 else float("inf")


def gumbel_emax(K: float) -> float:
    """First-order EVT mean of the max of ~K (weakly dependent) unit-variance Gaussians.

    The bare leading scale ``sqrt(2 ln K)`` systematically OVER-predicts E[max] at finite K; the
    standard Gumbel correction (Leadbetter-Lindgren-Rootzen; the Cramer expansion) is

        E[max] ~ a_K + gamma / a_K ,   a_K = sqrt(2 ln K) - (ln ln K + ln 4pi)/(2 sqrt(2 ln K)),

    with Euler-Mascheroni gamma = 0.5772. This is the correct comparator for the realized extreme of a
    band-limited GP whose number of independent local maxima is ~K (Rice). ``K`` here is the count of
    up-crossings / local maxima (= K_eff_rice / 2 for a symmetric band), the EVT exceedance count.
    """
    K = max(K, 1.0001)
    sq = math.sqrt(2.0 * math.log(K))
    a_K = sq - (math.log(math.log(K)) + math.log(4.0 * math.pi)) / (2.0 * sq)
    return a_K + 0.5772156649 / a_K


# ============================================================================================
# The smooth-kernel gate (mirrors witness_gate.run_gate structure)
# ============================================================================================
def run_smooth_gate(kernels: list[str], T: float = 500.0, M: int = 4000,
                    oversample: float = 20.0, seed: int = 0,
                    verbose: bool = True) -> tuple[bool, list[dict]]:
    """Per smooth kernel: K_eff, N_max, E[max V] measured-vs-predicted (TOL 5%)."""
    TOL = 0.05
    rows: list[dict] = []
    for kernel in kernels:
        if kernel == "rect":
            continue  # rect is not a smooth-gate kernel (handled by the dt-sweep diagnostic)
        params = default_params(kernel)
        bw = _bandwidth_estimate(kernel, params)
        dt = math.pi / (oversample * bw)
        n_grid = int(round(T / dt)) + 1
        # Covariance support: out to where the kernel autocorrelation has decayed. Use a generous
        # multiple of the kernel time-scale (and at least the window for compact kernels).
        support_ms = max(T, 40.0 / bw)
        ax = spectral_axes(kernel, params, T, dt, n_grid)
        c0 = cov_from_kernel(kernel, params, dt, n_grid, support_ms)
        rng = np.random.default_rng([seed, hash(kernel) & 0xFFFF])
        paths = synth_gp(c0, M, rng)

        cross = count_crossings(paths)
        _turn, maxima = count_turning_and_maxima(paths)
        keff_meas = math.pi * float(cross.mean())
        nmax_meas = float(maxima.mean())
        emax_meas, emax_se = e_max(paths)

        keff_pred = ax["Keff_rice"]
        nmax_pred = ax["N_max"]
        # EVT max scale: K = number of independent local maxima (Rice up-crossing count =
        # K_eff_rice/2 for a symmetric band), with the finite-K Gumbel correction.
        emax_pred = gumbel_emax(0.5 * ax["Keff_rice"])

        # N_max via lambda_4 is grid-fragile for kernels with a derivative discontinuity at lag 0
        # (alpha is causal -> autocovariance kink at s=0 -> lambda_4 ill-defined / grid-divergent).
        # We gate N_max only for the band-limited / infinitely-smooth kernels; for alpha it is logged
        # as information (pass=None) because the Rice maxima formula does not apply to a kinked C(s).
        nmax_gated = kernel not in ("alpha",)

        # The crossing/maxima counts are EXACT Rice identities -> strict 5% MC tolerance. E[max V] is a
        # first-order EVT asymptotic (the corrected Gumbel mean still carries an O(1/ln K) bias plus an
        # O(1) effective-count factor that depends on the spectral shape for a weakly-dependent GP), so
        # it is a SCALE/trend witness gated at a looser, physically-honest 10%.
        TOL_EVT = 0.10
        for name, meas, pred, gated, tol in (
            ("K_eff (pi*E[#cross])", keff_meas, keff_pred, True, TOL),
            ("N_max  (E[#maxima])", nmax_meas, nmax_pred, nmax_gated, TOL),
            ("E[max V] (Gumbel)", emax_meas, emax_pred, True, TOL_EVT),
        ):
            err = _rel_err(meas, pred)
            rows.append({
                "kernel": kernel, "check": name, "predicted": pred, "measured": meas,
                "rel_err": err, "tol": tol, "pass": (err <= tol) if gated else None,
                "kappa_S": ax["kappa_S"], "D_PR": ax["D_PR"],
            })
    gated = [r for r in rows if r["pass"] is not None]
    overall = all(r["pass"] for r in gated)
    if verbose:
        _print_smooth(rows, overall, TOL, oversample, M)
    return overall, rows


def _print_smooth(rows, overall, TOL, oversample, M):
    print(f"\nKernel-universality witness (smooth kernels) | tol={TOL:.0%} (counts) / 10% (E[maxV]) rel, "
          f"oversample={oversample:g}x Nyquist, M={M} paths\n")
    hdr = f"{'kernel':<10}{'check':<24}{'predicted':>12}{'measured':>12}{'rel_err':>10}{'verdict':>9}"
    print(hdr); print("-" * len(hdr))
    last = None
    for r in rows:
        if r["kernel"] != last:
            if last is not None:
                print()
            last = r["kernel"]
        verdict = "PASS" if r["pass"] else ("FAIL" if r["pass"] is False else "(info)")
        print(f"{r['kernel']:<10}{r['check']:<24}{r['predicted']:>12.4f}{r['measured']:>12.4f}"
              f"{r['rel_err']:>10.4f}{verdict:>9}")
    print("-" * len(hdr))
    print(f"\nSMOOTH-KERNEL OVERALL: {'PASS' if overall else 'FAIL'}\n")


# ============================================================================================
# The rect dt-sweep saturation diagnostic (#4-style witness lifted to the boxcar kernel)
# ============================================================================================
def run_rect_dt_sweep(T: float = 500.0, M: int = 4000, tau: float = 20.0,
                      dt_list: list[float] | None = None, seed: int = 0,
                      verbose: bool = True) -> dict:
    """rect at several dt: report the K_eff vs dt exponent (~ -1/2) and the E[max V] plateau.

    The boxcar spectrum is sinc^2(omega tau/2), so lambda_2 ~ int omega^2 sinc^2 domega diverges; on a
    grid the integral is cut at the Nyquist omega_max = pi/dt, giving lambda_2 ~ 1/dt and hence
    ``Keff_rice = T sqrt(lambda_2/lambda_0) ~ dt^{-1/2}``. Meanwhile E[max V] is governed by the
    FINITE participation count (the box has a well-defined number of independent plateaus over [0,T]),
    so it PLATEAUS as dt -> 0. The two diverge -- the rect analogue of #4's rise-time saturation.
    """
    if dt_list is None:
        dt_list = [2.0, 1.0, 0.5, 0.25, 0.125]
    params = {"tau": tau}
    recs: list[dict] = []
    for dt in dt_list:
        n_grid = int(round(T / dt)) + 1
        support_ms = max(T, 4.0 * tau)
        ax = spectral_axes("rect", params, T, dt, n_grid)
        c0 = cov_from_kernel("rect", params, dt, n_grid, support_ms)
        rng = np.random.default_rng([seed, int(round(1000 * dt))])
        paths = synth_gp(c0, M, rng)
        cross = count_crossings(paths)
        keff_meas = math.pi * float(cross.mean())
        emax_meas, emax_se = e_max(paths)
        recs.append({
            "dt": dt, "n_grid": n_grid,
            "Keff_rice_pred": ax["Keff_rice"], "Keff_meas": keff_meas,
            "D_PR": ax["D_PR"], "emax": emax_meas, "emax_se": emax_se,
        })

    # Scaling exponent: fit log(Keff_meas) = a + p*log(dt). Expect p ~ -1/2.
    logdt = np.log(np.array([r["dt"] for r in recs]))
    logk = np.log(np.array([r["Keff_meas"] for r in recs]))
    p_meas = float(np.polyfit(logdt, logk, 1)[0])
    logkpred = np.log(np.array([r["Keff_rice_pred"] for r in recs]))
    p_pred = float(np.polyfit(logdt, logkpred, 1)[0])

    # Plateau evidence: E[max V] spread across the finest half of the dt grid (should be ~flat).
    emax = np.array([r["emax"] for r in recs])
    emax_se = np.array([r["emax_se"] for r in recs])
    fine = emax[len(emax) // 2:]
    plateau_rel_spread = float((fine.max() - fine.min()) / fine.mean())
    # Compare finest-dt E[max V] vs coarsest: if it were tracking the diverging Keff it would GROW.
    emax_growth_ratio = float(emax[-1] / emax[0])

    out = {
        "records": recs, "exponent_measured": p_meas, "exponent_predicted": p_pred,
        "plateau_rel_spread": plateau_rel_spread, "emax_growth_ratio": emax_growth_ratio,
    }
    if verbose:
        _print_rect(out, T, tau, M)
    return out


def _print_rect(out: dict, T: float, tau: float, M: int):
    print(f"\nrect dt-sweep saturation diagnostic | T={T} tau={tau} M={M}\n")
    hdr = f"{'dt':>8}{'n_grid':>9}{'Keff_pred':>12}{'Keff_meas':>12}{'D_PR':>10}{'E[maxV]':>12}"
    print(hdr); print("-" * len(hdr))
    for r in out["records"]:
        print(f"{r['dt']:>8.4f}{r['n_grid']:>9d}{r['Keff_rice_pred']:>12.3f}{r['Keff_meas']:>12.3f}"
              f"{r['D_PR']:>10.3f}{r['emax']:>12.4f}")
    print("-" * len(hdr))
    p = out["exponent_measured"]
    p_ok = abs(p - (-0.5)) <= 0.12  # within +/-0.12 of -1/2
    plat_ok = out["plateau_rel_spread"] <= 0.05
    print(f"\nK_eff vs dt exponent: measured={p:.3f}  predicted={out['exponent_predicted']:.3f}  "
          f"(target -0.5)  -> {'PASS' if p_ok else 'FAIL'}")
    print(f"E[max V] plateau: rel-spread(fine half)={out['plateau_rel_spread']:.4f} "
          f"(<=0.05 plateau), growth(finest/coarsest)={out['emax_growth_ratio']:.3f} "
          f"-> {'PASS (saturates)' if plat_ok else 'FAIL (drifts)'}")
    print(f"\nDIAGNOSIS: K_eff_rice DIVERGES as dt->0 (exp ~ -1/2) while E[max V] SATURATES "
          f"-> the boxcar's expressiveness is set by the FINITE participation count D_PR, "
          f"not the (grid-divergent) Rice crossing count. [#4-style saturation]\n")


# ============================================================================================
# THE DECISIVE SIGN TEST (T2): E[max V] across shapes at a COMMON matched K_eff_rice.
# Tunes each kernel's width so the (numeric) Keff_rice hits a target, synthesizes the GP, and
# measures E[max V]. Pre-registered prediction: gauss (low-pass) highest; gabor falls monotonically
# as Q rises; sinusoid (degenerate) lowest -> narrowband correlation SUPPRESSES the EVT lift, the
# mechanism behind alpha_c(Gabor) <= alpha_c(Gaussian). A non-monotone bump in Q would predict a
# capacity sign-flip and MUST be flagged before the sweep.
# ============================================================================================
def _grid_for(kernel: str, params: dict, T: float, oversample: float = 20.0) -> tuple[float, int]:
    bw = _bandwidth_estimate(kernel, params)
    dt = math.pi / (oversample * bw)
    n_grid = int(round(T / dt)) + 1
    return dt, n_grid


def _keff_rice(kernel: str, params: dict, T: float, oversample: float = 20.0) -> float:
    dt, n_grid = _grid_for(kernel, params, T, oversample)
    return spectral_axes(kernel, params, T, dt, n_grid)["Keff_rice"]


def _bisect_width(f, lo: float, hi: float, target: float, iters: int = 60) -> float:
    """Solve f(w)=target for positive width w, f monotone; geometric bisection with bracket expand."""
    flo, fhi = f(lo) - target, f(hi) - target
    tries = 0
    while flo * fhi > 0 and tries < 40:
        lo *= 0.5
        hi *= 1.5
        flo, fhi = f(lo) - target, f(hi) - target
        tries += 1
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        fmid = f(mid) - target
        if abs(fmid) <= 1e-4 * target:
            return mid
        if (fmid > 0) == (flo > 0):
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid
    return math.sqrt(lo * hi)


def _tune_params(kernel: str, target_keff: float, T: float, Q: float | None = None) -> dict:
    if kernel == "gauss":
        sigma = _bisect_width(lambda w: _keff_rice("gauss", {"sigma": w}, T), 1.0, 200.0, target_keff)
        return {"sigma": sigma}
    if kernel == "gabor":  # fix Q = omega0*sigma; vary carrier omega0 (sigma = Q/omega0)
        omega0 = _bisect_width(
            lambda w: _keff_rice("gabor", {"omega0": w, "sigma": Q / w}, T),
            target_keff / (2.0 * T), 4.0 * target_keff / T, target_keff)
        return {"omega0": omega0, "sigma": Q / omega0}
    if kernel == "sinusoid":
        omega0 = _bisect_width(
            lambda w: _keff_rice("sinusoid", {"omega0": w, "n_lobes_win": 8}, T),
            target_keff / (4.0 * T), 8.0 * target_keff / T, target_keff)
        return {"omega0": omega0, "n_lobes_win": 8}
    raise ValueError(kernel)


def realize_spectral(kernel: str, params: dict, T: float, dt: float, n_grid: int,
                     M: int, rng: np.random.Generator, m_freqs: int = 600) -> np.ndarray:
    """Robust unit-variance GP synthesis via the spectral representation (freqs ~ S(omega)).

    Mirrors ``sinusoidal_capacity/witness_gate.realize_V`` (the proven #6 method): draw m frequencies
    with density proportional to the kernel power spectrum S(omega)=|K_hat|^2 and sum equal-weight
    cosines, V(t) = (1/sqrt(m)) sum_k [A_k cos(w_k t)+B_k sin(w_k t)], Var(V)=1. Unlike circulant
    embedding of a truncated covariance, this is unconditionally valid (S>=0) and handles NARROWBAND
    (high-Q) spectra, which is exactly where the sign test lives.
    """
    n_half = n_grid - 1
    lags = np.arange(-n_half, n_half + 1) * dt
    k = eval_kernel(lags, kernel, params)
    Khat = np.fft.rfft(k) * dt
    omega = 2.0 * math.pi * np.fft.rfftfreq(k.size, d=dt)
    S = np.abs(Khat) ** 2
    tot = S.sum()
    if tot <= 0:
        raise RuntimeError("zero-power spectrum")
    p = S / tot
    idx = rng.choice(omega.size, size=m_freqs, p=p)
    d_omega = float(omega[1] - omega[0]) if omega.size > 1 else 0.0
    freqs = np.clip(omega[idx] + rng.uniform(-0.5, 0.5, size=m_freqs) * d_omega, 0.0, None)
    t = np.arange(n_grid) * dt
    A = rng.standard_normal((M, m_freqs))
    B = rng.standard_normal((M, m_freqs))
    return (A @ np.cos(np.outer(freqs, t)) + B @ np.sin(np.outer(freqs, t))) / math.sqrt(m_freqs)


def run_sign_test(targets=(16.0, 32.0), T: float = 500.0, M: int = 8000, seed: int = 0,
                  verbose: bool = True) -> dict:
    # (kernel, Q) plans per target; Q capped so the Gaussian envelope sigma<=~T/3 (else truncated).
    plans = {
        16.0: [("gauss", 0.0), ("gabor", 2.0), ("gabor", 4.0), ("sinusoid", None)],
        32.0: [("gauss", 0.0), ("gabor", 2.0), ("gabor", 4.0), ("gabor", 8.0), ("sinusoid", None)],
    }
    out: dict = {}
    for K in targets:
        plan = plans.get(K, [("gauss", 0.0), ("gabor", 4.0), ("sinusoid", None)])
        rows: list[dict] = []
        for kernel, Q in plan:
            try:
                params = _tune_params(kernel, K, T, Q=Q)
                dt, n_grid = _grid_for(kernel, params, T)
                ax = spectral_axes(kernel, params, T, dt, n_grid)
                rng = np.random.default_rng([seed, int(round(K)), int(round((Q or 0) * 10)),
                                             hash(kernel) & 0xFF])
                # Robust spectral synthesis (handles narrowband; the circulant route fails there).
                paths = realize_spectral(kernel, params, T, dt, n_grid, M, rng)
                emax, se = e_max(paths)
                sigma = params.get("sigma", float("nan"))
                rows.append({
                    "kernel": kernel, "Q": Q, "Keff_meas": ax["Keff_rice"], "D_PR": ax["D_PR"],
                    "kappa_S": ax["kappa_S"], "emax": emax, "emax_se": se, "sigma": sigma,
                    "truncated": (not math.isnan(sigma)) and sigma > T / 3.0,
                })
            except Exception as exc:  # noqa: BLE001 — one bad cell must not kill the witness
                rows.append({"kernel": kernel, "Q": Q, "error": str(exc)})
        out[K] = rows
        if verbose:
            _print_sign(rows, K, M)
    if verbose:
        _print_sign_verdict(out)
    return out


def _print_sign(rows, K, M):
    print(f"\nSIGN TEST (T2): E[max V] at matched K_eff_rice = {K:g} | M={M} paths\n")
    hdr = (f"{'kernel':<10}{'Q':>5}{'Keff_meas':>11}{'D_PR':>9}{'kappa_S':>9}"
           f"{'sigma':>9}{'E[maxV]':>10}{'+/-se':>9}{'trunc':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['kernel']:<10}{(r['Q'] or 0):>5g}   ERROR: {r['error']}")
            continue
        q = "lp" if r["Q"] == 0 else ("inf" if r["Q"] is None else f"{r['Q']:g}")
        print(f"{r['kernel']:<10}{q:>5}{r['Keff_meas']:>11.3f}{r['D_PR']:>9.3f}{r['kappa_S']:>9.3f}"
              f"{r['sigma']:>9.2f}{r['emax']:>10.4f}{r['emax_se']:>9.4f}"
              f"{('YES' if r['truncated'] else '-'):>7}")
    print("-" * len(hdr))


def _print_sign_verdict(out):
    print("\nSIGN-TEST VERDICT (pre-registered: gauss highest; E[maxV] falls monotonically with Q; "
          "sinusoid lowest)\n")
    for K, rows in out.items():
        good = [r for r in rows if "error" not in r]
        gauss = next((r for r in good if r["kernel"] == "gauss"), None)
        gabors = sorted([r for r in good if r["kernel"] == "gabor"], key=lambda r: r["Q"])
        sinus = next((r for r in good if r["kernel"] == "sinusoid"), None)
        emax_seq = ([gauss["emax"]] if gauss else []) + [g["emax"] for g in gabors] + (
            [sinus["emax"]] if sinus else [])
        labels = (["gauss"] if gauss else []) + [f"gabor-Q{g['Q']:g}" for g in gabors] + (
            ["sinusoid"] if sinus else [])
        monotone = all(emax_seq[i] >= emax_seq[i + 1] - 3e-3 for i in range(len(emax_seq) - 1))
        order = " > ".join(f"{l}({e:.3f})" for l, e in zip(labels, emax_seq))
        print(f"  K={K:g}: {order}")
        print(f"         monotone-decreasing (supports alpha_c(Gabor)<=alpha_c(Gauss)): "
              f"{'YES' if monotone else 'NO -- FLAG: possible sign-flip / non-monotonicity in Q'}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Kernel-generic capacity witness ($0, in-memory).")
    ap.add_argument("--kernel", default="all",
                    help="one of sinc|gauss|gabor|alpha|rect|sinusoid, or 'all' (full witness).")
    ap.add_argument("--dt-sweep", action="store_true", help="rect dt-sweep saturation diagnostic.")
    ap.add_argument("--sign-test", action="store_true", help="matched-K_eff E[max V] sign test only.")
    ap.add_argument("--T", type=float, default=500.0)
    ap.add_argument("--M", type=int, default=4000)
    ap.add_argument("--oversample", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_all = args.kernel == "all"

    # 'all' runs the COMPLETE witness so a single (remote) job yields every Step-2 artifact.
    if args.sign_test or run_all:
        run_sign_test(T=args.T, M=max(args.M, 8000), seed=args.seed)
        if args.sign_test and not run_all:
            return 0

    if args.dt_sweep or args.kernel == "rect" or run_all:
        run_rect_dt_sweep(T=args.T, M=args.M, seed=args.seed)
        if (args.dt_sweep or args.kernel == "rect") and not run_all:
            return 0

    smooth = ["sinc", "gauss", "gabor", "alpha", "sinusoid"]
    if not run_all:
        if args.kernel not in smooth:
            print(f"(kernel {args.kernel!r} has no smooth gate; use --dt-sweep for rect)")
            return 0
        smooth = [args.kernel]
    ok, _ = run_smooth_gate(smooth, T=args.T, M=args.M, oversample=args.oversample, seed=args.seed)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
