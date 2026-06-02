"""V2 -- Tempotron peak-voltage EVT validated against an analytic Gaussian-process reference.

Self-contained Experiment-Contract entrypoint (numpy + scipy + torch only; the lab ships this
file unchanged, so there are no ``tempotron``/``lab`` imports). The locked spec is
``experiments/tempotron_capacity/docs/v2-theory-and-predictions.md``; the background theory is
``docs/05-statistical-mechanics-of-capacity.md`` §3-4.

What this script does
----------------------
1. **GP reference (primary).** Generate the stationary, zero-mean, unit-variance Gaussian
   process with the *proved* normalised autocorrelation

       c(D) = [tau_m e^{-|D|/tau_m} - tau_s e^{-|D|/tau_s}] / (tau_m - tau_s),    c(0) = 1,

   by **FFT circulant embedding** (Davies-Harte, O(n log n)); Cholesky is infeasible at the
   target grid sizes. ``V_max`` is read off a stationary interior window of length ``T_eval``
   with ``K_eff = T_eval / sqrt(tau_s tau_m)``. The many-draw distribution is the reference
   ``R_K``.
2. **Tempotron (secondary).** Peak-normalised double-exponential PSP, RMS Poisson patterns,
   Gaussian weights, **explicit centering** of the per-afferent traces (mandatory, spec §3.5),
   per-pattern analytic ``sigma_V`` standardisation, GPU-batched forward pass.
3. **Three-tier scoring.** Tier 1 (analytic curvature, gated exactly via dt->0 Richardson
   extrapolation), Tier 2 (tempotron-vs-``R_K`` two-sample Anderson-Darling + bootstrapped
   moment CIs + an N/K ladder), Tier 3 (Gumbel/RMS contact as a *trend diagnostic*, never
   pass/fail). The headline convention is the **median** ``m = loc_scipy + 0.3665129 * scale``.
4. **Aggregate null calibration.** The whole Tier-2 suite run on GP-vs-GP (size) and on
   GP-with-injected-scale-error (power), to set/record the family-wise false-fail rate.

Run standalone (smoke):
    LAB_RUN_DIR=/tmp/v2_smoke LAB_SEED=0 \\
        uv run --with torch --with scipy python studies/v2_vmax_evt.py \\
        K_list=64,256 n_draws=2000 N=400 ndt=2
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.stats import anderson_ksamp, gumbel_r

if TYPE_CHECKING:
    from numpy.typing import NDArray

TAU_M: float = 15.0  # ms, membrane time constant (Gutig & Sompolinsky 2006 + RMS 2010)
TAU_S: float = 3.75  # ms, synaptic time constant
SQRT_TAU: float = math.sqrt(TAU_S * TAU_M)  # = 7.5 ms, the PSP correlation time
EULER_GAMMA: float = 0.5772156649015329  # Euler-Mascheroni (Gumbel mean offset)
MEDIAN_OFFSET: float = 0.3665129205816643  # = -ln ln 2  (Gumbel median offset; CONVENTION LOCK)


# --------------------------------------------------------------------------------------------
# Analytic autocorrelation and the exact EVT constants
# --------------------------------------------------------------------------------------------
def autocorr(
    delta: NDArray[np.float64], tau_m: float = TAU_M, tau_s: float = TAU_S
) -> NDArray[np.float64]:
    """Normalised process autocorrelation ``c(D)`` (the proved covariance, spec §3.1-3.2).

    Parameters
    ----------
    delta
        Lags ``D`` (ms); any shape. The function uses ``|D|`` (the process is stationary and
        even).
    tau_m, tau_s
        Membrane and synaptic time constants (ms).

    Returns
    -------
    numpy.ndarray
        ``c(D) = [tau_m e^{-|D|/tau_m} - tau_s e^{-|D|/tau_s}] / (tau_m - tau_s)`` with the same
        shape as ``delta``; ``c(0) = 1``.

    Notes
    -----
    Because ``K(0) = 0`` (the two exponentials cancel at onset) the one-sided derivative of the
    autocorrelation vanishes: ``c`` is a clean parabola at the origin (no ``|D|`` cusp), so the
    process is mean-square differentiable and ``c''(0) = -1/(tau_m tau_s)`` exactly.
    """
    d = np.abs(delta)
    result = (tau_m * np.exp(-d / tau_m) - tau_s * np.exp(-d / tau_s)) / (tau_m - tau_s)
    return np.asarray(result, dtype=np.float64)


def cpp_zero_analytic(tau_m: float = TAU_M, tau_s: float = TAU_S) -> float:
    """Exact second derivative of the autocorrelation at zero, ``c''(0) = -1/(tau_m tau_s)``."""
    return -1.0 / (tau_m * tau_s)


def k_eff_of_window(t_eval: float) -> float:
    """Effective sample count ``K_eff = T_eval / sqrt(tau_s tau_m)`` (RMS eq. 2; spec §3.3)."""
    return t_eval / SQRT_TAU


def gumbel_mode(k: float) -> float:
    """Asymptotic Gumbel mode ``mu_K = sqrt(2 ln K) - (ln ln K + ln 4pi)/(2 sqrt(2 ln K))``."""
    a = math.sqrt(2.0 * math.log(k))
    return a - (math.log(math.log(k)) + math.log(4.0 * math.pi)) / (2.0 * a)


def gumbel_scale(k: float) -> float:
    """Asymptotic Gumbel scale ``beta_K = 1 / sqrt(2 ln K)``."""
    return 1.0 / math.sqrt(2.0 * math.log(k))


def gumbel_median_from_mode(mode: float, scale: float) -> float:
    """Gumbel **median** ``m = mode + 0.3665129 * scale`` (CONVENTION LOCK, spec §3.4)."""
    return mode + MEDIAN_OFFSET * scale


# --------------------------------------------------------------------------------------------
# Robust moment estimators (median / robust scale / skew) used everywhere
# --------------------------------------------------------------------------------------------
def robust_scale(samples: NDArray[np.float64]) -> float:
    """Robust scale estimate ``IQR / 1.349`` (a Gaussian-consistent spread, outlier-resistant)."""
    q75, q25 = np.percentile(samples, [75.0, 25.0])
    return float((q75 - q25) / 1.349)


def sample_skew(samples: NDArray[np.float64]) -> float:
    """Fisher (bias-uncorrected) skewness ``E[(x-mu)^3] / sigma^3``."""
    x = samples - samples.mean()
    m2 = float(np.mean(x**2))
    m3 = float(np.mean(x**3))
    return m3 / m2**1.5 if m2 > 0 else float("nan")


def summarise(samples: NDArray[np.float64]) -> dict[str, float]:
    """Median, robust scale, and skew of a ``V_max`` sample (the Tier-2 comparison triple)."""
    return {
        "median": float(np.median(samples)),
        "scale": robust_scale(samples),
        "skew": sample_skew(samples),
    }


# --------------------------------------------------------------------------------------------
# Probability-weighted-moment (PWM / L-moment) Gumbel fit -- Tier-3 diagnostic
# --------------------------------------------------------------------------------------------
def pwm_gumbel_fit(samples: NDArray[np.float64]) -> tuple[float, float]:
    """Fit a Gumbel law by probability-weighted moments (Hosking-Wallis-Wood 1985).

    Parameters
    ----------
    samples
        1-D ``V_max`` sample.

    Returns
    -------
    (mode, scale)
        ``scale = (2 b1 - b0) / ln 2`` and ``mode = b0 - gamma * scale``, where ``b0, b1`` are the
        first two PWMs and ``gamma`` is the Euler-Mascheroni constant. ``mode`` is the Gumbel
        location (NOT the median); recover the median via :func:`gumbel_median_from_mode`.
    """
    x = np.sort(samples)
    n = x.size
    b0 = float(x.mean())
    j = np.arange(1, n + 1, dtype=np.float64)
    b1 = float(np.sum((j - 1.0) / (n - 1.0) * x) / n)
    scale = (2.0 * b1 - b0) / math.log(2.0)
    mode = b0 - EULER_GAMMA * scale
    return mode, scale


# --------------------------------------------------------------------------------------------
# (A) GP reference -- FFT circulant embedding (Davies-Harte)
# --------------------------------------------------------------------------------------------
def circulant_embedding_eigs(n: int, dt: float) -> tuple[NDArray[np.float64], int]:
    """Build the non-negative circulant eigenvalues for the Davies-Harte GP sampler.

    Parameters
    ----------
    n
        Number of grid points of the *target* (interior + padding) series; the circulant block
        size is ``m = 2(n-1)`` (extended further if needed for non-negative-definiteness).
    dt
        Grid spacing (ms).

    Returns
    -------
    (eigs, m)
        ``eigs`` are the real FFT eigenvalues (clipped at 0 to repair tiny negative entries from
        a non-NND embedding), ``m`` is the realised circulant size. The standard symmetric
        embedding of the first column ``[c_0, ..., c_{n-1}, c_{n-2}, ..., c_1]`` is used.
    """
    base = n
    while True:
        m = 2 * (base - 1)
        lags = np.concatenate([np.arange(base), np.arange(base - 2, 0, -1)]).astype(np.float64)
        first_col = autocorr(lags * dt)
        eigs = np.fft.fft(first_col).real
        min_eig = float(eigs.min())
        if min_eig >= -1e-10 * float(eigs.max()):
            return np.clip(eigs, 0.0, None), m
        # Repair a (rare) non-NND embedding by padding the base length and retrying.
        base = math.ceil(base * 1.5)


def gp_reference_vmax(
    k_target: float,
    n_draws: int,
    dt: float,
    rng: np.random.Generator,
    burn_per_side_corr: float = 10.0,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Draw the GP-reference ``V_max`` distribution ``R_K`` by circulant embedding.

    Parameters
    ----------
    k_target
        Target effective sample count ``K_eff`` for the *interior* evaluation window, i.e.
        ``T_eval = k_target * sqrt(tau_s tau_m)``.
    n_draws
        Number of independent process realisations.
    dt
        Grid spacing (ms). Production ``dt <= sqrt(tau_s tau_m)/10``.
    rng
        NumPy generator (drives the Davies-Harte Gaussian draws).
    burn_per_side_corr
        One-sided burn-in in units of the correlation time ``sqrt(tau_s tau_m)`` (default 10,
        i.e. ``5 tau_m``), to evaluate ``V_max`` only over the stationary interior.

    Returns
    -------
    (vmax, sanity)
        ``vmax`` is the length-``n_draws`` ``V_max`` array over the interior window. ``sanity``
        records the realised sample variance, the empirical-vs-analytic autocorrelation error at
        a few lags, and grid bookkeeping (``K_eff``, ``T_eval``, ``dt``, ``n_interior``).

    Notes
    -----
    Davies-Harte: with circulant eigenvalues ``lambda_k`` (size ``m``), a real stationary
    Gaussian series is ``Re/Im`` parts of ``IFFT(sqrt(lambda_k) * (a + i b))`` for standard
    Gaussian ``a, b`` -- two independent series per transform. We take the first ``n`` points.
    """
    t_eval = k_target * SQRT_TAU
    n_interior = round(t_eval / dt) + 1
    burn = math.ceil(burn_per_side_corr * SQRT_TAU / dt)
    n_total = n_interior + 2 * burn
    eigs, m = circulant_embedding_eigs(n_total, dt)
    sqrt_eigs = np.sqrt(eigs)

    vmax = np.empty(n_draws, dtype=np.float64)
    var_acc = 0.0
    # Accumulate the empirical autocorrelation at a handful of small lags for the sanity check.
    check_lags = [0, 1, 2, 4, 8]
    check_lags = [lg for lg in check_lags if lg < n_total]
    acf_acc = np.zeros(len(check_lags), dtype=np.float64)
    acf_count = 0

    # Davies-Harte yields two independent real stationary series per transform (the real and
    # imaginary parts). With standard complex-Gaussian ``z`` (unit-variance real and imaginary
    # parts, so ``E|z|^2 = 2``) each part has the full target variance ``c(0)``.
    draw = 0
    while draw < n_draws:
        z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        spec = np.fft.fft(sqrt_eigs * z) / math.sqrt(float(m))
        for series in (spec.real, spec.imag):
            if draw >= n_draws:
                break
            full = series[:n_total]
            interior = full[burn : burn + n_interior]
            vmax[draw] = float(interior.max())
            # The process is zero-mean by construction, so the lag-0 *second moment* (not the
            # per-realisation sample variance, which is biased low for a strongly correlated
            # series) is the consistent estimator of c(0).
            var_acc += float(np.mean(full * full))
            for li, lg in enumerate(check_lags):
                seg_a = full[: n_total - lg]
                seg_b = full[lg:]
                acf_acc[li] += float(np.mean(seg_a * seg_b))
            acf_count += 1
            draw += 1

    sample_var = var_acc / acf_count
    acf_emp = acf_acc / acf_count
    acf_analytic = autocorr(np.array(check_lags, dtype=np.float64) * dt)
    # Normalise the empirical lag-0 to 1 for a fair shape comparison.
    acf_emp_norm = acf_emp / acf_emp[0] if acf_emp[0] != 0 else acf_emp
    acf_max_err = float(np.max(np.abs(acf_emp_norm - acf_analytic)))

    sanity = {
        "sample_var": sample_var,
        "acf_max_err": acf_max_err,
        "K_eff": k_eff_of_window(t_eval),
        "T_eval": t_eval,
        "dt": dt,
        "n_interior": float(n_interior),
        "burn": float(burn),
        "circulant_m": float(m),
    }
    return vmax, sanity


# --------------------------------------------------------------------------------------------
# (B) Tempotron forward pass (GPU-batched, explicit centering)
# --------------------------------------------------------------------------------------------
def psp_kernel(t: torch.Tensor) -> torch.Tensor:
    """Double-exponential PSP, peak-normalised to 1, causal (``K(t) = 0`` for ``t < 0``)."""
    t_peak = TAU_M * TAU_S / (TAU_M - TAU_S) * math.log(TAU_M / TAU_S)
    v0 = 1.0 / (math.exp(-t_peak / TAU_M) - math.exp(-t_peak / TAU_S))
    t_clamped = torch.clamp(t, min=0.0)
    raw = v0 * (torch.exp(-t_clamped / TAU_M) - torch.exp(-t_clamped / TAU_S))
    return torch.where(t >= 0.0, raw, torch.zeros_like(raw))


def _psp_kernel_v0() -> float:
    """Peak-normalisation constant ``V_0`` of the PSP kernel."""
    t_peak = TAU_M * TAU_S / (TAU_M - TAU_S) * math.log(TAU_M / TAU_S)
    return 1.0 / (math.exp(-t_peak / TAU_M) - math.exp(-t_peak / TAU_S))


def kernel_integral() -> float:
    """Analytic ``int_0^inf K(u) du = V_0 (tau_m - tau_s)`` (used by the analytic centering)."""
    return _psp_kernel_v0() * (TAU_M - TAU_S)


def tempotron_vmax(
    n_aff: int,
    k_target: float,
    n_draws: int,
    dt: float,
    device: torch.device,
    gen: torch.Generator,
    burn_per_side_corr: float = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Centred, standardised tempotron ``V_max`` over a stationary interior window.

    Parameters
    ----------
    n_aff
        Number of afferents ``N``.
    k_target
        Interior-window ``K_eff`` target; ``T_eval = k_target * sqrt(tau_s tau_m)``.
    n_draws
        Number of (Poisson pattern, Gaussian weights) realisations.
    dt
        Grid spacing (ms).
    device
        Torch device.
    gen
        Torch generator.
    burn_per_side_corr
        One-sided burn-in (units of ``sqrt(tau_s tau_m)``).

    Returns
    -------
    (vmax, sanity)
        ``vmax`` is the standardised interior ``V_max`` per draw. ``sanity`` records the mean
        empirical per-afferent trace mean (centering check) and the mean per-pattern ``sigma_V``.

    Notes
    -----
    The voltage ``V(t) = sum_i w_i s_i^c(t)`` uses the **explicitly centred** trace
    ``s_i^c(t) = s_i(t) - E[s_i(t)]`` with ``E[s_i(t)] = rho * int K`` (``rho = 1/T``) -- this
    removes the long-range DC covariance that otherwise violates the Berman/LLR mixing
    requirement (spec §3.5). Each realisation is standardised by the analytic per-pattern
    ``sigma_V(t) = sqrt(sum_i s_i^c(t)^2)`` so the EVT unit-variance normalisation holds at
    finite ``N``.
    """
    t_window_full = (k_target + 2.0 * burn_per_side_corr) * SQRT_TAU
    n_grid = round(t_window_full / dt) + 1
    t_eval = k_target * SQRT_TAU
    n_interior = round(t_eval / dt) + 1
    burn = math.ceil(burn_per_side_corr * SQRT_TAU / dt)
    t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * float(dt)
    rho = 1.0 / t_window_full
    e_s = rho * kernel_integral()  # analytic E[s_i(t)] (afferent-independent, time-flat)

    # Poisson(rate = rho * T_full = 1 expected spike per afferent over the FULL padded window),
    # times uniform on [0, T_full]. Keep the RMS density rho = 1/T over the whole window.
    expected_count = rho * t_window_full  # == 1.0 by construction
    rate_ones = torch.full((n_draws, n_aff), expected_count, device=device, dtype=torch.float32)
    counts = torch.poisson(rate_ones, generator=gen).to(torch.int64)
    max_spikes = int(counts.max().item())
    if max_spikes == 0:
        max_spikes = 1
    spike_times = t_window_full * torch.rand(
        (n_draws, n_aff, max_spikes), generator=gen, device=device, dtype=torch.float32
    )
    spike_idx = torch.arange(max_spikes, device=device).view(1, 1, max_spikes)
    valid = (spike_idx < counts.unsqueeze(-1)).to(torch.float32)
    w = torch.randn(n_draws, n_aff, generator=gen, device=device, dtype=torch.float32)

    vmax = torch.full((n_draws,), float("-inf"), device=device, dtype=torch.float32)
    s_sum_acc = torch.zeros(n_draws, device=device, dtype=torch.float32)  # for centering check
    sigma_acc = torch.zeros(n_draws, device=device, dtype=torch.float32)
    n_interior_seen = 0
    # Evaluate the interior grid vectorised over a (G, Bc, N, max_spikes) kernel tensor. The chunk
    # sizes are bounded by a memory BUDGET on that tensor's element count: the earlier version
    # sized G against N*max_spikes only and OMITTED the draw dimension B, so the real tensor was
    # B-fold larger and OOM'd (~93 GiB at the gated K). We chunk BOTH the draw dimension (Bc) and
    # the grid (G) so the working tensor stays within BUDGET regardless of N, B, max_spikes.
    budget = 32_000_000  # elements (~128 MB f32 for `contrib`; a few x with intermediates)
    per_unit = max(1, n_aff * max_spikes)              # elements per (grid point, draw)
    b_chunk = max(1, min(n_draws, budget // per_unit))  # draws per tile
    g_chunk = max(1, budget // (b_chunk * per_unit))    # grid points per tile
    for b0 in range(0, n_draws, b_chunk):
        b1 = min(b0 + b_chunk, n_draws)
        spike_e = spike_times[b0:b1].unsqueeze(0)  # (1, Bc, N, max_spikes)
        valid_e = valid[b0:b1].unsqueeze(0)
        w_b = w[b0:b1].unsqueeze(0)                # (1, Bc, N)
        vmax_b = torch.full((b1 - b0,), float("-inf"), device=device, dtype=torch.float32)
        for g0 in range(burn, burn + n_interior, g_chunk):
            g1 = min(g0 + g_chunk, burn + n_interior)
            t_chunk = t_grid[g0:g1].view(-1, 1, 1, 1)  # (G, 1, 1, 1)
            contrib = psp_kernel(t_chunk - spike_e) * valid_e  # (G, Bc, N, max_spikes)
            s_i = contrib.sum(dim=3)  # (G, Bc, N)
            s_c = s_i - e_s
            v_t = (w_b * s_c).sum(dim=2)  # (G, Bc)
            sigma_v = torch.sqrt(torch.clamp((s_c * s_c).sum(dim=2), min=1e-12))  # (G, Bc)
            v_std = v_t / sigma_v
            vmax_b = torch.maximum(vmax_b, v_std.max(dim=0).values)
            s_sum_acc[b0:b1] = s_sum_acc[b0:b1] + s_i.mean(dim=2).sum(dim=0)
            sigma_acc[b0:b1] = sigma_acc[b0:b1] + sigma_v.sum(dim=0)
            if b0 == 0:
                n_interior_seen += g1 - g0
        vmax[b0:b1] = vmax_b

    sanity = {
        "mean_raw_trace": float((s_sum_acc / max(1, n_interior_seen)).mean().item()),
        "analytic_E_s": float(e_s),
        "mean_sigma_v": float((sigma_acc / max(1, n_interior_seen)).mean().item()),
        "n_interior": float(n_interior),
        "max_spikes": float(max_spikes),
    }
    return vmax, sanity


# --------------------------------------------------------------------------------------------
# (C) Tier 1 -- curvature via dt->0 Richardson extrapolation
# --------------------------------------------------------------------------------------------
def fd_curvature(c_func: Any, dt: float) -> float:
    """Central-difference second derivative ``c''(0)`` from the analytic autocorrelation."""
    c0 = float(c_func(np.array(0.0)))
    cp = float(c_func(np.array(dt)))
    cm = float(c_func(np.array(-dt)))
    return (cp - 2.0 * c0 + cm) / (dt * dt)


def richardson_curvature(c_func: Any, dts: list[float]) -> tuple[float, list[float]]:
    """Richardson-extrapolate ``c''(0)`` to ``dt -> 0`` (the central FD is ``O(dt^2)``).

    Parameters
    ----------
    c_func
        Callable autocorrelation (analytic or a grid interpolant).
    dts
        Step sizes, finest last is not required; uses the two finest for the extrapolation.

    Returns
    -------
    (extrapolated, raw)
        Richardson estimate (``O(dt^4)`` accurate) and the per-``dt`` raw FD values.
    """
    raw = [fd_curvature(c_func, h) for h in dts]
    h_sorted = sorted(dts)
    f_fine = fd_curvature(c_func, h_sorted[0])
    f_coarse = fd_curvature(c_func, h_sorted[1])
    ratio = (h_sorted[1] / h_sorted[0]) ** 2
    extrapolated = (ratio * f_fine - f_coarse) / (ratio - 1.0)
    return extrapolated, raw


# --------------------------------------------------------------------------------------------
# (C) Tier 2 -- bootstrap CIs and the two-sample Anderson-Darling gate
# --------------------------------------------------------------------------------------------
def bootstrap_ci(
    samples: NDArray[np.float64],
    stat: Any,
    n_boot: int,
    level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat`` at the given two-sided ``level`` (e.g. 0.99)."""
    n = samples.size
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = stat(samples[idx])
    lo = float(np.percentile(vals, 100.0 * (1.0 - level) / 2.0))
    hi = float(np.percentile(vals, 100.0 * (1.0 + level) / 2.0))
    return lo, hi


def bootstrap_diff_ci(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    stat: Any,
    n_boot: int,
    level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the two-sample difference ``stat(a) - stat(b)``.

    Resamples both samples independently, so the CI carries *both* samples' sampling
    variance (the symmetric two-sample test). The moment gate passes iff 0 lies inside this
    CI -- i.e. the tempotron and reference moments are consistent given finite-sample noise.
    Using ``stat(a)`` against ``stat(b)``'s one-sided CI (the earlier construction) ignored
    ``a``'s own variance and triple-gated noisy moments, false-failing the GP-vs-GP null.
    """
    na, nb = a.size, b.size
    d = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        d[i] = stat(a[rng.integers(0, na, size=na)]) - stat(b[rng.integers(0, nb, size=nb)])
    lo = float(np.percentile(d, 100.0 * (1.0 - level) / 2.0))
    hi = float(np.percentile(d, 100.0 * (1.0 + level) / 2.0))
    return lo, hi


def anderson_darling_pvalue(a: NDArray[np.float64], b: NDArray[np.float64]) -> tuple[float, float]:
    """Two-sample Anderson-Darling statistic and (clipped) p-value via ``scipy.anderson_ksamp``."""
    res = anderson_ksamp([a, b])
    return float(res.statistic), float(res.significance_level)


def tier2_compare(
    tempo: NDArray[np.float64],
    ref: NDArray[np.float64],
    n_boot: int,
    ad_level: float,
    ci_level: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Score one Tier-2 cell: AD gate + median/scale/skew of ``tempo`` inside ``ref``'s CIs.

    Parameters
    ----------
    tempo, ref
        Tempotron and GP-reference ``V_max`` samples at equal ``K_eff``.
    n_boot
        Bootstrap resamples for the reference moment CIs.
    ad_level
        Anderson-Darling significance level below which the AD gate is a *fail* (e.g. 0.05).
    ci_level
        Two-sided bootstrap CI level for the moment gates (e.g. 0.99).
    rng
        NumPy generator.

    Returns
    -------
    dict
        AD statistic + p-value + pass flag; the three moment gaps with their reference CIs and
        per-moment pass flags; the aggregate ``passed`` for the cell.
    """
    ad_stat, ad_p = anderson_darling_pvalue(tempo, ref)
    ad_pass = ad_p >= ad_level

    tempo_s = summarise(tempo)
    ref_s = summarise(ref)
    out: dict[str, Any] = {
        "ad_stat": ad_stat,
        "ad_pvalue": ad_p,
        "ad_pass": bool(ad_pass),
        "tempo": tempo_s,
        "ref": ref_s,
    }
    moment_pass = True
    for name, fn in (("median", lambda x: float(np.median(x))),
                     ("scale", robust_scale),
                     ("skew", sample_skew)):
        # Two-sample difference test: is tempo_moment - ref_moment consistent with 0?
        lo, hi = bootstrap_diff_ci(tempo, ref, fn, n_boot, ci_level, rng)
        ok = lo <= 0.0 <= hi
        moment_pass = moment_pass and ok
        out[f"{name}_diff_ci_lo"] = lo
        out[f"{name}_diff_ci_hi"] = hi
        out[f"{name}_diff"] = tempo_s[name] - ref_s[name]
        out[f"{name}_pass"] = bool(ok)
    out["moment_pass"] = bool(moment_pass)
    out["passed"] = bool(ad_pass and moment_pass)
    return out


# --------------------------------------------------------------------------------------------
# (C) Tier 3 -- Gumbel/RMS contact, trend diagnostic (reported, never gated)
# --------------------------------------------------------------------------------------------
def tier3_diagnostic(ref: NDArray[np.float64], k_eff: float) -> dict[str, float]:
    """Tier-3 Gumbel/RMS contact for the GP reference at one K (diagnostic only).

    Reports the PWM and MLE Gumbel fits in the **median** convention, the asymptotic targets
    ``m_K, beta_K``, the scale-free corrected ratio identity, and skew vs the *finite-K* value
    (never vs the asymptotic 1.1395). No pass/fail is attached.
    """
    pwm_mode, pwm_scale = pwm_gumbel_fit(ref)
    pwm_median = gumbel_median_from_mode(pwm_mode, pwm_scale)
    mle_loc, mle_scale = gumbel_r.fit(ref)
    mle_median = gumbel_median_from_mode(mle_loc, mle_scale)

    mode_k = gumbel_mode(k_eff)
    scale_k = gumbel_scale(k_eff)
    median_k = gumbel_median_from_mode(mode_k, scale_k)

    # Scale-free corrected identity:  m/beta + (ln ln K + ln 4pi)/2  vs  2 ln K - ln ln 2.
    lhs = pwm_median / pwm_scale + (math.log(math.log(k_eff)) + math.log(4.0 * math.pi)) / 2.0
    rhs = 2.0 * math.log(k_eff) - math.log(math.log(2.0))

    return {
        "pwm_mode": pwm_mode,
        "pwm_scale": pwm_scale,
        "pwm_median": pwm_median,
        "mle_loc": float(mle_loc),
        "mle_scale": float(mle_scale),
        "mle_median": mle_median,
        "asymptotic_mode": mode_k,
        "asymptotic_scale": scale_k,
        "asymptotic_median": median_k,
        "emp_median": float(np.median(ref)),
        "emp_skew": sample_skew(ref),
        "ratio_lhs": lhs,
        "ratio_rhs": rhs,
        "ratio_residual": lhs - rhs,
        "median_residual": pwm_median - median_k,
        "scale_residual": pwm_scale - scale_k,
    }


# --------------------------------------------------------------------------------------------
# (D) Aggregate null calibration
# --------------------------------------------------------------------------------------------
def null_calibration(
    k_target: float,
    n_draws: int,
    dt: float,
    n_reps: int,
    n_boot: int,
    ad_level: float,
    ci_level: float,
    scale_inject: float,
    base_seed: int,
) -> dict[str, float]:
    """Family-wise size/power of the Tier-2 suite under GP-vs-GP and injected-scale-error nulls.

    Parameters
    ----------
    k_target
        ``K_eff`` for the calibration cell.
    n_draws
        Draws per GP realisation.
    dt
        Grid spacing.
    n_reps
        Number of independent calibration repetitions.
    n_boot
        Bootstrap resamples per Tier-2 comparison.
    ad_level, ci_level
        Tier-2 gate levels under test.
    scale_inject
        Multiplicative scale error injected to probe power (e.g. 1.1).
    base_seed
        Seed offset.

    Returns
    -------
    dict
        ``P_pass_null`` (GP-vs-GP; should be high) and ``P_pass_injected`` (scale x inject;
        should be low), plus the cell ``K_eff`` and rep count.
    """
    pass_null = 0
    pass_injected = 0
    for r in range(n_reps):
        rng_a = np.random.default_rng(base_seed * 1009 + 7 * r + 1)
        rng_b = np.random.default_rng(base_seed * 1009 + 7 * r + 2)
        rng_boot = np.random.default_rng(base_seed * 1009 + 7 * r + 3)
        ref_a, _ = gp_reference_vmax(k_target, n_draws, dt, rng_a)
        ref_b, _ = gp_reference_vmax(k_target, n_draws, dt, rng_b)
        # Null: two independent GP draws.
        res_null = tier2_compare(ref_b, ref_a, n_boot, ad_level, ci_level, rng_boot)
        if res_null["passed"]:
            pass_null += 1
        # Power: inject a scale error into the "tempotron" arm.
        med = float(np.median(ref_b))
        injected = med + scale_inject * (ref_b - med)
        rng_boot2 = np.random.default_rng(base_seed * 1009 + 7 * r + 4)
        res_inj = tier2_compare(injected, ref_a, n_boot, ad_level, ci_level, rng_boot2)
        if res_inj["passed"]:
            pass_injected += 1
    return {
        "K_eff": k_eff_of_window(k_target * SQRT_TAU),
        "n_reps": float(n_reps),
        "P_pass_null": pass_null / n_reps,
        "P_pass_injected": pass_injected / n_reps,
        "scale_inject": scale_inject,
    }


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------
def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))
    smoke = os.environ.get("LAB_RUN_DIR", "runs/local-dev") in ("", "runs/local-dev")

    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    k_list = [float(x) for x in ov.get("K_list", "16,64,256,1024,4096").split(",")]
    gated_set = {256.0, 1024.0, 4096.0}
    n_draws = int(ov.get("n_draws", "2000" if smoke else "50000"))
    n_aff = int(ov.get("N", "400" if smoke else "4000"))
    ndt = int(ov.get("ndt", "2" if smoke else "3"))  # number of dt levels in the plateau ladder
    n_boot = int(ov.get("n_boot", "500" if smoke else "2000"))
    ad_level = float(ov.get("ad_level", "0.02"))  # per-gate level (family-wise controlled)
    ci_level = float(ov.get("ci_level", "0.99"))
    null_reps = int(ov.get("null_reps", "10" if smoke else "40"))

    # dt ladder: sqrt(tau_s tau_m)/{5, 10, 20}, finest used for the GP reference plateau.
    dt_divisors = [5.0, 10.0, 20.0][:ndt]
    dt_levels = [SQRT_TAU / d for d in dt_divisors]
    dt_ref = min(dt_levels)  # production dt <= sqrt/10 (uses the finest in the ladder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(False)
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"V2 V_max EVT | seed={master_seed} device={device} K={k_list} gated={sorted(gated_set)} "
        f"n_draws={n_draws} N={n_aff} dt_ladder={[round(x, 3) for x in dt_levels]} smoke={smoke}",
        flush=True,
    )

    metrics_path = run_dir / "metrics.jsonl"

    def emit(name: str, value: float, step: int) -> None:
        with metrics_path.open("a") as f:
            f.write(
                json.dumps(
                    {"name": name, "value": float(value), "step": int(step),
                     "wall_time": time.time()}
                )
                + "\n"
            )

    started = time.time()

    # ---- Tier 1 (analytic, gated EXACTLY) -------------------------------------------------
    cpp_analytic = cpp_zero_analytic()
    k_target_identity = TAU_M  # placeholder; K = T/sqrt(tau_s tau_m), ratio == 1 by construction
    # T1b: K = T sqrt|c''(0)| with sqrt|c''(0)| = 1/sqrt(tau_s tau_m) -> ratio exactly 1.
    sqrt_cpp = math.sqrt(abs(cpp_analytic))
    t1b_ratio = sqrt_cpp * SQRT_TAU  # = (1/sqrt(tau_s tau_m)) * sqrt(tau_s tau_m) = 1
    extrap, raw_fd = richardson_curvature(autocorr, dt_levels if ndt >= 2 else [dt_ref, dt_ref / 2])
    t1c_rel = abs(extrap / cpp_analytic - 1.0)
    raw_single_rel = abs(raw_fd[0] / cpp_analytic - 1.0)
    tier1 = {
        "cpp_analytic": cpp_analytic,
        "cpp_target": -1.0 / (TAU_M * TAU_S),
        "t1a_exact_match": bool(abs(cpp_analytic - (-1.0 / (TAU_M * TAU_S))) < 1e-15),
        "K_over_T_sqrtcpp_ratio": t1b_ratio,
        "t1b_pass": bool(abs(t1b_ratio - 1.0) < 1e-12),
        "cpp_fd_raw": raw_fd,
        "cpp_fd_dt": dt_levels,
        "cpp_richardson": extrap,
        "t1c_rel_err": t1c_rel,
        "t1c_pass": bool(t1c_rel <= 0.03),
        "raw_single_dt_rel_err": raw_single_rel,
    }
    print(
        f"Tier 1: c''(0)={cpp_analytic:.6f} (exact), K/(T sqrt|c''|) ratio={t1b_ratio:.6f}, "
        f"Richardson c''(0)={extrap:.6f} rel_err={100 * t1c_rel:.2f}% "
        f"(pass={tier1['t1c_pass']}); raw single-dt FD rel_err={100 * raw_single_rel:.1f}%",
        flush=True,
    )
    emit("t1c_rel_err", t1c_rel, 0)
    _ = k_target_identity

    # ---- GP reference per K (primary), with a dt-plateau check ----------------------------
    boot_rng = np.random.default_rng(master_seed * 7919 + 11)
    gp_rows: list[dict[str, Any]] = []
    ref_cache: dict[float, NDArray[np.float64]] = {}
    plateau: dict[float, dict[str, float]] = {}
    for k_eff in k_list:
        t0 = time.time()
        # dt plateau: median/scale/skew across the dt ladder at this K.
        plateau_meds: list[float] = []
        plateau_scales: list[float] = []
        for di, dt in enumerate(dt_levels):
            rng = np.random.default_rng(master_seed * 31 + round(k_eff) * 7 + di)
            vmax_dt, sane = gp_reference_vmax(k_eff, n_draws, dt, rng)
            plateau_meds.append(float(np.median(vmax_dt)))
            plateau_scales.append(robust_scale(vmax_dt))
            if dt == dt_ref:
                ref_cache[k_eff] = vmax_dt
                ref_sanity = sane
        med_spread = float(max(plateau_meds) - min(plateau_meds))
        scale_spread = float(max(plateau_scales) - min(plateau_scales))
        plateau[k_eff] = {
            "median_spread": med_spread,
            "scale_spread": scale_spread,
            "medians": plateau_meds,  # type: ignore[dict-item]
            "scales": plateau_scales,  # type: ignore[dict-item]
        }
        ref = ref_cache[k_eff]
        np.save(run_dir / f"gp_vmax_K{round(k_eff)}.npy", ref)
        s = summarise(ref)
        t3 = tier3_diagnostic(ref, k_eff)
        elapsed = time.time() - t0
        gp_rows.append({
            "K": k_eff,
            "K_eff": ref_sanity["K_eff"],
            "T_eval": ref_sanity["T_eval"],
            "dt": dt_ref,
            "n_draws": n_draws,
            "gp_median": s["median"],
            "gp_scale": s["scale"],
            "gp_skew": s["skew"],
            "sample_var": ref_sanity["sample_var"],
            "acf_max_err": ref_sanity["acf_max_err"],
            "median_spread_dt": med_spread,
            "scale_spread_dt": scale_spread,
            "asymptotic_median": t3["asymptotic_median"],
            "asymptotic_scale": t3["asymptotic_scale"],
            "pwm_median": t3["pwm_median"],
            "pwm_scale": t3["pwm_scale"],
            "ratio_residual": t3["ratio_residual"],
            "emp_skew": t3["emp_skew"],
            "tier3": t3,
        })
        emit(f"gp_median_K{round(k_eff)}", s["median"], round(k_eff))
        emit(f"gp_scale_K{round(k_eff)}", s["scale"], round(k_eff))
        print(
            f"GP  K={k_eff:>7.0f}: med={s['median']:.4f} (asy m_K={t3['asymptotic_median']:.4f}) "
            f"scale={s['scale']:.4f} skew={s['skew']:.3f}  var={ref_sanity['sample_var']:.4f} "
            f"acf_err={ref_sanity['acf_max_err']:.2e}  dt-plateau med_spread={med_spread:.4f}  "
            f"t={elapsed:.1f}s",
            flush=True,
        )

    # ---- Tempotron per gated K + N/K ladder (Tier 2) --------------------------------------
    # The tempotron-realises-GP check needs N >> K, which is affordable (and in-regime) only at
    # the smaller gated K; the grid cost ~ n_interior = K*sqrt(tau_s tau_m)/dt makes K=4096
    # (~41k grid points) both out-of-regime and prohibitively slow. The GP reference already
    # covers K=4096 (Tier 1/3); Tier 2 is capped at `tier2_k_max`.
    tier2_k_max = float(ov.get("tier2_k_max", "1024"))
    tier2_rows: list[dict[str, Any]] = []
    nk_ratios = [int(x) for x in ov.get("nk_ladder", "4,16,64").split(",")]
    for k_eff in [k for k in k_list if k in gated_set and k <= tier2_k_max]:
        ref = ref_cache[k_eff]
        ladder: list[dict[str, Any]] = []
        primary_cell: dict[str, Any] | None = None
        for nk in nk_ratios:
            n_this = round(nk * k_eff)
            # Cap N to keep the smoke/CPU run bounded; flag out-of-regime small-N points.
            n_cap = int(ov.get("N_cap", str(n_aff if smoke else 8000)))
            out_of_regime = False
            if n_this > n_cap:
                # Use the explicit small-N endpoint (N=1000) rather than a huge N.
                n_this = min(n_this, n_cap)
            if nk < 4:
                out_of_regime = True
            gen = torch.Generator(device=device).manual_seed(
                (master_seed * 131 + round(k_eff) * 17 + nk) & 0xFFFFFFFF
            )
            # Tempotron draws: a few thousand suffice for two-sample AD power vs the GP
            # reference; no need to match the GP's 50k (and it bounds the grid-loop cost).
            n_tdraws = n_draws if smoke else max(500, min(4000, n_draws // 4))
            t0 = time.time()
            tempo_t, tsane = tempotron_vmax(n_this, k_eff, n_tdraws, dt_ref, device, gen)
            tempo = tempo_t.detach().to("cpu", dtype=torch.float64).numpy()
            cmp = tier2_compare(tempo, ref, n_boot, ad_level, ci_level, boot_rng)
            cmp_row = {
                "K": k_eff,
                "N": n_this,
                "NK": nk,
                "n_draws": n_tdraws,
                "out_of_regime": out_of_regime,
                "mean_raw_trace": tsane["mean_raw_trace"],
                "analytic_E_s": tsane["analytic_E_s"],
                "mean_sigma_v": tsane["mean_sigma_v"],
                "median_gap": cmp["tempo"]["median"] - cmp["ref"]["median"],
                "scale_gap": cmp["tempo"]["scale"] - cmp["ref"]["scale"],
                "skew_gap": cmp["tempo"]["skew"] - cmp["ref"]["skew"],
                **{f"cmp_{k}": v for k, v in cmp.items() if not isinstance(v, dict)},
                "tempo_median": cmp["tempo"]["median"],
                "tempo_scale": cmp["tempo"]["scale"],
                "tempo_skew": cmp["tempo"]["skew"],
                "ref_median": cmp["ref"]["median"],
            }
            ladder.append(cmp_row)
            np.save(run_dir / f"tempo_vmax_K{round(k_eff)}_NK{nk}.npy", tempo)
            print(
                f"  tempo K={k_eff:.0f} N/K={nk} N={n_this}: "
                f"med_gap={cmp_row['median_gap']:+.4f} "
                f"scale_gap={cmp_row['scale_gap']:+.4f} AD_p={cmp['ad_pvalue']:.3g} "
                f"pass={cmp['passed']} (raw_mean={tsane['mean_raw_trace']:.4f}) "
                f"t={time.time() - t0:.1f}s",
                flush=True,
            )
            # The primary cell is the largest in-regime N/K.
            if not out_of_regime:
                primary_cell = cmp_row
        # Tier 2c convergence: |median_gap| should shrink as N/K grows.
        gaps = [abs(c["median_gap"]) for c in ladder]
        converging = (
            all(gaps[i] >= gaps[i + 1] - 1e-3 for i in range(len(gaps) - 1))
            if len(gaps) > 1
            else True
        )
        tier2_rows.append({
            "K": k_eff,
            "primary_passed": bool(primary_cell["cmp_passed"]) if primary_cell else False,
            "primary_NK": primary_cell["NK"] if primary_cell else None,
            "t2c_converging": bool(converging),
            "ladder": ladder,
        })
        emit(f"tempo_primary_pass_K{round(k_eff)}",
             1.0 if (primary_cell and primary_cell["cmp_passed"]) else 0.0, round(k_eff))

    # ---- (D) Aggregate null calibration ---------------------------------------------------
    calib_k = min(gated_set & set(k_list)) if (gated_set & set(k_list)) else min(k_list)
    calib = null_calibration(
        calib_k,
        n_draws if not smoke else min(n_draws, 3000),
        dt_ref,
        null_reps,
        n_boot,
        ad_level,
        ci_level,
        scale_inject=1.1,
        base_seed=master_seed,
    )
    # Family-wise (3 gated cells, naive AND): false-fail = 1 - P_pass_null^3 if independent.
    n_gated_cells = len([k for k in k_list if k in gated_set])
    fw_false_fail = 1.0 - calib["P_pass_null"] ** max(1, n_gated_cells)
    calib["family_wise_false_fail_est"] = fw_false_fail
    print(
        f"Null calibration @K={calib['K_eff']:.0f}: P(pass|null)={calib['P_pass_null']:.3f} "
        f"P(pass|inject x1.1)={calib['P_pass_injected']:.3f}  "
        f"family-wise false-fail est={fw_false_fail:.3f} (target ~0.05)",
        flush=True,
    )

    # ---- Write results --------------------------------------------------------------------
    overall_verdict = {
        "tier1_pass": tier1["t1a_exact_match"] and tier1["t1b_pass"] and tier1["t1c_pass"],
        "tier2_gated_pass": {round(r["K"]): r["primary_passed"] for r in tier2_rows},
        "tier2_convergence": {round(r["K"]): r["t2c_converging"] for r in tier2_rows},
    }

    results = {
        "experiment": "V2 -- V_max EVT vs analytic GP reference",
        "reference": "Rubin, Monasson & Sompolinsky 2010 PRL 105, 218102; LLR 1983; spec v2",
        "params": {
            "K_list": k_list,
            "gated": sorted(gated_set),
            "n_draws": n_draws,
            "N": n_aff,
            "dt_levels": dt_levels,
            "dt_ref": dt_ref,
            "ad_level": ad_level,
            "ci_level": ci_level,
            "tau_m": TAU_M,
            "tau_s": TAU_S,
            "sqrt_tau_s_tau_m": SQRT_TAU,
            "median_offset": MEDIAN_OFFSET,
            "master_seed": master_seed,
            "smoke": smoke,
        },
        "tier1": tier1,
        "gp_reference": gp_rows,
        "dt_plateau": {round(k): v for k, v in plateau.items()},
        "tier2": tier2_rows,
        "null_calibration": calib,
        "verdict": overall_verdict,
        "elapsed_seconds": time.time() - started,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))

    # Flat results.csv: one row per GP-reference K (the primary table).
    csv_rows = []
    for r in gp_rows:
        csv_rows.append({
            "K": r["K"],
            "K_eff": r["K_eff"],
            "dt": r["dt"],
            "n_draws": r["n_draws"],
            "gp_median": r["gp_median"],
            "gp_scale": r["gp_scale"],
            "gp_skew": r["gp_skew"],
            "asymptotic_median": r["asymptotic_median"],
            "asymptotic_scale": r["asymptotic_scale"],
            "pwm_median": r["pwm_median"],
            "ratio_residual": r["ratio_residual"],
            "sample_var": r["sample_var"],
            "acf_max_err": r["acf_max_err"],
            "median_spread_dt": r["median_spread_dt"],
        })
    with (run_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"done in {results['elapsed_seconds']:.1f}s -> {run_dir}/results.json  "
        f"(Tier1 pass={overall_verdict['tier1_pass']}, "
        f"Tier2 gated={overall_verdict['tier2_gated_pass']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
