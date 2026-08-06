"""V13 -- KERNEL-PARAMETERIZED tempotron capacity alpha_c(K_eff) across PSP kernel families.

Self-contained Experiment-Contract entrypoint (numpy + torch only; the lab ships this file
unchanged, so there are NO ``tempotron`` / ``lab`` imports -- everything is inlined here). This is
a direct fork of ``v6_bandlimited_capacity.py`` (reduction #6, windowed-sinc / flat band). The ONLY
science change is that the band-limited (sinc) kernel is generalised to a ``kernel=`` knob dispatching
a single :func:`eval_kernel` over a family of temporal PSP kernels, and the kernel -> K_eff map is now
a NUMERIC certified route (autocorrelation / FFT spectral moments) rather than the flat-band-specific
closed form ``Omega_b = sqrt3 K_eff / T``. Everything else -- patterns, density control, the faithful
Adam-cosine + restarts learner, threshold gauge, OOM budgets, incremental CSV, the argv/env contract --
is reused byte-faithfully from v6.

Cross-kernel "capacity universality"
------------------------------------
For a stationary zero-mean Gaussian readout V(t) the capacity-driving effective dimension is a spectral
functional of the kernel's power spectrum S(omega) = |K_hat(omega)|^2. Two such functionals are logged
per cell:

- the **Rice crossing axis** ``K_eff_rice = T sqrt(lambda_2/lambda_0)`` (the v6 axis; for a flat band
  this is exactly ``Omega_b T/sqrt3``), and
- the **participation-ratio / power-weighted effective dimension** ``D_PR`` (the v8 ``k_pr`` construction
  lifted to a generic spectrum; finite even when K_eff_rice diverges -- the relevant axis for the rect /
  discontinuous boundary kernel where lambda_2 is grid-divergent).

The capacity hypothesis under test is UNIVERSALITY: along a *certified* expressiveness axis the findable
``alpha_hat_c`` should depend on the kernel only through its effective dimension (K_eff_rice or D_PR), not
through the kernel family -- so the per-kernel ladders should COLLAPSE onto one curve. The degenerate
anchors fix the endpoints: ``sinusoid`` (a single cosine, m=1) must give ``alpha_c -> 2`` (Cover/Gardner),
and ``sinc`` must reproduce the v6 K_eff(Omega_b) relation (the #6 ANCHOR).

The kernels (s = time lag in ms; peak-normalised to 1 at s=0 unless noted)
-------------------------------------------------------------------------
- ``sinc``     : windowed sinc (band-limited), param ``omega_b`` [rad/ms]   -- the #6 anchor.
- ``gauss``    : e^{-s^2/(2 sigma^2)},                 param ``sigma`` [ms].
- ``gabor``    : e^{-s^2/(2 sigma^2)} cos(omega0 s),   params ``sigma``, ``omega0`` (Q = omega0*sigma).
- ``alpha``    : causal alpha (t/tau) e^{1 - t/tau} (peak 1 at t=tau), param ``tau`` [ms] (v8 form).
- ``rect``     : boxcar 1_{[-tau/2, tau/2]},           param ``tau`` [ms] (the discontinuous kernel).
- ``sinusoid`` : cos(omega0 s) over the window,        param ``omega0`` (the m=1 degenerate anchor).

The kernel knob is inverted by bisection so the *measured* K_eff hits the target sweep value (for
``rect``, whose K_eff_rice diverges with dt, the bisection targets ``D_PR`` instead and ALSO logs
K_eff_rice). The Nyquist grid step is keyed off the kernel's effective bandwidth sqrt(lambda_2/lambda_0)
rather than ``omega_b`` so every kernel gets a comparable resolution.

Run standalone (tiny smoke; does NOT need a GPU):
    LAB_RUN_DIR=/tmp/v13_smoke uv run --with torch python studies/v13_kernel_capacity.py \\
        kernel=gauss Keff_list=8 N_list=40 alphas=1.0,2.0,3.0 seeds=0,1 epochs=200 oversample=4
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
import torch

SQRT3: float = math.sqrt(3.0)


# ============================================================================================
# KERNEL FAMILY -- a single eval_kernel(s, kernel, params) dispatching the temporal PSP kernels.
# All kernels are peak-normalised to 1 at s=0 (rect: 1 inside the box) and even/non-causal where
# natural; alpha is causal. ``params`` carries the width knob (sigma/tau/omega_b) plus omega0/n_lobes.
# ============================================================================================
def _band_kernel(s: torch.Tensor, omega_b: float, n_lobes: int, window: str) -> torch.Tensor:
    """Windowed sinc (flat low-pass), byte-faithful port of v6.band_kernel. EVEN / non-causal."""
    half_support = n_lobes * math.pi / omega_b
    x = omega_b * s
    sinc = torch.where(x.abs() < 1e-12, torch.ones_like(x), torch.sin(x) / x)
    if window == "hann":
        win = 0.5 * (1.0 + torch.cos(math.pi * s / half_support))
    else:  # "lanczos" -- a sinc window of the same half-support
        xw = math.pi * s / half_support
        win = torch.where(xw.abs() < 1e-12, torch.ones_like(xw), torch.sin(xw) / xw)
    k = sinc * win
    return torch.where(s.abs() <= half_support, k, torch.zeros_like(k))


def eval_kernel(s: torch.Tensor, kernel: str, params: dict) -> torch.Tensor:
    """Dispatch the temporal PSP kernel ``K(s)`` (s = time lag [ms]); peak 1 at s=0.

    The single physics call site (replaces v6's ``band_kernel(...)`` inside precompute_traces).

    Parameters
    ----------
    s : torch.Tensor
        Time lags [ms] (any shape). Even kernels obey ``K(-s)==K(s)``; ``alpha`` is causal (0 for s<0).
    kernel : str
        One of ``sinc``, ``gauss``, ``gabor``, ``alpha``, ``rect``, ``sinusoid``.
    params : dict
        Kernel knobs: ``omega_b`` (sinc), ``sigma`` (gauss/gabor), ``omega0`` (gabor/sinusoid),
        ``tau`` (alpha/rect), plus ``n_lobes``/``window`` (sinc) and ``n_lobes_win`` for the
        windowed gabor/sinusoid support.
    """
    if kernel == "sinc":
        return _band_kernel(s, params["omega_b"], params.get("n_lobes", 6), params.get("window", "lanczos"))
    if kernel == "gauss":
        sigma = params["sigma"]
        return torch.exp(-(s * s) / (2.0 * sigma * sigma))
    if kernel == "gabor":
        # Band-pass: a Gaussian-windowed cosine. Two regimes via Q = omega0*sigma:
        # Q << 1 -> low-pass-like (Gaussian dominates); Q >> 1 -> narrow band-pass (many ripples).
        sigma = params["sigma"]
        omega0 = params["omega0"]
        return torch.exp(-(s * s) / (2.0 * sigma * sigma)) * torch.cos(omega0 * s)
    if kernel == "alpha":
        # Causal alpha function (peak 1 at t=tau), the v8_gp_capacity form (alpha-function endpoint).
        tau = params["tau"]
        sc = torch.clamp(s, min=0.0)
        raw = (sc / tau) * torch.exp(1.0 - sc / tau)
        return torch.where(s >= 0.0, raw, torch.zeros_like(raw))
    if kernel == "rect":
        # Boxcar 1_{[-tau/2, tau/2]} -- the discontinuous boundary kernel (lambda_2 grid-divergent).
        tau = params["tau"]
        return torch.where(s.abs() <= 0.5 * tau, torch.ones_like(s), torch.zeros_like(s))
    if kernel == "sinusoid":
        # Pure cosine over a Lanczos-windowed support (the m=1 degenerate anchor; alpha_c -> 2).
        omega0 = params["omega0"]
        n_lobes_win = params.get("n_lobes_win", 6)
        half_support = n_lobes_win * math.pi / max(omega0, 1e-9)
        base = torch.cos(omega0 * s)
        xw = math.pi * s / half_support
        win = torch.where(xw.abs() < 1e-12, torch.ones_like(xw), torch.sin(xw) / xw)
        k = base * win
        return torch.where(s.abs() <= half_support, k, torch.zeros_like(k))
    raise ValueError(f"unknown kernel {kernel!r}")


# Width knob name per kernel (the parameter the K_eff bisection inverts).
_WIDTH_KEY = {
    "sinc": "omega_b",
    "gauss": "sigma",
    "gabor": "sigma",
    "alpha": "tau",
    "rect": "tau",
    "sinusoid": "omega0",
}
# Is the width knob a BANDWIDTH (K_eff increases with the knob) or a TIME-SCALE (K_eff decreases)?
# sinc/sinusoid: omega up -> more crossings -> K_eff UP (monotone increasing).
# gauss/gabor(sigma)/alpha/rect: width up -> fewer crossings -> K_eff DOWN (monotone decreasing).
_KNOB_INCREASING = {"sinc": True, "sinusoid": True, "gauss": False,
                    "gabor": False, "alpha": False, "rect": False}


# ============================================================================================
# NUMERIC CERTIFIED AXIS -- spectral moments / effective dimensions from the kernel autocorrelation.
# Replaces v6 line 754 (the flat-band closed-form omega_b = sqrt3 K_eff / T). For ANY kernel we
# build the power spectrum S(omega) = |K_hat(omega)|^2 on the SIM grid (the discretization IS the
# regularization -- crucial for rect, whose continuous lambda_2 diverges), then form lambda_p and the
# effective dimensions. This reuses the cov_numeric autocorrelation logic of v8_gp_witness.
# ============================================================================================
def kernel_spectral_axes(kernel: str, params: dict, T: float, dt: float,
                         n_grid: int) -> dict:
    """Numeric certified-axis quantities for ``kernel`` on the sim grid (dt, n_grid over [0,T]).

    Builds the kernel on a symmetric lag grid (step ``dt``, the SAME discretization the traces use),
    takes its rFFT to get the power spectrum ``S(omega) = |K_hat|^2``, and returns:

    - ``lambda_0`` = trapz S domega,  ``lambda_2`` = trapz omega^2 S,  ``lambda_4`` = trapz omega^4 S;
    - ``Keff_rice`` = ``T sqrt(lambda_2/lambda_0)`` (the Rice crossing axis; flat band -> Omega_b T/sqrt3);
    - ``D_PR``      = power-weighted participation ratio of the spectral measure, lifted to a TEMPORAL
      effective dimension. With p(omega) = S(omega)/lambda_0 the normalized spectral density, the
      participation ratio of the (continuous) spectral measure is PR = (int p)^2 / int p^2 =
      lambda_0^2 / int S^2 domega. As a temporal count over [0,T] we scale by the band's nominal
      crossing density: D_PR = (T/pi) * sqrt(lambda_2/lambda_0) * (lambda_0^2 / int S^2 domega) /
      (lambda_0 / int S domega) ... -- we use the v8 spirit directly: D_PR = (T/pi) * <omega>_S, where
      <omega>_S = int omega S / int S is the power-weighted mean (absolute) frequency, the
      participation-weighted bandwidth that stays FINITE for rect (unlike sqrt(lambda_2/lambda_0)).
      [documented exact formula below]
    - ``kappa_S``   = ``lambda_4 lambda_0 / lambda_2^2`` (spectral kurtosis; flat band -> 9/5=1.8).
    """
    # Symmetric lag grid matching the trace discretization. Support: full window (kernels are
    # compactly supported or decay fast); use lags out to +/- T so the FFT resolves the band.
    n_half = n_grid - 1
    lags = (np.arange(-n_half, n_half + 1) * dt).astype(np.float64)
    s_t = torch.from_numpy(lags)
    k = eval_kernel(s_t, kernel, params).detach().cpu().numpy().astype(np.float64)
    # Power spectrum via FFT of the (real, even or causal) kernel sampled on the lag grid. |K_hat|^2
    # is the power spectrum; the frequency grid omega = 2 pi f with f = fftfreq(len, dt).
    n = k.size
    Khat = np.fft.rfft(k) * dt                       # continuous-FT approximation (Riemann)
    f = np.fft.rfftfreq(n, d=dt)                      # cycles/ms
    omega = 2.0 * math.pi * f                         # rad/ms
    S = np.abs(Khat) ** 2                             # power spectrum S(omega) >= 0
    # Spectral moments (trapz on the sim grid -- the discretization regularizes rect's lambda_2).
    lambda_0 = float(np.trapz(S, omega))
    lambda_2 = float(np.trapz(omega**2 * S, omega))
    lambda_4 = float(np.trapz(omega**4 * S, omega))
    if lambda_0 <= 0.0:
        return {"lambda_0": lambda_0, "lambda_2": lambda_2, "lambda_4": lambda_4,
                "Keff_rice": float("nan"), "D_PR": float("nan"), "kappa_S": float("nan"),
                "bandwidth": float("nan")}
    bandwidth = math.sqrt(lambda_2 / lambda_0)        # rms band half-width (rad/ms)
    Keff_rice = T * bandwidth
    # D_PR: power-weighted mean absolute frequency <omega>_S = int omega S / int S, mapped to a
    # temporal count over [0,T] via D_PR = (T/pi) <omega>_S. <omega>_S is FINITE for rect (the
    # sinc^2 spectrum has integrable omega*S) whereas sqrt(lambda_2/lambda_0) is dt-divergent; this
    # is the v8 participation-ratio spirit (a robust effective dimension) for a generic spectrum.
    mean_omega = float(np.trapz(omega * S, omega) / lambda_0)
    D_PR = (T / math.pi) * mean_omega
    kappa_S = lambda_4 * lambda_0 / (lambda_2 ** 2) if lambda_2 > 0 else float("nan")
    return {"lambda_0": lambda_0, "lambda_2": lambda_2, "lambda_4": lambda_4,
            "Keff_rice": Keff_rice, "D_PR": D_PR, "kappa_S": kappa_S,
            "bandwidth": bandwidth}


def _axis_for_width(kernel: str, width: float, omega0: float, T: float, oversample: float,
                    n_lobes: int, window: str) -> dict:
    """Compute the certified axes for a candidate width, on a self-consistent Nyquist grid.

    The grid step is keyed off the kernel's effective bandwidth (so every kernel gets the same
    oversample-x-Nyquist resolution). We first get a coarse bandwidth estimate, then a stable grid.
    """
    params = _params_for(kernel, width, omega0, n_lobes, window)
    # Coarse bandwidth from an analytic estimate to set an initial dt; then one refine pass on grid.
    bw0 = _coarse_bandwidth(kernel, width, omega0)
    dt = math.pi / (oversample * max(bw0, 1e-6))
    n_grid = max(8, round(T / dt) + 1)
    ax = kernel_spectral_axes(kernel, params, T, dt, n_grid)
    return ax | {"dt": dt, "n_grid": n_grid, "params": params}


def _params_for(kernel: str, width: float, omega0: float, n_lobes: int, window: str) -> dict:
    p = {"n_lobes": n_lobes, "window": window, "omega0": omega0, "n_lobes_win": n_lobes}
    p[_WIDTH_KEY[kernel]] = width
    return p


def _coarse_bandwidth(kernel: str, width: float, omega0: float) -> float:
    """Analytic-ish bandwidth estimate (rad/ms) for sizing the initial Nyquist grid."""
    if kernel == "sinc":
        return width                       # omega_b is the band edge
    if kernel == "sinusoid":
        return max(width, 1e-6)            # omega0 is the (single) frequency
    if kernel == "gauss":
        return 1.0 / max(width, 1e-9)      # sigma_omega = 1/sigma
    if kernel == "gabor":
        return omega0 + 1.0 / max(width, 1e-9)  # center + Gaussian spread
    if kernel == "alpha":
        return 1.0 / max(width, 1e-9)      # 1/tau
    if kernel == "rect":
        return 2.0 * math.pi / max(width, 1e-9)  # first sinc null of the box (2pi/tau)
    return 1.0


def invert_knob_to_keff(kernel: str, target: float, *, T: float, oversample: float,
                        n_lobes: int, window: str, omega0: float,
                        axis: str = "Keff_rice", lo: float = 1e-3, hi: float = 1e4,
                        n_iter: int = 60) -> dict:
    """Bisect the kernel width knob so the MEASURED ``axis`` value hits ``target``.

    ``axis`` is ``Keff_rice`` (default, smooth kernels) or ``D_PR`` (rect, where K_eff_rice diverges
    with dt). Handles both monotone-increasing (sinc/sinusoid: K_eff up with the knob) and
    monotone-decreasing (gauss/gabor/alpha/rect: K_eff down with the knob) width->K_eff maps. Returns
    the inverted width plus the full certified-axis dict at that width.
    """
    incr = _KNOB_INCREASING[kernel]

    def measured(width: float) -> float:
        return _axis_for_width(kernel, width, omega0, T, oversample, n_lobes, window)[axis]

    a, b = lo, hi
    fa = measured(a) - target
    fb = measured(b) - target
    # For a decreasing map, measured(lo) is large and measured(hi) is small (sign of f flips the same
    # way as increasing once we orient by `incr`); bisection just needs a sign change in [a,b].
    if not (math.isfinite(fa) and math.isfinite(fb)):
        # Fall back: scan a log grid for a bracketing pair.
        grid = np.geomspace(lo, hi, 64)
        vals = np.array([measured(w) for w in grid])
        diffs = vals - target
        sign = np.sign(diffs)
        idx = np.where(sign[:-1] * sign[1:] < 0)[0]
        if idx.size == 0:
            # No bracket -> return the closest.
            j = int(np.nanargmin(np.abs(diffs)))
            width = float(grid[j])
            return _axis_for_width(kernel, width, omega0, T, oversample, n_lobes, window) | {
                "width": width, "bisect": "closest"}
        a, b = float(grid[idx[0]]), float(grid[idx[0] + 1])
        fa = measured(a) - target
    for _ in range(n_iter):
        mid = math.sqrt(a * b)            # geometric midpoint (knobs span decades)
        fm = measured(mid) - target
        if not math.isfinite(fm):
            break
        if fa * fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
        if abs(fm) <= 1e-4 * max(1.0, abs(target)):
            break
    width = math.sqrt(a * b)
    return _axis_for_width(kernel, width, omega0, T, oversample, n_lobes, window) | {
        "width": width, "bisect": "ok", "incr": incr}


# --------------------------------------------------------------------------------------------
# K_eff certifier -- inlined from v6 (measure_keff) for the realized-trace cross-check.
# --------------------------------------------------------------------------------------------
def measure_keff(V_samples_or_freqs: np.ndarray, T: float, *, dt: float | None = None) -> dict:
    """Certify ``K_eff = T sqrt(lambda_2/lambda_0)`` from time samples (byte-faithful v6 inline)."""
    arr = np.asarray(V_samples_or_freqs, dtype=float)
    if arr.ndim == 1:
        w = arr
        lambda_0 = 1.0
        lambda_2 = float(np.mean(w**2))
        lambda_4 = float(np.mean(w**4))
        keff_crossing = T * np.sqrt(lambda_2 / lambda_0)
        weights = np.ones_like(w)
        keff_pr = (weights.sum() ** 2) / np.sum(weights**2)
        return {"keff_crossing": keff_crossing, "keff_pr": float(keff_pr),
                "lambda_0": lambda_0, "lambda_2": lambda_2, "lambda_4": lambda_4,
                "method": "frequencies"}
    elif arr.ndim == 2:
        if dt is None:
            raise ValueError("dt is required when passing 2-D time samples")
        V = arr
        M = V.shape[0]
        lambda_0 = float(np.mean(V**2))
        signs = np.signbit(V)
        crossings = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)
        nu_0 = np.mean(crossings) / T
        lambda_2 = (np.pi * nu_0) ** 2 * lambda_0
        keff_crossing = T * np.sqrt(lambda_2 / lambda_0)
        return {"keff_crossing": keff_crossing, "keff_pr": float("nan"),
                "lambda_0": lambda_0, "lambda_2": lambda_2, "lambda_4": float("nan"),
                "method": "time-samples", "nu_0_measured": nu_0, "M": M}
    else:
        raise ValueError("expected 1-D frequencies or 2-D (M, n_t) time samples")


def certify_keff_from_traces(s: torch.Tensor, t_grid: torch.Tensor, T: float,
                             n_probe: int = 64) -> dict:
    """Measure K_eff from realized init voltages (byte-faithful v6 inline)."""
    sb, p, g, n = s.shape
    dev = s.device
    flat = s.reshape(sb * p, g, n)
    m = min(n_probe, flat.shape[0])
    sel = flat[:m]
    gen = torch.Generator(device=dev); gen.manual_seed(20260622)
    xi = torch.randn((m, n), generator=gen, device=dev, dtype=s.dtype)
    V = torch.einsum("mgn,mn->mg", sel, xi)
    V = V - V.mean(dim=1, keepdim=True)
    dt = float(t_grid[1] - t_grid[0])
    return measure_keff(V.detach().cpu().numpy(), T, dt=dt)


# --------------------------------------------------------------------------------------------
# Reproducible per-cell seeding (v6 recipe).
# --------------------------------------------------------------------------------------------
def cell_seed(master: int, n: int, alpha: float, k: float, tag: int = 0) -> int:
    """A scheduling-independent 63-bit seed for a cell (matches v6's SeedSequence recipe)."""
    ss = np.random.SeedSequence([master, n, round(1000 * alpha), round(k), tag])
    return int(ss.generate_state(1, dtype=np.uint64)[0]) >> 1


# --------------------------------------------------------------------------------------------
# Pattern generation (RMS 2010 Poisson input) -- REUSED VERBATIM from v6.
# --------------------------------------------------------------------------------------------
def make_patterns(
    n_tasks: int,
    n_patterns: int,
    n_aff: int,
    t_window: float,
    rng: np.random.Generator,
    device: torch.device,
    mean_spikes: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Draw ``n_tasks`` independent Poisson pattern sets (RMS 2010), balanced +/-1 labels."""
    shape = (n_tasks, n_patterns, n_aff)
    counts = rng.poisson(float(mean_spikes), size=shape).astype(np.int64)
    max_spikes = max(1, int(counts.max()))
    spike_times = (t_window * rng.random((n_tasks, n_patterns, n_aff, max_spikes))).astype(np.float32)
    spike_idx = np.arange(max_spikes).reshape(1, 1, 1, max_spikes)
    valid = (spike_idx < counts[..., None]).astype(np.float32)
    labels = np.where(rng.random((n_tasks, n_patterns)) < 0.5, 1.0, -1.0).astype(np.float32)
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return to(spike_times), to(valid), to(labels), max_spikes


def precompute_traces(
    spike_times: torch.Tensor,  # (Sb, P, N, max_spikes)
    valid: torch.Tensor,        # (Sb, P, N, max_spikes)
    t_grid: torch.Tensor,       # (G,)
    kernel: str,
    params: dict,
    elem_budget: int = 64_000_000,
) -> torch.Tensor:
    """Per-afferent traces ``s[b,g,p,i] = sum_f eval_kernel(t_g - spike[b,p,i,f]) * valid``.

    Identical chunking strategy to v6.precompute_traces; the ONLY change is the kernel call site
    (``eval_kernel(tt - st, kernel, params)`` replaces ``band_kernel(...)``).
    """
    sb, p, n, ms = spike_times.shape
    g = t_grid.shape[0]
    s = torch.empty((sb, p, g, n), device=spike_times.device, dtype=torch.float32)
    per_g = max(1, sb * p * n * ms)
    g_chunk = max(1, elem_budget // per_g)
    st = spike_times.unsqueeze(0)
    vd = valid.unsqueeze(0)
    for g0 in range(0, g, g_chunk):
        g1 = min(g0 + g_chunk, g)
        tt = t_grid[g0:g1].view(-1, 1, 1, 1, 1)
        contrib = eval_kernel(tt - st, kernel, params) * vd               # (gc, Sb, P, N, ms)
        s[:, :, g0:g1, :] = contrib.sum(dim=4).permute(1, 2, 0, 3)
    return s


# --------------------------------------------------------------------------------------------
# Forward pass + faithful online Gutig-Sompolinsky training -- REUSED VERBATIM from v6.
# --------------------------------------------------------------------------------------------
def _forward(s: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``V[b,p,g] = sum_i s[b,p,g,i] w[b,i]``; return ``(vmax, argmax_t)`` each ``(Sb, P)``."""
    v = torch.einsum("bpgn,bn->bpg", s, w)
    vmax, targ = v.max(dim=2)
    return vmax, targ


EXP_DECAY_K: float = 5.0


def scheduled_lr(
    lr: float,
    ep: int,
    *,
    epochs: int,
    schedule: str = "none",
    warmup: int = 0,
    step_size: int = 0,
    gamma: float = 0.1,
    floor: float = 0.0,
    n_cycles: int = 1,
    exp_k: float = EXP_DECAY_K,
) -> float:
    """Scheduled learning rate at epoch ``ep`` (verbatim port of v6.scheduled_lr)."""
    if warmup and ep < warmup:
        return lr * (ep + 1) / warmup
    span = max(1, epochs - warmup)
    t = min(1.0, max(0.0, (ep - warmup) / span))
    if schedule == "cosine":
        if n_cycles > 1:
            tc = span / n_cycles
            tcur = ((ep - warmup) % tc) / tc if tc > 0 else 0.0
            cos = 0.5 * (1.0 + math.cos(math.pi * tcur))
        else:
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return lr * (floor + (1.0 - floor) * cos)
    if schedule == "linear":
        return lr * (floor + (1.0 - floor) * (1.0 - t))
    if schedule == "exp":
        return lr * math.exp(-exp_k * t)
    if schedule == "step":
        if step_size <= 0:
            return lr
        return lr * (gamma ** (ep // step_size))
    return lr


def train_batch(
    s: torch.Tensor,        # (Sb, P, G, N) precomputed traces
    labels: torch.Tensor,   # (Sb, P) in {-1,+1}
    w_init: torch.Tensor,   # (Sb, N)
    *,
    lr: float,
    momentum: float = 0.0,
    epochs: int,
    threshold: torch.Tensor | None = None,
    patience: int = 0,
    log_every: int = 0,
    log_tag: str = "",
    train_seed: int = 0,
    capture: bool = False,
    metric_cb=None,
    mode: str = "online",
    optimizer: str = "momentum",
    lr_schedule: str = "none",
    lr_warmup: int = 0,
    lr_floor: float = 0.0,
    lr_cycles: int = 1,
    lr_exp_k: float = EXP_DECAY_K,
    lr_step_size: int = 0,
    lr_gamma: float = 0.1,
    batch_size: int = 0,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    rms_alpha: float = 0.99,
    rms_eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Gutig-Sompolinsky rule, vectorised across tasks -- REUSED VERBATIM from v6.train_batch.

    See v6 for the full docstring (online faithful GS default; minibatch Adam/RMSprop strong-learner
    path; per-epoch capture). Threshold fixed (median V_max at init); divergence guard on NaN weights.
    """
    sb, p, _, n = s.shape
    dev = s.device
    shuffle_gen = torch.Generator(device=dev)
    shuffle_gen.manual_seed(int(train_seed) & 0x7FFFFFFFFFFFFFFF)
    w = w_init.clone()
    vmax, _ = _forward(s, w)
    if threshold is None:
        threshold = vmax.median(dim=1).values
    init_fire_rate = (vmax >= threshold[:, None]).float().mean(dim=1)
    converged = torch.zeros(sb, dtype=torch.bool, device=dev)
    epochs_run = torch.full((sb,), epochs, dtype=torch.int64, device=dev)
    arange_sb = torch.arange(sb, device=dev)
    vel = torch.zeros_like(w)
    best_conv = 0
    stall = 0
    adam_m = torch.zeros_like(w)
    adam_v = torch.zeros_like(w)
    adam_step = 0
    last_ep = -1

    cap = bool(capture) and mode in ("online", "minibatch")
    if cap:
        err_buf = torch.full((sb, epochs), float("nan"), device=dev)
        loss_buf = torch.full((sb, epochs), float("nan"), device=dev)
        wnorm_buf = torch.full((sb, epochs), float("nan"), device=dev)
        kp_buf = torch.full((sb, epochs), float("nan"), device=dev)
        km_buf = torch.full((sb, epochs), float("nan"), device=dev)
        npos = (labels > 0).float().sum(dim=1).clamp(min=1.0)
        nneg = (labels < 0).float().sum(dim=1).clamp(min=1.0)

    def _lr_at(ep: int) -> float:
        return scheduled_lr(lr, ep, epochs=epochs, schedule=lr_schedule, warmup=lr_warmup,
                            step_size=lr_step_size, gamma=lr_gamma,
                            floor=lr_floor, n_cycles=lr_cycles, exp_k=lr_exp_k)

    for ep in range(epochs):
        active = (~converged).float().unsqueeze(-1)
        if mode == "minibatch":
            lr_ep = _lr_at(ep)
            order = torch.randperm(p, device=dev, generator=shuffle_gen)
            bs = max(1, min(batch_size, p))
            n_steps = (p + bs - 1) // bs
            act = active.squeeze(-1)
            viol_enc = torch.zeros(sb, device=dev)
            if cap:
                err_acc = torch.zeros(sb, device=dev)
                cost_acc = torch.zeros(sb, device=dev)
                kp_acc = torch.zeros(sb, device=dev)
                km_acc = torch.zeros(sb, device=dev)
            for st_i in range(n_steps):
                bidx = order[st_i * bs:(st_i + 1) * bs]
                sb_s = s[:, bidx]
                lab_b = labels[:, bidx]
                bb = sb_s.shape[1]
                vb = torch.einsum("bkgn,bn->bkg", sb_s, w)
                vmx, tg = vb.max(dim=2)
                smarg = (vmx - threshold[:, None]) * lab_b
                viol = (smarg < 0.0).float()
                nb = float(bb)
                viol_enc = viol_enc + viol.sum(dim=1) * act
                gidx = tg.view(sb, bb, 1, 1).expand(sb, bb, 1, n)
                gpat = torch.gather(sb_s, 2, gidx).squeeze(2)
                g = ((lab_b * viol).unsqueeze(-1) * gpat).sum(dim=1)
                g = g / nb
                has_viol = (viol.sum(dim=1) > 0).float().unsqueeze(-1)
                gate = active * has_viol
                if optimizer == "adam":
                    b1, b2 = adam_betas
                    adam_m.mul_(b1).add_(g, alpha=1 - b1)
                    adam_v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    adam_step += 1
                    mhat = adam_m / (1 - b1 ** adam_step)
                    vhat = adam_v / (1 - b2 ** adam_step)
                    step = lr_ep * mhat / (vhat.sqrt() + adam_eps)
                elif optimizer == "rmsprop":
                    adam_v.mul_(rms_alpha).addcmul_(g, g, value=1 - rms_alpha)
                    step = lr_ep * g / (adam_v.sqrt() + rms_eps)
                else:
                    vel = gate * (momentum * vel + g) + (1.0 - gate) * vel
                    step = lr_ep * vel
                w = w + step * gate
                if cap:
                    err_acc = err_acc + viol.sum(dim=1)
                    cost_acc = cost_acc + torch.relu(-smarg).sum(dim=1)
                    kp_acc = kp_acc + (smarg * (lab_b > 0).float()).sum(dim=1)
                    km_acc = km_acc + (smarg * (lab_b < 0).float()).sum(dim=1)
            if cap:
                err_buf[:, ep] = err_acc / p
                loss_buf[:, ep] = cost_acc / p
                wnorm_buf[:, ep] = w.norm(dim=1)
                kp_buf[:, ep] = kp_acc / npos
                km_buf[:, ep] = km_acc / nneg
                if metric_cb is not None:
                    metric_cb(ep + 1, float((err_acc / p).mean()), float((cost_acc / p).mean()))
            newly = (viol_enc == 0) & (~converged) & torch.isfinite(w).all(dim=1)
            last_ep = ep
            epochs_run = torch.where(newly, torch.tensor(ep + 1, device=dev), epochs_run)
            converged = converged | newly
            if bool(converged.all()):
                break
            n_conv = int(converged.sum().item())
            if n_conv > best_conv:
                best_conv = n_conv
                stall = 0
            else:
                stall += 1
            if log_every and (ep + 1) % log_every == 0:
                print(f"    [{log_tag}] ep={ep + 1} conv={n_conv}/{sb} stall={stall}", flush=True)
            if patience and stall >= patience and n_conv < sb:
                if log_every:
                    print(f"    [{log_tag}] early-stop at ep={ep + 1} (no progress {stall} ep)", flush=True)
                break
            continue
        order = torch.randperm(p, device=dev, generator=shuffle_gen)
        updates = torch.zeros(sb, device=dev)
        if cap:
            err_acc = torch.zeros(sb, device=dev)
            cost_acc = torch.zeros(sb, device=dev)
            kp_acc = torch.zeros(sb, device=dev)
            km_acc = torch.zeros(sb, device=dev)
        for pi in order:
            sp = s[:, pi]
            vp = torch.einsum("bgn,bn->bg", sp, w)
            vmx, tg = vp.max(dim=1)
            smarg = (vmx - threshold) * labels[:, pi]
            if cap:
                lab_pi = labels[:, pi]
                err_acc = err_acc + (smarg < 0.0).float()
                cost_acc = cost_acc + torch.relu(-smarg)
                kp_acc = kp_acc + smarg * (lab_pi > 0).float()
                km_acc = km_acc + smarg * (lab_pi < 0).float()
            e = (smarg < 0.0).float() * active.squeeze(-1)
            updates = updates + e
            grad = sp[arange_sb, tg]
            corr = (lr * labels[:, pi]).unsqueeze(-1) * grad
            e_col = e.unsqueeze(-1)
            vel = e_col * (corr + momentum * vel) + (1.0 - e_col) * vel
            w = w + e_col * vel
        if cap:
            err_buf[:, ep] = err_acc / p
            loss_buf[:, ep] = cost_acc / p
            wnorm_buf[:, ep] = w.norm(dim=1)
            kp_buf[:, ep] = kp_acc / npos
            km_buf[:, ep] = km_acc / nneg
            if metric_cb is not None:
                metric_cb(ep + 1, float((err_acc / p).mean()), float((cost_acc / p).mean()))
        last_ep = ep
        newly = (updates == 0) & (~converged)
        epochs_run = torch.where(newly, torch.tensor(ep + 1, device=dev), epochs_run)
        converged = converged | newly
        if bool(converged.all()):
            break
        n_conv = int(converged.sum().item())
        if n_conv > best_conv:
            best_conv = n_conv
            stall = 0
        else:
            stall += 1
        if log_every and (ep + 1) % log_every == 0:
            print(f"    [{log_tag}] ep={ep + 1} conv={n_conv}/{sb} stall={stall}", flush=True)
        if patience and stall >= patience and n_conv < sb:
            if log_every:
                print(f"    [{log_tag}] early-stop at ep={ep + 1} (no progress {stall} ep)", flush=True)
            break

    if mode == "minibatch":
        ran = last_ep + 1
        epochs_run = torch.where(converged, epochs_run, torch.full_like(epochs_run, ran))

    vmax, _ = _forward(s, w)
    pred = torch.where(vmax >= threshold[:, None], 1.0, -1.0)
    final_errfrac = (pred != labels).float().mean(dim=1)
    out = {
        "converged": converged,
        "epochs_run": epochs_run,
        "weights": w,
        "threshold": threshold,
        "init_fire_rate": init_fire_rate,
        "final_errfrac": final_errfrac,
    }
    if cap:
        ne = last_ep + 1
        out["cap"] = {
            "traj_err": err_buf[:, :ne],
            "traj_loss": loss_buf[:, :ne],
            "traj_wnorm": wnorm_buf[:, :ne],
            "traj_kp": kp_buf[:, :ne],
            "traj_km": km_buf[:, :ne],
            "n_epochs": ne,
        }
    return out


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------
def _alpha_grid(ov: dict[str, str]) -> list[float]:
    if "alphas" in ov:
        return [float(x) for x in ov["alphas"].split(",")]
    a0 = float(ov.get("alpha_min", "1.6"))
    a1 = float(ov.get("alpha_max", "3.6"))
    step = float(ov.get("alpha_step", "0.2"))
    n = int(round((a1 - a0) / step)) + 1
    return [round(a0 + i * step, 4) for i in range(n)]


def _crossing_alpha(alphas: list[float], psolve: list[float]) -> float:
    """Linear-interpolated 1/2-crossing of P_solve(alpha) (descending crossing); NaN if none."""
    for i in range(len(alphas) - 1):
        a0, a1 = alphas[i], alphas[i + 1]
        p0, p1 = psolve[i], psolve[i + 1]
        if (p0 - 0.5) >= 0.0 > (p1 - 0.5):
            if p0 == p1:
                return float(0.5 * (a0 + a1))
            return float(a0 + (a1 - a0) * (p0 - 0.5) / (p0 - p1))
    return float("nan")


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))

    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    kernel = ov.get("kernel", "sinc")            # sinc | gauss | gabor | alpha | rect | sinusoid
    keff_list = [float(x) for x in ov.get("Keff_list", "2,8,32").split(",")]
    n_list = [int(x) for x in ov.get("N_list", "500").split(",")]
    alphas = _alpha_grid(ov)
    if "seeds" in ov:
        seeds = [int(x) for x in ov["seeds"].split(",") if x != ""]
    else:
        seeds = list(range(int(ov.get("n_seeds", "24"))))
    n_seeds = len(seeds)
    T_ms = float(ov.get("T_ms", "500"))
    oversample = float(ov.get("oversample", "4"))
    n_lobes = int(ov.get("n_lobes", "6"))
    window = ov.get("window", "lanczos")
    # gabor: omega0 is the carrier; the bisection sets sigma so Q = omega0*sigma varies along K_eff.
    # To probe the TWO regimes set Q explicitly: if gabor_Q>0, omega0 is derived per cell from the
    # bisected sigma as omega0 = Q/sigma (band-pass at fixed Q); else omega0 is the fixed carrier.
    gabor_omega0 = float(ov.get("gabor_omega0", "1.0"))
    gabor_Q = float(ov.get("gabor_Q", "0"))      # if >0, hold Q=omega0*sigma fixed (regime knob)
    # sinusoid: omega0 IS the bisected knob (single frequency); n_lobes_win sets its window span.
    # rect: bisect to a target D_PR (K_eff_rice diverges with dt) and ALSO log K_eff_rice.
    axis = ov.get("axis", "")                    # '' -> auto (D_PR for rect, Keff_rice otherwise)
    if not axis:
        axis = "D_PR" if kernel == "rect" else "Keff_rice"
    density_coeff = float(ov.get("density_coeff", "0.05"))
    mean_spikes_fixed = float(ov.get("mean_spikes_fixed", "0"))
    capture = int(ov.get("capture", "0"))
    epochs = int(ov.get("epochs", "2000"))
    lr = float(ov.get("lr", "0.05"))
    momentum = float(ov.get("momentum", "0.0"))
    mode = ov.get("mode", "online")
    optimizer = ov.get("optimizer", "momentum")
    lr_schedule = ov.get("lr_schedule", "none")
    lr_warmup = int(ov.get("lr_warmup", "0"))
    lr_floor = float(ov.get("lr_floor", "0.0"))
    lr_cycles = int(ov.get("lr_cycles", "1"))
    lr_step_size = int(ov.get("lr_step_size", "0"))
    lr_gamma = float(ov.get("lr_gamma", "0.1"))
    batch_size = int(ov.get("batch_size", "0"))
    adam_b1 = float(ov.get("adam_b1", "0.9"))
    adam_b2 = float(ov.get("adam_b2", "0.999"))
    adam_eps = float(ov.get("adam_eps", "1e-8"))
    rms_alpha = float(ov.get("rms_alpha", "0.99"))
    rms_eps = float(ov.get("rms_eps", "1e-8"))
    n_restarts = int(ov.get("n_restarts", "5"))
    patience = int(ov.get("patience", "0"))
    log_every = int(ov.get("log_every", "0"))
    sigma_w = float(ov.get("sigma_w", "1.0"))
    elem_budget = int(float(ov.get("elem_budget", "64e6")))
    keff_probe = int(ov.get("keff_probe", "64"))
    require_cuda = int(ov.get("require_cuda", "0"))

    if kernel not in _WIDTH_KEY:
        print(f"FATAL: unknown kernel {kernel!r}; choose from {sorted(_WIDTH_KEY)}", flush=True)
        sys.exit(2)
    if require_cuda and not torch.cuda.is_available():
        print("FATAL: require_cuda=1 but no CUDA device. Exiting non-zero (no CPU-smoke at GPU price).",
              flush=True)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "s_budget" in ov:
        s_budget = int(float(ov["s_budget"]))
    elif torch.cuda.is_available():
        s_budget = int(0.55 * torch.cuda.get_device_properties(0).total_memory)
    else:
        s_budget = int(2e9)
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gb:.0f} GB) | s_budget={s_budget/1e9:.1f} GB",
              flush=True)
    else:
        print("WARNING: no CUDA device; running on CPU (smoke only).", flush=True)
    print(
        f"V13 kernel capacity | kernel={kernel} axis={axis} seed={master_seed} device={device} "
        f"Keff={keff_list} N={n_list} alphas={alphas} seeds={seeds} epochs={epochs} lr={lr} "
        f"mom={momentum} T={T_ms} oversample={oversample} n_lobes={n_lobes} window={window} "
        f"density_coeff={density_coeff} restarts={n_restarts} mode={mode} optimizer={optimizer} "
        f"lr_schedule={lr_schedule} batch_size={batch_size} gabor_omega0={gabor_omega0} gabor_Q={gabor_Q}",
        flush=True,
    )

    _log_metric = None
    if capture:
        try:
            from lab.metrics import log_metric as _log_metric  # type: ignore
        except Exception:
            _log_metric = None

    started = time.time()
    rows: list[dict[str, float]] = []
    history_rows: list[dict[str, float]] = []
    per_keff: dict[float, dict] = {}
    n_cells = len(keff_list) * len(n_list) * len(alphas)
    cell_i = 0
    first_cell_done = False

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

    for keff_t in keff_list:
        # ---- NUMERIC CERTIFIED AXIS: bisect the kernel width knob to hit the target K_eff. ----
        # (Replaces v6's flat-band closed form omega_b = sqrt3 K_eff / T.) For gabor with a fixed-Q
        # regime, omega0 is re-derived from the bisected sigma; we bisect once with the carrier set,
        # then (if gabor_Q>0) re-fit omega0 = Q/sigma and recompute the axes.
        inv = invert_knob_to_keff(kernel, keff_t, T=T_ms, oversample=oversample,
                                   n_lobes=n_lobes, window=window, omega0=gabor_omega0, axis=axis)
        width = inv["width"]
        params = inv["params"]
        Q = float("nan")
        if kernel == "gabor":
            if gabor_Q > 0:
                # Hold Q = omega0 * sigma fixed: set omega0 = Q / sigma at the bisected sigma, then
                # re-bisect sigma with that coupling. Simpler stable route: fix Q, bisect sigma so the
                # axis hits target with omega0 = Q/sigma inside the kernel.
                def axis_at_sigma(sig: float) -> float:
                    p = _params_for("gabor", sig, gabor_Q / max(sig, 1e-9), n_lobes, window)
                    bw0 = _coarse_bandwidth("gabor", sig, gabor_Q / max(sig, 1e-9))
                    dt0 = math.pi / (oversample * max(bw0, 1e-6))
                    ng0 = max(8, round(T_ms / dt0) + 1)
                    return kernel_spectral_axes("gabor", p, T_ms, dt0, ng0)[axis]
                a, b = 1e-3, 1e3
                fa = axis_at_sigma(a) - keff_t
                for _ in range(60):
                    mid = math.sqrt(a * b)
                    fm = axis_at_sigma(mid) - keff_t
                    if not math.isfinite(fm):
                        break
                    if fa * fm <= 0.0:
                        b = mid
                    else:
                        a, fa = mid, fm
                    if abs(fm) <= 1e-4 * max(1.0, keff_t):
                        break
                width = math.sqrt(a * b)
                omega0 = gabor_Q / max(width, 1e-9)
                Q = gabor_Q
                params = _params_for("gabor", width, omega0, n_lobes, window)
                bw0 = _coarse_bandwidth("gabor", width, omega0)
                dt = math.pi / (oversample * max(bw0, 1e-6))
                n_grid = max(8, round(T_ms / dt) + 1)
                inv = kernel_spectral_axes("gabor", params, T_ms, dt, n_grid) | {
                    "dt": dt, "n_grid": n_grid, "params": params, "width": width}
            else:
                Q = gabor_omega0 * width

        dt = inv["dt"]
        n_grid = inv["n_grid"]
        Keff_rice = inv["Keff_rice"]
        D_PR = inv["D_PR"]
        kappa_S = inv["kappa_S"]
        t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * dt

        if mean_spikes_fixed > 0:
            mean_spikes = mean_spikes_fixed
        else:
            mean_spikes = max(1e-3, density_coeff * keff_t)

        print(f"  [axis] kernel={kernel} target({axis})={keff_t:.3f} -> {_WIDTH_KEY[kernel]}={width:.5g} "
              f"| Keff_rice={Keff_rice:.3f} D_PR={D_PR:.3f} kappa_S={kappa_S:.3f} Q={Q} "
              f"dt={dt:.4g} n_grid={n_grid}", flush=True)

        keff_alpha_psolve: list[float] = []
        keff_measured_cell: float = float("nan")
        for n_aff in n_list:
            for alpha in alphas:
                cell_i += 1
                p = round(alpha * n_aff)
                if p == 0:
                    continue
                rng = np.random.default_rng(cell_seed(master_seed, n_aff, alpha, keff_t))
                spikes, valid, labels, _ = make_patterns(n_seeds, p, n_aff, T_ms, rng, device,
                                                          mean_spikes=mean_spikes)

                per_task_bytes = n_grid * p * n_aff * 4
                sb_size = max(1, min(n_seeds, s_budget // max(1, per_task_bytes)))

                conv = torch.zeros(n_seeds, dtype=torch.bool, device=device)
                conv_multi = torch.zeros(n_seeds, dtype=torch.bool, device=device)
                ep_run = torch.full((n_seeds,), float(epochs), device=device)
                fire = torch.zeros(n_seeds, device=device)
                errf = torch.zeros(n_seeds, device=device)
                keff_meas_batches: list[float] = []

                for b0 in range(0, n_seeds, sb_size):
                    b1 = min(b0 + sb_size, n_seeds)
                    s = precompute_traces(spikes[b0:b1], valid[b0:b1], t_grid,
                                          kernel, params, elem_budget)
                    lab_b = labels[b0:b1]
                    cert = certify_keff_from_traces(s, t_grid, T_ms, n_probe=keff_probe)
                    keff_meas_batches.append(cert["keff_crossing"])
                    wrng = np.random.default_rng(cell_seed(master_seed, n_aff, alpha, keff_t, tag=1) + b0)
                    w0 = torch.from_numpy(
                        (sigma_w * wrng.standard_normal((b1 - b0, n_aff))).astype(np.float32)).to(device)
                    stream_this = capture and (not first_cell_done) and (_log_metric is not None)

                    def _mcb(ep: int, errm: float, lossm: float) -> None:
                        _log_metric("train_err", float(errm), step=int(ep))  # type: ignore
                        _log_metric("loss", float(lossm), step=int(ep))      # type: ignore

                    res = train_batch(s, lab_b, w0, lr=lr, momentum=momentum, epochs=epochs,
                                      patience=patience, log_every=log_every,
                                      log_tag=f"{kernel}K{round(keff_t)}N{n_aff}a{alpha:.2f}b{b0}",
                                      train_seed=cell_seed(master_seed, n_aff, alpha, keff_t, tag=10) + b0,
                                      capture=bool(capture),
                                      metric_cb=_mcb if stream_this else None,
                                      mode=mode, optimizer=optimizer, lr_schedule=lr_schedule,
                                      lr_warmup=lr_warmup, lr_floor=lr_floor, lr_cycles=lr_cycles,
                                      lr_step_size=lr_step_size, lr_gamma=lr_gamma, batch_size=batch_size,
                                      adam_betas=(adam_b1, adam_b2), adam_eps=adam_eps,
                                      rms_alpha=rms_alpha, rms_eps=rms_eps)
                    if stream_this:
                        first_cell_done = True
                    if capture and "cap" in res:
                        cd = res["cap"]
                        te = cd["traj_err"].cpu().numpy()
                        tl = cd["traj_loss"].cpu().numpy()
                        tkp = cd["traj_kp"].cpu().numpy()
                        tkm = cd["traj_km"].cpu().numpy()
                        twn = cd["traj_wnorm"].cpu().numpy()
                        nb_seeds, ne = te.shape
                        for j in range(nb_seeds):
                            sd = seeds[b0 + j]
                            for e_i in range(ne):
                                history_rows.append({
                                    "kernel": kernel,
                                    "Keff_target": keff_t,
                                    "N": n_aff,
                                    "alpha": alpha,
                                    "seed_batch": sd,
                                    "epoch": e_i + 1,
                                    "train_err": float(te[j, e_i]),
                                    "loss": float(tl[j, e_i]),
                                    "kappa_plus": float(tkp[j, e_i]),
                                    "kappa_minus": float(tkm[j, e_i]),
                                    "wnorm": float(twn[j, e_i]),
                                })
                    conv[b0:b1] = res["converged"]
                    ep_run[b0:b1] = res["epochs_run"].float()
                    fire[b0:b1] = res["init_fire_rate"]
                    errf[b0:b1] = res["final_errfrac"]

                    any_conv = res["converged"].clone()
                    thr = res["threshold"]
                    for r in range(1, n_restarts):
                        rrng = np.random.default_rng(
                            cell_seed(master_seed, n_aff, alpha, keff_t, tag=2 + r) + b0)
                        wr = torch.from_numpy(
                            (sigma_w * rrng.standard_normal((b1 - b0, n_aff))).astype(np.float32)).to(device)
                        rr = train_batch(s, lab_b, wr, lr=lr, momentum=momentum, epochs=epochs,
                                         threshold=thr, patience=patience,
                                         train_seed=cell_seed(master_seed, n_aff, alpha, keff_t, tag=20 + r) + b0,
                                         mode=mode, optimizer=optimizer, lr_schedule=lr_schedule,
                                         lr_warmup=lr_warmup, lr_floor=lr_floor, lr_cycles=lr_cycles,
                                         lr_step_size=lr_step_size, lr_gamma=lr_gamma, batch_size=batch_size,
                                         adam_betas=(adam_b1, adam_b2), adam_eps=adam_eps,
                                         rms_alpha=rms_alpha, rms_eps=rms_eps)
                        newly_r = rr["converged"] & ~any_conv
                        ep_run[b0:b1] = torch.where(newly_r, rr["epochs_run"].float(), ep_run[b0:b1])
                        any_conv = any_conv | rr["converged"]
                        if bool(any_conv.all()):
                            break
                    conv_multi[b0:b1] = any_conv
                    del s
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                keff_measured_cell = float(np.mean(keff_meas_batches))
                n_solved = int(conv_multi.sum().item())
                p_solve = n_solved / n_seeds
                keff_alpha_psolve.append(p_solve)
                conv_np = conv_multi.cpu().numpy()
                ep_np = ep_run.cpu().numpy()
                fire_np = fire.cpu().numpy()
                errf_np = errf.cpu().numpy()
                for j, sd in enumerate(seeds):
                    solved = int(conv_np[j])
                    rows.append({
                        "seed": sd,
                        "kernel": kernel,
                        "knob": width,
                        "Keff_target": keff_t,
                        "Keff_measured": keff_measured_cell,
                        "Keff_rice": Keff_rice,
                        "D_PR": D_PR,
                        "kappa_S": kappa_S,
                        "Q": Q,
                        "N": n_aff,
                        "alpha": alpha,
                        "P": p,
                        "solved": solved,
                        "epochs_to_solve": int(ep_np[j]) if solved else -1,
                        "final_train_err": float(errf_np[j]),
                        "init_fire_rate": float(fire_np[j]),
                        "n_grid": n_grid,
                        "dt": dt,
                        "mean_spikes": mean_spikes,
                        "n_restarts": n_restarts,
                        "optimizer": optimizer,
                        "lr_schedule": lr_schedule,
                        "batch_size": batch_size,
                    })
                _flush_new_rows()
                p_se = math.sqrt(max(p_solve * (1 - p_solve), 0.0) / n_seeds)
                print(
                    f"[{cell_i}/{n_cells}] {kernel} Keff_t={keff_t:.1f} Keff_meas={keff_measured_cell:.2f} "
                    f"Keff_rice={Keff_rice:.2f} D_PR={D_PR:.2f} N={n_aff} a={alpha:.3f} P={p} "
                    f"p_solve={p_solve:.3f}+/-{p_se:.3f} n_grid={n_grid} fire0={float(fire.mean().item()):.2f}",
                    flush=True,
                )
        if n_list:
            ahat = _crossing_alpha(alphas, keff_alpha_psolve)
            per_keff[keff_t] = {
                "kernel": kernel,
                "Keff_target": keff_t,
                "Keff_measured": keff_measured_cell,
                "Keff_rice": Keff_rice,
                "D_PR": D_PR,
                "kappa_S": kappa_S,
                "Q": Q,
                "knob": width,
                "n_grid": n_grid,
                "dt": dt,
                "mean_spikes": mean_spikes,
                "alphas": alphas,
                "p_solve": keff_alpha_psolve,
                "alpha_hat_c": ahat,
                "ln_ln_Keff": (math.log(math.log(keff_measured_cell))
                               if keff_measured_cell > math.e else float("nan")),
            }

    elapsed = time.time() - started
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = None
    results = {
        "experiment": "v13_kernel_capacity",
        "kernel": kernel,
        "axis": axis,
        "git_sha": git_sha,
        "versions": {"numpy": np.__version__, "torch": torch.__version__,
                     "python": sys.version.split()[0]},
        "params": {
            "master_seed": master_seed, "kernel": kernel, "axis": axis,
            "Keff_list": keff_list, "N_list": n_list, "alphas": alphas,
            "seeds": seeds, "n_seeds": n_seeds, "T_ms": T_ms, "oversample": oversample,
            "n_lobes": n_lobes, "window": window, "density_coeff": density_coeff,
            "gabor_omega0": gabor_omega0, "gabor_Q": gabor_Q,
            "epochs": epochs, "lr": lr, "momentum": momentum, "n_restarts": n_restarts,
            "patience": patience, "sigma_w": sigma_w, "keff_probe": keff_probe,
            "mode": mode, "optimizer": optimizer, "lr_schedule": lr_schedule,
            "lr_warmup": lr_warmup, "lr_floor": lr_floor, "lr_cycles": lr_cycles,
            "lr_step_size": lr_step_size, "lr_gamma": lr_gamma, "batch_size": batch_size,
            "adam_betas": [adam_b1, adam_b2], "adam_eps": adam_eps,
            "rms_alpha": rms_alpha, "rms_eps": rms_eps,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "per_keff": [per_keff[k] for k in keff_list if k in per_keff],
        "rows": rows,
        "elapsed_seconds": elapsed,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    _flush_new_rows()
    csv_file.close()
    if capture and history_rows:
        hcols = ["kernel", "Keff_target", "N", "alpha", "seed_batch", "epoch",
                 "train_err", "loss", "kappa_plus", "kappa_minus", "wnorm"]
        with (run_dir / "history.csv").open("w", newline="") as hf:
            hwtr = csv.DictWriter(hf, fieldnames=hcols)
            hwtr.writeheader()
            hwtr.writerows(history_rows)
    print(f"done in {elapsed:.1f}s -> {run_dir}/results.json ({len(rows)} rows, "
          f"{len(per_keff)} K_eff cells)"
          + (f", history.csv ({len(history_rows)} epoch-rows)" if capture and history_rows else ""),
          flush=True)
    for k in keff_list:
        if k in per_keff:
            pk = per_keff[k]
            print(f"  {kernel} K_eff target={k:.1f} measured={pk['Keff_measured']:.2f} "
                  f"Keff_rice={pk['Keff_rice']:.2f} D_PR={pk['D_PR']:.2f} "
                  f"alpha_hat_c={pk['alpha_hat_c']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
