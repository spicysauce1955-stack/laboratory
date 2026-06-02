"""S1 -- Tempotron V_max distribution vs Gumbel EVT prediction (V2 pre-flight).

Self-contained Experiment-Contract entrypoint (numpy + scipy + torch only). The lab ships this
file unchanged; no ``tempotron`` imports.

Theory (Rubin, Monasson & Sompolinsky 2010 PRL 105, 218102, eq. 4 + Fig 1b)
--------------------------------------------------------------------------
Drive the tempotron with random Gaussian weights ``w_i ~ N(0, 1)`` and Poisson spike trains
of rate 1/T on the window ``[0, T]`` (the RMS input model). The subthreshold voltage

    V(t) = sum_i w_i  sum_{f}  K(t - t_i^f),   K(t) the normalised PSP kernel,

is a centred Gaussian process over the weight ensemble. RMS-2010 show its peak voltage
``V_max = max_{t in [0,T]} V(t)`` is approximately the maximum of ``K = T / sqrt(tau_s tau_m)``
effectively-independent Gaussian samples (the PSP correlation time is sqrt(tau_s tau_m)). By
extreme-value theory (Leadbetter, Lindgren & Rootzen 1983 ch.4; Fisher-Tippett-Gnedenko),
the max of N iid standard Gaussians converges to a Gumbel distribution with

    location  mu_N = sqrt(2 ln N) - (ln ln N + ln(4 pi)) / (2 sqrt(2 ln N))
    scale     beta_N = 1 / sqrt(2 ln N).

For the *normalised* tempotron voltage ``V_norm(t) = V(t) / sigma_V(t)`` (sigma the
per-trial std of V), the prediction is therefore

    V_max_norm  approximately  Gumbel(mu_K, beta_K)   with K = T / sqrt(tau_s tau_m).

Hypothesis
----------
At K in {16, 64, 256, 1024} we will observe ``V_max_norm`` distributions that are well-fit
by Gumbel and whose location/scale ratio approximates ``mu_K / beta_K = 2 ln K`` (independent
of normalisation -- the cleanest test). Finite-N corrections per Leadbetter et al. should
shrink with K.

Validation gates (pre-committed)
--------------------------------
  G-S1a (Gumbel shape): KS test p-value of fitted Gumbel >= 0.01 at each K.
  G-S1b (location/scale ratio): empirical ``loc/scale`` within +/- 15% of ``2 ln K``.
  G-S1c (monotonic location): empirical Gumbel location is monotonically increasing in K
        and tracks ``sqrt(2 ln K)`` (normalised) within 10%.

Outputs (to ``$LAB_RUN_DIR``)
-----------------------------
- ``results.csv``: one row per K with the Gumbel fit + KS test + theory comparison.
- ``vmax_samples_K{K}.npy``: raw V_max samples per K (for the analysis figure).
- ``metrics.jsonl``: per-K incremental metric series.

Run standalone (smoke):
    LAB_RUN_DIR=/tmp/s1_smoke LAB_SEED=0 \\
        uv run --with torch --with scipy python studies/sanity/s1_vmax_gumbel_evt.py \\
        K_list=16,64 n_trials=200
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import gumbel_r, kstest

TAU_M = 15.0  # ms, membrane time constant (Gutig & Sompolinsky 2006 + RMS 2010)
TAU_S = 3.75  # ms, synaptic time constant


def psp_kernel(t: torch.Tensor, tau_m: float = TAU_M, tau_s: float = TAU_S) -> torch.Tensor:
    """Double-exponential PSP normalised to unit peak. ``t`` may be any shape."""
    t_peak = tau_m * tau_s / (tau_m - tau_s) * math.log(tau_m / tau_s)
    v0 = 1.0 / (math.exp(-t_peak / tau_m) - math.exp(-t_peak / tau_s))
    t_clamped = torch.clamp(t, min=0.0)
    raw = v0 * (torch.exp(-t_clamped / tau_m) - torch.exp(-t_clamped / tau_s))
    return torch.where(t >= 0.0, raw, torch.zeros_like(raw))


def vmax_normalised_batch(
    n_aff: int,
    k_eff: int,
    n_trials: int,
    device: torch.device,
    gen: torch.Generator,
    dt: float | None = None,
) -> torch.Tensor:
    """For each of ``n_trials`` independent (Poisson pattern, Gaussian weights) realisations,
    compute ``V_max / sigma_V`` -- the per-trial normalised peak voltage.

    Time window ``T = K * sqrt(tau_m tau_s)``. Per-afferent spike count is Poisson(1); each
    spike time is uniform on ``[0, T]`` (RMS-2010 input model).

    ``sigma_V`` is the empirical std of ``V(t)`` across the time grid for *that* trial -- a
    consistent per-trial normalisation independent of the random pattern's exact realisation.
    """
    rho = math.sqrt(TAU_M * TAU_S)
    t_window = float(k_eff) * rho
    if dt is None:
        dt = TAU_S / 8.0  # ~ 0.47 ms, well below the ~1.7 ms PSP rise time
    n_grid = int(math.ceil(t_window / dt)) + 1
    t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * float(dt)  # (T_grid,)

    # Spike counts per (trial, afferent) -- Poisson(1). Then spike times uniform on [0, T].
    rate_ones = torch.ones(n_trials, n_aff, device=device, dtype=torch.float32)
    counts = torch.poisson(rate_ones, generator=gen).to(torch.int64)  # (B, N)
    max_spikes_int = int(counts.max().item())
    if max_spikes_int == 0:
        return torch.zeros(n_trials, device=device, dtype=torch.float32)
    # spike_times[b, i, f] uniform on [0, T], masked by f < counts[b, i].
    spike_times = t_window * torch.rand(
        (n_trials, n_aff, max_spikes_int), generator=gen, device=device, dtype=torch.float32
    )
    spike_idx = torch.arange(max_spikes_int, device=device).view(1, 1, max_spikes_int)
    valid = (spike_idx < counts.unsqueeze(-1)).to(torch.float32)  # 1.0 where active, 0 otherwise
    # Gaussian weights w_b in R^N, unit variance.
    w = torch.randn(n_trials, n_aff, generator=gen, device=device, dtype=torch.float32)

    # Loop over grid points, accumulating V_max and second-moment running sum (for sigma).
    # Memory at each step: (B, N, max_spikes) for the kernel contributions.
    v_max = torch.full((n_trials,), float("-inf"), device=device, dtype=torch.float32)
    v_sum = torch.zeros(n_trials, device=device, dtype=torch.float32)  # for mean V
    v2_sum = torch.zeros(n_trials, device=device, dtype=torch.float32)  # for E[V^2]
    for t_val in t_grid:
        # Kernel contributions per spike: K(t_val - spike_times) * valid_mask.
        contrib = psp_kernel(t_val - spike_times) * valid  # (B, N, max_spikes)
        # Per-afferent kernel sum (over that afferent's spikes).
        per_aff = contrib.sum(dim=2)  # (B, N)
        # Voltage at t_val for each trial.
        v_t = (w * per_aff).sum(dim=1)  # (B,)
        v_max = torch.maximum(v_max, v_t)
        v_sum = v_sum + v_t
        v2_sum = v2_sum + v_t * v_t

    # Per-trial std of V across time-grid points.
    n_grid_f = float(n_grid)
    mean_v = v_sum / n_grid_f
    var_v = (v2_sum / n_grid_f) - mean_v * mean_v
    sigma_v = torch.sqrt(torch.clamp(var_v, min=1e-12))
    return (v_max - mean_v) / sigma_v


def _theory_mu(k_eff: float) -> float:
    """Leadbetter-Lindgren-Rootzen leading-order Gumbel location for the max of K iid N(0,1)."""
    if k_eff < 2:
        return float("nan")
    a = math.sqrt(2.0 * math.log(k_eff))
    return a - (math.log(math.log(k_eff)) + math.log(4.0 * math.pi)) / (2.0 * a)


def _theory_beta(k_eff: float) -> float:
    """Gumbel scale for the max of K iid N(0,1)."""
    if k_eff < 2:
        return float("nan")
    return 1.0 / math.sqrt(2.0 * math.log(k_eff))


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))

    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    smoke = os.environ.get("LAB_RUN_DIR", "runs/local-dev") in ("", "runs/local-dev")
    k_list = [int(x) for x in ov.get("K_list", "16,64,256,1024" if not smoke else "16,64").split(",")]
    n_aff = int(ov.get("N_aff", "500"))
    n_trials = int(ov.get("n_trials", "200" if smoke else "5000"))
    chunk_size = int(ov.get("chunk_size", "1000"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(False)  # speed > determinism for sanity sample
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"S1 V_max Gumbel | seed={master_seed} device={device} K={k_list} N_aff={n_aff} "
        f"n_trials={n_trials} smoke={smoke}",
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

    rows: list[dict[str, float | int]] = []
    started = time.time()
    for k_eff in k_list:
        t0 = time.time()
        gen_seed = (master_seed * 31 + k_eff) & 0xFFFFFFFF
        gen = torch.Generator(device=device).manual_seed(gen_seed)
        samples_chunks: list[np.ndarray] = []
        for chunk_start in range(0, n_trials, chunk_size):
            n_chunk = min(chunk_size, n_trials - chunk_start)
            v_norm = vmax_normalised_batch(n_aff, k_eff, n_chunk, device, gen)
            samples_chunks.append(v_norm.detach().to("cpu", dtype=torch.float64).numpy())
        samples = np.concatenate(samples_chunks).astype(np.float64)
        elapsed = time.time() - t0
        # Save raw samples for the histogram figure.
        np.save(run_dir / f"vmax_samples_K{k_eff}.npy", samples)

        # Fit Gumbel and check shape (KS test against the fit).
        loc, scale = gumbel_r.fit(samples)
        loc_over_scale_emp = float(loc / scale) if scale > 0 else float("nan")
        ks_stat, ks_p = kstest(samples, lambda x: gumbel_r.cdf(x, loc=loc, scale=scale))

        # Theory: location/scale = 2 ln K (sigma-independent identity for Gumbel-of-Gauss-max).
        theory_loc_over_scale = 2.0 * math.log(float(k_eff))
        theory_mu = _theory_mu(float(k_eff))
        theory_beta = _theory_beta(float(k_eff))

        loc_rel_err = (loc - theory_mu) / theory_mu if theory_mu else float("nan")
        ratio_rel_err = (loc_over_scale_emp - theory_loc_over_scale) / theory_loc_over_scale

        row = {
            "K": k_eff,
            "n_trials": n_trials,
            "n_aff": n_aff,
            "elapsed_seconds": elapsed,
            "vmax_norm_mean": float(samples.mean()),
            "vmax_norm_std": float(samples.std()),
            "gumbel_loc_emp": float(loc),
            "gumbel_scale_emp": float(scale),
            "gumbel_loc_over_scale_emp": loc_over_scale_emp,
            "theory_loc_over_scale": theory_loc_over_scale,
            "theory_mu": theory_mu,
            "theory_beta": theory_beta,
            "loc_rel_err": float(loc_rel_err),
            "ratio_rel_err": float(ratio_rel_err),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
        }
        rows.append(row)

        emit(f"loc_K{k_eff}", loc, k_eff)
        emit(f"scale_K{k_eff}", scale, k_eff)
        emit(f"ratio_K{k_eff}", loc_over_scale_emp, k_eff)
        emit(f"ks_p_K{k_eff}", ks_p, k_eff)

        print(
            f"K={k_eff:>5}: V_max_norm ~ Gumbel(loc={loc:.3f}, scale={scale:.3f})  "
            f"loc/scale={loc_over_scale_emp:.2f} (theory={theory_loc_over_scale:.2f}, "
            f"err={100 * ratio_rel_err:+.1f}%)  loc vs theory_mu={theory_mu:.3f} "
            f"({100 * loc_rel_err:+.1f}%)  KS p={ks_p:.3g}  t={elapsed:.1f}s",
            flush=True,
        )

    fieldnames = list(rows[0].keys())
    with (run_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "experiment": "S1 -- V_max Gumbel EVT",
                "reference": "Rubin, Monasson & Sompolinsky 2010 PRL 105, 218102, Fig 1b",
                "params": {
                    "K_list": k_list,
                    "n_aff": n_aff,
                    "n_trials": n_trials,
                    "tau_m": TAU_M,
                    "tau_s": TAU_S,
                    "master_seed": master_seed,
                },
                "elapsed_seconds": time.time() - started,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"done in {time.time() - started:.1f}s -> {run_dir}/results.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
