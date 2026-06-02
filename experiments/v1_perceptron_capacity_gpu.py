"""V1 (GPU re-do): perceptron storage capacity vs Cover 1965 / Gardner 1988.

Self-contained Experiment-Contract entrypoint (numpy + torch only) -- no ``lab`` or ``tempotron``
imports, so the lab can ship and run this file unchanged. Replaces the CPU LP version
(``v1_perceptron_capacity.py``) with a **batched primal max-margin SVM on GPU** so the entire
sweep finishes in minutes on one RTX4090 and we get the Gardner ``kappa(alpha)`` curve as a
second theory contact essentially for free.

What it measures
----------------
For each ``(N, alpha)`` cell, draw ``n_reps`` independent instances:

* Patterns ``xi_{i,mu} ~ N(0, I_N)`` -- Gaussian, so Cover's exact formula
  ``P_sep(P, N) = 2^{1-P} sum_{k=0}^{N-1} C(P-1, k)`` is *exact* at finite N (general position).
* Labels ``y_{i,mu} in {-1, +1}`` iid uniform.
* Test homogeneous separability by minimising the squared-hinge primal

      L_i(w_i) = (1/P) sum_mu max(0, 1 - y_{i,mu} (w_i . xi_{i,mu}))^2  +  lambda ||w_i||^2

  with Adam for ``T_max`` steps, batched across all ``n_reps`` instances of the cell. After
  training we record (a) ``is_separable = (min_mu y_mu(w.xi_mu) > 0)`` and (b) the achieved
  *normalised* margin ``kappa_i = min_mu y_{i,mu}(w_i.xi_{i,mu}) / ||w_i||``. Aggregating across
  the cell gives empirical ``p_sep(N, alpha)`` (vs Cover) AND ``<kappa>(alpha)`` (vs Gardner).

Outputs (under ``$LAB_RUN_DIR``)
--------------------------------
- ``results.csv`` : N, alpha, P, n_trials, n_sep, p_sep, p_sep_se,
                    mean_margin, median_margin, p10_margin, p90_margin.
- ``results.json``: params + per-cell summary + elapsed seconds.
- ``metrics.jsonl``: incremental series, ``psep_N{N}`` and ``kappa_N{N}`` keyed by
                    ``step = round(1000 * alpha)``; plus a ``progress`` series and a
                    one-shot ``device_name`` / ``cuda_available`` info row.

Run standalone (smoke):
    LAB_RUN_DIR=/tmp/v1gpu LAB_SEED=0 \\
        uv run --with torch python studies/v1_perceptron_capacity_gpu.py \\
        N_list=10,30 alpha_coarse=0.3 alpha_fine=0.1 n_reps=20 T_max=1000
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------------------------------
# Core GPU primitive
# ----------------------------------------------------------------------------------------------


def gpu_separable_batch(
    patterns: torch.Tensor,
    labels: torch.Tensor,
    *,
    t_max: int = 5000,
    lr: float = 0.05,
    t_soft_start: float = 0.1,
    t_soft_end: float = 5e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched homogeneous-perceptron separability + Gardner max-margin via sphere-normalised
    annealed smoothed-min.

    For each instance ``b``, parameterises a free vector ``u_b`` (Euclidean, Hebb-initialised
    to ``sum_mu y_{b,mu} x_{b,mu}``), and optimises the **normalised** margin

        m_{b,mu}(u_b) = y_{b,mu} (u_b . x_{b,mu}) / ||u_b||

    The objective is the annealed soft-min (LogSumExp lower-bound on min):

        L_b(u_b) = T * log sum_mu exp( -m_{b,mu}(u_b) / T )   ~   -min_mu m_{b,mu}(u_b)

    with ``T`` decayed geometrically from ``t_soft_start`` to ``t_soft_end`` over ``t_max`` Adam
    steps. The loss is scale-invariant in ``u_b`` (every term depends only on the direction), so
    the optimisation converges to the max-margin *direction* on the unit sphere -- exactly the
    quantity Gardner (1988) computes analytically as ``kappa*(alpha)``.

    Returns ``(is_separable, kappa)`` with shapes ``(B,)``:

    * ``is_separable[b] = (min_mu m_{b,mu} > 0)``: a strictly positive normalised margin was
      attained, i.e. the instance is homogeneously linearly separable.
    * ``kappa[b] = min_mu m_{b,mu}``: the achieved (sphere-normalised) margin -- finite-budget
      estimate of Gardner's ``kappa*(alpha)`` for separable cases, and a (negative) "infeasibility
      score" otherwise.

    Parameters
    ----------
    patterns:
        Tensor of shape ``(B, P, N)``; row ``mu`` of batch ``b`` is one pattern.
    labels:
        Tensor of shape ``(B, P)`` with values in ``{-1, +1}`` (float dtype).
    t_max, lr, t_soft_start, t_soft_end:
        Adam steps, learning rate, and the annealing endpoints for the smoothed-min temperature.

    Notes
    -----
    Allocates ``y_x = labels.unsqueeze(-1) * patterns`` once. No gradients leave this function;
    the returned tensors are detached.
    """
    if patterns.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"expected patterns(B,P,N), labels(B,P); got {patterns.shape}, {labels.shape}"
        )
    b, p, n = patterns.shape
    if labels.shape != (b, p):
        raise ValueError(f"labels shape {labels.shape} incompatible with patterns {patterns.shape}")

    device = patterns.device
    y_x = labels.unsqueeze(-1) * patterns  # (B, P, N)
    # Hebb init: u0 = sum_mu y_mu x_mu. Already pointing into the separating cone for typical
    # separable instances, so the optimiser refines a near-solution rather than discovering one.
    u = y_x.sum(dim=1).clone().detach().requires_grad_(True)  # (B, N)
    optim = torch.optim.Adam([u], lr=lr)
    log_ratio = float(np.log(t_soft_end / t_soft_start))

    for step in range(t_max):
        optim.zero_grad(set_to_none=True)
        u_norm = u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        margins = torch.einsum("bpn,bn->bp", y_x, u / u_norm)  # (B, P), sphere-normalised
        # Geometric anneal of the soft-min temperature: tight estimate of min by run-end.
        t = float(t_soft_start * np.exp(log_ratio * step / max(1, t_max - 1)))
        # L_b = T * logsumexp(-m_b / T)  ~  -min_mu m_{b,mu}.   Sum over B (independent).
        loss = t * (-margins / t).logsumexp(dim=1).sum()
        loss.backward()
        optim.step()

    with torch.no_grad():
        u_norm = u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        margins = torch.einsum("bpn,bn->bp", y_x, u / u_norm)
        kappa = margins.min(dim=1).values  # already sphere-normalised
        is_sep = kappa > 0.0
    return is_sep.detach(), kappa.detach()


# ----------------------------------------------------------------------------------------------
# Sweep driver (Experiment-Contract entrypoint)
# ----------------------------------------------------------------------------------------------


def build_alpha_grid(
    a_min: float, a_max: float, coarse: float, fine: float, fine_lo: float, fine_hi: float
) -> np.ndarray:
    """Coarse global grid unioned with a fine grid across the transition window."""
    coarse_g = np.arange(a_min, a_max + 1e-9, coarse)
    fine_g = np.arange(fine_lo, fine_hi + 1e-9, fine)
    return np.unique(np.round(np.concatenate([coarse_g, fine_g]), 4)).astype(np.float64)


def _overrides(argv: list[str]) -> dict[str, str]:
    return dict(tok.split("=", 1) for tok in argv if "=" in tok)


def _derive_seed(*ints: int) -> int:
    """Deterministic 64-bit seed mix from integer entropy (independent of Python hash)."""
    ss = np.random.SeedSequence(list(ints))
    arr = ss.generate_state(2, dtype=np.uint32)
    return int(arr[0]) << 32 | int(arr[1])


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))

    ov = _overrides(sys.argv[1:])
    n_list = [int(x) for x in ov.get("N_list", "25,50,100,200,500,1000,2000").split(",")]
    a_min = float(ov.get("alpha_min", "1.0"))
    a_max = float(ov.get("alpha_max", "3.0"))
    coarse = float(ov.get("alpha_coarse", "0.1"))
    fine = float(ov.get("alpha_fine", "0.02"))
    fine_lo = float(ov.get("alpha_fine_lo", "1.7"))
    fine_hi = float(ov.get("alpha_fine_hi", "2.3"))
    n_reps = int(ov.get("n_reps", "100"))
    t_max = int(ov.get("T_max", "5000"))
    lr = float(ov.get("lr", "0.05"))
    t_soft_start = float(ov.get("t_soft_start", "0.1"))
    t_soft_end = float(ov.get("t_soft_end", "5e-3"))
    dtype_str = ov.get("dtype", "float32")
    force_cpu = ov.get("force_cpu", "0") == "1"

    alphas = build_alpha_grid(a_min, a_max, coarse, fine, fine_lo, fine_hi)
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_str]
    device = torch.device("cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu")

    params = {
        "master_seed": master_seed, "N_list": n_list, "n_reps": n_reps,
        "alpha_min": a_min, "alpha_max": a_max, "alpha_coarse": coarse, "alpha_fine": fine,
        "alpha_fine_lo": fine_lo, "alpha_fine_hi": fine_hi, "n_alphas": int(alphas.size),
        "T_max": t_max, "lr": lr, "t_soft_start": t_soft_start, "t_soft_end": t_soft_end,
        "dtype": dtype_str,
        "ensemble": "gaussian", "algorithm": "sphere_smoothed_min_annealed_adam",
        "device": str(device), "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }

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

    # One-shot device info as metrics rows -- visible via `lab metrics` immediately on start.
    emit("cuda_available", 1.0 if torch.cuda.is_available() else 0.0, 0)
    if torch.cuda.is_available():
        emit("gpu_total_mem_gb", torch.cuda.get_device_properties(0).total_memory / 1e9, 0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({device})", flush=True)
    else:
        print(f"GPU not available; running on {device}", flush=True)

    print(
        f"V1 GPU sweep | seed={master_seed} device={device} dtype={dtype_str} "
        f"N={n_list} alphas={alphas.size} ({alphas.min():.2f}..{alphas.max():.2f}) "
        f"n_reps={n_reps} T_max={t_max} cells={len(n_list) * alphas.size}",
        flush=True,
    )

    tasks = [(int(n), float(a)) for n in n_list for a in alphas]
    total = len(tasks)
    rows: list[dict[str, float | int]] = []
    started = time.time()

    for done, (n, alpha) in enumerate(tasks, 1):
        p = round(alpha * n)
        # Deterministic per-cell seed regardless of execution order.
        seed_int = _derive_seed(master_seed, n, int(round(1000 * alpha)))
        gen = torch.Generator(device=device).manual_seed(seed_int)
        if p == 0:
            n_sep = n_reps
            margins = np.zeros(n_reps, dtype=np.float64)
        else:
            patterns = torch.randn(n_reps, p, n, generator=gen, device=device, dtype=dtype)
            labels = (
                2.0 * torch.randint(0, 2, (n_reps, p), generator=gen, device=device, dtype=dtype)
                - 1.0
            )
            is_sep, kappa = gpu_separable_batch(
                patterns, labels, t_max=t_max, lr=lr,
                t_soft_start=t_soft_start, t_soft_end=t_soft_end,
            )
            n_sep = int(is_sep.sum().item())
            margins = kappa.detach().to("cpu", dtype=torch.float64).numpy()
            # Free GPU memory before next cell (big-N cells are the dominant footprint).
            del patterns, labels, is_sep, kappa
            if device.type == "cuda":
                torch.cuda.empty_cache()

        p_sep = n_sep / n_reps
        se = float(np.sqrt(max(p_sep * (1.0 - p_sep), 0.0) / n_reps))
        rows.append(
            {
                "N": n, "alpha": alpha, "P": p, "n_trials": n_reps, "n_sep": n_sep,
                "p_sep": p_sep, "p_sep_se": se,
                "mean_margin": float(np.mean(margins)),
                "median_margin": float(np.median(margins)),
                "p10_margin": float(np.percentile(margins, 10)),
                "p90_margin": float(np.percentile(margins, 90)),
            }
        )

        step = int(round(1000 * alpha))
        emit(f"psep_N{n}", p_sep, step)
        emit(f"kappa_N{n}", float(np.mean(margins)), step)
        emit("progress", done / total, done)
        print(
            f"[{done}/{total}] N={n:>5} alpha={alpha:5.3f} P={p:>5} "
            f"p_sep={p_sep:.3f}+-{se:.3f} <kappa>={np.mean(margins):+.4f}",
            flush=True,
        )

    rows.sort(key=lambda r: (r["N"], r["alpha"]))
    fieldnames = [
        "N", "alpha", "P", "n_trials", "n_sep", "p_sep", "p_sep_se",
        "mean_margin", "median_margin", "p10_margin", "p90_margin",
    ]
    with (run_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "results.json").write_text(
        json.dumps(
            {"params": params, "elapsed_seconds": time.time() - started, "rows": rows},
            indent=2,
        )
    )
    print(
        f"done: {len(rows)} cells in {time.time() - started:.1f}s -> {run_dir}/results.csv",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
