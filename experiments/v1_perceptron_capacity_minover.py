"""V1 (re-do): perceptron storage capacity via batched Krauth-Mezard min-over.

Self-contained Experiment-Contract entrypoint (numpy + torch only). No ``lab`` or ``tempotron``
imports -- the lab ships this file unmodified to the remote backend (see ``docs/v1-redo-design.md``
for the locked spec).

Algorithm: batched primal Krauth-Mezard min-over [Krauth & Mezard 1987, J Phys A 20:L745]
with a Hebb-warm-start and a one-sided Novikoff prefilter [Novikoff 1962]. Per step on each
of ``B`` independent instances::

    margins[b, mu] = y_{b,mu} (w_b . xi_{b,mu})            # batched matmul
    mu* = argmin_mu margins[b, mu]                          # worst-stability pattern
    w_b <- w_b + y_{b,mu*} xi_{b,mu*}                       # rank-1 Hebbian update

Per-cell ``T_max`` is theory-grounded (design doc §2.3 / §A.5)::

    T_max(N, alpha) = ceil(10 * R^2(N, P) / kappa_G(alpha)^2),  capped at 5e6
    R^2(N, P)      = N * (1 + 2 * sqrt(log(P) / N))

For ``alpha >= 2`` (above Cover capacity, ``kappa_G`` undefined / zero) we use
``T_max = max(N, 1e4)`` -- non-separable instances cycle quickly to a stable negative margin.

Outputs to ``$LAB_RUN_DIR`` (Experiment Contract §7):

* ``results.csv``  -- one row per cell with the LOCKED schema (design doc §4.2).
* ``results.json`` -- params + resolved env + rows + elapsed_seconds.
* ``metrics.jsonl``-- per-cell ``psep_N{N}`` and ``kappa_N{N}`` series, ``progress``, and a
                      one-shot device-info row at start (E.9 GPU determinism metadata).

Smoke mode: if ``LAB_RUN_DIR`` is unset OR set to ``runs/local-dev``, defaults shrink to
``N_list=20,40 alpha_max=2.6 n_reps=20 T_max_cap=5e4`` for quick local validation.

Run standalone (smoke)::

    LAB_RUN_DIR=/tmp/v1mo uv run --with torch python \\
        experiments/tempotron_capacity/studies/v1_perceptron_capacity_minover.py \\
        force_cpu=1
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp
from scipy.stats import norm

ALGO_VERSION = "krauth_mezard_minover_v1.0"

# ----------------------------------------------------------------------------------------------
# Analytic references (self-contained mirrors of tempotron.capacity; the lab ships only
# this single file so we cannot rely on the experiment's src/ package here).
# ----------------------------------------------------------------------------------------------


def gardner_alpha(kappa: float) -> float:
    """Gardner (1988) critical load at margin ``kappa`` for the homogeneous perceptron."""
    if not math.isfinite(kappa):
        raise ValueError(f"kappa must be finite, got {kappa}")
    integral, _ = quad(
        lambda t: (t + kappa) ** 2 * norm.pdf(t),
        -kappa,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-10,
    )
    return float(1.0 / integral)


def gardner_kappa(alpha: float) -> float:
    """Inverse of :func:`gardner_alpha`; ``0`` for ``alpha >= 2``, ``+inf`` for ``alpha <= 0``."""
    if alpha >= 2.0:
        return 0.0
    if alpha <= 0.0:
        return float("inf")
    return float(brentq(lambda k: gardner_alpha(k) - alpha, 0.0, 20.0, xtol=1e-10))


def cover_separable_probability(n: int, alpha: float) -> float:
    """Cover's exact ``P_sep(P, N) = 2^{1-P} sum_{k=0}^{N-1} C(P-1, k)`` (log-space)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    p = round(alpha * n)
    if p == 0:
        return 1.0
    k_max = min(n - 1, p - 1)
    if k_max < 0:
        return 0.0
    k = np.arange(0, k_max + 1, dtype=np.float64)
    log_binom = gammaln(p) - gammaln(k + 1.0) - gammaln(p - k)
    log_sum = float(logsumexp(log_binom))
    log_prob = (1.0 - p) * np.log(2.0) + log_sum
    return float(np.clip(np.exp(log_prob), 0.0, 1.0))


# ----------------------------------------------------------------------------------------------
# Per-cell T_max budget (design doc §2.3 / §A.5).
# ----------------------------------------------------------------------------------------------


def compute_t_max(n: int, alpha: float, *, cap: int = 5_000_000) -> int:
    """Per-cell min-over iteration budget, theory-grounded with hard cap.

    For ``alpha < 2`` (separable regime, ``kappa_G > 0``)::

        T_max = ceil(10 * R^2(N, P) / kappa_G(alpha)^2)
        R^2(N, P) = N * (1 + 2 * sqrt(log(P) / N))

    For ``alpha`` inside the **transition window** ``[2, 2 + 3/sqrt(N)]`` we use the
    cap. Cover's de Moivre-Laplace asymptotic ``P_sep ~ Phi((2-alpha)*sqrt(N/2))`` is
    still non-negligible there (a non-trivial fraction of instances *are* separable),
    so they need the same iteration budget as the alpha < 2 cells. The earlier
    ``alpha >= 2 -> T_max = max(N, 1e4)`` shortcut was too aggressive: the sub-experiment
    showed it pile-up 6-14sigma false negatives at the alpha=2.0 cells.

    Only **far** past the transition (``alpha > 2 + 3/sqrt(N)``, i.e. Cover P_sep < 0.001)
    do we use ``T_max = max(N, 1e4)`` -- those instances cycle quickly to a stable
    negative margin and don't need the heavy budget.
    """
    p = round(alpha * n)
    if p == 0:
        return 0
    transition_upper = 2.0 + 3.0 / math.sqrt(n)
    if alpha > transition_upper:
        return int(max(n, 10_000))
    kappa = gardner_kappa(alpha) if alpha < 2.0 else 0.0
    if kappa <= 0.0:
        # In the transition window with alpha >= 2.0: gardner_kappa returns 0 / undefined,
        # but Cover P_sep is still non-trivial -- use the cap so the algorithm has room
        # to find positive margins in the still-separable fraction of instances.
        return cap
    r2 = n * (1.0 + 2.0 * math.sqrt(math.log(p) / n))
    t = math.ceil(10.0 * r2 / (kappa * kappa))
    return int(min(cap, max(64, t)))


# ----------------------------------------------------------------------------------------------
# Core kernels.
# ----------------------------------------------------------------------------------------------


def novikoff_prefilter_separable(yx: torch.Tensor, t_pre: int) -> torch.Tensor:
    """One-sided Novikoff perceptron prefilter for cheap separability detection.

    Repeats ``w_b <- w_b + (y xi)_{b, mu*}`` on the worst-violator pattern per instance for at
    most ``t_pre`` updates, then declares each instance ``True`` only if every margin is
    strictly positive at exit (verified separable). The returned tensor uses the convention

        result[b] = True   iff the prefilter VERIFIED instance b as separable (early-exit OK)
        result[b] = False  iff the verdict is UNDETERMINED -- caller MUST run full min-over.

    **Invariant (design doc §E.10): this function never returns "non-separable".** A timeout
    or non-positive minimum margin at the prefilter step bound is *undetermined*, not a
    declaration of infeasibility -- the heavy min-over solve is required for every instance
    for which ``result[b] is False``. Otherwise an instance with a long Novikoff mistake bound
    (large ``R/gamma``, i.e. nearly-parallel patterns near alpha_c) would be silently
    misclassified and bias ``p_sep_hat`` downward.

    Parameters
    ----------
    yx:
        ``(B, P, N)`` tensor of ``y_{b,mu} * xi_{b,mu}`` (caller pre-multiplies labels). All
        margins reduce to ``(yx . w)``; we do not need patterns and labels separately.
    t_pre:
        Maximum Novikoff updates per instance. ``0`` disables the prefilter.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``(B,)``. ``True`` means *verified separable* (downstream may
        skip the full solve); ``False`` means *undetermined* (downstream MUST run min-over).

    Raises
    ------
    ValueError
        If ``yx`` is not 3-D.

    References
    ----------
    Novikoff, A. B. J. (1962). On convergence proofs on perceptrons. Symp Math Theory
    Automata XII:615 -- mistake-bound theorem ``(R/gamma)^2``.
    Krauth, W. & Mezard, M. (1987). J Phys A 20:L745 -- the min-over generalisation we
    delegate to when the prefilter is undetermined.
    """
    if yx.ndim != 3:
        raise ValueError(f"expected yx shape (B, P, N); got {yx.shape}")
    b, p, n = yx.shape
    device = yx.device
    dtype = yx.dtype

    if t_pre <= 0 or p == 0:
        return torch.zeros(b, dtype=torch.bool, device=device)

    w = torch.zeros(b, n, device=device, dtype=dtype)
    done = torch.zeros(b, dtype=torch.bool, device=device)

    for _ in range(t_pre):
        margins = torch.einsum("bpn,bn->bp", yx, w)  # (B, P)
        min_vals, min_idx = margins.min(dim=1)  # (B,)
        # An instance is verified separable as soon as all its margins are strictly positive.
        done = done | (min_vals > 0.0)
        if bool(done.all().item()):
            break
        active = ~done
        if not bool(active.any().item()):
            break
        # Rank-1 update on the violator: w_b += yx[b, mu*, :] for active b.
        idx_expand = min_idx.view(b, 1, 1).expand(b, 1, n)
        update = yx.gather(1, idx_expand).squeeze(1)  # (B, N)
        w = torch.where(active.unsqueeze(-1), w + update, w)

    with torch.no_grad():
        margins = torch.einsum("bpn,bn->bp", yx, w)
        min_vals = margins.min(dim=1).values
        verified = (min_vals > 0.0) & done
    return verified.detach()


def minover_batch(
    patterns: torch.Tensor,
    labels: torch.Tensor,
    *,
    t_max: int,
    record_every: int,
    eps_abs: float = 1e-3,
    eps_window_mult: int = 10,
    t_pre: int = 0,
) -> dict[str, torch.Tensor]:
    """Batched primal Krauth-Mezard min-over for the hard-margin homogeneous perceptron.

    Implements the algorithm of Krauth & Mezard (1987), eq. (7)-(16), batched over ``B``
    independent ``(N, P)`` instances. With Hebb warm-start ``w <- sum_mu y_mu xi_mu`` the
    iteration starts inside the separating cone for typical instances, so no symmetry-
    breaking noise is needed. The per-step update is purely worst-violator Hebbian, which
    Krauth & Mezard prove converges to Gardner's ``kappa*`` for any separable instance.

    Per-step diagnostics are recorded every ``record_every`` iterations (cheap matvec
    + argmin reduction). Per-instance two-criterion early-stop fires when

        (a) the running best ``kappa_hat`` is strictly positive AND has not improved by more
            than ``eps_abs`` over the last ``eps_window_mult * record_every`` recording steps,
        OR
        (b) the step counter reaches ``t_max``.

    The two-criterion stop bounds compute (criterion a fires fast on clearly-separable
    instances) yet still spends the full budget on borderline cells where ``kappa_hat``
    keeps improving.

    Parameters
    ----------
    patterns:
        ``(B, P, N)`` Gaussian pattern tensor.
    labels:
        ``(B, P)`` label tensor in ``{-1, +1}`` (same dtype as ``patterns``).
    t_max:
        Maximum min-over iterations per cell (theory-grounded; see :func:`compute_t_max`).
    record_every:
        Diagnostic recording stride; the eps-improvement criterion is evaluated every
        ``record_every`` steps. Use ``max(10, N // 5)`` per design doc.
    eps_abs:
        Absolute kappa-improvement threshold over the trailing window (default 1e-3).
    eps_window_mult:
        Number of ``record_every`` strides composing the trailing window (default 10, so the
        actual window length in steps is ``10 * record_every``).
    t_pre:
        Novikoff prefilter step budget. ``0`` disables the prefilter; otherwise instances the
        prefilter verifies as separable are still run through min-over for kappa polishing
        (G4 needs every separable instance polished toward Gardner's kappa*).

    Returns
    -------
    dict[str, torch.Tensor]
        Per-instance diagnostics, all of shape ``(B,)`` unless noted:

        * ``is_separable`` -- bool, ``min_mu y_mu (w . xi_mu) > 0`` at exit.
        * ``kappa`` -- float, achieved normalised min margin ``min_mu y_mu (w . xi_mu)/||w||``.
        * ``kappa_best`` -- float, running max kappa across recording steps (== kappa for
          well-converged instances; can exceed ``kappa`` when the final step regresses).
        * ``first_positive_margin_iter`` -- long, first step at which the all-positive margin
          condition held (``-1`` if never).
        * ``iter_count_to_eps_stop`` -- long, step at which criterion (a) fired; ``t_max`` if
          criterion (b) ended the loop.
        * ``hit_T_max`` -- bool, ``True`` iff criterion (b) ended the loop.
        * ``novikoff_early_exit`` -- bool, ``True`` iff the Novikoff prefilter verified the
          instance (metadata only; min-over still ran for kappa polishing).
        * ``mean_final_min_margin`` (shape ``(B,)``) -- same as ``kappa`` (kept for schema
          clarity per design doc §4.2).

    References
    ----------
    Krauth, W. & Mezard, M. (1987). Learning algorithms with optimal stability in neural
    networks. J Phys A 20:L745 -- eqs. (7)-(16), Theorem 1 (convergence to ``kappa*``).
    Novikoff, A. B. J. (1962). On convergence proofs on perceptrons. Symp Math Theory
    Automata XII:615 -- prefilter mistake bound.
    Engel, A. & Van den Broeck, C. (2001). Statistical Mechanics of Learning §3.5 -- min-over
    as the canonical primal solver for Gardner's optimal-margin perceptron.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    if patterns.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"expected patterns(B,P,N), labels(B,P); got {patterns.shape}, {labels.shape}"
        )
    b, p, n = patterns.shape
    if labels.shape != (b, p):
        raise ValueError(
            f"labels shape {labels.shape} incompatible with patterns {patterns.shape}"
        )
    if record_every <= 0:
        raise ValueError(f"record_every must be positive, got {record_every}")

    device = patterns.device
    dtype = patterns.dtype
    yx = labels.unsqueeze(-1) * patterns  # (B, P, N); single pre-multiply, reused throughout.

    # Novikoff prefilter (metadata only -- we still polish kappa for every separable instance).
    if t_pre > 0:
        novikoff_early = novikoff_prefilter_separable(yx, t_pre=t_pre)
    else:
        novikoff_early = torch.zeros(b, dtype=torch.bool, device=device)

    # Hebb warm-start: w_b = sum_mu y_{b,mu} xi_{b,mu}. Starts inside the separating cone for
    # typical instances (Krauth-Mezard 1987 footnote 4); no symmetry breaking needed.
    w = yx.sum(dim=1)  # (B, N)

    iter_to_first_positive = torch.full((b,), -1, dtype=torch.long, device=device)
    iter_to_eps_stop = torch.full((b,), t_max, dtype=torch.long, device=device)
    eps_stopped = torch.zeros(b, dtype=torch.bool, device=device)
    kappa_best = torch.full((b,), -float("inf"), dtype=dtype, device=device)
    # Sliding window of best-kappa snapshots, length eps_window_mult.
    window_len = max(1, eps_window_mult)
    window: list[torch.Tensor] = []

    if p == 0 or t_max <= 0:
        # Degenerate: no patterns or no budget. All instances trivially separable; kappa = +inf.
        kappa_final = torch.full((b,), float("inf"), dtype=dtype, device=device)
        return {
            "is_separable": torch.ones(b, dtype=torch.bool, device=device),
            "kappa": kappa_final,
            "kappa_best": kappa_final.clone(),
            "first_positive_margin_iter": iter_to_first_positive,
            "iter_count_to_eps_stop": iter_to_eps_stop,
            "hit_T_max": torch.zeros(b, dtype=torch.bool, device=device),
            "novikoff_early_exit": novikoff_early,
            "mean_final_min_margin": kappa_final.clone(),
        }

    # GPU-utilisation note: the original implementation did ``.item()`` syncs every
    # ``record_every`` steps (3 syncs per recording tick: stop_mask.any, eps_stopped.all,
    # active.any). On a B200 this collapsed throughput to ~0.0001% of peak FLOPS because each
    # sync blocks the kernel pipeline. The rewrite below runs ``chunk_size`` pure-GPU steps
    # with NO syncs, then a single per-chunk diagnostic + cell-level stop check. Per-instance
    # masking of "already stopped" instances was also dropped: with batched ops the kernel
    # cost is fixed in B, so freezing individual instances saves no compute, only adds syncs.
    chunk_size = max(record_every * 20, 500)
    step = 0
    cell_stopped = False
    while step < t_max and not cell_stopped:
        chunk_end = int(min(step + chunk_size, t_max))
        # Pure GPU inner block. No .item(), no host branches. Inlined einsum + argmin + gather
        # + add (the eager path -- already validated end-to-end on B200 at ~570 k step-inst/s).
        # torch.compile was attempted; mode='reduce-overhead' hit CUDA-graphs aliasing on the
        # iterative w<-step(w) pattern, mode='default' was untested on actual GPU and risked
        # tracing overhead per (B, P, N) shape -- kept out to ship a proven path.
        for _ in range(chunk_end - step):
            margins_loop = torch.einsum("bpn,bn->bp", yx, w)
            _, min_idx_loop = margins_loop.min(dim=1)
            idx_expand = min_idx_loop.view(b, 1, 1).expand(b, 1, n)
            w = w + yx.gather(1, idx_expand).squeeze(1)
        step = chunk_end

        # Chunk-end diagnostics + global eps-stop check (one sync per chunk, not per step).
        margins = torch.einsum("bpn,bn->bp", yx, w)
        min_vals, _ = margins.min(dim=1)
        w_norm = w.norm(dim=1).clamp_min(1e-30)
        kappa_now = min_vals / w_norm
        kappa_best = torch.maximum(kappa_best, kappa_now)
        # First-positive iteration -- track the *earliest* chunk-end at which kappa was > 0.
        # Best-effort precision (chunk-grained) -- the recorded value is the chunk-end step.
        newly_pos = (min_vals > 0.0) & (iter_to_first_positive < 0)
        iter_to_first_positive = torch.where(
            newly_pos,
            torch.full_like(iter_to_first_positive, step),
            iter_to_first_positive,
        )
        window.append(kappa_best.clone())
        if len(window) > window_len:
            window.pop(0)
        # Cell-level stop: ALL instances satisfy kappa_best > 0 AND plateaued. Single .item()
        # call per chunk -- amortised over ``chunk_size`` steps.
        if len(window) == window_len:
            improved = (kappa_best - window[0]).abs()
            per_instance_done = (kappa_best > 0.0) & (improved <= eps_abs)
            newly_done = per_instance_done & (~eps_stopped)
            iter_to_eps_stop = torch.where(
                newly_done,
                torch.full_like(iter_to_eps_stop, step),
                iter_to_eps_stop,
            )
            eps_stopped = eps_stopped | per_instance_done
            cell_stopped = bool(per_instance_done.all().item())

    # Final readout. Separability is determined by the RUNNING BEST normalised margin,
    # not the final one: under batched min-over the worst-pattern argmin shifts as w grows,
    # so the per-step kappa is not monotonic. A positive margin found mid-training proves
    # the instance is separable -- losing it because the final step's worst pattern dipped
    # back negative would be a false negative (the bug that killed the first run).
    with torch.no_grad():
        margins = torch.einsum("bpn,bn->bp", yx, w)
        min_vals = margins.min(dim=1).values
        w_norm = w.norm(dim=1).clamp_min(1e-30)
        kappa_final = min_vals / w_norm
        kappa_best = torch.maximum(kappa_best, kappa_final)
        is_sep = kappa_best > 0.0
        # Novikoff prefilter is provably one-sided (only declares True when it found a
        # positive margin) -- never override its True with a min-over False.
        is_sep = is_sep | novikoff_early
        hit_tmax = ~eps_stopped

    return {
        "is_separable": is_sep.detach(),
        "kappa": kappa_final.detach(),
        "kappa_best": kappa_best.detach(),
        "first_positive_margin_iter": iter_to_first_positive.detach(),
        "iter_count_to_eps_stop": iter_to_eps_stop.detach(),
        "hit_T_max": hit_tmax.detach(),
        "novikoff_early_exit": novikoff_early.detach(),
        "mean_final_min_margin": kappa_final.detach().clone(),
    }


# ----------------------------------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------------------------------


def build_alpha_grid(
    n_full: list[int],
    n_coarse_only: list[int],
    a_min: float,
    a_max: float,
    coarse: float,
    fine: float,
    fine_lo: float,
    fine_hi: float,
) -> dict[int, np.ndarray[Any, Any]]:
    """Return ``{N: alpha-grid}``: full = coarse + fine; coarse-only sets skip the fine grid."""
    coarse_g = np.arange(a_min, a_max + 1e-9, coarse)
    fine_g = np.arange(fine_lo, fine_hi + 1e-9, fine)
    full = np.unique(np.round(np.concatenate([coarse_g, fine_g]), 6)).astype(np.float64)
    coarse_only = np.unique(np.round(coarse_g, 6)).astype(np.float64)
    out: dict[int, np.ndarray[Any, Any]] = {}
    for n in n_full:
        out[n] = full
    for n in n_coarse_only:
        out[n] = coarse_only
    return out


def _overrides(argv: list[str]) -> dict[str, str]:
    """Hydra-style ``key=value`` argv parse."""
    return dict(tok.split("=", 1) for tok in argv if "=" in tok)


def _derive_seed(*ints: int) -> int:
    """Deterministic 64-bit seed mix from integer entropy (independent of Python hash)."""
    ss = np.random.SeedSequence(list(ints))
    arr = ss.generate_state(2, dtype=np.uint32)
    return int(arr[0]) << 32 | int(arr[1])


def _percentile(x: np.ndarray[Any, Any], q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _is_smoke_mode(run_dir_env: str | None) -> bool:
    """Smoke mode: ``LAB_RUN_DIR`` unset OR exactly ``runs/local-dev``."""
    return run_dir_env is None or run_dir_env == "runs/local-dev"


def main() -> int:
    run_dir_env = os.environ.get("LAB_RUN_DIR")
    smoke = _is_smoke_mode(run_dir_env)
    run_dir = Path(run_dir_env if run_dir_env is not None else "runs/local-dev")
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))

    ov = _overrides(sys.argv[1:])

    # ----- N lists ----------------------------------------------------------------
    if smoke:
        default_n_full = "20,40"
        default_n_coarse = ""
    else:
        default_n_full = "25,50,100,200,500,1000"
        default_n_coarse = "2000"
    if "N_list" in ov:
        n_full = [int(x) for x in ov["N_list"].split(",") if x]
        n_coarse_only: list[int] = [
            int(x) for x in ov.get("N_list_coarse_only", "").split(",") if x
        ]
    else:
        n_full = [int(x) for x in ov.get("N_list_full", default_n_full).split(",") if x]
        n_coarse_only = [
            int(x) for x in ov.get("N_list_coarse_only", default_n_coarse).split(",") if x
        ]

    # ----- alpha grids + sizing knobs ---------------------------------------------
    a_min = float(ov.get("alpha_min", "1.0"))
    a_max = float(ov.get("alpha_max", "2.6" if smoke else "3.0"))
    coarse = float(ov.get("alpha_coarse", "0.1"))
    fine = float(ov.get("alpha_fine", "0.02"))
    fine_lo = float(ov.get("alpha_fine_lo", "1.7"))
    fine_hi = float(ov.get("alpha_fine_hi", "2.05"))
    n_reps = int(ov.get("n_reps", "20" if smoke else "50"))
    t_max_cap = int(float(ov.get("T_max_cap", "50000" if smoke else "1000000")))
    eps_abs = float(ov.get("eps_abs", "1e-3"))
    t_pre_cap = int(ov.get("t_pre_cap", "5000"))
    t_pre_mul = int(ov.get("t_pre_mul", "10"))
    dtype_str = ov.get("dtype", "float32")
    force_cpu = ov.get("force_cpu", "0") == "1"
    record_every_override = ov.get("record_every_override")

    grids = build_alpha_grid(n_full, n_coarse_only, a_min, a_max, coarse, fine, fine_lo, fine_hi)
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_str]
    device = torch.device("cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu")

    # GPU determinism (E.9): CUBLAS_WORKSPACE_CONFIG required for deterministic cuBLAS GEMM
    # under torch.use_deterministic_algorithms.
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") not in (
        ":4096:8",
        ":16:8",
    ):
        print(
            "WARN: CUBLAS_WORKSPACE_CONFIG not set to ':4096:8'; setting in-process for "
            "deterministic GEMM.",
            flush=True,
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
        cudnn_det = True
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"WARN: could not enable deterministic algorithms: {exc}", flush=True)
        cudnn_det = False

    cuda_version = ""
    gpu_uuid = ""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cuda_version = str(getattr(torch.version, "cuda", "") or "")
        gpu_uuid = str(getattr(props, "uuid", "") or torch.cuda.get_device_name(0))

    params: dict[str, Any] = {
        "master_seed": master_seed,
        "smoke_mode": smoke,
        "N_list_full": n_full,
        "N_list_coarse_only": n_coarse_only,
        "n_reps": n_reps,
        "alpha_min": a_min,
        "alpha_max": a_max,
        "alpha_coarse": coarse,
        "alpha_fine": fine,
        "alpha_fine_lo": fine_lo,
        "alpha_fine_hi": fine_hi,
        "T_max_cap": t_max_cap,
        "eps_abs": eps_abs,
        "t_pre_cap": t_pre_cap,
        "t_pre_mul": t_pre_mul,
        "dtype": dtype_str,
        "record_every_override": record_every_override,
        "ensemble": "gaussian",
        "algorithm": ALGO_VERSION,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "gpu_uuid": gpu_uuid,
        "cudnn_deterministic": cudnn_det,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    metrics_path = run_dir / "metrics.jsonl"
    # Truncate the metrics file (in case of re-run into the same LAB_RUN_DIR).
    metrics_path.write_text("")

    def emit(name: str, value: float, step: int) -> None:
        with metrics_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "name": name,
                        "value": float(value),
                        "step": int(step),
                        "wall_time": time.time(),
                    }
                )
                + "\n"
            )

    def emit_info(name: str, value: str, step: int = 0) -> None:
        with metrics_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "name": name,
                        "info": value,
                        "step": int(step),
                        "wall_time": time.time(),
                    }
                )
                + "\n"
            )

    # One-shot device-info row at start (E.9 GPU determinism metadata).
    emit_info("device", str(device))
    emit_info("torch_version", torch.__version__)
    emit_info("cuda_version", cuda_version)
    emit_info("gpu_uuid", gpu_uuid)
    emit("cudnn_deterministic", 1.0 if cudnn_det else 0.0, 0)
    emit("cuda_available", 1.0 if torch.cuda.is_available() else 0.0, 0)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        emit("gpu_total_mem_gb", props.total_memory / 1e9, 0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({device})", flush=True)
    else:
        print(f"GPU not available; running on {device}", flush=True)

    # Build task list and pre-tabulate T_max per cell.
    tasks: list[tuple[int, float]] = []
    for n in n_full + n_coarse_only:
        for a in grids[n]:
            tasks.append((int(n), float(a)))
    total = len(tasks)
    rows: list[dict[str, Any]] = []
    started = time.time()
    t_max_table: dict[tuple[int, float], int] = {
        (n, a): compute_t_max(n, a, cap=t_max_cap) for (n, a) in tasks
    }

    print(
        f"V1 min-over | seed={master_seed} device={device} dtype={dtype_str} "
        f"N_full={n_full} N_coarse={n_coarse_only} cells={total} n_reps={n_reps} "
        f"eps_abs={eps_abs} smoke={smoke}",
        flush=True,
    )

    for done, (n, alpha) in enumerate(tasks, 1):
        p = round(alpha * n)
        t_max_used = t_max_table[(n, alpha)]
        t_pre = int(min(t_pre_mul * n, t_pre_cap))
        record_every = (
            int(record_every_override) if record_every_override else max(10, n // 5)
        )
        seed_int = _derive_seed(master_seed, n, round(1000 * alpha))

        t0 = time.time()

        if p == 0:
            n_sep = n_reps
            kappa_arr = np.full(n_reps, float("inf"), dtype=np.float64)
            diag_first_pos = np.zeros(n_reps, dtype=np.int64)
            diag_eps_stop = np.zeros(n_reps, dtype=np.int64)
            diag_hit_tmax = np.zeros(n_reps, dtype=bool)
            diag_novikoff = np.zeros(n_reps, dtype=bool)
            diag_final_margin = kappa_arr.copy()
        else:
            gen = torch.Generator(device=device).manual_seed(seed_int)
            patterns = torch.randn(n_reps, p, n, generator=gen, device=device, dtype=dtype)
            labels = (
                2.0
                * torch.randint(
                    0, 2, (n_reps, p), generator=gen, device=device, dtype=dtype
                )
                - 1.0
            )

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            res = minover_batch(
                patterns,
                labels,
                t_max=t_max_used,
                record_every=record_every,
                eps_abs=eps_abs,
                eps_window_mult=10,
                t_pre=t_pre,
            )
            n_sep = int(res["is_separable"].sum().item())
            kappa_arr = res["kappa"].detach().to("cpu", dtype=torch.float64).numpy()
            diag_first_pos = (
                res["first_positive_margin_iter"].detach().to("cpu").numpy().astype(np.int64)
            )
            diag_eps_stop = (
                res["iter_count_to_eps_stop"].detach().to("cpu").numpy().astype(np.int64)
            )
            diag_hit_tmax = res["hit_T_max"].detach().to("cpu").numpy().astype(bool)
            diag_novikoff = res["novikoff_early_exit"].detach().to("cpu").numpy().astype(bool)
            diag_final_margin = (
                res["mean_final_min_margin"].detach().to("cpu", dtype=torch.float64).numpy()
            )
            del patterns, labels, res
            if device.type == "cuda":
                torch.cuda.empty_cache()

        t_wall = time.time() - t0

        p_sep_hat = n_sep / max(1, n_reps)
        p_sep_cover = cover_separable_probability(n, alpha)

        sep_mask = kappa_arr > 0.0
        kappa_sep = kappa_arr[sep_mask] if np.any(sep_mask) else kappa_arr.copy()
        first_pos_sep = diag_first_pos[sep_mask] if np.any(sep_mask) else np.array([], np.int64)
        first_pos_sep = first_pos_sep[first_pos_sep >= 0]

        # iter_count_* statistics are over iter_count_to_eps_stop (per-instance convergence
        # cost: either the eps-stop step or t_max if criterion (b) ended the loop).
        iter_counts = diag_eps_stop.astype(np.float64)

        gpu_mem_peak_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        )

        row: dict[str, Any] = {
            # Identity
            "N": n,
            "alpha": float(alpha),
            "P": p,
            "n_reps": n_reps,
            "seed": seed_int,
            "algo_version": ALGO_VERSION,
            "T_max_used": t_max_used,
            # Aggregate
            "n_separable": n_sep,
            "p_sep_hat": p_sep_hat,
            "p_sep_cover": p_sep_cover,
            "kappa_emp_mean": float(np.mean(kappa_sep)) if kappa_sep.size else float("nan"),
            "kappa_emp_std": float(np.std(kappa_sep)) if kappa_sep.size else float("nan"),
            "kappa_emp_median": _percentile(kappa_sep, 50.0),
            "kappa_emp_q05": _percentile(kappa_sep, 5.0),
            "kappa_emp_q95": _percentile(kappa_sep, 95.0),
            # Convergence
            "iter_count_min": int(iter_counts.min()) if iter_counts.size else 0,
            "iter_count_median": int(np.median(iter_counts)) if iter_counts.size else 0,
            "iter_count_max": int(iter_counts.max()) if iter_counts.size else 0,
            "iter_count_p99": int(_percentile(iter_counts, 99.0)) if iter_counts.size else 0,
            "first_positive_margin_iter_median": (
                float(np.median(first_pos_sep)) if first_pos_sep.size else float("nan")
            ),
            "n_T_max_hit": int(diag_hit_tmax.sum()),
            "n_eps_stopped": int(n_reps - int(diag_hit_tmax.sum())),
            "n_novikoff_early_exit": int(diag_novikoff.sum()),
            "mean_final_min_margin": float(np.mean(diag_final_margin)),
            # Resources
            "t_wall_seconds": float(t_wall),
            "t_wall_max_instance": float(t_wall),  # batched across instances
            "gpu_mem_peak_mb": float(gpu_mem_peak_mb),
            # Reproducibility
            "torch_version": torch.__version__,
            "cuda_version": cuda_version,
            "gpu_uuid": gpu_uuid,
            "cudnn_deterministic": cudnn_det,
        }
        rows.append(row)

        step_int = round(1000 * alpha)
        emit(f"psep_N{n}", p_sep_hat, step_int)
        if kappa_sep.size:
            emit(f"kappa_N{n}", float(np.mean(kappa_sep)), step_int)
        emit("progress", done / total, done)
        kstep_per_sec = (int(row["iter_count_max"]) * n_reps) / max(t_wall, 1e-9) / 1000.0
        # Effective TFLOPS (einsum is the dominant op: 2*B*P*N FMAs per step).
        flops_per_step = 2.0 * n_reps * p * n
        effective_tflops = (
            float(row["iter_count_max"]) * flops_per_step / max(t_wall, 1e-9) / 1e12
        )
        print(
            f"[{done}/{total}] N={n:>5} a={alpha:5.3f} P={p:>5} "
            f"p_sep={p_sep_hat:.3f}/{p_sep_cover:.3f} "
            f"<k>={row['kappa_emp_mean']:+.4f} "
            f"iters p99={row['iter_count_p99']} "
            f"Tmax_hit={row['n_T_max_hit']}/{n_reps} "
            f"nov={row['n_novikoff_early_exit']}/{n_reps} "
            f"t={t_wall:.2f}s ({kstep_per_sec:.0f}k step-inst/s, "
            f"{effective_tflops:.2f} TFLOPS)",
            flush=True,
        )

    rows.sort(key=lambda r: (int(r["N"]), float(r["alpha"])))
    fieldnames = [
        # Identity
        "N", "alpha", "P", "n_reps", "seed", "algo_version", "T_max_used",
        # Aggregate
        "n_separable", "p_sep_hat", "p_sep_cover",
        "kappa_emp_mean", "kappa_emp_std", "kappa_emp_median",
        "kappa_emp_q05", "kappa_emp_q95",
        # Convergence
        "iter_count_min", "iter_count_median", "iter_count_max", "iter_count_p99",
        "first_positive_margin_iter_median",
        "n_T_max_hit", "n_eps_stopped", "n_novikoff_early_exit",
        "mean_final_min_margin",
        # Resources
        "t_wall_seconds", "t_wall_max_instance", "gpu_mem_peak_mb",
        # Reproducibility
        "torch_version", "cuda_version", "gpu_uuid", "cudnn_deterministic",
    ]
    with (run_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "params": params,
                "elapsed_seconds": time.time() - started,
                "rows": rows,
            },
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
