"""V6 -- Band-limited (sinusoidal-kernel) tempotron capacity alpha_c(K_eff) on a CERTIFIED K axis.

Self-contained Experiment-Contract entrypoint (numpy + torch only; the lab ships this file
unchanged, so there are NO ``tempotron`` / ``lab`` imports -- everything is inlined here). It is
the reduction-#6 sibling of ``v3_capacity_sweep.py``: the double-exponential PSP is replaced by a
flat-band (low-pass) kernel realized as a WINDOWED SINC, and the readout count K is now the EXACT,
dialable effective dimension ``K_eff = Omega_b*T/sqrt(3)`` (flat band) rather than the inferred Rice
up-crossing count ``T/sqrt(tau_s tau_m)`` of v3.

The science (see ``../sinusoidal_capacity/EXPERIMENT.md`` and
``lit/tempotron-reductions/derivations/06-sinusoidal-kernel-advance.md``)
--------------------------------------------------------------------------------------------------
For a stationary band-limited Gaussian readout V(t) the effective number of independent local
maxima over [0, T] -- the count that drives the EVT capacity lift -- is the crossing count
``K_eff = T sqrt(lambda_2/lambda_0)``, which for a flat low-pass band on [0, Omega_b] equals
``Omega_b T / sqrt(3)`` (Rice 1939 / Kac-Rice; reduction #6). This is the ONLY reduction where the
expressiveness axis K_eff is a known a-priori function of a single kernel knob (the bandwidth
Omega_b), so the findable capacity can be plotted against a *certified* K axis instead of a
back-fitted one (Rubin 2010 Fig 2a used K^discrete = K/8 as a free conversion).

The theory contact is the same double-log lift as v3,

    alpha_c(K_eff) = ln ln K_eff / (2 ln 2) + offset      (RMS 2010 eq. 3, existence reference),

anchored at ``alpha_c -> 2`` as ``K_eff -> 1`` (narrowband = Cover/Gardner perceptron). The sharp,
offset-robust observable is the SHAPE (monotone rise above 2, double-log not linear); the slope
1/(2 ln 2) = 0.7213 is the stretch (existence) target.

What this study measures
------------------------
For each target ``K_eff`` we set ``Omega_b = sqrt(3) K_eff / T`` (so ``K_eff = Omega_b T/sqrt3``),
build the band-limited traces on a Nyquist grid, and:

1. **Certify the x-axis.** From the realized init voltages we call :func:`measure_keff` (the
   crossing form ``T sqrt(lambda_2/lambda_0)``, inlined from
   ``../sinusoidal_capacity/witness_gate/witness_gate.py``) and log the MEASURED K_eff per cell --
   the certified abscissa for the capacity plot.
2. **Capacity.** We draw ``n_seeds`` independent tasks (Poisson patterns, balanced +/-1 labels),
   calibrate ``U_th`` to the median ``V_max`` at the random init (then fix it -- the balanced
   P(fire)=1/2 operating point the ln ln K lift requires; NOT a fixed kappa=0), and run the
   (batched) faithful online Gutig-Sompolinsky rule to a budget. A task counts solved if ANY of
   ``n_restarts`` random inits reaches zero training error (existence-from-below proxy).
   ``P_solve(alpha)`` is the solved fraction; its 1/2-crossing in alpha is ``alpha_hat_c(K_eff)``.

Input density: two modes (the K_eff/density confound fix)
---------------------------------------------------------
The total input richness is set by ``mean_spikes`` (expected spikes/afferent). Two modes:

- **Proportional (default, ``mean_spikes_fixed=0``).** ``mean_spikes = density_coeff * K_eff``
  (default ``density_coeff=0.05``). This keeps the number of afferents active per *kernel-width*
  (~``2 N mean_spikes / K_eff``) roughly constant as the band Omega_b grows, but the TOTAL input
  richness (spikes/afferent over [0,T]) co-varies with K_eff -- so a capacity rise along the K_eff
  axis is confounded with rising input richness (the bug this study addresses).
- **Fixed (``mean_spikes_fixed=m > 0``).** Every cell uses the SAME ``mean_spikes = m`` regardless
  of K_eff, holding total input richness CONSTANT across the K_eff sweep. Any alpha_c(K_eff) lift is
  then attributable to the expressiveness axis alone, not to more spikes (see
  [[tempotron-capacity-finite-N-starvation]]). This is the de-confounded mode for the capacity claim.

The actual ``mean_spikes`` used is recorded per row in ``results.csv``.

The Nyquist cost win
--------------------
A band-limited V is fixed by its Nyquist samples. We size the grid by ``oversample x Nyquist``:
``dt = pi/(oversample*Omega_b)``, ``n_grid = round(T/dt)+1`` (~ ``2.2 * oversample/2 * K_eff``
points). At oversample=4 that is ~ ``2.3 K_eff`` -- far smaller than the v3 fixed grid -- which is
what makes a single-N sweep affordable on the $25 budget.

Run standalone (tiny smoke; does NOT need a GPU):
    LAB_RUN_DIR=/tmp/v6_smoke uv run --with torch python studies/v6_bandlimited_capacity.py \\
        Keff_list=8 N_list=40 alphas=1.0,2.0,3.0 seeds=0,1 epochs=200 oversample=4
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


# --------------------------------------------------------------------------------------------
# Band-limited PSP kernel (windowed sinc, flat low-pass, EVEN / non-causal)
# --------------------------------------------------------------------------------------------
# K_band(s) = sinc(Omega_b s / pi) * window(s), peak-normalised to 1 at s=0, with a Lanczos
# (sinc) window spanning +/- L main lobes (support |s| <= L*pi/Omega_b; zero outside). The flat
# band has half-width Omega_b [rad/ms]; the kernel is the stationary-band idealisation of the PSP
# (the faithful reduction-#6 object) and is deliberately even/non-causal -- documented in the
# module docstring. This mirrors v3's psp_kernel signature/vectorisation so precompute_traces is
# a drop-in (the only physics change vs v3).
def band_kernel(s: torch.Tensor, omega_b: float, n_lobes: int = 6, window: str = "lanczos") -> torch.Tensor:
    """``K_band(s) = sinc(Omega_b s/pi) * window(s)``, peak 1 at s=0, compact support.

    Parameters
    ----------
    s : torch.Tensor
        Time lags [ms] (any shape). The kernel is even: ``K_band(-s) == K_band(s)``.
    omega_b : float
        Band half-width Omega_b [rad/ms]. The main lobe half-width is ``pi/Omega_b``.
    n_lobes : int
        Window half-span in main lobes L; support is ``|s| <= L*pi/Omega_b`` (zero outside).
    window : str
        ``"lanczos"`` (sinc window, the default) or ``"hann"`` (raised cosine over the support).

    Notes
    -----
    ``sinc(Omega_b s/pi) = sin(Omega_b s)/(Omega_b s)`` is the flat low-pass impulse response to
    half-width ``Omega_b``, normalised to 1 at ``s=0``. The window enforces compact support so the
    per-afferent traces are cheap; the Lanczos window is itself a sinc, ``sinc(s/(L*pi/Omega_b))``,
    which keeps the central lobe nearly intact while suppressing ringing.
    """
    half_support = n_lobes * math.pi / omega_b
    x = omega_b * s  # = pi * (s / main-lobe-half-width)
    # sinc with the s=0 singularity handled (sin x / x -> 1).
    sinc = torch.where(x.abs() < 1e-12, torch.ones_like(x), torch.sin(x) / x)
    if window == "hann":
        # Raised cosine over [-half_support, half_support], 1 at centre, 0 at the edge.
        win = 0.5 * (1.0 + torch.cos(math.pi * s / half_support))
    else:  # "lanczos" -- a sinc window of the same half-support
        xw = math.pi * s / half_support
        win = torch.where(xw.abs() < 1e-12, torch.ones_like(xw), torch.sin(xw) / xw)
    k = sinc * win
    return torch.where(s.abs() <= half_support, k, torch.zeros_like(k))


# --------------------------------------------------------------------------------------------
# K_eff certifier -- inlined from ../sinusoidal_capacity/witness_gate/witness_gate.py (measure_keff)
# --------------------------------------------------------------------------------------------
def measure_keff(V_samples_or_freqs: np.ndarray, T: float, *, dt: float | None = None) -> dict:
    """Certify ``K_eff = T sqrt(lambda_2/lambda_0)`` from time samples or band frequencies.

    Byte-faithful inline of ``witness_gate.measure_keff`` (the lab ships this file alone, so the
    certifier is copied here rather than imported). 2-D ``(M, n_t)`` -> time-sample path (needs
    ``dt``); 1-D -> frequency path. Returns ``keff_crossing`` plus the spectral moments used.
    """
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
    """Measure K_eff from realized init voltages built on ``s`` with random Gaussian weights.

    ``V[b,p,g] = sum_i s[b,p,g,i] xi_i`` with ``xi ~ N(0,I)`` is, for fixed (b,p), a sample path of
    the band-limited Gaussian readout. We flatten the (b,p) probe paths into the ``(M, n_grid)``
    time-sample matrix and feed it to :func:`measure_keff` (crossing form). Mean-centred per path so
    the zero-crossing rate measures the band, not a DC offset.
    """
    sb, p, g, n = s.shape
    dev = s.device
    flat = s.reshape(sb * p, g, n)
    m = min(n_probe, flat.shape[0])
    sel = flat[:m]  # (m, G, N)
    gen = torch.Generator(device=dev); gen.manual_seed(20260622)
    xi = torch.randn((m, n), generator=gen, device=dev, dtype=s.dtype)
    V = torch.einsum("mgn,mn->mg", sel, xi)  # (m, G)
    V = V - V.mean(dim=1, keepdim=True)
    dt = float(t_grid[1] - t_grid[0])
    return measure_keff(V.detach().cpu().numpy(), T, dt=dt)


# --------------------------------------------------------------------------------------------
# Reproducible per-cell seeding (matches v3 cell_seed recipe; K rounded -> use round(K_eff))
# --------------------------------------------------------------------------------------------
def cell_seed(master: int, n: int, alpha: float, k: float, tag: int = 0) -> int:
    """A scheduling-independent 63-bit seed for a cell (matches v3's SeedSequence recipe)."""
    ss = np.random.SeedSequence([master, n, round(1000 * alpha), round(k), tag])
    return int(ss.generate_state(1, dtype=np.uint64)[0]) >> 1


# --------------------------------------------------------------------------------------------
# Pattern generation (RMS 2010 Poisson input), batched over tasks -- mirrors v3.make_patterns
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
    """Draw ``n_tasks`` independent Poisson pattern sets (RMS 2010), balanced +/-1 labels.

    Each afferent emits a homogeneous Poisson train on ``[0, T]`` with ``mean_spikes`` expected
    spikes. Drawn on the CPU via a numpy Generator then moved to ``device`` (device-INDEPENDENT,
    bit-reproducible on any backend). Returns ``(spike_times, valid, labels, max_spikes)`` with
    ``spike_times, valid`` of shape ``(n_tasks, P, N, max_spikes)``.
    """
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
    omega_b: float,
    n_lobes: int,
    window: str,
    elem_budget: int = 64_000_000,
) -> torch.Tensor:
    """Per-afferent band-limited traces ``s[b,g,p,i] = sum_f K_band(t_g - spike[b,p,i,f]) * valid``.

    Identical chunking strategy to v3.precompute_traces (kernel swap only): the working tensor
    ``(g_chunk, Sb, P, N, max_spikes)`` is kept within ``elem_budget`` elements. Returns ``s`` of
    shape ``(Sb, P, G, N)`` (float32); the per-pattern slice ``s[:, p]`` is contiguous, which the
    online sweep relies on.
    """
    sb, p, n, ms = spike_times.shape
    g = t_grid.shape[0]
    s = torch.empty((sb, p, g, n), device=spike_times.device, dtype=torch.float32)
    per_g = max(1, sb * p * n * ms)
    g_chunk = max(1, elem_budget // per_g)
    st = spike_times.unsqueeze(0)  # (1, Sb, P, N, ms)
    vd = valid.unsqueeze(0)
    for g0 in range(0, g, g_chunk):
        g1 = min(g0 + g_chunk, g)
        tt = t_grid[g0:g1].view(-1, 1, 1, 1, 1)                          # (gc,1,1,1,1)
        contrib = band_kernel(tt - st, omega_b, n_lobes, window) * vd    # (gc, Sb, P, N, ms)
        s[:, :, g0:g1, :] = contrib.sum(dim=4).permute(1, 2, 0, 3)       # (gc,Sb,P,N)->(Sb,P,gc,N)
    return s


# --------------------------------------------------------------------------------------------
# Forward pass + faithful online Gutig-Sompolinsky training (vectorised over tasks)
# --------------------------------------------------------------------------------------------
def _forward(s: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``V[b,p,g] = sum_i s[b,p,g,i] w[b,i]``; return ``(vmax, argmax_t)`` each ``(Sb, P)``."""
    v = torch.einsum("bpgn,bn->bpg", s, w)
    vmax, targ = v.max(dim=2)
    return vmax, targ


EXP_DECAY_K: float = 5.0  # 'exp' schedule: lr*exp(-K*t); at t=1 -> ~0.0067*lr (treated as "->0")


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
    """Scheduled learning rate at epoch ``ep`` (V10/V11: the tuned nuisance LR schedule).

    A linear **warmup** over the first ``warmup`` epochs (``lr*(ep+1)/warmup``) precedes the chosen
    decay. After warmup (``t`` is the fraction of the post-warmup budget elapsed):
    - ``"none"`` (constant): ``lr``.
    - ``"cosine"``: half-cosine anneal ``lr -> floor*lr`` over ``[warmup, epochs]`` (V11: ``floor``).
      With ``n_cycles>1`` (V11 SGDR warm restarts) the cosine repeats over ``n_cycles`` equal cycles,
      each resetting to the peak; restarts are exact only when ``span=epochs-warmup`` divides evenly
      by ``n_cycles`` (otherwise the cycle boundary lands mid-step and the reset is approximate).
    - ``"linear"`` (V11): linear anneal ``lr -> floor*lr``.
    - ``"exp"`` (V11): ``lr * exp(-exp_k*t)`` (default ``exp_k=EXP_DECAY_K=5`` ends at ~``0.0067*lr``).
      Smaller ``exp_k`` decays gently (longer high-LR exploration); larger settles faster. NOTE: ``exp``
      does **not** apply ``floor`` -- its tail is set by ``exp_k`` alone.
    - ``"step"``: ``lr * gamma**floor(ep/step_size)`` (no decay if ``step_size<=0``).
    Defaults ``floor=0.0`` / ``n_cycles=1`` make cosine/linear anneal to 0 in a single cycle, i.e.
    they preserve the pre-V11 behaviour exactly.

    Extracted as a module function so the schedule is unit-testable in isolation; ``train_batch``
    calls it once per epoch. Copied verbatim from ``v3_capacity_sweep.scheduled_lr``.
    """
    if warmup and ep < warmup:
        return lr * (ep + 1) / warmup
    span = max(1, epochs - warmup)
    t = min(1.0, max(0.0, (ep - warmup) / span))
    if schedule == "cosine":
        if n_cycles > 1:
            tc = span / n_cycles                      # cycle length (epochs)
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
    return lr  # "none" / constant


def train_batch(
    s: torch.Tensor,        # (Sb, P, G, N) precomputed traces
    labels: torch.Tensor,   # (Sb, P) in {-1,+1}
    w_init: torch.Tensor,   # (Sb, N)
    *,
    lr: float,
    momentum: float = 0.0,
    epochs: int,
    threshold: torch.Tensor | None = None,  # (Sb,) fixed U_th; if None, calibrate at init
    patience: int = 0,      # if >0, early-stop a cell after this many no-progress epochs
    log_every: int = 0,
    log_tag: str = "",
    train_seed: int = 0,    # seeds the per-epoch shuffle RNG -> deterministic training (gate fix)
    capture: bool = False,  # if True, record per-epoch seed-batch trajectories (port of v3 cap path)
    metric_cb=None,         # optional callback(epoch, err_mean, loss_mean) -> stream a live learning curve
    mode: str = "online",   # 'online' (faithful GS rule, default) | 'minibatch' (V10 strong learner)
    # --- V10 minibatch path (mode='minibatch'): tuned optimizer / schedule / batch HPs (port of v3) ---
    optimizer: str = "momentum",  # momentum | adam | rmsprop (the findability probe)
    lr_schedule: str = "none",    # none | cosine | linear (decay lr over the budget)
    lr_warmup: int = 0,           # linear lr warmup over the first this-many epochs
    lr_floor: float = 0.0,        # anneal cosine/linear to lr_floor*lr instead of 0
    lr_cycles: int = 1,           # SGDR warm-restart cycle count for the cosine schedule
    lr_exp_k: float = EXP_DECAY_K,  # exp-schedule decay rate
    lr_step_size: int = 0,        # 'step' schedule: epochs between gamma-decays
    lr_gamma: float = 0.1,        # 'step' schedule: multiplicative decay factor
    batch_size: int = 0,          # minibatch size b (1=online .. P=full); required for mode='minibatch'
    adam_betas: tuple[float, float] = (0.9, 0.999),  # Adam (beta1, beta2)
    adam_eps: float = 1e-8,       # Adam epsilon
    rms_alpha: float = 0.99,      # RMSprop smoothing rho
    rms_eps: float = 1e-8,        # RMSprop epsilon
) -> dict[str, torch.Tensor]:
    """Gutig-Sompolinsky rule, vectorised across tasks. Threshold fixed (median V_max at init).

    ``mode='online'`` (default, what GS/RMS used): per-pattern updates within a shuffled epoch; a
    task is converged once a full epoch passes with zero updates (the textbook perceptron criterion).
    Threshold is calibrated to the median V_max at the random init (the balanced P(fire)=1/2 operating
    point the ln ln K lift requires), then held fixed. ``momentum>0`` chains the GS velocity across
    error trials.

    ``mode='minibatch'`` (V10 strong learner, ported from ``v3_capacity_sweep.train_batch``): one
    shuffled pass split into ceil(P/b) minibatch steps; the GS hinge subgradient is the per-minibatch
    MEAN over margin-violators ``g = (1/b) sum_{viol} y s(t*)``, ascended by a selectable optimizer
    (momentum/Adam/RMSprop) under a cosine/linear/none LR schedule. A step is gated per-seed by
    (still active) AND (a violator present in this minibatch); at ``batch_size=1``, ``optimizer=momentum``,
    constant LR this reproduces the faithful online GS rule. A NaN-weight seed (LR-too-large divergence)
    is NOT counted as converged (the divergence guard). Convergence = a full epoch with zero violators.

    Returns per-task ``converged``, ``epochs_run``, ``init_fire_rate``, ``final_errfrac``,
    ``weights``, ``threshold``.

    Per-epoch capture (``capture=True``; default OFF, cheap path unchanged): ported from
    ``v3_capacity_sweep.train_batch``. Records, per epoch, the seed-batch trajectories
    ``traj_err`` (training-error fraction), ``traj_loss`` (GS hinge loss
    ``relu(-signed_margin)``), ``traj_kp`` (mean signed margin on +1 patterns, kappa_plus),
    ``traj_km`` (mean signed margin on -1 patterns, kappa_minus) and ``traj_wnorm`` (||w||), each
    of shape ``(Sb, n_epochs)`` (per-seed; NaN-padded then sliced to the epochs actually run),
    returned under the ``"cap"`` key. Both the online and minibatch branches populate these via the
    cheap per-epoch reductions already computed in the hot loop (no extra forward pass). If
    ``metric_cb`` is given it is called once per epoch with the seed-batch-mean ``(epoch, err, loss)``
    so the driver can stream a live learning curve to lab metrics.
    """
    sb, p, _, n = s.shape  # (Sb, P, G, N) -- n (afferents) used by the minibatch gather
    dev = s.device
    # Per-call seeded RNG for the epoch shuffle: without this, torch's unseeded global RNG makes the
    # online GS trajectory (and convergence outcome at the margin) non-reproducible (pre-spend gate fix).
    shuffle_gen = torch.Generator(device=dev)
    shuffle_gen.manual_seed(int(train_seed) & 0x7FFFFFFFFFFFFFFF)
    w = w_init.clone()
    vmax, _ = _forward(s, w)
    if threshold is None:
        threshold = vmax.median(dim=1).values  # (Sb,) median V_max at init -> P(fire)=1/2
    init_fire_rate = (vmax >= threshold[:, None]).float().mean(dim=1)
    converged = torch.zeros(sb, dtype=torch.bool, device=dev)
    epochs_run = torch.full((sb,), epochs, dtype=torch.int64, device=dev)
    arange_sb = torch.arange(sb, device=dev)
    vel = torch.zeros_like(w)
    best_conv = 0
    stall = 0
    # V10 minibatch optimizer state (Adam/RMSprop moments; unused by the online path).
    adam_m = torch.zeros_like(w)
    adam_v = torch.zeros_like(w)
    adam_step = 0           # global Adam step counter for bias correction
    last_ep = -1

    # Per-epoch capture buffers (port of v3.train_batch). NaN-padded to the budget and sliced to the
    # epochs actually run on return; everything downstream is derived offline from these arrays.
    cap = bool(capture) and mode in ("online", "minibatch")
    if cap:
        err_buf = torch.full((sb, epochs), float("nan"), device=dev)
        loss_buf = torch.full((sb, epochs), float("nan"), device=dev)
        wnorm_buf = torch.full((sb, epochs), float("nan"), device=dev)
        kp_buf = torch.full((sb, epochs), float("nan"), device=dev)
        km_buf = torch.full((sb, epochs), float("nan"), device=dev)
        npos = (labels > 0).float().sum(dim=1).clamp(min=1.0)  # (Sb,) per-seed #(+1) patterns
        nneg = (labels < 0).float().sum(dim=1).clamp(min=1.0)  # (Sb,) per-seed #(-1) patterns

    def _lr_at(ep: int) -> float:
        """Scheduled learning rate for this epoch (== lr when lr_schedule='none')."""
        return scheduled_lr(lr, ep, epochs=epochs, schedule=lr_schedule, warmup=lr_warmup,
                            step_size=lr_step_size, gamma=lr_gamma,
                            floor=lr_floor, n_cycles=lr_cycles, exp_k=lr_exp_k)

    for ep in range(epochs):
        active = (~converged).float().unsqueeze(-1)  # (Sb, 1)
        if mode == "minibatch":  # V10 unified path: b in {1..P}, tuned optimizer / schedule / freeze
            # One shuffled pass split into ceil(P/b) minibatch steps (port of v3.train_batch). The GS
            # hinge subgradient is the per-minibatch MEAN over margin-violators, g = (1/b) sum_{viol}
            # y s(t*); each optimizer ascends +g. A step is gated per-seed by (still active) AND (has a
            # violator in this minibatch). At b=1, constant lambda, mu reproduces the faithful online GS
            # rule (velocity advances only on error steps). Convergence = a full epoch with zero violators.
            # The shuffle uses the SAME seeded generator as the online path -> bit-reproducible.
            lr_ep = _lr_at(ep)
            order = torch.randperm(p, device=dev, generator=shuffle_gen)
            bs = max(1, min(batch_size, p))
            n_steps = (p + bs - 1) // bs
            act = active.squeeze(-1)                          # (Sb,) 1.0 while unconverged
            viol_enc = torch.zeros(sb, device=dev)            # violators seen by active seeds -> converge
            if cap:  # per-epoch capture accumulators (seed-batch, summed over minibatch steps)
                err_acc = torch.zeros(sb, device=dev)
                cost_acc = torch.zeros(sb, device=dev)
                kp_acc = torch.zeros(sb, device=dev)
                km_acc = torch.zeros(sb, device=dev)
            for st_i in range(n_steps):
                bidx = order[st_i * bs:(st_i + 1) * bs]       # (b,) pattern indices this minibatch
                sb_s = s[:, bidx]                              # (Sb, b, G, N)
                lab_b = labels[:, bidx]                        # (Sb, b)
                bb = sb_s.shape[1]
                vb = torch.einsum("bkgn,bn->bkg", sb_s, w)     # (Sb, b, G)
                vmx, tg = vb.max(dim=2)                        # (Sb, b)
                smarg = (vmx - threshold[:, None]) * lab_b     # (Sb, b) signed margin
                viol = (smarg < 0.0).float()                   # (Sb, b) margin-violating patterns
                nb = float(bb)
                viol_enc = viol_enc + viol.sum(dim=1) * act
                gidx = tg.view(sb, bb, 1, 1).expand(sb, bb, 1, n)
                gpat = torch.gather(sb_s, 2, gidx).squeeze(2)  # (Sb, b, N) = s_i(t*) per pattern
                g = ((lab_b * viol).unsqueeze(-1) * gpat).sum(dim=1)  # (Sb, N) GS ascent over violators
                g = g / nb                                     # mean over the minibatch
                has_viol = (viol.sum(dim=1) > 0).float().unsqueeze(-1)  # (Sb,1) does this seed update?
                gate = active * has_viol                       # active AND a violator present
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
                else:  # momentum -- gate the velocity too so only error steps advance it (b=1 == GS)
                    vel = gate * (momentum * vel + g) + (1.0 - gate) * vel
                    step = lr_ep * vel
                w = w + step * gate
                if cap:  # accumulate this minibatch's contributions to the per-epoch trajectory
                    err_acc = err_acc + viol.sum(dim=1)
                    cost_acc = cost_acc + torch.relu(-smarg).sum(dim=1)
                    kp_acc = kp_acc + (smarg * (lab_b > 0).float()).sum(dim=1)
                    km_acc = km_acc + (smarg * (lab_b < 0).float()).sum(dim=1)
            if cap:  # record the seed-batch trajectory for this epoch (no extra forward pass)
                err_buf[:, ep] = err_acc / p
                loss_buf[:, ep] = cost_acc / p
                wnorm_buf[:, ep] = w.norm(dim=1)
                kp_buf[:, ep] = kp_acc / npos
                km_buf[:, ep] = km_acc / nneg
                if metric_cb is not None:
                    metric_cb(ep + 1, float((err_acc / p).mean()), float((cost_acc / p).mean()))
            # A diverged seed (NaN/inf weights from too-large an LR) has NaN margins, and NaN<0 is
            # False -> it would look like "zero violators". Require finite weights to count as solved,
            # so divergent HP configs report low p_solve (not spuriously "converged"). [divergence guard]
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
        if cap:  # per-epoch capture accumulators (seed-batch, summed over patterns; ungated)
            err_acc = torch.zeros(sb, device=dev)
            cost_acc = torch.zeros(sb, device=dev)
            kp_acc = torch.zeros(sb, device=dev)
            km_acc = torch.zeros(sb, device=dev)
        for pi in order:
            sp = s[:, pi]                               # (Sb, G, N) contiguous
            vp = torch.einsum("bgn,bn->bg", sp, w)
            vmx, tg = vp.max(dim=1)                     # (Sb,) tempotron argmax_t V
            smarg = (vmx - threshold) * labels[:, pi]   # signed margin
            if cap:
                lab_pi = labels[:, pi]
                err_acc = err_acc + (smarg < 0.0).float()
                cost_acc = cost_acc + torch.relu(-smarg)
                kp_acc = kp_acc + smarg * (lab_pi > 0).float()
                km_acc = km_acc + smarg * (lab_pi < 0).float()
            e = (smarg < 0.0).float() * active.squeeze(-1)
            updates = updates + e
            grad = sp[arange_sb, tg]                    # (Sb, N) = s_i(t_max)
            corr = (lr * labels[:, pi]).unsqueeze(-1) * grad  # GS correction dw = lr*y*s(t_max)
            e_col = e.unsqueeze(-1)
            vel = e_col * (corr + momentum * vel) + (1.0 - e_col) * vel
            w = w + e_col * vel
        if cap:  # record the seed-batch trajectory for this epoch (no extra forward pass)
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
        # For still-unconverged seeds, record the epochs actually executed (last_ep+1) rather than the
        # full budget, so downstream epoch accounting is exact even when the run early-stopped (mirrors
        # v3). Converged seeds keep their convergence epoch.
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
        # Slice the NaN-padded buffers to the epochs actually run for this batch (last_ep+1). All
        # downstream numerics/figures are derived offline from these per-seed per-epoch arrays.
        ne = last_ep + 1
        out["cap"] = {
            "traj_err": err_buf[:, :ne],     # (Sb, ne) per-seed training-error fraction vs epoch
            "traj_loss": loss_buf[:, :ne],   # (Sb, ne) per-seed GS hinge loss vs epoch
            "traj_wnorm": wnorm_buf[:, :ne],  # (Sb, ne) per-seed ||w|| vs epoch
            "traj_kp": kp_buf[:, :ne],       # (Sb, ne) per-seed kappa_plus (mean +1 signed margin)
            "traj_km": km_buf[:, :ne],       # (Sb, ne) per-seed kappa_minus (mean -1 signed margin)
            "n_epochs": ne,                  # epochs this batch ran (== budget unless early-stopped)
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
    keff_list = [float(x) for x in ov.get("Keff_list", "2,8,32").split(",")]
    n_list = [int(x) for x in ov.get("N_list", "500").split(",")]
    alphas = _alpha_grid(ov)
    # seeds=0,1,2,.. enumerated manifest (lab-shardable, one shard per seed); n_seeds is the legacy
    # contiguous fallback used only when seeds= is absent.
    if "seeds" in ov:
        seeds = [int(x) for x in ov["seeds"].split(",") if x != ""]
    else:
        seeds = list(range(int(ov.get("n_seeds", "24"))))
    n_seeds = len(seeds)
    T_ms = float(ov.get("T_ms", "500"))
    oversample = float(ov.get("oversample", "4"))
    n_lobes = int(ov.get("n_lobes", "6"))
    window = ov.get("window", "lanczos")
    density_coeff = float(ov.get("density_coeff", "0.05"))  # mean_spikes = density_coeff * K_eff (proportional mode)
    # De-confound knob: if >0, hold mean_spikes FIXED at this value for EVERY cell regardless of K_eff
    # (constant total input richness across the K_eff sweep); 0 (default) keeps the proportional mode.
    mean_spikes_fixed = float(ov.get("mean_spikes_fixed", "0"))
    capture = int(ov.get("capture", "0"))  # if 1, write per-epoch learning curves to history.csv + stream metrics
    epochs = int(ov.get("epochs", "2000"))
    lr = float(ov.get("lr", "0.05"))
    momentum = float(ov.get("momentum", "0.0"))
    # V10 strong-learner path (port of v3). mode=online (default) preserves the faithful GS rule
    # exactly; mode=minibatch enables the tuned optimizer / LR schedule / minibatch findability probe.
    mode = ov.get("mode", "online")              # online | minibatch
    optimizer = ov.get("optimizer", "momentum")  # momentum | adam | rmsprop (minibatch mode)
    lr_schedule = ov.get("lr_schedule", "none")  # none | cosine | linear (minibatch mode)
    lr_warmup = int(ov.get("lr_warmup", "0"))    # linear lr warmup over the first this-many epochs
    lr_floor = float(ov.get("lr_floor", "0.0"))  # anneal cosine/linear to lr_floor*lr instead of 0
    lr_cycles = int(ov.get("lr_cycles", "1"))    # SGDR warm-restart cycle count for cosine
    lr_step_size = int(ov.get("lr_step_size", "0"))
    lr_gamma = float(ov.get("lr_gamma", "0.1"))
    batch_size = int(ov.get("batch_size", "0"))  # b in {1..P}; 0 => online mode preserved
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
    keff_probe = int(ov.get("keff_probe", "64"))  # # init voltage paths fed to measure_keff
    require_cuda = int(ov.get("require_cuda", "0"))

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
        f"V6 band-limited capacity | seed={master_seed} device={device} Keff={keff_list} N={n_list} "
        f"alphas={alphas} seeds={seeds} epochs={epochs} lr={lr} mom={momentum} T={T_ms} "
        f"oversample={oversample} n_lobes={n_lobes} window={window} density_coeff={density_coeff} "
        f"restarts={n_restarts} mode={mode} optimizer={optimizer} lr_schedule={lr_schedule} "
        f"batch_size={batch_size}",
        flush=True,
    )

    # Optional live metric streaming to lab metrics. Import is wrapped so the study stays import-clean
    # when run standalone (the lab ships this file alone, with no `lab` package on the path); falls back
    # to a no-op. Used only when capture=1 (the first cell streams a live train_err/loss learning curve).
    _log_metric = None
    if capture:
        try:
            from lab.metrics import log_metric as _log_metric  # type: ignore
        except Exception:
            _log_metric = None

    started = time.time()
    rows: list[dict[str, float]] = []
    history_rows: list[dict[str, float]] = []  # tidy long-format per-epoch learning curves (capture=1)
    per_keff: dict[float, dict] = {}
    n_cells = len(keff_list) * len(n_list) * len(alphas)
    cell_i = 0
    first_cell_done = False  # only the FIRST trained cell streams scalars to lab metrics (one live curve)

    # Incremental, durable results.csv: write the header once (lazily, when the first cell's rows
    # fix the column order) and flush+fsync after EACH (K_eff, N, alpha) cell so a timeout/OOM kill
    # leaves a valid, readable partial CSV with all completed cells. The in-memory ``rows`` list is
    # still kept for the end-of-run results.json. ``_csv_written`` tracks rows already on disk.
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
        # Bandwidth set by the flat-band crossing count: K_eff = Omega_b T / sqrt3.
        omega_b = SQRT3 * keff_t / T_ms                          # [rad/ms]
        # Input density: fixed mode (constant richness across the sweep, the de-confounded mode) when
        # mean_spikes_fixed>0; else proportional (density_coeff*K_eff). The realized value is logged.
        if mean_spikes_fixed > 0:
            mean_spikes = mean_spikes_fixed
        else:
            mean_spikes = max(1e-3, density_coeff * keff_t)
        dt = math.pi / (oversample * omega_b)                   # oversample x Nyquist (Nyquist = pi/Omega_b)
        n_grid = round(T_ms / dt) + 1
        t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * dt
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
                                          omega_b, n_lobes, window, elem_budget)
                    lab_b = labels[b0:b1]
                    # Certify K_eff from realized init voltages on this seed-batch (the certified
                    # x-axis: measured crossing-form K_eff = T sqrt(lambda2/lambda0) ~ Omega_b T/sqrt3).
                    cert = certify_keff_from_traces(s, t_grid, T_ms, n_probe=keff_probe)
                    keff_meas_batches.append(cert["keff_crossing"])
                    wrng = np.random.default_rng(cell_seed(master_seed, n_aff, alpha, keff_t, tag=1) + b0)
                    w0 = torch.from_numpy(
                        (sigma_w * wrng.standard_normal((b1 - b0, n_aff))).astype(np.float32)).to(device)
                    # Live metric streaming: only the FIRST trained cell streams (one clean learning curve
                    # in `lab metrics <job>`); no-op if log_metric unimportable. Capture is recorded for
                    # the primary (first-init) run of every cell when capture=1.
                    stream_this = capture and (not first_cell_done) and (_log_metric is not None)

                    def _mcb(ep: int, errm: float, lossm: float) -> None:
                        _log_metric("train_err", float(errm), step=int(ep))  # type: ignore
                        _log_metric("loss", float(lossm), step=int(ep))      # type: ignore

                    res = train_batch(s, lab_b, w0, lr=lr, momentum=momentum, epochs=epochs,
                                      patience=patience, log_every=log_every,
                                      log_tag=f"K{round(keff_t)}N{n_aff}a{alpha:.2f}b{b0}",
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
                        # Tidy long-format learning-curve rows for this seed-batch (per-seed; the driver
                        # writes them all to history.csv at the end). Seed-batch-averaged is acceptable;
                        # here we emit per-seed (cheaper to aggregate downstream, richer).
                        cd = res["cap"]
                        te = cd["traj_err"].cpu().numpy()    # (sb_batch, ne)
                        tl = cd["traj_loss"].cpu().numpy()
                        tkp = cd["traj_kp"].cpu().numpy()
                        tkm = cd["traj_km"].cpu().numpy()
                        twn = cd["traj_wnorm"].cpu().numpy()
                        nb_seeds, ne = te.shape
                        for j in range(nb_seeds):
                            sd = seeds[b0 + j]
                            for e_i in range(ne):
                                history_rows.append({
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

                    # Multi-restart existence-from-below proxy: ANY of n_restarts inits solving counts.
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
                        # epochs_to_solve: earliest restart's convergence epoch (min over restarts)
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
                n_solved = int(conv_multi.sum().item())  # solved = ANY restart (existence-from-below)
                p_solve = n_solved / n_seeds
                keff_alpha_psolve.append(p_solve)
                # per-seed rows (lab-shardable): one row per (seed, Keff, alpha)
                conv_np = conv_multi.cpu().numpy()
                ep_np = ep_run.cpu().numpy()
                fire_np = fire.cpu().numpy()
                errf_np = errf.cpu().numpy()  # primary-init residual training-error fraction per seed
                for j, sd in enumerate(seeds):
                    solved = int(conv_np[j])
                    rows.append({
                        "seed": sd,
                        "Keff_target": keff_t,
                        "Keff_measured": keff_measured_cell,
                        "omega_b": omega_b,
                        "N": n_aff,
                        "alpha": alpha,
                        "P": p,
                        "solved": solved,
                        "epochs_to_solve": int(ep_np[j]) if solved else -1,
                        "final_train_err": float(errf_np[j]),  # how close an unsolved cell got (0 if solved)
                        "init_fire_rate": float(fire_np[j]),
                        "n_grid": n_grid,
                        "dt": dt,
                        "mean_spikes": mean_spikes,
                        "n_restarts": n_restarts,
                        "optimizer": optimizer,
                        "lr_schedule": lr_schedule,
                        "batch_size": batch_size,
                    })
                _flush_new_rows()  # durable per-cell write (timeout/OOM-safe partial CSV)
                p_se = math.sqrt(max(p_solve * (1 - p_solve), 0.0) / n_seeds)
                print(
                    f"[{cell_i}/{n_cells}] Keff_t={keff_t:.1f} Keff_meas={keff_measured_cell:.2f} "
                    f"N={n_aff} a={alpha:.3f} P={p} p_solve={p_solve:.3f}+/-{p_se:.3f} "
                    f"n_grid={n_grid} fire0={float(fire.mean().item()):.2f}",
                    flush=True,
                )
        if n_list:  # crossing computed for the (last) N over the alpha grid
            ahat = _crossing_alpha(alphas, keff_alpha_psolve)
            per_keff[keff_t] = {
                "Keff_target": keff_t,
                "Keff_measured": keff_measured_cell,
                "omega_b": omega_b,
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
        "experiment": "v6_bandlimited_capacity",
        "git_sha": git_sha,
        "versions": {"numpy": np.__version__, "torch": torch.__version__,
                     "python": sys.version.split()[0]},
        "params": {
            "master_seed": master_seed, "Keff_list": keff_list, "N_list": n_list, "alphas": alphas,
            "seeds": seeds, "n_seeds": n_seeds, "T_ms": T_ms, "oversample": oversample,
            "n_lobes": n_lobes, "window": window, "density_coeff": density_coeff,
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
    _flush_new_rows()  # write any trailing rows; CSV was written incrementally per cell
    csv_file.close()
    # Per-epoch learning curves (capture=1): tidy long-format, one row per (cell, seed, epoch).
    if capture and history_rows:
        hcols = ["Keff_target", "N", "alpha", "seed_batch", "epoch",
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
            print(f"  K_eff target={k:.1f} measured={pk['Keff_measured']:.2f} "
                  f"alpha_hat_c={pk['alpha_hat_c']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
