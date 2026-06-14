"""V3 -- Tempotron memory capacity alpha_c(K): reproduce RMS 2010 Fig. 2a and probe findability.

Self-contained Experiment-Contract entrypoint (numpy + torch only; the lab ships this file
unchanged, so there are NO ``tempotron`` / ``lab`` imports -- everything is inlined here and the
validated numpy package is used only as the offline G0 reference in tests/test_v3_capacity.py).

The science (see ``docs/05-statistical-mechanics-of-capacity.md`` and ``docs/v3-capacity-design.md``)
--------------------------------------------------------------------------------------------------
The tempotron fires iff ``V_max = max_t V(t) >= U_th``. Because the +1 (target) constraint
``exists t: w.s(t) >= U_th`` is a *union of half-spaces* it is NON-convex, so there is no linear
separability (LP) oracle as there is for the perceptron -- RMS measured the capacity by *training
the tempotron*. We do the same: the capacity oracle here is the Gutig-Sompolinsky gradient rule,
which makes this the *findable* capacity, with the replica formula

    alpha_c(K) = ln ln K / (2 ln 2)        (RMS 2010 eq. 3, the existence reference)

as the theory contact. The SHARP, offset-robust test is the *shape*: regressing the measured
half-crossing ``alpha_hat_c(K)`` on ``ln ln K`` should give slope ``1/(2 ln 2) = 0.7213`` (finite-N
sims sit on the curve up to an additive offset alpha_0 ~ 2.58 that cancels in the slope).

Method
------
For each cell ``(K, N, alpha)`` we draw ``n_seeds`` independent random tasks (RMS Poisson patterns,
+/-1 labels), calibrate ``U_th`` to the median ``V_max`` at the random init (then fix it), and run
the (batched) Gutig-Sompolinsky rule to a budget. ``P_solve(alpha)`` is the fraction of tasks driven
to zero training error; its 1/2-crossing in ``alpha`` is ``alpha_hat_c(K)``.

Performance: the per-afferent traces ``s_i^mu(t) = sum_f K(t - t_i^f)`` do not depend on the
weights, so we PRECOMPUTE them once per task (chunked over the time grid to bound memory) and reuse
them across all training epochs -- each epoch is then a cheap matmul + argmax + gather. Tasks are
processed in seed-batches sized to a memory budget on the trace tensor. We use **batch-mode** GS
(sum the misclassified updates, apply once per epoch) because it vectorises across tasks; this is a
standard variant of the online rule (documented in the design doc; the slope test is robust to it).

Multi-restart (existence-from-below proxy): at ``anchor_K`` we re-run training from ``n_restarts``
random inits reusing the same traces; a task counts as solved-multi if ANY init reaches zero error.

Run standalone (tiny smoke; does NOT need a GPU):
    LAB_RUN_DIR=/tmp/v3_smoke LAB_SEED=0 \\
        uv run --with torch python studies/v3_capacity_sweep.py \\
        K_list=16 N_list=50 alphas=0.5,1.5,2.5,3.5 n_seeds=8 epochs=300
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

TAU_M: float = 15.0  # ms, membrane time constant (Gutig & Sompolinsky 2006 + RMS 2010)
TAU_S: float = 3.75  # ms, synaptic time constant
SQRT_TAU: float = math.sqrt(TAU_S * TAU_M)  # = 7.5 ms, the PSP correlation time sqrt(tau_s tau_m)


# --------------------------------------------------------------------------------------------
# PSP kernel (peak-normalised double exponential, causal) -- mirrors tempotron.kernel
# --------------------------------------------------------------------------------------------
def _psp_v0() -> float:
    t_peak = TAU_M * TAU_S / (TAU_M - TAU_S) * math.log(TAU_M / TAU_S)
    return 1.0 / (math.exp(-t_peak / TAU_M) - math.exp(-t_peak / TAU_S))


PSP_V0: float = _psp_v0()


def psp_kernel(t: torch.Tensor) -> torch.Tensor:
    """``K(t) = V0 (e^{-t/tau_m} - e^{-t/tau_s})`` for ``t >= 0``, else 0 (peak-normalised to 1)."""
    t_clamped = torch.clamp(t, min=0.0)
    raw = PSP_V0 * (torch.exp(-t_clamped / TAU_M) - torch.exp(-t_clamped / TAU_S))
    return torch.where(t >= 0.0, raw, torch.zeros_like(raw))


# --------------------------------------------------------------------------------------------
# Reproducible per-cell seeding
# --------------------------------------------------------------------------------------------
def cell_seed(master: int, n: int, alpha: float, k: float, tag: int = 0) -> int:
    """A scheduling-independent 63-bit seed for a cell (matches the studies' SeedSequence recipe)."""
    ss = np.random.SeedSequence([master, n, round(1000 * alpha), round(k), tag])
    return int(ss.generate_state(1, dtype=np.uint64)[0]) >> 1


# --------------------------------------------------------------------------------------------
# Pattern generation (RMS 2010 Poisson input), batched over tasks
# --------------------------------------------------------------------------------------------
def make_patterns(
    n_tasks: int,
    n_patterns: int,
    n_aff: int,
    t_window: float,
    rng: np.random.Generator,
    device: torch.device,
    mean_spikes: float = 1.0,
    ensemble: str = "poisson",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Draw ``n_tasks`` independent pattern sets.

    ``ensemble`` selects the input model:

    - ``"poisson"`` (default, RMS 2010): each afferent emits a homogeneous Poisson train on
      ``[0, T]`` with ``mean_spikes`` expected spikes (rate ``mean_spikes/T``). Setting
      ``mean_spikes`` proportional to K keeps the number of afferents active per PSP-width
      (~``2 N mean_spikes / K``) constant as the window grows -- the "constant temporal density" mode.
    - ``"single"`` (Gutig & Sompolinsky 2006, the latency code, Figs 3a/4): **exactly one** spike
      per afferent at a time drawn iid ``U[0, T]``. This is the ensemble whose capacity is
      ``alpha_c ~ 3``.
    - ``"synchrony_half"`` (GS 2006 Fig 4a gray curve, the rate-coding control): in each pattern a
      random half of afferents fire **synchronously** at one common time (the other half are
      silent). Spike *timing* carries no information -> the tempotron rule reduces to the perceptron
      rule and ``alpha_c -> 2``.

    Labels are +/-1 with equal probability. Returns ``(spike_times, valid, labels, max_spikes)`` with
    ``spike_times, valid`` of shape ``(n_tasks, P, N, max_spikes)`` and ``labels`` of ``(n_tasks, P)``.
    """
    # Draw on the CPU via a numpy Generator, then move to ``device``. PyTorch's RNG is *per-device*
    # (CUDA and CPU streams differ for the same seed), so a numpy source makes the patterns
    # device-INDEPENDENT -> the run is bit-reproducible on any backend and the saved weights can be
    # re-verified on CPU (G_capture reweight). The draw order is fixed for reproducibility.
    shape = (n_tasks, n_patterns, n_aff)
    if ensemble == "single":
        # Exactly one spike per afferent, time ~ U[0, T] (GS 2006 latency ensemble).
        max_spikes = 1
        spike_times = (t_window * rng.random((n_tasks, n_patterns, n_aff, 1))).astype(np.float32)
        valid = np.ones((n_tasks, n_patterns, n_aff, 1), dtype=np.float32)
    elif ensemble == "synchrony_half":
        # Random half of afferents fire at one common per-pattern time; rest silent (perceptron).
        max_spikes = 1
        t_common = (t_window * rng.random((n_tasks, n_patterns, 1, 1))).astype(np.float32)
        spike_times = np.broadcast_to(t_common, (n_tasks, n_patterns, n_aff, 1)).copy()
        active = (rng.random(shape) < 0.5).astype(np.float32)
        valid = active[..., None]
    else:  # "poisson" (RMS 2010)
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
    elem_budget: int = 64_000_000,
) -> torch.Tensor:
    """Per-afferent traces ``s[b,g,p,i] = sum_f K(t_g - spike[b,p,i,f]) * valid``.

    Built by chunking the time grid so the working kernel tensor
    ``(g_chunk, Sb, P, N, max_spikes)`` stays within ``elem_budget`` elements.
    Returns ``s`` of shape ``(Sb, P, G, N)`` (float32) -- the per-pattern slice ``s[:, p]`` is
    contiguous, which the online sweep relies on.
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
        tt = t_grid[g0:g1].view(-1, 1, 1, 1, 1)             # (gc,1,1,1,1)
        contrib = psp_kernel(tt - st) * vd                   # (gc, Sb, P, N, ms)
        s[:, :, g0:g1, :] = contrib.sum(dim=4).permute(1, 2, 0, 3)  # (gc,Sb,P,N)->(Sb,P,gc,N)
    return s


# --------------------------------------------------------------------------------------------
# Forward pass + Gutig-Sompolinsky training (online by default; batch-mode optional)
# --------------------------------------------------------------------------------------------
def _forward(s: torch.Tensor, w: torch.Tensor,
             tfixed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``V[b,p,g] = sum_i s[b,p,g,i] w[b,i]``; return ``(vmax, argmax_t)`` each ``(Sb, P)``.

    ``tfixed`` (Sb,P): if given, read out at this fixed time per pattern instead of argmax_t V
    (the drive-peak readout for the GS-2006 synchrony control; see ``readout`` in ``train_batch``).
    """
    v = torch.einsum("bpgn,bn->bpg", s, w)
    if tfixed is None:
        vmax, targ = v.max(dim=2)
        return vmax, targ
    vmax = v.gather(2, tfixed.unsqueeze(2)).squeeze(2)
    return vmax, tfixed


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

    Extracted as a module function so the schedule is unit-testable in isolation
    (``tests/test_v10_minibatch.py``, ``tests/test_v11_schedule.py``); ``train_batch`` calls it once
    per epoch.
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
    mode: str = "online",
    uth_scale: float = 1.0,       # V5 gauge knob: multiply the calibrated U_th
    kappa_target: float = 0.0,    # V5 margin band: require signed margin >= kappa_target*|U_th|
    noise_sigmas: tuple[float, ...] = (),  # V6 output-noise levels for the robust-capacity probe
    vthr_fixed: float | None = None,  # V8 faithful-GS: fix V_thr (=1) and rescale init weights to it
    rescale_init: bool = True,    # T2-A: if False, keep w_init as-is (no median rescale) -- the GS
                                  # Suppl-Methods sigma_w=1e-3 faithful init that starts silent
    readout: str = "vmax",  # 'vmax' (tempotron argmax_t V) | 'drive_peak' (GS synchrony control)
    log_every: int = 0,    # V8: if >0, print converged-fraction every log_every epochs (live monitor)
    patience: int = 0,     # V8: if >0, stop a cell after this many epochs with no new convergence
    log_tag: str = "",
    record_history: bool = False,  # V8: record per-epoch mean/std error & GS-cost across the seed batch
    metric_cb=None,  # V8: callback(epoch, err_mean, loss_mean) -> stream learning curve to metrics.jsonl
    capture: bool = False,    # V9: record FULL per-seed per-epoch trajectories (raw store)
    patience_min_delta: float = 0.0,  # V9: a loss drop below best-min_delta counts as "improvement"
    optimizer: str = "momentum",  # V9 batch mode: momentum | adam | rmsprop (findability probe)
    lr_schedule: str = "none",    # V9 batch mode: none | cosine (decay lr -> 0 over the budget)
    lr_warmup: int = 0,           # V9 batch mode: linear lr warmup over the first this-many epochs
    lr_floor: float = 0.0,        # V11: anneal cosine/linear to lr_floor*lr instead of 0
    lr_cycles: int = 1,           # V11: SGDR warm-restart cycle count for the cosine schedule
    lr_exp_k: float = EXP_DECAY_K,  # V11: exp-schedule decay rate (smaller = gentler/more exploration)
    # --- V10 unified minibatch path (mode='minibatch'): tuned optimizer/schedule/batch HPs ---
    batch_size: int = 0,          # minibatch size b (1=online .. P=full); required for mode='minibatch'
    lr_step_size: int = 0,        # 'step' schedule: epochs between gamma-decays (tuned)
    lr_gamma: float = 0.1,        # 'step' schedule: multiplicative decay factor (tuned)
    adam_betas: tuple[float, float] = (0.9, 0.999),  # tuned Adam (beta1, beta2)
    adam_eps: float = 1e-8,       # tuned Adam epsilon
    rms_alpha: float = 0.99,      # tuned RMSprop smoothing rho
    rms_eps: float = 1e-8,        # tuned RMSprop epsilon
    report_epochs: tuple[int, ...] = (),  # Hyperband rungs: epochs at which to report p_solve
    report_cb=None,               # callback(epoch:int, p_solve:float)->bool; True => prune & stop
    valid_mask: torch.Tensor | None = None,  # (Sb,P) 1/0: padded (0) slots are inert (minibatch only)
) -> dict[str, torch.Tensor]:
    """Gutig-Sompolinsky rule, vectorised across tasks. Threshold fixed (median V_max at init).

    ``mode='online'`` (default, what GS/RMS used): per-pattern updates within a shuffled epoch; a
    task is converged once a full epoch passes with **zero updates** (the textbook perceptron
    criterion), detected without an extra forward pass. With ``momentum>0`` the online path chains a
    velocity across **error trials** (``dw_current = dw + momentum*dw_prev``, GS 2006 Methods; the
    effective step is amplified up to ``1/(1-momentum)`` when the direction is consistent).
    ``mode='batch'`` sums the misclassified updates and applies them once per epoch (vectorised but
    oscillation-prone near capacity; kept for study).

    Gauge of the threshold:
    - ``vthr_fixed=None`` (default): if ``threshold`` is given use it, else **calibrate** U_th to the
      median V_max at init (the gauge-preserving convention; weights init ~ N(0,1)).
    - ``vthr_fixed=v`` (faithful GS): hold ``V_thr = v`` (=1) and rescale the init weights so the
      median V_max equals ``v`` (a balanced P(fire)=1/2 start at the *fixed* threshold). By the
      U_th<->||w|| gauge invariance this measures the same capacity as the calibrate convention.

    Returns per-task tensors: ``converged``, ``epochs_run``, ``wnorm``, ``mean_margin``,
    ``threshold``, ``init_fire_rate``, ``final_errfrac``.
    """
    sb, p, _, n = s.shape
    dev = s.device
    w = w_init.clone()
    # Drive-peak readout (GS-2006 synchrony control): read out at the fixed per-pattern time of
    # maximal *total* PSP drive (argmax_t sum_i s_i(t)) instead of argmax_t V(t). For a coincident
    # volley this is the volley peak, so the gradient s(t*) never collapses into the causal
    # dead-zone (the V_max=0 atom) that makes the bare rule blind to half the +patterns. This
    # makes the timing-removed ensemble a faithful perceptron; for the generic ensembles the
    # tempotron's argmax_t V readout is the intended one, so this is opt-in (online mode only).
    tdrive = s.sum(dim=3).argmax(dim=2) if readout == "drive_peak" else None  # (Sb,P) or None
    vmax, _ = _forward(s, w, tdrive)

    def _vmax_median(v: torch.Tensor) -> torch.Tensor:
        """Per-seed median V_max over *valid* patterns (mask=None => over all P)."""
        if valid_mask is None:
            return v.median(dim=1).values
        return torch.stack([v[i][valid_mask[i] > 0].median() for i in range(sb)])

    def _vmax_balance(v: torch.Tensor) -> torch.Tensor:
        """Per-seed init-rescale target for the fixed-V_thr gauge.

        Default: the median V_max (the P(fire)=1/2 operating point). But the median is
        ill-posed for a *degenerate* V_max distribution. In ``synchrony_half`` every active
        afferent of a pattern fires at one coincident time, so V(t)=S*K(t-t_common) with a
        single scalar S=sum_active w_i; the causal PSP kernel makes V_max>=0, so the ~half of
        patterns with S<=0 collapse to an atom at V_max=0. The median then sits on that atom
        (~0) and rescaling ``vthr/median`` drives ||w||->~1e12 (the J3/J3b divergence). Detect
        the atom (median << upper-quantile scale) and rescale on the median of the strictly
        -positive V_max (its continuous support) instead. For the validated continuous ensembles
        (single/poisson) the median is well above 0, so the guard never fires and their
        calibration is byte-for-byte unchanged.
        """
        out = torch.empty(sb, device=dev, dtype=v.dtype)
        for i in range(sb):
            vi = v[i] if valid_mask is None else v[i][valid_mask[i] > 0]
            med_i = vi.median()
            q_hi = torch.quantile(vi, 0.75)
            if bool(med_i <= 1e-3 * q_hi):  # atom at ~0 dominates the lower half
                pos = vi[vi > 0]
                out[i] = pos.median() if pos.numel() > 0 else med_i
            else:
                out[i] = med_i
        return out

    if vthr_fixed is not None:
        # Fixed threshold V_thr. By default (rescale_init=True) rescale init weights so a balanced
        # fraction fires (median V_max -> V_thr; guarded for degenerate ensembles, see _vmax_balance).
        # rescale_init=False is the GS Suppl-Methods faithful init: keep the sigma_w=1e-3 weights
        # as-is (no rescale), so the net starts essentially silent at the fixed threshold.
        if rescale_init:
            med = _vmax_balance(vmax).clamp(min=1e-12)  # (Sb,)
            w = w * (vthr_fixed / med)[:, None]
            vmax, _ = _forward(s, w, tdrive)
        threshold = torch.full((sb,), float(vthr_fixed), device=dev)
    elif threshold is None:
        threshold = _vmax_median(vmax)  # (Sb,)
    threshold = threshold * uth_scale  # V5 gauge: rescale the (fixed) threshold
    band = kappa_target * threshold.abs()  # required signed-margin band (0 => bare zero-error rule)
    if valid_mask is None:
        init_fire_rate = (vmax >= threshold[:, None]).float().mean(dim=1)
    else:
        fired = (vmax >= threshold[:, None]).float() * valid_mask
        init_fire_rate = fired.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
    converged = torch.zeros(sb, dtype=torch.bool, device=dev)
    epochs_run = torch.full((sb,), epochs, dtype=torch.int64, device=dev)
    arange_sb = torch.arange(sb, device=dev)
    vel = torch.zeros_like(w)
    best_conv = 0           # most tasks converged so far (for the no-progress early stop)
    stall = 0               # epochs since best_conv last improved
    history: list[tuple[int, float, float, float, float]] = []  # (epoch, err_mean, err_std, cost_mean, cost_std)

    # V9 raw capture: full per-seed per-epoch trajectories (online mode only). NaN-padded to the
    # budget; sliced to the actual epochs run on return. Everything downstream is derived offline
    # from these arrays, so the run computes/discards no summary scalar that an analysis might want.
    cap = capture and mode in ("online", "batch", "minibatch")
    last_ep = -1
    best_loss = float("inf")
    # V9 batch-mode optimizer state (Adam/RMSprop moments) + a no-op for online.
    adam_m = torch.zeros_like(w)
    adam_v = torch.zeros_like(w)
    adam_step = 0           # V10 minibatch: global Adam step counter for bias correction
    pruned = False          # V10: set True if a Hyperband report_cb requested a prune
    report_set = set(int(e) for e in report_epochs)  # V10: rung epochs (epochs-run = ep+1)

    def _lr_at(ep: int) -> float:
        """Scheduled learning rate for this epoch (V10/V11 scheduled_lr)."""
        return scheduled_lr(lr, ep, epochs=epochs, schedule=lr_schedule, warmup=lr_warmup,
                            step_size=lr_step_size, gamma=lr_gamma,
                            floor=lr_floor, n_cycles=lr_cycles, exp_k=lr_exp_k)

    if cap:
        err_buf = torch.full((sb, epochs), float("nan"), device=dev)
        loss_buf = torch.full((sb, epochs), float("nan"), device=dev)
        wnorm_buf = torch.full((sb, epochs), float("nan"), device=dev)
        nupd_buf = torch.full((sb, epochs), float("nan"), device=dev)
        kp_buf = torch.full((sb, epochs), float("nan"), device=dev)
        km_buf = torch.full((sb, epochs), float("nan"), device=dev)
        npos = (labels > 0).float().sum(dim=1).clamp(min=1.0)  # (Sb,) per-seed #(+1) patterns
        nneg = (labels < 0).float().sum(dim=1).clamp(min=1.0)  # (Sb,) per-seed #(-1) patterns

    for ep in range(epochs):
        active = (~converged).float().unsqueeze(-1)  # (Sb, 1)
        if mode == "online":
            lr_ep = _lr_at(ep)  # V9: per-epoch scheduled LR (== lr when lr_schedule='none')
            order = torch.randperm(p, device=dev)
            updates = torch.zeros(sb, device=dev)  # #updates this epoch (per task)
            err_acc = torch.zeros(sb, device=dev)   # raw training errors this epoch (all seeds, ungated)
            cost_acc = torch.zeros(sb, device=dev)  # GS hinge cost relu(band - signed_margin)
            kp_acc = torch.zeros(sb, device=dev)    # V9: running sum of signed margin on +1 patterns
            km_acc = torch.zeros(sb, device=dev)    # V9: running sum of signed margin on -1 patterns
            for pi in order:
                sp = s[:, pi]                               # (Sb, G, N) contiguous
                vp = torch.einsum("bgn,bn->bg", sp, w)
                if tdrive is None:
                    vmx, tg = vp.max(dim=1)                 # (Sb,) tempotron argmax_t V
                else:
                    tg = tdrive[:, pi]                      # fixed drive-peak readout
                    vmx = vp[arange_sb, tg]
                # update when the signed margin is below the band (band=0 => bare zero-error rule)
                smarg = (vmx - threshold) * labels[:, pi]
                if record_history or cap:
                    err_acc = err_acc + (smarg < band).float()
                    cost_acc = cost_acc + torch.relu(band - smarg)
                if cap:
                    lab_pi = labels[:, pi]
                    kp_acc = kp_acc + smarg * (lab_pi > 0).float()
                    km_acc = km_acc + smarg * (lab_pi < 0).float()
                e = (smarg < band).float() * active.squeeze(-1)
                updates = updates + e
                grad = sp[arange_sb, tg]                    # (Sb, N) = s_i(t_max)
                corr = (lr_ep * labels[:, pi]).unsqueeze(-1) * grad  # GS correction dw = lr*y*s(t_max)
                e_col = e.unsqueeze(-1)
                # momentum (GS 2006): on error trials vel = corr + momentum*vel, w += vel; on correct
                # trials carry the velocity and leave w unchanged. momentum=0 => bare online rule.
                vel = e_col * (corr + momentum * vel) + (1.0 - e_col) * vel
                w = w + e_col * vel
            if record_history:
                ef = err_acc / p   # per-seed training-error fraction this epoch (all seeds)
                cs = cost_acc / p  # per-seed mean GS hinge cost (threshold units)
                history.append((ep, float(ef.mean()), float(ef.std()),
                                float(cs.mean()), float(cs.std())))
            if cap:
                ep_err, ep_loss, ep_nupd = err_acc / p, cost_acc / p, updates
                ep_kp, ep_km = kp_acc / npos, km_acc / nneg
            newly = (updates == 0) & (~converged)
        elif mode == "minibatch":  # V10 unified path: b in {1..P}, tuned optimizer / schedule / freeze
            # One shuffled pass split into ceil(P/b) minibatch steps (Sec 6.1). The GS hinge subgradient
            # is the per-minibatch MEAN over margin-violators, g = (1/b) sum_{viol} y s(t*); each
            # optimizer ascends +g. A step is gated per-seed by (still active) AND (has a violator in
            # this minibatch): at b=1, constant lambda, mu=0.99 this reproduces the faithful online GS
            # rule bit-for-bit (the velocity advances only on error steps), and at b=P it matches the
            # full-batch freeze gate. Convergence = a full epoch with zero violators encountered.
            lr_ep = _lr_at(ep)
            order = torch.randperm(p, device=dev)
            bs = max(1, min(batch_size, p))
            n_steps = (p + bs - 1) // bs
            act = active.squeeze(-1)                          # (Sb,) 1.0 while unconverged
            viol_enc = torch.zeros(sb, device=dev)            # violators seen by active seeds -> converge
            if record_history or cap:
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
                viol = (smarg < band[:, None]).float()         # (Sb, b) margin-violating patterns
                if valid_mask is not None:
                    vmask_b = valid_mask[:, bidx]              # (Sb, b): 0 => padded/inert slot
                    viol = viol * vmask_b                       # padded slots are never violators
                    nb = vmask_b.sum(dim=1).clamp(min=1.0)      # (Sb,) valid patterns in this minibatch
                else:
                    nb = float(bb)
                viol_enc = viol_enc + viol.sum(dim=1) * act
                gidx = tg.view(sb, bb, 1, 1).expand(sb, bb, 1, n)
                gpat = torch.gather(sb_s, 2, gidx).squeeze(2)  # (Sb, b, N) = s_i(t*) per pattern
                g = ((lab_b * viol).unsqueeze(-1) * gpat).sum(dim=1)  # (Sb, N) GS ascent over violators
                g = g / nb if valid_mask is None else g / nb[:, None]   # mean over valid minibatch
                has_viol = (viol.sum(dim=1) > 0).float().unsqueeze(-1)     # (Sb,1) does this seed update?
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
                if record_history or cap:
                    err_acc = err_acc + viol.sum(dim=1)
                    cost_acc = cost_acc + torch.relu(band[:, None] - smarg).sum(dim=1)
                if cap:
                    kp_acc = kp_acc + (smarg * (lab_b > 0).float()).sum(dim=1)
                    km_acc = km_acc + (smarg * (lab_b < 0).float()).sum(dim=1)
            if cap:
                ep_err, ep_loss, ep_nupd = err_acc / p, cost_acc / p, viol_enc
                ep_kp, ep_km = kp_acc / npos, km_acc / nneg
            # A diverged seed (NaN/inf weights from too-large an LR) has NaN margins, and NaN<band is
            # False -> it would look like "zero violators". Require finite weights to count as solved,
            # so divergent HP configs report low p_solve and get pruned (not spuriously "converged").
            newly = (viol_enc == 0) & (~converged) & torch.isfinite(w).all(dim=1)
        else:  # batch mode -- full-gradient step with a selectable optimizer (V9 findability probe)
            vmax, targ = _forward(s, w)
            signed = (vmax - threshold[:, None]) * labels    # (Sb, P) signed margin
            err = (signed < band[:, None]).float()           # violating / misclassified patterns
            idx = targ.view(sb, p, 1, 1).expand(sb, p, 1, n)
            grad = torch.gather(s, 2, idx).squeeze(2)        # (Sb, P, N) = s_i(t_max) per pattern
            g = ((err * labels).unsqueeze(-1) * grad).sum(dim=1)  # raw GS ascent gradient (Sb, N)
            lr_ep = _lr_at(ep)
            if optimizer == "adam":
                b1, b2, eps = 0.9, 0.999, 1e-8
                adam_m.mul_(b1).add_(g, alpha=1 - b1)
                adam_v.mul_(b2).addcmul_(g, g, value=1 - b2)
                t = ep + 1
                mhat = adam_m / (1 - b1 ** t)
                vhat = adam_v / (1 - b2 ** t)
                step = lr_ep * mhat / (vhat.sqrt() + eps)
            elif optimizer == "rmsprop":
                a, eps = 0.99, 1e-8
                adam_v.mul_(a).addcmul_(g, g, value=1 - a)
                step = lr_ep * g / (adam_v.sqrt() + eps)
            else:  # "momentum" (the original batch rule: vel = mom*vel + g, step = lr*vel)
                vel = momentum * vel + g
                step = lr_ep * vel
            # Freeze a seed the instant it is error-free: with g=0 the momentum/Adam state still has
            # inertia and would drift the weights back OFF the solution. Gate the step by "still has
            # errors" (and by not-yet-converged) so a found solution stays put.
            still = (err.sum(dim=1) > 0).float().unsqueeze(-1)
            w = w + step * active * still
            if cap:
                ep_err, ep_loss, ep_nupd = err.mean(dim=1), torch.relu(band[:, None] - signed).mean(dim=1), err.sum(dim=1)
                ep_kp = (signed * (labels > 0).float()).sum(dim=1) / npos
                ep_km = (signed * (labels < 0).float()).sum(dim=1) / nneg
            newly = (err.sum(dim=1) == 0) & (~converged)
        # Record epochs *executed* to converge = ep + 1 (epochs 0..ep ran). Matches the reference
        # package's epoch+1 convention and the unconverged ``last_ep + 1`` count below, so converged
        # and unconverged learning times share one scale for GS Fig.4a comparisons (audit fix).
        epochs_run = torch.where(newly, torch.tensor(ep + 1, device=dev), epochs_run)
        converged = converged | newly
        last_ep = ep
        if cap:
            # Record this epoch's per-seed trajectory from the cheap per-epoch reductions (no extra
            # forward pass). ``ep_*`` are set by whichever update path (online/batch) ran above.
            err_buf[:, ep] = ep_err
            loss_buf[:, ep] = ep_loss
            wnorm_buf[:, ep] = w.norm(dim=1)
            nupd_buf[:, ep] = ep_nupd
            kp_buf[:, ep] = ep_kp
            km_buf[:, ep] = ep_km
        # V10 Hyperband rung hook: report p_solve at the requested epochs (epochs-run = ep+1) and
        # prune (stop training) the instant the callback says so -- the early-stopping that makes the
        # FTE budget go further. Reported p_solve is the fraction of seeds converged so far.
        if report_set and report_cb is not None and (ep + 1) in report_set:
            p_solve_now = float(converged.float().mean().item())
            if bool(report_cb(ep + 1, p_solve_now)):
                pruned = True
                break
        if bool(converged.all()):
            break
        if cap:
            # V9 patience: "no improvement" = neither the seed-averaged GS loss dropped below its
            # running best (by more than patience_min_delta) NOR a new seed converged, for `patience`
            # consecutive epochs. Tracking the *continuous* loss keeps a still-descending
            # (budget-limited) cell running; also resetting on any new convergence protects a slow
            # straggler from being cut when a few permanently-stuck seeds plateau the average.
            n_conv = int(converged.sum().item())
            cur_loss = float(ep_loss.mean().item())
            # "improvement" is a RELATIVE drop below the running-best loss (patience_min_delta as a
            # fraction), so the criterion is scale-free across the orders of magnitude the loss spans
            # from far-below to near capacity, and robust to mu=0.99 jitter.
            improved = cur_loss < best_loss * (1.0 - patience_min_delta)
            if cur_loss < best_loss:
                best_loss = cur_loss
            if n_conv > best_conv:
                best_conv = n_conv
            if improved or bool(newly.any()):
                stall = 0
            else:
                stall += 1
            if log_every and (ep + 1) % log_every == 0:
                err_all = float(ep_err.mean().item())
                if metric_cb is not None:                      # stream the learning curve, death-proof
                    metric_cb(ep + 1, err_all, cur_loss)
                print(f"    [{log_tag}] ep={ep + 1} conv={n_conv}/{sb} err={err_all:.3f} "
                      f"loss={cur_loss:.4f} stall={stall}", flush=True)
            if patience and stall >= patience and n_conv < sb:
                if log_every:
                    print(f"    [{log_tag}] early-stop at ep={ep + 1} "
                          f"(loss flat for {stall} epochs)", flush=True)
                break
        # live monitor + no-progress early stop (only meaningful for large epoch budgets)
        elif (log_every and (ep + 1) % log_every == 0) or patience:
            n_conv = int(converged.sum().item())
            if n_conv > best_conv:
                best_conv = n_conv
                stall = 0
            else:
                stall += 1
            if log_every and (ep + 1) % log_every == 0:
                vtmp, _ = _forward(s, w)                        # (Sb, P) per-pattern V_max
                rerr = ((torch.where(vtmp >= threshold[:, None], 1.0, -1.0) != labels).float()
                        .mean(dim=1))                          # per-seed error fraction
                rmean = float((rerr * (~converged).float()).sum().item() / max(1, sb - n_conv))
                err_all = float(rerr.mean())                   # seed-averaged training error (all seeds)
                signed = (vtmp - threshold[:, None]) * labels  # GS hinge loss relu(band - margin)
                loss_all = float(torch.relu(band[:, None] - signed).mean())
                if metric_cb is not None:                      # stream the learning curve, death-proof
                    metric_cb(ep + 1, err_all, loss_all)
                print(f"    [{log_tag}] ep={ep + 1} conv={n_conv}/{sb} err={err_all:.3f} "
                      f"loss={loss_all:.4f} resid_unconv={rmean:.3f} stall={stall}", flush=True)
            if patience and stall >= patience and n_conv < sb:
                if log_every:
                    print(f"    [{log_tag}] early-stop at ep={ep + 1} "
                          f"(no new convergence for {stall} epochs)", flush=True)
                break

    if mode == "minibatch":
        # For still-unconverged seeds, record the epochs actually executed (last_ep+1) rather than the
        # full budget, so the HPO driver's FTE accounting (sum of seed-epochs) is exact even when the
        # run was pruned or all-converged early. Converged seeds keep their convergence epoch.
        ran = last_ep + 1
        epochs_run = torch.where(converged, epochs_run,
                                 torch.full_like(epochs_run, ran))

    vmax, _ = _forward(s, w)
    pred = torch.where(vmax >= threshold[:, None], 1.0, -1.0)
    final_errfrac = (pred != labels).float().mean(dim=1)  # per-task residual error fraction
    signed_margin = (vmax - threshold[:, None]) * labels  # (Sb,P); >0 when correct
    margin = signed_margin.mean(dim=1)
    pos = (labels > 0).float()
    neg = (labels < 0).float()
    kappa_plus = (signed_margin * pos).sum(dim=1) / pos.sum(dim=1).clamp(min=1.0)
    kappa_minus = (signed_margin * neg).sum(dim=1) / neg.sum(dim=1).clamp(min=1.0)
    # V6 output-noise robustness: per-task mean error when the decision variable is perturbed by
    # sigma*U_th Gaussian noise (Rubin 2017 output-noise model), averaged over draws.
    noisy_err = {}
    if noise_sigmas:
        ng = torch.Generator(device=dev); ng.manual_seed(12345)
        for sg in noise_sigmas:
            acc = torch.zeros(sb, device=dev)
            for _ in range(4):
                vn = vmax + sg * threshold[:, None] * torch.randn(vmax.shape, generator=ng, device=dev)
                acc += (torch.where(vn >= threshold[:, None], 1.0, -1.0) != labels).float().mean(dim=1)
            noisy_err[sg] = acc / 4.0
    out = {
        "converged": converged,
        "epochs_run": epochs_run,
        "wnorm": w.norm(dim=1),
        "mean_margin": margin,
        "kappa_plus": kappa_plus,    # mean signed margin on target (+1) patterns
        "kappa_minus": kappa_minus,  # mean signed margin on null (-1) patterns
        "weights": w,                # final weights (for robustness probe)
        "threshold": threshold,
        "init_fire_rate": init_fire_rate,
        "final_errfrac": final_errfrac,
        "noisy_err": noisy_err,
        "history": history,  # per-epoch (epoch, err_mean, err_std, cost_mean, cost_std) if recorded
        "pruned": pruned,    # V10: True if a Hyperband report_cb requested an early prune
    }
    if cap:
        # Slice the NaN-padded buffers to the epochs actually run for this batch (last_ep+1). All
        # downstream numerics/figures are derived offline from these per-seed per-epoch arrays.
        ne = last_ep + 1
        out["cap"] = {
            "traj_err": err_buf[:, :ne],     # (Sb, ne) per-seed training-error fraction vs epoch
            "traj_loss": loss_buf[:, :ne],   # (Sb, ne) per-seed GS hinge loss vs epoch
            "traj_wnorm": wnorm_buf[:, :ne],  # (Sb, ne) per-seed ||w|| vs epoch
            "traj_nupd": nupd_buf[:, :ne],   # (Sb, ne) per-seed #weight-updates vs epoch
            "traj_kp": kp_buf[:, :ne],       # (Sb, ne) per-seed running kappa_plus vs epoch
            "traj_km": km_buf[:, :ne],       # (Sb, ne) per-seed running kappa_minus vs epoch
            "n_epochs": ne,                  # epochs this batch ran (== budget unless early-stopped)
        }
    return out


def mean_pairwise_overlap(weights: list[torch.Tensor], convs: list[torch.Tensor]) -> torch.Tensor:
    """Mean |cosine overlap| among the *converged* multi-restart solutions, per task (V7).

    ``weights[r]``/``convs[r]`` are ``(Sb,N)``/``(Sb,)`` for restart r. Returns ``(Sb,)``: the mean of
    ``|w_a . w_b| / (||w_a|| ||w_b||)`` over restart pairs (a<b) that both converged (NaN if <2). q≈0
    means independent runs land in near-orthogonal clusters (RMS Fig.3 shattering, q0≈0); q≈1 means
    the same region.
    """
    W = torch.stack(weights)                      # (R, Sb, N)
    C = torch.stack(convs).float()                # (R, Sb)
    Wn = W / W.norm(dim=2, keepdim=True).clamp(min=1e-12)
    gram = torch.einsum("rbn,sbn->rsb", Wn, Wn).abs()   # (R, R, Sb)
    r = W.shape[0]
    triu = torch.triu(torch.ones(r, r, device=W.device), diagonal=1).unsqueeze(-1)
    pair = C.unsqueeze(1) * C.unsqueeze(0) * triu        # (R, R, Sb) both-converged upper pairs
    den = pair.sum(dim=(0, 1))
    out = (gram * pair).sum(dim=(0, 1)) / den.clamp(min=1.0)
    out[den < 1] = float("nan")
    return out


def robustness_radius(
    s: torch.Tensor,          # (Sb, P, G, N)
    w: torch.Tensor,          # (Sb, N) converged weights
    threshold: torch.Tensor,  # (Sb,)
    labels: torch.Tensor,     # (Sb, P)
    rng: np.random.Generator,
    *,
    n_levels: int = 10,
    max_eta: float = 0.6,
    n_draws: int = 3,
    err_tol: float = 0.05,
) -> torch.Tensor:
    """Local-entropy / robustness proxy (Baldassi): the relative weight-noise level eta at which a
    converged solution first exceeds ``err_tol`` training error.

    For each level eta in (0, max_eta], perturb ``w -> w + eta*(||w||/sqrt N)*xi`` (xi ~ N(0,I)),
    averaged over ``n_draws``, and record the smallest eta whose mean error exceeds ``err_tol``
    (``max_eta`` if never). Larger radius => the solution sits in a wider, denser basin.
    """
    sb, _, _, n = s.shape
    scale = (w.norm(dim=1, keepdim=True) / math.sqrt(n))  # (Sb,1)
    radius = torch.full((sb,), float(max_eta), device=s.device)
    for li in range(1, n_levels + 1):
        eta = max_eta * li / n_levels
        err_acc = torch.zeros(sb, device=s.device)
        for _ in range(n_draws):
            xi = torch.from_numpy(rng.standard_normal((sb, n), dtype=np.float32)).to(s.device)
            vmax, _ = _forward(s, w + eta * scale * xi)
            pred = torch.where(vmax >= threshold[:, None], 1.0, -1.0)
            err_acc += (pred != labels).float().mean(dim=1)
        err = err_acc / n_draws
        newly = (err > err_tol) & (radius >= max_eta - 1e-9)
        radius = torch.where(newly, torch.tensor(eta, device=s.device), radius)
    return radius


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------
def _alpha_grid(ov: dict[str, str]) -> list[float]:
    if "alphas" in ov:
        return [float(x) for x in ov["alphas"].split(",")]
    a0 = float(ov.get("alpha_min", "0.5"))
    a1 = float(ov.get("alpha_max", "5.0"))
    step = float(ov.get("alpha_step", "0.5"))
    n = int(round((a1 - a0) / step)) + 1
    return [round(a0 + i * step, 4) for i in range(n)]


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))

    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    k_list = [float(x) for x in ov.get("K_list", "16,64,256,1024").split(",")]
    n_list = [int(x) for x in ov.get("N_list", "100,200").split(",")]
    alphas = _alpha_grid(ov)
    n_seeds = int(ov.get("n_seeds", "100"))
    epochs = int(ov.get("epochs", "2000"))
    lr = float(ov.get("lr", "0.05"))
    momentum = float(ov.get("momentum", "0.0"))
    mode = ov.get("mode", "online")  # 'online' (RMS rule) or 'batch'
    # Input firing density. Default mean_spikes=1 (RMS rate-1/T model). If spikes_per_K>0, the
    # per-afferent mean is spikes_per_K*K (constant temporal density -> tests the lnlnK lift without
    # finite-N starvation; see the design doc).
    mean_spikes_fixed = float(ov.get("mean_spikes", "1.0"))
    spikes_per_K = float(ov.get("spikes_per_K", "0"))
    grid_per_corr = int(ov.get("grid_per_corr", "8"))   # time-grid points per sqrt(tau_s tau_m)
    # V8 faithful Gutig-Sompolinsky 2006 knobs (defaults preserve the RMS V3 behaviour exactly):
    ensemble = ov.get("ensemble", "poisson")  # 'poisson'|'single' (GS latency)|'synchrony_half'
    lr_mode = ov.get("lr_mode", "fixed")      # 'fixed' (use lr) | 'gs' (lambda = coeff*T/(tau_m N V0))
    lr_gs_coeff = float(ov.get("lr_gs_coeff", "3e-3"))  # GS Fig-4 capacity schedule coefficient
    vthr_fixed_ov = ov.get("vthr_fixed", "")  # faithful GS fixes V_thr (e.g. 1.0); '' => calibrate U_th
    vthr_fixed = float(vthr_fixed_ov) if vthr_fixed_ov else None
    readout = ov.get("readout", "vmax")  # 'vmax' | 'drive_peak' (GS-2006 synchrony control)
    sigma_w = float(ov.get("sigma_w", "1.0"))  # T2-A faithful GS: init efficacy std (GS Suppl: 1e-3)
    rescale_init = ov.get("rescale_init", "true").lower() != "false"  # T2-A: False keeps w0 as-is (no rescale)
    # Live learning-curve logging cadence (epochs). Default ON (~50 points/cell) so EVERY run streams
    # its training error+loss to metrics.jsonl -- captured live and surviving a teardown/rsync failure.
    log_every = int(ov.get("log_every", "0"))
    patience = int(ov.get("patience", "0"))     # early-stop a cell after this many no-progress epochs
    history_flag = int(ov.get("history", "0"))  # also write the fine-grained per-epoch CSV per cell
    # V9 raw capture: write FULL per-seed per-epoch trajectories + final weights per cell to .npz, so
    # every numeric/figure is derived offline (analysis/v9_*.py) without rerunning the GPU. Online only.
    capture = int(ov.get("capture", "0"))
    patience_min_delta = float(ov.get("patience_min_delta", "0.0"))
    # V9 findability probe: in batch mode, swap the optimizer / add an LR schedule to test whether the
    # near-capacity plateau of the bare GS rule is an OPTIMIZATION limit (vs a true capacity limit).
    optimizer = ov.get("optimizer", "momentum")  # momentum | adam | rmsprop (batch/minibatch mode)
    lr_schedule = ov.get("lr_schedule", "none")  # none | cosine | step
    lr_warmup = int(ov.get("lr_warmup", "0"))
    # V10 minibatch knobs (mode='minibatch'): tuned batch size b + optimizer/schedule HPs, so a winning
    # HPO config can be re-run for the confirmation study (Sec 14.5) straight from the CLI.
    batch_size = int(ov.get("batch_size", "0"))   # b in {1..P}; required when mode='minibatch'
    lr_step_size = int(ov.get("lr_step_size", "0"))
    lr_gamma = float(ov.get("lr_gamma", "0.1"))
    adam_b1 = float(ov.get("adam_b1", "0.9"))
    adam_b2 = float(ov.get("adam_b2", "0.999"))
    adam_eps = float(ov.get("adam_eps", "1e-8"))
    rms_alpha = float(ov.get("rms_alpha", "0.99"))
    rms_eps = float(ov.get("rms_eps", "1e-8"))
    if log_every == 0:
        log_every = max(1, epochs // 50)        # default: ~50 learning-curve points per cell
    anchor_k = {float(x) for x in ov.get("anchor_K", "").split(",") if x}
    n_restarts = int(ov.get("n_restarts", "10"))
    robust_probe = int(ov.get("robust_probe", "0"))  # V4: perturbation-robustness of solutions
    overlap_probe = int(ov.get("overlap_probe", "0"))  # V7: pairwise overlap of multi-restart solutions
    uth_scale = float(ov.get("uth_scale", "1.0"))     # V5 gauge knob
    kappa_target = float(ov.get("kappa_target", "0.0"))  # V5 margin-band target
    noise_sigmas = tuple(float(x) for x in ov.get("noise_sigmas", "").split(",") if x)  # V6
    noise_tol = float(ov.get("noise_tol", "0.05"))  # V6: max noisy error for a task to count robust
    elem_budget = int(float(ov.get("elem_budget", "64e6")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Trace-tensor memory budget: adapt to the actual GPU (Vast hands out anything from a 24 GB
    # RTX 4090 to a 96 GB Blackwell). In online mode the trace tensor is the dominant allocation,
    # so 0.55 x total VRAM is safe; overridable via s_budget=.
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
        f"V3 capacity | seed={master_seed} device={device} K={k_list} N={n_list} "
        f"alphas={alphas} n_seeds={n_seeds} epochs={epochs} lr={lr} mom={momentum} "
        f"grid_per_corr={grid_per_corr} anchor_K={sorted(anchor_k)} restarts={n_restarts} "
        f"mean_spikes={mean_spikes_fixed} spikes_per_K={spikes_per_K} ensemble={ensemble} "
        f"lr_mode={lr_mode} lr_gs_coeff={lr_gs_coeff} vthr_fixed={vthr_fixed} V0={PSP_V0:.4f}",
        flush=True,
    )

    metrics_path = run_dir / "metrics.jsonl"

    def emit(name: str, value: float, step: int) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps({"name": name, "value": float(value), "step": int(step),
                                "wall_time": time.time()}) + "\n")

    started = time.time()
    rows: list[dict[str, float]] = []
    n_cells = len(k_list) * len(n_list) * len(alphas)
    cell_i = 0

    for k in k_list:
        t_window = k * SQRT_TAU
        mean_spikes = spikes_per_K * k if spikes_per_K > 0 else mean_spikes_fixed
        dt = SQRT_TAU / grid_per_corr
        n_grid = round(t_window / dt) + 1
        t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * dt
        do_restart = k in anchor_k
        for n_aff in n_list:
            # GS 2006 Fig-4 capacity schedule: lambda = coeff * T / (tau_m * N * V0). 'fixed' keeps lr.
            lr_cell = (lr_gs_coeff * t_window / (TAU_M * n_aff * PSP_V0)) if lr_mode == "gs" else lr
            for alpha in alphas:
                cell_i += 1
                p = round(alpha * n_aff)
                if p == 0:
                    continue
                rng = np.random.default_rng(cell_seed(master_seed, n_aff, alpha, k))
                spikes, valid, labels, _ = make_patterns(n_seeds, p, n_aff, t_window, rng, device,
                                                          mean_spikes=mean_spikes, ensemble=ensemble)

                # seed-batch size from the trace-tensor memory budget
                per_task_bytes = n_grid * p * n_aff * 4
                sb_size = max(1, min(n_seeds, s_budget // max(1, per_task_bytes)))

                conv = torch.zeros(n_seeds, dtype=torch.bool, device=device)
                conv_multi = torch.zeros(n_seeds, dtype=torch.bool, device=device)
                ep_run = torch.zeros(n_seeds, device=device)
                wn = torch.zeros(n_seeds, device=device)
                mg = torch.zeros(n_seeds, device=device)
                fire = torch.zeros(n_seeds, device=device)
                errf = torch.zeros(n_seeds, device=device)
                kp = torch.zeros(n_seeds, device=device)
                km = torch.zeros(n_seeds, device=device)
                rad = torch.full((n_seeds,), float("nan"), device=device)
                ovl = torch.full((n_seeds,), float("nan"), device=device)
                noisy_acc = {sg: torch.zeros(n_seeds, device=device) for sg in noise_sigmas}
                # V9 raw store: final weights/threshold per seed + the per-batch trajectory buffers.
                weights_all = torch.zeros(n_seeds, n_aff, device=device) if capture else None
                thr_all = torch.zeros(n_seeds, device=device) if capture else None
                cap_batches: list[tuple[int, int, dict]] = []

                for b0 in range(0, n_seeds, sb_size):
                    b1 = min(b0 + sb_size, n_seeds)
                    s = precompute_traces(spikes[b0:b1], valid[b0:b1], t_grid, elem_budget)
                    lab_b = labels[b0:b1]
                    wrng = np.random.default_rng(cell_seed(master_seed, n_aff, alpha, k, tag=1) + b0)
                    w0 = torch.from_numpy(
                        (sigma_w * wrng.standard_normal((b1 - b0, n_aff))).astype(np.float32)).to(device)
                    # stream the seed-averaged learning curve (error+loss vs epoch) to metrics.jsonl
                    # for the representative first seed-batch -- live and survives a teardown/rsync loss.
                    def _mcb(ep: int, errm: float, lossm: float, _a: float = alpha, _n: int = n_aff) -> None:
                        emit(f"trainerr_N{_n}_a{_a:.2f}", errm, ep)
                        emit(f"trainloss_N{_n}_a{_a:.2f}", lossm, ep)
                    res = train_batch(s, lab_b, w0, lr=lr_cell, momentum=momentum, epochs=epochs,
                                      mode=mode, uth_scale=uth_scale, kappa_target=kappa_target,
                                      noise_sigmas=noise_sigmas, vthr_fixed=vthr_fixed,
                                      rescale_init=rescale_init, readout=readout,
                                      log_every=log_every, patience=patience,
                                      log_tag=f"K{round(k)}N{n_aff}a{alpha:.2f}b{b0}",
                                      record_history=bool(history_flag) and b0 == 0,
                                      metric_cb=_mcb if b0 == 0 else None,
                                      capture=bool(capture), patience_min_delta=patience_min_delta,
                                      optimizer=optimizer, lr_schedule=lr_schedule, lr_warmup=lr_warmup,
                                      batch_size=batch_size, lr_step_size=lr_step_size, lr_gamma=lr_gamma,
                                      adam_betas=(adam_b1, adam_b2), adam_eps=adam_eps,
                                      rms_alpha=rms_alpha, rms_eps=rms_eps)
                    if capture:
                        weights_all[b0:b1] = res["weights"]
                        thr_all[b0:b1] = res["threshold"]
                        cap_batches.append((b0, b1, {k_: v.detach().to("cpu")
                                                     for k_, v in res["cap"].items()
                                                     if isinstance(v, torch.Tensor)}))
                    if history_flag and b0 == 0 and res["history"]:
                        hpath = run_dir / f"history_N{n_aff}_a{alpha:.2f}.csv"
                        with hpath.open("w", newline="") as hf:
                            hw = csv.writer(hf)
                            hw.writerow(["epoch", "err_mean", "err_std", "cost_mean", "cost_std",
                                         "alpha", "N", "n_seeds_batch"])
                            for row_h in res["history"]:
                                hw.writerow([*row_h, alpha, n_aff, b1 - b0])
                    conv[b0:b1] = res["converged"]
                    ep_run[b0:b1] = res["epochs_run"].float()
                    wn[b0:b1] = res["wnorm"]
                    mg[b0:b1] = res["mean_margin"]
                    fire[b0:b1] = res["init_fire_rate"]
                    errf[b0:b1] = res["final_errfrac"]
                    kp[b0:b1] = res["kappa_plus"]
                    km[b0:b1] = res["kappa_minus"]
                    for sg in noise_sigmas:
                        noisy_acc[sg][b0:b1] = res["noisy_err"][sg]

                    if robust_probe:
                        rr_rad = robustness_radius(s, res["weights"], res["threshold"], lab_b, wrng)
                        # only meaningful for converged tasks; leave others NaN
                        rad[b0:b1] = torch.where(res["converged"], rr_rad,
                                                 torch.full_like(rr_rad, float("nan")))

                    if do_restart:
                        any_conv = res["converged"].clone()
                        thr = res["threshold"]
                        ws = [res["weights"]]; cs = [res["converged"]]
                        for r in range(1, n_restarts):
                            rrng = np.random.default_rng(
                                cell_seed(master_seed, n_aff, alpha, k, tag=2 + r) + b0)
                            wr = torch.from_numpy(
                                (sigma_w * rrng.standard_normal((b1 - b0, n_aff))).astype(np.float32)).to(device)
                            rr = train_batch(s, lab_b, wr, lr=lr_cell, momentum=momentum,
                                             epochs=epochs, threshold=thr, mode=mode,
                                             uth_scale=1.0, kappa_target=kappa_target,
                                             vthr_fixed=vthr_fixed, rescale_init=rescale_init)
                            any_conv = any_conv | rr["converged"]
                            ws.append(rr["weights"]); cs.append(rr["converged"])
                            if (not overlap_probe) and bool(any_conv.all()):
                                break  # overlap needs all restarts; else stop once any solves
                        conv_multi[b0:b1] = any_conv
                        if overlap_probe:
                            ovl[b0:b1] = mean_pairwise_overlap(ws, cs)
                    del s
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                n_solved = int(conv.sum().item())
                p_solve = n_solved / n_seeds
                p_se = math.sqrt(max(p_solve * (1 - p_solve), 0.0) / n_seeds)
                n_unconv = int((~conv).sum().item())
                # residual error fraction among UNCONVERGED tasks: ~0 => budget-limited (would
                # likely solve with more epochs); large => capacity-limited (genuinely stuck).
                resid = float((errf * (~conv).float()).sum().item() / max(1, n_unconv))
                # per-class margins + robustness on CONVERGED tasks (where a solution exists)
                cmask = conv.float()
                nconv = max(1.0, float(conv.sum().item()))
                kp_conv = float((kp * cmask).sum().item() / nconv) if n_solved else float("nan")
                km_conv = float((km * cmask).sum().item() / nconv) if n_solved else float("nan")
                if robust_probe and n_solved:
                    rconv = rad[conv]
                    rad_mean = float(rconv[~torch.isnan(rconv)].mean().item())
                else:
                    rad_mean = float("nan")
                # GS 2006 Fig-4a learning-time observable: epochs-to-converge among solved tasks
                # (the curve that diverges at alpha_c). mean_epochs (all tasks) saturates at the
                # budget above capacity, so the median over converged runs is the cleaner signal.
                median_epochs_conv = float(ep_run[conv].median().item()) if n_solved else float("nan")
                mean_epochs_conv = float(ep_run[conv].mean().item()) if n_solved else float("nan")
                row = {
                    "K": k, "N": n_aff, "alpha": alpha, "P": p, "n_seeds": n_seeds,
                    "n_solved": n_solved, "p_solve": p_solve, "p_solve_se": p_se,
                    "mean_epochs": float(ep_run.mean().item()),
                    "median_epochs": float(ep_run.median().item()),
                    "mean_epochs_conv": mean_epochs_conv,
                    "median_epochs_conv": median_epochs_conv,
                    "lr_cell": float(lr_cell), "ensemble": ensemble,
                    "resid_errfrac_unconv": resid,
                    "mean_wnorm": float(wn.mean().item()),
                    "mean_margin": float(mg.mean().item()),
                    "kappa_plus_conv": kp_conv,
                    "kappa_minus_conv": km_conv,
                    "robustness_radius": rad_mean,
                    "mean_restart_overlap": (float(ovl[~torch.isnan(ovl)].mean().item())
                                             if overlap_probe and (~torch.isnan(ovl)).any() else float("nan")),
                    "mean_init_fire_rate": float(fire.mean().item()),
                    "mean_spikes": mean_spikes,
                    # V6: fraction of tasks that stay correct (noisy error < 2%) under output noise
                    # sigma -> its 1/2-crossing in alpha is the robust capacity alpha_b(sigma).
                    **{f"p_robust_s{int(round(sg*100)):03d}":
                       float((noisy_acc[sg] < noise_tol).float().mean().item()) for sg in noise_sigmas},
                    "n_grid": n_grid, "dt": dt, "lr": lr, "epochs": epochs,
                    "n_restarts": n_restarts if do_restart else 0,
                    "n_solved_multi": int(conv_multi.sum().item()) if do_restart else -1,
                    "p_solve_multi": (int(conv_multi.sum().item()) / n_seeds) if do_restart else -1.0,
                }
                rows.append(row)
                if capture:
                    # Assemble the per-cell raw store: per-seed per-epoch trajectories (NaN-padded to
                    # the cell's max epochs, concatenated over seed-batches) + final per-seed state +
                    # final weights. Plus the in-run reference scalars, so the offline pipeline can
                    # prove it reconstructs them losslessly (G_capture). One .npz per cell -> a partial
                    # run is still fully analysable cell-by-cell (survives a teardown/rsync failure).
                    cells_dir = run_dir / "cells"
                    cells_dir.mkdir(exist_ok=True)
                    max_ne = max((cd["traj_err"].shape[1] for _, _, cd in cap_batches), default=0)
                    ran_epochs = np.zeros(n_seeds, dtype=np.int64)

                    def _assemble(key: str) -> np.ndarray:
                        arr = np.full((n_seeds, max_ne), np.nan, dtype=np.float32)
                        for b0_, b1_, cd in cap_batches:
                            a = cd[key].numpy()
                            arr[b0_:b1_, : a.shape[1]] = a
                            ran_epochs[b0_:b1_] = a.shape[1]
                        return arr

                    traj = {f"{ch}": _assemble(ch) for ch in
                            ("traj_err", "traj_loss", "traj_wnorm", "traj_nupd", "traj_kp", "traj_km")}
                    npz_path = cells_dir / f"cell_N{n_aff}_a{alpha:.3f}_K{round(k)}.npz"
                    np.savez_compressed(
                        npz_path,
                        # --- per-seed per-epoch trajectories (Sb, max_ne), NaN past each seed's stop ---
                        **traj,
                        ran_epochs=ran_epochs,                       # epochs actually run per seed
                        # --- final per-seed state ---
                        converged=conv.cpu().numpy(),
                        epochs_to_conv=ep_run.cpu().numpy(),         # convergence epoch (==budget if not)
                        final_errfrac=errf.cpu().numpy(),
                        final_wnorm=wn.cpu().numpy(),
                        final_kappa_plus=kp.cpu().numpy(),
                        final_kappa_minus=km.cpu().numpy(),
                        init_fire_rate=fire.cpu().numpy(),
                        threshold=thr_all.cpu().numpy(),
                        robustness_radius=rad.cpu().numpy(),
                        weights=weights_all.cpu().numpy(),           # (n_seeds, N) final weights
                        # --- cell metadata (scalars) ---
                        N=np.int64(n_aff), alpha=np.float64(alpha), K=np.float64(k), P=np.int64(p),
                        n_seeds=np.int64(n_seeds), lr_cell=np.float64(lr_cell),
                        epochs_budget=np.int64(epochs), dt=np.float64(dt), n_grid=np.int64(n_grid),
                        momentum=np.float64(momentum), master_seed=np.int64(master_seed),
                        patience=np.int64(patience), vthr_fixed=np.float64(vthr_fixed or float("nan")),
                        ensemble=np.str_(ensemble), optimizer=np.str_(optimizer),
                        lr_schedule=np.str_(lr_schedule), mode=np.str_(mode),
                        # --- in-run reference scalars (for the lossless-reconstruction gate) ---
                        ref_p_solve=np.float64(p_solve), ref_n_solved=np.int64(n_solved),
                        ref_median_epochs_conv=np.float64(median_epochs_conv),
                        ref_resid=np.float64(resid),
                    )
                emit(f"psolve_K{round(k)}_N{n_aff}", p_solve, round(1000 * alpha))
                print(
                    f"[{cell_i}/{n_cells}] K={k:.0f} N={n_aff} a={alpha:.3f} P={p} "
                    f"p_solve={p_solve:.3f}+/-{p_se:.3f} resid={resid:.3f} "
                    f"<ep>={row['mean_epochs']:.0f} med_ep_conv={median_epochs_conv:.0f} "
                    f"fire0={row['mean_init_fire_rate']:.2f}"
                    + (f" p_multi={row['p_solve_multi']:.3f}" if do_restart else ""),
                    flush=True,
                )

    elapsed = time.time() - started
    results = {
        "experiment": "v3_capacity_sweep",
        "params": {
            "master_seed": master_seed, "K_list": k_list, "N_list": n_list, "alphas": alphas,
            "n_seeds": n_seeds, "epochs": epochs, "lr": lr, "momentum": momentum, "mode": mode,
            "mean_spikes_fixed": mean_spikes_fixed, "spikes_per_K": spikes_per_K,
            "robust_probe": robust_probe, "uth_scale": uth_scale, "kappa_target": kappa_target,
            "noise_sigmas": list(noise_sigmas), "overlap_probe": overlap_probe,
            "grid_per_corr": grid_per_corr, "tau_m": TAU_M, "tau_s": TAU_S,
            "anchor_K": sorted(anchor_k), "n_restarts": n_restarts,
            "ensemble": ensemble, "lr_mode": lr_mode, "lr_gs_coeff": lr_gs_coeff,
            "optimizer": optimizer, "lr_schedule": lr_schedule, "lr_warmup": lr_warmup,
            "patience": patience, "patience_min_delta": patience_min_delta,
            "vthr_fixed": vthr_fixed, "psp_v0": PSP_V0,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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
    if capture:
        # Self-describing manifest for the raw store: provenance + the cell-file index, so the offline
        # pipeline (analysis/v9_*.py) can load and verify the dataset without any other state.
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_sha = None
        cells = sorted(str(p.relative_to(run_dir)) for p in (run_dir / "cells").glob("*.npz")) \
            if (run_dir / "cells").exists() else []
        manifest = {
            "experiment": "v9_capacity_capture",
            "git_sha": git_sha,
            "created": time.time(),
            "versions": {"numpy": np.__version__, "torch": torch.__version__,
                         "python": sys.version.split()[0]},
            "params": results["params"],
            "capture_schema": {
                "trajectories": ["traj_err", "traj_loss", "traj_wnorm", "traj_nupd",
                                 "traj_kp", "traj_km"],
                "trajectory_shape": "(n_seeds, max_epochs_run), NaN past each seed's stop epoch",
                "final_state": ["converged", "epochs_to_conv", "final_errfrac", "final_wnorm",
                                "final_kappa_plus", "final_kappa_minus", "init_fire_rate",
                                "threshold", "robustness_radius", "weights", "ran_epochs"],
                "reference_scalars": ["ref_p_solve", "ref_n_solved", "ref_median_epochs_conv",
                                      "ref_resid"],
            },
            "cells": cells,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
        print(f"raw store: {len(cells)} cell .npz + manifest.json under {run_dir}", flush=True)
    print(f"done in {elapsed:.1f}s -> {run_dir}/results.json ({len(rows)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
