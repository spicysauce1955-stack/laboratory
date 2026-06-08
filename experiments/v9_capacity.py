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
    gen: torch.Generator,
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
    shape = (n_tasks, n_patterns, n_aff)
    if ensemble == "single":
        # Exactly one spike per afferent, time ~ U[0, T] (GS 2006 latency ensemble).
        max_spikes = 1
        spike_times = t_window * torch.rand(
            (n_tasks, n_patterns, n_aff, 1), generator=gen, device=device, dtype=torch.float32
        )
        valid = torch.ones((n_tasks, n_patterns, n_aff, 1), device=device, dtype=torch.float32)
    elif ensemble == "synchrony_half":
        # Random half of afferents fire at one common per-pattern time; rest silent (perceptron).
        max_spikes = 1
        t_common = t_window * torch.rand(
            (n_tasks, n_patterns, 1, 1), generator=gen, device=device, dtype=torch.float32
        )
        spike_times = t_common.expand(n_tasks, n_patterns, n_aff, 1).contiguous()
        active = (torch.rand(shape, generator=gen, device=device) < 0.5).to(torch.float32)
        valid = active.unsqueeze(-1)
    else:  # "poisson" (RMS 2010)
        counts = torch.poisson(torch.full(shape, float(mean_spikes), device=device),
                               generator=gen).to(torch.int64)
        max_spikes = max(1, int(counts.max().item()))
        spike_times = t_window * torch.rand(
            (n_tasks, n_patterns, n_aff, max_spikes), generator=gen, device=device, dtype=torch.float32
        )
        spike_idx = torch.arange(max_spikes, device=device).view(1, 1, 1, max_spikes)
        valid = (spike_idx < counts.unsqueeze(-1)).to(torch.float32)
    labels = torch.where(
        torch.rand((n_tasks, n_patterns), generator=gen, device=device) < 0.5,
        torch.tensor(1.0, device=device),
        torch.tensor(-1.0, device=device),
    )
    return spike_times, valid, labels, max_spikes


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
def _forward(s: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``V[b,p,g] = sum_i s[b,p,g,i] w[b,i]``; return ``(vmax, argmax_t)`` each ``(Sb, P)``."""
    v = torch.einsum("bpgn,bn->bpg", s, w)
    vmax, targ = v.max(dim=2)
    return vmax, targ


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
    log_every: int = 0,    # V8: if >0, print converged-fraction every log_every epochs (live monitor)
    patience: int = 0,     # V8: if >0, stop a cell after this many epochs with no new convergence
    log_tag: str = "",
    record_history: bool = False,  # V8: record per-epoch mean/std error & GS-cost across the seed batch
    metric_cb=None,  # V8: callback(epoch, err_mean, loss_mean) -> stream learning curve to metrics.jsonl
    capture: bool = False,    # V9: record FULL per-seed per-epoch trajectories (raw store, online only)
    patience_min_delta: float = 0.0,  # V9: a loss drop below best-min_delta counts as "improvement"
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
    vmax, _ = _forward(s, w)
    if vthr_fixed is not None:
        # Faithful GS: fixed threshold, rescale init weights so median V_max == vthr_fixed.
        med = vmax.median(dim=1).values.clamp(min=1e-12)  # (Sb,)
        w = w * (vthr_fixed / med)[:, None]
        vmax, _ = _forward(s, w)
        threshold = torch.full((sb,), float(vthr_fixed), device=dev)
    elif threshold is None:
        threshold = vmax.median(dim=1).values  # (Sb,)
    threshold = threshold * uth_scale  # V5 gauge: rescale the (fixed) threshold
    band = kappa_target * threshold.abs()  # required signed-margin band (0 => bare zero-error rule)
    init_fire_rate = (vmax >= threshold[:, None]).float().mean(dim=1)
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
    cap = capture and mode == "online"
    last_ep = -1
    best_loss = float("inf")
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
            order = torch.randperm(p, device=dev)
            updates = torch.zeros(sb, device=dev)  # #updates this epoch (per task)
            err_acc = torch.zeros(sb, device=dev)   # raw training errors this epoch (all seeds, ungated)
            cost_acc = torch.zeros(sb, device=dev)  # GS hinge cost relu(band - signed_margin)
            kp_acc = torch.zeros(sb, device=dev)    # V9: running sum of signed margin on +1 patterns
            km_acc = torch.zeros(sb, device=dev)    # V9: running sum of signed margin on -1 patterns
            for pi in order:
                sp = s[:, pi]                               # (Sb, G, N) contiguous
                vp = torch.einsum("bgn,bn->bg", sp, w)
                vmx, tg = vp.max(dim=1)                     # (Sb,)
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
                corr = (lr * labels[:, pi]).unsqueeze(-1) * grad  # GS correction dw = lr*y*s(t_max)
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
            newly = (updates == 0) & (~converged)
        else:  # batch mode
            vmax, targ = _forward(s, w)
            err = (((vmax - threshold[:, None]) * labels) < band[:, None]).float()
            idx = targ.view(sb, p, 1, 1).expand(sb, p, 1, n)
            grad = torch.gather(s, 2, idx).squeeze(2)       # (Sb, P, N)
            dw = lr * ((err * labels).unsqueeze(-1) * grad).sum(dim=1)
            vel = momentum * vel + dw
            w = w + vel * active
            newly = (err.sum(dim=1) == 0) & (~converged)
        epochs_run = torch.where(newly, torch.tensor(ep, device=dev), epochs_run)
        converged = converged | newly
        last_ep = ep
        if cap:
            # Record this epoch's per-seed trajectory from the cheap reductions accumulated during
            # the online sweep (no extra forward pass). NaN-padded rows are sliced off on return.
            err_buf[:, ep] = err_acc / p
            loss_buf[:, ep] = cost_acc / p
            wnorm_buf[:, ep] = w.norm(dim=1)
            nupd_buf[:, ep] = updates
            kp_buf[:, ep] = kp_acc / npos
            km_buf[:, ep] = km_acc / nneg
        if bool(converged.all()):
            break
        if cap:
            # V9 patience: "no improvement" = neither the seed-averaged GS loss dropped below its
            # running best (by more than patience_min_delta) NOR a new seed converged, for `patience`
            # consecutive epochs. Tracking the *continuous* loss keeps a still-descending
            # (budget-limited) cell running; also resetting on any new convergence protects a slow
            # straggler from being cut when a few permanently-stuck seeds plateau the average.
            n_conv = int(converged.sum().item())
            cur_loss = float((cost_acc / p).mean().item())
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
                err_all = float((err_acc / p).mean().item())
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
    gen: torch.Generator,
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
            xi = torch.randn(sb, n, generator=gen, device=s.device)
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
    # Live learning-curve logging cadence (epochs). Default ON (~50 points/cell) so EVERY run streams
    # its training error+loss to metrics.jsonl -- captured live and surviving a teardown/rsync failure.
    log_every = int(ov.get("log_every", "0"))
    patience = int(ov.get("patience", "0"))     # early-stop a cell after this many no-progress epochs
    history_flag = int(ov.get("history", "0"))  # also write the fine-grained per-epoch CSV per cell
    # V9 raw capture: write FULL per-seed per-epoch trajectories + final weights per cell to .npz, so
    # every numeric/figure is derived offline (analysis/v9_*.py) without rerunning the GPU. Online only.
    capture = int(ov.get("capture", "0"))
    patience_min_delta = float(ov.get("patience_min_delta", "0.0"))
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
                gen = torch.Generator(device=device)
                gen.manual_seed(cell_seed(master_seed, n_aff, alpha, k))
                spikes, valid, labels, _ = make_patterns(n_seeds, p, n_aff, t_window, gen, device,
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
                    wgen = torch.Generator(device=device)
                    wgen.manual_seed(cell_seed(master_seed, n_aff, alpha, k, tag=1) + b0)
                    w0 = torch.randn(b1 - b0, n_aff, generator=wgen, device=device)
                    # stream the seed-averaged learning curve (error+loss vs epoch) to metrics.jsonl
                    # for the representative first seed-batch -- live and survives a teardown/rsync loss.
                    def _mcb(ep: int, errm: float, lossm: float, _a: float = alpha, _n: int = n_aff) -> None:
                        emit(f"trainerr_N{_n}_a{_a:.2f}", errm, ep)
                        emit(f"trainloss_N{_n}_a{_a:.2f}", lossm, ep)
                    res = train_batch(s, lab_b, w0, lr=lr_cell, momentum=momentum, epochs=epochs,
                                      mode=mode, uth_scale=uth_scale, kappa_target=kappa_target,
                                      noise_sigmas=noise_sigmas, vthr_fixed=vthr_fixed,
                                      log_every=log_every, patience=patience,
                                      log_tag=f"K{round(k)}N{n_aff}a{alpha:.2f}b{b0}",
                                      record_history=bool(history_flag) and b0 == 0,
                                      metric_cb=_mcb if b0 == 0 else None,
                                      capture=bool(capture), patience_min_delta=patience_min_delta)
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
                        rr_rad = robustness_radius(s, res["weights"], res["threshold"], lab_b, wgen)
                        # only meaningful for converged tasks; leave others NaN
                        rad[b0:b1] = torch.where(res["converged"], rr_rad,
                                                 torch.full_like(rr_rad, float("nan")))

                    if do_restart:
                        any_conv = res["converged"].clone()
                        thr = res["threshold"]
                        ws = [res["weights"]]; cs = [res["converged"]]
                        for r in range(1, n_restarts):
                            wgen.manual_seed(cell_seed(master_seed, n_aff, alpha, k, tag=2 + r) + b0)
                            wr = torch.randn(b1 - b0, n_aff, generator=wgen, device=device)
                            rr = train_batch(s, lab_b, wr, lr=lr_cell, momentum=momentum,
                                             epochs=epochs, threshold=thr, mode=mode,
                                             uth_scale=1.0, kappa_target=kappa_target,
                                             vthr_fixed=vthr_fixed)
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
                        ensemble=np.str_(ensemble),
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
