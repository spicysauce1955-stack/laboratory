"""V8 GP CAPACITY -- findable alpha_hat_c(tau_s) saturates (K_PR) vs diverges (Rice) as tau_s -> 0.

Self-contained Experiment-Contract entrypoint (numpy + torch only; the lab ships this file unchanged,
so there are NO ``tempotron`` / ``lab`` imports). This is the CAPACITY payoff (test (b)) of reduction
#4; see ``lit/tempotron-reductions/derivations/04-gp-capacity-prediction.md`` section (b) and the
CAPACITY-EXPERIMENT-PLAYBOOK. It is the **tau_s-knob sibling** of ``v6_bandlimited_capacity.py`` and a
fixed-T adaptation of ``v3_capacity_sweep.py`` (double-exp PSP, Adam/cosine minibatch learner, per-epoch
capture -- all reused verbatim; the ONLY change is the driver geometry).

Why a new driver (not a config wrapper on v3)
---------------------------------------------
v3 ties the geometry to K: ``t_window = K sqrt(tau_s tau_m)`` and ``dt = sqrt(tau_s tau_m)/grid_per_corr``,
so the window grows with K and the grid shrinks only as ``1/sqrt(tau_s)``. Reduction #4 instead holds
**T fixed** (=500 ms) and sweeps **tau_s** at fixed tau_m=15, with the cost-driving grid ``dt = tau_s/OS``
(``n_grid = OS*T/tau_s ~ 1/tau_s``) needed to Nyquist-resolve the sharpening kink (prediction note section
(b)/3). That fixed-T, tau_s-swept geometry with a 1/tau_s grid is not expressible through v3's K-driven
config, so the driver is rewritten here; the physics (kernel, traces, training) is byte-identical to v3.

The prediction
--------------
At fixed T, tau, as tau_s -> 0 the capacity-relevant effective dimension is the participation ratio

    K_PR = T (tau + tau_s) / (tau^2 + 3 tau tau_s + tau_s^2)  ->  T/tau   (SATURATES),

NOT the Rice up-crossing count ``K_Rice = T/(pi sqrt(tau tau_s)) -> inf`` (DIVERGES). With the double-log
lift ``alpha_c(K) = ln ln K/(2 ln 2) + offset`` (RMS 2010), the findable ``alpha_hat_c(tau_s)`` should
track ``1/2 ln ln K_PR`` (saturate/peak) and NOT ``1/2 ln ln K_Rice`` (keep rising). The offset cancels
in the contrast; the discriminating observable is the SHAPE of ``alpha_hat_c(tau_s)`` (saturate vs rise)
and the kernel-shape corollary (the alpha-function floor at tau_s=tau vs the well-separated peak).

HONEST decisiveness (prediction note): the alpha_c gap between the two hypotheses is small
(Delta alpha ~ 0.13 at the cheap tau_s, ~0.21 only at the expensive r=0.01 corner) -- inside findable
noise except at the corner. This test is CORROBORATIVE; the decisive evidence is the free witness
(``v8_gp_witness.py``). Measure the shape, do not chase the gap.

Controls (mandatory, from the #6 lessons / playbook)
----------------------------------------------------
- **Fixed density** (the #6 confound fix): ``mean_spikes_fixed`` holds the expected spikes/afferent CONSTANT
  across the tau_s sweep, so a capacity change is attributable to the kernel band, not to input richness.
  Default ON. A density control (``density_list``) is provided to subtract the nuisance. Do NOT let
  density co-vary with tau_s.
- **Balanced threshold** (invariant): U_th = median V_max at init (calibrate; P(fire)=1/2). The ln ln K
  lift only exists at this balanced point. ``vthr_fixed=1 rescale_init=1`` is the gauge-equivalent variant.
- **dt/grid resolution**: ``dt = tau_s/OS`` (OS=oversample, default 8) resolves the kink; per cell we log
  ``n_grid`` and certify the realized-trace E[max V] against the GP witness (the certified-axis gate).
- **Learner**: Adam-cosine minibatch (``mode=minibatch optimizer=adam lr_schedule=cosine``), generous
  epochs + patience, n_restarts>=3 (existence-from-below proxy). Validate via critical slowing.

Per cell we log: ``tau_s, n_grid, K_PR, K_Rice, e_max_realized`` (certified axis) and the P_solve crossing.

Run standalone (tiny smoke; does NOT need a GPU):
    LAB_RUN_DIR=/tmp/v8c uv run --with torch python studies/v8_gp_capacity.py \\
        tau_s_list=15 N_list=40 alphas=2.0,3.0 seeds=0,1 epochs=200 \\
        mode=minibatch optimizer=adam lr_schedule=cosine batch_size=16 lr=0.1

Real capacity run (GPU on the lab):
    require_cuda=1 ... tau_s_list=15,4.5,1.5 N_list=500 alphas=... seeds=0..23 epochs=8000 ...
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

TAU_M: float = 15.0  # ms, membrane time constant (held fixed; reduction #4 sweeps tau_s)
TAU_S: float = 1.5   # ms, synaptic time constant (set per cell from tau_s_list)
T_MS: float = 500.0  # ms, fixed observation window (reduction #4 holds T fixed, sweeps tau_s)


# --------------------------------------------------------------------------------------------
# PSP kernel (peak-normalised double exponential, causal) -- byte-identical to v3.psp_kernel.
# TAU_M/TAU_S are module globals set per cell by the driver (mirrors v3's global-override pattern).
# --------------------------------------------------------------------------------------------
def _alpha_kernel() -> bool:
    """True when tau_s == tau_m (the alpha-function endpoint of the kernel-shape corollary)."""
    return abs(TAU_M - TAU_S) < 1e-9 * TAU_M


def _psp_v0() -> float:
    if _alpha_kernel():
        return 1.0  # alpha-function: K(t)=(t/tau)e^{1-t/tau} is peak-normalised by construction
    t_peak = TAU_M * TAU_S / (TAU_M - TAU_S) * math.log(TAU_M / TAU_S)
    return 1.0 / (math.exp(-t_peak / TAU_M) - math.exp(-t_peak / TAU_S))


PSP_V0: float = _psp_v0()


def psp_kernel(t: torch.Tensor) -> torch.Tensor:
    """Peak-normalised PSP for ``t >= 0`` (else 0).

    Double-exponential ``K(t) = V0 (e^{-t/tau_m} - e^{-t/tau_s})`` away from tau_s=tau_m; at the
    alpha-function endpoint tau_s=tau_m it is the L'Hopital limit ``K(t) = (t/tau) e^{1 - t/tau}``
    (peak 1 at t=tau), the narrowest-band corner of the kernel-shape corollary (#4 section 2).
    """
    t_clamped = torch.clamp(t, min=0.0)
    if _alpha_kernel():
        raw = (t_clamped / TAU_M) * torch.exp(1.0 - t_clamped / TAU_M)
    else:
        raw = PSP_V0 * (torch.exp(-t_clamped / TAU_M) - torch.exp(-t_clamped / TAU_S))
    return torch.where(t >= 0.0, raw, torch.zeros_like(raw))


# --------------------------------------------------------------------------------------------
# Reproducible per-cell seeding (matches v3/v6 cell_seed recipe; K rounded -> use round(tau_s))
# --------------------------------------------------------------------------------------------
def cell_seed(master: int, n: int, alpha: float, k: float, tag: int = 0) -> int:
    """A scheduling-independent 63-bit seed for a cell (matches the v3/v6 SeedSequence recipe)."""
    ss = np.random.SeedSequence([master, n, round(1000 * alpha), round(k), tag])
    return int(ss.generate_state(1, dtype=np.uint64)[0]) >> 1


# --------------------------------------------------------------------------------------------
# Pattern generation (RMS 2010 Poisson input), batched over tasks -- byte-identical to v6.
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

    Each afferent emits a homogeneous Poisson train on ``[0, T]`` with ``mean_spikes`` expected spikes.
    Drawn on the CPU via a numpy Generator then moved to ``device`` (device-INDEPENDENT, bit-reproducible
    on any backend). Returns ``(spike_times, valid, labels, max_spikes)`` of shape
    ``(n_tasks, P, N, max_spikes)``.
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
    elem_budget: int = 64_000_000,
) -> torch.Tensor:
    """Per-afferent traces ``s[b,g,p,i] = sum_f K(t_g - spike[b,p,i,f]) * valid`` (v3 byte-identical).

    Chunked over the time grid so the working tensor ``(g_chunk, Sb, P, N, max_spikes)`` stays within
    ``elem_budget`` elements. Returns ``s`` of shape ``(Sb, P, G, N)`` (float32); the per-pattern slice
    ``s[:, p]`` is contiguous, which the online sweep relies on.
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
        tt = t_grid[g0:g1].view(-1, 1, 1, 1, 1)                # (gc,1,1,1,1)
        contrib = psp_kernel(tt - st) * vd                      # (gc, Sb, P, N, ms)
        s[:, :, g0:g1, :] = contrib.sum(dim=4).permute(1, 2, 0, 3)  # (gc,Sb,P,N)->(Sb,P,gc,N)
    return s


def certify_emax_from_traces(s: torch.Tensor, n_probe: int = 64) -> float:
    """Certify E[max_t V] on the REALIZED traces (the dt-resolution / certified-axis gate).

    ``V[b,p,g] = sum_i s[b,p,g,i] xi_i`` with ``xi ~ N(0,I)`` is, for fixed (b,p), a sample path of the
    membrane GP on the actual training grid. We normalise each path to unit variance (the gauge the GP
    witness uses) and report the mean of max_t V over the probe paths. Per the playbook (gate 3) this
    realized E[max V] must match the GP witness ``v8_gp_witness.py`` per cell -- if the dt is too coarse
    the discretized kink biases V_max low (a spurious saturation), and this number catches it.
    """
    sb, p, g, n = s.shape
    dev = s.device
    flat = s.reshape(sb * p, g, n)
    m = min(n_probe, flat.shape[0])
    sel = flat[:m]  # (m, G, N)
    gen = torch.Generator(device=dev); gen.manual_seed(20260622)
    xi = torch.randn((m, n), generator=gen, device=dev, dtype=s.dtype)
    V = torch.einsum("mgn,mn->mg", sel, xi)              # (m, G)
    V = V - V.mean(dim=1, keepdim=True)
    V = V / V.std(dim=1, keepdim=True).clamp(min=1e-12)  # unit variance per path (witness gauge)
    return float(V.max(dim=1).values.mean().item())


# --------------------------------------------------------------------------------------------
# Forward pass + Gutig-Sompolinsky training -- byte-identical to v6.train_batch (online + V10
# minibatch path: Adam/RMSprop/momentum, cosine/linear schedule, per-epoch capture).
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
    """Scheduled learning rate at epoch ``ep`` (copied verbatim from v3/v6 scheduled_lr)."""
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
    vthr_fixed: float | None = None,  # faithful GS: fix V_thr and rescale init to median V_max
    patience: int = 0,
    log_every: int = 0,
    log_tag: str = "",
    train_seed: int = 0,
    capture: bool = False,
    metric_cb=None,
    mode: str = "online",   # 'online' (faithful GS) | 'minibatch' (V10 strong learner)
    optimizer: str = "momentum",  # momentum | adam | rmsprop (minibatch)
    lr_schedule: str = "none",    # none | cosine | linear | step | exp
    lr_warmup: int = 0,
    lr_floor: float = 0.0,
    lr_cycles: int = 1,
    lr_exp_k: float = EXP_DECAY_K,
    lr_step_size: int = 0,
    lr_gamma: float = 0.1,
    batch_size: int = 0,          # b in {1..P}; required for mode='minibatch'
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    rms_alpha: float = 0.99,
    rms_eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Gutig-Sompolinsky rule, vectorised across tasks (port of v6.train_batch + v3 vthr gauge).

    ``mode='online'`` (faithful GS): per-pattern updates within a seeded-shuffled epoch; a task converges
    once a full epoch passes with zero updates. ``mode='minibatch'`` (V10 strong learner): one shuffled
    pass split into ceil(P/b) steps; the GS hinge subgradient is the per-minibatch mean over violators,
    ascended by Adam/RMSprop/momentum under a schedule. At b=1, momentum, constant LR this reduces to the
    online rule bit-for-bit. A NaN-weight seed (LR-too-large divergence) is NOT counted converged.

    Threshold gauge: ``vthr_fixed=None`` calibrates U_th to the median V_max at init (P(fire)=1/2).
    ``vthr_fixed=v`` holds V_thr=v and rescales the init weights so median V_max=v (gauge-equivalent).

    Returns per-task ``converged``, ``epochs_run``, ``init_fire_rate``, ``final_errfrac``, ``weights``,
    ``threshold``; with ``capture=True`` adds per-epoch ``cap`` trajectories (err/loss/kp/km/wnorm).
    """
    sb, p, _, n = s.shape
    dev = s.device
    shuffle_gen = torch.Generator(device=dev)
    shuffle_gen.manual_seed(int(train_seed) & 0x7FFFFFFFFFFFFFFF)
    w = w_init.clone()
    vmax, _ = _forward(s, w)
    if vthr_fixed is not None:
        med = vmax.median(dim=1).values.clamp(min=1e-12)  # (Sb,)
        w = w * (vthr_fixed / med)[:, None]
        vmax, _ = _forward(s, w)
        threshold = torch.full((sb,), float(vthr_fixed), device=dev)
    elif threshold is None:
        threshold = vmax.median(dim=1).values  # (Sb,) median V_max at init -> P(fire)=1/2
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
        active = (~converged).float().unsqueeze(-1)  # (Sb, 1)
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
                sb_s = s[:, bidx]                              # (Sb, b, G, N)
                lab_b = labels[:, bidx]                        # (Sb, b)
                bb = sb_s.shape[1]
                vb = torch.einsum("bkgn,bn->bkg", sb_s, w)     # (Sb, b, G)
                vmx, tg = vb.max(dim=2)                        # (Sb, b)
                smarg = (vmx - threshold[:, None]) * lab_b     # (Sb, b) signed margin
                viol = (smarg < 0.0).float()
                nb = float(bb)
                viol_enc = viol_enc + viol.sum(dim=1) * act
                gidx = tg.view(sb, bb, 1, 1).expand(sb, bb, 1, n)
                gpat = torch.gather(sb_s, 2, gidx).squeeze(2)  # (Sb, b, N) = s_i(t*)
                g = ((lab_b * viol).unsqueeze(-1) * gpat).sum(dim=1)  # (Sb, N)
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
                else:  # momentum -- gate the velocity too so only error steps advance it (b=1 == GS)
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
        # --- online (faithful GS) ---
        order = torch.randperm(p, device=dev, generator=shuffle_gen)
        updates = torch.zeros(sb, device=dev)
        if cap:
            err_acc = torch.zeros(sb, device=dev)
            cost_acc = torch.zeros(sb, device=dev)
            kp_acc = torch.zeros(sb, device=dev)
            km_acc = torch.zeros(sb, device=dev)
        for pi in order:
            sp = s[:, pi]                               # (Sb, G, N) contiguous
            vp = torch.einsum("bgn,bn->bg", sp, w)
            vmx, tg = vp.max(dim=1)                     # (Sb,) tempotron argmax_t V
            smarg = (vmx - threshold) * labels[:, pi]
            if cap:
                lab_pi = labels[:, pi]
                err_acc = err_acc + (smarg < 0.0).float()
                cost_acc = cost_acc + torch.relu(-smarg)
                kp_acc = kp_acc + smarg * (lab_pi > 0).float()
                km_acc = km_acc + smarg * (lab_pi < 0).float()
            e = (smarg < 0.0).float() * active.squeeze(-1)
            updates = updates + e
            grad = sp[arange_sb, tg]                    # (Sb, N) = s_i(t_max)
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
# Closed-form effective-dimension counts (the two competing capacity axes)
# --------------------------------------------------------------------------------------------
def k_pr(T: float, tau_m: float, tau_s: float) -> float:
    """Participation-ratio count K_PR = T(tau+tau_s)/(tau^2+3 tau tau_s+tau_s^2) -> T/tau (saturates)."""
    return T * (tau_m + tau_s) / (tau_m**2 + 3 * tau_m * tau_s + tau_s**2)


def k_rice(T: float, tau_m: float, tau_s: float) -> float:
    """Rice (two-direction) crossing count K_Rice = T/(pi sqrt(tau tau_s)) -> inf (diverges)."""
    return T / (math.pi * math.sqrt(tau_m * tau_s))


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------
def _alpha_grid(ov: dict[str, str]) -> list[float]:
    if "alphas" in ov:
        return [float(x) for x in ov["alphas"].split(",")]
    a0 = float(ov.get("alpha_min", "2.6"))
    a1 = float(ov.get("alpha_max", "3.6"))
    step = float(ov.get("alpha_step", "0.1"))
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
    global TAU_M, TAU_S, T_MS, PSP_V0
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))

    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    TAU_M = float(ov.get("tau_m", "15"))
    T_MS = float(ov.get("T_ms", "500"))
    tau_s_list = [float(x) for x in ov.get("tau_s_list", "15,4.5,1.5").split(",")]
    n_list = [int(x) for x in ov.get("N_list", "500").split(",")]
    alphas = _alpha_grid(ov)
    if "seeds" in ov:
        seeds = [int(x) for x in ov["seeds"].split(",") if x != ""]
    else:
        seeds = list(range(int(ov.get("n_seeds", "24"))))
    n_seeds = len(seeds)
    oversample = float(ov.get("oversample", "8"))   # dt = tau_s / oversample (Nyquist-resolve the kink)
    # Fixed-density control (the #6 confound fix). mean_spikes_fixed > 0 holds the expected spikes per
    # afferent CONSTANT across the tau_s sweep (de-confounded mode, default). density_list runs the
    # density control at the FIRST tau_s only: one P_solve curve per density to subtract the nuisance.
    mean_spikes_fixed = float(ov.get("mean_spikes_fixed", "2.0"))
    density_list = [float(x) for x in ov.get("density_list", "").split(",") if x != ""]
    capture = int(ov.get("capture", "0"))
    epochs = int(ov.get("epochs", "8000"))
    lr = float(ov.get("lr", "0.1"))
    momentum = float(ov.get("momentum", "0.0"))
    mode = ov.get("mode", "minibatch")           # minibatch (Adam-cosine strong learner) | online
    optimizer = ov.get("optimizer", "adam")      # adam | rmsprop | momentum
    lr_schedule = ov.get("lr_schedule", "cosine")  # cosine | linear | none | step | exp
    lr_warmup = int(ov.get("lr_warmup", "0"))
    lr_floor = float(ov.get("lr_floor", "0.0"))
    lr_cycles = int(ov.get("lr_cycles", "1"))
    lr_step_size = int(ov.get("lr_step_size", "0"))
    lr_gamma = float(ov.get("lr_gamma", "0.1"))
    batch_size = int(ov.get("batch_size", "16"))
    adam_b1 = float(ov.get("adam_b1", "0.9"))
    adam_b2 = float(ov.get("adam_b2", "0.999"))
    adam_eps = float(ov.get("adam_eps", "1e-8"))
    rms_alpha = float(ov.get("rms_alpha", "0.99"))
    rms_eps = float(ov.get("rms_eps", "1e-8"))
    n_restarts = int(ov.get("n_restarts", "3"))
    patience = int(ov.get("patience", "0"))
    log_every = int(ov.get("log_every", "0"))
    sigma_w = float(ov.get("sigma_w", "1.0"))
    vthr_fixed_ov = ov.get("vthr_fixed", "")
    vthr_fixed = float(vthr_fixed_ov) if vthr_fixed_ov else None
    elem_budget = int(float(ov.get("elem_budget", "64e6")))
    emax_probe = int(ov.get("emax_probe", "64"))  # # init voltage paths fed to the E[max V] certifier
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
        f"V8 GP capacity | seed={master_seed} device={device} T={T_MS} tau_m={TAU_M} "
        f"tau_s_list={tau_s_list} N={n_list} alphas={alphas} seeds={seeds} epochs={epochs} lr={lr} "
        f"oversample={oversample} mean_spikes_fixed={mean_spikes_fixed} density_list={density_list} "
        f"mode={mode} optimizer={optimizer} lr_schedule={lr_schedule} batch_size={batch_size} "
        f"restarts={n_restarts} vthr_fixed={vthr_fixed}",
        flush=True,
    )

    metrics_path = run_dir / "metrics.jsonl"

    def emit(name: str, value: float, step: int) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps({"name": name, "value": float(value), "step": int(step),
                                "wall_time": time.time()}) + "\n")

    started = time.time()
    rows: list[dict[str, float]] = []
    history_rows: list[dict[str, float]] = []
    per_taus: dict[float, dict] = {}

    # Incremental, durable results.csv: write the header once (lazily, when the first cell's rows
    # fix the column order) and flush+fsync after EACH (tau_s, density, N, alpha) cell so a
    # timeout/OOM kill leaves a valid, readable partial CSV with all completed cells. The in-memory
    # ``rows`` list is still kept for the end-of-run results.json. ``_csv_written`` tracks rows on disk.
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
    # The density control runs only at the FIRST tau_s; elsewhere a single fixed density.
    first_tau_s = tau_s_list[0]
    n_cells = sum(
        len(n_list) * len(alphas) * (len(density_list) if (density_list and ts == first_tau_s) else 1)
        for ts in tau_s_list
    )
    cell_i = 0
    first_cell_done = False

    for tau_s in tau_s_list:
        # Set the kernel for this cell (module globals; recompute the peak-normalisation V0).
        TAU_S = tau_s
        PSP_V0 = _psp_v0()
        dt = tau_s / oversample                  # the 1/tau_s cost driver (Nyquist-resolve the kink)
        n_grid = int(round(T_MS / dt)) + 1
        t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * dt
        kpr = k_pr(T_MS, TAU_M, tau_s)
        krice = k_rice(T_MS, TAU_M, tau_s)
        # Density grid for this tau_s: the control list at the first tau_s, else the single fixed value.
        if density_list and tau_s == first_tau_s:
            densities = density_list
        else:
            densities = [mean_spikes_fixed]
        for mean_spikes in densities:
            is_density_ctrl = len(densities) > 1
            taus_alpha_psolve: list[float] = []
            emax_cell: float = float("nan")
            for n_aff in n_list:
                for alpha in alphas:
                    cell_i += 1
                    p = round(alpha * n_aff)
                    if p == 0:
                        continue
                    rng = np.random.default_rng(
                        cell_seed(master_seed, n_aff, alpha, tau_s, tag=round(100 * mean_spikes)))
                    spikes, valid, labels, _ = make_patterns(n_seeds, p, n_aff, T_MS, rng, device,
                                                             mean_spikes=mean_spikes)
                    per_task_bytes = n_grid * p * n_aff * 4
                    sb_size = max(1, min(n_seeds, s_budget // max(1, per_task_bytes)))

                    conv_multi = torch.zeros(n_seeds, dtype=torch.bool, device=device)
                    ep_run = torch.full((n_seeds,), float(epochs), device=device)
                    fire = torch.zeros(n_seeds, device=device)
                    errf = torch.zeros(n_seeds, device=device)
                    emax_batches: list[float] = []

                    for b0 in range(0, n_seeds, sb_size):
                        b1 = min(b0 + sb_size, n_seeds)
                        s = precompute_traces(spikes[b0:b1], valid[b0:b1], t_grid, elem_budget)
                        lab_b = labels[b0:b1]
                        # Certify E[max V] on the realized traces (the dt-resolution gate); compare to
                        # the GP witness E[max V] per cell.
                        emax_batches.append(certify_emax_from_traces(s, n_probe=emax_probe))
                        wrng = np.random.default_rng(
                            cell_seed(master_seed, n_aff, alpha, tau_s, tag=1) + b0)
                        w0 = torch.from_numpy(
                            (sigma_w * wrng.standard_normal((b1 - b0, n_aff))).astype(np.float32)).to(device)
                        stream_this = capture and (not first_cell_done)

                        def _mcb(ep: int, errm: float, lossm: float,
                                 _a: float = alpha, _t: float = tau_s) -> None:
                            emit(f"trainerr_tau{_t:.2f}_a{_a:.2f}", errm, ep)
                            emit(f"trainloss_tau{_t:.2f}_a{_a:.2f}", lossm, ep)

                        res = train_batch(
                            s, lab_b, w0, lr=lr, momentum=momentum, epochs=epochs,
                            vthr_fixed=vthr_fixed, patience=patience, log_every=log_every,
                            log_tag=f"tau{tau_s:.2f}N{n_aff}a{alpha:.2f}b{b0}",
                            train_seed=cell_seed(master_seed, n_aff, alpha, tau_s, tag=10) + b0,
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
                                        "tau_s": tau_s, "N": n_aff, "alpha": alpha,
                                        "mean_spikes": mean_spikes, "seed_batch": sd,
                                        "epoch": e_i + 1, "train_err": float(te[j, e_i]),
                                        "loss": float(tl[j, e_i]),
                                        "kappa_plus": float(tkp[j, e_i]),
                                        "kappa_minus": float(tkm[j, e_i]),
                                        "wnorm": float(twn[j, e_i]),
                                    })
                        ep_run[b0:b1] = res["epochs_run"].float()
                        fire[b0:b1] = res["init_fire_rate"]
                        errf[b0:b1] = res["final_errfrac"]

                        # Multi-restart existence-from-below proxy: ANY of n_restarts inits solving counts.
                        any_conv = res["converged"].clone()
                        thr = res["threshold"]
                        for r in range(1, n_restarts):
                            rrng = np.random.default_rng(
                                cell_seed(master_seed, n_aff, alpha, tau_s, tag=2 + r) + b0)
                            wr = torch.from_numpy(
                                (sigma_w * rrng.standard_normal((b1 - b0, n_aff))).astype(np.float32)).to(device)
                            rr = train_batch(
                                s, lab_b, wr, lr=lr, momentum=momentum, epochs=epochs,
                                threshold=thr, patience=patience,
                                train_seed=cell_seed(master_seed, n_aff, alpha, tau_s, tag=20 + r) + b0,
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

                    emax_cell = float(np.mean(emax_batches))
                    n_solved = int(conv_multi.sum().item())
                    p_solve = n_solved / n_seeds
                    if not is_density_ctrl:
                        taus_alpha_psolve.append(p_solve)
                    conv_np = conv_multi.cpu().numpy()
                    ep_np = ep_run.cpu().numpy()
                    fire_np = fire.cpu().numpy()
                    errf_np = errf.cpu().numpy()
                    for j, sd in enumerate(seeds):
                        solved = int(conv_np[j])
                        rows.append({
                            "seed": sd,
                            "tau_s": tau_s,
                            "tau_m": TAU_M,
                            "T": T_MS,
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
                            "is_density_control": int(is_density_ctrl),
                            "K_PR": kpr,
                            "K_Rice": krice,
                            "e_max_realized": emax_cell,
                            "n_restarts": n_restarts,
                            "optimizer": optimizer,
                            "lr_schedule": lr_schedule,
                            "batch_size": batch_size,
                        })
                    _flush_new_rows()  # durable per-cell write (timeout/OOM-safe partial CSV)
                    p_se = math.sqrt(max(p_solve * (1 - p_solve), 0.0) / n_seeds)
                    emit(f"psolve_tau{tau_s:.2f}_N{n_aff}_d{mean_spikes:.1f}", p_solve, round(1000 * alpha))
                    print(
                        f"[{cell_i}/{n_cells}] tau_s={tau_s:.3f} N={n_aff} a={alpha:.3f} P={p} "
                        f"dens={mean_spikes:.2f} p_solve={p_solve:.3f}+/-{p_se:.3f} n_grid={n_grid} "
                        f"K_PR={kpr:.2f} K_Rice={krice:.2f} e_max={emax_cell:.3f} "
                        f"fire0={float(fire.mean().item()):.2f}"
                        + ("  [DENSITY-CTRL]" if is_density_ctrl else ""),
                        flush=True,
                    )
            if not is_density_ctrl and n_list:
                ahat = _crossing_alpha(alphas, taus_alpha_psolve)
                per_taus[tau_s] = {
                    "tau_s": tau_s,
                    "tau_m": TAU_M,
                    "T": T_MS,
                    "n_grid": n_grid,
                    "dt": dt,
                    "mean_spikes": mean_spikes,
                    "K_PR": kpr,
                    "K_Rice": krice,
                    "e_max_realized": emax_cell,
                    "alphas": alphas,
                    "p_solve": taus_alpha_psolve,
                    "alpha_hat_c": ahat,
                    "lnln_K_PR": math.log(math.log(kpr)) if kpr > math.e else float("nan"),
                    "lnln_K_Rice": math.log(math.log(krice)) if krice > math.e else float("nan"),
                }

    elapsed = time.time() - started
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = None
    results = {
        "experiment": "v8_gp_capacity",
        "git_sha": git_sha,
        "versions": {"numpy": np.__version__, "torch": torch.__version__,
                     "python": sys.version.split()[0]},
        "params": {
            "master_seed": master_seed, "T_ms": T_MS, "tau_m": TAU_M, "tau_s_list": tau_s_list,
            "N_list": n_list, "alphas": alphas, "seeds": seeds, "n_seeds": n_seeds,
            "oversample": oversample, "mean_spikes_fixed": mean_spikes_fixed,
            "density_list": density_list, "epochs": epochs, "lr": lr, "momentum": momentum,
            "mode": mode, "optimizer": optimizer, "lr_schedule": lr_schedule,
            "lr_warmup": lr_warmup, "lr_floor": lr_floor, "lr_cycles": lr_cycles,
            "lr_step_size": lr_step_size, "lr_gamma": lr_gamma, "batch_size": batch_size,
            "adam_betas": [adam_b1, adam_b2], "adam_eps": adam_eps,
            "rms_alpha": rms_alpha, "rms_eps": rms_eps, "n_restarts": n_restarts,
            "patience": patience, "sigma_w": sigma_w, "vthr_fixed": vthr_fixed,
            "emax_probe": emax_probe, "psp_v0_last": PSP_V0,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "per_taus": [per_taus[k] for k in tau_s_list if k in per_taus],
        "rows": rows,
        "elapsed_seconds": elapsed,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    _flush_new_rows()  # write any trailing rows; CSV was written incrementally per cell
    csv_file.close()
    if capture and history_rows:
        hcols = ["tau_s", "N", "alpha", "mean_spikes", "seed_batch", "epoch",
                 "train_err", "loss", "kappa_plus", "kappa_minus", "wnorm"]
        with (run_dir / "history.csv").open("w", newline="") as hf:
            hwtr = csv.DictWriter(hf, fieldnames=hcols)
            hwtr.writeheader()
            hwtr.writerows(history_rows)
    print(f"done in {elapsed:.1f}s -> {run_dir}/results.json ({len(rows)} rows, "
          f"{len(per_taus)} tau_s cells)"
          + (f", history.csv ({len(history_rows)} epoch-rows)" if capture and history_rows else ""),
          flush=True)
    for k in tau_s_list:
        if k in per_taus:
            pk = per_taus[k]
            print(f"  tau_s={k:.3f} K_PR={pk['K_PR']:.2f} K_Rice={pk['K_Rice']:.2f} "
                  f"e_max={pk['e_max_realized']:.3f} alpha_hat_c={pk['alpha_hat_c']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
