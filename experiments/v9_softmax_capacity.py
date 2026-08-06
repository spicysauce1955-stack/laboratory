"""V9 -- Soft-max(beta) K-bin tempotron capacity alpha_c(beta) = 2 + C_K(rho) beta^2.

Self-contained Experiment-Contract entrypoint (numpy + torch only; the lab ships this file
unchanged, so there are NO ``tempotron`` / ``lab`` imports -- everything is inlined here). This is
the reduction-#5 study: the hard ``max_t V(t)`` tempotron readout is replaced by the smooth
log-sum-exp (soft-max) at inverse temperature ``beta``, and -- crucially -- the spike->PSP->trace
pipeline is DROPPED entirely. There is no time grid, no kernel, no traces. The model is the clean
``K-bin direct field`` reduction of derivation #5 section 1.

The science (see ``lit/tempotron-reductions/derivations/05-softmax-capacity-prediction.md``)
--------------------------------------------------------------------------------------------------
Per pattern mu we draw K correlated Gaussian "bin fields" directly,

    V^mu = (V_1, ..., V_K) = w . X^mu,   X^mu in R^{K x N} with rows equicorrelated (corr rho),

scaled so each V_k ~ N(0,1) under ||w||=sqrt(N) and <V_j V_k> = rho for j != k. We realize the
equicorrelation with the standard one-factor construction

    x_k = sqrt(rho) g_common + sqrt(1-rho) g_k,    g_common, g_k ~ N(0, I_N / N),

so V_k = w . x_k has variance ||w||^2/N = 1 and corr(V_j,V_k) = rho (j != k). This is the minimal
model that still carries the beta knob (soft-max temperature) and the correlation knob rho, with NO
kernel confound and NO density (#6) confound -- it is a direct K-field model, not a spike-count
model, so there is no 2N/K starvation.

Readout (CENTERED log-sum-exp -- load-bearing; keeps theta_beta O(1) at all beta):

    V_beta = (1/beta) [ logsumexp(beta V_k over k) - ln K ]
           = (1/beta) ln[ (1/K) sum_k e^{beta V_k} ].

V_beta -> mean_k V_k as beta -> 0 (the perceptron/Gardner anchor, alpha_c = 2) and -> max_k V_k as
beta -> inf (the hard-max tempotron). Decision: fire iff V_beta >= theta_beta, with theta_beta set to
the MEDIAN of V_beta at the random init -> P(fire) = 1/2 (the balanced-threshold invariant, A5), then
held fixed. Balanced +/-1 labels.

The prediction (small-beta RS law, codex m5; HEURISTIC coefficient -- see the note):

    alpha_c(beta) = 2 + C_K(rho) beta^2,   C_K(rho) = (K-1)(1-rho)^2 / [K(1 + (K-1) rho)],

valid for beta <~ beta* = sqrt(2 ln K) (beta*(K=2)=1.18, beta*(K=3)=1.48); beyond beta* the surface
fragments (RSB) and the rise saturates toward the hard-max ceiling (alpha_c(inf) ~ 2.86/3.44 for
K=2/3). The decisive, offset-robust observable is the beta^2 SLOPE C_K (anchored at the free
beta->0 = 2 point): C_2(0)=0.5, C_3(0)=0.667, C_2(0.3)=0.189, C_3(0.3)=0.204.

What this study measures
------------------------
FINDABLE capacity (derivation #5 section 1: the soft-max margin is C^infty in w and effectively
convex at small beta, so a gradient finder tracks existence there; an existence oracle is non-convex
for any beta>0 -- intractable). For each cell (K, rho, beta, N, alpha) we draw ``n_seeds`` independent
tasks (P = round(alpha N) patterns each), calibrate theta_beta = median V_beta at init, and train w on
the SMOOTH HINGE on the soft-max margin

    L(w) = (1/P) sum_mu relu( kappa - y_mu (V_beta^mu - theta_beta) )

(margin target kappa=0 by default; the hinge is sub-differentiable, the readout V_beta is smooth, so
torch autograd gives a clean gradient) by Adam under a cosine LR schedule (the program's near-universal
strong learner; reused from v3/v6). A task counts SOLVED if ANY of ``n_restarts`` random inits reaches
zero TRAINING error (existence-from-below proxy). ``P_solve(alpha)`` is the solved fraction; its
1/2-crossing in alpha is ``alpha_hat_c(beta, K, rho)``.

The learner machinery (Adam, cosine/linear/none schedule, warmup, floor, per-epoch capture, per-cell
seeding, the Experiment Contract argv/LAB_RUN_DIR/seeds= conventions, require_cuda) mirrors
``v6_bandlimited_capacity.py`` so the harness is familiar; the ONLY change is the model (direct K-bin
fields + soft-max readout) and that training uses autograd on the smooth readout rather than the
hand-coded hard-max GS subgradient.

Controls (the #5 prediction-note / playbook lessons), in priority order
-----------------------------------------------------------------------
- **beta->0 anchor** (must hold or the harness is wrong): the beta->0 cell reproduces alpha_c = 2.0
  (centered V_beta -> mean -> single Gaussian field -> Gardner perceptron, convex). Free calibration.
- **Per-cell HPO (THE confound here)**: beta changes the loss conditioning (curvature ~ beta^2), so the
  optimal lr is beta-dependent -- fixed HPs would differentially under-tune large-beta cells and ERASE
  the rise. Tune lr per beta (or at minimum verify the argmax-HP is shared at beta=0.25 and beta=2.0).
  Lower beta should need a smaller lr (flatter loss).
- **rho control**: rho=0 vs 0.3 must show the (1-rho)^2 suppression (K=3: 0.667 -> 0.204, a 3.3x drop).
- **Balanced threshold**: theta_beta = median V_beta at init -> init_fire_rate ~ 0.5 (logged per cell).
- **Restart overlap** (the fragmentation witness, beta* onset): logged as ``restart_overlap`` (mean
  cosine of independent converged restart weight vectors); expect ~1 below beta*, dropping above.

Output: ``results.csv`` per (seed, K, rho, beta, alpha) row:
    seed, K, rho, beta, N, alpha, P, solved, epochs_to_solve, init_fire_rate, final_train_err, ...
plus ``history.csv`` (tidy long-format per-epoch learning curves) when ``capture=1``.

Run standalone (tiny smoke; does NOT need a GPU):
    LAB_RUN_DIR=/tmp/v9 uv run --with torch python studies/v9_softmax_capacity.py \\
        K_list=3 rho=0.0 beta_list=0.1,1.0 N_list=40 alphas=1.6,2.0,2.4 seeds=0,1,2,3 \\
        mode=minibatch optimizer=adam lr_schedule=cosine batch_size=32 lr=0.1 epochs=400

Real capacity run (GPU on the lab) -- see EXACT_RUN_ARGV at the bottom of this docstring section.
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


# --------------------------------------------------------------------------------------------
# Soft-max(beta) coefficient (analytic prediction, for logging / verification)
# --------------------------------------------------------------------------------------------
def C_K(K: int, rho: float) -> float:
    """RS small-beta coefficient C_K(rho) = (K-1)(1-rho)^2 / [K (1 + (K-1) rho)] (codex m5).

    Anchors: C_K(1) = 0 (all bins identical -> one field), K=1 -> 0 (no bins). C_2(0)=0.5,
    C_3(0)=2/3, C_2(0.3)=0.1885, C_3(0.3)=0.2042.
    """
    if K <= 1:
        return 0.0
    return (K - 1) * (1.0 - rho) ** 2 / (K * (1.0 + (K - 1) * rho))


def beta_star(K: int) -> float:
    """Fragmentation onset beta* = sqrt(2 ln K) (EVT crossover; #5 section 3)."""
    return math.sqrt(2.0 * math.log(K)) if K > 1 else float("inf")


# --------------------------------------------------------------------------------------------
# Reproducible per-cell seeding (matches v3/v6 cell_seed recipe; cell keyed by K,rho,beta too)
# --------------------------------------------------------------------------------------------
def cell_seed(master: int, n: int, alpha: float, k: int, rho: float, beta: float, tag: int = 0) -> int:
    """A scheduling-independent 63-bit seed for a cell (extends v6's SeedSequence recipe).

    Keyed by (master, N, round(1000 alpha), K, round(1000 rho), round(1000 beta), tag) so every
    (K, rho, beta, alpha, N) cell -- and every role (patterns/init/restart/shuffle) via ``tag`` --
    gets an independent, schedule-order-independent stream. Same inputs => byte-identical results.
    """
    ss = np.random.SeedSequence(
        [master, n, round(1000 * alpha), int(k), round(1000 * rho), round(1000 * beta), tag]
    )
    return int(ss.generate_state(1, dtype=np.uint64)[0]) >> 1


# --------------------------------------------------------------------------------------------
# K-bin equicorrelated Gaussian field model (the reduction-#5 direct feature map)
# --------------------------------------------------------------------------------------------
def make_fields(
    n_tasks: int,
    n_patterns: int,
    n_aff: int,
    K: int,
    rho: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``n_tasks`` independent K-bin field tasks, balanced +/-1 labels.

    One-factor equicorrelation: for each pattern, K bin-field input vectors

        x_k = sqrt(rho) g_common + sqrt(1-rho) g_k,   g_common, g_k iid N(0, I_N / N),

    so under ||w|| = sqrt(N) the readouts V_k = w . x_k satisfy Var(V_k) = ||w||^2/N = 1 and
    corr(V_j, V_k) = rho for j != k (the equicorrelation matrix C = (1-rho)I + rho 11^T). At rho=0
    the K bins are independent; at rho=1 they collapse to a single field (alpha_c=2 for all beta).

    Returns ``(X, labels)``: ``X`` of shape ``(n_tasks, P, K, N)`` (float32, scaled by 1/sqrt(N) so
    a unit-norm-in-the-||w||=sqrt(N)-gauge weight gives unit-variance fields) and ``labels`` of
    shape ``(n_tasks, P)`` in {-1,+1}. Drawn on the CPU via a numpy Generator then moved to
    ``device`` (device-INDEPENDENT, bit-reproducible on any backend).
    """
    inv_sqrt_n = 1.0 / math.sqrt(n_aff)
    sr = math.sqrt(rho)
    s1r = math.sqrt(max(0.0, 1.0 - rho))
    # g_common shared across the K bins of a pattern; g_k per-bin. Both ~ N(0, I_N/N).
    g_common = rng.standard_normal((n_tasks, n_patterns, 1, n_aff)).astype(np.float32) * inv_sqrt_n
    g_bins = rng.standard_normal((n_tasks, n_patterns, K, n_aff)).astype(np.float32) * inv_sqrt_n
    X = sr * g_common + s1r * g_bins  # (n_tasks, P, K, N), broadcast over the K axis
    labels = np.where(rng.random((n_tasks, n_patterns)) < 0.5, 1.0, -1.0).astype(np.float32)
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return to(X), to(labels)


def softmax_readout(X: torch.Tensor, w: torch.Tensor, beta: float) -> torch.Tensor:
    """Centered soft-max readout V_beta = (1/beta)[logsumexp(beta V_k) - ln K] over the K axis.

    ``V[b,p,k] = sum_i X[b,p,k,i] w[b,i]``; then V_beta[b,p] = (1/beta) ln[(1/K) sum_k e^{beta V_k}].
    Numerically stable via torch.logsumexp (subtracts the per-pattern max internally). As beta->0
    this -> mean_k V_k; as beta->inf it -> max_k V_k. ``w`` is per-task ``(Sb, N)``.
    """
    V = torch.einsum("bpkn,bn->bpk", X, w)  # (Sb, P, K)
    K = V.shape[2]
    # (1/beta) [ logsumexp_k(beta V) - ln K ]
    return (torch.logsumexp(beta * V, dim=2) - math.log(K)) / beta  # (Sb, P)


# --------------------------------------------------------------------------------------------
# LR schedule (copied verbatim from v6_bandlimited_capacity.scheduled_lr)
# --------------------------------------------------------------------------------------------
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
    """Scheduled learning rate at epoch ``ep`` (v6's tuned LR schedule, byte-faithful).

    Linear warmup over the first ``warmup`` epochs, then ``none``/``cosine``/``linear``/``exp``/
    ``step`` decay over ``[warmup, epochs]`` (cosine/linear anneal lr -> floor*lr; n_cycles>1 = SGDR).
    Defaults (floor=0, n_cycles=1) anneal to 0 in one cycle.
    """
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


# --------------------------------------------------------------------------------------------
# Forward error / training loop (Adam/RMSprop/momentum minibatch on the SMOOTH soft-max hinge)
# --------------------------------------------------------------------------------------------
def _train_err(X: torch.Tensor, w: torch.Tensor, threshold: torch.Tensor, labels: torch.Tensor,
               beta: float) -> torch.Tensor:
    """Hard 0/1 training-error fraction per task: pred = sign(V_beta - theta), err = pred != y.

    ``threshold`` is ``(Sb,)``, ``labels`` is ``(Sb, P)``. Returns ``(Sb,)`` error fraction.
    """
    Vb = softmax_readout(X, w, beta)  # (Sb, P)
    pred = torch.where(Vb >= threshold[:, None], 1.0, -1.0)
    return (pred != labels).float().mean(dim=1)


def train_batch(
    X: torch.Tensor,        # (Sb, P, K, N) equicorrelated bin fields
    labels: torch.Tensor,   # (Sb, P) in {-1,+1}
    w_init: torch.Tensor,   # (Sb, N)
    beta: float,
    *,
    lr: float,
    epochs: int,
    threshold: torch.Tensor | None = None,  # (Sb,) fixed theta_beta; if None, calibrate = median V_beta at init
    kappa: float = 0.0,     # smooth-hinge margin target
    patience: int = 0,      # if >0, early-stop a cell after this many no-progress epochs
    log_every: int = 0,
    log_tag: str = "",
    train_seed: int = 0,    # seeds the per-epoch shuffle -> deterministic training
    capture: bool = False,
    metric_cb=None,
    optimizer: str = "adam",      # adam | rmsprop | momentum
    lr_schedule: str = "cosine",  # none | cosine | linear | exp | step
    lr_warmup: int = 0,
    lr_floor: float = 0.0,
    lr_cycles: int = 1,
    lr_exp_k: float = EXP_DECAY_K,
    lr_step_size: int = 0,
    lr_gamma: float = 0.1,
    momentum: float = 0.9,
    batch_size: int = 0,          # b in {1..P}; 0 => full batch
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    rms_alpha: float = 0.99,
    rms_eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Train w on the smooth soft-max hinge, vectorised across tasks. threshold fixed at init.

    The soft-max readout V_beta is C^infty in w; the per-pattern hinge ``relu(kappa - y(V_beta-theta))``
    is sub-differentiable. We descend the per-minibatch MEAN hinge with a selectable optimizer
    (Adam/RMSprop/momentum) under a cosine/linear/none LR schedule -- the program's strong-learner
    machinery (ported from v6.train_batch). The gradient is computed by torch autograd through the
    logsumexp readout (clean and exact; no hand-coded subgradient). theta_beta is calibrated to the
    median V_beta at the random init (balanced P(fire)=1/2), then held fixed. Convergence = a full
    epoch with ZERO hard training errors (the existence-from-below proxy).

    Per-epoch capture (``capture=True``): records, per epoch, the seed-batch trajectories ``traj_err``
    (hard 0/1 error fraction), ``traj_loss`` (mean smooth hinge), ``traj_kp``/``traj_km`` (mean signed
    soft-max margin on +1/-1 patterns), ``traj_wnorm`` (||w||), each ``(Sb, ne)``.
    """
    sb, p, K, n = X.shape
    dev = X.device
    shuffle_gen = torch.Generator(device=dev)
    shuffle_gen.manual_seed(int(train_seed) & 0x7FFFFFFFFFFFFFFF)

    w = w_init.clone()
    with torch.no_grad():
        Vb0 = softmax_readout(X, w, beta)  # (Sb, P)
        if threshold is None:
            threshold = Vb0.median(dim=1).values  # (Sb,) median V_beta at init -> P(fire)=1/2
        init_fire_rate = (Vb0 >= threshold[:, None]).float().mean(dim=1)

    converged = torch.zeros(sb, dtype=torch.bool, device=dev)
    epochs_run = torch.full((sb,), epochs, dtype=torch.int64, device=dev)
    vel = torch.zeros_like(w)
    adam_m = torch.zeros_like(w)
    adam_v = torch.zeros_like(w)
    adam_step = 0
    best_conv = 0
    stall = 0
    last_ep = -1

    cap = bool(capture)
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
                            step_size=lr_step_size, gamma=lr_gamma, floor=lr_floor,
                            n_cycles=lr_cycles, exp_k=lr_exp_k)

    for ep in range(epochs):
        lr_ep = _lr_at(ep)
        order = torch.randperm(p, device=dev, generator=shuffle_gen)
        bs = p if batch_size <= 0 else max(1, min(batch_size, p))
        n_steps = (p + bs - 1) // bs
        active = (~converged).float().unsqueeze(-1)  # (Sb,1)

        if cap:
            err_acc = torch.zeros(sb, device=dev)
            cost_acc = torch.zeros(sb, device=dev)
            kp_acc = torch.zeros(sb, device=dev)
            km_acc = torch.zeros(sb, device=dev)

        for st_i in range(n_steps):
            bidx = order[st_i * bs:(st_i + 1) * bs]
            Xb = X[:, bidx]            # (Sb, b, K, N)
            lab_b = labels[:, bidx]    # (Sb, b)
            bb = Xb.shape[1]

            w_req = w.detach().requires_grad_(True)
            Vb = softmax_readout(Xb, w_req, beta)              # (Sb, b)
            smarg = (Vb - threshold[:, None]) * lab_b          # (Sb, b) signed soft-max margin
            hinge = torch.relu(kappa - smarg)                  # (Sb, b)
            loss = hinge.mean()                                # scalar (summed across the seed batch)
            grad, = torch.autograd.grad(loss, w_req)           # (Sb, N) descent direction d loss / d w
            g = -grad  # ascend -loss == descend loss; keep the optimizer sign convention as "step += g"

            # A minibatch with no margin-violator (all hinge==0) contributes zero grad -> no update,
            # so the per-seed gate is implicit in g; we still gate by ``active`` to freeze solved seeds.
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
            else:  # momentum
                vel = momentum * vel + g
                step = lr_ep * vel
            w = w + step * active

            if cap:
                with torch.no_grad():
                    err_acc = err_acc + ((smarg.detach() < 0.0).float().sum(dim=1))
                    cost_acc = cost_acc + hinge.detach().sum(dim=1)
                    kp_acc = kp_acc + (smarg.detach() * (lab_b > 0).float()).sum(dim=1)
                    km_acc = km_acc + (smarg.detach() * (lab_b < 0).float()).sum(dim=1)

        last_ep = ep
        with torch.no_grad():
            errfrac = _train_err(X, w, threshold, labels, beta)  # (Sb,) hard error fraction
        newly = (errfrac == 0.0) & (~converged) & torch.isfinite(w).all(dim=1)
        epochs_run = torch.where(newly, torch.tensor(ep + 1, device=dev), epochs_run)
        converged = converged | newly

        if cap:
            err_buf[:, ep] = err_acc / p
            loss_buf[:, ep] = cost_acc / p
            wnorm_buf[:, ep] = w.norm(dim=1)
            kp_buf[:, ep] = kp_acc / npos
            km_buf[:, ep] = km_acc / nneg
            if metric_cb is not None:
                metric_cb(ep + 1, float((err_acc / p).mean()), float((cost_acc / p).mean()))

        if bool(converged.all()):
            break
        n_conv = int(converged.sum().item())
        if n_conv > best_conv:
            best_conv = n_conv
            stall = 0
        else:
            stall += 1
        if log_every and (ep + 1) % log_every == 0:
            print(f"    [{log_tag}] ep={ep + 1} conv={n_conv}/{sb} stall={stall} lr={lr_ep:.4f}",
                  flush=True)
        if patience and stall >= patience and n_conv < sb:
            if log_every:
                print(f"    [{log_tag}] early-stop at ep={ep + 1} (no progress {stall} ep)", flush=True)
            break

    ran = last_ep + 1
    epochs_run = torch.where(converged, epochs_run, torch.full_like(epochs_run, ran))
    with torch.no_grad():
        final_errfrac = _train_err(X, w, threshold, labels, beta)
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
    pairs = sorted(zip(alphas, psolve), key=lambda t: t[0])
    a = [x for x, _ in pairs]
    p = [y for _, y in pairs]
    for i in range(len(a) - 1):
        a0, a1 = a[i], a[i + 1]
        p0, p1 = p[i], p[i + 1]
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
    k_list = [int(x) for x in ov.get("K_list", "2,3").split(",")]
    rho = float(ov.get("rho", "0.0"))
    beta_list = [float(x) for x in ov.get("beta_list", "0.25,0.5,0.75,1.0,1.25,1.5,2.0").split(",")]
    n_list = [int(x) for x in ov.get("N_list", "500").split(",")]
    alphas = _alpha_grid(ov)
    if "seeds" in ov:
        seeds = [int(x) for x in ov["seeds"].split(",") if x != ""]
    else:
        seeds = list(range(int(ov.get("n_seeds", "24"))))
    n_seeds = len(seeds)

    kappa = float(ov.get("kappa", "0.0"))
    capture = int(ov.get("capture", "0"))
    epochs = int(ov.get("epochs", "2000"))
    lr = float(ov.get("lr", "0.1"))
    optimizer = ov.get("optimizer", "adam")
    lr_schedule = ov.get("lr_schedule", "cosine")
    lr_warmup = int(ov.get("lr_warmup", "0"))
    lr_floor = float(ov.get("lr_floor", "0.0"))
    lr_cycles = int(ov.get("lr_cycles", "1"))
    lr_exp_k = float(ov.get("lr_exp_k", str(EXP_DECAY_K)))
    lr_step_size = int(ov.get("lr_step_size", "0"))
    lr_gamma = float(ov.get("lr_gamma", "0.1"))
    momentum = float(ov.get("momentum", "0.9"))
    batch_size = int(ov.get("batch_size", "32"))
    adam_b1 = float(ov.get("adam_b1", "0.9"))
    adam_b2 = float(ov.get("adam_b2", "0.999"))
    adam_eps = float(ov.get("adam_eps", "1e-8"))
    rms_alpha = float(ov.get("rms_alpha", "0.99"))
    rms_eps = float(ov.get("rms_eps", "1e-8"))
    n_restarts = int(ov.get("n_restarts", "5"))
    patience = int(ov.get("patience", "0"))
    log_every = int(ov.get("log_every", "0"))
    sigma_w = float(ov.get("sigma_w", "1.0"))
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
        f"V9 softmax(beta) capacity | seed={master_seed} device={device} K={k_list} rho={rho} "
        f"beta={beta_list} N={n_list} alphas={alphas} seeds={seeds} epochs={epochs} lr={lr} "
        f"optimizer={optimizer} lr_schedule={lr_schedule} batch_size={batch_size} kappa={kappa} "
        f"restarts={n_restarts}",
        flush=True,
    )
    for K in k_list:
        print(f"  predicted: K={K} rho={rho} C_K={C_K(K, rho):.4f} beta*={beta_star(K):.3f} "
              f"alpha_c(beta) = 2 + {C_K(K, rho):.4f} beta^2 (valid beta<~beta*)", flush=True)

    _log_metric = None
    if capture:
        try:
            from lab.metrics import log_metric as _log_metric  # type: ignore
        except Exception:
            _log_metric = None

    started = time.time()
    rows: list[dict[str, float]] = []
    history_rows: list[dict[str, float]] = []
    per_cell: list[dict] = []
    n_cells = len(k_list) * len(beta_list) * len(n_list) * len(alphas)
    cell_i = 0
    first_cell_done = False

    # Incremental, durable results.csv: write the header once (lazily, when the first cell's rows
    # fix the column order) and flush+fsync after EACH cell so a timeout/OOM kill leaves a valid,
    # readable partial CSV with all completed cells. The in-memory ``rows`` list is still kept for
    # the end-of-run results.json. ``_csv_written`` tracks how many rows are already on disk.
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

    for K in k_list:
        for beta in beta_list:
            for n_aff in n_list:
                cell_psolve: list[float] = []
                for alpha in alphas:
                    cell_i += 1
                    p = round(alpha * n_aff)
                    if p == 0:
                        continue
                    rng = np.random.default_rng(cell_seed(master_seed, n_aff, alpha, K, rho, beta))
                    X, labels = make_fields(n_seeds, p, n_aff, K, rho, rng, device)

                    per_task_bytes = p * K * n_aff * 4
                    sb_size = max(1, min(n_seeds, s_budget // max(1, per_task_bytes)))

                    conv_multi = torch.zeros(n_seeds, dtype=torch.bool, device=device)
                    ep_run = torch.full((n_seeds,), float(epochs), device=device)
                    fire = torch.zeros(n_seeds, device=device)
                    errf = torch.zeros(n_seeds, device=device)
                    overlap_acc: list[float] = []  # restart-overlap witness (mean |cos| of converged restarts)

                    for b0 in range(0, n_seeds, sb_size):
                        b1 = min(b0 + sb_size, n_seeds)
                        Xb = X[b0:b1]
                        lab_b = labels[b0:b1]
                        wrng = np.random.default_rng(
                            cell_seed(master_seed, n_aff, alpha, K, rho, beta, tag=1) + b0)
                        w0 = torch.from_numpy(
                            (sigma_w * wrng.standard_normal((b1 - b0, n_aff))).astype(np.float32)).to(device)

                        stream_this = capture and (not first_cell_done) and (_log_metric is not None)

                        def _mcb(ep: int, errm: float, lossm: float) -> None:
                            _log_metric("train_err", float(errm), step=int(ep))  # type: ignore
                            _log_metric("loss", float(lossm), step=int(ep))      # type: ignore

                        res = train_batch(
                            Xb, lab_b, w0, beta, lr=lr, epochs=epochs, kappa=kappa,
                            patience=patience, log_every=log_every,
                            log_tag=f"K{K}b{beta:.2f}N{n_aff}a{alpha:.2f}s{b0}",
                            train_seed=cell_seed(master_seed, n_aff, alpha, K, rho, beta, tag=10) + b0,
                            capture=bool(capture), metric_cb=_mcb if stream_this else None,
                            optimizer=optimizer, lr_schedule=lr_schedule, lr_warmup=lr_warmup,
                            lr_floor=lr_floor, lr_cycles=lr_cycles, lr_exp_k=lr_exp_k,
                            lr_step_size=lr_step_size, lr_gamma=lr_gamma, momentum=momentum,
                            batch_size=batch_size, adam_betas=(adam_b1, adam_b2), adam_eps=adam_eps,
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
                                        "K": K, "rho": rho, "beta": beta, "N": n_aff, "alpha": alpha,
                                        "seed": sd, "epoch": e_i + 1,
                                        "train_err": float(te[j, e_i]), "loss": float(tl[j, e_i]),
                                        "kappa_plus": float(tkp[j, e_i]),
                                        "kappa_minus": float(tkm[j, e_i]),
                                        "wnorm": float(twn[j, e_i]),
                                    })

                        ep_run[b0:b1] = res["epochs_run"].float()
                        fire[b0:b1] = res["init_fire_rate"]
                        errf[b0:b1] = res["final_errfrac"]
                        thr = res["threshold"]

                        # Existence-from-below proxy + restart-overlap witness. Keep the first-init
                        # converged weight per seed; for each later restart, if it converges record the
                        # |cosine| with the first converged restart's weight (fragmentation probe).
                        any_conv = res["converged"].clone()
                        w_first = res["weights"].clone()
                        first_w_conv = res["converged"].clone()
                        for r in range(1, n_restarts):
                            rrng = np.random.default_rng(
                                cell_seed(master_seed, n_aff, alpha, K, rho, beta, tag=2 + r) + b0)
                            wr = torch.from_numpy(
                                (sigma_w * rrng.standard_normal((b1 - b0, n_aff))).astype(np.float32)).to(device)
                            rr = train_batch(
                                Xb, lab_b, wr, beta, lr=lr, epochs=epochs, kappa=kappa,
                                threshold=thr, patience=patience,
                                train_seed=cell_seed(master_seed, n_aff, alpha, K, rho, beta, tag=20 + r) + b0,
                                optimizer=optimizer, lr_schedule=lr_schedule, lr_warmup=lr_warmup,
                                lr_floor=lr_floor, lr_cycles=lr_cycles, lr_exp_k=lr_exp_k,
                                lr_step_size=lr_step_size, lr_gamma=lr_gamma, momentum=momentum,
                                batch_size=batch_size, adam_betas=(adam_b1, adam_b2), adam_eps=adam_eps,
                                rms_alpha=rms_alpha, rms_eps=rms_eps)
                            newly_r = rr["converged"] & ~any_conv
                            ep_run[b0:b1] = torch.where(newly_r, rr["epochs_run"].float(), ep_run[b0:b1])
                            # restart-overlap: both this restart and the first init converged
                            both = (rr["converged"] & first_w_conv)
                            if bool(both.any()):
                                wa = w_first[both]
                                wb = rr["weights"][both]
                                cos = torch.nn.functional.cosine_similarity(wa, wb, dim=1).abs()
                                overlap_acc.extend(cos.detach().cpu().numpy().tolist())
                            any_conv = any_conv | rr["converged"]
                            if bool(any_conv.all()):
                                break
                        conv_multi[b0:b1] = any_conv
                        del Xb
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    n_solved = int(conv_multi.sum().item())
                    p_solve = n_solved / n_seeds
                    cell_psolve.append(p_solve)
                    restart_overlap = float(np.mean(overlap_acc)) if overlap_acc else float("nan")
                    conv_np = conv_multi.cpu().numpy()
                    ep_np = ep_run.cpu().numpy()
                    fire_np = fire.cpu().numpy()
                    errf_np = errf.cpu().numpy()
                    for j, sd in enumerate(seeds):
                        solved = int(conv_np[j])
                        rows.append({
                            "seed": sd, "K": K, "rho": rho, "beta": beta, "N": n_aff,
                            "alpha": alpha, "P": p, "solved": solved,
                            "epochs_to_solve": int(ep_np[j]) if solved else -1,
                            "init_fire_rate": float(fire_np[j]),
                            "final_train_err": float(errf_np[j]),
                            "restart_overlap": restart_overlap,
                            "C_K": C_K(K, rho), "beta_star": beta_star(K),
                            "n_restarts": n_restarts, "optimizer": optimizer,
                            "lr_schedule": lr_schedule, "batch_size": batch_size, "lr": lr,
                        })
                    _flush_new_rows()  # durable per-cell write (timeout/OOM-safe partial CSV)
                    p_se = math.sqrt(max(p_solve * (1 - p_solve), 0.0) / n_seeds)
                    print(
                        f"[{cell_i}/{n_cells}] K={K} rho={rho} beta={beta:.3f} N={n_aff} "
                        f"a={alpha:.3f} P={p} p_solve={p_solve:.3f}+/-{p_se:.3f} "
                        f"fire0={float(fire.mean().item()):.3f} overlap={restart_overlap:.3f}",
                        flush=True,
                    )
                ahat = _crossing_alpha(alphas, cell_psolve)
                pred = 2.0 + C_K(K, rho) * beta * beta
                per_cell.append({
                    "K": K, "rho": rho, "beta": beta, "N": n_aff, "alphas": alphas,
                    "p_solve": cell_psolve, "alpha_hat_c": ahat,
                    "C_K": C_K(K, rho), "beta_star": beta_star(K),
                    "alpha_pred_quadratic": pred,
                })
                print(f"  >> K={K} rho={rho} beta={beta:.3f} N={n_aff} alpha_hat_c={ahat:.3f} "
                      f"(quad-pred={pred:.3f}, valid={'yes' if beta <= beta_star(K) else 'PAST beta*'})",
                      flush=True)

    elapsed = time.time() - started
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = None
    results = {
        "experiment": "v9_softmax_capacity",
        "git_sha": git_sha,
        "versions": {"numpy": np.__version__, "torch": torch.__version__,
                     "python": sys.version.split()[0]},
        "params": {
            "master_seed": master_seed, "K_list": k_list, "rho": rho, "beta_list": beta_list,
            "N_list": n_list, "alphas": alphas, "seeds": seeds, "n_seeds": n_seeds, "kappa": kappa,
            "epochs": epochs, "lr": lr, "optimizer": optimizer, "lr_schedule": lr_schedule,
            "lr_warmup": lr_warmup, "lr_floor": lr_floor, "lr_cycles": lr_cycles,
            "lr_step_size": lr_step_size, "lr_gamma": lr_gamma, "momentum": momentum,
            "batch_size": batch_size, "adam_betas": [adam_b1, adam_b2], "adam_eps": adam_eps,
            "rms_alpha": rms_alpha, "rms_eps": rms_eps, "n_restarts": n_restarts,
            "patience": patience, "sigma_w": sigma_w, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "per_cell": per_cell,
        "rows": rows,
        "elapsed_seconds": elapsed,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    _flush_new_rows()  # write any trailing rows; CSV was written incrementally per cell
    csv_file.close()
    if capture and history_rows:
        hcols = ["K", "rho", "beta", "N", "alpha", "seed", "epoch",
                 "train_err", "loss", "kappa_plus", "kappa_minus", "wnorm"]
        with (run_dir / "history.csv").open("w", newline="") as hf:
            hwtr = csv.DictWriter(hf, fieldnames=hcols)
            hwtr.writeheader()
            hwtr.writerows(history_rows)
    print(f"done in {elapsed:.1f}s -> {run_dir}/results.json ({len(rows)} rows, {len(per_cell)} cells)"
          + (f", history.csv ({len(history_rows)} epoch-rows)" if capture and history_rows else ""),
          flush=True)
    for c in per_cell:
        print(f"  K={c['K']} rho={c['rho']} beta={c['beta']:.3f} N={c['N']} "
              f"alpha_hat_c={c['alpha_hat_c']:.3f} (quad-pred {c['alpha_pred_quadratic']:.3f})",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
