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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Draw ``n_tasks`` independent pattern sets.

    Each afferent emits a homogeneous Poisson train on ``[0, T]`` at rate ``1/T`` (mean 1 spike);
    labels are +/-1 with equal probability.

    Returns ``(spike_times, valid, labels, max_spikes)`` with
    ``spike_times, valid`` of shape ``(n_tasks, P, N, max_spikes)`` and ``labels`` of ``(n_tasks, P)``.
    """
    shape = (n_tasks, n_patterns, n_aff)
    counts = torch.poisson(torch.ones(shape, device=device), generator=gen).to(torch.int64)
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
) -> dict[str, torch.Tensor]:
    """Gutig-Sompolinsky rule, vectorised across tasks. Threshold fixed (median V_max at init).

    ``mode='online'`` (default, what RMS used): per-pattern updates within a shuffled epoch; a task
    is converged once a full epoch passes with **zero updates** (the textbook perceptron criterion),
    which is detected without an extra forward pass. ``mode='batch'`` sums the misclassified updates
    and applies them once per epoch (vectorised but oscillation-prone near capacity; kept for study).

    Returns per-task tensors: ``converged``, ``epochs_run``, ``wnorm``, ``mean_margin``,
    ``threshold``, ``init_fire_rate``, ``final_errfrac``.
    """
    sb, p, _, n = s.shape
    dev = s.device
    w = w_init.clone()
    vmax, _ = _forward(s, w)
    if threshold is None:
        threshold = vmax.median(dim=1).values  # (Sb,)
    init_fire_rate = (vmax >= threshold[:, None]).float().mean(dim=1)
    converged = torch.zeros(sb, dtype=torch.bool, device=dev)
    epochs_run = torch.full((sb,), epochs, dtype=torch.int64, device=dev)
    arange_sb = torch.arange(sb, device=dev)
    vel = torch.zeros_like(w)

    for ep in range(epochs):
        active = (~converged).float().unsqueeze(-1)  # (Sb, 1)
        if mode == "online":
            order = torch.randperm(p, device=dev)
            updates = torch.zeros(sb, device=dev)  # #updates this epoch (per task)
            for pi in order:
                sp = s[:, pi]                               # (Sb, G, N) contiguous
                vp = torch.einsum("bgn,bn->bg", sp, w)
                vmx, tg = vp.max(dim=1)                     # (Sb,)
                pred_p = torch.where(vmx >= threshold, 1.0, -1.0)
                e = (pred_p != labels[:, pi]).float() * active.squeeze(-1)  # 1 on active errors
                updates = updates + e
                grad = sp[arange_sb, tg]                    # (Sb, N) = s_i(t_max)
                w = w + (lr * labels[:, pi] * e).unsqueeze(-1) * grad
            newly = (updates == 0) & (~converged)
        else:  # batch mode
            vmax, targ = _forward(s, w)
            err = (torch.where(vmax >= threshold[:, None], 1.0, -1.0) != labels).float()
            idx = targ.view(sb, p, 1, 1).expand(sb, p, 1, n)
            grad = torch.gather(s, 2, idx).squeeze(2)       # (Sb, P, N)
            dw = lr * ((err * labels).unsqueeze(-1) * grad).sum(dim=1)
            vel = momentum * vel + dw
            w = w + vel * active
            newly = (err.sum(dim=1) == 0) & (~converged)
        epochs_run = torch.where(newly, torch.tensor(ep, device=dev), epochs_run)
        converged = converged | newly
        if bool(converged.all()):
            break

    vmax, _ = _forward(s, w)
    pred = torch.where(vmax >= threshold[:, None], 1.0, -1.0)
    final_errfrac = (pred != labels).float().mean(dim=1)  # per-task residual error fraction
    margin = ((vmax - threshold[:, None]) * labels).mean(dim=1)
    return {
        "converged": converged,
        "epochs_run": epochs_run,
        "wnorm": w.norm(dim=1),
        "mean_margin": margin,
        "threshold": threshold,
        "init_fire_rate": init_fire_rate,
        "final_errfrac": final_errfrac,
    }


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
    grid_per_corr = int(ov.get("grid_per_corr", "8"))   # time-grid points per sqrt(tau_s tau_m)
    anchor_k = {float(x) for x in ov.get("anchor_K", "").split(",") if x}
    n_restarts = int(ov.get("n_restarts", "10"))
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
        f"grid_per_corr={grid_per_corr} anchor_K={sorted(anchor_k)} restarts={n_restarts}",
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
        dt = SQRT_TAU / grid_per_corr
        n_grid = round(t_window / dt) + 1
        t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * dt
        do_restart = k in anchor_k
        for n_aff in n_list:
            for alpha in alphas:
                cell_i += 1
                p = round(alpha * n_aff)
                if p == 0:
                    continue
                gen = torch.Generator(device=device)
                gen.manual_seed(cell_seed(master_seed, n_aff, alpha, k))
                spikes, valid, labels, _ = make_patterns(n_seeds, p, n_aff, t_window, gen, device)

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

                for b0 in range(0, n_seeds, sb_size):
                    b1 = min(b0 + sb_size, n_seeds)
                    s = precompute_traces(spikes[b0:b1], valid[b0:b1], t_grid, elem_budget)
                    lab_b = labels[b0:b1]
                    wgen = torch.Generator(device=device)
                    wgen.manual_seed(cell_seed(master_seed, n_aff, alpha, k, tag=1) + b0)
                    w0 = torch.randn(b1 - b0, n_aff, generator=wgen, device=device)
                    res = train_batch(s, lab_b, w0, lr=lr, momentum=momentum, epochs=epochs,
                                      mode=mode)
                    conv[b0:b1] = res["converged"]
                    ep_run[b0:b1] = res["epochs_run"].float()
                    wn[b0:b1] = res["wnorm"]
                    mg[b0:b1] = res["mean_margin"]
                    fire[b0:b1] = res["init_fire_rate"]
                    errf[b0:b1] = res["final_errfrac"]

                    if do_restart:
                        any_conv = res["converged"].clone()
                        thr = res["threshold"]
                        for r in range(1, n_restarts):
                            wgen.manual_seed(cell_seed(master_seed, n_aff, alpha, k, tag=2 + r) + b0)
                            wr = torch.randn(b1 - b0, n_aff, generator=wgen, device=device)
                            rr = train_batch(s, lab_b, wr, lr=lr, momentum=momentum,
                                             epochs=epochs, threshold=thr, mode=mode)
                            any_conv = any_conv | rr["converged"]
                            if bool(any_conv.all()):
                                break
                        conv_multi[b0:b1] = any_conv
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
                row = {
                    "K": k, "N": n_aff, "alpha": alpha, "P": p, "n_seeds": n_seeds,
                    "n_solved": n_solved, "p_solve": p_solve, "p_solve_se": p_se,
                    "mean_epochs": float(ep_run.mean().item()),
                    "resid_errfrac_unconv": resid,
                    "mean_wnorm": float(wn.mean().item()),
                    "mean_margin": float(mg.mean().item()),
                    "mean_init_fire_rate": float(fire.mean().item()),
                    "n_grid": n_grid, "dt": dt, "lr": lr, "epochs": epochs,
                    "n_restarts": n_restarts if do_restart else 0,
                    "n_solved_multi": int(conv_multi.sum().item()) if do_restart else -1,
                    "p_solve_multi": (int(conv_multi.sum().item()) / n_seeds) if do_restart else -1.0,
                }
                rows.append(row)
                emit(f"psolve_K{round(k)}_N{n_aff}", p_solve, round(1000 * alpha))
                print(
                    f"[{cell_i}/{n_cells}] K={k:.0f} N={n_aff} a={alpha:.3f} P={p} "
                    f"p_solve={p_solve:.3f}+/-{p_se:.3f} resid={resid:.3f} "
                    f"<ep>={row['mean_epochs']:.0f} fire0={row['mean_init_fire_rate']:.2f}"
                    + (f" p_multi={row['p_solve_multi']:.3f}" if do_restart else ""),
                    flush=True,
                )

    elapsed = time.time() - started
    results = {
        "experiment": "v3_capacity_sweep",
        "params": {
            "master_seed": master_seed, "K_list": k_list, "N_list": n_list, "alphas": alphas,
            "n_seeds": n_seeds, "epochs": epochs, "lr": lr, "momentum": momentum, "mode": mode,
            "grid_per_corr": grid_per_corr, "tau_m": TAU_M, "tau_s": TAU_S,
            "anchor_K": sorted(anchor_k), "n_restarts": n_restarts,
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
    print(f"done in {elapsed:.1f}s -> {run_dir}/results.json ({len(rows)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
