"""S5 -- Threshold gauge invariance and the U_th/lambda learning timescale.

A proper rerun of the earlier threshold-norm "side study" (which used N=40, no theory
overlay, and a learning-timescale prediction masked by a large random initialisation). Here
we (i) use a larger N, (ii) initialise the weights near zero so the U_th/lambda growth
timescale is not masked, and (iii) overlay the analytic predictions.

Theory (noise-free tempotron; Gutig-Sompolinsky 2006; gauge argument)
---------------------------------------------------------------------
V_max(w) = max_t w.s(t) is positively homogeneous: V_max(cw) = c V_max(w). Hence rescaling
w -> cw is equivalent to U_th -> U_th/c, and the *existence* of a zero-error solution is
independent of U_th (gauge invariance). The learning *dynamics* are not gauge-free: from a
near-zero start each tempotron update adds ~ lambda * s(t_max) with s = O(1), so reaching
V_max ~ U_th takes ~ U_th/lambda updates. Predictions, at fixed load alpha below capacity:

  P1 (gauge invariance of existence): every U_th reaches zero training error.
  P2 (norm scaling):     final ||w|| proportional to U_th.
  P3 (timescale scaling): epochs-to-converge proportional to U_th / lambda
                          (exposed only from near-zero init).

Self-contained Experiment-Contract entrypoint (numpy + torch). Reuses the tempotron forward
pass / learning rule (double-exponential PSP, grid V_max, dw = lambda y s(t_max)).

Outputs (to $LAB_RUN_DIR)
-------------------------
- results.csv: one row per (U_th, seed): converged, epochs, final_norm, ...
- results.json: params + rows + per-U_th aggregates + linear-fit slopes for P2, P3.
- metrics.jsonl: incremental series.

Run standalone (smoke):
    LAB_RUN_DIR=/tmp/s5 LAB_SEED=0 \\
        uv run --with torch python studies/sanity/s5_gauge_threshold_norm.py \\
        N=60 alpha=0.5 U_th_list=2,8 n_seeds=2 n_epochs=80
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

TAU_M = 15.0
TAU_S = 3.75


def psp_kernel(t: torch.Tensor) -> torch.Tensor:
    t_peak = TAU_M * TAU_S / (TAU_M - TAU_S) * math.log(TAU_M / TAU_S)
    v0 = 1.0 / (math.exp(-t_peak / TAU_M) - math.exp(-t_peak / TAU_S))
    tc = torch.clamp(t, min=0.0)
    raw = v0 * (torch.exp(-tc / TAU_M) - torch.exp(-tc / TAU_S))
    return torch.where(t >= 0.0, raw, torch.zeros_like(raw))


def voltage_on_grid(spike_times: torch.Tensor, valid: torch.Tensor,
                    w: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    delta = t_grid.view(-1, 1, 1) - spike_times.view(1, *spike_times.shape)
    kernel = psp_kernel(delta) * valid.view(1, *valid.shape)
    return kernel.sum(dim=2) @ w  # (T_grid,)


def vmax_tmax(spike_times: torch.Tensor, valid: torch.Tensor,
              w: torch.Tensor, t_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v = voltage_on_grid(spike_times, valid, w, t_grid)
    idx = torch.argmax(v)
    return v[idx], t_grid[idx]


def s_at(spike_times: torch.Tensor, valid: torch.Tensor, t_max: torch.Tensor) -> torch.Tensor:
    return (psp_kernel(t_max - spike_times) * valid).sum(dim=1)


def make_patterns(p: int, n: int, t_window: float, rng: np.random.Generator):
    counts = rng.poisson(1.0, size=(p, n))
    max_sp = max(1, int(counts.max()))
    st = np.zeros((p, n, max_sp), dtype=np.float32)
    vm = np.zeros_like(st)
    for a in range(p):
        for i in range(n):
            c = int(counts[a, i])
            if c:
                st[a, i, :c] = rng.uniform(0.0, t_window, size=c)
                vm[a, i, :c] = 1.0
    return st, vm


def train_one(spike_times: torch.Tensor, valid: torch.Tensor, labels: torch.Tensor,
              t_grid: torch.Tensor, u_th: float, lr: float, n_epochs: int,
              rng: np.random.Generator, device: torch.device,
              init_scale: float) -> tuple[bool, int, int, float]:
    """Train from a near-zero init; return (converged, epochs_used, total_updates, final_norm).

    ``total_updates`` (the number of weight changes = mistakes made over training) is the
    physical learning-time observable: from a near-zero start each update grows ||w|| by at
    most lambda*||s||, so reaching the threshold scale needs ~ U_th/lambda updates. Epochs is
    not this observable because one epoch makes up to P updates.
    """
    n = valid.shape[1]
    p = valid.shape[0]
    w = torch.from_numpy((init_scale * rng.standard_normal(n)).astype(np.float32)).to(device)
    epochs_used = n_epochs
    total_updates = 0
    converged = False
    for epoch in range(n_epochs):
        order = rng.permutation(p)
        errors = 0
        for j in order:
            vmax, tmax = vmax_tmax(spike_times[j], valid[j], w, t_grid)
            out = 1.0 if float(vmax) > u_th else -1.0
            if out != float(labels[j]):
                errors += 1
                total_updates += 1
                w = w + lr * float(labels[j]) * s_at(spike_times[j], valid[j], tmax)
        if errors == 0:
            converged = True
            epochs_used = epoch + 1
            break
    return converged, epochs_used, total_updates, float(torch.linalg.norm(w).item())


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))
    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    smoke = os.environ.get("LAB_RUN_DIR", "runs/local-dev") in ("", "runs/local-dev")

    n = int(ov.get("N", "60" if smoke else "200"))
    k_res = int(ov.get("K", "20"))
    alpha = float(ov.get("alpha", "0.5"))
    u_th_list = [float(x) for x in ov.get("U_th_list", "2,8" if smoke else "1,2,4,8,16").split(",")]
    n_seeds = int(ov.get("n_seeds", "2" if smoke else "8"))
    n_epochs = int(ov.get("n_epochs", "80" if smoke else "400"))
    lr = float(ov.get("lr", "0.1"))
    init_scale = float(ov.get("init_scale", "1e-3"))

    t_window = float(k_res) * math.sqrt(TAU_M * TAU_S)
    dt = TAU_S / 8.0
    n_grid = int(math.ceil(t_window / dt)) + 1
    p = max(2, round(alpha * n))
    if p % 2:
        p += 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"S5 gauge | seed={master_seed} dev={device} N={n} K={k_res} alpha={alpha} P={p} "
          f"U_th={u_th_list} n_seeds={n_seeds} n_epochs={n_epochs} lr={lr} "
          f"init_scale={init_scale} smoke={smoke}", flush=True)

    metrics_path = run_dir / "metrics.jsonl"

    def emit(name: str, value: float, step: int) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps({"name": name, "value": float(value), "step": int(step),
                                "wall_time": time.time()}) + "\n")

    t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * dt
    rows: list[dict[str, float | int | bool]] = []
    started = time.time()
    for u_th in u_th_list:
        for seed in range(n_seeds):
            rng = np.random.default_rng(
                np.random.SeedSequence([master_seed, int(1000 * u_th), seed])
            )
            st_np, vm_np = make_patterns(p, n, t_window, rng)
            labels_np = np.array([1.0 if i < p // 2 else -1.0 for i in range(p)],
                                 dtype=np.float32)
            rng.shuffle(labels_np)
            st = torch.from_numpy(st_np).to(device)
            vm = torch.from_numpy(vm_np).to(device)
            labels = torch.from_numpy(labels_np).to(device)
            conv, epochs, updates, norm = train_one(st, vm, labels, t_grid, u_th, lr,
                                                     n_epochs, rng, device, init_scale)
            rows.append({"U_th": u_th, "seed": seed, "converged": conv,
                         "epochs": epochs, "updates": updates, "final_norm": norm,
                         "N": n, "alpha": alpha, "P": p, "lr": lr})
            emit(f"updates_Uth{int(1000*u_th)}", updates, seed)
            emit(f"norm_Uth{int(1000*u_th)}", norm, seed)
            print(f"  U_th={u_th:>5.1f} seed={seed}: converged={conv} epochs={epochs} "
                  f"updates={updates} ||w||={norm:.2f}", flush=True)

    # Aggregates + linear fits for P2 (norm ~ U_th) and P3 (epochs ~ U_th).
    aggs = {}
    for u_th in u_th_list:
        sub = [r for r in rows if r["U_th"] == u_th]
        conv = [r for r in sub if r["converged"]]
        norms = np.array([r["final_norm"] for r in conv]) if conv else np.array([np.nan])
        upd = np.array([r["updates"] for r in conv]) if conv else np.array([np.nan])
        aggs[u_th] = {
            "frac_converged": sum(r["converged"] for r in sub) / len(sub),
            "norm_mean": float(np.mean(norms)), "norm_std": float(np.std(norms)),
            "updates_mean": float(np.mean(upd)), "updates_std": float(np.std(upd)),
        }
    uth_arr = np.array([u for u in u_th_list if np.isfinite(aggs[u]["norm_mean"])])
    if uth_arr.size >= 2:
        norm_means = np.array([aggs[u]["norm_mean"] for u in uth_arr])
        upd_means = np.array([aggs[u]["updates_mean"] for u in uth_arr])
        norm_slope, norm_int = np.polyfit(uth_arr, norm_means, 1)
        upd_slope, upd_int = np.polyfit(uth_arr, upd_means, 1)
        norm_r = float(np.corrcoef(uth_arr, norm_means)[0, 1])
        upd_r = float(np.corrcoef(uth_arr, upd_means)[0, 1])
    else:
        norm_slope = norm_int = upd_slope = upd_int = norm_r = upd_r = float("nan")

    fieldnames = list(rows[0].keys())
    with (run_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (run_dir / "results.json").write_text(json.dumps({
        "experiment": "S5 -- threshold gauge invariance + U_th/lambda timescale",
        "params": {"N": n, "K": k_res, "alpha": alpha, "P": p, "U_th_list": u_th_list,
                   "n_seeds": n_seeds, "n_epochs": n_epochs, "lr": lr,
                   "init_scale": init_scale, "master_seed": master_seed},
        "aggregates": {str(k): v for k, v in aggs.items()},
        "fits": {"norm_vs_Uth_slope": float(norm_slope), "norm_vs_Uth_r": norm_r,
                 "updates_vs_Uth_slope": float(upd_slope), "updates_vs_Uth_r": upd_r,
                 "P3_predicted_slope_1_over_lambda": 1.0 / lr},
        "elapsed_seconds": time.time() - started, "rows": rows,
    }, indent=2, default=str))
    print(f"\nfits: ||w||~U_th slope={norm_slope:.2f} (r={norm_r:.3f}); "
          f"updates~U_th slope={upd_slope:.2f} (r={upd_r:.3f}, predicted 1/lambda={1/lr:.1f})",
          flush=True)
    print(f"done in {time.time()-started:.1f}s -> {run_dir}/results.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
