"""S2 -- Tempotron decision visualisation: V(t) for + and - patterns after training.

Reproduces the canonical decision figure of Gütig & Sompolinsky 2006 (Nat. Neurosci. 9, 420,
Fig 1). Trains a small tempotron on the random Poisson-pattern classification task at
``alpha = 0.5`` (well below the K=20 capacity ~ 1.16), then evaluates the trained voltage
trace ``V(t)`` for one + pattern (expected ``V_max > U_th``) and one - pattern (expected
``V_max < U_th``). Output: a side-by-side decision figure analogous to GS 2006 Fig 1c.

Validates Rungs 1 (PSP kernel) + 3a (decision rule) + 3b (gradient-on-margin learning) end
to end with a publishable-quality figure that future writeups can lean on.

Self-contained Experiment-Contract entrypoint (numpy + torch); no ``tempotron`` imports.

Outputs (to ``$LAB_RUN_DIR``)
-----------------------------
- ``training_curve.csv``: error rate per epoch (for the convergence plot).
- ``traces.npz``: ``t_grid``, ``v_plus``, ``v_minus``, ``threshold``, ``params`` -- raw arrays
  for the figure script (rendered locally on completion).
- ``results.json``: full provenance + per-epoch training log + final classifier verdict.
- ``metrics.jsonl``: per-epoch metric series for live monitoring.

Run standalone (smoke):
    LAB_RUN_DIR=/tmp/s2_smoke LAB_SEED=0 \\
        uv run --with torch python studies/sanity/s2_tempotron_decision_viz.py \\
        N=50 K=10 alpha=0.5 n_epochs=30
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
    """Double-exponential PSP, peak-normalised to 1. Causal."""
    t_peak = TAU_M * TAU_S / (TAU_M - TAU_S) * math.log(TAU_M / TAU_S)
    v0 = 1.0 / (math.exp(-t_peak / TAU_M) - math.exp(-t_peak / TAU_S))
    t_clamped = torch.clamp(t, min=0.0)
    raw = v0 * (torch.exp(-t_clamped / TAU_M) - torch.exp(-t_clamped / TAU_S))
    return torch.where(t >= 0.0, raw, torch.zeros_like(raw))


def voltage_on_grid(
    spike_times: torch.Tensor,
    valid_mask: torch.Tensor,
    weights: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """V(t) for one pattern on a time grid. Shapes:
    - spike_times: (N_aff, max_spikes)
    - valid_mask:  (N_aff, max_spikes) -- 1.0 where the spike is real, 0 where padding.
    - weights:     (N_aff,)
    - t_grid:      (T_grid,)
    Returns V: (T_grid,).
    """
    # delta[g, n, f] = t_grid[g] - spike_times[n, f]
    delta = t_grid.view(-1, 1, 1) - spike_times.view(1, *spike_times.shape)
    kernel = psp_kernel(delta) * valid_mask.view(1, *valid_mask.shape)
    per_aff = kernel.sum(dim=2)  # (T_grid, N_aff)
    return per_aff @ weights  # (T_grid,)


def vmax_argmax(
    spike_times: torch.Tensor,
    valid_mask: torch.Tensor,
    weights: torch.Tensor,
    t_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (V_max, t_max) on the grid for one pattern."""
    v_grid = voltage_on_grid(spike_times, valid_mask, weights, t_grid)
    idx = torch.argmax(v_grid)
    return v_grid[idx], t_grid[idx]


def s_i_at_tmax(
    spike_times: torch.Tensor,
    valid_mask: torch.Tensor,
    t_max: torch.Tensor,
) -> torch.Tensor:
    """Per-afferent kernel sum at ``t_max``: ``s_i = sum_f K(t_max - t_i^f)``."""
    delta = t_max - spike_times
    kernel = psp_kernel(delta) * valid_mask
    return kernel.sum(dim=1)


def generate_patterns(
    n_patterns: int,
    n_aff: int,
    t_window: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_patterns RMS-2010 Poisson patterns. Returns (spike_times, valid_mask, max_spikes).
    spike_times shape (n_patterns, n_aff, max_spikes).
    """
    counts = rng.poisson(1.0, size=(n_patterns, n_aff))  # rate=1 per afferent
    max_spikes = max(1, int(counts.max()))
    spike_times = np.zeros((n_patterns, n_aff, max_spikes), dtype=np.float32)
    valid_mask = np.zeros_like(spike_times)
    for p in range(n_patterns):
        for i in range(n_aff):
            c = int(counts[p, i])
            if c:
                spike_times[p, i, :c] = rng.uniform(0.0, t_window, size=c)
                valid_mask[p, i, :c] = 1.0
    return spike_times, valid_mask, np.array([max_spikes])


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))

    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    smoke = os.environ.get("LAB_RUN_DIR", "runs/local-dev") in ("", "runs/local-dev")
    n_aff = int(ov.get("N", "50" if smoke else "100"))
    k_res = int(ov.get("K", "10" if smoke else "20"))
    alpha = float(ov.get("alpha", "0.5"))
    n_epochs = int(ov.get("n_epochs", "30" if smoke else "60"))
    lr = float(ov.get("lr", "0.05"))
    momentum = float(ov.get("momentum", "0.99"))

    t_window = float(k_res) * math.sqrt(TAU_M * TAU_S)
    dt = TAU_S / 8.0
    n_grid = int(math.ceil(t_window / dt)) + 1
    p_patterns = max(2, round(alpha * n_aff))
    if p_patterns % 2 == 1:
        p_patterns += 1  # balanced +/- labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(master_seed)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"S2 tempotron decision viz | seed={master_seed} device={device} "
        f"N={n_aff} K={k_res} alpha={alpha} P={p_patterns} T={t_window:.1f}ms "
        f"n_epochs={n_epochs} lr={lr} momentum={momentum}",
        flush=True,
    )

    # Patterns + labels (balanced).
    spike_times_np, valid_mask_np, _ = generate_patterns(p_patterns, n_aff, t_window, rng)
    labels_np = np.array([+1.0 if i < p_patterns // 2 else -1.0 for i in range(p_patterns)],
                         dtype=np.float32)
    rng.shuffle(labels_np)  # randomise + / - assignment
    spike_times = torch.from_numpy(spike_times_np).to(device)  # (P, N, max_spikes)
    valid_mask = torch.from_numpy(valid_mask_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)

    # Threshold: U_th = median V_max over a sample of random unit-norm weights -> P(fire)=1/2
    # We re-use the locked Rung 3a calibration recipe (decision.py::calibrate_threshold).
    calib_n = max(64, p_patterns * 4)
    calib_vmaxes = []
    for _ in range(calib_n):
        w_rand = torch.randn(n_aff, device=device)
        # Pick a random pattern (uniform) for the calibration draw.
        idx = rng.integers(0, p_patterns)
        t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * dt
        v_max, _ = vmax_argmax(spike_times[idx], valid_mask[idx], w_rand, t_grid)
        calib_vmaxes.append(float(v_max))
    threshold = float(np.median(calib_vmaxes))
    print(f"  threshold (median V_max over {calib_n} random weights) = {threshold:.3f}",
          flush=True)

    # Initial weights -- small Gaussian.
    weights = (0.01 * torch.randn(n_aff, device=device, dtype=torch.float32))
    weights.requires_grad_(False)
    velocity = torch.zeros_like(weights)

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

    # Training loop.
    t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * dt
    epoch_log: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(n_epochs):
        perm = rng.permutation(p_patterns)
        errors = 0
        for j in perm:
            v_max, t_max = vmax_argmax(
                spike_times[j], valid_mask[j], weights, t_grid
            )
            output = 1.0 if float(v_max) > threshold else -1.0
            if output != float(labels[j]):
                errors += 1
                # Tempotron rule on error: dw_i = lr * y * s_i(t_max).
                s_i = s_i_at_tmax(spike_times[j], valid_mask[j], t_max)
                grad = float(labels[j]) * s_i
                velocity = momentum * velocity + lr * grad
                weights = weights + velocity
        err_rate = errors / p_patterns
        epoch_log.append({"epoch": epoch, "errors": errors, "error_rate": err_rate})
        emit("error_rate", err_rate, epoch)
        emit("errors", float(errors), epoch)
        if epoch % max(1, n_epochs // 10) == 0 or errors == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:>3}: errors {errors}/{p_patterns} (rate {err_rate:.3f})",
                  flush=True)
        if errors == 0:
            break
    trained_epochs = len(epoch_log)

    # Pick one + and one - test pattern for the decision figure (from the training set is fine --
    # GS 2006 Fig 1 also visualises training patterns).
    idx_plus = next(j for j in range(p_patterns) if float(labels[j]) > 0)
    idx_minus = next(j for j in range(p_patterns) if float(labels[j]) < 0)
    with torch.no_grad():
        v_plus = voltage_on_grid(spike_times[idx_plus], valid_mask[idx_plus], weights, t_grid)
        v_minus = voltage_on_grid(spike_times[idx_minus], valid_mask[idx_minus], weights, t_grid)
    vmax_plus = float(v_plus.max())
    vmax_minus = float(v_minus.max())
    classify_plus = +1 if vmax_plus > threshold else -1
    classify_minus = +1 if vmax_minus > threshold else -1
    final_correct = (classify_plus == 1) and (classify_minus == -1)

    print(f"  +pattern V_max = {vmax_plus:.3f}  (threshold={threshold:.3f}) -> "
          f"classified {'+1 [correct]' if classify_plus == 1 else '-1 [WRONG]'}", flush=True)
    print(f"  -pattern V_max = {vmax_minus:.3f}  (threshold={threshold:.3f}) -> "
          f"classified {'-1 [correct]' if classify_minus == -1 else '+1 [WRONG]'}", flush=True)

    # Save figure data.
    np.savez(
        run_dir / "traces.npz",
        t_grid=t_grid.detach().cpu().numpy().astype(np.float64),
        v_plus=v_plus.detach().cpu().numpy().astype(np.float64),
        v_minus=v_minus.detach().cpu().numpy().astype(np.float64),
        spike_times_plus=spike_times[idx_plus].detach().cpu().numpy().astype(np.float64),
        valid_mask_plus=valid_mask[idx_plus].detach().cpu().numpy().astype(np.float64),
        spike_times_minus=spike_times[idx_minus].detach().cpu().numpy().astype(np.float64),
        valid_mask_minus=valid_mask[idx_minus].detach().cpu().numpy().astype(np.float64),
        weights=weights.detach().cpu().numpy().astype(np.float64),
        threshold=np.float64(threshold),
        tau_m=np.float64(TAU_M),
        tau_s=np.float64(TAU_S),
    )

    # Save the training curve as a CSV (separately from results.csv).
    with (run_dir / "training_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "errors", "error_rate"])
        writer.writeheader()
        writer.writerows(epoch_log)

    # results.json: full provenance.
    final = {
        "experiment": "S2 -- tempotron decision visualisation (GS 2006 Fig 1)",
        "reference": "Gutig & Sompolinsky 2006, Nat Neurosci 9:420, Fig 1c",
        "params": {
            "N": n_aff, "K": k_res, "alpha": alpha, "P": p_patterns,
            "T_window_ms": t_window, "n_epochs_max": n_epochs, "lr": lr, "momentum": momentum,
            "tau_m": TAU_M, "tau_s": TAU_S, "master_seed": master_seed,
        },
        "trained_epochs": trained_epochs,
        "converged": trained_epochs < n_epochs,
        "final_error_rate": epoch_log[-1]["error_rate"],
        "threshold": threshold,
        "vmax_plus_test": vmax_plus,
        "vmax_minus_test": vmax_minus,
        "final_correct": bool(final_correct),
        "elapsed_seconds": time.time() - started,
        "epoch_log": epoch_log,
    }
    (run_dir / "results.json").write_text(json.dumps(final, indent=2))
    # results.csv (one summary row, schema-stable for the figure orchestration).
    summary_row = {k: v for k, v in final.items() if not isinstance(v, (list, dict))}
    fieldnames = list(summary_row.keys())
    with (run_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

    print(
        f"done in {final['elapsed_seconds']:.1f}s (trained {trained_epochs} epochs; "
        f"converged={final['converged']}; final_correct={final['final_correct']}) -> "
        f"{run_dir}/results.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
