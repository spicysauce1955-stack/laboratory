"""V11 -- controlled-factorial schedule-mechanism study (self-contained lab entrypoint).

Isolates which property of the LR schedule closes the tempotron findability gap. NOT Optuna: a fixed
factorial at a matched, pre-calibrated peak LR per (optimizer, batch) cell. Reads $LAB_RUN_DIR/$LAB_SEED,
writes cells.jsonl / curves.jsonl / calib.jsonl / manifest.json there. See
docs/v11-schedule-mechanism-design.md.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- load the study module (single source of truth for model + train_batch) ---
_STUDY = Path(__file__).resolve().parent / "v3_capacity_sweep.py"          # flat lab layout
if not _STUDY.exists():
    _STUDY = Path(__file__).resolve().parents[1] / "studies" / "v3_capacity_sweep.py"
_spec = importlib.util.spec_from_file_location("v3_capacity_sweep", _STUDY)
assert _spec is not None and _spec.loader is not None
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)

TAU_M, SQRT_TAU, PSP_V0 = v3.TAU_M, v3.SQRT_TAU, v3.PSP_V0
T_WINDOW = 500.0
K_TEMPORAL = T_WINDOW / SQRT_TAU


def gs_base_lr(n: int) -> float:
    """Faithful GS base learning rate (1/N-scaled), as in v10_hpo."""
    return 3e-3 * T_WINDOW / (TAU_M * n * PSP_V0)


def build_cell_batch(n, alpha, n_seeds, master_seed, grid_per_corr, device):
    """One single-spike task per seed at load alpha: returns (s, labels, w0) on `device`.

    s: (n_seeds, P, G, N) traces; labels: (n_seeds, P); w0: (n_seeds, N) N(0,1) init.
    """
    dt = SQRT_TAU / grid_per_corr
    n_grid = round(T_WINDOW / dt) + 1
    t_grid = torch.arange(n_grid, dtype=torch.float32, device=device) * dt
    p = round(alpha * n)
    rng = np.random.default_rng(v3.cell_seed(master_seed, n, alpha, K_TEMPORAL))
    st, vd, lab, _ = v3.make_patterns(n_seeds, p, n, T_WINDOW, rng, device, ensemble="single")
    s = v3.precompute_traces(st, vd, t_grid)                       # (n_seeds, P, G, N)
    w0 = torch.from_numpy(rng.standard_normal((n_seeds, n), dtype=np.float32)).to(device)
    return s, lab, w0


def _sched_params(spec):
    """The spec's schedule knobs in ``scheduled_lr``'s vocabulary.

    The single source for both training (via :func:`_kwargs_for`) and the recorded LR trace /
    cells.jsonl fields, so the artifacts can never disagree with the schedule actually trained.
    """
    return dict(schedule=spec["shape"], warmup=spec["warmup"], floor=spec["floor"],
                n_cycles=spec["n_cycles"], exp_k=spec.get("exp_k", v3.EXP_DECAY_K))


def _kwargs_for(spec, peak_lr, r_max):
    """Map a cell spec + resolved peak LR to train_batch kwargs."""
    sched = _sched_params(spec)
    kw = dict(mode="minibatch", batch_size=spec["batch"], optimizer=spec["optimizer"],
              lr=peak_lr, epochs=r_max, vthr_fixed=1.0,
              lr_schedule=sched["schedule"], lr_warmup=sched["warmup"], lr_floor=sched["floor"],
              lr_cycles=sched["n_cycles"], lr_exp_k=sched["exp_k"], momentum=spec["mu"])
    if spec["optimizer"] == "adam":
        kw["adam_betas"] = (0.95, 0.99)
        kw["adam_eps"] = 1e-6
    elif spec["optimizer"] == "rmsprop":
        kw["rms_alpha"] = 0.99
        kw["rms_eps"] = 1e-8
    return kw


def run_cell(spec, batch, r_max, gs_base_lr, capture):
    """Train one cell; return a cells.jsonl record (+ a per-epoch 'curve' when capture=True)."""
    s, lab, w0 = batch
    # The faithful corner (S1) uses the GS base lambda; every other cell uses its matched peak.
    peak = gs_base_lr if spec.get("faithful") else spec["peak_lr"]
    out = v3.train_batch(s, lab, w0, capture=capture, **_kwargs_for(spec, peak, r_max))
    epochs_run = out["epochs_run"].tolist()
    conv = out["converged"].tolist()
    solved_eps = [e for e, c in zip(epochs_run, conv, strict=True) if c]
    median = float(np.median(solved_eps)) if len(solved_eps) > len(epochs_run) / 2 else None
    sched = _sched_params(spec)
    rec = {
        "cell_id": spec["cell_id"], "role": spec["role"], "optimizer": spec["optimizer"],
        "batch": spec["batch"], "shape": spec["shape"], "peak_lr": float(peak), "mu": spec["mu"],
        "floor": sched["floor"], "warmup": sched["warmup"], "n_cycles": sched["n_cycles"],
        "exp_k": sched["exp_k"],
        "alpha": spec["alpha"], "N": s.shape[-1], "S": s.shape[0], "R_max": r_max,
        "solve_fraction": sum(conv) / len(conv), "epochs_to_solve_median": median,
        "epochs_run": epochs_run, "censored": len(conv) - sum(conv),
        "wnorm_mean": float(out["wnorm"].mean().item()),
        "mean_margin_mean": float(out["mean_margin"].mean().item()),
    }
    if capture and "cap" in out:
        cap = out["cap"]; ne = cap["n_epochs"]
        log_every = max(1, ne // 200)                           # ~200 points/curve, keeps artifacts small
        idx = list(range(0, ne, log_every))
        loss = cap["traj_loss"].mean(dim=0)
        err = cap["traj_err"].mean(dim=0)
        er = torch.tensor(epochs_run, dtype=torch.float32)
        solved_frac = [float((er <= e).float().mean().item()) for e in idx]
        lr_trace = [v3.scheduled_lr(peak, e, epochs=r_max, **sched) for e in idx]
        rec["curve"] = {"epochs": idx, "log_every": log_every,
                        "loss": loss[idx].tolist(),
                        "acc": (1.0 - err[idx]).tolist(),
                        "solved_frac": solved_frac, "lr": lr_trace}
    return rec


CAL_ALPHA = 2.4                  # calibration load (spec §3); also the always-captured factorial load
DECISIVE_LOADS = [2.2, CAL_ALPHA, 2.6]
SINGLE_LOAD = [CAL_ALPHA]


def calibrate_peak(*, optimizer, batch_size, mu, cell_batch, gs_base, r_max, n_points=6,
                   sgd_lo=0.3, sgd_hi=50.0, adapt_lo=1e-4, adapt_hi=1e-1):
    """1-D log sweep of the peak LR under cosine->0; pick the peak minimising median epochs-to-solve.

    SGD peaks are multiples of the GS base lambda (so they transfer across N); Adam/RMSprop are absolute.
    Returns a calib.jsonl record. The chosen peak is the best interior point; `edge` flags a boundary hit.
    """
    if optimizer == "momentum":
        mults = np.geomspace(sgd_lo, sgd_hi, n_points)
        peaks = [float(m * gs_base) for m in mults]
    else:
        peaks = [float(x) for x in np.geomspace(adapt_lo, adapt_hi, n_points)]
    median_epochs, solve_frac = [], []
    for peak in peaks:
        spec = {"cell_id": "calib", "role": "calib", "optimizer": optimizer, "batch": batch_size,
                "shape": "cosine", "peak_lr": peak, "mu": mu, "floor": 0.0, "warmup": 0,
                "n_cycles": 1, "alpha": CAL_ALPHA}
        rec = run_cell(spec, cell_batch, r_max=r_max, gs_base_lr=gs_base, capture=False)
        median_epochs.append(rec["epochs_to_solve_median"])
        solve_frac.append(rec["solve_fraction"])
    # score: prefer solved (median not None); among solved, smaller median; tie-break higher solve_frac.
    def _score(i):
        m = median_epochs[i]
        return (0 if m is not None else 1, m if m is not None else 1e9, -solve_frac[i])
    best = min(range(len(peaks)), key=_score)
    edge = best in (0, len(peaks) - 1)
    return {"optimizer": optimizer, "batch": batch_size, "peaks": peaks,
            "median_epochs": median_epochs, "solve_frac": solve_frac,
            "chosen_peak": peaks[best], "edge": edge}


def build_cell_specs():
    """The frozen V11 factorial (spec §3). peak_lr=None => resolved from the cell's calibration.

    Spine: SGD b=1, mu=0.99 unless noted. Generality: 3 crux shapes x optimizer x batch.
    'loads' lists the alphas to run for that cell (decisive shapes get 3, others 1).
    """
    spine = [
        dict(cell_id="S1_const_faithful", shape="none", faithful=True, loads=SINGLE_LOAD),
        dict(cell_id="S2_const_peak",     shape="none", loads=DECISIVE_LOADS),
        dict(cell_id="S3_linear",         shape="linear", loads=DECISIVE_LOADS),
        dict(cell_id="S4_cosine",         shape="cosine", loads=DECISIVE_LOADS),
        dict(cell_id="S5_exp",            shape="exp", loads=SINGLE_LOAD),
        dict(cell_id="S6_cosine_floor",   shape="cosine", floor=0.1, loads=SINGLE_LOAD),
        dict(cell_id="S7_cosine_warmup",  shape="cosine", warmup="auto", loads=SINGLE_LOAD),
        dict(cell_id="S8_sgdr",           shape="cosine", n_cycles=2, loads=SINGLE_LOAD),
        # S9 isolates H-overshoot: it RUNS at mu=0 but SHARES S4's peak (calibrated at mu=0.99) via
        # calib_mu, so the only thing differing from S4 is the momentum -- not the tuned step size.
        dict(cell_id="S9_cosine_mu0",     shape="cosine", mu=0.0, calib_mu=0.99, loads=SINGLE_LOAD),
    ]
    for sp in spine:
        sp.update(role="spine", optimizer="momentum", batch=1)
        sp.setdefault("mu", 0.99); sp.setdefault("floor", 0.0)
        sp.setdefault("warmup", 0); sp.setdefault("n_cycles", 1); sp.setdefault("peak_lr", None)
        sp.setdefault("calib_mu", sp["mu"])   # which mu the peak is calibrated under (== run mu unless shared)
    gen = []
    crux = [("none", SINGLE_LOAD), ("cosine", DECISIVE_LOADS), ("linear", DECISIVE_LOADS)]
    grid = [("momentum", 64, 0.99), ("adam", 16, 0.0), ("adam", 256, 0.0),
            ("rmsprop", 64, 0.0), ("rmsprop", 256, 0.0)]   # SGD b=1 covered by the spine
    for opt, b, mu in grid:
        for shape, loads in crux:
            gen.append(dict(cell_id=f"G_{opt}_b{b}_{shape}", role="generality", optimizer=opt,
                            batch=b, mu=mu, calib_mu=mu, shape=shape, floor=0.0, warmup=0,
                            n_cycles=1, peak_lr=None, loads=loads))
    # exp-decay-rate sweep on Adam (V11.1): does a gentler exp (k=2, long exploration) reach a higher
    # findable capacity than the aggressive default (k=5/8)? Shares Adam's calibrated peak (calib_mu=0.0).
    for b in (16, 256):
        for k in (2.0, 5.0, 8.0):
            gen.append(dict(cell_id=f"G_adam_b{b}_exp{int(k)}", role="generality", optimizer="adam",
                            batch=b, mu=0.0, calib_mu=0.0, shape="exp", exp_k=k, floor=0.0,
                            warmup=0, n_cycles=1, peak_lr=None, loads=DECISIVE_LOADS))
    return spine + gen


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/v11-local"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))
    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    N = int(ov.get("N", "200")); S = int(ov.get("S", "32"))
    R_max = min(5000, int(ov.get("R_max", "5000")))        # spec §4: hard cap 5000, never exceeded
    grid_per_corr = int(ov.get("grid_per_corr", "8"))
    n_calib = int(ov.get("calib_points", "6"))
    allow_cpu = int(ov.get("allow_cpu", "0"))
    only = ov.get("only", "")                              # optional cell_id prefix filter (job-splitting)
    # Optional load-grid override (e.g. alphas=2.6,2.8,3.0,3.2 to probe the N=500 capacity crossing).
    load_override = [float(x) for x in ov["alphas"].split(",")] if "alphas" in ov else None

    if not torch.cuda.is_available() and not allow_cpu:
        print("FATAL: no CUDA device; refusing CPU run (pass allow_cpu=1 for a tiny smoke).", flush=True)
        return 2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gsb = gs_base_lr(N)

    specs = [s for s in build_cell_specs() if not only or s["cell_id"].startswith(only)]
    for s in specs:
        if s.get("warmup") == "auto":            # resolve auto warmup (10% of budget)
            s["warmup"] = max(1, R_max // 10)
        if load_override is not None:            # probe a custom load grid (e.g. the capacity crossing)
            s["loads"] = list(load_override)

    cells_f = (run_dir / "cells.jsonl").open("w")
    curves_f = (run_dir / "curves.jsonl").open("w")
    calib_f = (run_dir / "calib.jsonl").open("w")

    # --- peak calibration per (optimizer, batch, calib_mu) needed by the specs ---
    # Keying on calib_mu (not the run mu) lets S9 share S4's peak so H-overshoot varies only momentum.
    def _ckey(s):
        return (s["optimizer"], s["batch"], s["calib_mu"])
    need = {_ckey(s) for s in specs if not s.get("faithful")}
    peaks = {}
    # build_cell_batch is deterministic in (master_seed, N, alpha), so the calibration batch is
    # byte-identical to the factorial's CAL_ALPHA batch — build it once and reuse it below.
    cal_batch = build_cell_batch(N, CAL_ALPHA, S, master_seed, grid_per_corr, dev) if need else None
    for (opt, b, mu) in sorted(need):
        crec = calibrate_peak(optimizer=opt, batch_size=b, mu=mu, cell_batch=cal_batch,
                              gs_base=gsb, r_max=R_max, n_points=n_calib)
        peaks[(opt, b, mu)] = crec["chosen_peak"]
        calib_f.write(json.dumps(crec) + "\n"); calib_f.flush()
        print(f"[v11] calib {opt} b={b} mu={mu}: peak={crec['chosen_peak']:.4g} edge={crec['edge']}", flush=True)

    for s in specs:
        if not s.get("faithful"):
            s["peak_lr"] = peaks[_ckey(s)]

    # --- factorial, LOAD-MAJOR: only ONE trace batch is resident at a time (N>=500 memory). All cells
    # that share a load run against the same batch, which is then freed before the next load is built.
    # CAL_ALPHA goes first so the still-resident calibration batch is consumed, not rebuilt. ---
    all_alphas = sorted({a for s in specs for a in s["loads"]}, key=lambda a: (a != CAL_ALPHA, a))
    if cal_batch is not None and CAL_ALPHA not in all_alphas:  # load override skipped CAL_ALPHA
        cal_batch = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for alpha in all_alphas:
        if cal_batch is not None:                          # first turn is CAL_ALPHA (ordering above)
            batch, cal_batch = cal_batch, None
        else:
            batch = build_cell_batch(N, alpha, S, master_seed, grid_per_corr, dev)
        for s in specs:
            if alpha not in s["loads"]:
                continue
            spec = {**s, "alpha": alpha}
            cap = (s["role"] == "spine") or (alpha == CAL_ALPHA)  # capture curves for plotted cells
            rec = run_cell(spec, batch, r_max=R_max, gs_base_lr=gsb, capture=cap)
            curve = rec.pop("curve", None)
            cells_f.write(json.dumps(rec) + "\n"); cells_f.flush()
            if curve is not None:
                curve.update(cell_id=rec["cell_id"], optimizer=rec["optimizer"],
                             batch=rec["batch"], shape=rec["shape"], alpha=alpha)
                curves_f.write(json.dumps(curve) + "\n"); curves_f.flush()
            print(f"[v11] {rec['cell_id']} a={alpha} solve={rec['solve_fraction']:.2f} "
                  f"med_ep={rec['epochs_to_solve_median']}", flush=True)
        del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cells_f.close(); curves_f.close(); calib_f.close()
    (run_dir / "manifest.json").write_text(json.dumps({
        "git_sha": v3._git_sha() or "unknown", "torch": torch.__version__, "numpy": np.__version__,
        "N": N, "S": S, "R_max": R_max, "alphas": sorted(all_alphas),
        "grid_per_corr": grid_per_corr, "n_calib": n_calib, "only": only or None,
        "master_seed": master_seed, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print("[v11] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
