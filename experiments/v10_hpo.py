"""V10 -- pre-registered optimizer/HPO study: the *findable* tempotron capacity under tuned learning.

Self-contained lab entrypoint (Experiment Contract: reads ``$LAB_RUN_DIR``/``$LAB_SEED``, writes all
artifacts there; the only local import is the sibling study file ``v3_capacity_sweep.py`` -- the lab
ships both under ``experiments/``). Implements EXPERIMENT-SPEC.md Sec 10/14.1 *exactly*:

* **CASH conditional space (Optuna define-by-run).** One *arm* per optimizer
  ``arm in {sgd, adam, rmsprop}`` (+ a ``random`` baseline over the union at 2x compute). Each trial
  samples the arm's batch size ``b in {1,16,64,256,P}``, learning rate, optimizer HPs and LR schedule
  (Sec 10 table), then trains the faithful-GS minibatch rule (``v3.train_batch(mode='minibatch')``).
* **Objective.** Mean ``p_solve`` over the contested band ``alpha in {2.3, 2.5}`` at ``N=200``,
  averaged over ``n_seeds=16`` tasks. Both loads are packed into ONE seed-batch (16 seeds x 2 alpha =
  32 rows; the shorter load zero-padded with ``valid_mask=0`` inert slots) so a single training run
  yields the combined objective and Hyperband can prune on it natively.
* **Multi-fidelity.** ``HyperbandPruner(R_min=200 -> R_max=5000, eta=3)``; ``train_batch``'s rung hook
  reports intermediate ``p_solve`` at the Hyperband rungs and prunes mid-training.
* **Budget = training compute, not trial count (pruning-valid).** The driver stops when cumulative
  ``sum(seed-epochs) >= budget_fte`` full-training-equivalents, ``1 FTE = n_seeds x R_max =
  16 x 5000 = 80,000`` seed-epochs (random baseline gets ``2x``). *Trials-completed is an outcome.*
* **Teardown-robust artifacts** streamed to ``$LAB_RUN_DIR``: ``optuna_journal.log`` (JournalStorage),
  ``trials.jsonl`` (per-trial config/objective/fte/intermediate curve), ``best_config.json``,
  ``manifest.json`` (git SHA, library versions, the frozen config).

Run (REMOTE GPU only -- do not run locally; tiny demo shown):
    LAB_RUN_DIR=/tmp/v10 LAB_SEED=0 uv run --with torch --with optuna --with scipy \\
        python experiments/v10_hpo.py arm=sgd budget_fte=0.4 tune_N=40 n_seeds=4 \\
        R_min=20 R_max=180 eta=3 max_trials=12
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- load the sibling study module (single source of truth for the model + train_batch) ----------
_STUDY = Path(__file__).resolve().parent / "v3_capacity_sweep.py"
if not _STUDY.exists():  # lab layout: both files live under experiments/
    _STUDY = Path(__file__).resolve().parents[1] / "studies" / "v3_capacity_sweep.py"
_spec = importlib.util.spec_from_file_location("v3_capacity_sweep", _STUDY)
assert _spec is not None and _spec.loader is not None
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)

import optuna  # noqa: E402  (after v3 so torch import errors surface first)

TAU_M, SQRT_TAU, PSP_V0 = v3.TAU_M, v3.SQRT_TAU, v3.PSP_V0
T_WINDOW = 500.0           # GS single-spike window (ms); K = T/sqrt(tau_s tau_m) ~ 66.7
K_TEMPORAL = T_WINDOW / SQRT_TAU
BATCH_CHOICES = ["1", "16", "64", "256", "P"]  # Sec 10: b in {1,16,64,256,P} (P => full batch)


# --------------------------------------------------------------------------------------------
# Fixed tuning batch: precompute ONCE (HPs are tuned on the same tasks across all trials).
# --------------------------------------------------------------------------------------------
def build_tuning_batch(n: int, alphas: tuple[float, ...], n_seeds: int, master_seed: int,
                       grid_per_corr: int, device: torch.device):
    """Pack ``n_seeds`` tasks at each band load into one ``(len(alphas)*n_seeds, P_max, G, N)`` batch.

    Each (alpha, seed) row is an independent single-spike task; shorter loads are zero-padded to
    ``P_max`` and flagged inert via ``valid_mask=0`` (so padding never counts as an error or shifts
    the fixed-threshold calibration). Init weights are drawn with the same device-independent RNG.
    """
    dt = SQRT_TAU / grid_per_corr
    n_grid = round(T_WINDOW / dt) + 1
    t_grid = torch.arange(n_grid, dtype=torch.float32, device=device) * dt
    p_per = [round(a * n) for a in alphas]
    p_max = max(p_per)
    rows = len(alphas) * n_seeds

    s_all = torch.zeros((rows, p_max, n_grid, n), dtype=torch.float32, device=device)
    lab_all = torch.ones((rows, p_max), dtype=torch.float32, device=device)  # padded slots = +1
    mask = torch.zeros((rows, p_max), dtype=torch.float32, device=device)
    w0 = torch.empty((rows, n), dtype=torch.float32, device=device)

    for ai, alpha in enumerate(alphas):
        p_a = p_per[ai]
        rng = np.random.default_rng(v3.cell_seed(master_seed, n, alpha, K_TEMPORAL))
        st, vd, lab, _ = v3.make_patterns(n_seeds, p_a, n, T_WINDOW, rng, device, ensemble="single")
        s_a = v3.precompute_traces(st, vd, t_grid)                       # (n_seeds, P_a, G, N)
        r0 = ai * n_seeds
        s_all[r0:r0 + n_seeds, :p_a] = s_a
        lab_all[r0:r0 + n_seeds, :p_a] = lab
        mask[r0:r0 + n_seeds, :p_a] = 1.0
        w0[r0:r0 + n_seeds] = torch.from_numpy(
            rng.standard_normal((n_seeds, n), dtype=np.float32)).to(device)
    return s_all, lab_all, mask, w0, p_max


def hyperband_rungs(r_min: int, r_max: int, eta: int) -> list[int]:
    """Epochs at which to report p_solve to the pruner: r_min, r_min*eta, ... , r_max."""
    rungs, r = [], r_min
    while r < r_max:
        rungs.append(int(r))
        r *= eta
    rungs.append(int(r_max))
    return rungs


# --------------------------------------------------------------------------------------------
# CASH define-by-run search space (Sec 10).
# --------------------------------------------------------------------------------------------
def suggest_config(trial: optuna.Trial, arm: str, gs_base_lr: float, r_max: int) -> dict:
    """Sample one config from the arm's conditional space; returns train_batch kwargs + bookkeeping."""
    opt = arm if arm in ("sgd", "adam", "rmsprop") else \
        trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])  # random arm: union
    b_tok = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    warmup_max = min(500, max(0, r_max // 4))
    warmup = trial.suggest_int("warmup", 0, warmup_max)

    cfg: dict = {"optimizer_arm": opt, "batch_token": b_tok, "warmup": warmup}
    if opt == "sgd":
        lr_mult = trial.suggest_float("lr_mult", 0.3, 50.0, log=True)
        cfg["lr"] = lr_mult * gs_base_lr
        cfg["lr_mult"] = lr_mult
        cfg["momentum"] = trial.suggest_categorical("momentum", [0.0, 0.9, 0.99, 0.999])
        cfg["optimizer"] = "momentum"
        sched = trial.suggest_categorical("sched", ["none", "cosine", "step"])
    else:
        cfg["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        if opt == "adam":
            cfg["optimizer"] = "adam"
            cfg["adam_betas"] = (trial.suggest_categorical("beta1", [0.9, 0.95]),
                                 trial.suggest_categorical("beta2", [0.99, 0.999]))
            cfg["adam_eps"] = trial.suggest_categorical("adam_eps", [1e-8, 1e-6])
        else:
            cfg["optimizer"] = "rmsprop"
            cfg["rms_alpha"] = trial.suggest_categorical("rms_alpha", [0.9, 0.99])
            cfg["rms_eps"] = trial.suggest_categorical("rms_eps", [1e-8, 1e-6])
        sched = trial.suggest_categorical("sched", ["none", "cosine"])

    cfg["lr_schedule"] = sched
    if sched == "step":
        cfg["lr_step_size"] = trial.suggest_int("step_size", max(1, r_max // 10), max(2, r_max // 2))
        cfg["lr_gamma"] = trial.suggest_categorical("gamma", [0.1, 0.3, 0.5])
    return cfg


def train_kwargs(cfg: dict, p_max: int, r_max: int) -> dict:
    """Map a sampled config to v3.train_batch keyword arguments."""
    bs = p_max if cfg["batch_token"] == "P" else int(cfg["batch_token"])
    kw = dict(
        mode="minibatch", batch_size=bs, optimizer=cfg["optimizer"], lr=cfg["lr"],
        epochs=r_max, vthr_fixed=1.0, lr_schedule=cfg["lr_schedule"], lr_warmup=cfg["warmup"],
        momentum=cfg.get("momentum", 0.0),
    )
    if cfg["optimizer"] == "adam":
        kw["adam_betas"] = cfg["adam_betas"]
        kw["adam_eps"] = cfg["adam_eps"]
    elif cfg["optimizer"] == "rmsprop":
        kw["rms_alpha"] = cfg["rms_alpha"]
        kw["rms_eps"] = cfg["rms_eps"]
    if cfg["lr_schedule"] == "step":
        kw["lr_step_size"] = cfg["lr_step_size"]
        kw["lr_gamma"] = cfg["lr_gamma"]
    return kw


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=str(Path(__file__).resolve().parent),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(os.environ.get("LAB_SEED", "0"))
    ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)

    arm = ov.get("arm", "sgd")
    assert arm in ("sgd", "adam", "rmsprop", "random"), f"bad arm {arm}"
    tune_n = int(ov.get("tune_N", "200"))
    n_seeds = int(ov.get("n_seeds", "16"))
    alphas = tuple(float(x) for x in ov.get("alpha_obj", "2.3,2.5").split(","))
    r_min = int(ov.get("R_min", "200"))
    r_max = int(ov.get("R_max", "5000"))
    eta = int(ov.get("eta", "3"))
    grid_per_corr = int(ov.get("grid_per_corr", "8"))
    budget_fte = float(ov.get("budget_fte", "120" if arm == "random" else "60"))
    max_trials = int(ov.get("max_trials", "100000"))  # safety cap; FTE budget is the real control

    fte_unit = float(n_seeds * r_max)  # 1 FTE = n_seeds x R_max seed-epochs
    gs_base_lr = 3e-3 * T_WINDOW / (TAU_M * tune_n * PSP_V0)
    rungs = hyperband_rungs(r_min, r_max, eta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gb:.0f} GB)", flush=True)
    else:
        print("WARNING: no CUDA device; CPU (smoke only).", flush=True)
    print(f"V10 HPO | arm={arm} seed={master_seed} N={tune_n} seeds={n_seeds} alphas={alphas} "
          f"R={r_min}->{r_max} eta={eta} rungs={rungs} budget_fte={budget_fte} "
          f"gs_base_lr={gs_base_lr:.3e} fte_unit={fte_unit:.0f}", flush=True)

    s_all, lab_all, mask, w0, p_max = build_tuning_batch(
        tune_n, alphas, n_seeds, master_seed, grid_per_corr, device)
    rows = s_all.shape[0]
    print(f"tuning batch: rows={rows} P_max={p_max} grid={s_all.shape[2]} "
          f"trace={s_all.numel() * 4 / 1e9:.2f} GB", flush=True)

    trials_path = run_dir / "trials.jsonl"
    journal_path = run_dir / "optuna_journal.log"
    storage = None
    try:  # JournalStorage API moved across Optuna versions; fall back to in-memory (trials.jsonl is durable)
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        storage = JournalStorage(JournalFileBackend(str(journal_path)))
    except Exception:
        try:
            from optuna.storages import JournalFileStorage, JournalStorage
            storage = JournalStorage(JournalFileStorage(str(journal_path)))
        except Exception as exc:  # pragma: no cover
            print(f"  (JournalStorage unavailable: {exc}; using in-memory + trials.jsonl)", flush=True)

    sampler = (optuna.samplers.RandomSampler(seed=master_seed) if arm == "random"
               else optuna.samplers.TPESampler(multivariate=True, seed=master_seed))
    pruner = optuna.pruners.HyperbandPruner(min_resource=r_min, max_resource=r_max,
                                            reduction_factor=eta)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                storage=storage, study_name=f"v10_{arm}", load_if_exists=True)

    state = {"fte": 0.0, "completed": 0, "pruned": 0, "t0": time.time()}

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial, arm, gs_base_lr, r_max)
        kw = train_kwargs(cfg, p_max, r_max)
        rung_curve: list[tuple[int, float]] = []

        def cb(epoch: int, p_solve: float) -> bool:
            rung_curve.append((epoch, p_solve))
            trial.report(p_solve, step=epoch)
            return trial.should_prune()

        res = v3.train_batch(s_all, lab_all, w0, valid_mask=mask,
                             report_epochs=tuple(rungs), report_cb=cb, **kw)
        # Objective + exact compute consumed (seed-epochs actually run, incl. early prune/converge).
        solved = (res["converged"] & torch.isfinite(res["wnorm"])).float()
        objective_value = float(solved.mean().item())
        row_epochs = int(res["epochs_run"].sum().item())
        fte_trial = row_epochs / fte_unit
        state["fte"] += fte_trial
        pruned = bool(res["pruned"])

        rec = {
            "trial": trial.number, "arm": arm, "objective": objective_value,
            "pruned": pruned, "fte_consumed": fte_trial, "fte_cumulative": state["fte"],
            "row_epochs": row_epochs, "config": {k: _jsonable(v) for k, v in cfg.items()},
            "rung_curve": rung_curve, "wall_s": round(time.time() - state["t0"], 1),
        }
        with trials_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  trial {trial.number}: arm={arm} obj={objective_value:.3f} "
              f"{'PRUNED' if pruned else 'done'} fte={fte_trial:.3f} "
              f"(cum {state['fte']:.2f}/{budget_fte}) {cfg['optimizer']} b={cfg['batch_token']}",
              flush=True)
        if pruned:
            state["pruned"] += 1
            raise optuna.TrialPruned()
        state["completed"] += 1
        return objective_value

    def budget_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if state["fte"] >= budget_fte or (state["completed"] + state["pruned"]) >= max_trials:
            study.stop()

    study.optimize(objective, n_trials=max_trials, callbacks=[budget_cb],
                   catch=(RuntimeError,))  # a divergent config can OOM/NaN; record & continue

    # ---- winners + manifest -----------------------------------------------------------------
    try:
        best = study.best_trial
        best_cfg = {"objective": best.value, "trial": best.number, "params": best.params}
    except ValueError:
        best_cfg = {"objective": None, "trial": None, "params": {}, "note": "no completed trials"}
    (run_dir / "best_config.json").write_text(json.dumps(best_cfg, indent=2))

    manifest = {
        "study": "v10_hpo", "arm": arm, "git_sha": _git_sha(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "torch": torch.__version__, "optuna": optuna.__version__},
        "frozen_config": {
            "tune_N": tune_n, "n_seeds": n_seeds, "alpha_obj": list(alphas), "master_seed": master_seed,
            "R_min": r_min, "R_max": r_max, "eta": eta, "rungs": rungs, "grid_per_corr": grid_per_corr,
            "budget_fte": budget_fte, "fte_unit": fte_unit, "gs_base_lr": gs_base_lr,
            "ensemble": "single", "vthr_fixed": 1.0, "kappa": 0.0, "T": T_WINDOW, "K": K_TEMPORAL,
            "batch_choices": BATCH_CHOICES,
        },
        "result": {"fte_consumed": state["fte"], "completed": state["completed"],
                   "pruned": state["pruned"], "best": best_cfg},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE arm={arm}: completed={state['completed']} pruned={state['pruned']} "
          f"fte={state['fte']:.2f} best_obj={best_cfg['objective']}", flush=True)
    return 0


def _jsonable(v):
    if isinstance(v, tuple):
        return list(v)
    return v


if __name__ == "__main__":
    raise SystemExit(main())
