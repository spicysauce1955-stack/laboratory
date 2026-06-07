"""V10 gates: the unified minibatch training path + LR schedule + Hyperband rung hook.

The V10 HPO study (``experiments/v10_hpo.py``) tunes the optimizer *and* the batch size
``b in {1, 16, 64, 256, P}`` (EXPERIMENT-SPEC.md Sec 6.1/10). The bare ``online`` (b=1) and
``batch`` (b=P) paths in ``v3_capacity_sweep.py`` are the validated V8/V9 reference and are left
untouched; this file gates the *new* ``mode='minibatch'`` path that the HPO driver calls.

* **G-mb-parity** -- ``minibatch`` at ``b=1`` (SGD, mu=0.99, constant lambda, kappa=0, fixed V_thr)
  reproduces the faithful ``online`` GS rule **bit-for-bit on the weight trajectory** (same patterns,
  same per-epoch shuffle): the whole point of unifying is that the b=1 endpoint *is* the GS baseline.
* **G-mb-sched** -- the extracted ``scheduled_lr`` matches the analytic constant/cosine/step/warmup
  forms (the schedules are tuned nuisance HPs; a silent schedule bug would bias every arm).
* **G-mb-rung** -- the Hyperband rung hook reports ``p_solve = mean_seeds(converged)`` at the
  requested epochs and a pruning callback that returns ``True`` stops training immediately (Hyperband
  early-stopping is load-bearing for the FTE budget; a broken hook would silently disable pruning).
* **G-mb-solve** -- the minibatch path is a correct solver (adaptive optimizers included): it drives
  an under-loaded single-spike task to zero error.

Run (remote GPU box; do NOT run locally per the experiment protocol):
    uv run --with torch python -m pytest experiments/tempotron_capacity/tests/test_v10_minibatch.py -v
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

# Load the study module directly (studies/ is not an importable package). Works in both the worktree
# layout (tests/ + ../studies/) and the flat lab layout (experiments/ siblings).
_STUDY = Path(__file__).resolve().parent / "v3_capacity_sweep.py"
if not _STUDY.exists():
    _STUDY = Path(__file__).resolve().parents[1] / "studies" / "v3_capacity_sweep.py"
_spec = importlib.util.spec_from_file_location("v3_capacity_sweep", _STUDY)
assert _spec is not None and _spec.loader is not None
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)

TAU_M, TAU_S = v3.TAU_M, v3.TAU_S
DUR = 66.667 * v3.SQRT_TAU  # the GS T = 500 ms window


def _single_spike_task(n_aff: int, alpha: float, seeds: int, n_grid: int = 400, seed: int = 0):
    """Build one single-spike capacity cell: precomputed traces, labels, balanced-ish init weights."""
    p = round(alpha * n_aff)
    rng = np.random.default_rng(v3.cell_seed(seed, n_aff, alpha, 67))
    st, vd, lab, _ = v3.make_patterns(seeds, p, n_aff, DUR, rng, torch.device("cpu"),
                                      ensemble="single")
    tg = torch.arange(n_grid, dtype=torch.float32) * (DUR / (n_grid - 1))
    s = v3.precompute_traces(st, vd, tg)
    w0 = torch.from_numpy(rng.standard_normal((seeds, n_aff), dtype=np.float32))
    return s, lab, w0


# --------------------------------------------------------------------------------------------
# G-mb-parity: b=1 minibatch SGD == faithful online GS on the weight trajectory.
# --------------------------------------------------------------------------------------------
def test_minibatch_b1_equals_online_gs() -> None:
    """The b=1 endpoint of the SGD arm reproduces the validated faithful-GS ``online`` rule exactly."""
    s, lab, w0 = _single_spike_task(n_aff=60, alpha=1.5, seeds=5, seed=1)
    lr = 3e-3 * DUR / (TAU_M * 60 * v3.PSP_V0)
    common = dict(lr=lr, momentum=0.99, epochs=400, vthr_fixed=1.0)

    torch.manual_seed(12345)
    ref = v3.train_batch(s, lab, w0, mode="online", **common)
    torch.manual_seed(12345)  # identical per-epoch shuffle sequence
    got = v3.train_batch(s, lab, w0, mode="minibatch", batch_size=1, optimizer="momentum", **common)

    # Identical weight trajectory => identical convergence, norms, and margins (state v differs only
    # by the constant factor lambda, which cancels in w; see the algebra in the V10 design note).
    assert torch.equal(got["converged"], ref["converged"]), "b=1 minibatch must converge like online"
    assert torch.equal(got["epochs_run"], ref["epochs_run"]), "same convergence epoch per seed"
    assert torch.allclose(got["wnorm"], ref["wnorm"], atol=1e-4, rtol=1e-4), "||w|| trajectory match"
    assert torch.allclose(got["mean_margin"], ref["mean_margin"], atol=1e-4, rtol=1e-4), "margins match"


# --------------------------------------------------------------------------------------------
# G-mb-sched: the extracted LR schedule matches its analytic form.
# --------------------------------------------------------------------------------------------
def test_scheduled_lr_constant() -> None:
    for ep in (0, 5, 99):
        assert v3.scheduled_lr(0.1, ep, epochs=100, schedule="none") == pytest.approx(0.1)


def test_scheduled_lr_warmup_is_linear() -> None:
    # Linear warmup over the first 10 epochs: lr*(ep+1)/10 for ep < 10, then full lr.
    assert v3.scheduled_lr(0.2, 0, epochs=100, schedule="none", warmup=10) == pytest.approx(0.02)
    assert v3.scheduled_lr(0.2, 9, epochs=100, schedule="none", warmup=10) == pytest.approx(0.2)
    assert v3.scheduled_lr(0.2, 50, epochs=100, schedule="none", warmup=10) == pytest.approx(0.2)


def test_scheduled_lr_cosine_decays_to_zero() -> None:
    lr0 = 0.3
    assert v3.scheduled_lr(lr0, 0, epochs=100, schedule="cosine") == pytest.approx(lr0)
    mid = v3.scheduled_lr(lr0, 50, epochs=100, schedule="cosine")
    assert mid == pytest.approx(0.5 * lr0, abs=0.02)
    assert v3.scheduled_lr(lr0, 100, epochs=100, schedule="cosine") == pytest.approx(0.0, abs=1e-6)


def test_scheduled_lr_step_decays_by_gamma() -> None:
    # Step schedule: lr * gamma**floor(ep/step_size).
    lr0, ss, g = 0.4, 10, 0.1
    assert v3.scheduled_lr(lr0, 0, epochs=100, schedule="step", step_size=ss, gamma=g) == pytest.approx(0.4)
    assert v3.scheduled_lr(lr0, 9, epochs=100, schedule="step", step_size=ss, gamma=g) == pytest.approx(0.4)
    assert v3.scheduled_lr(lr0, 10, epochs=100, schedule="step", step_size=ss, gamma=g) == pytest.approx(0.04)
    assert v3.scheduled_lr(lr0, 25, epochs=100, schedule="step", step_size=ss, gamma=g) == pytest.approx(0.004)


# --------------------------------------------------------------------------------------------
# G-mb-rung: the Hyperband rung hook reports p_solve and prunes.
# --------------------------------------------------------------------------------------------
def test_rung_hook_reports_psolve_at_requested_epochs() -> None:
    """report_cb is called at each requested epoch with p_solve = mean_seeds(converged so far)."""
    s, lab, w0 = _single_spike_task(n_aff=50, alpha=1.0, seeds=6, seed=2)
    lr = 3e-3 * DUR / (TAU_M * 50 * v3.PSP_V0)
    calls: list[tuple[int, float]] = []

    def cb(epoch: int, p_solve: float) -> bool:
        calls.append((epoch, p_solve))
        return False  # never prune

    res = v3.train_batch(s, lab, w0, mode="minibatch", batch_size=8, optimizer="momentum",
                         lr=lr, momentum=0.99, epochs=200, vthr_fixed=1.0,
                         report_epochs=(20, 50, 100), report_cb=cb)
    reported = {e for e, _ in calls}
    # Hooks fire at requested rungs that were actually reached (the run may converge & stop earlier).
    assert reported, "rung hook must fire at least once"
    assert reported.issubset({20, 50, 100}), f"unexpected rung epochs: {reported}"
    for epoch, p in calls:
        assert 0.0 <= p <= 1.0, "p_solve must be a fraction"
    # The reported p_solve is monotone non-decreasing (convergence is sticky via the freeze gate).
    ps = [p for _, p in calls]
    assert ps == sorted(ps), "p_solve over rungs must be non-decreasing"
    # Final p_solve >= last reported (more epochs can only help).
    assert float(res["converged"].float().mean()) >= ps[-1] - 1e-9


def test_rung_hook_prune_stops_training() -> None:
    """Returning True from report_cb prunes: training stops at that rung, not the full budget."""
    s, lab, w0 = _single_spike_task(n_aff=50, alpha=2.6, seeds=6, seed=3)  # near/above capacity: slow
    lr = 3e-3 * DUR / (TAU_M * 50 * v3.PSP_V0)
    seen: list[int] = []

    def cb(epoch: int, p_solve: float) -> bool:
        seen.append(epoch)
        return True  # prune at the first rung

    res = v3.train_batch(s, lab, w0, mode="minibatch", batch_size=8, optimizer="momentum",
                         lr=lr, momentum=0.99, epochs=5000, vthr_fixed=1.0,
                         report_epochs=(30, 100, 300), report_cb=cb)
    assert seen == [30], "must prune at the first rung and stop reporting"
    assert bool(res["pruned"]), "result must flag pruned=True"
    # epochs_run for still-unconverged seeds reflects the prune epoch, not the 5000-epoch budget.
    unconv = ~res["converged"]
    if bool(unconv.any()):
        assert int(res["epochs_run"][unconv].max()) <= 30, "pruned run must not consume the full budget"


# --------------------------------------------------------------------------------------------
# G-mb-solve: the minibatch path is a correct solver, adaptive optimizers included.
# --------------------------------------------------------------------------------------------
def test_minibatch_sgd_solves_underloaded() -> None:
    s, lab, w0 = _single_spike_task(n_aff=60, alpha=1.0, seeds=6, seed=4)
    lr = 5 * 3e-3 * DUR / (TAU_M * 60 * v3.PSP_V0)  # a few x the faithful base
    res = v3.train_batch(s, lab, w0, mode="minibatch", batch_size=16, optimizer="momentum",
                         lr=lr, momentum=0.99, epochs=3000, vthr_fixed=1.0)
    assert float(res["converged"].float().mean()) >= 0.8, "underloaded task must mostly solve"


def test_valid_mask_excludes_padded_patterns() -> None:
    """Masked (padded) pattern slots are inert: they neither block convergence nor shift V_thr.

    The V10 driver packs both band loads (alpha=2.3 P=460 and alpha=2.5 P=500) into ONE 32-row
    seed-batch so the rung hook reports the combined objective; the shorter load is padded up to P_max
    with ``valid_mask=0`` slots. We pad with **unsatisfiable** patterns (label +1, zero trace ->
    V_max=0 < V_thr=1, a permanent error if counted). With the mask the task still solves and V_thr
    calibrates to the *real* patterns only; without the mask the perma-errors block convergence.
    (Bit-equality is not expected -- the extra slots reshuffle the minibatches -- only inertness.)
    """
    s, lab, w0 = _single_spike_task(n_aff=40, alpha=1.0, seeds=4, seed=6)
    sb, p, g, n = s.shape
    lr = 3e-3 * DUR / (TAU_M * 40 * v3.PSP_V0)
    common = dict(mode="minibatch", batch_size=8, optimizer="momentum",
                  lr=lr, momentum=0.99, epochs=400, vthr_fixed=1.0)

    torch.manual_seed(7)
    ref = v3.train_batch(s, lab, w0, **common)

    pad = 6  # unsatisfiable +1 / zero-trace padding
    s_pad = torch.cat([s, torch.zeros(sb, pad, g, n)], dim=1)
    lab_pad = torch.cat([lab, torch.ones(sb, pad)], dim=1)
    mask = torch.cat([torch.ones(sb, p), torch.zeros(sb, pad)], dim=1)

    torch.manual_seed(7)
    masked = v3.train_batch(s_pad, lab_pad, w0, valid_mask=mask, **common)
    torch.manual_seed(7)
    unmasked = v3.train_batch(s_pad, lab_pad, w0, **common)  # no mask: perma-errors block it

    assert float(masked["converged"].float().mean()) >= 0.75, "masked padding must not block solving"
    assert float(unmasked["converged"].float().mean()) == 0.0, "unmasked perma-errors must block all"
    # V_thr calibrates on the real patterns only -> matches the unpadded reference.
    assert torch.allclose(masked["threshold"], ref["threshold"], atol=1e-5), "padding shifted V_thr"


def test_minibatch_adam_runs_and_solves() -> None:
    s, lab, w0 = _single_spike_task(n_aff=60, alpha=1.0, seeds=6, seed=5)
    res = v3.train_batch(s, lab, w0, mode="minibatch", batch_size=64, optimizer="adam",
                         lr=1e-2, epochs=3000, vthr_fixed=1.0,
                         adam_betas=(0.9, 0.999), adam_eps=1e-8)
    assert float(res["converged"].float().mean()) >= 0.5, "Adam must make real progress under-loaded"
