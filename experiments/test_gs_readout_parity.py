"""A5 parity gate: vectorised GS-2006 readout (batched engine) == serial reference.

The faithful online GS rule reads out ``V_max`` and the credit-assignment time ``t_max`` under the
Gütig-Sompolinsky Suppl.-Methods convention (hyperpolarizing init PSP + dead-trajectory fallback).
The batched study implements this with two precomputed tensors (``gs_readout_terms`` + the
``readout='gs'`` path in ``_forward``); the ground truth is the obvious O(grid) numpy reference
``_serial_vmaxgs`` below -- a line-for-line transcription of :func:`tempotron.decision.v_max_gs`.

Self-contained so the laboratory can run it next to ``v3_capacity_sweep.py`` with no package mounts
(it only imports that one self-contained study module). When the ``tempotron`` package *is*
importable (repo / pytest), an extra check asserts the inline reference matches the canonical
``decision.v_max_gs`` bit-for-bit, so the reference cannot silently share a bug with the engine.

Runnable as a script (prints PASS/FAIL, exits non-zero on failure) for a lab gate; also a pytest.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

# Locate the self-contained study module (lab: same dir; repo: ../studies).
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / "studies"):
    if (_cand / "v3_capacity_sweep.py").exists():
        sys.path.insert(0, str(_cand))
        break
import v3_capacity_sweep as v3  # noqa: E402

TAU_M, TAU_S = v3.TAU_M, v3.TAU_S


def _serial_vmaxgs(spk_list, w, tau_m, tau_s, duration, n_grid):
    """Reference (V_max, t_max) under the GS-2006 Suppl.-Methods readout (numpy, O(grid))."""
    t = np.linspace(0.0, duration, n_grid)
    tp = tau_m * tau_s / (tau_m - tau_s) * math.log(tau_m / tau_s)
    v0 = 1.0 / (math.exp(-tp / tau_m) - math.exp(-tp / tau_s))

    def kern(x):
        xc = np.clip(x, 0.0, None)
        return np.where(x >= 0.0, v0 * (np.exp(-xc / tau_m) - np.exp(-xc / tau_s)), 0.0)

    v = np.zeros_like(t)
    all_times = []
    for i, spk in enumerate(spk_list):
        for ts_ in spk:
            v = v + w[i] * kern(t - ts_)
            all_times.append(ts_)
    at = np.asarray(all_times)
    if at.size:
        v = v - 0.01 * kern(t - (at.min() - tp))           # GS rule 1: hyperpolarizing init PSP
    idx = int(np.argmax(v))
    vmax = float(v[idx])
    if at.size and vmax <= 0.0:
        return vmax, float(at.max()) + tp                  # GS rule 2: dead-trajectory fallback
    return vmax, float(t[idx])


def _build(n_aff: int = 200, n_patterns: int = 24, k: float = 66.667, seed: int = 7):
    """Single-spike patterns + random weights on a grid shared by both implementations."""
    t_window = k * v3.SQRT_TAU
    dt = v3.SQRT_TAU / 16.0
    n_grid = round(t_window / dt) + 1
    t_grid = torch.arange(n_grid, dtype=torch.float32) * dt
    duration = float((n_grid - 1) * dt)  # == t_grid[-1]; makes serial linspace coincide with t_grid
    rng = np.random.default_rng(seed)
    spikes, valid, _labels, _ms = v3.make_patterns(
        1, n_patterns, n_aff, t_window, rng, torch.device("cpu"), ensemble="single"
    )
    w = torch.from_numpy(rng.standard_normal(n_aff).astype(np.float32)).view(1, n_aff)
    return spikes, valid, w, t_grid, duration, n_grid


def _batched(spikes, valid, w, t_grid):
    s = v3.precompute_traces(spikes, valid, t_grid)           # (1,P,G,N)
    vbias, tfall = v3.gs_readout_terms(spikes, valid, t_grid)  # (1,P,G), (1,P)
    vmax, targ = v3._forward(s, w, None, vbias, tfall)         # (1,P) each
    return vmax[0].numpy(), t_grid[targ][0].numpy()


def _serial(spikes, valid, w, duration, n_grid):
    st, vd, wv = spikes[0].numpy(), valid[0].numpy(), w[0].numpy().astype(np.float64)
    P, N = st.shape[0], st.shape[1]
    vmax, tmax = np.empty(P), np.empty(P)
    for pi in range(P):
        spk = [st[pi, i][vd[pi, i] > 0].astype(np.float64) for i in range(N)]
        vmax[pi], tmax[pi] = _serial_vmaxgs(spk, wv, TAU_M, TAU_S, duration, n_grid)
    return vmax, tmax


def _compare(spikes, valid, w, t_grid, duration, n_grid, tag: str):
    """Return (v_ok, t_ok, n_dead, msg) comparing batched vs serial for one weight vector."""
    vb, tb = _batched(spikes, valid, w, t_grid)
    vs, ts = _serial(spikes, valid, w, duration, n_grid)
    dt = float(t_grid[1] - t_grid[0])
    # The batched engine's credit-assignment time is necessarily a grid index, so the serial fallback
    # time t_last+t_peak (which can exceed the window) is clamped to the grid before comparison.
    ts_cl = np.clip(ts, float(t_grid[0]), float(t_grid[-1]))
    dv, dtm = np.abs(vb - vs), np.abs(tb - ts_cl)
    v_scale = max(1e-9, float(np.abs(vs).max()))
    v_ok = bool(dv.max() <= 1e-3 * v_scale + 1e-5)
    t_ok = bool(dtm.max() <= 2.0 * dt + 1e-6)  # argmax may differ a cell at near-ties (float32 vs float64)
    n_dead = int((vs <= 0.0).sum())
    msg = (f"  [{tag}] V_max max|Δ|={dv.max():.3e} ({'OK' if v_ok else 'FAIL'})  "
           f"t_max max|Δ|={dtm.max():.4f}ms/tol{2*dt:.3f} ({'OK' if t_ok else 'FAIL'})  "
           f"dead={n_dead}/{len(vs)}")
    return v_ok, t_ok, n_dead, msg


def _validate_reference_against_canonical(spikes, valid, w, duration, n_grid) -> str:
    """If the tempotron package is importable, assert the inline ref == canonical decision.v_max_gs."""
    try:
        for cand in (_HERE.parent / "src",):
            if (cand / "tempotron").exists():
                sys.path.insert(0, str(cand))
        from tempotron.decision import v_max_gs as canon
    except Exception as exc:  # noqa: BLE001 -- lab context has no tempotron package; skip
        return f"  (canonical decision.v_max_gs not importable: {type(exc).__name__}; inline-ref check skipped)"
    st, vd, wv = spikes[0].numpy(), valid[0].numpy(), w[0].numpy().astype(np.float64)
    worst = 0.0
    for pi in range(st.shape[0]):
        spk = [st[pi, i][vd[pi, i] > 0].astype(np.float64) for i in range(st.shape[1])]
        vi, ti = _serial_vmaxgs(spk, wv, TAU_M, TAU_S, duration, n_grid)
        vc, tc = canon(spk, wv, TAU_M, TAU_S, duration, n_grid=n_grid)
        worst = max(worst, abs(vi - vc), abs(ti - tc))
    assert worst < 1e-9, f"inline reference diverges from canonical v_max_gs (max|Δ|={worst:.2e})"
    return f"  inline reference == canonical decision.v_max_gs (max|Δ|={worst:.1e})  OK"


def check() -> bool:
    spikes, valid, w, t_grid, duration, n_grid = _build()
    dt = float(t_grid[1] - t_grid[0])
    print(f"N=200 single-spike, P={spikes.shape[1]} patterns, grid dt={dt:.4f} ms ({n_grid} pts)")

    # Case A: generic N(0,1) weights (V_max>0 -> argmax path, GS rule 1 only).
    vA, tA, deadA, mA = _compare(spikes, valid, w, t_grid, duration, n_grid, "rand-w  ")
    print(mA)
    # Case B: all-negative weights force V<=0 everywhere -> exercises the dead-trajectory
    # fallback (GS rule 2: t_max = t_last + t_peak, clamped to the grid).
    w_neg = -torch.abs(w)
    vB, tB, deadB, mB = _compare(spikes, valid, w_neg, t_grid, duration, n_grid, "neg-w   ")
    print(mB)

    ref_note = _validate_reference_against_canonical(spikes, valid, w, duration, n_grid)
    print(ref_note)

    coverage_ok = deadB > 0  # the fallback path must actually be hit by case B
    if not coverage_ok:
        print("  FAIL: case B produced no dead trajectories -> rule-2 fallback never exercised")
    ok = bool(vA and tA and vB and tB and coverage_ok)
    print(f"GATE {'PASS' if ok else 'FAIL'}  "
          f"(rule-1 argmax: case A; rule-2 fallback: {deadB}/{spikes.shape[1]} dead in case B)")
    return ok


def test_gs_readout_parity():
    assert check()


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
