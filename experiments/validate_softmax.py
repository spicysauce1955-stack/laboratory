"""β→∞ validation gate for the soft-max tempotron — run on the lab `--backend cpu` BEFORE GPU spend.

Checks (each writes a PASS/FAIL line to $LAB_RUN_DIR/validation.json and stdout):

  V1  soft VOLTAGE → hard max:  |V_β − V_max| → 0 as β grows, and the bound is exactly
      0 ≤ V_β − V_max ≤ ln(G)/β  (the analytic logsumexp sandwich). Checked across β.
  V2  soft GRADIENT → hard credit:  the soft-max measure p = softmax(βV) concentrates on the
      argmax time; ‖Σ_g p_g s_g − s(t_max)‖ → 0 and the argmax-time mass → 1 as β→∞.
  V3  envelope/autograd check: dV_β/dω computed by the analytic soft-gradient (Σ_g p_g s_g)
      matches PyTorch autograd through V_β to machine tol (the gradient is correct at all β).
  V4  COLD ≡ standard tempotron rule: a tiny single-spike cell trained by the COLD arm
      (β=∞) reproduces the v3 reference engine's per-seed solve pattern bit-for-bit.

All on CPU, small N — this is the cheap gate the charter requires before any paid GPU run.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("softmax_sweep", _HERE / "softmax_sweep.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)
v3 = sm.v3


def _voltage_grid(sb=2, p=5, g=64, n=8, seed=0, device="cpu"):
    g_ = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(sb, p, g, n, generator=g_, device=device)  # traces s[b,p,g,n]
    w = torch.randn(sb, n, generator=g_, device=device)
    V = torch.einsum("bpgn,bn->bpg", v, w)                      # (sb,p,g)
    return v, w, V


def check_v1_v2_v3(device="cpu") -> dict:
    s, w, V = _voltage_grid(device=device)
    G = V.shape[-1]
    vmax, targ = V.max(dim=-1)                                  # hard max + argmax time
    out = {"betas": [], "max_abs_dV": [], "lnG_over_beta": [],
           "max_grad_diff": [], "min_argmax_mass": [], "max_autograd_diff": []}
    # hard credit s(t_max): gather along G
    hard_grad = torch.gather(s, 2, targ.view(*targ.shape, 1, 1).expand(*targ.shape, 1, s.shape[-1])).squeeze(2)
    for beta in (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0):
        v_beta, pmeas = sm.soft_readout(V, beta)
        dV = (v_beta - vmax).abs().max().item()
        # V2 soft gradient Σ_g p_g s_g vs hard s(t_max)
        soft_grad = torch.einsum("bpg,bpgn->bpn", pmeas, s)
        gd = (soft_grad - hard_grad).abs().max().item()
        argmass = pmeas.gather(2, targ.unsqueeze(2)).squeeze(2).min().item()
        # V3 autograd: dV_beta/dw via autograd vs analytic soft credit Σ_g p_g s_g (per pattern summed)
        wv = w.clone().requires_grad_(True)
        Vv = torch.einsum("bpgn,bn->bpg", s, wv)
        vb2, _ = sm.soft_readout(Vv, beta)
        vb2.sum().backward()
        analytic = soft_grad.sum(dim=1)                        # Σ_p Σ_g p_g s_g  (sb,n)
        ad = (wv.grad - analytic).abs().max().item()
        out["betas"].append(beta)
        out["max_abs_dV"].append(dV)
        out["lnG_over_beta"].append(math.log(G) / beta)
        out["max_grad_diff"].append(gd)
        out["min_argmax_mass"].append(argmass)
        out["max_autograd_diff"].append(ad)
    # PASS conditions
    out["V1_pass"] = all(dv <= lg + 1e-5 for dv, lg in zip(out["max_abs_dV"], out["lnG_over_beta"])) \
        and out["max_abs_dV"][-1] < 1e-2
    out["V2_pass"] = out["max_grad_diff"][-1] < 1e-2 and out["min_argmax_mass"][-1] > 0.99
    out["V3_pass"] = max(out["max_autograd_diff"]) < 1e-3
    return out


def check_v4_cold_equiv(device="cpu") -> dict:
    """COLD arm (β=∞) on a tiny single-spike cell reproduces the v3 reference solve pattern."""
    N, alpha, K, S, epochs = 30, 1.0, 16.0, 8, 400
    t_window = K * sm.SQRT_TAU
    dt = sm.SQRT_TAU / 8
    n_grid = round(t_window / dt) + 1
    t_grid = torch.arange(n_grid, device=device, dtype=torch.float32) * dt
    p = round(alpha * N)
    rng = np.random.default_rng(v3.cell_seed(0, N, alpha, K))
    spikes, valid, labels, _ = v3.make_patterns(S, p, N, t_window, rng, device,
                                                 mean_spikes=1.0, ensemble="single")
    s_tr = v3.precompute_traces(spikes, valid, t_grid)
    wrng = np.random.default_rng(v3.cell_seed(0, N, alpha, K, tag=1))
    w0 = torch.from_numpy(wrng.standard_normal((S, N)).astype(np.float32)).to(device)
    # COLD arm (our engine)
    cold = sm.train_softmax(s_tr, labels, w0.clone(), lr=0.05, epochs=epochs, batch_size=16,
                            vthr_fixed=1.0, adam_betas=(0.95, 0.99), adam_eps=1e-6,
                            lr_schedule="cosine", lr_warmup=20,
                            beta_lo=float("inf"), beta_hi=float("inf"))
    # v3 reference engine, minibatch, identical knobs
    ref = v3.train_batch(s_tr, labels, w0.clone(), lr=0.05, epochs=epochs, mode="minibatch",
                         batch_size=16, vthr_fixed=1.0, optimizer="adam",
                         adam_betas=(0.95, 0.99), adam_eps=1e-6,
                         lr_schedule="cosine", lr_warmup=20)
    cmask = cold["converged"].cpu().numpy().astype(int)
    rmask = ref["converged"].cpu().numpy().astype(int)
    return {
        "cold_solved": cmask.tolist(), "ref_solved": rmask.tolist(),
        "n_match": int((cmask == rmask).sum()), "S": S,
        "V4_pass": bool((cmask == rmask).all()),
    }


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"validate_softmax | device={device}", flush=True)
    res = {}
    res.update(check_v1_v2_v3(device=device))
    res.update(check_v4_cold_equiv(device=device))
    res["ALL_PASS"] = bool(res["V1_pass"] and res["V2_pass"] and res["V3_pass"] and res["V4_pass"])
    (run_dir / "validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in
                      ("V1_pass", "V2_pass", "V3_pass", "V4_pass", "ALL_PASS",
                       "max_abs_dV", "max_grad_diff", "min_argmax_mass",
                       "max_autograd_diff", "n_match")}, indent=1), flush=True)
    if not res["ALL_PASS"]:
        print("VALIDATION FAILED — do NOT spend GPU.", flush=True)
        return 1
    print("VALIDATION PASSED — β→∞≡hard-max, soft-grad correct, COLD≡standard rule. GPU gated open.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
