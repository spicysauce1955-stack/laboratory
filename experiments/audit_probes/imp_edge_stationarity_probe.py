"""AUDIT PROBE -- pure numpy, <30 s, no project imports, no training.

Question: v6_bandlimited_capacity.band_kernel (lines 99-131) is a Lanczos-windowed sinc
with compact support |s| <= n_lobes*pi/Omega_b (n_lobes=6 default). With
Omega_b = sqrt(3)*K_eff/T, the kernel HALF-SUPPORT is

    L_half = 6*pi/Omega_b = 6*pi*T/(sqrt(3)*K_eff) ~= 10.88 * T / K_eff

At K_eff = 8 that is 1.36 T -- LARGER THAN THE WHOLE PATTERN WINDOW. Since spikes are
drawn only inside [0,T], the readout V(t) is then strongly NON-STATIONARY across [0,T]
(edge/taper effects), which is exactly what the flat-band Rice theory assumes away.

This probe reimplements v6's band_kernel + Poisson pattern generation in numpy and
measures Var(V(t)) across the window at each K_eff of the headline ladder, plus the
realised spectrum's K_eff. Fixed seed, read-only w.r.t. the project.
"""

import numpy as np

RNG = np.random.default_rng(4242)
T = 500.0
OVERSAMPLE = 4.0
N_LOBES = 6
N_AFF = 500
N_PAT = 400
MEAN_SPIKES = 2.0   # committed corrected/*.csv density


def band_kernel(s, omega_b, n_lobes=N_LOBES):
    """Verbatim numpy port of v6_bandlimited_capacity.band_kernel (lanczos window)."""
    half = n_lobes * np.pi / omega_b
    x = omega_b * s
    sinc = np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))
    xw = np.pi * s / half
    win = np.where(np.abs(xw) < 1e-12, 1.0, np.sin(xw) / np.where(xw == 0, 1.0, xw))
    return np.where(np.abs(s) <= half, sinc * win, 0.0)


print(f"{'K_eff':>7} {'n_grid':>7} {'half_supp/T':>12} {'Var(V) edge/centre':>19} "
      f"{'Var ratio min/max':>18} {'meas K_eff':>11}")

for keff in (8.0, 16.0, 32.0, 64.0, 128.0):
    omega = np.sqrt(3.0) * keff / T
    dt = np.pi / (OVERSAMPLE * omega)
    n_grid = round(T / dt) + 1
    t = np.arange(n_grid) * dt
    half = N_LOBES * np.pi / omega

    counts = RNG.poisson(MEAN_SPIKES, size=(N_PAT, N_AFF))
    ms = max(1, counts.max())
    st = T * RNG.random((N_PAT, N_AFF, ms))
    valid = (np.arange(ms)[None, None, :] < counts[..., None]).astype(float)

    # per-afferent traces s[p, g, i]
    s = np.einsum("pigf->pig", band_kernel(t[None, None, :, None] - st[:, :, None, :], omega)
                  * valid[:, :, None, :])
    w = RNG.standard_normal((N_PAT, N_AFF))
    V = np.einsum("pgi,pi->pg", s.transpose(0, 2, 1), w)   # (N_PAT, n_grid)

    var_t = V.var(axis=0)                                  # variance profile across the window
    ncen = max(1, n_grid // 5)
    centre = var_t[n_grid // 2 - ncen // 2: n_grid // 2 + ncen // 2 + 1].mean()
    edge = 0.5 * (var_t[0] + var_t[-1])

    Vc = V - V.mean(axis=1, keepdims=True)
    signs = np.signbit(Vc)
    ncross = (signs[:, 1:] != signs[:, :-1]).sum(axis=1).mean()
    meas_k = np.pi * (ncross / T) * T

    print(f"{keff:7.1f} {n_grid:7d} {half/T:12.3f} {edge/centre:19.3f} "
          f"{var_t.min()/var_t.max():18.3f} {meas_k:11.2f}")

print("\nInterpretation key: for a STATIONARY flat-band field the variance profile is flat,")
print("so edge/centre -> 1.00 and min/max -> 1.00. Departures are window/taper artefacts")
print("that the Rice K_eff = Omega T/sqrt(3) axis does not model.")
