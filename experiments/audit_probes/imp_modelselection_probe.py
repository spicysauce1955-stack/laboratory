"""AUDIT PROBE -- re-runs the headline model selection of
experiments/sinusoidal_capacity/analysis/fit_formtest_clean.py (lines 59-70)
on the 5 published crossings, because the raw data it reads (`runs/*/manifest.json`)
is NOT present in the repository.

Inputs are transcribed from experiments/sinusoidal_capacity/EXPERIMENT.md (the "Analyze
(2026-07-28)" table) -- the ONLY surviving record of the headline ladder.

Checks:
 A. reproduce SSE / dAIC exactly as the script computes them (sanity on the arithmetic)
 B. propagate the project's OWN stated per-cell crossing uncertainty (+/-0.05..0.1,
    EXPERIMENT.md pre-registration) -> chi^2 goodness of fit for each model
 C. re-do the selection on the PHYSICAL EVT trial count n = K_eff/(2 pi)
    (the actual expected number of zero up-crossings; K_eff is 2 pi x that by the
    chapter's own definition, study/05-rice-qualls-keff.tex L336-347) -- i.e. test
    whether the winner survives a change of the x-variable by a constant factor.
 D. same for n = K_eff * sqrt(3/5) / (2 pi) (expected number of LOCAL MAXIMA, the
    natural EVT trial count).
"""

import math

K = [8.3, 15.6, 30.8, 61.3, 121.7]
AC = [2.786, 3.067, 3.250, 3.480, 3.600]


def fit(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    aic = n * math.log(sse / n) + 4
    return b, a, sse, aic


MODELS = [("ln ln x", lambda k: math.log(math.log(k))),
          ("ln x", math.log),
          ("x (lin)", lambda k: k)]

AXES = [("K_eff (as published)", lambda k: k),
        ("n_upcross = K/2pi", lambda k: k / (2 * math.pi)),
        ("n_maxima = K*sqrt(3/5)/2pi", lambda k: k * math.sqrt(3 / 5) / (2 * math.pi))]

for axname, xf0 in AXES:
    print(f"\n=== x-axis: {axname} ===")
    xs0 = [xf0(k) for k in K]
    print("   x values:", [round(v, 3) for v in xs0])
    out = {}
    for nm, xf in MODELS:
        try:
            xs = [xf(v) for v in xs0]
        except ValueError:
            print(f"  {nm:9s}: UNDEFINED on this axis (log of non-positive)")
            continue
        b, a, sse, aic = fit(xs, AC)
        out[nm] = (sse, aic)
        print(f"  {nm:9s}: slope={b:+.3f} intercept={a:+.3f} SSE={sse:.5f} AIC={aic:+.3f} "
              f"RMSresid={math.sqrt(sse/5):.4f}")
    if "ln ln x" in out and "ln x" in out:
        print(f"  dAIC(ln x - ln ln x) = {out['ln x'][1]-out['ln ln x'][1]:+.2f}  "
              f"(project reports +8.7 on the published axis)")

print("\n=== B. goodness of fit vs the project's OWN stated crossing uncertainty ===")
xs_lnln = [math.log(math.log(k)) for k in K]
xs_ln = [math.log(k) for k in K]
for nm, xs in (("ln ln K", xs_lnln), ("ln K", xs_ln)):
    b, a, sse, _ = fit(xs, AC)
    print(f"  {nm}: SSE={sse:.5f}, RMS residual={math.sqrt(sse/5):.4f} in alpha units")
    for sig in (0.03, 0.05, 0.10):
        chi2 = sse / sig ** 2
        print(f"     sigma_cell={sig:.2f} -> chi2={chi2:6.2f} on dof=3 "
              f"({'ACCEPTABLE' if chi2 < 7.81 else 'REJECTED at 5%'})")
