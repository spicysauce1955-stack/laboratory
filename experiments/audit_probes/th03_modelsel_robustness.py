"""
TH audit part 3 — how robust is the {lnlnK, lnK, K} model selection?

Two questions the project does not ask:
  (1) The lnK regressor is INVARIANT under K -> cK (ln(cK)=lnK+ln c, absorbed by the
      intercept). The lnlnK regressor is NOT. So the comparison is convention-dependent.
      How much does an O(1) axis rescale move Delta AIC?
  (2) With n=5 points and no reported per-point s.e., what measurement noise would
      make the two forms statistically indistinguishable?

Run: uv run --with numpy --with scipy python th03_modelsel_robustness.py
"""
import numpy as np

K = np.array([8.3, 15.6, 30.8, 61.3, 121.7])
a = np.array([2.79, 3.07, 3.25, 3.48, 3.60])
n = len(K)

def fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    return c[0], float(r @ r)

def aic(rss, n, k=3):     # k = slope + intercept + sigma
    return n*np.log(rss/n) + 2*k

print("=" * 78)
print("A. Baseline model selection, as the project reports it")
print("=" * 78)
regs = {"lnlnK": np.log(np.log(K)), "lnK": np.log(K), "K": K}
res = {}
for name, x in regs.items():
    s, rss = fit(x, a)
    res[name] = (s, rss, aic(rss, n))
    print("  %-6s slope=%8.4f  RSS=%.6f  AIC=%8.3f" % (name, s, rss, aic(rss, n)))
print("  dAIC(lnK - lnlnK)  = %+.3f   (project claims ~8.7)"
      % (res["lnK"][2]-res["lnlnK"][2]))
print("  dAIC(K    - lnlnK) = %+.3f" % (res["K"][2]-res["lnlnK"][2]))
print("  RSS ratio lnK/lnlnK = %.2f  (project claims 5.5x)"
      % (res["lnK"][1]/res["lnlnK"][1]))

print()
print("=" * 78)
print("B. CONVENTION SENSITIVITY: rescale the axis K -> c*K.")
print("   lnK fit is INVARIANT; lnlnK fit is not. Does lnlnK still win?")
print("=" * 78)
_, rss_lnK = fit(np.log(K), a)
print("   (lnK RSS is exactly invariant under rescaling: %.6f)" % rss_lnK)
print("   %-42s %10s %10s %9s" % ("axis convention", "slope", "RSS", "dAIC vs lnK"))
convs = [
    ("K_eff = OmegaT/sqrt3          (as used)", 1.0),
    ("K_eff/2pi = zero up-crossings ", 1/(2*np.pi)),
    ("K_eff/pi  = zero crossings    ", 1/np.pi),
    ("E[N_max] = 0.2135 K_eff (lambda4 count)", 3*np.sqrt(5)/(10*np.pi)),
    ("Rubin discrete axis K/8       ", 1/8),
    ("2 x K_eff                     ", 2.0),
    ("10 x K_eff                    ", 10.0),
]
for name, c in convs:
    x = np.log(np.log(c*K))
    s, rss = fit(x, a)
    d = aic(rss_lnK, n) - aic(rss, n)
    verdict = "lnlnK WINS" if d > 2 else ("inconclusive" if d > -2 else "lnK WINS")
    print("   %-42s %10.4f %10.6f %9.2f  %s" % (name, s, rss, d, verdict))
print()
print("   => The lnlnK-vs-lnK verdict AND the fitted slope both depend on an O(1)")
print("      multiplicative convention in the axis, which no chapter fixes physically.")

print()
print("=" * 78)
print("C. NOISE SENSITIVITY: what per-point s.e. destroys the selection?")
print("=" * 78)
rng = np.random.default_rng(0)
xll, xl = np.log(np.log(K)), np.log(K)
sl, _ = fit(xll, a)
il = a.mean() - sl*xll.mean()
truth = sl*xll + il          # take lnlnK as truth, add noise, see how often it wins
print("   sigma   P(lnlnK selected by AIC, 4000 resamples)   mean dAIC")
for sig in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
    wins, ds = 0, []
    for _ in range(4000):
        y = truth + rng.normal(scale=sig, size=n)
        _, r1 = fit(xll, y); _, r2 = fit(xl, y)
        d = aic(r2, n) - aic(r1, n); ds.append(d)
        if d > 2: wins += 1
    print("   %.3f          %5.1f%%                          %+7.2f"
          % (sig, 100*wins/4000, np.mean(ds)))
print()
resid_sd = np.sqrt(res["lnlnK"][1]/(n-2))
print("   Residual s.d. of the actual fit under lnlnK = %.4f" % resid_sd)
print("   => the selection is decisive only if the true per-point uncertainty on")
print("      alpha_hat_c is <~0.03. A half-crossing read off P_solve(alpha) at")
print("      finite N and finite seeds is rarely that precise.")

print()
print("=" * 78)
print("D. Can a plain lnK law + the theory's OWN O3 correction masquerade as lnlnK?")
print("=" * 78)
# Fit a 'lnK + constant' curve and a 'lnlnK + O3 drift' curve; compare to data.
o3 = np.log(np.log(np.log(K)))/np.log(2)
cands = {
    "lnlnK (1 regressor)": np.log(np.log(K)),
    "lnK   (1 regressor)": np.log(K),
    "lnlnK + O3 drift (theory's own eq)": np.log(np.log(K)) + 2*np.log(2)*o3,
}
for nm, x in cands.items():
    s, rss = fit(x, a)
    print("   %-38s slope=%7.4f  RSS=%.6f" % (nm, s, rss))
print()
print("   Theory's full prediction  a = lnlnK/(2ln2) + lnlnlnK/ln2 - 0.383 :")
pred = np.log(np.log(K))/(2*np.log(2)) + o3 - 0.383
sfit, _ = fit(np.log(np.log(K)), pred)
sdat, _ = fit(np.log(np.log(K)), a)
print("   local slope of the THEORY curve on this grid = %.4f" % sfit)
print("   local slope of the DATA                      = %.4f" % sdat)
print("   ratio data/theory = %.3f   -> the data is ~2x FLATTER than the theory's")
print("   own O3-corrected prediction, not a match to it." % ())
