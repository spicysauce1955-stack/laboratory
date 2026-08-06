"""
TH audit part 2 — independent re-derivation of the Ch7 rigorous bracket numbers,
and a DIRECT numerical test of Ch6 Definition 'EVT-effective readout bundle'
(that max_{[0,T]} V behaves like the max of K_eff independent standard Gaussians).

Run: uv run --with numpy --with scipy python th02_bracket_and_bundle.py
Auditor-written; imports no project code.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad

Ks = [1, 8, 31, 100, 10**6]

print("=" * 78)
print("A. Ch7 Thm cellcount: alpha* solving  a ln2 = 1 + ln a + ln K")
print("=" * 78)
print("  claimed: 3.05, 7.31, 9.67, 11.63, 26.1")
got = []
for K in Ks:
    f = lambda a: a*np.log(2) - 1 - np.log(a) - np.log(K)
    a = brentq(f, 1.0001, 1e4)
    got.append(a)
    print("    K=%8d  alpha* = %8.4f" % (K, a))
print("  asymptotic (lnK+lnlnK+O(1))/ln2 ; coefficient 1/ln2 = %.4f" % (1/np.log(2)))
for K in (10**6, 10**12):
    f = lambda a: a*np.log(2) - 1 - np.log(a) - np.log(K)
    a = brentq(f, 1.0001, 1e5)
    print("    K=1e%-3d  alpha*/lnK = %.4f  (-> 1/ln2 = 1.4427)" % (int(np.log10(K)), a/np.log(K)))

print()
print("=" * 78)
print("B. Ch7 Thm gordon (rho = I):  1/Dbar,  Dbar = .5 E[(th-M)_+^2] + .5 K E[(g-th)_+^2]")
print("=" * 78)
print("  claimed: 2.00, 5.07, 7.67, 10.09, 29.9")
def theta_K(K):
    return norm.ppf(2.0**(-1.0/K))

def E_theta_minus_M_sq(K, th):
    # M = max of K iid N(0,1); density K phi(m) Phi(m)^{K-1}
    f = lambda m: (th-m)**2 * K*norm.pdf(m)*norm.cdf(m)**(K-1)
    v, _ = quad(f, -40, th, limit=400)
    return v

def E_g_minus_theta_plus_sq(th):
    # E[(g-th)_+^2] for g~N(0,1) = (1+th^2)*Phibar(th) - th*phi(th)
    return (1+th**2)*norm.sf(th) - th*norm.pdf(th)

for K in Ks:
    th = theta_K(K)
    Dbar = 0.5*E_theta_minus_M_sq(K, th) + 0.5*K*E_g_minus_theta_plus_sq(th)
    print("    K=%8d  theta_K=%7.4f  Dbar=%.6f  1/Dbar = %8.4f" % (K, th, Dbar, 1/Dbar))

print()
print("  asymptotic claim: alpha_c+ <= (1+o(1)) * 2.22 ln K")
for K in (10**4, 10**6, 10**9, 10**12):
    th = theta_K(K)
    Dbar = 0.5*E_theta_minus_M_sq(K, th) + 0.5*K*E_g_minus_theta_plus_sq(th)
    print("    K=1e%-3d  (1/Dbar)/lnK = %.4f" % (int(np.log10(K)), (1/Dbar)/np.log(K)))
c0 = None
# claimed c0 = E[(m_G - G)_+^2], G standard Gumbel, m_G = -ln ln 2 its median
mG = -np.log(np.log(2))
f = lambda x: (mG-x)**2 * np.exp(-x)*np.exp(-np.exp(-x))
c0, _ = quad(f, -5, mG, limit=400)
print("  claimed c0 = E[(m_G-G)_+^2] ~ 0.412 ; recomputed = %.5f" % c0)
print("  claimed asymptotic const 1/(c0/4 + ln2/2) = %.4f  (claimed 2.22)"
      % (1/(c0/4 + np.log(2)/2)))

print()
print("=" * 78)
print("C. Ch7 Prop convex (rho=I): alpha_cvx = [ .5((1+th^2)Phi(th)+th phi(th))")
print("                                          + .5 K((1+th^2)Phibar(th)-th phi(th)) ]^-1")
print("=" * 78)
print("  claimed: 2.000, 0.638, 0.384, 0.278, 0.082")
for K in Ks:
    th = theta_K(K)
    t1 = (1+th**2)*norm.cdf(th) + th*norm.pdf(th)          # E[(th-g)_+^2]
    t2 = (1+th**2)*norm.sf(th) - th*norm.pdf(th)           # E[(g-th)_+^2]
    a = 1.0/(0.5*t1 + 0.5*K*t2)
    print("    K=%8d  alpha_cvx = %8.4f     (times lnK = %.4f)"
          % (K, a, a*np.log(K) if K > 1 else float('nan')))

print()
print("=" * 78)
print("D. Ch7 Thm pinning (rho=I, lambda_min=1): claimed 1.0, 0.022, 0.0036, 0.00083")
print("=" * 78)
for K in Ks:
    th = theta_K(K)
    # x theta^2 < (1 - sqrt x)^2 , x = alpha K
    g = lambda x: x*th**2 - (1-np.sqrt(x))**2
    xs = brentq(g, 1e-14, 1-1e-12) if th > 0 else 1.0
    print("    K=%8d  theta=%7.4f  x*=%.6f  alpha_min = x*/K = %.6g   (asympt 1/(2K lnK) = %.3g)"
          % (K, th, xs, xs/K, 1/(2*K*np.log(K)) if K > 1 else float('nan')))

print()
print("=" * 78)
print("E. Lemma 'the union is essential': Var<c,h> = c^T rho c <= (sum c)^2 = 1 ?")
print("=" * 78)
rng = np.random.default_rng(0)
worst = -1
for _ in range(20000):
    K = rng.integers(2, 9)
    A = rng.normal(size=(K, K)); R = A @ A.T
    d = np.sqrt(np.diag(R)); R = R/np.outer(d, d)      # a valid correlation matrix
    c = rng.dirichlet(np.ones(K))
    worst = max(worst, float(c @ R @ c))
print("  max over 20000 random (rho, c in simplex) of c^T rho c = %.10f  (must be <= 1)" % worst)
print("  NOTE: the bound uses rho_lm <= 1, true for any correlation matrix. VERIFIED.")

print()
print("=" * 78)
print("F. DIRECT TEST of Ch6 Definition 'EVT-effective readout bundle':")
print("   is max_{[0,T]} V(t) distributed like max of K_eff iid N(0,1)?")
print("=" * 78)
# Build the exact flat-band process of Ch4/Ch5: V(t) = sum_k [A_k cos(w_k t) + B_k sin(w_k t)]
# with w_k an even comb on (0, Omega], A_k,B_k iid N(0, sigma^2) chosen so Var V = 1.
rng = np.random.default_rng(1)
T = 1.0
for Keff_target in (8.0, 30.8, 121.7):
    Om = Keff_target*np.sqrt(3)/T
    m = 400                                     # dense comb -> flat band
    wk = (np.arange(1, m+1) - 0.5)*Om/m         # midpoint rule on (0, Omega]
    sig2 = 1.0/m                                # Var V = sum_k sig2 = 1
    ngrid = 40000
    t = np.linspace(0, T, ngrid)
    C = np.cos(np.outer(t, wk)); S = np.sin(np.outer(t, wk))
    nrep = 4000
    A = rng.normal(scale=np.sqrt(sig2), size=(nrep, m))
    B = rng.normal(scale=np.sqrt(sig2), size=(nrep, m))
    Vmax = (A @ C.T + B @ S.T).max(axis=1)
    # empirical Var of V (sanity)
    Vsamp = (A[:50] @ C.T + B[:50] @ S.T)
    # reference: max of K iid standard normals
    Keff = Om*T/np.sqrt(3)
    ref = rng.normal(size=(nrep, max(1, int(round(Keff))))).max(axis=1)
    # what K would REPRODUCE the observed median of the max?
    med = np.median(Vmax)
    Kfit = np.log(0.5)/np.log(norm.cdf(med))    # solve Phi(med)^K = 1/2
    print("  K_eff = %7.2f  (Omega=%.3f, T=%.1f)   Var[V]=%.4f" % (Keff, Om, T, Vsamp.var()))
    print("    median max_[0,T] V      = %.4f" % med)
    print("    median max of K_eff iid = %.4f   (theta_K = %.4f)"
          % (np.median(ref), theta_K(int(round(Keff)))))
    print("    -> EFFECTIVE iid trial count that matches the observed median: K_fit = %.2f"
          % Kfit)
    print("       K_fit / K_eff = %.3f ;  K_fit / (K_eff/2pi) = %.3f ;  "
          "K_fit / E[N_max]=0.2135*K_eff = %.3f"
          % (Kfit/Keff, Kfit/(Keff/(2*np.pi)), Kfit/(0.2135*Keff)))
    print("    lnln(K_eff)=%.4f   lnln(K_fit)=%.4f   difference=%+.4f"
          % (np.log(np.log(Keff)), np.log(np.log(Kfit)),
             np.log(np.log(Kfit))-np.log(np.log(Keff))))
    print()

print("=" * 78)
print("G. Nearest-neighbour correlation of the Ch6 'quasi-independent' slices")
print("=" * 78)
# flat band: r(tau)/lambda0 = sin(Omega tau)/(Omega tau); slice length tau_c = sqrt(l0/l2)=sqrt3/Omega
tc_scaled = np.sqrt(3)          # Omega * tau_c
print("  Ch6 chops [0,T] into K_eff slices of length tau_c = sqrt(l0/l2).")
print("  For a flat band r(tau)/l0 = sin(Om tau)/(Om tau); at tau = tau_c, Om*tau_c = sqrt(3):")
print("    r(tau_c)/lambda_0 = sin(sqrt3)/sqrt3 = %.4f" % (np.sin(tc_scaled)/tc_scaled))
print("  => ADJACENT 'quasi-independent' readouts are ~57%% correlated BY CONSTRUCTION.")
print("  small-lag expansion 1 - (l2/2l0)tau_c^2 = 1 - 1/2 = 0.5 (consistent)")
