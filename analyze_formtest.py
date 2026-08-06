"""Aggregate the form-discrimination sweep from raw shard CSVs, locate alpha_hat_c per
K_eff, run validity gates + model selection {ln ln K, ln K, K}. No lab helper (it choked
on a duplicate shard); we dedupe by (K,seed,alpha) ourselves."""
import csv, glob, os, json, math
from collections import defaultdict

SW = "sweep-20260728-111736-77fcae"
rows_by_cell = defaultdict(dict)   # (Ktarget) -> {(seed,alpha): row}
for m in glob.glob("runs/*/manifest.json"):
    try:
        d = json.load(open(m))
    except Exception:
        continue
    if d.get("sweep_id") != SW and SW not in json.dumps(d.get("config", {})):
        continue
    jd = os.path.dirname(m)
    rc = os.path.join(jd, "output", "results.csv")
    if not os.path.exists(rc):
        rc = os.path.join(jd, "results.csv")
    if not os.path.exists(rc):
        continue
    for r in csv.DictReader(open(rc)):
        try:
            K = float(r["Keff_target"]); seed = int(r["seed"]); a = float(r["alpha"])
        except (ValueError, KeyError):
            continue
        rows_by_cell[K][(seed, a)] = r   # dedupe: last write wins (idempotent per (seed,alpha))

def half_crossing(alphas, psolve):
    xs = sorted(alphas)
    for i in range(len(xs) - 1):
        a0, a1 = xs[i], xs[i + 1]; p0, p1 = psolve[a0], psolve[a1]
        if (p0 - 0.5) * (p1 - 0.5) <= 0 and p0 != p1:
            return a0 + (0.5 - p0) * (a1 - a0) / (p1 - p0)
    return float("nan")

print("=== per-cell aggregation + validity gates ===")
cells = {}
for K in sorted(rows_by_cell):
    rows = rows_by_cell[K]
    seeds = sorted(set(s for s, a in rows))
    alphas = sorted(set(a for s, a in rows))
    psolve, ndat, meas_K, fire0, eps_solved = {}, {}, [], [], {}
    for a in alphas:
        sv = [int(rows[(s, a)]["solved"]) for s in seeds if (s, a) in rows]
        psolve[a] = sum(sv) / len(sv) if sv else float("nan")
        ndat[a] = len(sv)
        for s in seeds:
            if (s, a) in rows:
                r = rows[(s, a)]
                meas_K.append(float(r["Keff_measured"])); fire0.append(float(r["init_fire_rate"]))
                e = int(r["epochs_to_solve"])
                if e > 0: eps_solved.setdefault(a, []).append(e)
    ac = half_crossing(alphas, psolve)
    mK = sum(meas_K) / len(meas_K) if meas_K else float("nan")
    f0 = sum(fire0) / len(fire0) if fire0 else float("nan")
    # critical slowing: median epochs_to_solve at low vs near-crossing alpha (solved cells only)
    cs = {a: (sorted(v)[len(v)//2]) for a, v in sorted(eps_solved.items())}
    cells[K] = dict(ac=ac, mK=mK, f0=f0, nseed=len(seeds), nalpha=len(alphas),
                    minfill=min(ndat.values()), maxfill=max(ndat.values()),
                    psolve=psolve, cs=cs)
    print(f"K_target={K:6.1f} | measK={mK:6.2f} ({100*(mK-K)/K:+.1f}%) | fire0={f0:.3f} | "
          f"seeds={len(seeds):2d} alphas={len(alphas):2d} fill={min(ndat.values())}-{max(ndat.values())}/24 | "
          f"alpha_hat_c={ac:.3f}")
    print("     P_solve:", {a: round(psolve[a], 2) for a in alphas})
    print("     crit-slow (median epochs_to_solve):", {a: cs[a] for a in list(cs)[:12]})

print("\n=== model selection on alpha_hat_c(K_eff) ===")
Ks = sorted(k for k in cells if not math.isnan(cells[k]["ac"]))
mKs = [cells[k]["mK"] for k in Ks]          # use MEASURED K_eff (certified axis)
ac = [cells[k]["ac"] for k in Ks]
print("measured K_eff:", [round(x, 2) for x in mKs])
print("alpha_hat_c   :", [round(x, 3) for x in ac])

def wls_fit(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys); sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
    den = n*sxx - sx*sx
    if den == 0: return float("nan"), float("nan"), float("inf")
    b = (n*sxy - sx*sy)/den; a0 = (sy - b*sx)/n
    sse = sum((y - (a0 + b*x))**2 for x, y in zip(xs, ys))
    return a0, b, sse

for name, xf in [("ln ln K", lambda k: math.log(math.log(k))),
                 ("ln K", lambda k: math.log(k)),
                 ("K (linear)", lambda k: k)]:
    xs = [xf(k) for k in mKs]
    a0, b, sse = wls_fit(xs, ac)
    n, p = len(xs), 2
    aic = n*math.log(sse/n) + 2*p if sse > 0 else float("-inf")
    # leave-one-out RMSE
    loo = []
    for i in range(n):
        xs2 = xs[:i]+xs[i+1:]; ys2 = ac[:i]+ac[i+1:]
        a2, b2, _ = wls_fit(xs2, ys2)
        loo.append((ac[i] - (a2 + b2*xs[i]))**2)
    loormse = math.sqrt(sum(loo)/len(loo))
    print(f"  {name:12s}: slope={b:+.3f} intercept={a0:+.3f} SSE={sse:.4f} AIC={aic:+.2f} LOO-RMSE={loormse:.4f}")
print("\n(theory: existence slope d alpha/d(ln ln K)=1/(2 ln2)=0.7213; findable expected below; "
      "prior findable ~0.45, existence probe ~0.86)")
