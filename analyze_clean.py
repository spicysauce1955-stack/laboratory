"""Clean analysis: ONLY my three job sets (sweep1 + sweep2 + resubmits), all N=1000,
validated Adam-cosine learner, median threshold, fixed density. Dedupe by (K,seed,alpha)
preferring higher-epoch (sweep2) rows. Gate on P_solve monotonicity + seed count."""
import json, glob, os, csv, math
from collections import defaultdict

SW1 = "sweep-20260728-111736-77fcae"   # epochs 8000, alpha 2.3-3.2
SW2 = "sweep-20260728-122723-43fd6f"   # epochs 12000, alpha 2.8-4.0
RESUB = set(open("resubmit_jobs.txt").read().split()) if os.path.exists("resubmit_jobs.txt") else set()

# (K) -> (seed,alpha) -> (epochs_cfg, row)  keep the higher-epoch row on collision
rows = defaultdict(dict)
for m in glob.glob("runs/*/manifest.json"):
    try: d = json.load(open(m))
    except Exception: continue
    jid = os.path.basename(os.path.dirname(m))
    swid = d.get("sweep_id")
    if not (swid == SW1 or swid == SW2 or jid in RESUB):
        continue
    ep = 12000 if (swid == SW2 or jid in RESUB) else 8000
    rc = os.path.join(os.path.dirname(m), "output", "results.csv")
    if not os.path.exists(rc): rc = os.path.join(os.path.dirname(m), "results.csv")
    if not os.path.exists(rc): continue
    for r in csv.DictReader(open(rc)):
        try: K = float(r["Keff_target"]); s = int(r["seed"]); a = float(r["alpha"])
        except (ValueError, KeyError): continue
        key = (s, a)
        if key not in rows[K] or ep > rows[K][key][0]:
            rows[K][key] = (ep, r)

def crossing(al, ps):
    xs = sorted(al)
    for i in range(len(xs) - 1):
        a0, a1 = xs[i], xs[i + 1]
        if (ps[a0] - .5) * (ps[a1] - .5) <= 0 and ps[a0] != ps[a1]:
            return a0 + (.5 - ps[a0]) * (a1 - a0) / (ps[a1] - ps[a0])
    return float("nan")

print("=== clean per-K (my runs only), full P_solve ===")
res = []
for K in sorted(rows):
    rws = {k: v[1] for k, v in rows[K].items()}
    seeds = sorted(set(s for s, a in rws)); als = sorted(set(a for s, a in rws))
    ps, nfill = {}, {}
    for a in als:
        sv = [int(rws[(s, a)]["solved"]) for s in seeds if (s, a) in rws]
        ps[a] = sum(sv) / len(sv) if sv else float("nan"); nfill[a] = len(sv)
    mK = sum(float(rws[k]["Keff_measured"]) for k in rws) / len(rws)
    f0 = sum(float(rws[k]["init_fire_rate"]) for k in rws) / len(rws)
    # monotonicity gate: is P_solve roughly non-increasing? count inversions on the high side
    hi = [a for a in als if a >= 2.6]
    inv = sum(1 for i in range(len(hi)-1) if ps[hi[i+1]] > ps[hi[i]] + 0.25)
    ac = crossing(als, ps)
    res.append((K, mK, ac, len(seeds), inv))
    flag = "OK" if inv <= 1 else f"NON-MONO(inv={inv})"
    print(f"K={K:6.0f} measK={mK:6.1f} fire0={f0:.3f} seeds={len(seeds):2d} cross={ac:6.3f} [{flag}]")
    print("   P:", " ".join(f"{a}:{ps[a]:.2f}({nfill[a]})" for a in als if a >= 2.5))

print("\n=== model selection on clean crossings (monotone cells only) ===")
good = [(mK, ac) for K, mK, ac, ns, inv in res if not math.isnan(ac) and inv <= 1 and ns >= 12]
print("used (measK, crossing):", [(round(a, 1), round(b, 3)) for a, b in good])
if len(good) >= 3:
    ys = [b for _, b in good]
    for nm, xf in [("ln ln K", lambda k: math.log(math.log(k))), ("ln K", math.log), ("K (lin)", lambda k: k)]:
        xs = [xf(a) for a, _ in good]; n = len(xs)
        sx = sum(xs); sy = sum(ys); sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
        b = (n*sxy - sx*sy) / (n*sxx - sx*sx); a0 = (sy - b*sx)/n
        sse = sum((y-(a0+b*x))**2 for x, y in zip(xs, ys))
        aic = n*math.log(sse/n) + 4 if sse > 0 else float("-inf")
        print(f"  {nm:8s}: slope={b:+.3f} intercept={a0:+.3f} SSE={sse:.4f} AIC={aic:+.2f}")
else:
    print("  <3 clean monotone crossings — insufficient for model selection")
