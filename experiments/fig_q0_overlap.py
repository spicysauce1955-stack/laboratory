"""Figure: restart-overlap witness q0 vs alpha, per K_eff (the solution-cluster observable).

Companion to `fig07_kernel_capacity.py`, which plots the *findable* alpha_c. This one plots q0 --
the mean pairwise normalized overlap of independently-solved weight vectors -- against alpha for
each K_eff cell, so the solution cluster's collapse can be read against the capacity crossing.

Reads the aggregated per-cell CSV produced by `lab sweep-aggregate` (column `restart_overlap`,
which the run header prints as `q0`). Pure matplotlib; writes the PNG to $LAB_RUN_DIR.

Run on the lab CPU backend:
    lab submit -c "python experiments/fig_q0_overlap.py results_csv=<path>" \
        --backend cpu --cloud gcp --with matplotlib

Unlike v14 this uses `lab.experiment.get_overrides`, so it participates in the config-consumption
handshake and writes `effective_config.json`.
"""
import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lab.experiment import get_overrides  # noqa: E402

KNOWN = ("results_csv", "kernel", "out_name")
ov = get_overrides(KNOWN)

run_dir = os.environ.get("LAB_RUN_DIR", os.path.dirname(os.path.abspath(__file__)))
results_csv = ov.get("results_csv", "experiments/kernel_universality_capacity/results/q0.csv")
want_kernel = ov.get("kernel", "gauss")
out_name = ov.get("out_name", "q0_restart_overlap.png")

# Published findable alpha_c for gauss (fig07), so q0 can be read against the crossing.
ALPHA_C = {8: 2.68, 16: 2.92, 32: 3.13, 64: 3.33}

with open(results_csv) as fh:
    rows = [r for r in csv.DictReader(fh) if r.get("kernel", want_kernel) == want_kernel]

# (K_eff, alpha) -> list of q0 over seeds. Unsolved cells carry an empty/NaN overlap by design
# (q0 is undefined with <2 solved restarts) and must be dropped, not read as zero.
buckets: dict[tuple[float, float], list[float]] = defaultdict(list)
for r in rows:
    raw = (r.get("restart_overlap") or "").strip()
    if not raw:
        continue
    try:
        q0 = float(raw)
    except ValueError:
        continue
    if math.isnan(q0):
        continue
    buckets[(float(r["Keff_target"]), float(r["alpha"]))].append(q0)

if not buckets:
    raise SystemExit(f"no usable restart_overlap rows in {results_csv} for kernel={want_kernel}")

fig, ax = plt.subplots(figsize=(7.2, 4.4))
colors = {8: "teal", 16: "tab:blue", 32: "tab:orange", 64: "tab:red"}
for keff in sorted({k for k, _ in buckets}):
    pts = sorted((a, vals) for (k, a), vals in buckets.items() if k == keff)
    alphas = [a for a, _ in pts]
    means = [sum(v) / len(v) for _, v in pts]
    ns = [len(v) for _, v in pts]
    col = colors.get(int(keff), None)
    ax.plot(alphas, means, "o-", color=col, lw=1.8, ms=5,
            label=rf"$K_{{\mathrm{{eff}}}}={int(keff)}$  (n={min(ns)}-{max(ns)} seeds)")
    if int(keff) in ALPHA_C:
        ax.axvline(ALPHA_C[int(keff)], color=col, ls=":", lw=1, alpha=0.7)

ax.set_xlabel(r"$\alpha = P/N$")
ax.set_ylabel(r"restart overlap $q_0$")
ax.set_title(
    f"Solution-cluster overlap vs load ({want_kernel})\n"
    r"dotted lines: published findable $\alpha_c$ per $K_{\mathrm{eff}}$"
)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()

out = os.path.join(run_dir, out_name)
fig.savefig(out, dpi=150)
print(f"wrote {out} from {len(rows)} rows, {len(buckets)} (K_eff, alpha) cells")
