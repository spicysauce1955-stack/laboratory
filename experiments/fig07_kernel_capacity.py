"""Figure for paper 07: findable alpha_c vs K_eff per kernel (cross-kernel capacity).

Data are the measured half-crossing alpha_hat_c from the collapse sweep
(experiments/kernel_universality_capacity, 2026-06-25). Pure matplotlib; writes the PNG to
$LAB_RUN_DIR (lab artifact) and to this script's own dir (so pdf/build.sh regenerates it in place).
Run on the lab CPU backend (no local compute):  lab submit -c "python .../07_kernel_capacity.py" --backend cpu --with matplotlib
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# measured findable alpha_hat_c  (K_eff : alpha_c)
series = {
    "gauss (low-pass)":  ([8, 16, 32, 64], [2.68, 2.92, 3.13, 3.33], "o-", "teal"),
    "sinc (band-limited)": ([8, 16, 32, 64], [2.76, 2.96, 3.29, 3.47], "o-", "tab:blue"),
    "alpha (causal)":    ([16, 32, 64], [2.45, 2.80, 3.20], "o-", "tab:brown"),
    "gabor Q4 (band-pass)": ([16, 32, 64], [3.49, 3.64, 3.89], "s-", "tab:red"),
    "gabor Q8 (band-pass)": ([16, 32, 64], [3.33, 3.44, 3.64], "^-", "tab:orange"),
    "sinusoid (radial)": ([16, 32, 64], [3.40, 3.40, 3.43], "D-", "purple"),
}

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.axhline(4.0, ls="--", c="gray", lw=1, label=r"radial bound $\alpha_c=4$")
ax.axhline(2.0, ls=":", c="gray", lw=1, label=r"perceptron $\alpha_c=2$")
for name, (k, a, sty, col) in series.items():
    ax.plot(k, a, sty, color=col, lw=1.8, ms=6, label=name)
ax.set_xscale("log", base=2)
ax.set_xticks([8, 16, 32, 64])
ax.set_xticklabels(["8", "16", "32", "64"])
ax.set_xlabel(r"$K_{\mathrm{eff}}$ (log scale)")
ax.set_ylabel(r"findable $\alpha_c$")
ax.set_ylim(2.0, 4.1)
ax.set_title("Cross-kernel tempotron capacity: band-pass > low-pass at matched $K_{\\mathrm{eff}}$")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8, ncol=2, loc="lower right")
fig.tight_layout()

here = os.path.dirname(os.path.abspath(__file__))
for d in {here, os.environ.get("LAB_RUN_DIR", here)}:
    fig.savefig(os.path.join(d, "07_kernel_capacity.png"), dpi=150)
print("wrote 07_kernel_capacity.png")
