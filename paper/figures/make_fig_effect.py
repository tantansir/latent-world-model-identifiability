"""fig_effect.pdf: training target versus parameter precision for the mass action effect.
Reads exp/e003_motion/results/effect_heads_VFX_s{0,1,2}_l0.02_{trunk}_s0.json (three trunk seeds).
Left: relative error E of the predicted push effect; right: slope ratio S of its mass dependence;
x = alpha (0: parameter estimate from the frozen state, 1: true parameters)."""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight",
})
BLUE, ORANGE, GRAY, GREEN = "#3B6FB6", "#E1812C", "#8C8C8C", "#3A923A"
R = pathlib.Path(__file__).resolve().parents[2] / "exp" / "e003_motion" / "results"
TRUNKS = [("vr_oi_kr_mr_ac_ih", "-", "o", "coordinate-target trunk"), ("oi_ac_ih", "--", "s", "baseline trunk")]
ALPHAS = ["0", "0.5", "1"]


def load(trunk):
    return [json.load(open(R / f"effect_heads_VFX_s{s}_l0.02_{trunk}_s0.json")) for s in (0, 1, 2)]


fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw={"wspace": 0.32})
x = np.array([0.0, 0.5, 1.0])
for trunk, ls, mk, tlab in TRUNKS:
    D = load(trunk)
    for target, col, lab in (("latent", ORANGE, "future-latent target"), ("effect", BLUE, "action-effect target")):
        for ax, key in zip(axes, ("E_delta", "S_delta")):
            v = np.array([[d[f"{target}_alpha{a}"][key] for a in ALPHAS] for d in D])
            ax.errorbar(x, v.mean(0), yerr=v.std(0, ddof=1), color=col, ls=ls, marker=mk, ms=4.5,
                        lw=1.6, capsize=2, mfc="white" if ls == "--" else col, zorder=3)
D = load("vr_oi_kr_mr_ac_ih")
const = np.mean([d["constant_train_mean"]["E_delta"] for d in D])
ctrl_e = np.mean([d["control_true_theta"]["E_delta"] for d in D])
ctrl_s = np.mean([d["control_true_theta"]["S_delta"] for d in D])
ax = axes[0]
ax.axhline(const, color=GRAY, ls=":", lw=1.1)
ax.text(0.5, const + 0.012, "constant: training-mean effect", color=GRAY, fontsize=7.5, va="bottom", ha="center")
ax.axhline(ctrl_e, color=GREEN, ls="-.", lw=1.1)
ax.text(0.5, ctrl_e - 0.035, "supervised model given state and true $\\theta$", color=GREEN, fontsize=7.5, va="top", ha="center")
ax.set_ylim(0, 0.56)
ax.set_ylabel("error of predicted effect $E^{\\Delta}$")
ax = axes[1]
ax.axhline(1.0, color="k", ls=":", lw=1.0)
ax.text(-0.07, 1.03, "simulator", fontsize=7.5, va="bottom", ha="left")
ax.set_ylim(-0.15, 1.45)
ax.set_ylabel("mass dependence $S^{\\Delta}$")
for ax in axes:
    ax.set_xticks(x, ["0\n(estimate)", "0.5", "1\n(true $\\theta$)"])
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel("parameter precision $\\alpha$", labelpad=1)
# legend: colour = training target, line style = frozen trunk
from matplotlib.lines import Line2D
h = [Line2D([], [], color=ORANGE, lw=1.6, label="future-latent target"),
     Line2D([], [], color=BLUE, lw=1.6, label="action-effect target"),
     Line2D([], [], color="k", ls="-", marker="o", ms=4, lw=1.2, label="coordinate-target trunk"),
     Line2D([], [], color="k", ls="--", marker="s", ms=4, mfc="white", lw=1.2, label="baseline trunk")]
fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
out = pathlib.Path(__file__).parent / "fig_effect.pdf"
fig.savefig(out)
print("wrote", out, "const", round(const, 3), "ctrl", round(ctrl_e, 3), round(ctrl_s, 3))
