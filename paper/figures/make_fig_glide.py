"""fig_glide.pdf: drag is recoverable, weakly read on contact windows, readable on glides, and used by glide forecasts.
(a) drag R^2 on contact windows (probe set) and on pure glides (evaluation set, seed 778): recovery reference (GRU on
    the VFX inputs: image-derived object trajectory, proprioception, touch, actions) versus ridge readout of the frozen
    VFX predictor state; three seeds for the readouts.
(b) implied 16-step displacement factor f_hat = (p_hat_31 - p_15).v / |v|^2 of the VFX forecasts per drag bin (mean over
    three seeds, error bars s.d. over seeds) against the true factor and the median-drag factor.
Sources: results/cert_matched.json, results/glide_recovery.json, results/VFX_s{s}_l0.02_ac.json,
and results/glide_forecasts.json (exported from analyze_gamma.py/probe_targets.py output).
Prints the numbers used in the text."""
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight",
})
BLUE, ORANGE, GRAY, GREEN = "#3B6FB6", "#E1812C", "#8C8C8C", "#3A923A"
RES = pathlib.Path(__file__).resolve().parents[2] / "exp" / "e003_motion" / "results"
records = json.loads((RES / "glide_forecasts.json").read_text(encoding="utf-8"))["models"]


def forecast_record(row):
    return (np.array(row["gamma_bins"]), np.array(row["forecast_factors"]),
            np.array(row["true_factors"]), row["bin_mean_slope"],
            row["partial_slope"], row["partial_slope_se"])


base = [(forecast_record(r), r["glide_readout_r2"]) for r in records if "vr_oi" not in r["tag"]]
coord = [(forecast_record(r), r["glide_readout_r2"]) for r in records if "vr_oi" in r["tag"]]
cm = json.load(open(RES / "cert_matched.json"))
rec_contact = cm["contact"]["V+P+T+A"]["gru"][1]
gr = json.load(open(RES / "glide_recovery.json"))
rec_glide = gr["GRU, VFX inputs (image-derived)"]
read_contact = [json.load(open(RES / f"VFX_s{s}_l0.02_ac.json"))["probe"]["trunk"]["contact"]["gamma"] for s in (0, 1, 2)]
read_glide = [r for _, r in base]
print("recovery contact / glide:", rec_contact, round(rec_glide, 3), "| structured glide (image, state):",
      round(gr["structured, image-derived speed"], 3), round(gr["structured, simulator speed"], 3),
      "| GRU glide state:", round(gr["GRU, simulator object state"], 3))
print("VFX readout contact:", np.round(read_contact, 3), "glide:", np.round(read_glide, 3))
print("VFX forecast bin-mean slopes:", [c[0][3] for c in base], "partial:", [(c[0][4], c[0][5]) for c in base])
print("coordinate model: glide readout", [r for _, r in coord], "slopes", [c[0][3] for c in coord], "partial", [(c[0][4], c[0][5]) for c in coord])

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw={"wspace": 0.35, "width_ratios": [0.85, 1.15]})
ax = axes[0]
xs = np.array([0, 1]); w = 0.26
coord_contact = [json.load(open(RES / f"VFX_s{s}_l0.02_vr_oi_kr_mr_ac_ih.json"))["probe"]["trunk"]["contact"]["gamma"] for s in (0, 1, 2)]
coord_glide = [r for _, r in coord]
bars = [(xs - w, [rec_contact, rec_glide], None, GRAY, "GRU on the VFX inputs"),
        (xs, [np.mean(read_contact), np.mean(read_glide)], (read_contact, read_glide), BLUE, "VFX latent"),
        (xs + w, [np.mean(coord_contact), np.mean(coord_glide)], (coord_contact, coord_glide), ORANGE, "coordinate-target latent")]
for x, v, pts, col, lab in bars:
    ax.bar(x, v, w, color=col, label=lab)
    for xi, vi in zip(x, v):
        ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", va="bottom", fontsize=7.2)
    if pts is not None:
        for xi, p in zip(x, pts):
            ax.scatter([xi] * len(p), p, s=6, color="k", zorder=3)
print("coordinate model contact readout", np.round(coord_contact, 3))
ax.set_xticks(xs, ["contact windows", "glide windows"])
ax.set_ylabel("drag $R^2$")
ax.set_ylim(0, 1.1)
ax.set_title("(a) drag recovery and readout", fontsize=9.5)

ax = axes[1]
g = base[0][0][0]
gg = np.linspace(0.5, 4.0, 200)
ax.plot(gg, (1 - np.exp(-0.8 * gg)) / gg, color="k", lw=1.4, label="true dynamics")
ax.axhline(0.367, color=GRAY, ls=":", lw=1.1, label="median-drag prediction")
F = np.array([c[0][1] for c in base])
ax.errorbar(g, F.mean(0), yerr=F.std(0, ddof=1), color=BLUE, marker="o", ms=4, lw=1.5, capsize=2)
FC = np.array([c[0][1] for c in coord])
ax.errorbar(g, FC.mean(0), yerr=FC.std(0, ddof=1), color=ORANGE, marker="s", ms=4, lw=1.5, capsize=2)
ax.set_xlabel("drag $\\gamma$")
ax.set_ylabel("16-step displacement / speed")
ax.set_xlim(0.4, 4.1)
ax.set_ylim(-0.05, 1.0)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(color=GRAY, label="GRU on the VFX inputs"), Patch(color=BLUE, label="VFX"),
           Patch(color=ORANGE, label="VFX + coordinate targets"),
           Line2D([], [], color="k", lw=1.4, label="true dynamics"), Line2D([], [], color=GRAY, ls=":", lw=1.1, label="median-drag prediction")]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, 1.09),
           handlelength=1.6, columnspacing=1.1)
ax.set_title("(b) forecasts of pure glides", fontsize=9.5)
out = pathlib.Path(__file__).parent / "fig_glide.pdf"
fig.savefig(out)
print("wrote", out)
