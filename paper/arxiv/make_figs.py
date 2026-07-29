"""Publication figures — overlap-free revision."""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "font.size": 11, "axes.titlesize": 11,
    "axes.labelsize": 10.5, "legend.fontsize": 9, "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
BLUE, ORANGE, GRAY, GREEN, PURPLE = "#3B6FB6", "#E1812C", "#8C8C8C", "#3A923A", "#7B4FA6"
LORANGE, LGRAY = "#E19C56", "#9E9E9E"   # VXt / VXp (variant->color is global)

# ------- within-embodiment scaling curves (Flexiv cfg1, fixed compute) -------
# Data-driven: reads the e010 result JSONs so the four-arm factorial
# (V / VX / VF / VFX x nep {800, 2000, 4258}) always matches the source files.
E010R = pathlib.Path(r"C:\Users\Kaizh\Desktop\physical representation"
                     r"\exp\e010_rh20t\results")


def _load(tag):
    p = E010R / f"{tag}.json"
    return json.load(open(p)) if p.exists() else None


def _series(variant, fn):
    vals = []
    for suf in ("_n800", "_n2000", ""):
        d = _load(f"{variant}_s0_c1{suf}")
        vals.append(fn(d) if d else np.nan)
    return vals


ARMS = [("V", GRAY, "o", "V (vision in, latent tgt)"),
        ("VX", ORANGE, "^", "VX (+cross-modal targets)"),
        ("VF", PURPLE, "D", "VF (fused in, latent tgt)"),
        ("VFX", BLUE, "s", "VFX (fused + targets)")]
rand = _load("RAND_c1")

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw={"wspace": 0.38})
x = [800, 2000, 4258]

ax = axes[0]
for v, c, mk, lab in ARMS:
    ax.plot(x, _series(v, lambda d: d["iid"]["pos_readout_r2"]), mk + "-",
            color=c, lw=1.8, ms=6, label=lab)
    ax.plot(x, _series(v, lambda d: d["ood"]["pos_readout_r2"]), mk + "--",
            color=c, lw=1.0, ms=4, alpha=0.55)
d30 = _load("V_s0_c1_st30k")
if d30:
    ax.scatter([4258], [d30["iid"]["pos_readout_r2"]], marker="x", color=GRAY,
               s=34, zorder=5)
ax.set_title("position readout $R^2$", fontsize=9.5)
ax.set_ylim(-0.12, 1.09)
ax.set_xscale("log")
ax.set_xticks(x, ["800", "2000", "4258"])
ax.minorticks_off()
ax.set_xlabel("training episodes")

ax = axes[1]
ffp = E010R / "futforce.json"
ff = json.load(open(ffp)) if ffp.exists() else None
if ff:
    def _ffs(variant, key="d4_model"):
        return [ff.get(f"{variant}_s0_c1{suf}", {}).get("ood", {})
                .get(key, np.nan) for suf in ("_n800", "_n2000", "")]
    for v, c in (("VX", ORANGE), ("VFX", BLUE)):
        mid = _ffs(v)
        ci = [ff.get(f"{v}_s0_c1{suf}", {}).get("ood", {})
              .get("d4_model_ci95", [np.nan, np.nan])
              for suf in ("_n800", "_n2000", "")]
        lo, hi = [c0 for c0, _ in ci], [c1 for _, c1 in ci]
        ax.fill_between(x, lo, hi, color=c, alpha=0.16, lw=0)
        mk = "^" if v == "VX" else "s"
        ax.plot(x, mid, mk + "-", color=c, lw=1.8, ms=6)
    pers = ff["VFX_s0_c1"]["ood"]["d4_persist"]
    ax.axhline(pers, color="k", lw=1, ls="--")
    ax.text(4100, pers - 0.03, "persistence baseline", fontsize=8,
            ha="right", va="top")
ax.set_title("future-force $R^2$ (held-out)", fontsize=9.5)
ax.set_ylim(-0.2, 0.62)
ax.axhline(0, color="k", lw=0.6)
ax.set_xscale("log")
ax.set_xticks(x, ["800", "2000", "4258"])
ax.minorticks_off()
ax.set_xlabel("training episodes")

ax = axes[2]
for v, c, mk, lab in ARMS:
    ax.plot(x, _series(v, lambda d: d["iid"]["direct16_err"] * 100), mk + "-",
            color=c, lw=1.8, ms=6, label=lab)
ax.axhline(3.3, color="k", lw=1, ls="--", label="static")
ax.set_title("16-step error (cm, log)", fontsize=9.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(x, ["800", "2000", "4258"])
ax.set_yticks([2, 3, 5, 10], ["2", "3", "5", "10"])
ax.minorticks_off()
ax.set_ylim(1.8, 15)
ax.set_xlabel("training episodes")
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8.5,
           handlelength=1.6, columnspacing=1.2)
axes[0].text(920, -0.10, "dashed: held-out tasks", fontsize=7.5,
             color="0.35")
fig.savefig("fig_scissors.pdf", bbox_inches="tight")
plt.close(fig)

# ---- appendix: current-force readout vs untrained bound (single panel) ----
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for v, c, mk, lab in ARMS:
    ax.plot(x, _series(v, lambda d: d["ood"]["touch_readout_r2_force"]),
            mk + "-", color=c, lw=1.7, ms=3.5, label=lab)
if rand:
    b = rand["VFX"]["ood"]["touch_readout_r2_force"]
    ax.axhline(b, color="k", lw=0.9, ls=":", alpha=0.8)
    ax.text(830, b + 0.05, "untrained encoder (passthrough bound)",
            fontsize=7.5, color="k")
ax.set_ylim(-0.42, 1.05)
ax.axhline(0, color="k", lw=0.6)
ax.set_xscale("log")
ax.set_xticks(x, ["800", "2000", "4258"])
ax.minorticks_off()
ax.legend(frameon=False, loc="center left", handlelength=1.3, fontsize=8)
ax.set_xlabel("training episodes")
fig.savefig("fig_force_readout.pdf")
plt.close(fig)

# ---------------- lambda ----------------
lam = [0.3, 0.1, 0.02, 0.01, 0.005]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.3, 1.95),
                               gridspec_kw={"wspace": 0.45})
ax1.semilogx(lam, [0.12, 0.24, 0.50, 0.54, 0.57], "o-", color=ORANGE, lw=1.7)
ax1.semilogx(lam, [0.12, 0.18, 0.29, 0.32, 0.32], "s-", color=BLUE, lw=1.7)
ax1.semilogx(lam, [0.54, 0.62, 0.79, 0.79, 0.80], "^-", color=GREEN, lw=1.7)
# direct labels at the left curve ends: nothing sits on the data
ax1.text(0.0052, 0.86, "velocity", color=GREEN, fontsize=9)
ax1.text(0.0052, 0.635, "stiffness $\\log k$", color=ORANGE, fontsize=9)
ax1.text(0.0052, 0.20, "mass $\\log m$", color=BLUE, fontsize=9)
ax1.set_xlabel("SIGReg weight $\\lambda$")
ax1.set_ylabel("probe $R^2$")
ax1.set_ylim(0.05, 1.05)
ax1.annotate("lighter regularization", xy=(0.006, 0.97), xytext=(0.13, 0.97),
             fontsize=8, va="center", ha="right", color="0.25",
             arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25"))
ax2.semilogx(lam, [0.128, 0.111, 0.099, 0.096, 0.095], "d-", color=PURPLE,
             lw=1.7)
ax2.axhline(0.205, color="k", ls="--", lw=1)
ax2.text(0.0052, 0.209, "static baseline", fontsize=9, va="bottom")
ax2.text(0.0052, 0.107, "direct-16 error", color=PURPLE, fontsize=9,
         va="bottom")
ax2.set_xlabel("SIGReg weight $\\lambda$")
ax2.set_ylabel("16-step error")
ax2.set_ylim(0.08, 0.225)
fig.savefig("fig_lambda.pdf")
plt.close(fig)

# ---------------- targets ----------------
variants = ["V", "VF", "VX$_p$", "VX$_t$", "VX", "VFX"]
VCOLORS = [GRAY, PURPLE, LGRAY, LORANGE, ORANGE, BLUE]
k_seeds = [(-0.02, -0.02), (-0.02, -0.01), (-0.02, -0.01), (0.40, 0.43),
           (0.46, 0.45), (0.50, 0.49)]
m_seeds = [(0.15, 0.14), (0.19, 0.21), (0.18, 0.18), (0.26, 0.25),
           (0.26, 0.28), (0.29, 0.31)]
pos_vals = [0.04, np.nan, 0.09, 0.17, 0.58, 0.98]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0), gridspec_kw={"wspace": 0.28})
xp = np.arange(len(variants))
for ax, seeds, title, oracle in (
        (axes[0], k_seeds, "stiffness $\\log k$", 0.87),
        (axes[1], m_seeds, "mass $\\log m$", 0.86),
        (axes[2], None, "object position", 1.0)):
    if seeds is not None:
        means = [np.mean(s) for s in seeds]
        ax.bar(xp, means, color=VCOLORS, width=0.62)
        for i, (a, b) in enumerate(seeds):        # both seeds as dots
            ax.scatter([i - 0.10, i + 0.10], [a, b], s=9, color="k", zorder=5)
    else:
        ax.bar(xp, np.nan_to_num(pos_vals), color=VCOLORS, width=0.62)
    ax.axhline(oracle, color="k", ls=":", lw=1)
    ax.text(0.1, oracle - 0.10, "certificate", fontsize=8.5, ha="left")
    ax.set_xticks(xp, variants, fontsize=8.5)
    for tick, tc in zip(ax.get_xticklabels(), VCOLORS):
        tick.set_color(tc)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.07, 1.06)
    ax.axhline(0, color="k", lw=0.6)
for i, v in enumerate(pos_vals):
    if np.isnan(v):
        axes[2].text(i, 0.03, "n/a", fontsize=8, color=GRAY, ha="center")
axes[0].set_ylabel("probe $R^2$")
axes[0].annotate("touch as\ntarget", xy=(2.62, 0.28), xytext=(-0.35, 0.28),
                 fontsize=8.5, color=ORANGE, va="center",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
axes[2].annotate("targets\ncompose", xy=(3.62, 0.48), xytext=(0.2, 0.52),
                 fontsize=8.5, va="center",
                 arrowprops=dict(arrowstyle="->", lw=0.8))
fig.savefig("fig_targets.pdf")
plt.close(fig)

# -------- identifiability map (vector cells; certificate isolated) --------
from matplotlib.patches import Rectangle
from matplotlib import colormaps as _cmaps
rows = ["object position", r"mass $\log m$", r"stiffness $\log k$",
        r"drag $\gamma$"]
cols = ["prediction\nonly", "+ multi-\nhorizon",
        "+ cross-modal\ntargets", "+ fusion"]
M = np.array([
    [0.04, 0.89, 0.97, 0.98],
    [0.04, 0.15, 0.26, 0.29],
    [-0.02, -0.01, 0.46, 0.50],
    [0.00, 0.00, 0.13, 0.12],
])
CERT = np.array([1.00, 0.86, 0.87, 0.89])
GAP = 0.45
fig, ax = plt.subplots(figsize=(7.2, 1.9))
blues, grays = _cmaps["Blues"], _cmaps["Greys"]
for i in range(4):
    for j in range(4):
        v = float(M[i, j])
        ax.add_patch(Rectangle((j, 3 - i), 1, 1, facecolor=blues(max(v, 0.0)),
                               edgecolor="white", lw=1.6))
        ax.text(j + 0.5, 3 - i + 0.5, f"{v:.2f}", ha="center", va="center",
                fontsize=10, color="white" if v > 0.55 else "#1a1a1a",
                fontweight="bold" if v >= 0.4 else "normal")
    v = float(CERT[i])
    ax.add_patch(Rectangle((4 + GAP, 3 - i), 1, 1,
                           facecolor=grays(0.30 + 0.45 * v),
                           edgecolor="white", lw=1.6))
    ax.text(4 + GAP + 0.5, 3 - i + 0.5, f"{v:.2f}", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4 + GAP + 0.5],
              cols + ["recoverability\ncertificate"], fontsize=9.5)
ax.set_yticks([3.5, 2.5, 1.5, 0.5], rows, fontsize=10)
ax.get_yticklabels()[3].set_color(ORANGE)   # tie the drag row to the callout
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.add_patch(Rectangle((0.02, 0.02), 3.96, 0.96, fill=False, edgecolor=ORANGE,
                       lw=1.8, zorder=5))
ax.text(5.62, 0.5, "blind region", fontsize=9.5, color=ORANGE,
        ha="left", va="center", style="italic")
ax.set_xlim(-0.1, 6.95)
ax.set_ylim(-0.05, 4.05)
fig.savefig("fig_map.pdf")
plt.close(fig)

# ------- planning: the same task on two frozen latents (paired) -------
vz = np.load(pathlib.Path(r"C:\Users\Kaizh\Desktop\physical representation"
                          r"\exp\e003_motion\results\plan_viz_pair.npz"))
goal = vz["goal"]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.05),
                         gridspec_kw={"width_ratios": [1, 1, 1.28],
                                      "wspace": 0.26})


def _episode(ax, obj, fing, ok, title):
    ax.add_patch(plt.Circle(goal, 0.12, color=GREEN, alpha=0.18, zorder=0))
    ax.add_patch(plt.Circle(goal, 0.015, color=GREEN, zorder=4))
    n = len(obj)
    for i in range(n - 1):
        ax.plot(obj[i:i + 2, 0], obj[i:i + 2, 1], zorder=3, lw=2.4,
                color=plt.cm.Blues(0.35 + 0.6 * i / n))
    ax.plot(fing[:, 0], fing[:, 1], color=GRAY, lw=0.9, ls=":", zorder=2)
    ax.add_patch(plt.Circle(obj[0], 0.09, fill=False, color=BLUE, lw=1.1))
    ax.scatter(*fing[0], color=GRAY, s=13, zorder=5)
    # data coordinates: an equal-aspect axes shrinks away from its box, so
    # axes-fraction placement would land outside the drawn arena. No bold --
    # the bold cut of the serif family has no check/cross glyph.
    ax.text(0.955, 0.95, "\u2713" if ok else "\u00d7", fontsize=14 if ok else 17,
            color=GREEN if ok else ORANGE, ha="right", va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8.5)


_episode(axes[0], vz["obj_vis"], vz["fing_vis"], False,
         "vision-only latent")
_episode(axes[1], vz["obj_full"], vz["fing_full"], True,
         "fused $+$ targets")
axes[0].annotate("goal", xy=goal, xytext=(0.04, 0.93),
                 textcoords="axes fraction", fontsize=7, color=GREEN,
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7,
                                 shrinkB=6))
axes[0].annotate("object start", xy=vz["obj_vis"][0], xytext=(0.03, 0.05),
                 textcoords="axes fraction", fontsize=7, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7,
                                 shrinkB=8))

axD = axes[2]
for key, col in (("obj_vis", GRAY), ("obj_full", BLUE)):
    d = np.linalg.norm(vz[key] - goal, axis=1)
    axD.plot(np.arange(len(d)), d, color=col, lw=1.7)
axD.axhline(0.12, color=GREEN, ls="--", lw=1)
# direct labels: nothing sits on the curves
axD.text(26, 0.50, "vision-only", fontsize=7.5, color=GRAY)
axD.text(3, 0.31, "fused $+$ targets", fontsize=7.5, color=BLUE)
axD.text(2, 0.145, "goal region", fontsize=7, color=GREEN)
axD.set_xlabel("control step")
axD.set_ylabel("object\u2013goal distance")
axD.set_ylim(0, 0.62)
axD.set_title("distance to goal", fontsize=8.5)
fig.savefig("fig_plan.pdf")
plt.close(fig)

# ---------- appendix: gamma intervention ledger (forest plot) ----------
items = [
    ("multi-horizon heads", 0.05, GRAY),
    ("motion input channel", 0.12, GRAY),
    ("spatial tokens", 0.075, GRAY),
    ("motion-forecast targets", 0.072, GRAY),
    ("flow velocity sensor", 0.096, GRAY),
    ("glide-enriched data", 0.099, GRAY),
    ("Gaussian-NLL heads", 0.10, GRAY),
    (r"loss reweighting $\times 30$", 0.273, GRAY),
    ("log-speed coordinate", 0.333, ORANGE),
    ("supervised system-ID", 0.45, GREEN),
]
fig, ax = plt.subplots(figsize=(4.9, 2.5))
ys = np.arange(len(items))[::-1]
for y, (name, v, c) in zip(ys, items):
    ax.plot([0, v], [y, y], color="0.85", lw=1.0, zorder=1)
    ax.scatter([v], [y], s=34, color=c, zorder=3)
ax.axvline(0.43, color="0.35", ls="--", lw=1.0)
ax.text(0.445, 9.35, "pixel-derived\ncertificate", fontsize=7.5,
        color="0.35", va="top")
ax.axvline(0.89, color="k", ls=":", lw=1.0)
ax.text(0.878, 9.35, "state\ncertificate", fontsize=7.5, color="k",
        ha="right", va="top")
ax.set_yticks(ys, [n for n, _, _ in items], fontsize=8.5)
ax.set_xlabel(r"$\gamma$ readout $R^2$", fontsize=9.5)
ax.set_xlim(0, 0.95)
ax.set_ylim(-0.5, 9.6)
fig.savefig("fig_gamma_ledger.pdf")
plt.close(fig)

print("all figures regenerated")
