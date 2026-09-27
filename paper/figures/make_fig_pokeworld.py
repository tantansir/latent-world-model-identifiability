"""fig_pokeworld.pdf / .png (Figure 2): hidden physics and empirical recovery in PokeWorld.

(a)-(c) Matched episodes: identical initial state and identical actions, replayed open loop in the simulator
(fim_certificate.replay); only one parameter differs. Top: stroboscopic overlay of the object every third step (light:
low value, dark: high value; finger in gold). Bottom: the stream in which the parameter shows: object speed on a log
scale for drag, object speed for mass, per-step peak touch force for stiffness.
(d) Recovery references (held-out R^2 of a GRU on the 16-step windows a model receives) by input set, on contact
windows (results/cert_matched.json) and on pure glides (results/glide_recovery.json).
Run from the repository root: python paper/figures/make_fig_pokeworld.py."""
import sys
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

HERE = pathlib.Path(__file__).resolve().parent
E3 = HERE.parents[1] / "exp" / "e003_motion"
sys.path.insert(0, str(HERE.parents[1] / "exp" / "e001_xmodal_jepa")); sys.path.insert(0, str(E3))
import pokeworld as pw
from fim_certificate import replay

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "font.size": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
})
LO, HI, GOLD, INK, MUTED = "#8FB0DD", "#1F3F73", "#A07B2A", "#333333", "#777777"
T = 24
DT = pw.DT


def run(pf, vf, po, vo, acts, m, g, k):
    f = lambda v: np.asarray(v, float)[None]
    fin, obj, tou = replay(f(pf), f(vf), f(po), f(vo), np.asarray(acts, float)[:, None, :],
                           np.array([m], float), np.array([g], float), np.array([k], float))
    return dict(p0f=np.asarray(pf, float), p0o=np.asarray(po, float), fin=fin[:, 0], obj=obj[:, 0], tou=tou[:, 0])


strike = np.array([[1.0, 0.0]] * 5 + [[-1.0, 0.0]] * (T - 5))
PAIRS = [
    ("drag", r"(a) drag $\gamma$", (r"$\gamma{=}0.5$", r"$\gamma{=}4$"),
     [run([0.07, 0.575], [0, 0], [0.22, 0.575], [0.62, 0.0], np.zeros((T, 2)), 1.2, g, 2000.0) for g in (0.5, 4.0)]),
    ("mass", r"(b) mass $m$", (r"$m{=}0.5$", r"$m{=}3$"),
     [run([0.12, 0.50], [0, 0], [0.38, 0.50], [0, 0], strike, m, 2.5, 2000.0) for m in (0.5, 3.0)]),
    ("stiff", r"(c) stiffness $k$", (r"$k{=}500$", r"$k{=}6000$"),
     [run([0.12, 0.50], [0, 0], [0.38, 0.50], [0, 0], strike, 1.2, 2.5, k) for k in (500.0, 6000.0)]),
]

from matplotlib.patches import FancyBboxPatch
FW_, FH_ = 5.5, 1.9
fig = plt.figure(figsize=(FW_, FH_))
L0, CW, GAP = 0.006, 0.186, 0.026                      # column geometry in figure fractions
VX0, VX1 = 0.0, 1.30                                   # arena x-range shown (arena units) + label margin
LANES = (0.30, 0.10)                                   # display height of the low / high episode lane
for c, (name, title, labs, (lo, hi)) in enumerate(PAIRS):
    x0 = L0 + c * (CW + GAP)
    VY0, VY1 = 0.0, 0.40
    ah = CW * FW_ * (VY1 - VY0) / (VX1 - VX0) / FH_
    ax = fig.add_axes([x0, 0.90 - ah, CW, ah])
    for ep, col, ly in ((lo, LO, LANES[0]), (hi, HI, LANES[1])):
        y0 = ep["p0o"][1]; sh = lambda q: np.c_[q[..., 0], ly + (q[..., 1] - y0)]
        ax.add_patch(FancyBboxPatch((0.005, ly - 0.095), 0.99, 0.19,
                                    boxstyle="round,pad=0,rounding_size=0.03", fc="#F4F5F7", ec="none", zorder=0))
        o, f = sh(ep["obj"][:, :2]), sh(ep["fin"][:, :2])
        for i in range(3, T, 4):                       # object every 0.2 s: spacing shows speed
            ax.add_patch(Circle(o[i], pw.R_OBJ * 0.8, fc=col, ec="none", alpha=0.18 + 0.3 * i / T, zorder=3))
        ax.add_patch(Circle(o[-1], pw.R_OBJ * 0.8, fc="none", ec=col, lw=1.0, zorder=4))
        for i in range(1, T, 4):
            ax.add_patch(Circle(f[i], pw.R_FINGER * 0.8, fc=GOLD, ec="none", alpha=0.10 + 0.25 * i / T, zorder=3))
        p0 = sh(ep["p0o"][None])[0]; pf0 = sh(ep["p0f"][None])[0]
        ax.add_patch(Circle(p0, pw.R_OBJ * 0.8, fc="white", ec=INK, lw=0.7, ls=(0, (1.5, 1)), zorder=5))
        ax.add_patch(Circle(pf0, pw.R_FINGER * 0.8, fc="white", ec=GOLD, lw=0.7, zorder=5))
        ax.text(1.30, ly, labs[0] if ep is lo else labs[1], color=col if ep is hi else "#5B84BD",
                fontsize=6.5, ha="right", va="center", zorder=6)
    ax.set_xlim(VX0, VX1); ax.set_ylim(VY0, VY1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=8, pad=2, loc="left")

    tx = fig.add_axes([x0 + 0.03, 0.10, CW - 0.035, 0.42])
    t = np.arange(1, T + 1) * DT
    for ep, col in ((lo, LO), (hi, HI)):
        if name == "stiff":
            y = ep["tou"][:, 2]; tx.plot(t[:10], y[:10], color=col, lw=1.2, marker="o", ms=1.8)
        else:
            y = np.linalg.norm(ep["obj"][:, 2:], axis=1)
            tx.plot(t, y, color=col, lw=1.2)
    if name == "drag":
        tx.set_yscale("log"); tx.set_ylim(3e-3, 1.0); tx.set_ylabel("speed (log)", fontsize=6.5, labelpad=1)
    elif name == "mass":
        tx.set_ylabel("speed", fontsize=6.5, labelpad=1)
    else:
        tx.set_ylabel("touch peak", fontsize=6.5, labelpad=1)
    for s in ("top", "right"):
        tx.spines[s].set_visible(False)
    tx.set_xlabel("time (s)", fontsize=6.5, labelpad=0.5)
    tx.tick_params(pad=1)

# ---------------------------------------------------------------- (d) recovery references
cm = json.load(open(E3 / "results" / "cert_matched.json"))["contact"]
gr = json.load(open(E3 / "results" / "glide_recovery.json"))
rows = [("frames", cm["V+A"]["gru"], gr.get("GRU, V inputs (image-derived object trajectory, actions)")),
        (r"+ $p$, $\tau$", cm["V+P+T+A"]["gru"], gr["GRU, VFX inputs (image-derived)"]),
        ("true state", cm["state+A"]["gru"], gr["GRU, simulator object state"])]
M = np.array([[r[1][0], r[1][1], r[1][2], np.nan if r[2] is None else r[2]] for r in rows])
hx = fig.add_axes([0.742, 0.10, 0.255, 0.66])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rec", ["#FFFFFF", "#C9D7EC", "#3B6FB6", "#1F3F73"])
hx.imshow(np.clip(np.nan_to_num(M, nan=0), 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(3):
    for j in range(4):
        v = M[i, j]
        hx.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                color="white" if (not np.isnan(v) and v > 0.6) else INK)
hx.set_xticks(range(4), [r"$m$", r"$\gamma$", r"$k$", r"$\gamma$"], fontsize=7.5)
hx.set_yticks(range(3), [r[0] for r in rows], fontsize=6.8)
hx.tick_params(length=0, pad=2)
for s in hx.spines.values():
    s.set_visible(False)
hx.axvline(2.5, color="white", lw=2.5)
hx.text(1, -0.58, "contact windows", ha="center", va="bottom", fontsize=6.3, color=MUTED)
hx.text(3, -0.58, "glides", ha="center", va="bottom", fontsize=6.3, color=MUTED)
fig.text(0.742 - 0.075, 0.885, r"(d) recovery ($R^2$)", fontsize=8, ha="left", va="bottom")

fig.savefig(HERE / "fig_pokeworld.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(HERE / "fig_pokeworld.png", dpi=220, bbox_inches="tight", pad_inches=0.01)
print("wrote fig_pokeworld; recovery matrix:\n", np.round(M, 3))
