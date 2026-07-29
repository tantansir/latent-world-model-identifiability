"""fig_pairs.pdf -- matched PokeWorld episodes rendered as whole-trajectory
overlays. Each pair starts from an identical frame and is driven by an
identical action sequence; only one hidden parameter differs, and nothing in
the rendering reveals which. Shares its episode selection and its patched
sample_props with docs/assets/make_pokeworld_videos.py, so the figure and the
project-page clips show the same episodes."""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "exp" / "e001_xmodal_jepa"))
import pokeworld as pw                                            # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "figure.dpi": 150, "savefig.bbox": "tight",
})
BLUE, GOLD, GRAY = "#3B6FB6", "#A07B2A", "#8C8C8C"
OBJ_FILL, FIN_FILL = "#C9D7EC", "#F2E3C8"
N_EP, T = 64, 64
_orig = pw.sample_props


def run(props, seed, glide):
    def patched(n, rng):
        p = _orig(n, rng)                  # consume the stream identically
        p[:] = np.asarray(props, float)
        return p
    pw.sample_props = patched
    try:
        return pw.simulate(N_EP, T, seed=seed,
                           glide_frac=1.0 if glide else 0.0)
    finally:
        pw.sample_props = _orig


def pick(sim, glide):
    best, score = 0, -1e9
    for e in range(N_EP):
        obj = sim["obj"][e]
        sp = np.linalg.norm(obj[:, 2:], axis=1)
        s = 0.5 * sp.max() + (1.0 if ((obj[:, :2] > 0.10) &
                                      (obj[:, :2] < 0.90)).all() else -3.0)
        if glide:
            s += 2.0 * sp[0] - np.linalg.norm(obj[0, :2] - 0.5)
        else:
            c = np.where(sim["touch"][e, :, 6] > 0.5)[0]
            if len(c) == 0 or c[0] > 18:
                continue
            s += 3.0 * np.linalg.norm(obj[-1, :2] - obj[c[0], :2]) - 0.05 * c[0]
        if s > score:
            best, score = e, s
    return best


PAIRS = [
    ("drag", 7, True, (1.2, 0.5, 2000.0), (1.2, 4.0, 2000.0),
     r"$\gamma{=}0.5$", r"$\gamma{=}4.0$"),
    ("mass", 3, False, (0.5, 1.5, 2000.0), (3.0, 1.5, 2000.0),
     r"$m{=}0.5$", r"$m{=}3.0$"),
    ("stiff", 11, False, (1.2, 1.5, 500.0), (1.2, 1.5, 6000.0),
     r"$k{=}500$", r"$k{=}6000$"),
]

fig, axes = plt.subplots(1, 6, figsize=(7.0, 1.42),
                         gridspec_kw={"wspace": 0.08})
col = 0
for name, seed, glide, lo, hi, lab_lo, lab_hi in PAIRS:
    sim_lo = run(lo, seed, glide)
    e = pick(sim_lo, glide)
    sim_hi = run(hi, seed, glide)
    for sim, lab in ((sim_lo, lab_lo), (sim_hi, lab_hi)):
        ax = axes[col]
        obj, fing = sim["obj"][e], sim["finger"][e]
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white",
                               edgecolor="0.15", lw=1.0))
        ax.plot(fing[:, 0], fing[:, 1], color=GRAY, lw=0.7, ls=":", alpha=0.9)
        for i in range(T - 1):
            ax.plot(obj[i:i + 2, 0], obj[i:i + 2, 1], color=BLUE,
                    lw=1.7, alpha=0.25 + 0.6 * i / T, solid_capstyle="round")
        ax.add_patch(Circle(obj[0, :2], pw.R_OBJ, facecolor=OBJ_FILL,
                            edgecolor=BLUE, lw=1.0, alpha=0.85))
        ax.add_patch(Circle(fing[0, :2], pw.R_FINGER, facecolor=FIN_FILL,
                            edgecolor=GOLD, lw=1.0, alpha=0.85))
        ax.add_patch(Circle(obj[-1, :2], pw.R_OBJ, fill=False,
                            edgecolor=BLUE, lw=1.4))
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(lab, fontsize=8.5, pad=3)
        col += 1

for ax, tag in zip(axes, ["(a)", "", "(b)", "", "(c)", ""]):
    if tag:
        ax.text(0.02, -0.055, tag, transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left")

fig.savefig(HERE / "fig_pairs.pdf")
print("fig_pairs.pdf written; episodes chosen per pair")
