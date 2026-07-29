"""Setup figure: PokeWorld environment overview + sensor streams + the
X-JEPA variant factorial. One full-width figure carrying what Sections 3's
prose previously had to describe unaided."""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch, Rectangle

sys.path.insert(0, str(pathlib.Path(
    r"C:\Users\Kaizh\Desktop\physical representation\exp\e001_xmodal_jepa")))
import pokeworld as pw

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "font.size": 10, "axes.titlesize": 10.5,
    "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
BLUE, ORANGE, GRAY, GREEN, PURPLE = "#3B6FB6", "#E1812C", "#8C8C8C", "#3A923A", "#7B4FA6"
LORANGE, LGRAY = "#E19C56", "#9E9E9E"

# ---- pick an episode with a clean impact followed by a glide ----
sim = pw.simulate(200, 64, seed=4242)
best, best_t = None, None
for e in range(200):
    pk = sim["touch"][e, :, 2]
    cb = sim["touch"][e, :, 6]
    sp = np.linalg.norm(sim["obj"][e, :, 2:], axis=1)
    hits = np.where(pk > 3.0)[0]
    for t in hits:
        po_t = sim["obj"][e, t, :2]
        pf_t = sim["finger"][e, t, :2]
        centered = (0.25 < po_t[0] < 0.6 and 0.35 < po_t[1] < 0.7 and
                    pf_t[1] > 0.25)
        if (t < 42 and cb[t + 3:t + 18].sum() == 0 and
                sp[t + 3:t + 14].mean() > 0.25 and centered):
            best, best_t = e, int(t)
            break
    if best is not None:
        break
e, t_hit = best, best_t
frame = pw.render(sim["finger"][e:e + 1], sim["obj"][e:e + 1])[0, t_hit]

fig = plt.figure(figsize=(7.2, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 0.95, 1.5], wspace=0.25)

# ---------------- panel A: arena + hidden parameters ----------------
axA = fig.add_subplot(gs[0, 0])
pf = sim["finger"][e, t_hit, :2]
po = sim["obj"][e, t_hit, :2]
axA.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1.2, color="k"))
axA.add_patch(Circle(po, pw.R_OBJ, facecolor="#C9D7EC", edgecolor=BLUE, lw=1.4))
axA.add_patch(Circle(pf, pw.R_FINGER, facecolor="#F2E3C8", edgecolor="#A07B2A", lw=1.4))
a = sim["act"][e, t_hit]
an = a / max(np.linalg.norm(a), 1e-9)
tip = pf - (pw.R_FINGER + 0.012) * an     # arrowhead stops at the circle edge
tail = tip - 0.11 * an
axA.add_patch(FancyArrow(tail[0], tail[1], tip[0] - tail[0], tip[1] - tail[1],
                         width=0.012, color="#A07B2A",
                         length_includes_head=True))
def rim(center, radius, toward, margin=0.02):
    """Point on the circle edge facing `toward` — leader lines end exactly at
    the rim of the circle they name, never inside or floating in space."""
    d = np.asarray(toward, float) - np.asarray(center, float)
    d /= max(np.linalg.norm(d), 1e-9)
    return center + (radius + margin) * d


axA.annotate("finger",
             xy=rim(pf, pw.R_FINGER, (0.545, 0.50), margin=0.006),
             xytext=(0.545, 0.50),
             fontsize=8, color="#A07B2A", ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color="#A07B2A", lw=0.7,
                             shrinkA=1.5, shrinkB=0))
axA.annotate("object: hidden $m,\\gamma,k$\nresampled per episode",
             xy=rim(po, pw.R_OBJ, (0.22, 0.78)), xytext=(0.045, 0.965),
             fontsize=8, color=BLUE, va="top",
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7,
                             shrinkA=2, shrinkB=0))
axi = axA.inset_axes([0.66, 0.02, 0.32, 0.32])
axi.imshow(frame, cmap="gray", vmin=0, vmax=255)
axi.set_xticks([]); axi.set_yticks([])
for s in axi.spines.values():
    s.set_visible(True); s.set_color(GRAY)
axA.text(0.82, 0.37, "$64{\\times}64$ obs.", fontsize=7.5, ha="center", color="0.3")
axA.set_xlim(-0.02, 1.02); axA.set_ylim(-0.02, 1.02)
axA.set_aspect("equal"); axA.set_xticks([]); axA.set_yticks([])
axA.set_anchor("N")     # square sits at the cell top: titles align across panels
for s in axA.spines.values():
    s.set_visible(False)
axA.set_title("PokeWorld", fontsize=10.5)

# ---------------- panel B: sensor streams -> observability ----------------
T0, T1 = max(0, t_hit - 8), min(64, t_hit + 18)
ts = np.arange(T0, T1)
gsB = gs[0, 1].subgridspec(3, 1, hspace=0.45)
axs = [fig.add_subplot(gsB[i, 0]) for i in range(3)]
pk = sim["touch"][e, T0:T1, 2]
axs[0].plot(ts, pk, color=ORANGE, lw=1.4)
axs[0].set_ylabel("touch", fontsize=8)
axs[0].set_ylim(top=pk.max() * 1.18)
axs[0].annotate("sub-step peak $\\rightarrow k$",
                xy=(t_hit + 0.35, pk.max()),
                xytext=(t_hit + 4.5, pk.max() * 1.02), fontsize=8, va="center",
                color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
sp = np.linalg.norm(sim["obj"][e, T0:T1, 2:], axis=1)
axs[1].plot(ts, sp, color=BLUE, lw=1.4)
axs[1].set_ylabel("$|v|$", fontsize=8)
axs[1].set_ylim(top=sp.max() * 1.22)
axs[1].annotate("glide decay $\\rightarrow \\gamma$",
                xy=(t_hit + 9, sp[t_hit - T0 + 9]),
                xytext=(T1 - 1, sp.max() * 0.88), fontsize=8, color=BLUE,
                ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8,
                                shrinkA=1, shrinkB=1))
fsp = np.linalg.norm(sim["finger"][e, T0:T1, 2:], axis=1)
axs[2].plot(ts, fsp, color="#A07B2A", lw=1.4)
axs[2].set_ylabel("proprio", fontsize=8)
rngf = fsp.max() - fsp.min()
axs[2].set_ylim(fsp.min() - 0.08 * rngf, fsp.max() + 0.58 * rngf)
i_notch = t_hit - T0 + int(np.argmin(fsp[t_hit - T0:t_hit - T0 + 4]))
axs[2].annotate("impulse vs. $\\Delta v \\rightarrow m$",
                xy=(T0 + i_notch, fsp[i_notch]),
                xytext=(T0 + 1, fsp.max() + 0.50 * rngf), fontsize=8,
                color="#A07B2A", va="top",
                arrowprops=dict(arrowstyle="->", color="#A07B2A", lw=0.8))
for ax in axs:
    ax.set_yticks([])
    ax.set_xlim(T0, T1 - 1)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
axs[0].set_xticks([]); axs[1].set_xticks([])
axs[2].set_xlabel("time step", fontsize=8.5)
axs[0].set_title("sensor streams", fontsize=10.5)

# ---------------- panel C: the X-JEPA variant factorial ----------------
axC = fig.add_subplot(gs[0, 2])
axC.set_xlim(0, 10); axC.set_ylim(0, 10)
axC.axis("off")


def box(x, y, w, h, text, fc="#F2F2F2", ec="0.35", fs=8):
    axC.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                 facecolor=fc, edgecolor=ec, lw=0.9))
    axC.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


box(0.1, 7.7, 2.3, 1.5, "encoder\n(V [+P,T])")
box(3.6, 7.7, 2.4, 1.5, "predictor\n(actions)")
box(6.72, 8.65, 3.16, 0.85, "$\\hat z_{t+\\Delta}$", fc="#E8EEF7", fs=8)
box(6.72, 7.45, 3.16, 0.85, "touch, prop$_{t+\\Delta}$", fc="#FBEEDD", fs=7.2)
axC.text(8.30, 9.78, "$\\Delta{\\in}\\{1,4,16\\}$", fontsize=7, ha="center",
         color="0.35")
axC.annotate("", xy=(3.45, 8.45), xytext=(2.55, 8.45),
             arrowprops=dict(arrowstyle="->", lw=0.9, color="0.35"))
for x0, x1, y0, y1 in ((6.10, 6.60, 8.85, 9.02), (6.10, 6.60, 8.05, 7.92)):
    axC.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="->", lw=0.9, color="0.35",
                                 mutation_scale=10, shrinkA=0, shrinkB=0))

rows = [("V", GRAY), ("VF", PURPLE), ("VX$_p$", LGRAY), ("VX$_t$", LORANGE),
        ("VX", ORANGE), ("VFX", BLUE)]
flags = {"V": (0, 0, 0), "VF": (1, 0, 0), "VX$_p$": (0, 0, 1),
         "VX$_t$": (0, 1, 0), "VX": (0, 1, 1), "VFX": (1, 1, 1)}
cols = ["P,T\ninputs", "touch\ntarget", "prop\ntarget"]
x0s = [4.3, 6.7, 8.9]
axC.text(1.4, 5.72, "variant", fontsize=8.5, ha="center", style="italic")
for j, c in enumerate(cols):
    axC.text(x0s[j], 7.0, c, fontsize=8, ha="center", style="italic",
             va="top")
axC.plot([0.25, 9.75], [5.42, 5.42], color="0.8", lw=0.7, clip_on=False)
for i, (name, col) in enumerate(rows):
    y = 4.9 - i * 0.82
    axC.text(1.4, y, name, fontsize=9, ha="center", color=col,
             fontweight="bold")
    for j, f in enumerate(flags[name]):
        axC.text(x0s[j], y, "\u2713" if f else "--", fontsize=9, ha="center",
                 color="0.15" if f else "0.6")
axC.set_title("X-JEPA variants", fontsize=10.5)

fig.savefig("fig_setup.pdf")
print("fig_setup.pdf written; episode", e, "impact at t =", t_hit)
