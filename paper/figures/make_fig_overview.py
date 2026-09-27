"""fig_overview.pdf / .svg / .png (Figure 1): the action-conditioned JEPA world model, one column per time step.

Every element follows the code (exp/e003_motion/model.py, --acorr) and is centred on its column:
  * step s records the frame o_s, finger proprioception p_s and touch tau_s; a_s is the action applied during step s,
    so o_s is its result. Real values of one PokeWorld episode (seed 4242, episode 454, t = 15; all disks inside the
    frame in every shown step): the finger speeds up, strikes the object at t and t+1, and is slowed by the impact.
  * p and tau are drawn as curves over time (p: finger speed, tau: per-step peak contact force); dots mark the shown
    steps; "..." columns stand for omitted steps, over which the curves are compressed. A time axis
    with one tick per shown step runs underneath.
  * encoder E (shared weights; it also encodes the target steps) embeds each step into z_s. In the F variants E also
    receives p_s and tau_s (stated in the caption, not drawn).
  * causal transformer P: position s receives z_s from below and is conditioned on a_s from above (AdaLN); its
    output node at position s attends to the positions <= s (arcs); the output at t is h_t.
  * heads g: z_hat_{t+1} from h_t alone; z_hat_{t+4}, z_hat_{t+16} also from a_{t+1:t+Delta} (D = {1,4,16}); each
    prediction (hollow) is regressed onto the target embedding below it. (The sensor targets of the X variants are
    stated in the caption, not drawn.)
  * SIGReg acts on all embeddings z (context and future).
  * linear probe: the frozen h_t is read out by a trained linear map W into theta-hat (icon: frozen h -> W -> theta-hat).
Solid lines only. Every element carries an SVG id (set_gid) for editing in Figma.
Run from the repository root: python paper/figures/make_fig_overview.py."""
import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon, Circle, PathPatch
from matplotlib.path import Path

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "exp" / "e001_xmodal_jepa"))
import pokeworld as pw

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8,
})
BLUE, ORANGE, GOLD, GREEN = "#3B6FB6", "#E1812C", "#A07B2A", "#3A923A"
ZFILL, BAND, EFILL, EEDGE, SIGBAND = "#D3DFF0", "#E3EBF6", "#EEF2F8", "#9DB4D9", "#F1F5FB"
AGRAY, INK, MUTED, ATT = "#6F6F6F", "#333333", "#7A7A7A", "#7F9CC8"

# ---------------------------------------------------------------- data: one PokeWorld episode
E_ID, T0 = 454, 15
sim = pw.simulate(512, 64, seed=4242)
frames = pw.render(sim["finger"][E_ID:E_ID + 1], sim["obj"][E_ID:E_ID + 1])[0]
tpeak = sim["touch"][E_ID, :, 2]                                   # peak reaction force within each step
fspeed = np.linalg.norm(sim["finger"][E_ID, :, 2:], axis=1)        # finger speed (part of p)
act = sim["act"][E_ID]

# ---------------------------------------------------------------- canvas (inches; placed at \linewidth = 5.5 in)
W, H = 5.5, 2.12
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_aspect("equal")

COLS = [("dots", None, 0.86), ("ctx", -2, 1.27), ("ctx", -1, 1.77), ("ctx", 0, 2.27),
        ("fut", 1, 3.35), ("dots", None, 3.75), ("fut", 4, 4.15), ("dots", None, 4.55), ("fut", 16, 4.95)]
XS = {s: x for k, s, x in COLS if s is not None}
XL, XR = 0.64, 5.14                                # left and right end of the timeline

# rows (y, inches)
Y_TLAB, Y_AXIS = 0.03, 0.115                       # time labels, time axis
TY0, PY0, CH = 0.16, 0.325, 0.13                   # tau and p curve bands
FY0, FW = 0.50, 0.28                               # frame
EY0, EH = 0.825, 0.085                             # encoder
ZY0, ZH, ZW = 0.96, 0.13, 0.30                     # embedding tokens
BY0, BH = 1.16, 0.36                               # predictor band
YN = BY0 + 0.125                                   # node row inside the predictor band
AY0, AH, AW = 1.75, 0.13, 0.30                     # action tokens
XG = XS[0] + 0.66                                  # heads


def add(p, name):
    ax.add_patch(p); p.set_gid(name); return p


def rbox(x0, y0, w, h, fc, ec="none", lw=0.7, r=0.035, z=3, name=None):
    return add(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                              zorder=z), name or "box")


def arrow(p0, p1, color=INK, lw=0.7, cs="arc3,rad=0", z=3, name=None, hl=0.22, hw=0.12, style="-|>"):
    return add(FancyArrowPatch(p0, p1, connectionstyle=cs, arrowstyle=f"{style},head_length={hl},head_width={hw}",
                               color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0, mutation_scale=10,
                               capstyle="round", joinstyle="round"), name or "arrow")


def line(xs, ys, color, lw=0.7, z=2, name="line"):
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round", gid=name)


def txt(x, y, s, size=8, color=INK, ha="center", va="center", name=None, **kw):
    t = ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=8, **kw)
    if name: t.set_gid(name)
    return t


def lab(s):                                        # subscript label of step offset s
    return "t" if s == 0 else (rf"t\!-\!{-s}" if s < 0 else rf"t\!+\!{s}")


def hbrace(x0, x1, y, h=0.045, name="brace"):      # horizontal curly brace opening downwards, tip up
    xm_ = (x0 + x1) / 2
    verts = [(x0, y - h), (x0, y), (x0 + 0.06, y), (xm_ - 0.06, y), (xm_, y), (xm_, y + h * 0.6),
             (xm_, y), (xm_ + 0.06, y), (x1 - 0.06, y), (x1, y), (x1, y - h)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.MOVETO, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    add(PathPatch(Path(verts, codes), fill=False, ec="#9A9A9A", lw=0.7, zorder=3), name)


# ---------------------------------------------------------------- time axis and the p, tau curves
ANCH = [(-15, XL), (-3, 1.06)] + [(s, XS[s]) for s in (-2, -1, 0, 1, 4)] + [(5, XS[4] + 0.15), (16, XS[16])]
steps = np.arange(-15, 17)
cx = np.interp(steps, [a for a, _ in ANCH], [b for _, b in ANCH])
arrow((XL, Y_AXIS), (XR + 0.12, Y_AXIS), color="#8A8A8A", lw=0.8, name="time_axis", hl=0.24, hw=0.13)
txt(XR + 0.15, Y_AXIS, "time", 6.0, MUTED, ha="left", style="italic", name="lab_time")
for s, x in XS.items():
    line([x, x], [Y_AXIS, Y_AXIS + 0.025], "#8A8A8A", lw=0.8, z=3, name=f"tick_{s}")
    txt(x, Y_TLAB, rf"${lab(s)}$", 6.8, MUTED, name=f"time_{s}")
for k, s, x in COLS:
    if k == "dots":
        txt(x, Y_TLAB, r"$\cdots$", 7, MUTED, name="dots_time")


def curve(vals, y0, color, name):
    y = y0 + 0.012 + (CH - 0.024) * vals
    ax.plot(cx, y, color=color, lw=1.2, solid_capstyle="round", solid_joinstyle="round", zorder=4, gid=f"{name}_line")
    for s in XS:
        add(Circle((cx[s + 15], y[s + 15]), 0.02, fc=color, ec="white", lw=0.4, zorder=6), f"{name}_dot_{s}")
    return y


seg = np.arange(T0 - 15, T0 + 17)
norm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-9)
curve(0.95 * np.clip(tpeak[seg], 0, None) / tpeak[seg].max(), TY0, ORANGE, "curve_tau")
curve(norm(fspeed[seg]), PY0, GOLD, "curve_p")

# ---------------------------------------------------------------- SIGReg band behind all embeddings (context and future)
rbox(0.66, ZY0 - 0.022, 5.13 - 0.66, ZH + 0.044, fc=SIGBAND, r=0.05, z=1, name="sigreg_band")
gx = np.linspace(-1, 1, 40)
ax.plot(5.29 + 0.10 * gx, ZY0 + 0.035 + 0.06 * np.exp(-4.0 * gx ** 2), color=BLUE, lw=0.7, zorder=5, gid="sigreg_bell")
line([5.17, 5.41], [ZY0 + 0.035, ZY0 + 0.035], BLUE, lw=0.4, z=5, name="sigreg_bell_axis")
txt(5.29, ZY0 + 0.005, "SIGReg", 5.6, BLUE, va="top", name="lab_sigreg")

# ---------------------------------------------------------------- causal transformer: one output node per position
XB1 = XS[0] + 0.24
rbox(0.66, BY0, XB1 - 0.66, BH, fc=BAND, r=0.08, z=1, name="predictor_band")
txt(0.91, YN + 0.005, "causal\ntransformer", 6.2, "#46628C", linespacing=1.0, name="lab_predictor")
RN = RT = 0.095                                    # all output nodes have the same size
for s in (-2, -1, 0):
    add(Circle((XS[s], YN), RN, fc="white", ec=BLUE, lw=1.0, zorder=6), "h_t" if s == 0 else f"node_{s}")
    txt(XS[s], YN, rf"$h_{{{lab(s)}}}$", 6.8, BLUE, name="lab_h" if s == 0 else f"lab_h_{s}")


def on_circle(x, r, ang):
    return x + r * np.cos(np.radians(ang)), YN + r * np.sin(np.radians(ang))


# causal self-attention: each position attends to the earlier positions (arcs from earlier to later). The long arc
# leaves and enters at the node tops and runs high; the short arcs use the node sides and run low, so none overlap.
for (s0, a0), (s1, r1, a1), rad in (((-2, 35), (-1, RN, 145), -0.35), ((-2, 80), (0, RT, 100), -0.22),
                                    ((-1, 35), (0, RT, 145), -0.35)):
    p0 = on_circle(XS[s0], RN, a0); p1 = on_circle(XS[s1], r1, a1)
    arrow(p0, p1, color=ATT, lw=0.6, cs=f"arc3,rad={rad}", name=f"attn_{s0}_{s1}", hl=0.14, hw=0.08)

# ---------------------------------------------------------------- columns
ZP0 = YN - ZH / 2                                  # predicted tokens: same height as the node row
for kind, s, x in COLS:
    if kind == "dots":
        for y in (FY0 + FW / 2, ZY0 + ZH / 2, AY0 + AH / 2):
            txt(x, y, r"$\cdots$", 8, MUTED, name="dots")
        continue
    fut = kind == "fut"
    ax.imshow(frames[T0 + s], cmap="gray", origin="lower", vmin=0, vmax=255, interpolation="nearest",
              extent=(x - FW / 2, x + FW / 2, FY0, FY0 + FW), zorder=3).set_gid(f"frame_{s}")
    add(Rectangle((x - FW / 2, FY0), FW, FW, fill=False, ec="#555555", lw=0.5, zorder=4), f"frame_border_{s}")
    add(Polygon([(x - 0.18, EY0), (x + 0.18, EY0), (x + 0.13, EY0 + EH), (x - 0.13, EY0 + EH)], closed=True,
                fc=EFILL, ec=EEDGE, lw=0.6, zorder=2), f"encoder_{s}")
    arrow((x, FY0 + FW), (x, EY0), color="#555555", lw=0.6, name=f"o_to_E_{s}", hl=0.2, hw=0.11)
    arrow((x, EY0 + EH), (x, ZY0), color="#555555", lw=0.6, name=f"E_to_z_{s}", hl=0.2, hw=0.11)
    rbox(x - ZW / 2, ZY0, ZW, ZH, fc=ZFILL, name=f"z_{s}")
    txt(x, ZY0 + ZH / 2, rf"$z_{{{lab(s)}}}$", 6.8, "#1F3F73", name=f"lab_z_{s}")
    a = act[T0 + s]; n = np.linalg.norm(a) + 1e-9
    if not fut:
        rbox(x - AW / 2, AY0, AW, AH, fc="#F2F2F2", ec="#BBBBBB", lw=0.5, name=f"a_{s}")
        arrow((x, ZY0 + ZH), (x, BY0), color="#6C87B5", lw=0.6, name=f"z_to_P_{s}", hl=0.2, hw=0.11)
        arrow((x, AY0), (x, BY0 + BH), color=AGRAY, lw=0.6, name=f"a_to_P_{s}", hl=0.2, hw=0.11)
    else:
        rbox(x - ZW / 2, ZP0, ZW, ZH, fc="white", ec=BLUE, lw=0.8, z=4, name=f"zhat_{s}")
        txt(x, ZP0 + ZH / 2, rf"$\hat z_{{{lab(s)}}}$", 6.8, BLUE, name=f"lab_zhat_{s}")
        arrow((x, ZY0 + ZH + 0.005), (x, ZP0 - 0.005), color=MUTED, lw=0.5, style="<|-|>", name=f"loss_z_{s}",
              hl=0.14, hw=0.08)
    gx0, gy0, L = x - 0.075, AY0 + AH / 2, 0.05
    add(FancyArrowPatch((gx0 - L * a[0] / n, gy0 - L * a[1] / n), (gx0 + L * a[0] / n, gy0 + L * a[1] / n),
                        arrowstyle="-|>,head_length=0.13,head_width=0.08", color=AGRAY, lw=0.7, mutation_scale=10,
                        zorder=5), f"a_glyph_{s}")
    txt(x + 0.05, AY0 + AH / 2, rf"$a_{{{lab(s)}}}$", 6.5, AGRAY, name=f"lab_a_{s}")

# future actions: one group, one connector into the heads (used for Delta > 1)
xa0, xa1 = XS[1] - AW / 2 - 0.03, XS[16] + AW / 2 + 0.03
rbox(xa0, AY0 - 0.025, xa1 - xa0, AH + 0.05, fc="#F2F2F2", ec="#BBBBBB", lw=0.5, r=0.05, z=2, name="future_actions_group")
YA = AY0 + AH / 2
line([xa0, XG], [YA, YA], AGRAY, lw=0.6, z=3, name="future_actions_line")
arrow((XG, YA), (XG, YN + 0.076), color=AGRAY, lw=0.6, name="future_actions_to_g", hl=0.2, hw=0.11)
txt(XG - 0.035, YN + 0.27, r"$a_{t+1:t+\Delta}$", 6.2, AGRAY, ha="right", name="lab_future_actions")

# ---------------------------------------------------------------- heads g: three latent heads, separate exits, no overlap
rbox(XG - 0.075, YN - 0.075, 0.15, 0.15, fc="white", ec=INK, lw=0.7, r=0.03, z=6, name="heads_g")
txt(XG, YN, r"$g$", 7.5, INK, name="lab_g")
arrow((XS[0] + RT, YN), (XG - 0.075, YN), color=INK, lw=0.7, name="h_to_g", hl=0.2, hw=0.11)
arrow((XG + 0.075, YN - 0.03), (XS[1] - ZW / 2, ZP0 + ZH / 2 - 0.03), color=BLUE, lw=0.85, name="head_d1")
arrow((XG + 0.075, YN + 0.035), (XS[4], ZP0 + ZH + 0.004), color=BLUE, lw=0.85, cs="arc3,rad=-0.24", name="head_d4")
arrow((XG + 0.06, YN + 0.075), (XS[16], ZP0 + ZH + 0.004), color=BLUE, lw=0.85, cs="arc3,rad=-0.235", name="head_d16")
for d in (1, 4, 16):
    txt(XS[d] + ZW / 2 + 0.025, ZP0 + ZH / 2, rf"$\Delta{{=}}{d}$", 6.2, BLUE, ha="left", name=f"lab_d{d}")

# what g consists of (exp/e003_motion/model.py): a linear next-step projector, and additive heads for Delta = 4, 16
GX0, GW_, GY0, GH_ = XS[0] + 0.215, 0.66, FY0 + 0.01, 0.33
rbox(GX0, GY0, GW_, GH_, fc="#F7F8FA", ec="#C9C9C9", lw=0.5, r=0.04, z=2, name="g_detail_box")
gxl = GX0 + 0.05
txt(gxl, GY0 + GH_ - 0.075, r"$\hat z_{t+1}=W_1h_t$", 6.2, INK, ha="left", name="g_detail_1")
txt(gxl, GY0 + GH_ - 0.17, r"$\hat z_{t+\Delta}=W_\Delta h_t$", 6.2, INK, ha="left", name="g_detail_2")
txt(gxl + 0.07, GY0 + GH_ - 0.265, r"$+\,\phi_\Delta(a_{t+1:t+\Delta})$", 6.2, INK, ha="left", name="g_detail_3")

# ---------------------------------------------------------------- linear probe: frozen h_t -> trained linear map W -> theta-hat
XT_ = XS[0] + 0.33                                  # theta-hat, up and to the right of h_t
arrow(on_circle(XS[0], RT, 40), (XT_, AY0 + 0.07), color=GREEN, lw=0.75, name="probe", hl=0.2, hw=0.11)
txt(XT_ - 0.005, AY0 + 0.14, r"$\hat\theta$", 8, GREEN, name="lab_thetahat")
txt(XT_ + 0.075, AY0 + 0.255, "linear probe", 6.0, GREEN, ha="left", name="lab_probe")   # upper right of theta-hat
IX0, IY0 = XT_ + 0.105, AY0 + 0.035                 # icon directly below the text: frozen h -> W -> theta-hat
c = 0.032
for k in range(4):                                 # frozen state h_t
    add(Rectangle((IX0, IY0 + k * c), c, c, fc="#E3EBF6", ec=BLUE, lw=0.45, zorder=6), f"icon_h_{k}")
sx_, sy_ = IX0 - 0.032, IY0 + 2 * c                # snowflake: frozen
for ang in (0, 60, 120):
    dx, dy = 0.024 * np.cos(np.radians(ang)), 0.024 * np.sin(np.radians(ang))
    line([sx_ - dx, sx_ + dx], [sy_ - dy, sy_ + dy], BLUE, lw=0.5, z=7, name="icon_frozen")
xw = IX0 + c + 0.045
arrow((IX0 + c + 0.006, IY0 + 2 * c), (xw - 0.006, IY0 + 2 * c), color=GREEN, lw=0.55, name="icon_arrow1", hl=0.14, hw=0.08)
add(Rectangle((xw, IY0 + 0.3 * c), 0.034, 3.4 * c, fc=GREEN, ec="none", zorder=6), "icon_readout_W")
txt(xw + 0.017, IY0 - 0.035, r"$W$", 6.0, GREEN, name="icon_W")
xo = xw + 0.034 + 0.045
arrow((xw + 0.04, IY0 + 2 * c), (xo - 0.006, IY0 + 2 * c), color=GREEN, lw=0.55, name="icon_arrow2", hl=0.14, hw=0.08)
for k, nm in enumerate(("k", "g", "m")):          # theta-hat = (m, gamma, k)-hat
    add(Rectangle((xo, IY0 + (0.5 + k) * c), c, c, fc="#E6F2E6", ec=GREEN, lw=0.45, zorder=6), f"icon_theta_{nm}")

# ---------------------------------------------------------------- left margin: row labels with their symbols, theta
for y, s_, c_, nm in ((AY0 + AH / 2, r"$a$", AGRAY, "a"), (YN, r"$P$", BLUE, "P"), (ZY0 + ZH / 2, r"$z$", BLUE, "z"),
                      (EY0 + EH / 2, r"$E$", BLUE, "E")):
    txt(0.52, y, s_, 8.5, c_, name="row_" + nm)
KX = 0.43
add(Rectangle((KX - 0.033, FY0 + FW / 2 - 0.033), 0.066, 0.066, fc="black", ec="#555555", lw=0.5, zorder=4), "key_o")
line([KX - 0.045, KX + 0.035], [PY0 + CH / 2, PY0 + CH / 2], GOLD, lw=1.3, z=4, name="key_p")
line([KX - 0.045, KX + 0.035], [TY0 + CH / 2, TY0 + CH / 2], ORANGE, lw=1.3, z=4, name="key_tau")
for y, s_, c_, nm in ((FY0 + FW / 2, r"$o$", INK, "o"), (PY0 + CH / 2, r"$p$", GOLD, "p"), (TY0 + CH / 2, r"$\tau$", ORANGE, "tau")):
    txt(KX + 0.09, y, s_, 8.5, c_, name=f"key_lab_{nm}")
bx, y0b, y1b = 0.33, TY0, FY0 + FW
ym = (y0b + y1b) / 2
verts = [(bx + 0.05, y1b), (bx, y1b), (bx, ym + 0.06), (bx, ym + 0.03), (bx - 0.045, ym), (bx, ym - 0.03),
         (bx, ym - 0.06), (bx, y0b), (bx + 0.05, y0b)]
add(PathPatch(Path(verts, [Path.MOVETO] + [Path.CURVE3] * 8), fill=False, ec=MUTED, lw=0.7, zorder=4), "theta_brace")
txt(0.15, ym + 0.05, r"$\theta$", 9.5, INK, name="lab_theta")
txt(0.15, ym - 0.08, "hidden", 5.6, MUTED, style="italic", name="lab_theta_hidden")

# ---------------------------------------------------------------- regions: braces over the 16-step context and the future
YBR = AY0 + AH + 0.075
hbrace(0.70, XS[0] + AW / 2, YBR, name="brace_context")
txt((0.70 + XS[0] + AW / 2) / 2, YBR + 0.085, "context: 16 steps", 6.0, MUTED, style="italic", name="lab_context")
hbrace(xa0, xa1, YBR, name="brace_future")
txt((xa0 + xa1) / 2, YBR + 0.085, "future: prediction targets", 6.0, MUTED, style="italic", name="lab_future")

for ext in ("pdf", "svg", "png"):
    fig.savefig(HERE / f"fig_overview.{ext}", dpi=300 if ext == "png" else None)
print("wrote fig_overview; finger speed / touch at t-2..t+16:",
      np.round(fspeed[[T0 - 2, T0 - 1, T0, T0 + 1, T0 + 4, T0 + 16]], 2),
      np.round(tpeak[[T0 - 2, T0 - 1, T0, T0 + 1, T0 + 4, T0 + 16]], 1))
