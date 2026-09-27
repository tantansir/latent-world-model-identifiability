"""Qualitative real-robot figure: filmstrip of a held-out RH20T episode with
the model's force readout, contact-onset anticipation, and 16-step position
forecasts overlaid against ground truth."""
import os, sys, pathlib
CFG = sys.argv[1] if len(sys.argv) > 1 else "kuka"
os.environ["E010_DATA"] = "data" if CFG == "kuka" else "data_cfg1"
CKPT = "VFX_s0.pt" if CFG == "kuka" else "VFX_s0_c1.pt"
OUT = "fig_qual.pdf" if CFG == "kuka" else "fig_qual_b.pdf"
FTHR = 10.0 if CFG == "kuka" else 6.0
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp" / "e010_rh20t"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_legacy as TR
from model import XJEPA

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN = "#3B6FB6", "#E1812C", "#3A923A"
device = "cuda"
HERE = ROOT / "exp" / "e010_rh20t"

model = XJEPA("VFX").to(device)
model.load_state_dict(torch.load(HERE / "results" / CKPT, weights_only=True))
model.eval()
stats = TR.whiten_stats()

# fit probes on probe_tr (same protocol as evaluate)
Ztr, Htr, Rtr, _ = TR.encode_split(model, "probe_tr", stats, device)
Zf = Ztr.reshape(-1, 128)
sub = np.random.default_rng(0).choice(len(Zf), 60000, False)
_, fp = TR.ridge(Zf[sub], TR.TOUCH[Rtr.reshape(-1)][:, :3][sub][:, :1] * 0 +
                 np.linalg.norm(TR.TOUCH[Rtr.reshape(-1)][:, :3], axis=1,
                                keepdims=True)[sub], Zf[:100], np.zeros((100, 1)))
wf, muf, sdf, ymuf, ysdf = fp
fm_all = np.linalg.norm(TR.TOUCH[:, :3], axis=1)
lab = np.array([1.0 if (fm_all[r] < FTHR and fm_all[r + 1:r + 5].max() > FTHR) else 0.0
                for r in Rtr[:, -1]])
_, op = TR.ridge(Htr, lab[:, None], Htr[:100], lab[:100, None])
wo, muo, sdo, ymuo, ysdo = op

# pick a held-out episode with a clear late onset
cand = None
for e in TR.SPLITS["probe_te"]:
    s, L = TR.START[e], TR.LENGTH[e]
    fm = fm_all[s:s + L]
    on = np.where((fm[:-1] < FTHR) & (fm[1:] > FTHR))[0]
    if L >= 90 and len(on) and 40 <= on[0] <= L - 30:
        cand = (e, int(on[0]))
        break
e, onset = cand
s, L = TR.START[e], int(TR.LENGTH[e])
print(f"episode {e}, len {L}, onset at t={onset}")

# per-frame readouts over the episode via sliding 16-frame windows
zs, hs = [], []
with torch.no_grad():
    for t0 in range(0, L - 16 + 1):
        b, _ = TR.make_batch(np.array([[e, t0]]), stats, device, T=16)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"], b["prop"], b["touch"])
            h = model.pred.trunk(z[:, :-1].float(), b["act"][:, :-1])
        zs.append(z[0, -1].float().cpu().numpy())
        hs.append(h[0, -1].float().cpu().numpy())
zs, hs = np.array(zs), np.array(hs)
ts = np.arange(15, L)
f_pred = ((zs - muf) / sdf) @ wf * ysdf + ymuf
o_pred = ((hs - muo) / sdo) @ wo * ysdo + ymuo

# figure: filmstrip (4 semantic phases, auto-cropped to the manipulation
# zone, contrast-stretched, inter-panel change overlaid) + force + onset traces
fm_ep = fm_all[s:s + L]
t_peak = int(np.argmax(fm_ep))
rel = np.where(fm_ep[t_peak:] < FTHR)[0]
t_release = t_peak + int(rel[0]) if len(rel) else min(L - 1, t_peak + 30)
frames_t, labels = [], []
for t, lb in ((max(0, onset - 30), "approach"), (onset, "onset $t_0$"),
              (t_peak, "peak $\\|F\\|$"), (t_release, "release")):
    if t not in frames_t:
        frames_t.append(int(t)); labels.append(lb)

# locate the manipulation zone from whole-episode motion energy (96-space)
ep_imgs = np.asarray(TR.IMG[s:s + L]).astype(np.float32)
mot = ep_imgs.std(0)
K = 56
c = np.pad(np.cumsum(np.cumsum(mot, 0), 1), ((1, 0), (1, 0)))
sm = c[K:, K:] - c[:-K, K:] - c[K:, :-K] + c[:-K, :-K]
iy, ix = np.unravel_index(np.argmax(sm), sm.shape)

# native-resolution COLOR display frames: the stored 96x96 grayscale tensors
# are the model input; for display we pull the same frames from the original
# RH20T videos (LeRobot episode->file mapping, 10 fps)
import pandas as pd
import imageio.v3 as iio
import torch.nn.functional as TF
CFGDIR = pathlib.Path(os.environ.get("RH20T_ROOT", ROOT / "data_rh20t")) / ("cfg7" if CFG == "kuka" else "cfg1")
NCAM = ("observation.images.cam_037522061512" if CFG == "kuka"
        else "observation.images.cam_035622060973")
em = pd.read_parquet(CFGDIR / "meta" / "episodes" / "chunk-000" /
                     "file-000.parquet")
row = em[em["episode_index"] == e].iloc[0]
fi = int(row[f"videos/{NCAM}/file_index"])
off = int(round(float(row[f"videos/{NCAM}/from_timestamp"]) * 10))
vid = CFGDIR / "videos" / NCAM / "chunk-000" / f"file-{fi:03d}.mp4"
need = {off + t: i for i, t in enumerate(frames_t)}
nat = [None] * len(frames_t)
for j, fr in enumerate(iio.imiter(vid, plugin="pyav")):
    if j in need:
        nat[need[j]] = np.asarray(fr).copy()
    if j >= max(need):
        break
H, W = nat[0].shape[:2]
cx = (ix + K / 2) * W / 96.0
S = H                      # full-height square: keeps the arm's approach path
y0 = 0
x0 = int(np.clip(cx - S / 2, 0, W - S))
nat_crops = [f[y0:y0 + S, x0:x0 + S] for f in nat]

def brighten(img):
    """Display-only brightness/contrast stretch (noted in the caption)."""
    f = img.astype(np.float32) / 255.0
    lo, hi = np.percentile(f, 2), np.percentile(f, 99.5)
    f = np.clip((f - lo) / max(hi - lo, 1e-3), 0, 1) ** 0.85
    mean = f.mean(axis=-1, keepdims=True)      # mild saturation lift so the
    f = np.clip(mean + (f - mean) * 1.18, 0, 1)  # stretch doesn't wash color
    return (f * 255).astype(np.uint8)


fig = plt.figure(figsize=(6.8, 3.15))
gs = fig.add_gridspec(3, len(frames_t), height_ratios=[1.9, 1, 1],
                      hspace=0.45, wspace=0.06)
frame_axes = []
for i, (t, lb) in enumerate(zip(frames_t, labels)):
    ax = fig.add_subplot(gs[0, i])
    frame_axes.append(ax)
    ax.imshow(brighten(nat_crops[i]))
    if i > 0:     # orange overlay: change since previous panel (96-space mask
                  # of the same region, upsampled -> smooth, speckle-free)
        d_full = np.abs(ep_imgs[frames_t[i]] - ep_imgs[frames_t[i - 1]])
        ys, ye = int(y0 * 96 / H), int(np.ceil((y0 + S) * 96 / H))
        xs, xe = int(x0 * 96 / W), int(np.ceil((x0 + S) * 96 / W))
        a96 = np.clip((d_full[ys:ye, xs:xe] - 7.0) / 50.0, 0, 1) * 0.5
        a = TF.interpolate(torch.tensor(a96, dtype=torch.float32)[None, None],
                           size=(S, S), mode="bilinear",
                           align_corners=False)[0, 0].numpy()
        rgba = np.zeros((S, S, 4))
        rgba[..., :3] = (0.882, 0.506, 0.173)
        rgba[..., 3] = a
        ax.imshow(rgba)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(lb, fontsize=8, color=ORANGE if t == onset else "black")
    for sp in ax.spines.values():
        sp.set_visible(t == onset)
        sp.set_color(ORANGE); sp.set_linewidth(1.6)
ax1 = fig.add_subplot(gs[1, :])
from matplotlib.patches import ConnectionPatch
for fax, t in zip(frame_axes, frames_t):
    fig.add_artist(ConnectionPatch(
        xyA=(0.5, -0.04), coordsA=fax.transAxes,
        xyB=(t, 1.0), coordsB=ax1.get_xaxis_transform(),
        ls=":", lw=0.7, color="0.6"))
ax1.plot(np.arange(L), fm_ep, color="k", lw=1.1, label="true $\\|F\\|$")
ax1.plot(ts, f_pred[:, 0], color=BLUE, lw=1.4, label="latent readout")
for t in frames_t:
    ax1.axvline(t, color="0.55", lw=0.7, ls=":", zorder=0)
ax1.axvline(onset, color=ORANGE, lw=1, ls="--")
ax1.set_ylabel("$\\|F\\|$ (N)")
ax1.legend(frameon=False, fontsize=7, loc="best")
ax1.set_xlim(0, L - 1); ax1.set_xticks([])
ax2 = fig.add_subplot(gs[2, :])
ax2.plot(ts, o_pred[:, 0], color=GREEN, lw=1.4,
         label="onset score (next 0.4 s)")
for t in frames_t:
    ax2.axvline(t, color="0.55", lw=0.7, ls=":", zorder=0)
ax2.axvline(onset, color=ORANGE, lw=1, ls="--")
ax2.set_xlabel("time step (10 Hz)"); ax2.set_ylabel("onset score")
ax2.set_xlim(0, L - 1)
ax2.legend(frameon=False, fontsize=7, loc="best")
fig.savefig(str(pathlib.Path(__file__).parent / OUT))
print(OUT, "written")
