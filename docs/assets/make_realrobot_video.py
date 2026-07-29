"""Project-page video: a held-out RH20T episode with the force magnitude read
out of the frozen latent, drawn against the wrist sensor as the clip plays.

Usage: python make_realrobot_video.py [kuka|cfg1]
Mirrors the probe protocol of paper/iclr2026/make_qual.py.
"""
import os
import pathlib
import sys

CFG = sys.argv[1] if len(sys.argv) > 1 else "kuka"
os.environ["E010_DATA"] = "data" if CFG == "kuka" else "data_cfg1"
CKPT = "VFX_s0.pt" if CFG == "kuka" else "VFX_s0_c1.pt"
FTHR = 10.0 if CFG == "kuka" else 6.0
OUTNAME = f"realrobot_{'kuka' if CFG == 'kuka' else 'flexiv'}.mp4"

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E010 = ROOT / "exp" / "e010_rh20t"
sys.path.insert(0, str(E010))

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import pandas as pd                                              # noqa: E402
import imageio.v2 as imageio                                     # noqa: E402
import imageio.v3 as iio                                         # noqa: E402
import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

import train as TR                                               # noqa: E402
from model import XJEPA                                          # noqa: E402

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#3B6FB6", "#E1812C"
device = "cuda"

model = XJEPA("VFX").to(device)
model.load_state_dict(torch.load(E010 / "results" / CKPT, weights_only=True))
model.eval()
stats = TR.whiten_stats()

# force probe, fit exactly as in the paper's evaluation
Ztr, Htr, Rtr, _ = TR.encode_split(model, "probe_tr", stats, device)
Zf = Ztr.reshape(-1, 128)
sub = np.random.default_rng(0).choice(len(Zf), 60000, False)
fmag = np.linalg.norm(TR.TOUCH[Rtr.reshape(-1)][:, :3], axis=1, keepdims=True)
_, fp = TR.ridge(Zf[sub], fmag[sub], Zf[:100], np.zeros((100, 1)))
wf, muf, sdf, ymuf, ysdf = fp
fm_all = np.linalg.norm(TR.TOUCH[:, :3], axis=1)

# a held-out episode with a clear late contact onset
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
print(f"episode {e}, {L} frames, onset t={onset}", flush=True)

zs = []
with torch.no_grad():
    for t0 in range(0, L - 16 + 1):
        b, _ = TR.make_batch(np.array([[e, t0]]), stats, device, T=16)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"], b["prop"], b["touch"])
        zs.append(z[0, -1].float().cpu().numpy())
zs = np.array(zs)
ts = np.arange(15, L)
f_pred = (((zs - muf) / sdf) @ wf * ysdf + ymuf)[:, 0]
fm_ep = fm_all[s:s + L]

# native-resolution colour frames straight from the RH20T clip
CFGDIR = ROOT / "data_rh20t" / ("cfg7" if CFG == "kuka" else "cfg1")
NCAM = ("observation.images.cam_037522061512" if CFG == "kuka"
        else "observation.images.cam_035622060973")
em = pd.read_parquet(CFGDIR / "meta" / "episodes" / "chunk-000" /
                     "file-000.parquet")
row = em[em["episode_index"] == e].iloc[0]
fi = int(row[f"videos/{NCAM}/file_index"])
off = int(round(float(row[f"videos/{NCAM}/from_timestamp"]) * 10))
vid = CFGDIR / "videos" / NCAM / "chunk-000" / f"file-{fi:03d}.mp4"
frames = []
for j, fr in enumerate(iio.imiter(vid, plugin="pyav")):
    if off <= j < off + L:
        frames.append(np.asarray(fr).copy())
    if j >= off + L:
        break
print(f"pulled {len(frames)} colour frames", flush=True)


def brighten(img):
    f = img.astype(np.float32) / 255.0
    lo, hi = np.percentile(f, 2), np.percentile(f, 99.5)
    f = np.clip((f - lo) / max(hi - lo, 1e-3), 0, 1) ** 0.85
    m = f.mean(axis=-1, keepdims=True)
    return np.clip(m + (f - m) * 1.18, 0, 1)


H, W = frames[0].shape[:2]
S = H
x0 = int(np.clip(W // 2 - S // 2, 0, W - S))
fig = plt.figure(figsize=(9.0, 4.0), dpi=110)
axv = fig.add_axes([0.005, 0.02, 0.44, 0.96])
axf = fig.add_axes([0.535, 0.17, 0.445, 0.72])
writer = imageio.get_writer(HERE / "video" / OUTNAME, fps=20, codec="libx264",
                            quality=8, macro_block_size=1,
                            ffmpeg_params=["-pix_fmt", "yuv420p"])
n = min(len(frames), L)
for t in range(n):
    axv.clear()
    axv.imshow(brighten(frames[t][:, x0:x0 + S]))
    axv.set_xticks([]); axv.set_yticks([]); axv.axis("off")

    axf.clear()
    axf.plot(np.arange(L), fm_ep, color="0.15", lw=1.3, label="wrist sensor")
    axf.plot(ts, f_pred, color=BLUE, lw=1.8, label="read from frozen latent")
    axf.axvline(onset, color=ORANGE, lw=1.2, ls="--")
    axf.axvline(t, color="0.45", lw=1.4)
    axf.scatter([t], [fm_ep[t]], s=26, color="0.15", zorder=5)
    if t >= 15:
        axf.scatter([t], [f_pred[t - 15]], s=26, color=BLUE, zorder=5)
    axf.set_xlim(0, L - 1)
    axf.set_ylim(min(-1, fm_ep.min() - 2), max(fm_ep.max(), f_pred.max()) * 1.15)
    axf.set_xlabel("time step (10 Hz)", fontsize=10)
    axf.set_ylabel("‖F‖ (N)", fontsize=10)
    axf.legend(frameon=False, fontsize=9, loc="upper left")
    fig.canvas.draw()
    writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
writer.close()
plt.close(fig)
p = HERE / "video" / OUTNAME
print(f"wrote {p.name} ({p.stat().st_size // 1024} KB, {n} frames)")
