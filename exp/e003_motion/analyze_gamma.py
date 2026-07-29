"""Functional-gamma test (e006a): does the model's Delta-16 prediction USE
gamma knowledge on pure-glide windows, even though linear/MLP probes can't
decode gamma as a disentangled scalar?

Compare decoded direct-16 predictions against analytic baselines that know:
  (a) the true gamma  : p31 = p15 + v15 * (1 - exp(-g*T)) / g   (pure drag)
  (b) the median gamma: same with g = median over dataset
  (c) static          : p31 = p15
Model error near (a) => functional gamma (entangled); near (b)/(c) => absent.
Also regress the model's implied displacement factor against true gamma.
"""
import sys, pathlib
import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import model as M
from model import XJEPA
from train import (load_split, whiten_stats, make_batch, ridge_fit_eval,
                   T_CTX, RESULTS)

device = "cuda"
DT16 = 16 * 0.05
_tag_probe = sys.argv[1] if len(sys.argv) > 1 else ""
img_tr, st_tr = load_split("train2" if "_g" in _tag_probe else "train")
stats = whiten_stats(st_tr)
img_ptr, st_ptr = load_split("probe_tr")
img_pte, st_pte = load_split("probe_te")

TAG = sys.argv[1] if len(sys.argv) > 1 else "VFX_s0_l0.005"
if "_fl" in TAG:
    import train as TR
    from flownet import FlowNet
    fn = FlowNet().to(device)
    fn.load_state_dict(torch.load(RESULTS / "flownet.pt", weights_only=True))
    fn.eval()
    TR.FLOW_NET = fn
in_ch = 4 if "_fl" in TAG else 2
if "_ls" in TAG:
    import train as TR
    TR.LOGSPEED = True
    in_ch += 1
model = XJEPA(TAG.split("_")[0], in_ch=in_ch).to(device)
model.load_state_dict(torch.load(RESULTS / f"{TAG}.pt", weights_only=True))
model.eval()
print(f"checkpoint: {TAG}", flush=True)


@torch.no_grad()
def frame_z(img, state, ep, t0, bs=256):
    zs = []
    for s in range(0, len(ep), bs):
        b = make_batch(img, state, stats, ep[s:s + bs], t0[s:s + bs], device, T=T_CTX)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"], b["prop"], b["touch"])
        zs.append(z.float().cpu().numpy())
    return np.concatenate(zs)


# per-frame position probe fit on probe_tr (same protocol as train.py)
n_tr = img_ptr.shape[0]
t0s = np.array([0, 8, 16, 24, 32, 40])
ep = np.repeat(np.arange(n_tr), len(t0s))
t0 = np.tile(t0s, n_tr)
Z = frame_z(img_ptr, st_ptr, ep, t0)
pos = np.stack([st_ptr["obj"][e, t:t + T_CTX, :2] for e, t in zip(ep, t0)])
Zf, posf = Z.reshape(-1, M.D), pos.reshape(-1, 2)
sub = np.random.default_rng(0).choice(len(Zf), 60000, replace=False)
_, (w, mu, sd, ymu, ysd) = ridge_fit_eval(Zf[sub], posf[sub], Zf[:100], posf[:100])

# dedicated glide eval set: simulate fresh episodes (no render), select those
# containing a pure-glide window (no contact in [t0,t0+31], moving, off-wall),
# then render only the selected episodes
sys.path.insert(0, str(HERE.parent / "e001_xmodal_jepa"))
import pokeworld as pw

sim = pw.simulate(8000, 64, seed=777)
cands_e, cands_t = [], []
for e in range(8000):
    touch_bin = sim["touch"][e, :, 6]
    obj = sim["obj"][e]
    for t0 in range(0, 64 - 32 + 1, 4):
        seg = slice(t0, t0 + 32)
        if touch_bin[seg].sum() > 0:
            continue
        if np.linalg.norm(obj[t0 + T_CTX - 1, 2:]) < 0.2:
            continue
        if not ((obj[seg, :2] > 0.12) & (obj[seg, :2] < 0.88)).all():
            continue
        cands_e.append(e)
        cands_t.append(t0)
        break
sel = np.array(cands_e)
t0w = np.array(cands_t)
print(f"pure-glide windows: {len(sel)} / 8000 episodes", flush=True)
img_g = pw.render(sim["finger"][sel], sim["obj"][sel])
st_g = {k: v[sel] for k, v in sim.items()}
idx = np.arange(len(sel))

# model direct-16 prediction
preds = []
with torch.no_grad():
    for s in range(0, len(idx), 256):
        ii, tt = idx[s:s + 256], t0w[s:s + 256]
        b = make_batch(img_g, st_g, stats, ii, tt, device, T=T_CTX)
        with torch.autocast("cuda", torch.bfloat16):
            z_ctx = model.encode(b["img"], b["prop"], b["touch"])
            h = model.pred.trunk(z_ctx.float(), b["act"])
        aw = np.stack([st_g["act"][e, t + T_CTX - 1:t + T_CTX - 1 + 16]
                       for e, t in zip(ii, tt)])
        aw = torch.from_numpy(aw).to(device).float()
        zd16 = model.dheads["16"](h[:, -1:].float(),
                                  aw.reshape(len(ii), 1, 32))[:, 0].float().cpu().numpy()
        preds.append(((zd16 - mu) / sd) @ w * ysd + ymu)
pred_model = np.concatenate(preds)

p15 = np.stack([st_g["obj"][e, t + T_CTX - 1, :2] for e, t in zip(idx, t0w)])
v15 = np.stack([st_g["obj"][e, t + T_CTX - 1, 2:] for e, t in zip(idx, t0w)])
p31 = np.stack([st_g["obj"][e, t + T_CTX - 1 + 16, :2] for e, t in zip(idx, t0w)])
g_true = st_g["props"][idx, 1]
g_med = np.median(st_g["props"][:, 1])
g_med = np.median(st_pte["props"][:, 1])

def glide(p, v, g):
    return p + v * ((1 - np.exp(-g * DT16)) / g)[:, None]

err = lambda p: np.linalg.norm(p - p31, axis=1)
e_model = err(pred_model)
e_gtrue = err(glide(p15, v15, g_true))
e_gmed = err(glide(p15, v15, np.full(len(idx), g_med)))
e_static = err(p15)
print(f"direct16 err  : model {e_model.mean():.4f} | gamma-true {e_gtrue.mean():.4f} "
      f"| gamma-median {e_gmed.mean():.4f} | static {e_static.mean():.4f}", flush=True)

# implied displacement factor vs true gamma: factor = |p_pred - p15| / |v15| ;
# invert f = (1-exp(-gT))/g numerically to get implied gamma correlation
disp = np.linalg.norm(pred_model - p15, axis=1)
speed = np.linalg.norm(v15, axis=1)
f_impl = np.clip(disp / np.maximum(speed, 1e-6), 0.05, 0.9)
f_true = (1 - np.exp(-g_true * DT16)) / g_true
ok = speed > 0.25
corr = np.corrcoef(f_impl[ok], f_true[ok])[0, 1]
print(f"implied-vs-true displacement factor corr (n={ok.sum()}): {corr:.3f}", flush=True)
r2f = 1 - ((f_impl[ok] - f_true[ok]) ** 2).sum() / ((f_true[ok] - f_true[ok].mean()) ** 2).sum()
print(f"implied factor R2 vs true factor: {r2f:.3f}", flush=True)
