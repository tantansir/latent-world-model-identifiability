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
import model as M
M.ACORR = "_ac" in TAG      # corrected action windows for _ac checkpoints
M.IHEAD = "_ih" in TAG      # interaction prediction heads for _ih checkpoints
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
if "_obs" in TAG:
    import train as TR
    TR.OBSIN = True
if "_cv" in TAG:
    import train as TR
    TR.OBSVEL = "cent"
model = XJEPA(TAG.split("_")[0], in_ch=in_ch, vr=("_vr" in TAG), vr_flow=("_vrf" in TAG or ("_fl" in TAG and "_vv" in TAG)),
              mul=("_mul" in TAG), vv=("_vv" in TAG),
              oracle=("_or" in TAG), objin=("_oi" in TAG or "_obs" in TAG), vrcond=("_vc" in TAG),
              vr16=("_vr16" in TAG), physhead=("_ph" in TAG),
              phys_mode=("fixg" if "_phfixg" in TAG else "truev" if "_phtruev" in TAG else "multi" if "_phmulti" in TAG else "full"),
              dispmlp=("_dm" in TAG)).to(device)
model.load_state_dict(torch.load(RESULTS / f"{TAG}.pt", weights_only=True), strict=False)
model.eval()
print(f"checkpoint: {TAG}", flush=True)


@torch.no_grad()
def frame_z(img, state, ep, t0, bs=256):
    zs = []
    for s in range(0, len(ep), bs):
        b = make_batch(img, state, stats, ep[s:s + bs], t0[s:s + bs], device, T=T_CTX)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"], b["prop"], b["touch"], b["obj"])
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

EVAL_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 777
N_SIM, N_BIN, PER_BIN = 48000, 7, 400
# GLIDE_FRAC=0 (default): episodes from the training generator (in-distribution; high-gamma bins are thin
# because a gamma>2.5 object stops within ~10 frames of release, so 32 contact-free frames with motion at
# the context end barely exist). GLIDE_FRAC=1: all-glide episodes (finger parked in a corner) - every bin
# is full but the finger configuration is out of distribution for models trained on the plain split.
import os
GLIDE_FRAC = float(os.environ.get("GLIDE_FRAC", "0"))
# gamma-stratified glide set. The old criterion (speed > 0.2 at the context end, 32 contact-free frames)
# could not be met by high-gamma objects (v decays by exp(-0.75*gamma) before the context ends), so the
# selected set had mean gamma 1.03 against a prior mean of 2.25. Now: all-glide episodes (object launched,
# finger parked), speed > 0.05 at the context end, the two analytic predictions (true vs median gamma)
# differ by > 0.005 (so gamma matters for the window), and an equal number of windows per gamma bin.
sim = pw.simulate(N_SIM, 64, seed=EVAL_SEED, glide_frac=GLIDE_FRAC)
print(f"eval set seed {EVAL_SEED} glide_frac {GLIDE_FRAC}", flush=True)
g_med = np.median(st_pte["props"][:, 1])
cands_e, cands_t = [], []
for e in range(N_SIM):
    touch_bin = sim["touch"][e, :, 6]
    obj = sim["obj"][e]
    g = sim["props"][e, 1]
    for t0 in range(0, 64 - 32 + 1, 4):
        seg = slice(t0, t0 + 32)
        if touch_bin[seg].sum() > 0:
            continue
        v = obj[t0 + T_CTX - 1, 2:]
        if np.linalg.norm(v) < 0.05:
            continue
        if not ((obj[seg, :2] > 0.12) & (obj[seg, :2] < 0.88)).all():
            continue
        f_true, f_med = (1 - np.exp(-g * DT16)) / g, (1 - np.exp(-g_med * DT16)) / g_med
        if np.linalg.norm(v) * abs(f_true - f_med) < 0.005:
            continue
        cands_e.append(e)
        cands_t.append(t0)
        break
cands_e, cands_t = np.array(cands_e), np.array(cands_t)
edges = np.linspace(0.5, 4.0, N_BIN + 1)
gbin_all = np.clip(np.digitize(sim["props"][cands_e, 1], edges) - 1, 0, N_BIN - 1)
rng = np.random.default_rng(EVAL_SEED)
keep = []
for b in range(N_BIN):
    ii = np.where(gbin_all == b)[0]
    keep.append(rng.permutation(ii)[:PER_BIN])
    print(f"  gamma bin {b} [{edges[b]:.1f},{edges[b+1]:.1f}): {len(ii)} candidates, keep {min(len(ii), PER_BIN)}", flush=True)
keep = np.concatenate(keep)
sel, t0w = cands_e[keep], cands_t[keep]
print(f"pure-glide windows: {len(sel)} stratified from {len(cands_e)} candidates / {N_SIM} episodes; "
      f"gamma mean {sim['props'][sel, 1].mean():.2f} (prior 2.25)", flush=True)
img_g = pw.render(sim["finger"][sel], sim["obj"][sel])
st_g = {k: v[sel] for k, v in sim.items()}
idx = np.arange(len(sel))

# model direct-16 prediction
preds = []
phys_preds, phys_g, phys_v, phys_d12, disp_preds = [], [], [], [], []
with torch.no_grad():
    for s in range(0, len(idx), 256):
        ii, tt = idx[s:s + 256], t0w[s:s + 256]
        b = make_batch(img_g, st_g, stats, ii, tt, device, T=T_CTX)
        with torch.autocast("cuda", torch.bfloat16):
            z_ctx = model.encode(b["img"], b["prop"], b["touch"], b["obj"])
            pc = model.props_w(b) if model.use_oracle else None
            h = model.pred.trunk(z_ctx.float(), b["act"], pc)
        aw = np.stack([st_g["act"][e, t + M.act_offset(T_CTX):t + M.act_offset(T_CTX) + 16]
                       for e, t in zip(ii, tt)])
        aw = torch.from_numpy(aw).to(device).float()
        h = model.condition(h, torch.cat([b["act"], aw[:, 1:]], 1))
        zd16 = model.dheads["16"](h[:, -1:].float(),
                                  aw.reshape(len(ii), 1, 32))[:, 0].float().cpu().numpy()
        preds.append(((zd16 - mu) / sd) @ w * ysd + ymu)
        if model.use_phys:
            gh, vh = model.phys_params(h[:, -1].float(), b["obj"][:, -1].float())
            dh = vh * (1 - torch.exp(-gh * 0.8)) / gh
            phys_preds.append(dh.cpu().numpy()); phys_g.append(gh[:, 0].cpu().numpy())
            phys_v.append(vh.cpu().numpy())
            phys_d12.append((vh * (1 - torch.exp(-gh * 0.6)) / gh).cpu().numpy())
        if model.use_disp:
            disp_preds.append(model.head_disp(h[:, -1].float()).cpu().numpy())
pred_phys = np.concatenate(phys_preds) if phys_preds else None
pred_model = np.concatenate(preds)

p15 = np.stack([st_g["obj"][e, t + T_CTX - 1, :2] for e, t in zip(idx, t0w)])
v15 = np.stack([st_g["obj"][e, t + T_CTX - 1, 2:] for e, t in zip(idx, t0w)])
p31 = np.stack([st_g["obj"][e, t + T_CTX - 1 + 16, :2] for e, t in zip(idx, t0w)])
g_true = st_g["props"][idx, 1]

def glide(p, v, g):
    return p + v * ((1 - np.exp(-g * DT16)) / g)[:, None]

err = lambda p: np.linalg.norm(p - p31, axis=1)
e_model = err(pred_model)
e_gtrue = err(glide(p15, v15, g_true))
e_gmed = err(glide(p15, v15, np.full(len(idx), g_med)))
e_static = err(p15)
print(f"direct16 err  : model {e_model.mean():.4f} | gamma-true {e_gtrue.mean():.4f} "
      f"| gamma-median {e_gmed.mean():.4f} | static {e_static.mean():.4f}", flush=True)
gbin = np.clip(np.digitize(g_true, edges) - 1, 0, N_BIN - 1)
print("per gamma bin  : model | gamma-true | gamma-median | static   (n)", flush=True)
for b in range(N_BIN):
    m = gbin == b
    if m.sum() == 0:
        continue
    print(f"  [{edges[b]:.1f},{edges[b+1]:.1f}) {e_model[m].mean():.4f} | {e_gtrue[m].mean():.4f} | {e_gmed[m].mean():.4f} | {e_static[m].mean():.4f}   ({m.sum()})", flush=True)
# Per-window position errors carry a 2-4 px readout floor (linear readout applied to predicted latents), larger
# than the gamma-dependent displacement difference (~1.4 px at 16 steps). Average it out per gamma bin: project
# the predicted displacement onto the velocity direction, divide by |v15| -> implied displacement factor
# f_hat = (p_hat - p15).v / |v|^2, whose bin mean is compared with f_true(gamma) and the median-gamma factor.
f_hat = ((pred_model - p15) * v15).sum(1) / np.maximum((v15 ** 2).sum(1), 1e-6)
f_true = (1 - np.exp(-g_true * DT16)) / g_true
f_medc = (1 - np.exp(-g_med * DT16)) / g_med
print("per gamma bin  : implied factor f_hat (mean ± sem) | f_true mean | f_median   [f = displacement / v15; pure glide]", flush=True)
rows = []
for b in range(N_BIN):
    m = gbin == b
    if m.sum() < 3:
        continue
    rows.append((f_hat[m].mean(), f_true[m].mean()))
    print(f"  [{edges[b]:.1f},{edges[b+1]:.1f}) {f_hat[m].mean():.3f} ± {f_hat[m].std()/np.sqrt(m.sum()):.3f} | {f_true[m].mean():.3f} | {f_medc:.3f}   ({m.sum()})", flush=True)
rows = np.array(rows)
sl = np.polyfit(rows[:, 1], rows[:, 0], 1)[0] if len(rows) > 2 else float("nan")
print(f"bin-mean slope of f_hat on f_true: {sl:.2f}  (1 = full functional use of gamma, 0 = none)", flush=True)
# Confound: gamma bins differ in speed at the context end (high gamma -> slow). A low-speed dead zone would
# lower f_hat at high gamma without any gamma use. Two controls:
#  (a) partial regression f_hat ~ a + b f_true + c log|v15| + d |v15|  -> b is the gamma effect at fixed speed
#  (b) speed-matched bins: windows with |v15| in a common band, per gamma bin
spd = np.linalg.norm(v15, axis=1)
X = np.stack([np.ones_like(f_true), f_true, np.log(spd), spd], 1)
beta, res_, *_ = np.linalg.lstsq(X, f_hat, rcond=None)
resid = f_hat - X @ beta
cov = np.linalg.inv(X.T @ X) * resid.var() * len(f_hat) / (len(f_hat) - X.shape[1])
print(f"partial regression: f_hat = {beta[0]:.3f} + {beta[1]:.2f}*f_true (se {np.sqrt(cov[1,1]):.2f}) + {beta[2]:.3f}*log|v| + {beta[3]:.3f}*|v|", flush=True)
Xs = np.stack([np.ones_like(f_true), np.log(spd), spd], 1)
b2 = np.linalg.lstsq(Xs, f_hat, rcond=None)[0]
print(f"  R2 of f_hat: speed only {1 - (f_hat - Xs @ b2).var() / f_hat.var():.3f}, speed + f_true {1 - resid.var() / f_hat.var():.3f}", flush=True)
band = (spd > 0.06) & (spd < 0.15)
print("speed-matched bins (|v15| in [0.06, 0.15)): f_hat mean ± sem | f_true | n", flush=True)
for b in range(N_BIN):
    m = (gbin == b) & band
    if m.sum() >= 10:
        print(f"  [{edges[b]:.1f},{edges[b+1]:.1f}) {f_hat[m].mean():.3f} ± {f_hat[m].std()/np.sqrt(m.sum()):.3f} | {f_true[m].mean():.3f} | {m.sum()}  (mean |v| {spd[m].mean():.3f})", flush=True)
# where on the true<->median axis does the model sit? 0 = as good as true gamma, 1 = as bad as median gamma
pos = (e_model.mean() - e_gtrue.mean()) / max(e_gmed.mean() - e_gtrue.mean(), 1e-9)
print(f"model position on true(0)..median(1) axis: {pos:.2f}", flush=True)

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

if pred_phys is not None:
    e_phys = err(p15 + pred_phys)
    g_hat = np.concatenate(phys_g)
    r2g = 1 - ((g_hat - g_true) ** 2).sum() / ((g_true - g_true.mean()) ** 2).sum()
    print(f"physics head : glide err {e_phys.mean():.4f} | gamma_hat R2 vs true {r2g:.3f} "
          f"| corr {np.corrcoef(g_hat, g_true)[0, 1]:.3f}", flush=True)
    v_hat = np.concatenate(phys_v)
    print(f"physics head : v_hat R2 vs true v15 {1 - ((v_hat - v15) ** 2).sum() / ((v15 - v15.mean(0)) ** 2).sum():.3f} "
          f"| mean gamma_hat {g_hat.mean():.2f} (true mean {g_true.mean():.2f}, sd {g_true.std():.2f} / hat sd {g_hat.std():.2f})", flush=True)
    # untrained horizon Delta=12: same (gamma_hat, v_hat), analytic integral
    p27 = np.stack([st_g["obj"][e, t + T_CTX - 1 + 12, :2] for e, t in zip(idx, t0w)])
    e12 = np.linalg.norm(p15 + np.concatenate(phys_d12) - p27, axis=1)
    e12_med = np.linalg.norm(glide(p15, v15, np.full(len(idx), g_med)) * 0 + p15 + v15 * ((1 - np.exp(-g_med * 0.6)) / g_med) - p27, axis=1)
    e12_true = np.linalg.norm(p15 + v15 * ((1 - np.exp(-g_true * 0.6)) / g_true)[:, None] - p27, axis=1)
    print(f"physics head @Delta=12 (untrained): err {e12.mean():.4f} | median-gamma {e12_med.mean():.4f} | true-gamma {e12_true.mean():.4f}", flush=True)
if disp_preds:
    e_disp = err(p15 + np.concatenate(disp_preds))
    print(f"mlp disp head: glide err {e_disp.mean():.4f}", flush=True)
