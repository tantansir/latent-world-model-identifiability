"""Recovery references for drag on the pure-glide evaluation set (the windows of probe_targets.py / analyze_gamma.py,
seed 778): how well can drag be estimated from the observations themselves on these windows?
  structured estimator: gamma_hat = -median over the context of log(|v_{t+1}| / |v_t|) / dt, from image-derived speeds
                        (object centroid after removing the finger disk located in the image) and from simulator speeds
  GRU (2 layers, width 96) on the 16-step context, 5-fold cross-validation over episodes (one window per episode),
                        early stopping on 15% of each training fold, with the VFX inputs (image-derived object
                        trajectory, finger proprioception, touch, actions) and with simulator object state instead
Usage: python glide_recovery.py [seed] [--only-v]   -> results/glide_recovery.json (merged)"""
import sys, json, pathlib
import numpy as np, torch, torch.nn as nn
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "e001_xmodal_jepa"))
import pokeworld as pw
T_CTX, DT, DT16 = 16, 0.05, 0.8
EVAL_SEED = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 778
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- identical window selection to probe_targets.py
sim = pw.simulate(48000, 64, seed=EVAL_SEED, glide_frac=1.0)
pte = np.load(HERE.parent / "e001_xmodal_jepa" / "data" / "probe_te_state.npz")
g_med = np.median(pte["props"][:, 1])
cands_e, cands_t = [], []
for e in range(48000):
    tb = sim["touch"][e, :, 6]; obj = sim["obj"][e]; g = sim["props"][e, 1]
    for t0 in range(0, 33, 4):
        seg = slice(t0, t0 + 32)
        if tb[seg].sum() > 0: continue
        v = obj[t0 + T_CTX - 1, 2:]
        if np.linalg.norm(v) < 0.05: continue
        if not ((obj[seg, :2] > 0.12) & (obj[seg, :2] < 0.88)).all(): continue
        ft, fm = (1 - np.exp(-g * DT16)) / g, (1 - np.exp(-g_med * DT16)) / g_med
        if np.linalg.norm(v) * abs(ft - fm) < 0.005: continue
        cands_e.append(e); cands_t.append(t0); break
cands_e, cands_t = np.array(cands_e), np.array(cands_t)
edges = np.linspace(0.5, 4.0, 8)
gb = np.clip(np.digitize(sim["props"][cands_e, 1], edges) - 1, 0, 6)
rng = np.random.default_rng(EVAL_SEED)
keep = np.concatenate([rng.permutation(np.where(gb == b)[0])[:400] for b in range(7)])
sel, t0w = cands_e[keep], cands_t[keep]
img = pw.render(sim["finger"][sel], sim["obj"][sel])
st = {k: v[sel] for k, v in sim.items()}
n = len(sel); print(f"{n} pure-glide windows", flush=True)


def image_centroids(frames):
    """frames [N,T,64,64] in 0..1 -> object centroid [N,T,2]; the finger (rendered at 1.0) is located from the
    image and its disk removed before taking the object's intensity band (as in cert_matched.py)."""
    lin = (np.arange(64) + 0.5) / 64; yy, xx = np.meshgrid(lin, lin, indexing="ij")
    fm = (frames > 0.9).astype(np.float32); ft = fm.sum((-2, -1)) + 1e-8
    fx = (fm * xx).sum((-2, -1)) / ft; fy = (fm * yy).sum((-2, -1)) / ft
    mask = (frames > 0.3) & (frames < 0.8)
    mask &= ((xx[None, None] - fx[..., None, None]) ** 2 + (yy[None, None] - fy[..., None, None]) ** 2) > 0.10 ** 2
    w = mask.astype(np.float32); tot = w.sum((-2, -1)) + 1e-8
    c = np.stack([(w * xx).sum((-2, -1)) / tot, (w * yy).sum((-2, -1)) / tot], -1)
    sm = c.copy(); sm[:, 1:-1] = (c[:, :-2] + c[:, 1:-1] + c[:, 2:]) / 3.0
    return sm


fr = np.stack([img[i, t:t + T_CTX] for i, t in enumerate(t0w)]) / 255.0
cen = image_centroids(fr)
vel = np.zeros_like(cen); vel[:, 1:-1] = (cen[:, 2:] - cen[:, :-2]) / (2 * DT); vel[:, 0], vel[:, -1] = vel[:, 1], vel[:, -2]
gsl = lambda k: np.stack([st[k][i, t:t + T_CTX] for i, t in enumerate(t0w)])
obj_state = gsl("obj")
X = {"VFX inputs (image-derived)": np.concatenate([cen, vel, gsl("finger"), gsl("touch"), gsl("act")], -1).astype(np.float32),
     "simulator object state": np.concatenate([obj_state, gsl("finger"), gsl("touch"), gsl("act")], -1).astype(np.float32),
     "V inputs (image-derived object trajectory, actions)": np.concatenate([cen, vel, gsl("act")], -1).astype(np.float32)}
ONLY = [k for k in X if "V inputs" in k] if "--only-v" in sys.argv else list(X)
y = st["props"][:, 1].astype(np.float32)
r2 = lambda yt, yp: float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
res = {"n_windows": int(n)}


def structured(speed):
    """gamma_hat = -median log speed ratio / dt over the context (speeds floored to avoid log 0)"""
    lr = np.log(np.maximum(speed[:, 1:], 1e-4) / np.maximum(speed[:, :-1], 1e-4))
    return -np.median(lr, 1) / DT


res["structured, image-derived speed"] = r2(y, structured(np.linalg.norm(vel[:, 1:-1], axis=-1)))
res["structured, simulator speed"] = r2(y, structured(np.linalg.norm(obj_state[..., 2:], axis=-1)))
print({k: round(v, 3) for k, v in res.items() if k.startswith("structured")}, flush=True)


class G(nn.Module):
    def __init__(self, d_in):
        super().__init__(); self.gru = nn.GRU(d_in, 96, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, x):
        hh, _ = self.gru(x); return self.head(torch.cat([hh[:, -1], hh.mean(1)], -1))[:, 0]


def gru_cv(Xa, k=5, iters=4000):
    perm = np.random.default_rng(0).permutation(n); folds = np.array_split(perm, k); pred = np.zeros(n)
    for f in range(k):
        te = folds[f]; tr_all = np.setdiff1d(perm, te); cut = int(len(tr_all) * 0.85)
        tr, va = tr_all[:cut], tr_all[cut:]
        mu, sd = Xa[tr].reshape(-1, Xa.shape[-1]).mean(0), Xa[tr].reshape(-1, Xa.shape[-1]).std(0) + 1e-6
        ymu, ysd = y[tr].mean(), y[tr].std() + 1e-6
        T = lambda a: torch.tensor((Xa[a] - mu) / sd, device=device)
        xtr, xva, xte = T(tr), T(va), T(te); ytr = torch.tensor((y[tr] - ymu) / ysd, device=device)
        torch.manual_seed(f); net = G(Xa.shape[-1]).to(device); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
        best, best_pred = -1e9, None
        for i in range(iters):
            idx = torch.randint(0, len(xtr), (256,), device=device)
            loss = (net(xtr[idx]) - ytr[idx]).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if i % 200 == 0:
                with torch.no_grad():
                    rv = r2(y[va], net(xva).cpu().numpy() * ysd + ymu)
                    if rv > best: best, best_pred = rv, net(xte).cpu().numpy() * ysd + ymu
        pred[te] = best_pred
    return r2(y, pred)


for name in ONLY:
    res[f"GRU, {name}"] = gru_cv(X[name])
    print(name, round(res[f"GRU, {name}"], 3), flush=True)
out = HERE / "results" / "glide_recovery.json"
if out.exists():                                   # merge: keep entries computed earlier
    old = json.load(open(out)); old.update(res); res = old
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
