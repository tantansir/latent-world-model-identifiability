"""Input-matched recovery references (Codex review M1 / E03).

Supervised GRU estimators of (log m, gamma, log k) trained on probe_tr windows and evaluated on probe_te windows -- the
same episodes, 16-frame contexts and window classes as the latent probes (train.py probe_windows) -- under three input
sets:
  V+A        : object centroid and centroid velocity derived from the images alone (the finger disk is located from the
               image itself: pixels > 0.9), plus actions
  V+P+T+A    : the above plus finger proprioception and the 7-d touch vector (everything the VFX model receives)
  state+A    : privileged simulator object state plus finger, touch and actions (the paper's state certificate inputs)
Window classes: all / contact (>= 2 contact frames) / natural glide (no contact, mean speed > 0.15). A linear (ridge)
estimator on the flattened window is reported alongside the GRU. Usage: python cert_matched.py
"""
import sys, json, pathlib
import numpy as np, torch, torch.nn as nn
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from train import load_split, ridge_fit_eval, r2, T_CTX, RESULTS
device = "cuda"; DT = 0.05


def windows(state, n):
    t0s = np.array([0, 8, 16, 24, 32, 40]); ep = np.repeat(np.arange(n), len(t0s)); t0 = np.tile(t0s, n)
    contact = np.array([state["touch"][e, t:t + T_CTX, 6].sum() for e, t in zip(ep, t0)])
    speed = np.array([np.linalg.norm(state["obj"][e, t:t + T_CTX, 2:], axis=1).mean() for e, t in zip(ep, t0)])
    return ep, t0, contact, speed


def image_centroids(frames):
    """frames [N,T,64,64] in 0..1 -> object centroid [N,T,2] using the image only: the finger renders at 1.0, the object
    at ~0.55; the finger disk (from pixels > 0.9, radius 0.10 around its centroid) is excluded before taking the
    0.3-0.8 intensity band."""
    lin = (np.arange(64) + 0.5) / 64; yy, xx = np.meshgrid(lin, lin, indexing="ij")
    fm = (frames > 0.9).astype(np.float32); ft = fm.sum((-2, -1)) + 1e-8
    fx = (fm * xx).sum((-2, -1)) / ft; fy = (fm * yy).sum((-2, -1)) / ft
    mask = (frames > 0.3) & (frames < 0.8)
    mask &= ((xx[None, None] - fx[..., None, None]) ** 2 + (yy[None, None] - fy[..., None, None]) ** 2) > 0.10 ** 2
    w = mask.astype(np.float32); tot = w.sum((-2, -1)) + 1e-8
    c = np.stack([(w * xx).sum((-2, -1)) / tot, (w * yy).sum((-2, -1)) / tot], -1)
    c[tot < 5] = np.nan
    sm = c.copy(); sm[:, 1:-1] = (c[:, :-2] + c[:, 1:-1] + c[:, 2:]) / 3.0
    return sm


def feats(img, state, ep, t0):
    fr = np.stack([img[e, t:t + T_CTX] for e, t in zip(ep, t0)]) / 255.0
    cen = image_centroids(fr)
    v = np.zeros_like(cen); v[:, 1:-1] = (cen[:, 2:] - cen[:, :-2]) / (2 * DT); v[:, 0], v[:, -1] = v[:, 1], v[:, -2]
    bad = np.isnan(cen).any((1, 2)); cen, v = np.nan_to_num(cen), np.nan_to_num(v)
    fin = np.stack([state["finger"][e, t:t + T_CTX] for e, t in zip(ep, t0)])
    tou = np.stack([state["touch"][e, t:t + T_CTX] for e, t in zip(ep, t0)])
    act = np.stack([state["act"][e, t:t + T_CTX] for e, t in zip(ep, t0)])
    obj = np.stack([state["obj"][e, t:t + T_CTX] for e, t in zip(ep, t0)])
    sets = {"V+A": np.concatenate([cen, v, act], -1), "V+P+T+A": np.concatenate([cen, v, fin, tou, act], -1),
            "state+A": np.concatenate([obj, fin, tou, act], -1)}
    return {k: x.astype(np.float32) for k, x in sets.items()}, bad


def gru_probe(Xtr, ytr, Xte, yte, iters=4000):
    cut = int(len(Xtr) * 0.85); f = Xtr[:cut].reshape(-1, Xtr.shape[-1]); mu, sd = f.mean(0), f.std(0) + 1e-6
    ymu, ysd = ytr[:cut].mean(0), ytr[:cut].std(0) + 1e-6
    xtr = torch.tensor((Xtr[:cut] - mu) / sd, device=device); ytr_ = torch.tensor((ytr[:cut] - ymu) / ysd, device=device, dtype=torch.float32)
    xva = torch.tensor((Xtr[cut:] - mu) / sd, device=device); yva = ytr[cut:]; xte = torch.tensor((Xte - mu) / sd, device=device)
    class G(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__(); self.gru = nn.GRU(d_in, 96, num_layers=2, batch_first=True); self.head = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, d_out))
        def forward(self, x):
            hh, _ = self.gru(x); return self.head(torch.cat([hh[:, -1], hh.mean(1)], -1))
    torch.manual_seed(0); net = G(Xtr.shape[-1], ytr.shape[1]).to(device); o = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    best, best_pred = -1e9, None
    for i in range(iters):
        idx = torch.randint(0, len(xtr), (256,), device=device); l = (net(xtr[idx]) - ytr_[idx]).pow(2).mean(); o.zero_grad(); l.backward(); o.step()
        if i % 200 == 0:
            with torch.no_grad():
                rv = r2(yva, net(xva).cpu().numpy() * ysd + ymu).mean()
                if rv > best: best, best_pred = rv, net(xte).cpu().numpy() * ysd + ymu
    return r2(yte, best_pred)


if __name__ == "__main__":
    img_ptr, st_ptr = load_split("probe_tr"); img_pte, st_pte = load_split("probe_te")
    lab = lambda st, ep: np.stack([np.log(st["props"][ep, 0]), st["props"][ep, 1], np.log(st["props"][ep, 2])], 1)
    ep_a, t0_a, c_a, sp_a = windows(st_ptr, img_ptr.shape[0]); ep_b, t0_b, c_b, sp_b = windows(st_pte, img_pte.shape[0])
    Fa, bad_a = feats(img_ptr, st_ptr, ep_a, t0_a); Fb, bad_b = feats(img_pte, st_pte, ep_b, t0_b)
    ya, yb = lab(st_ptr, ep_a), lab(st_pte, ep_b)
    classes = {"all": (np.ones(len(ep_a), bool), np.ones(len(ep_b), bool)), "contact": (c_a >= 2, c_b >= 2),
               "glide": ((c_a == 0) & (sp_a > 0.15), (c_b == 0) & (sp_b > 0.15))}
    res = {}
    for cls, (ma, mb) in classes.items():
        ma = ma & ~bad_a; mb = mb & ~bad_b
        res[cls] = {"n_train": int(ma.sum()), "n_test": int(mb.sum())}
        for name in ("V+A", "V+P+T+A", "state+A"):
            g = gru_probe(Fa[name][ma], ya[ma], Fb[name][mb], yb[mb])
            lin, _ = ridge_fit_eval(Fa[name][ma].reshape(ma.sum(), -1), ya[ma], Fb[name][mb].reshape(mb.sum(), -1), yb[mb])
            res[cls][name] = {"gru": [round(float(x), 3) for x in g], "linear": [round(float(x), 3) for x in lin]}
            print(f"[{cls}] {name:8s} GRU  m/g/k {np.round(g, 3)}   linear {np.round(lin, 3)}   (n {ma.sum()}/{mb.sum()})", flush=True)
    json.dump(res, open(RESULTS / "cert_matched.json", "w"), indent=1)
    print("saved", RESULTS / "cert_matched.json")
