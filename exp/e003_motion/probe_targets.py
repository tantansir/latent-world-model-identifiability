"""Product-encoding test: from the trunk hidden state at the context end (pure-glide windows, the same
gamma-stratified all-glide set as analyze_gamma.py, GLIDE_FRAC=1), ridge-probe
  gamma, f(gamma) = (1-exp(-0.8 gamma))/gamma, |v15|, |v15| f(gamma), and the 2-d glide displacement v15 f(gamma)
5-fold CV R^2.  If the displacement (product) is readable while gamma alone is not, gamma is encoded only
multiplied with velocity - used, not separable.
Usage: GLIDE_FRAC=1 python probe_targets.py TAG [seed]
"""
import os
os.environ.setdefault("GLIDE_FRAC", "1")
from common_eval import *
from common_eval import model, stats, device, T_CTX, make_batch, st_pte, HERE
sys.path.insert(0, str(HERE.parent / "e001_xmodal_jepa"))
import pokeworld as pw
EVAL_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 778
DT16 = 0.8
sim = pw.simulate(48000, 64, seed=EVAL_SEED, glide_frac=1.0)
g_med = np.median(st_pte["props"][:, 1])
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
img_g = pw.render(sim["finger"][sel], sim["obj"][sel]); st_g = {k: v[sel] for k, v in sim.items()}
idx = np.arange(len(sel))
H, Z = [], []
with torch.no_grad():
    for s in range(0, len(idx), 256):
        ii, tt = idx[s:s + 256], t0w[s:s + 256]
        b = make_batch(img_g, st_g, stats, ii, tt, device, T=T_CTX)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"], b["prop"], b["touch"], b["obj"])
            h = model.pred.trunk(z.float(), b["act"], model.props_w(b) if model.use_oracle else None)
        H.append(h[:, -1].float().cpu().numpy()); Z.append(z[:, -1].float().cpu().numpy())
H, Z = np.concatenate(H), np.concatenate(Z)
g = st_g["props"][idx, 1]; v = np.stack([st_g["obj"][e, t + T_CTX - 1, 2:] for e, t in zip(idx, t0w)])
f = (1 - np.exp(-g * DT16)) / g; spd = np.linalg.norm(v, axis=1)
targets = {"gamma": g[:, None], "f(gamma)": f[:, None], "|v|": spd[:, None], "|v| f(gamma)": (spd * f)[:, None],
           "v f(gamma) (2d)": v * f[:, None], "v (2d)": v, "log|v|": np.log(spd)[:, None], "log f(gamma)": np.log(f)[:, None]}
def cv_r2(X, y, k=5):
    """episode-grouped outer folds (one window per episode here, so windows = episodes); alpha chosen on an inner
    split of the outer training fold; standardization fitted on the inner training data; outer test fold scored once"""
    n = len(X); perm = rng.permutation(n); folds = np.array_split(perm, k); out = []
    def fit(Xtr, ytr, Xte, a):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-6
        w = np.linalg.solve(((Xtr - mu) / sd).T @ ((Xtr - mu) / sd) + a * np.eye(X.shape[1]), ((Xtr - mu) / sd).T @ ((ytr - ymu) / ysd))
        return ((Xte - mu) / sd) @ w * ysd + ymu
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        cut = int(len(tr) * 0.8); itr, ite = tr[:cut], tr[cut:]
        best_a, best_r = None, -1e9
        for a in (1.0, 10.0, 100.0, 1000.0):
            p = fit(X[itr], y[itr], X[ite], a); r = (1 - ((p - y[ite]) ** 2).sum(0) / ((y[ite] - y[ite].mean(0)) ** 2).sum(0)).mean()
            if r > best_r: best_a, best_r = a, r
        p = fit(X[tr], y[tr], X[te], best_a)
        out.append((1 - ((p - y[te]) ** 2).sum(0) / ((y[te] - y[te].mean(0)) ** 2).sum(0)).mean())
    return np.mean(out)
def cv_mlp(X, y, k=5, steps=3000):
    """Small MLP probe (192-128-64-out), 5-fold CV R2, early-stop on a held-in slice."""
    n = len(X); perm = rng.permutation(n); folds = np.array_split(perm, k); out = []
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; ymu, ysd = y[tr].mean(0), y[tr].std(0) + 1e-6
        xt = torch.tensor((X[tr] - mu) / sd, device=device, dtype=torch.float32); yt = torch.tensor((y[tr] - ymu) / ysd, device=device, dtype=torch.float32)
        xe = torch.tensor((X[te] - mu) / sd, device=device, dtype=torch.float32)
        cut = int(len(xt) * 0.85)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Linear(64, y.shape[1])).to(device)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); best, state = -1e9, None
        for it in range(steps):
            ii = torch.randint(0, cut, (256,), device=device)
            l = (net(xt[ii]) - yt[ii]).pow(2).mean(); opt.zero_grad(); l.backward(); opt.step()
            if it % 100 == 0:
                with torch.no_grad():
                    pv = net(xt[cut:]); r = 1 - (pv - yt[cut:]).pow(2).sum(0) / (yt[cut:] - yt[cut:].mean(0)).pow(2).sum(0)
                if r.mean().item() > best: best, state = r.mean().item(), {k_: v_.clone() for k_, v_ in net.state_dict().items()}
        net.load_state_dict(state)
        with torch.no_grad(): p = net(xe).cpu().numpy() * ysd + ymu
        out.append(float((1 - ((p - y[te]) ** 2).sum(0) / ((y[te] - y[te].mean(0)) ** 2).sum(0)).mean()))
    return np.mean(out)
print(f"{len(idx)} glide windows; probe from trunk h (dim {H.shape[1]}) and encoder z (dim {Z.shape[1]}), 5-fold CV ridge R2 (alpha chosen on an inner split of the training fold)")
for name, y in targets.items():
    print(f"  {name:16s}  h: {cv_r2(H, y):.3f}   z: {cv_r2(Z, y):.3f}", flush=True)
print("MLP probe (5-fold CV R2):", flush=True)
for name in ("gamma", "f(gamma)", "|v| f(gamma)", "v (2d)"):
    print(f"  {name:16s}  h: {cv_mlp(H, targets[name]):.3f}   z: {cv_mlp(Z, targets[name]):.3f}", flush=True)
