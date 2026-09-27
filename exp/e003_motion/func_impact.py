"""Stiffness intervention test (Codex round 3): timed short push vs retreat, scored on the peak contact force inside
frame 4, using the model's own Delta-4 touch head. Also the readout-ceiling check for the release test: encode the
simulator's true future frames with the frozen encoder and decode position through the same readout.

Starts: fresh episodes (training generator), context ends out of contact with the object within a small gap of the
finger (gap in [0.02, 0.10] arena units), >= 2 contact frames in the context, object off-wall. Two action plans for
4 steps: A1 = push 0.9*unit(obj - finger), A0 = retreat -0.9*unit. Keep starts where the simulator's first contact
under A1 is exactly frame 4 (index 3) and there is no contact under A0. Truth = f_peak inside frame 4 (touch channel 2)
under each plan; prediction = touch head Delta=4 peak channel with the corresponding action window. Reported:
S_k^impact (push branch), S_k^Delta (push - retreat), each as partial slope on log k over the simulator's, controlling
gap, normal/tangential relative velocity, direction, log m, gamma; and E_Delta (relative squared error of the predicted
action effect). Natural continuation S_k^nat: same starts, recorded actions, peak force at frame 4.
Usage: CF_POLICY=closed python func_impact.py TAG
"""
from common_eval import *
from common_eval import model, stats, device, T_CTX, make_batch, HERE, AOFF
import numpy as np, torch, json, os
from fim_certificate import replay
sys.path.insert(0, str(HERE.parent / "e001_xmodal_jepa"))
import pokeworld as pw
EVAL_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 781
N_SIM, NB, PER_BIN, U = 48000, 6, 300, 0.9
touch_mu, touch_sd = stats["touch_mu"], stats["touch_sd"]

sim = pw.simulate(N_SIM, 64, seed=EVAL_SEED)
touch, obj, fin, props, act = sim["touch"], sim["obj"], sim["finger"], sim["props"], sim["act"]
cb = touch[..., 6] > 0.5
W = []
for e in range(N_SIM):
    for t0 in range(0, 64 - T_CTX - 4 + 1, 2):
        tc = t0 + T_CTX - 1
        if cb[e, tc] or cb[e, tc - 1] or cb[e, t0:tc + 1].sum() < 2: continue
        d = obj[e, tc, :2] - fin[e, tc, :2]; gap = np.linalg.norm(d) - 0.15
        if not (0.02 <= gap <= 0.10): continue
        if not ((obj[e, tc, :2] > 0.2) & (obj[e, tc, :2] < 0.8)).all(): continue
        W.append((e, t0)); break
W = np.array(W); ep, t0 = W[:, 0], W[:, 1]; tc = t0 + T_CTX - 1; n = len(ep)
d = obj[ep, tc, :2] - fin[ep, tc, :2]; dist = np.linalg.norm(d, axis=1); nvec = d / dist[:, None]; gap = dist - 0.15
vrel = fin[ep, tc, 2:] - obj[ep, tc, 2:]; vn = (vrel * nvec).sum(1); vt = (vrel * np.c_[-nvec[:, 1], nvec[:, 0]]).sum(1)
lm, gg, lk = np.log(props[ep, 0]), props[ep, 1], np.log(props[ep, 2])
P = lambda pl: (fin[ep, tc, :2].astype(np.float64), fin[ep, tc, 2:].astype(np.float64), obj[ep, tc, :2].astype(np.float64), obj[ep, tc, 2:].astype(np.float64), pl,
                props[ep, 0].astype(np.float64), props[ep, 1].astype(np.float64), props[ep, 2].astype(np.float64))
acts1 = np.repeat((U * nvec)[None], 4, 0); acts0 = -acts1
_, _, tou1 = replay(*P(acts1)); _, _, tou0 = replay(*P(acts0))
onset1 = np.array([next((t for t in range(4) if tou1[t, i, 6] > 0.5), -1) for i in range(n)])
keep = (onset1 == 3) & (tou0[:, :, 6].max(0) < 0.5)
print(f"{n} candidate starts; contact exactly at frame 4 under push and none under retreat: {keep.sum()}", flush=True)
kb = np.clip(np.digitize(lk, np.linspace(np.log(500), np.log(6000), NB + 1)) - 1, 0, NB - 1)
rng = np.random.default_rng(EVAL_SEED)
sel = np.concatenate([rng.permutation(np.where(keep & (kb == b))[0])[:PER_BIN] for b in range(NB)])
print("per log k bin:", [int((keep & (kb == b)).sum()) for b in range(NB)], "-> kept", len(sel), flush=True)
ep, t0, tc, nvec, gap, vn, vt, lm, gg, lk = ep[sel], t0[sel], tc[sel], nvec[sel], gap[sel], vn[sel], vt[sel], lm[sel], gg[sel], lk[sel]
f1_true, f0_true = tou1[3, sel, 2], tou0[3, sel, 2]
img_g = pw.render(sim["finger"][np.unique(ep)], sim["obj"][np.unique(ep)]); remap = {e: i for i, e in enumerate(np.unique(ep))}
st_g = {k: v[np.unique(ep)] for k, v in sim.items()}; epr = np.array([remap[e] for e in ep])
f_nat_true = touch[ep, tc + 4, 2]


HS = []


def touch_pred(plan):
    out = []
    for s in range(0, len(epr), 256):
        ii, tt = epr[s:s + 256], t0[s:s + 256]
        b = make_batch(img_g, st_g, stats, ii, tt, device, T=T_CTX + 4)
        if plan is None:
            a = b["act"]
        else:
            u = torch.from_numpy(plan[s:s + 256]).to(device).float()[:, None].expand(-1, 4, -1)
            a = torch.cat([b["act"][:, :T_CTX], u], 1)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"][:, :T_CTX], b["prop"][:, :T_CTX], b["touch"][:, :T_CTX], b["obj"][:, :T_CTX])
            pc = model.props_w(b) if model.use_oracle else None
            h = model.pred.trunk(z.float(), a[:, :T_CTX], pc); h = model.condition(h, a)
            aw = a[:, AOFF:AOFF + 4].reshape(len(ii), 1, 8).float()
            y = model.dheads_touch["4"](h[:, -1:].float(), aw)[:, 0].float().cpu().numpy()
        if plan is None:
            HS.append(h[:, -1].float().cpu().numpy())
        out.append(y[:, 2] * touch_sd[2] + touch_mu[2])
    return np.concatenate(out)


f1_pred, f0_pred, fnat_pred = touch_pred(U * nvec), touch_pred(-U * nvec), touch_pred(None)


def partial(y, x, nu):
    X = np.c_[np.ones(len(y)), x, nu]; wv, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ wv; se = np.sqrt((r @ r) / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)[1, 1]); return float(wv[1]), float(se)


NU = np.c_[gap, vn, vt, nvec, lm, gg]
res = {}


def grouped_ridge_r2(X, y, g, k=5):
    # readout at the intervention start on these impact contexts (episode-grouped, inner alpha selection), as in func_cf.py
    rs = np.random.default_rng(0); ug = np.unique(g); gf = np.array_split(ug[rs.permutation(len(ug))], k); out = []
    def fit(Xtr, ytr, Xte, a):
        m_, s_ = Xtr.mean(0), Xtr.std(0) + 1e-6; ym, ys = ytr.mean(), ytr.std() + 1e-6
        wv = np.linalg.solve(((Xtr - m_) / s_).T @ ((Xtr - m_) / s_) + a * np.eye(X.shape[1]), ((Xtr - m_) / s_).T @ ((ytr - ym) / ys))
        return ((Xte - m_) / s_) @ wv * ys + ym
    for i in range(k):
        te = np.isin(g, gf[i]); tr = ~te; tg = np.unique(g[tr]); ite = np.isin(g, tg[rs.permutation(len(tg))[:len(tg) // 5]]) & tr; itr = tr & ~ite
        ba, br = None, -1e9
        for a in (1.0, 10.0, 100.0, 1000.0):
            p = fit(X[itr], y[itr], X[ite], a); r = 1 - ((p - y[ite]) ** 2).sum() / ((y[ite] - y[ite].mean()) ** 2).sum()
            if r > br: ba, br = a, r
        p = fit(X[tr], y[tr], X[te], ba); out.append(1 - ((p - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return float(np.mean(out))


H_all = np.concatenate(HS)
res["probe_at_intervention"] = {nm: grouped_ridge_r2(H_all, v, ep) for nm, v in (("log_m", lm), ("gamma", gg), ("log_k", lk))}
print("[readout at the impact start, episode-grouped ridge R2]", {k_: round(v_, 3) for k_, v_ in res["probe_at_intervention"].items()}, flush=True)
for name, yt, yp in (("push", f1_true, f1_pred), ("delta", f1_true - f0_true, f1_pred - f0_pred), ("natural", f_nat_true, fnat_pred)):
    st_, se_t = partial(yt, lk, NU); sp_, se_p = partial(yp, lk, NU)
    e_rel = float(((yp - yt) ** 2).sum() / (yt ** 2).sum()) if name == "delta" else float("nan")
    print(f"[k, {name}] true slope on log k {st_:.3f} ± {se_t:.3f}; pred {sp_:.3f} ± {se_p:.3f}; S_k {sp_ / st_:.2f}" + (f"; E_delta {e_rel:.3f}" if name == "delta" else ""), flush=True)
    res[name] = dict(true=st_, true_se=se_t, pred=sp_, pred_se=se_p, S=sp_ / st_, E_delta=e_rel)
print(f"true peak force: push mean {f1_true.mean():.2f}, retreat {f0_true.mean():.2f}; pred push {f1_pred.mean():.2f}, retreat {f0_pred.mean():.2f}", flush=True)
json.dump(res, open(RESULTS / f"func_impact_{TAG}.json", "w"), indent=1)
np.savez(RESULTS / f"func_impact_{TAG}.npz", lm=lm, gg=gg, lk=lk, gap=gap, vn=vn, vt=vt, nvec=nvec, f1_true=f1_true, f0_true=f0_true, f1_pred=f1_pred, f0_pred=f0_pred, fnat_true=f_nat_true, fnat_pred=fnat_pred)
print("saved", RESULTS / f"func_impact_{TAG}.json")
