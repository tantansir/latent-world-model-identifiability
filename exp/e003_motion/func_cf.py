"""Counterfactual functional test for mass and stiffness under a fixed open-loop intervention.

Context: a recorded 16-frame probe window (probe_tr + probe_te) whose last frames have the object nearly at rest and
out of contact, the finger within reach, and which contains >= 3 contact frames (evidence about m and k).
Intervention: from the context end the finger pushes with a constant action u = 0.9 * unit(obj - finger) for 16
steps. The same intervention is applied to every window, so the action window carries no parameter information
(the closed-loop behavior policy did: heavy objects keep getting pushed, light ones get chased).
Truth: open-loop replay of the simulator from the recorded state with the intervention (fim_certificate.replay).
Model: Delta-4 / Delta-16 latent heads with the intervention action window -> per-frame position readout ->
displacement along the push direction; touch heads (Delta=4) -> predicted peak force at the collision.
Mass: displacement after 4 / 16 steps stratified by log m. Stiffness: peak force stratified by log k (both matched
on the finger-object distance and finger speed by partial regression).
Usage: python func_cf.py TAG
"""
from common_eval import *
from common_eval import model, stats, device, T_CTX, make_batch, HERE, w, mu, sd, ymu, ysd, img_ptr, st_ptr, img_pte, st_pte, AOFF
import numpy as np, torch, json
from fim_certificate import replay
sys.path.insert(0, str(HERE.parent / "e001_xmodal_jepa"))
import pokeworld as pw
EVAL_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 780
N_SIM, NB, PER_BIN = 48000, 6, 300

touch_mu, touch_sd = stats["touch_mu"], stats["touch_sd"]
U = 0.9


def windows(state):
    touch, obj, fin = state["touch"], state["obj"], state["finger"]
    N, T = touch.shape[:2]; cb = touch[..., 6] > 0.5
    out = []
    for e in range(N):
        for t0 in range(0, T - T_CTX - 16 + 1, 2):
            tc = t0 + T_CTX - 1
            if CF_MODE == "release":
                # object being pushed (in contact, moving > 0.2) with room ahead: the finger retreats, the object glides
                if not (cb[e, tc] and cb[e, tc - 1]): continue
                v = obj[e, tc, 2:]
                if np.linalg.norm(v) < 0.2: continue
                ahead = obj[e, tc, :2] + v / np.linalg.norm(v) * 0.35
                if not ((ahead > 0.13) & (ahead < 0.87)).all() or not ((obj[e, tc, :2] > 0.2) & (obj[e, tc, :2] < 0.8)).all(): continue
                out.append((e, t0)); break
            if cb[e, tc] or cb[e, tc - 1] or cb[e, t0:tc + 1].sum() < 2: continue
            if np.linalg.norm(obj[e, tc, 2:]) > 0.10: continue
            d = obj[e, tc, :2] - fin[e, tc, :2]; dist = np.linalg.norm(d)
            if not (0.16 <= dist <= 0.45): continue
            if not ((obj[e, tc, :2] > 0.22) & (obj[e, tc, :2] < 0.78)).all(): continue
            out.append((e, t0)); break
    return np.array(out)


res = {}
rows_all = []
# fresh episodes from the training generator (closed-loop policy in the context, as in training), stratified by log m
import os
CF_POLICY = os.environ.get("CF_POLICY", "closed")   # behavior policy generating the test contexts
CF_MODE = os.environ.get("CF_MODE", "push")          # push: rest object, constant push; release: moving object in contact, finger retreats
SUF = ("" if CF_POLICY == "closed" else "_" + CF_POLICY) + ("" if CF_MODE == "push" else "_" + CF_MODE)
if CF_MODE == "release": U = -0.9
sim = pw.simulate(N_SIM, 64, seed=EVAL_SEED, policy=CF_POLICY)
W = windows(sim)
medges = np.linspace(np.log(0.5), np.log(3.0), NB + 1)
gedges = np.linspace(0.5, 4.0, NB + 1)
mb = np.clip(np.digitize(sim["props"][W[:, 0], 1] if CF_MODE == "release" else np.log(sim["props"][W[:, 0], 0]), gedges if CF_MODE == "release" else medges) - 1, 0, NB - 1)
rng = np.random.default_rng(EVAL_SEED)
keep = np.concatenate([rng.permutation(np.where(mb == b)[0])[:PER_BIN] for b in range(NB)])
print("candidates per log m bin:", [int((mb == b).sum()) for b in range(NB)], flush=True)
W = W[keep]
sel = np.unique(W[:, 0]); remap = {e: i for i, e in enumerate(sel)}
img_g = pw.render(sim["finger"][sel], sim["obj"][sel]); st_g = {k: v[sel] for k, v in sim.items()}
W = np.array([(remap[e], t) for e, t in W])
for split, img, st in (("sim", img_g, st_g),):
    ep, t0 = W[:, 0], W[:, 1]; tc = t0 + T_CTX - 1
    obj, fin, props, act = st["obj"], st["finger"], st["props"], st["act"]
    n = len(ep)
    d = obj[ep, tc, :2] - fin[ep, tc, :2]; dist = np.linalg.norm(d, axis=1); nvec = d / dist[:, None]
    if CF_MODE == "release":
        nvec = obj[ep, tc, 2:] / np.linalg.norm(obj[ep, tc, 2:], axis=1, keepdims=True)   # push direction = object velocity; U<0 retreats
    fspeed = np.linalg.norm(fin[ep, tc, 2:], axis=1)
    vf_n = (fin[ep, tc, 2:] * nvec).sum(1); vo_n = (obj[ep, tc, 2:] * nvec).sum(1)   # finger / object velocity along the push direction
    # truth by open-loop replay with the intervention
    acts = np.repeat((U * nvec)[None], 16, 0)                                    # [16,n,2]
    fr, ob, tou = replay(fin[ep, tc, :2].astype(np.float64), fin[ep, tc, 2:].astype(np.float64),
                         obj[ep, tc, :2].astype(np.float64), obj[ep, tc, 2:].astype(np.float64), acts,
                         props[ep, 0].astype(np.float64), props[ep, 1].astype(np.float64), props[ep, 2].astype(np.float64))
    dtrue4 = ((ob[3, :, :2] - obj[ep, tc, :2]) * nvec).sum(1); dtrue16 = ((ob[15, :, :2] - obj[ep, tc, :2]) * nvec).sum(1)
    onset = np.array([next((t for t in range(16) if tou[t, i, 6] > 0.5), -1) for i in range(n)])
    fpeak_true = np.array([tou[onset[i]:onset[i] + 2, i, 2].max() if onset[i] >= 0 else np.nan for i in range(n)])
    # model predictions with the intervention action window (training convention: window = a_{t..t+delta-1})
    preds = {4: [], 16: []}; tpred4 = []; Hs = []
    for s in range(0, n, 256):
        ii, tt = ep[s:s + 256], t0[s:s + 256]
        b = make_batch(img, st, stats, ii, tt, device, T=T_CTX + 16)
        u = torch.from_numpy(U * nvec[s:s + 256]).to(device).float()[:, None].expand(-1, 16, -1)
        act_cf = torch.cat([b["act"][:, :T_CTX], u], 1)                               # [B, T_CTX+16, 2]; ACORR window = the 16 interventions
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"][:, :T_CTX], b["prop"][:, :T_CTX], b["touch"][:, :T_CTX], b["obj"][:, :T_CTX])
            pc = model.props_w(b) if model.use_oracle else None
            h = model.pred.trunk(z.float(), act_cf[:, :T_CTX], pc)
            h = model.condition(h, act_cf)
            hl = h[:, -1:].float()
            Hs.append(hl[:, 0].cpu().numpy())
            if getattr(model, "use_theta_heads", False):
                th = (((hl - model.thetahat_W[0]) / model.thetahat_W[1]) @ model.thetahat_W[2] - model.thetahat_off) if getattr(model, "use_thetahat", False) else model.props_w(b)
                for hd in model.dheads.values(): hd.theta_ctx = th
            for delta in (4, 16):
                aw = act_cf[:, AOFF:AOFF + delta].reshape(len(ii), 1, 2 * delta).float()
                zd = model.dheads[str(delta)](hl, aw)[:, 0].float().cpu().numpy()
                preds[delta].append(((zd - mu) / sd) @ w * ysd + ymu)
            if model.use_touch_tgt:
                aw = act_cf[:, AOFF:AOFF + 4].reshape(len(ii), 1, 8).float()
                y = model.dheads_touch["4"](hl, aw)[:, 0].float().cpu().numpy()
                tpred4.append(y[:, 2] * touch_sd[2] + touch_mu[2])
    p4 = np.concatenate(preds[4]); p16 = np.concatenate(preds[16])
    # second plan A0 = hold (zero action): model and simulator; the action effect Y(A1) - Y(A0) isolates what the
    # push does, and its dependence on m is what an additive head cannot express
    acts0 = np.zeros_like(acts)
    _, ob0, _ = replay(fin[ep, tc, :2].astype(np.float64), fin[ep, tc, 2:].astype(np.float64),
                       obj[ep, tc, :2].astype(np.float64), obj[ep, tc, 2:].astype(np.float64), acts0,
                       props[ep, 0].astype(np.float64), props[ep, 1].astype(np.float64), props[ep, 2].astype(np.float64))
    dtrue16_0 = ((ob0[15, :, :2] - obj[ep, tc, :2]) * nvec).sum(1)
    preds0 = []
    for s in range(0, n, 256):
        ii, tt = ep[s:s + 256], t0[s:s + 256]
        b = make_batch(img, st, stats, ii, tt, device, T=T_CTX + 16)
        act0 = torch.cat([b["act"][:, :T_CTX], torch.zeros(len(ii), 16, 2, device=device)], 1)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"][:, :T_CTX], b["prop"][:, :T_CTX], b["touch"][:, :T_CTX], b["obj"][:, :T_CTX])
            pc = model.props_w(b) if model.use_oracle else None
            h = model.pred.trunk(z.float(), act0[:, :T_CTX], pc); h = model.condition(h, act0)
            if getattr(model, "use_theta_heads", False):
                hl0 = h[:, -1:].float()
                th = (((hl0 - model.thetahat_W[0]) / model.thetahat_W[1]) @ model.thetahat_W[2] - model.thetahat_off) if getattr(model, "use_thetahat", False) else model.props_w(b)
                for hd in model.dheads.values(): hd.theta_ctx = th
            aw = act0[:, AOFF:AOFF + 16].reshape(len(ii), 1, 32).float()
            zd = model.dheads["16"](h[:, -1:].float(), aw)[:, 0].float().cpu().numpy()
        preds0.append(((zd - mu) / sd) @ w * ysd + ymu)
    dpred16_0 = ((np.concatenate(preds0) - obj[ep, tc, :2]) * nvec).sum(1)
    dpred4 = ((p4 - obj[ep, tc, :2]) * nvec).sum(1); dpred16 = ((p16 - obj[ep, tc, :2]) * nvec).sum(1)
    fp4 = np.concatenate(tpred4) if tpred4 else np.full(n, np.nan)
    if CF_MODE == "release":
        # readout ceiling: render the replayed true future, encode frames 15-16 with the frozen encoder, decode position
        seg = {"finger": fr.transpose(1, 0, 2).astype(np.float32), "obj": ob.transpose(1, 0, 2).astype(np.float32),
               "touch": tou.transpose(1, 0, 2).astype(np.float32), "act": acts.transpose(1, 0, 2).astype(np.float32), "props": props[ep]}
        img_seg = pw.render(seg["finger"], seg["obj"])
        ceil = []
        for s in range(0, n, 256):
            bseg = make_batch(img_seg, seg, stats, np.arange(s, min(s + 256, n)), np.full(min(256, n - s), 14), device, T=2)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                zc = model.encode(bseg["img"], bseg["prop"], bseg["touch"], bseg["obj"])[:, 1].float().cpu().numpy()
            ceil.append(((zc - mu) / sd) @ w * ysd + ymu)
        dceil16 = ((np.concatenate(ceil) - obj[ep, tc, :2]) * nvec).sum(1)
        res["release_readout_ceiling"] = dict(zip(("slope_true", "slope_ceiling"), (float(np.polyfit(props[ep, 1], dtrue16, 1)[0]), float(np.polyfit(props[ep, 1], dceil16, 1)[0]))))
        print(f"[release readout ceiling] raw slope on gamma: true {res['release_readout_ceiling']['slope_true']:.4f}, encoded-true-future {res['release_readout_ceiling']['slope_ceiling']:.4f}", flush=True)
    for i in range(n):
        rows_all.append((np.log(props[ep[i], 0]), props[ep[i], 1], np.log(props[ep[i], 2]), dist[i], fspeed[i], dtrue4[i], dtrue16[i],
                         dpred4[i], dpred16[i], onset[i], fpeak_true[i], fp4[i], vf_n[i], vo_n[i], dtrue16_0[i], dpred16_0[i], nvec[i, 0], nvec[i, 1]))
    H_all = np.concatenate(Hs); ep_all = ep.copy()
    print(f"{split}: {n} windows", flush=True)
R = np.array(rows_all, dtype=float)
lm, gg, lk, dist, fsp, dt4, dt16, dp4, dp16, onset, fpk, fp4, vfn, von, dt16_0, dp16_0, nx, ny = R.T
np.savez(RESULTS / f"func_cf_{TAG}{SUF}.npz", R=R)
# partial slopes on log m with the context nuisances controlled (distance, finger speed, finger/object velocity along the
# push, log k, gamma): d true / d log m and d pred / d log m; their ratio = fraction of the mass dependence the model uses
NU = np.c_[dist, fsp, vfn, von, lk, gg, nx, ny]     # push-direction components: world-frame actions differ across windows
# readability of the parameters from h at the intervention start (episode-grouped 5-fold ridge, alpha on an inner split)
def grouped_ridge_r2(X, y, g, k=5):
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
res["probe_at_intervention"] = {nm: grouped_ridge_r2(H_all, v, ep_all) for nm, v in (("log_m", lm), ("gamma", gg), ("log_k", lk))}
print("[readout at the intervention start, episode-grouped ridge R2]", {k_: round(v_, 3) for k_, v_ in res["probe_at_intervention"].items()}, flush=True)
def partial(y, x, nu):
    X = np.c_[np.ones(len(y)), x, nu]; wv, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ wv; se = np.sqrt((r @ r) / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)[1, 1])
    return float(wv[1]), float(se)
for delta, dt, dp in ((4, dt4, dp4), (16, dt16, dp16), ("16_action_effect", dt16 - dt16_0, dp16 - dp16_0)):
    st_, se_t = partial(dt, lm, NU); sp_, se_p = partial(dp, lm, NU)
    print(f"[m, counterfactual, Delta={delta}] partial slope on log m | nuisances: true {st_:.4f} ± {se_t:.4f}, pred {sp_:.4f} ± {se_p:.4f}, ratio {sp_ / st_:.2f}", flush=True)
    res[f"m_partial_d{delta}"] = dict(true=st_, true_se=se_t, pred=sp_, pred_se=se_p, ratio=float(sp_ / st_))
print(f"total {len(R)} windows; collision within 4 steps in {(onset >= 0).sum()} ({((onset >= 0) & (onset < 4)).sum()} within 4)", flush=True)


def report(name, val, pred, true, nuis, nuis_names, edges):
    b = np.clip(np.digitize(val, edges) - 1, 0, len(edges) - 2)
    rows = []
    print(f"[{name}] bin [lo,hi) n | pred ± se | true ± se")
    for i in range(len(edges) - 1):
        m = b == i
        if m.sum() < 5: continue
        rows.append([float(edges[i]), float(edges[i + 1]), int(m.sum()), float(pred[m].mean()), float(pred[m].std() / np.sqrt(m.sum())), float(true[m].mean()), float(true[m].std() / np.sqrt(m.sum()))])
        print(f"   [{edges[i]:.2f},{edges[i+1]:.2f}) {m.sum():4d} | {rows[-1][3]:.4f} ± {rows[-1][4]:.4f} | {rows[-1][5]:.4f} ± {rows[-1][6]:.4f}")
    bt = np.array([r[5] for r in rows]); bp = np.array([r[3] for r in rows])
    slope = float(np.polyfit(bt, bp, 1)[0])
    X = np.c_[np.ones(len(val)), true, nuis]; wr = np.linalg.lstsq(X, pred, rcond=None)[0]
    Xn = np.c_[np.ones(len(val)), nuis]; wn = np.linalg.lstsq(Xn, pred, rcond=None)[0]
    r2 = lambda p: 1 - ((pred - p) ** 2).sum() / ((pred - pred.mean()) ** 2).sum()
    # slope of the bin means of the true quantity on the parameter (how much the truth depends on it) vs the prediction's
    bv = np.array([val[b == i].mean() for i in range(len(edges) - 1) if (b == i).sum() >= 5])
    st_true = float(np.polyfit(bv, bt, 1)[0]); st_pred = float(np.polyfit(bv, bp, 1)[0])
    print(f"   slope of bin-mean pred on bin-mean true: {slope:.2f}   (d true/d param {st_true:.4f}, d pred/d param {st_pred:.4f})")
    print(f"   partial regression pred ~ true + {nuis_names}: coef(true) {wr[1]:.2f}; R2 with/without true {r2(X @ wr):.3f}/{r2(Xn @ wn):.3f}")
    return dict(slope=slope, dtrue_dparam=st_true, dpred_dparam=st_pred, coef_true=float(wr[1]), r2_with=float(r2(X @ wr)), r2_without=float(r2(Xn @ wn)), bins=rows)


kedges = np.linspace(np.log(500), np.log(6000), 7)
res["m_d4"] = report("m, counterfactual push, displacement after 4 steps", lm, dp4, dt4, np.c_[dist, fsp, lk, gg], "dist,fspeed,logk,gamma", medges)
res["m_d16"] = report("m, counterfactual push, displacement after 16 steps", lm, dp16, dt16, np.c_[dist, fsp, lk, gg], "dist,fspeed,logk,gamma", medges)
ok = (onset >= 0) & (onset < 4) & np.isfinite(fp4)
if ok.sum() > 50:
    res["k_peak4"] = report("k, counterfactual push, peak force at the collision (within 4 steps, Delta=4 touch head)", lk[ok], fp4[ok], fpk[ok], np.c_[dist[ok], fsp[ok], lm[ok], gg[ok]], "dist,fspeed,logm,gamma", kedges)
json.dump(res, open(RESULTS / f"func_cf_{TAG}{SUF}.json", "w"), indent=1)
print("saved", RESULTS / f"func_cf_{TAG}{SUF}.json")
