"""Natural-continuation functional test for mass (no intervention): the same stratified windows as func_cf.py
(object near rest, finger within reach, contact evidence in the context), but the future is the RECORDED closed-loop
continuation. Truth = recorded 16-step displacement along the finger->object direction; prediction = Delta-16 latent
head with the recorded action window -> position readout. Nuisances include the recorded future actions (their mean
component along the push direction and their magnitude), so the action confound is regressed out linearly. If the
mass-use ratio is high here and ~0 under intervention (func_cf), the model uses a context-continuation proxy for m
that breaks under intervention. Usage: python func_nat.py TAG
"""
from common_eval import *
from common_eval import model, stats, device, T_CTX, make_batch, HERE, w, mu, sd, ymu, ysd, AOFF
import numpy as np, torch, json, os
sys.path.insert(0, str(HERE.parent / "e001_xmodal_jepa"))
import pokeworld as pw
EVAL_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 780
N_SIM, NB, PER_BIN = 48000, 6, 300
POLICY = os.environ.get("CF_POLICY", "closed"); SUF = "" if POLICY == "closed" else "_" + POLICY


def windows(state):
    touch, obj, fin = state["touch"], state["obj"], state["finger"]
    N, T = touch.shape[:2]; cb = touch[..., 6] > 0.5
    out = []
    for e in range(N):
        for t0 in range(0, T - T_CTX - 16 + 1, 2):
            tc = t0 + T_CTX - 1
            if cb[e, tc] or cb[e, tc - 1] or cb[e, t0:tc + 1].sum() < 2: continue
            if np.linalg.norm(obj[e, tc, 2:]) > 0.10: continue
            d = obj[e, tc, :2] - fin[e, tc, :2]; dist = np.linalg.norm(d)
            if not (0.16 <= dist <= 0.45): continue
            if not ((obj[e, tc, :2] > 0.22) & (obj[e, tc, :2] < 0.78)).all(): continue
            out.append((e, t0)); break
    return np.array(out)


sim = pw.simulate(N_SIM, 64, seed=EVAL_SEED, policy=POLICY)
W = windows(sim)
medges = np.linspace(np.log(0.5), np.log(3.0), NB + 1)
mb = np.clip(np.digitize(np.log(sim["props"][W[:, 0], 0]), medges) - 1, 0, NB - 1)
rng = np.random.default_rng(EVAL_SEED)
keep = np.concatenate([rng.permutation(np.where(mb == b)[0])[:PER_BIN] for b in range(NB)])
W = W[keep]; sel = np.unique(W[:, 0]); remap = {e: i for i, e in enumerate(sel)}
img_g = pw.render(sim["finger"][sel], sim["obj"][sel]); st = {k: v[sel] for k, v in sim.items()}
W = np.array([(remap[e], t) for e, t in W]); ep, t0 = W[:, 0], W[:, 1]; tc = t0 + T_CTX - 1
obj, fin, props, act = st["obj"], st["finger"], st["props"], st["act"]; n = len(ep)
d = obj[ep, tc, :2] - fin[ep, tc, :2]; dist = np.linalg.norm(d, axis=1); nvec = d / dist[:, None]
fsp = np.linalg.norm(fin[ep, tc, 2:], axis=1); vfn = (fin[ep, tc, 2:] * nvec).sum(1); von = (obj[ep, tc, 2:] * nvec).sum(1)
fut = np.stack([act[e, t + 1:t + 17] for e, t in zip(ep, tc)])                       # recorded future actions [n,16,2]
a_n = (fut * nvec[:, None]).sum(-1).mean(1); a_mag = np.linalg.norm(fut, axis=-1).mean(1)
a_n8 = (fut[:, :8] * nvec[:, None]).sum(-1).mean(1)
preds = []
for s in range(0, n, 256):
    ii, tt = ep[s:s + 256], t0[s:s + 256]
    b = make_batch(img_g, st, stats, ii, tt, device, T=T_CTX + 16)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        z = model.encode(b["img"][:, :T_CTX], b["prop"][:, :T_CTX], b["touch"][:, :T_CTX], b["obj"][:, :T_CTX])
        pc = model.props_w(b) if model.use_oracle else None
        h = model.pred.trunk(z.float(), b["act"][:, :T_CTX], pc); h = model.condition(h, b["act"])
        aw = b["act"][:, AOFF:AOFF + 16].reshape(len(ii), 1, 32).float()
        zd = model.dheads["16"](h[:, -1:].float(), aw)[:, 0].float().cpu().numpy()
    preds.append(((zd - mu) / sd) @ w * ysd + ymu)
p16 = np.concatenate(preds)
dpred = ((p16 - obj[ep, tc, :2]) * nvec).sum(1); dtrue = ((obj[ep, tc + 16, :2] - obj[ep, tc, :2]) * nvec).sum(1)
lm, gg, lk = np.log(props[ep, 0]), props[ep, 1], np.log(props[ep, 2])


def partial(y, x, nu):
    X = np.c_[np.ones(len(y)), x, nu]; wv, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ wv; se = np.sqrt((r @ r) / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)[1, 1]); return float(wv[1]), float(se)


res = {}
for name, nu in (("state only", np.c_[dist, fsp, vfn, von, lk, gg]), ("state + future actions", np.c_[dist, fsp, vfn, von, lk, gg, a_n, a_mag, a_n8])):
    st_, se_t = partial(dtrue, lm, nu); sp_, se_p = partial(dpred, lm, nu)
    print(f"[m, natural continuation, Delta=16, nuisances: {name}] true {st_:.4f} ± {se_t:.4f}, pred {sp_:.4f} ± {se_p:.4f}, ratio {sp_ / st_:.2f}", flush=True)
    res[name] = dict(true=st_, true_se=se_t, pred=sp_, pred_se=se_p, ratio=sp_ / st_)
# how much of the true m-dependence is carried by the future actions: regress true on m with and without actions
print(f"corr(log m, future action along push) {np.corrcoef(lm, a_n)[0, 1]:.3f}; corr(log m, true disp) {np.corrcoef(lm, dtrue)[0, 1]:.3f}; corr(log m, pred disp) {np.corrcoef(lm, dpred)[0, 1]:.3f}")
np.savez(RESULTS / f"func_nat_{TAG}{SUF}.npz", lm=lm, gg=gg, lk=lk, dist=dist, fsp=fsp, vfn=vfn, von=von, a_n=a_n, a_mag=a_mag, a_n8=a_n8, dtrue=dtrue, dpred=dpred)
json.dump(res, open(RESULTS / f"func_nat_{TAG}{SUF}.json", "w"), indent=1)
print("saved", RESULTS / f"func_nat_{TAG}{SUF}.json")
