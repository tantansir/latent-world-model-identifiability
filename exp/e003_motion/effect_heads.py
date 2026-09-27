"""Parameter-precision x training-target diagnostic for the mass action effect (Codex round 5, the discriminating experiment).

Frozen trunk of TAG. Contexts: the mass-test starts (func_cf push mode) on independent train / val / test episodes.
For each start: h_t (frozen trunk at the context end), s_t (object and finger position and velocity at the context end),
the push plan A1 = 0.9 unit(obj - finger) for 16 steps and the zero-force plan A0; truth = open-loop replay of both
branches (16-step displacement along the push direction, dt1 / dt0); latent targets z1 / z0 = frozen encoder embedding
of the true frame 16 under each branch.
New heads H(h, s, A, c_alpha) -> z_hat (same architecture, same fixed linear position readout), with
  c_alpha = theta_hat(h) + alpha * (theta - theta_hat(h)),  alpha in {0, 0.5, 1}   (theta_hat: episode-grouped
  cross-fitted ridge from h; alpha = 0 adds no test-time parameter information)
trained with either
  latent target : squared error of z_hat to z1 / z0
  effect target : squared error of the decoded displacement difference d_hat(A1) - d_hat(A0) to dt1 - dt0
Positive control: supervised MLP on (s_t, theta, push direction) -> dt1 - dt0.
Metrics on the test starts: S_m^Delta (partial slope ratio on log m with nuisances), E^Delta, RMSE, skill relative to
the train-mean effect, and an episode bootstrap. Usage: python effect_heads.py TAG [--steps N] [--alphas 0,0.5,1]
"""
import numpy as np, torch, torch.nn as nn, json, os, sys, time, argparse, pathlib
ap = argparse.ArgumentParser(description="Compare latent-target and action-effect heads on a frozen PokeWorld checkpoint.")
ap.add_argument("tag", help="checkpoint filename without the .pt suffix")
ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--alphas", default="0,0.5,1")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--train_sims", type=int, default=2)
args = ap.parse_args()
sys.argv = sys.argv[:1] + [args.tag]
from common_eval import *
from common_eval import model, stats, device, T_CTX, make_batch, HERE, w, mu, sd, ymu, ysd, AOFF
from fim_certificate import replay
sys.path.insert(0, str(HERE.parent / "e001_xmodal_jepa"))
import pokeworld as pw
import model as M
RESULTS = HERE / "results"; NB, U = 6, 0.9
medges = np.linspace(np.log(0.5), np.log(3.0), NB + 1)


def windows(state):
    touch, obj, fin = state["touch"], state["obj"], state["finger"]
    N, T = touch.shape[:2]; cb = touch[..., 6] > 0.5; out = []
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


def encode_future(fr, ob, tou, acts, props_sel, n):
    """embed the true frame 16 (index 15) of a replayed branch with the frozen encoder (frames 14-15 rendered)."""
    seg = {"finger": fr.transpose(1, 0, 2).astype(np.float32), "obj": ob.transpose(1, 0, 2).astype(np.float32),
           "touch": tou.transpose(1, 0, 2).astype(np.float32), "act": acts.transpose(1, 0, 2).astype(np.float32), "props": props_sel}
    img_seg = pw.render(seg["finger"], seg["obj"]); out = []
    for s in range(0, n, 256):
        idx = np.arange(s, min(s + 256, n))
        b = make_batch(img_seg, seg, stats, idx, np.full(len(idx), 14), device, T=2)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"], b["prop"], b["touch"], b.get("obj"))[:, 1].float().cpu().numpy()
        out.append(z)
    return np.concatenate(out)


def build(seed, per_bin, n_sim=48000):
    f = RESULTS / f"effect_data_{TAG}_seed{seed}_pb{per_bin}.npz"
    if f.exists():
        return dict(np.load(f))
    sim = pw.simulate(n_sim, 64, seed=seed); W = windows(sim)
    mb = np.clip(np.digitize(np.log(sim["props"][W[:, 0], 0]), medges) - 1, 0, NB - 1)
    rng = np.random.default_rng(seed)
    keep = np.concatenate([rng.permutation(np.where(mb == b)[0])[:per_bin] for b in range(NB)])
    print(f"seed {seed}: candidates per bin {[int((mb == b).sum()) for b in range(NB)]} -> kept {len(keep)}", flush=True)
    W = W[keep]; sel = np.unique(W[:, 0]); remap = {e: i for i, e in enumerate(sel)}
    img_g = pw.render(sim["finger"][sel], sim["obj"][sel]); st = {k: v[sel] for k, v in sim.items()}
    ep = np.array([remap[e] for e in W[:, 0]]); t0 = W[:, 1]; tc = t0 + T_CTX - 1; n = len(ep)
    obj, fin, props = st["obj"], st["finger"], st["props"]
    d = obj[ep, tc, :2] - fin[ep, tc, :2]; dist = np.linalg.norm(d, axis=1); nvec = d / dist[:, None]
    fspeed = np.linalg.norm(fin[ep, tc, 2:], axis=1); vf_n = (fin[ep, tc, 2:] * nvec).sum(1); vo_n = (obj[ep, tc, 2:] * nvec).sum(1)
    P = lambda pl: (fin[ep, tc, :2].astype(np.float64), fin[ep, tc, 2:].astype(np.float64), obj[ep, tc, :2].astype(np.float64), obj[ep, tc, 2:].astype(np.float64), pl,
                    props[ep, 0].astype(np.float64), props[ep, 1].astype(np.float64), props[ep, 2].astype(np.float64))
    acts1 = np.repeat((U * nvec)[None], 16, 0); acts0 = np.zeros_like(acts1)
    fr1, ob1, tou1 = replay(*P(acts1)); fr0, ob0, tou0 = replay(*P(acts0))
    dt1 = ((ob1[15, :, :2] - obj[ep, tc, :2]) * nvec).sum(1); dt0 = ((ob0[15, :, :2] - obj[ep, tc, :2]) * nvec).sum(1)
    z1 = encode_future(fr1, ob1, tou1, acts1, props[ep], n); z0 = encode_future(fr0, ob0, tou0, acts0, props[ep], n)
    H = []
    for s in range(0, n, 256):
        ii, tt = ep[s:s + 256], t0[s:s + 256]
        b = make_batch(img_g, st, stats, ii, tt, device, T=T_CTX + 16)
        a = torch.cat([b["act"][:, :T_CTX], torch.zeros(len(ii), 16, 2, device=device)], 1)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"][:, :T_CTX], b["prop"][:, :T_CTX], b["touch"][:, :T_CTX], b["obj"][:, :T_CTX])
            pc = model.props_w(b) if model.use_oracle else None
            h = model.pred.trunk(z.float(), a[:, :T_CTX], pc); h = model.condition(h, a)
        H.append(h[:, -1].float().cpu().numpy())
    out = dict(h=np.concatenate(H), s=np.concatenate([obj[ep, tc], fin[ep, tc]], 1).astype(np.float32), nvec=nvec.astype(np.float32),
               dist=dist, fspeed=fspeed, vf_n=vf_n, vo_n=vo_n, theta=np.stack([np.log(props[ep, 0]), props[ep, 1], np.log(props[ep, 2])], 1),
               pos=obj[ep, tc, :2].astype(np.float32), dt1=dt1, dt0=dt0, z1=z1, z0=z0, ep=W[:, 0] + seed * 10 ** 6)
    np.savez(f, **out); return out


t_start = time.time()
tr = build(790, 1000, 48000 * args.train_sims); va = build(793, 250); te = build(780, 300)
print(f"train {len(tr['ep'])}, val {len(va['ep'])}, test {len(te['ep'])} starts ({time.time() - t_start:.0f}s)", flush=True)
th_mu, th_sd = tr["theta"].mean(0), tr["theta"].std(0) + 1e-6
s_mu, s_sd = tr["s"].mean(0), tr["s"].std(0) + 1e-6
h_mu, h_sd = tr["h"].mean(0), tr["h"].std(0) + 1e-6


# ---- theta_hat: ridge from h, cross-fitted by episode on train, all-train fit for val / test
def ridge_fit(X, Y, a=10.0):
    Xn = (X - h_mu) / h_sd; return np.linalg.solve(Xn.T @ Xn + a * np.eye(X.shape[1]), Xn.T @ Y)
Yw = (tr["theta"] - th_mu) / th_sd
fold = np.random.default_rng(123).integers(0, 5, len(tr["ep"]))
uep = np.unique(tr["ep"]); epfold = dict(zip(uep, np.random.default_rng(123).integers(0, 5, len(uep)))); fold = np.array([epfold[e] for e in tr["ep"]])
th_hat_tr = np.zeros_like(Yw)
for f_ in range(5):
    Wf = ridge_fit(tr["h"][fold != f_], Yw[fold != f_]); th_hat_tr[fold == f_] = ((tr["h"][fold == f_] - h_mu) / h_sd) @ Wf
Wall = ridge_fit(tr["h"], Yw)
th_hat = {"train": th_hat_tr, "val": ((va["h"] - h_mu) / h_sd) @ Wall, "test": ((te["h"] - h_mu) / h_sd) @ Wall}
r2 = lambda y, p: 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
res = {"theta_hat_r2": {"train_crossfit": r2(Yw, th_hat_tr).round(3).tolist(), "val": r2((va["theta"] - th_mu) / th_sd, th_hat["val"]).round(3).tolist(),
                        "test": r2((te["theta"] - th_mu) / th_sd, th_hat["test"]).round(3).tolist()}, "n": {"train": len(tr["ep"]), "val": len(va["ep"]), "test": len(te["ep"])}}
print("theta_hat R2 (m/g/k) train-crossfit / val / test:", res["theta_hat_r2"], flush=True)
_mb = np.clip(np.digitize(te["theta"][:, 0], medges) - 1, 0, NB - 1); _err = th_hat["test"] - (te["theta"] - th_mu) / th_sd
res["theta_hat_bias_by_mass_bin"] = {"mean_signed_error_m_by_bin": [float(_err[_mb == b, 0].mean()) for b in range(NB)],
                                     "rmse_m_by_bin": [float(np.sqrt((_err[_mb == b, 0] ** 2).mean())) for b in range(NB)]}
res["standardization_note"] = "h, s, theta standardized with all-training-set statistics; only the ridge coefficients are cross-fitted by episode fold"
print("theta_hat mass bias by bin (train-sd units):", np.round(res["theta_hat_bias_by_mass_bin"]["mean_signed_error_m_by_bin"], 2), flush=True)

T_ = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
w_t, mu_t, sd_t = T_(w), T_(mu), T_(sd); ysd_t, ymu_t = T_(ysd), T_(ymu)
def decode_disp(z_hat, pos, nvec):
    p = ((z_hat - mu_t) / sd_t) @ w_t * ysd_t + ymu_t
    return ((p - pos) * nvec).sum(-1)


def pack(D, split, alpha):
    c = th_hat[split] + alpha * ((D["theta"] - th_mu) / th_sd - th_hat[split])
    return {"h": T_((D["h"] - h_mu) / h_sd), "s": T_((D["s"] - s_mu) / s_sd), "c": T_(c), "nvec": T_(D["nvec"]), "pos": T_(D["pos"]),
            "z1": T_(D["z1"]), "z0": T_(D["z0"]), "dy": T_(D["dt1"] - D["dt0"]), "dt1": T_(D["dt1"]), "dt0": T_(D["dt0"])}


class EffectHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_mlp = nn.Sequential(nn.Linear(32, 128), nn.GELU())
        self.out = nn.Sequential(nn.Linear(M.DP + 8 + 128 + 3, 256), nn.GELU(), nn.Linear(256, M.D))
    def forward(self, h, s, aw, c):
        return self.out(torch.cat([h, s, self.a_mlp(aw), c], -1))


def action_window(nvec, push):
    return (U * nvec if push else torch.zeros_like(nvec))[:, None, :].expand(-1, 16, -1).reshape(len(nvec), 32)


def predict(head, B):
    z1 = head(B["h"], B["s"], action_window(B["nvec"], True), B["c"]); z0 = head(B["h"], B["s"], action_window(B["nvec"], False), B["c"])
    return z1, z0, decode_disp(z1, B["pos"], B["nvec"]), decode_disp(z0, B["pos"], B["nvec"])


def train_head(target, alpha, seed):
    torch.manual_seed(seed); head = EffectHead().to(device)
    Btr, Bva = pack(tr, "train", alpha), pack(va, "val", alpha)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 200, 0.5 * (1 + np.cos(np.pi * s / args.steps)) + 1e-2))
    rng = np.random.default_rng(seed); n = len(Btr["h"]); best, best_state = 1e9, None
    def loss_fn(B, idx):
        sub = {k: v[idx] for k, v in B.items()}; z1, z0, d1, d0 = predict(head, sub)
        if target == "latent":
            return 0.5 * ((z1 - sub["z1"]).pow(2).mean() + (z0 - sub["z0"]).pow(2).mean())
        return ((d1 - d0) - sub["dy"]).pow(2).mean() / DY_VAR
    for step in range(args.steps):
        head.train(); idx = torch.tensor(rng.integers(0, n, 256), device=device)
        l = loss_fn(Btr, idx); opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); opt.step(); sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            head.eval()
            with torch.no_grad():
                lv = float(loss_fn(Bva, torch.arange(len(Bva["h"]), device=device)))
            if lv < best: best, best_state = lv, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval(); return head, best


DY_VAR = float(np.var(tr["dt1"] - tr["dt0"]))


def partial(y, x, nu):
    X = np.c_[np.ones(len(y)), x, nu]; wv, *_ = np.linalg.lstsq(X, y, rcond=None); return float(wv[1])


NU_te = np.c_[te["dist"], te["fspeed"], te["vf_n"], te["vo_n"], te["nvec"], te["theta"][:, 1], te["theta"][:, 2]]
dy_te = te["dt1"] - te["dt0"]; dy_train_mean = float((tr["dt1"] - tr["dt0"]).mean())


def metrics(dy_hat, d1_hat=None):
    lm = te["theta"][:, 0]; out = {}
    st_ = partial(dy_te, lm, NU_te); sp_ = partial(dy_hat, lm, NU_te); out["S_delta"] = sp_ / st_
    if d1_hat is not None: out["S_push"] = partial(d1_hat, lm, NU_te) / partial(te["dt1"], lm, NU_te)
    out["E_delta"] = float(((dy_hat - dy_te) ** 2).sum() / (dy_te ** 2).sum()); out["rmse"] = float(np.sqrt(((dy_hat - dy_te) ** 2).mean()))
    out["skill_vs_train_mean"] = float(1 - ((dy_hat - dy_te) ** 2).sum() / ((dy_te - dy_train_mean) ** 2).sum())
    # episode bootstrap on E_delta and S_delta
    rs = np.random.default_rng(0); ue = np.unique(te["ep"]); Eb, Sb = [], []
    for _ in range(500):
        draw = rs.choice(ue, len(ue), replace=True); idx = np.concatenate([np.where(te["ep"] == e)[0] for e in draw])
        Eb.append(((dy_hat[idx] - dy_te[idx]) ** 2).sum() / (dy_te[idx] ** 2).sum())
        Sb.append(partial(dy_hat[idx], lm[idx], NU_te[idx]) / partial(dy_te[idx], lm[idx], NU_te[idx]))
    out["E_delta_ci"] = [float(np.percentile(Eb, 2.5)), float(np.percentile(Eb, 97.5))]; out["S_delta_ci"] = [float(np.percentile(Sb, 2.5)), float(np.percentile(Sb, 97.5))]
    return out


PRED = {"dy_true": dy_te, "ep": te["ep"], "log_m": te["theta"][:, 0]}; HEADS = {}
res["constant_train_mean"] = metrics(np.full(len(dy_te), dy_train_mean)); PRED["constant_train_mean"] = np.full(len(dy_te), dy_train_mean)
res["constant_test_mean_diagnostic"] = {"E_delta": float(((dy_te - dy_te.mean()) ** 2).sum() / (dy_te ** 2).sum())}
print("constant (train mean):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["constant_train_mean"].items()}, flush=True)

# ---- positive control: supervised MLP on (s, theta, nvec) -> dy ; and a version with theta_hat
def train_control(use_hat, seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(8 + 3 + 2, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1)).to(device)
    def X(D, split): c = th_hat[split] if use_hat else (D["theta"] - th_mu) / th_sd; return T_(np.c_[(D["s"] - s_mu) / s_sd, c, D["nvec"]])
    Xtr, Xva, Xte = X(tr, "train"), X(va, "val"), X(te, "test"); ytr, yva = T_(tr["dt1"] - tr["dt0"]), T_(va["dt1"] - va["dt0"])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); rng = np.random.default_rng(seed); best, bs = 1e9, None
    for step in range(args.steps):
        idx = torch.tensor(rng.integers(0, len(Xtr), 256), device=device); l = (net(Xtr[idx])[:, 0] - ytr[idx]).pow(2).mean()
        opt.zero_grad(); l.backward(); opt.step()
        if step % 100 == 0:
            with torch.no_grad(): lv = float((net(Xva)[:, 0] - yva).pow(2).mean())
            if lv < best: best, bs = lv, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(bs)
    with torch.no_grad(): return net(Xte)[:, 0].cpu().numpy()


for name, use_hat in (("control_true_theta", False), ("control_theta_hat", True)):
    PRED[name] = train_control(use_hat, args.seed); res[name] = metrics(PRED[name])
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)

# ---- the grid: target x alpha
for target in ("latent", "effect"):
    for alpha in [float(a) for a in args.alphas.split(",")]:
        head, best_val = train_head(target, alpha, args.seed)
        with torch.no_grad():
            z1, z0, d1, d0 = predict(head, pack(te, "test", alpha))
        m = metrics((d1 - d0).cpu().numpy(), d1.cpu().numpy()); m["best_val_loss"] = best_val
        res[f"{target}_alpha{alpha:g}"] = m; PRED[f"{target}_alpha{alpha:g}"] = (d1 - d0).cpu().numpy(); HEADS[f"{target}_alpha{alpha:g}"] = {k: v.cpu() for k, v in head.state_dict().items()}
        print(f"{target} alpha={alpha:g}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
def paired(a, b):
    """episode bootstrap of E_delta(a) - E_delta(b) and S_delta(a) - S_delta(b), same draws"""
    rs = np.random.default_rng(1); ue = np.unique(te["ep"]); lm = te["theta"][:, 0]; dE, dS = [], []
    for _ in range(500):
        draw = rs.choice(ue, len(ue), replace=True); idx = np.concatenate([np.where(te["ep"] == e)[0] for e in draw])
        E = lambda p: ((p[idx] - dy_te[idx]) ** 2).sum() / (dy_te[idx] ** 2).sum()
        S = lambda p: partial(p[idx], lm[idx], NU_te[idx]) / partial(dy_te[idx], lm[idx], NU_te[idx])
        dE.append(E(PRED[a]) - E(PRED[b])); dS.append(S(PRED[a]) - S(PRED[b]))
    return {"dE_mean": float(np.mean(dE)), "dE_ci": [float(np.percentile(dE, 2.5)), float(np.percentile(dE, 97.5))],
            "dS_mean": float(np.mean(dS)), "dS_ci": [float(np.percentile(dS, 2.5)), float(np.percentile(dS, 97.5))]}
res["paired"] = {}
for a, b in (("effect_alpha0", "latent_alpha0"), ("effect_alpha0.5", "effect_alpha0"), ("latent_alpha1", "latent_alpha0"), ("effect_alpha0", "control_theta_hat")):
    if a in PRED and b in PRED:
        res["paired"][f"{a} - {b}"] = paired(a, b); print("paired", a, "-", b, {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in res["paired"][f"{a} - {b}"].items()}, flush=True)
res["config"] = {"steps": args.steps, "alphas": args.alphas, "head_seed": args.seed, "train_sims": args.train_sims, "lr": 3e-4, "wd": 0.05, "batch": 256,
                 "control": {"hidden": [256, 256], "lr": 1e-3, "wd": 1e-4, "steps": args.steps}}
np.savez(RESULTS / f"effect_heads_{TAG}_s{args.seed}_pred.npz", **PRED); torch.save(HEADS, RESULTS / f"effect_heads_{TAG}_s{args.seed}_heads.pt")
json.dump(res, open(RESULTS / f"effect_heads_{TAG}_s{args.seed}.json", "w"), indent=1)
print("saved", RESULTS / f"effect_heads_{TAG}_s{args.seed}.json", f"({time.time() - t_start:.0f}s)")
