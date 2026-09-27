"""CALIPER-style probe controls for PokeWorld checkpoints.

For a trained checkpoint TAG (results/TAG.pt) reports ridge R^2 of
(log m, gamma, log k) on contact / glide / all windows from:
  trained   : trunk state h_T of the checkpoint (the paper's number)
  random    : same architecture, random init, same probe (feature floor)
  raw_lin   : linear ridge on the raw 16-step window of finger+touch+actions
              (the linear certificate WITHOUT privileged object state)
  raw_lin+obj: same plus object state (the paper's linear certificate)
  swap      : trained features paired with labels of a different episode
              (should be ~0; sanity that the probe uses this episode's evidence)
Usage: python probe_controls.py TAG [--oracle] [--mul] [--vr]
"""
import sys, json, argparse
import numpy as np
import torch
import train as TR
import model as M

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--variant", default=None)
ap.add_argument("--oracle", action="store_true")
ap.add_argument("--mul", action="store_true")
ap.add_argument("--vr", action="store_true")
ap.add_argument("--dim", type=int, default=128)
args = ap.parse_args()
device = "cuda"
M.set_dim(args.dim)
variant = args.variant or args.tag.split("_")[0]

img_tr, st_tr = TR.load_split("train")
stats = TR.whiten_stats(st_tr)
img_ptr, st_ptr = TR.load_split("probe_tr")
img_pte, st_pte = TR.load_split("probe_te")


def windows(img, state):
    n = img.shape[0]
    t0s = np.array([0, 8, 16, 24, 32, 40])
    ep = np.repeat(np.arange(n), len(t0s)); t0 = np.tile(t0s, n)
    contact = np.array([state["touch"][e, t:t + TR.T_CTX, 6].sum() for e, t in zip(ep, t0)])
    speed = np.array([np.linalg.norm(state["obj"][e, t:t + TR.T_CTX, 2:], axis=1).mean() for e, t in zip(ep, t0)])
    y = TR.prop_labels(state, ep)
    raw = np.stack([np.concatenate([state["finger"][e, t:t + TR.T_CTX], state["touch"][e, t:t + TR.T_CTX],
                                    state["act"][e, t:t + TR.T_CTX]], 1) for e, t in zip(ep, t0)]).reshape(len(ep), -1)
    raw_obj = np.stack([np.concatenate([state["finger"][e, t:t + TR.T_CTX], state["obj"][e, t:t + TR.T_CTX],
                                        state["touch"][e, t:t + TR.T_CTX], state["act"][e, t:t + TR.T_CTX]], 1)
                        for e, t in zip(ep, t0)]).reshape(len(ep), -1)
    return dict(ep=ep, t0=t0, contact=contact, speed=speed, y=y, raw=raw, raw_obj=raw_obj)


Wtr, Wte = windows(img_ptr, st_ptr), windows(img_pte, st_pte)
subsets = {"all": (np.ones(len(Wtr["y"]), bool), np.ones(len(Wte["y"]), bool)),
           "contact": (Wtr["contact"] >= 2, Wte["contact"] >= 2),
           "glide": ((Wtr["speed"] > 0.15) & (Wtr["contact"] == 0), (Wte["speed"] > 0.15) & (Wte["contact"] == 0))}


def feats(model):
    model.eval()
    _, Htr = TR.encode_and_trunk(model, img_ptr, st_ptr, stats, Wtr["ep"], Wtr["t0"], device)
    _, Hte = TR.encode_and_trunk(model, img_pte, st_pte, stats, Wte["ep"], Wte["t0"], device)
    return Htr, Hte


def table(Xtr, Xte, ytr=None, yte=None):
    ytr = Wtr["y"] if ytr is None else ytr; yte = Wte["y"] if yte is None else yte
    out = {}
    for nm, (mtr, mte) in subsets.items():
        r, _ = TR.ridge_fit_eval(Xtr[mtr], ytr[mtr], Xte[mte], yte[mte])
        out[nm] = dict(zip(("log_mass", "gamma", "log_stiffness"), [round(float(v), 3) for v in r]))
    return out


res = {}
model = M.XJEPA(variant, in_ch=2, oracle=args.oracle, mul=args.mul, vr=args.vr).to(device)
model.load_state_dict(torch.load(TR.RESULTS / f"{args.tag}.pt", weights_only=True), strict=False)
if args.oracle:
    lbl = np.stack([np.log(st_tr["props"][:, 0]), st_tr["props"][:, 1], np.log(st_tr["props"][:, 2])], 1)
    model.sid_mu.copy_(torch.tensor(lbl.mean(0), dtype=torch.float32)); model.sid_sd.copy_(torch.tensor(lbl.std(0) + 1e-6, dtype=torch.float32))
Htr, Hte = feats(model)
res["trained"] = table(Htr, Hte)
# swap control: labels from a random other window
perm = np.random.default_rng(0).permutation(len(Wte["y"]))
res["swap"] = table(Htr, Hte, Wtr["y"], Wte["y"][perm])
torch.manual_seed(1)
rnd = M.XJEPA(variant, in_ch=2, mul=args.mul, vr=args.vr).to(device)
Rtr, Rte = feats(rnd)
res["random_init"] = table(Rtr, Rte)
res["raw_lin"] = table(Wtr["raw"], Wte["raw"])
res["raw_lin+obj"] = table(Wtr["raw_obj"], Wte["raw_obj"])
(TR.RESULTS / f"controls_{args.tag}.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(k, {s: (v[s]["log_mass"], v[s]["gamma"], v[s]["log_stiffness"]) for s in v})
