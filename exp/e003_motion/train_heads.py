"""Frozen-trunk head retraining (Codex round 2, the one follow-up mechanism experiment).

Load a trained checkpoint, freeze encoder + trunk (+ all existing heads), and train fresh Delta-{1,4,16} latent heads on
the same data and objective in two variants:
  H(h, A)        : the interaction head as in --ihead
  H(h, A, theta) : the same head with the true whitened parameters (log m, gamma, log k) appended to its input
If H(h, A, theta) recovers the counterfactual action response and H(h, A) does not, the frozen representation plus the
same supervision are not enough for a predictor to learn the response, while explicit parameters are; if H(h, A) also
recovers it, the original predictor's failure was a learning-process fact, not a representation fact.
Usage: python train_heads.py TAG [--theta] [--steps N]   -> results/{TAG}_fh.pt or {TAG}_fht.pt (full state dict)
"""
import sys, argparse, json, time, pathlib
import numpy as np, torch, torch.nn as nn
ap = argparse.ArgumentParser(description="Retrain prediction heads on a frozen PokeWorld checkpoint.")
ap.add_argument("tag", help="checkpoint filename without the .pt suffix")
ap.add_argument("--theta", action="store_true"); ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--thetahat", action="store_true", help="theta_hat = cross-fitted (episode-grouped) ridge estimate of the params from h, instead of the true params")
ap.add_argument("--randproj", action="store_true", help="control: a fixed random 3-d projection of h (standardized) in place of theta")
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
sys.argv = sys.argv[:1] + [args.tag]
from common_eval import *                     # builds `model` from TAG (sets ACORR / IHEAD / GHA0 from the tag)
from common_eval import model, stats, device, TAG, img_tr, st_tr
import model as M
from model import action_windows, DeltaHead
from train import make_batch, WIN, T_CTX, HORIZON, RESULTS

torch.manual_seed(args.seed); np.random.seed(args.seed)


from train_heads_lib import ThetaHead


for p in model.parameters():
    p.requires_grad_(False)
M.IHEAD = True
USE_TH = args.theta or args.thetahat or args.randproj
if args.randproj:
    gP = torch.Generator().manual_seed(11)
    P = torch.randn(M.DP, 3, generator=gP).to(device) / M.DP ** 0.5
    Hs = []
    rs = np.random.default_rng(7)
    for _ in range(20):
        ei = rs.integers(0, img_tr.shape[0], 256); ti = rs.integers(0, img_tr.shape[1] - WIN + 1, 256)
        bb = make_batch(img_tr, st_tr, stats, ei, ti, device, T=WIN)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            zz = model.encode(bb["img"], bb["prop"], bb["touch"], bb.get("obj")); hh = model.pred.trunk(zz[:, :-1], bb["act"][:, :-1], None); hh = model.condition(hh, bb["act"])
        Hs.append((hh.float()[:, 3::4].reshape(-1, M.DP) @ P).cpu())
    Hs = torch.cat(Hs); P_mu, P_sd = Hs.mean(0).to(device), (Hs.std(0) + 1e-6).to(device)
    model.thetahat_W = (torch.zeros(M.DP, device=device), torch.ones(M.DP, device=device), P / P_sd - 0.0)   # theta = (h @ P - mu)/sd == ((h - 0)/1) @ (P/sd) - mu/sd; the offset is absorbed below
    model.randproj_off = (P_mu / P_sd)
if args.thetahat:
    # cross-fitted per-window ridge estimate of the whitened params from the frozen trunk state: 5 folds by episode on the
    # training set; window t of an episode in fold f uses W_f fitted on the other folds. Evaluation uses W fitted on all
    # training episodes (test episodes are freshly simulated).
    lbl = np.stack([np.log(st_tr["props"][:, 0]), st_tr["props"][:, 1], np.log(st_tr["props"][:, 2])], 1)
    model.sid_mu.copy_(torch.tensor(lbl.mean(0), dtype=torch.float32)); model.sid_sd.copy_(torch.tensor(lbl.std(0) + 1e-6, dtype=torch.float32))
    n_ep_all = img_tr.shape[0]; fold = rng_f = np.random.default_rng(123).integers(0, 5, n_ep_all)
    Hs, Ys, Es = [], [], []
    rs = np.random.default_rng(7)
    for _ in range(60):
        ei = rs.integers(0, n_ep_all, 256); ti = rs.integers(0, img_tr.shape[1] - WIN + 1, 256)
        bb = make_batch(img_tr, st_tr, stats, ei, ti, device, T=WIN)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            zz = model.encode(bb["img"], bb["prop"], bb["touch"], bb.get("obj")); hh = model.pred.trunk(zz[:, :-1], bb["act"][:, :-1], None); hh = model.condition(hh, bb["act"])
        hh = hh.float()[:, 3::4].reshape(-1, M.DP).cpu().numpy()                          # a few positions per window
        Hs.append(hh); Ys.append(np.repeat(model.props_w(bb).cpu().numpy(), hh.shape[0] // 256, 0)); Es.append(np.repeat(ei, hh.shape[0] // 256))
    Hs, Ys, Es = np.concatenate(Hs), np.concatenate(Ys), np.concatenate(Es)
    def ridge(X, Y, a=10.0):
        mu_, sd_ = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu_) / sd_
        Wm = np.linalg.solve(Xn.T @ Xn + a * np.eye(X.shape[1]), Xn.T @ Y); return mu_, sd_, Wm
    Wf = [ridge(Hs[fold[Es] != f], Ys[fold[Es] != f]) for f in range(5)]
    Wall = ridge(Hs, Ys)
    r2 = 1 - ((Ys - ((Hs - Wall[0]) / Wall[1]) @ Wall[2]) ** 2).mean(0) / Ys.var(0)
    print(f"theta_hat ridge (in-sample, all) R2 m/g/k {np.round(r2, 3)}", flush=True)
    Wf_t = [(torch.tensor(a_, device=device).float(), torch.tensor(b_, device=device).float(), torch.tensor(c_, device=device).float()) for a_, b_, c_ in Wf]
    Wall_t = tuple(torch.tensor(x_, device=device).float() for x_ in Wall)
    model.thetahat_W = Wall_t
    def theta_hat(h, ep_idx):
        out = torch.zeros(h.shape[0], h.shape[1], 3, device=device)
        for f in range(5):
            mk = torch.tensor(fold[ep_idx] == f, device=device)
            if mk.any():
                mu_, sd_, Wm = Wf_t[f]; out[mk] = ((h[mk] - mu_) / sd_) @ Wm
        return out
if USE_TH:
    heads = nn.ModuleDict({str(d): ThetaHead(d, M.D) for d in M.DELTAS})
    lbl = np.stack([np.log(st_tr["props"][:, 0]), st_tr["props"][:, 1], np.log(st_tr["props"][:, 2])], 1)
    # whitening stored in the model's sid buffers so that evaluation (model.props_w) uses the same transform
    model.sid_mu.copy_(torch.tensor(lbl.mean(0), dtype=torch.float32)); model.sid_sd.copy_(torch.tensor(lbl.std(0) + 1e-6, dtype=torch.float32))
else:
    heads = nn.ModuleDict({str(d): DeltaHead(d) for d in M.DELTAS})
heads.to(device)
opt = torch.optim.AdamW(heads.parameters(), lr=3e-4, weight_decay=0.05)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 300, 0.5 * (1 + np.cos(np.pi * s / args.steps)) + 1e-2))
rng = np.random.default_rng(args.seed)
n_ep, T = img_tr.shape[:2]
model.eval(); heads.train()
log = []; t0 = time.time()
for step in range(args.steps):
    ep_idx = rng.integers(0, n_ep, args.bs); tt = rng.integers(0, T - WIN + 1, args.bs)
    b = make_batch(img_tr, st_tr, stats, ep_idx, tt, device, T=WIN)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        z = model.encode(b["img"], b["prop"], b["touch"], b.get("obj"))
        pc = model.props_w(b) if model.use_oracle else None
        h = model.pred.trunk(z[:, :-1], b["act"][:, :-1], pc)
        h = model.condition(h, b["act"])
    z, h = z.float(), h.float()
    loss = 0.0; parts = {}
    for d in M.DELTAS:
        tv = z.shape[1] - d
        if args.theta:
            heads[str(d)].theta_ctx = model.props_w(b)
        elif args.thetahat:
            heads[str(d)].theta_ctx = theta_hat(h[:, :tv], ep_idx)
        elif args.randproj:
            heads[str(d)].theta_ctx = h[:, :tv] @ model.thetahat_W[2] - model.randproj_off
        y = heads[str(d)](h[:, :tv], action_windows(b["act"], d, tv))
        l = (y - z[:, d:]).pow(2).mean(); parts[f"z{d}"] = float(l); loss = loss + 0.5 * l
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(heads.parameters(), 5.0); opt.step(); sched.step()
    if step % 500 == 0 or step == args.steps - 1:
        print(f"[{TAG} heads{'+theta' if args.theta else ''}] step {step} {parts} ({time.time() - t0:.0f}s)", flush=True); log.append({"step": step, **parts})
# splice the new heads into the model and save the full state dict under a new tag
model.dheads = heads
suf = "_fht" if args.theta else "_fhh" if args.thetahat else "_fhr" if args.randproj else "_fh"
sd_ = model.state_dict()
torch.save(sd_, RESULTS / f"{TAG}{suf}.pt")
if args.thetahat:
    torch.save({"mu": Wall_t[0].cpu(), "sd": Wall_t[1].cpu(), "W": Wall_t[2].cpu()}, RESULTS / f"{TAG}{suf}_ridge.pt")
if args.randproj:
    torch.save({"mu": torch.zeros(M.DP), "sd": torch.ones(M.DP), "W": model.thetahat_W[2].cpu(), "off": model.randproj_off.cpu()}, RESULTS / f"{TAG}{suf}_ridge.pt")
json.dump({"base": TAG, "theta": args.theta, "thetahat": args.thetahat, "steps": args.steps, "log": log}, open(RESULTS / f"{TAG}{suf}.json", "w"), indent=1)
print("saved", RESULTS / f"{TAG}{suf}.pt")
