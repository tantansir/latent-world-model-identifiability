"""Concat-token re-probe for e005 checkpoints: is the pooled readout (mean of
16 tokens) destroying spatial information (position = which-token-is-active)?
Re-probes position/velocity/props from CONCATENATED token features."""
import json, pathlib, sys
import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import model as M
from model import XJEPA, K
from train import (load_split, whiten_stats, make_batch, ridge_fit_eval,
                   prop_labels, r2, T_CTX, DATA, RESULTS)

device = "cuda"
img_tr, st_tr = load_split("train")
stats = whiten_stats(st_tr)
img_ptr, st_ptr = load_split("probe_tr")
img_pte, st_pte = load_split("probe_te")


@torch.no_grad()
def features(model, img, state, ep, t0, bs=192):
    tok_last, trunk_cat = [], []
    for s in range(0, len(ep), bs):
        b = make_batch(img, state, stats, ep[s:s + bs], t0[s:s + bs], device, T=T_CTX)
        with torch.autocast("cuda", torch.bfloat16):
            z_sp, tokens = model.enc(b["img"], b["prop"], b["touch"])
            h = model.pred.trunk(tokens, b["act"])
        tok_last.append(z_sp[:, -1].reshape(len(b["img"]), -1).float().cpu().numpy())
        trunk_cat.append(h[:, -1, :K].reshape(len(b["img"]), -1).float().cpu().numpy())
    return np.concatenate(tok_last), np.concatenate(trunk_cat)


def windows(img, state):
    n = img.shape[0]
    t0s = np.array([0, 8, 16, 24, 32, 40])
    ep = np.repeat(np.arange(n), len(t0s))
    t0 = np.tile(t0s, n)
    contact = np.array([state["touch"][e, t:t + T_CTX, 6].sum() for e, t in zip(ep, t0)])
    y_prop = prop_labels(state, ep)
    y_pos = np.stack([state["obj"][e, t + T_CTX - 1, :2] for e, t in zip(ep, t0)])
    y_vel = np.stack([state["obj"][e, t + T_CTX - 1, 2:] for e, t in zip(ep, t0)])
    return ep, t0, contact, y_prop, y_pos, y_vel


ep_tr, t0_tr, c_tr, yp_tr, pos_tr, vel_tr = windows(img_ptr, st_ptr)
ep_te, t0_te, c_te, yp_te, pos_te, vel_te = windows(img_pte, st_pte)

for variant in ("V", "VFX"):
    m = XJEPA(variant).to(device)
    m.load_state_dict(torch.load(RESULTS / f"{variant}_s0.pt", weights_only=True))
    m.eval()
    Ttr, Htr = features(m, img_ptr, st_ptr, ep_tr, t0_tr)
    Tte, Hte = features(m, img_pte, st_pte, ep_te, t0_te)
    r2_pos, _ = ridge_fit_eval(Ttr, pos_tr, Tte, pos_te)
    r2_vel, _ = ridge_fit_eval(Ttr, vel_tr, Tte, vel_te)
    mtr, mte = c_tr >= 2, c_te >= 2
    r2_prop, _ = ridge_fit_eval(Htr[mtr], yp_tr[mtr], Hte[mte], yp_te[mte])
    print(f"{variant}: tokcat pos R2={r2_pos.mean():.3f} vel R2={r2_vel.mean():.3f} | "
          f"trunkcat contact m/g/k = {r2_prop[0]:.3f}/{r2_prop[1]:.3f}/{r2_prop[2]:.3f}",
          flush=True)
print("done")
