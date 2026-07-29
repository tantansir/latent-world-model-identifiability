"""Post-review evaluation batch:
  (a) MLP (nonlinear) probes for m/gamma/k on the final variant family
  (b) drift-parameter probe for e013 (slow-linear cell of the attribute matrix)
      + linear raw-data recoverability certificate for drift
Usage: python review_fixes.py mlp | drift
"""
import sys, json, pathlib
import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import model as M
from model import XJEPA
from train import (load_split, whiten_stats, make_batch, ridge_fit_eval, r2,
                   T_CTX, RESULTS)

device = "cuda"
MODE = sys.argv[1] if len(sys.argv) > 1 else "mlp"


def mlp_probe(Xtr, ytr, Xte, yte, iters=4001):
    cut = int(len(Xtr) * 0.85)
    mu, sd = Xtr[:cut].mean(0), Xtr[:cut].std(0) + 1e-6
    ymu, ysd = ytr[:cut].mean(0), ytr[:cut].std(0) + 1e-6
    xtr = torch.tensor((Xtr[:cut] - mu) / sd, device=device, dtype=torch.float32)
    ytr_ = torch.tensor((ytr[:cut] - ymu) / ysd, device=device, dtype=torch.float32)
    xva = torch.tensor((Xtr[cut:] - mu) / sd, device=device, dtype=torch.float32)
    yva = ytr[cut:]
    xte = torch.tensor((Xte - mu) / sd, device=device, dtype=torch.float32)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(),
                        nn.Linear(256, 128), nn.GELU(),
                        nn.Linear(128, ytr.shape[1])).to(device)
    o = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    best, best_pred = -1e9, None
    for i in range(iters):
        idx = torch.randint(0, len(xtr), (256,), device=device)
        l = (net(xtr[idx]) - ytr_[idx]).pow(2).mean()
        o.zero_grad(); l.backward(); o.step()
        if i % 200 == 0:
            with torch.no_grad():
                rv = r2(yva, net(xva).cpu().numpy() * ysd + ymu).mean()
                if rv > best:
                    best = rv
                    best_pred = net(xte).cpu().numpy() * ysd + ymu
    return r2(yte, best_pred)


def trunk_features(model, img, state, stats, ep, t0):
    hs = []
    with torch.no_grad():
        for s in range(0, len(ep), 192):
            b = make_batch(img, state, stats, ep[s:s + 192], t0[s:s + 192],
                           device, T=T_CTX)
            with torch.autocast("cuda", torch.bfloat16):
                z = model.encode(b["img"], b["prop"], b["touch"])
                h = model.pred.trunk(z[:, :-1].float(), b["act"][:, :-1])
            hs.append(h[:, -1].float().cpu().numpy())
    return np.concatenate(hs)


def windows(state, n):
    t0s = np.array([0, 8, 16, 24, 32, 40])
    ep = np.repeat(np.arange(n), len(t0s))
    t0 = np.tile(t0s, n)
    contact = np.array([state["touch"][e, t:t + T_CTX, 6].sum()
                        for e, t in zip(ep, t0)])
    return ep, t0, contact


if MODE == "mlp":
    img_tr_, st_tr_ = load_split("train")
    stats = whiten_stats(st_tr_)
    img_ptr, st_ptr = load_split("probe_tr")
    img_pte, st_pte = load_split("probe_te")
    ep_tr, t0_tr, c_tr = windows(st_ptr, img_ptr.shape[0])
    ep_te, t0_te, c_te = windows(st_pte, img_pte.shape[0])
    lab = lambda st, ep: np.stack([np.log(st["props"][ep, 0]), st["props"][ep, 1],
                                   np.log(st["props"][ep, 2])], 1)
    out = {}
    for tag in ("V_s0_l0.02", "VXp_s0_l0.02", "VXt_s0_l0.02", "VX_s0_l0.02",
                "VFX_s0_l0.02"):
        m = XJEPA(tag.split("_")[0]).to(device)
        m.load_state_dict(torch.load(RESULTS / f"{tag}.pt", weights_only=True),
                          strict=False)
        m.eval()
        Htr = trunk_features(m, img_ptr, st_ptr, stats, ep_tr, t0_tr)
        Hte = trunk_features(m, img_pte, st_pte, stats, ep_te, t0_te)
        mtr, mte = c_tr >= 2, c_te >= 2
        r_lin, _ = ridge_fit_eval(Htr[mtr], lab(st_ptr, ep_tr)[mtr],
                                  Hte[mte], lab(st_pte, ep_te)[mte])
        r_mlp = mlp_probe(Htr[mtr], lab(st_ptr, ep_tr)[mtr],
                          Hte[mte], lab(st_pte, ep_te)[mte])
        out[tag] = {"ridge": [round(float(x), 3) for x in r_lin],
                    "mlp": [round(float(x), 3) for x in r_mlp]}
        print(tag, out[tag], flush=True)
    (RESULTS / "mlp_probes.json").write_text(json.dumps(out, indent=1))

elif MODE in ("drift", "gain", "gain2", "gain3", "mgain"):
    SUF = {"drift": "3", "gain": "5", "gain2": "7", "gain3": "8",
           "mgain": "6"}[MODE]
    KEY = {"gain2": "gain", "gain3": "gain"}.get(MODE, MODE)
    img_tr_, st_tr_ = load_split(f"train{SUF}")
    stats = whiten_stats(st_tr_)
    img_ptr, st_ptr = load_split(f"probe{SUF}_tr")
    img_pte, st_pte = load_split(f"probe{SUF}_te")
    ep_tr, t0_tr, _ = windows(st_ptr, img_ptr.shape[0])
    ep_te, t0_te, _ = windows(st_pte, img_pte.shape[0])
    ytr = np.atleast_2d(st_ptr[KEY][ep_tr].T).T
    yte = np.atleast_2d(st_pte[KEY][ep_te].T).T
    # raw-data recoverability certificate (linear suffices for a linear param;
    # for the ratio-type mgain the linear certificate fails BY DESIGN - that
    # failure certifies the ratio attribute - and a GRU sequence certificate
    # establishes recoverability, mirroring the gamma protocol)
    def raw_seq(st, ep, t0):
        return np.stack([np.concatenate(
            [st["finger"][e, t:t + T_CTX], st["obj"][e, t:t + T_CTX],
             st["touch"][e, t:t + T_CTX], st["act"][e, t:t + T_CTX]],
            1) for e, t in zip(ep, t0)]).astype(np.float32)   # [N, T, 17]

    def raw_feats(st, ep, t0):
        return raw_seq(st, ep, t0).reshape(len(ep), -1)

    def gru_cert(Xtr, ytr, Xte, yte, iters=4000):
        cut = int(len(Xtr) * 0.85)
        f = Xtr[:cut].reshape(-1, Xtr.shape[-1])
        mu, sd = f.mean(0), f.std(0) + 1e-6
        ymu, ysd = ytr[:cut].mean(0), ytr[:cut].std(0) + 1e-6
        xtr = torch.tensor((Xtr[:cut] - mu) / sd, device=device)
        ytr_ = torch.tensor((ytr[:cut] - ymu) / ysd, device=device,
                            dtype=torch.float32)
        xva = torch.tensor((Xtr[cut:] - mu) / sd, device=device)
        yva = ytr[cut:]
        xte = torch.tensor((Xte - mu) / sd, device=device)

        class G(nn.Module):
            def __init__(self, d_in, d_out):
                super().__init__()
                self.gru = nn.GRU(d_in, 96, num_layers=2, batch_first=True)
                self.head = nn.Sequential(nn.Linear(192, 128), nn.GELU(),
                                          nn.Linear(128, d_out))

            def forward(self, x):
                h, _ = self.gru(x)
                return self.head(torch.cat([h[:, -1], h.mean(1)], -1))

        torch.manual_seed(0)
        net = G(Xtr.shape[-1], ytr.shape[1]).to(device)
        o = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
        best, best_pred = -1e9, None
        for i in range(iters):
            idx = torch.randint(0, len(xtr), (256,), device=device)
            l = (net(xtr[idx]) - ytr_[idx]).pow(2).mean()
            o.zero_grad(); l.backward(); o.step()
            if i % 200 == 0:
                with torch.no_grad():
                    pv = torch.cat([net(xva[s:s + 2048])
                                    for s in range(0, len(xva), 2048)])
                    rv = r2(yva, pv.cpu().numpy() * ysd + ymu).mean()
                    if rv > best:
                        best = rv
                        pt = torch.cat([net(xte[s:s + 2048])
                                        for s in range(0, len(xte), 2048)])
                        best_pred = pt.cpu().numpy() * ysd + ymu
        return r2(yte, best_pred)

    r_cert, _ = ridge_fit_eval(raw_feats(st_ptr, ep_tr, t0_tr), ytr,
                               raw_feats(st_pte, ep_te, t0_te), yte)
    print(f"{KEY} raw-data linear certificate R2:", r_cert.round(3), flush=True)
    out = {"cert_linear_raw": [round(float(x), 3) for x in r_cert]}
    if MODE in ("mgain", "gain", "gain2", "gain3"):
        r_gru = gru_cert(raw_seq(st_ptr, ep_tr, t0_tr), ytr,
                         raw_seq(st_pte, ep_te, t0_te), yte)
        out["cert_gru_raw"] = [round(float(x), 3) for x in r_gru]
        print(f"{MODE} raw-data GRU certificate R2:", r_gru.round(3), flush=True)
    m = XJEPA("VFX").to(device)
    m.load_state_dict(torch.load(RESULTS / f"VFX_s0_l0.02_d{SUF}.pt",
                                 weights_only=True), strict=False)
    m.eval()
    Htr = trunk_features(m, img_ptr, st_ptr, stats, ep_tr, t0_tr)
    Hte = trunk_features(m, img_pte, st_pte, stats, ep_te, t0_te)
    r_lin, _ = ridge_fit_eval(Htr, ytr, Hte, yte)
    r_mlp = mlp_probe(Htr, ytr, Hte, yte)
    out["trunk_ridge"] = [round(float(x), 3) for x in r_lin]
    out["trunk_mlp"] = [round(float(x), 3) for x in r_mlp]
    print(json.dumps(out), flush=True)
    (RESULTS / f"{MODE}_probe.json").write_text(json.dumps(out, indent=1))
