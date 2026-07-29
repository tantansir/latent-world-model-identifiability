"""Round-5 analyses.
  logspeed_cert : linear gamma certificate under flow log-speed coordinates
                  (the coordinate-change prediction of the two-regime law)
  pixel_certs   : gamma / m / k certificates using PIXEL-derived object state
                  (centroid + centroid velocity) instead of simulator state
  composition   : finger / object / relative-displacement probes on VXp, VXt,
                  VX (the target-composition decomposition, reviewer request)
Usage: python round5.py <mode> [limit]
"""
import json, sys, pathlib
import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import train as TR
from train import (load_split, whiten_stats, make_batch, ridge_fit_eval, r2,
                   T_CTX, RESULTS)
from model import XJEPA

device = "cuda"
MODE = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DT = 0.05

img_tr_, st_tr_ = load_split("train")
stats = whiten_stats(st_tr_)
img_ptr, st_ptr = load_split("probe_tr")
img_pte, st_pte = load_split("probe_te")


def windows(state, n):
    t0s = np.array([0, 8, 16, 24, 32, 40])
    ep = np.repeat(np.arange(n), len(t0s))
    t0 = np.tile(t0s, n)
    contact = np.array([state["touch"][e, t:t + T_CTX, 6].sum()
                        for e, t in zip(ep, t0)])
    speed = np.array([np.linalg.norm(state["obj"][e, t:t + T_CTX, 2:], axis=1).mean()
                      for e, t in zip(ep, t0)])
    return ep, t0, contact, speed


def obj_centroids(frames, finger_pos):
    """frames [N,T,64,64] float 0..1, finger_pos [N,T,2] (proprioception, a
    legitimate sensor) -> object centroids [N,T,2] in arena units. The object
    renders at ~0.55 intensity, the finger at 1.0; the finger's antialiased
    edge also passes an intensity band, so its disk is masked out explicitly.
    Centroids are 3-frame smoothed before differencing."""
    mask = (frames > 0.3) & (frames < 0.8)
    lin = (np.arange(64) + 0.5) / 64
    yy, xx = np.meshgrid(lin, lin, indexing="ij")
    dx = xx[None, None] - finger_pos[:, :, 0, None, None]
    dy = yy[None, None] - finger_pos[:, :, 1, None, None]
    mask &= (dx * dx + dy * dy) > 0.10 ** 2
    w = mask.astype(np.float32)
    tot = w.sum((-2, -1)) + 1e-8
    cx = (w * xx).sum((-2, -1)) / tot
    cy = (w * yy).sum((-2, -1)) / tot
    c = np.stack([cx, cy], -1)
    c[tot < 5] = np.nan
    sm = c.copy()
    sm[:, 1:-1] = (c[:, :-2] + c[:, 1:-1] + c[:, 2:]) / 3.0
    return sm


def gru_probe(Xtr, ytr, Xte, yte, iters=4000):
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
            hh, _ = self.gru(x)
            return self.head(torch.cat([hh[:, -1], hh.mean(1)], -1))

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


def flow_speeds(img, state, ep, t0):
    """Per-frame object-region flow speed for the given windows: [N, T-1]."""
    from flownet import FlowNet
    if TR.FLOW_NET is None:
        fn = FlowNet().to(device)
        fn.load_state_dict(torch.load(RESULTS / "flownet.pt", weights_only=True))
        fn.eval()
        TR.FLOW_NET = fn
    speeds = []
    for s in range(0, len(ep), 128):
        b = make_batch(img, state, stats, ep[s:s + 128], t0[s:s + 128],
                       device, T=T_CTX)
        fr = b["img"][:, :, 0].cpu().numpy()
        fl = (b["img"][:, :, 2:4] * 4.0).cpu().numpy()      # undo /4 scaling
        mag = np.sqrt((fl ** 2).sum(2))                     # [B,T,64,64]
        mask = (fr > 0.3) & (fr < 0.8)
        w = mask.astype(np.float32)
        sp = (mag * w).sum((-2, -1)) / (w.sum((-2, -1)) + 1e-8)
        speeds.append(sp[:, 1:])                            # flow at t vs t-1
    return np.concatenate(speeds)


if MODE == "logspeed_cert":
    out = {}
    for split, img, st in (("tr", img_ptr, st_ptr), ("te", img_pte, st_pte)):
        ep, t0, c, sp = windows(st, img.shape[0])
        gl = (sp > 0.15) & (c == 0)
        ep, t0 = ep[gl], t0[gl]
        if LIMIT:
            ep, t0 = ep[:LIMIT], t0[:LIMIT]
        S = flow_speeds(img, st, ep, t0)                    # [N, 15]
        X = np.log(S + 0.02)
        y = st["props"][ep, 1:2]
        out[split] = {"n": int(len(ep)), "X": X, "y": y}
    r_lin, _ = ridge_fit_eval(out["tr"]["X"], out["tr"]["y"],
                              out["te"]["X"], out["te"]["y"])
    # physics-informed analytic in the same coordinates: median log-speed slope
    def analytic(X):
        d = -(X[:, 1:] - X[:, :-1]) / DT
        return np.median(d, 1, keepdims=True)
    r_ana = r2(out["te"]["y"], analytic(out["te"]["X"]))
    res = {"n_glide_windows": [out["tr"]["n"], out["te"]["n"]],
           "cert_linear_logspeed": [round(float(x), 3) for x in r_lin],
           "cert_analytic_logspeed": [round(float(x), 3) for x in r_ana]}
    print(json.dumps(res), flush=True)
    if not LIMIT:
        (RESULTS / "logspeed_cert.json").write_text(json.dumps(res, indent=1))

elif MODE == "pixel_certs":
    ep_a, t0_a, c_a, sp_a = windows(st_ptr, img_ptr.shape[0])
    ep_b, t0_b, c_b, sp_b = windows(st_pte, img_pte.shape[0])
    if LIMIT:
        ep_a, t0_a, c_a, sp_a = ep_a[:LIMIT], t0_a[:LIMIT], c_a[:LIMIT], sp_a[:LIMIT]
        ep_b, t0_b, c_b, sp_b = ep_b[:LIMIT], t0_b[:LIMIT], c_b[:LIMIT], sp_b[:LIMIT]

    def feats(img, state, ep, t0):
        fr = np.stack([img[e, t:t + T_CTX] for e, t in zip(ep, t0)]) / 255.0
        fpos = np.stack([state["finger"][e, t:t + T_CTX, :2]
                         for e, t in zip(ep, t0)])
        cen = obj_centroids(fr, fpos)                       # [N,T,2]
        v = np.zeros_like(cen)
        v[:, 1:-1] = (cen[:, 2:] - cen[:, :-2]) / (2 * DT)
        v[:, 0], v[:, -1] = v[:, 1], v[:, -2]
        bad = np.isnan(cen).any((1, 2))
        cen, v = np.nan_to_num(cen), np.nan_to_num(v)
        base = np.stack([np.concatenate(
            [state["finger"][e, t:t + T_CTX], cen[i], v[i],
             state["touch"][e, t:t + T_CTX], state["act"][e, t:t + T_CTX]],
            1) for i, (e, t) in enumerate(zip(ep, t0))]).astype(np.float32)
        return base, v, bad

    lab = lambda st, ep: np.stack([np.log(st["props"][ep, 0]),
                                   st["props"][ep, 1],
                                   np.log(st["props"][ep, 2])], 1)
    Xa, va, bad_a = feats(img_ptr, st_ptr, ep_a, t0_a)
    Xb, vb, bad_b = feats(img_pte, st_pte, ep_b, t0_b)
    # GRU certificate from pixel-derived object state, contact windows
    mtr = (c_a >= 2) & ~bad_a
    mte = (c_b >= 2) & ~bad_b
    r_gru = gru_probe(Xa[mtr], lab(st_ptr, ep_a)[mtr],
                      Xb[mte], lab(st_pte, ep_b)[mte])
    # analytic gamma from pixel-centroid velocities, glide windows
    def gamma_hat(v, m):
        sp = np.linalg.norm(v, axis=2) + 1e-6
        d = -(np.log(sp[:, 1:]) - np.log(sp[:, :-1])) / DT
        d = np.where(np.minimum(sp[:, 1:], sp[:, :-1]) > 0.3, d, np.nan)
        med = np.nanmedian(d[:, 1:-2], 1, keepdims=True)
        return np.nan_to_num(med, nan=1.5)
    gl_b = (sp_b > 0.4) & (c_b == 0) & ~bad_b
    r_gam = r2(st_pte["props"][ep_b[gl_b], 1:2], gamma_hat(vb[gl_b], None))
    res = {"gru_pixelstate_m_g_k": [round(float(x), 3) for x in r_gru],
           "analytic_pixel_gamma": round(float(r_gam[0]), 3),
           "n_contact": [int(mtr.sum()), int(mte.sum())],
           "n_glide_te": int(gl_b.sum())}
    print(json.dumps(res), flush=True)
    if not LIMIT:
        (RESULTS / "pixel_certs.json").write_text(json.dumps(res, indent=1))

elif MODE == "composition":
    ep_a, t0_a, _, _ = windows(st_ptr, img_ptr.shape[0])
    ep_b, t0_b, _, _ = windows(st_pte, img_pte.shape[0])

    def trunk_feats(model, img, state, ep, t0):
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

    def labels(st, ep, t0):
        t = t0 + T_CTX - 1
        fin = st["finger"][ep, t, :2]
        obj = st["obj"][ep, t, :2]
        return {"finger": fin, "object": obj, "relative": obj - fin}

    la, lb = labels(st_ptr, ep_a, t0_a), labels(st_pte, ep_b, t0_b)
    out = {}
    for tag in ("V_s0_l0.02", "VXp_s0_l0.02", "VXt_s0_l0.02", "VX_s0_l0.02",
                "VFX_s0_l0.02"):
        m = XJEPA(tag.split("_")[0]).to(device)
        m.load_state_dict(torch.load(RESULTS / f"{tag}.pt", weights_only=True),
                          strict=False)
        m.eval()
        Ha = trunk_feats(m, img_ptr, st_ptr, ep_a, t0_a)
        Hb = trunk_feats(m, img_pte, st_pte, ep_b, t0_b)
        row = {}
        for k in ("finger", "object", "relative"):
            rr, _ = ridge_fit_eval(Ha, la[k], Hb, lb[k])
            row[k] = round(float(rr.mean()), 3)
        out[tag.split("_")[0]] = row
        print(tag, row, flush=True)
    (RESULTS / "composition_probe.json").write_text(json.dumps(out, indent=1))
