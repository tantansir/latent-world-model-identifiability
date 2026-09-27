"""Input-matched recovery reference on the intervention histories (Codex rounds 1-2, drag chain closure).

Uses the same window selection as func_cf.py (push mode: near-rest object with contact evidence; release mode: object being
pushed) on two independent simulated sets: a fit set (seed 790) and the test set used by func_cf (seed 780). Features per
16-frame context: image-derived object centroid and velocity (finger located from the image), finger state, touch, actions
(V+P+T+A), plus a privileged variant (state+A). A GRU estimator of (log m, gamma, log k) is fitted on the fit set and scored
on the test set, so the recovery reference lives on exactly the histories where the models' responses were tested.
Usage: CF_MODE=push|release python cert_cf.py
"""
import os, sys, json, pathlib
import numpy as np, torch
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "e001_xmodal_jepa"))
import pokeworld as pw
from train import r2, T_CTX, RESULTS
from cert_matched import image_centroids, gru_probe
CF_MODE = os.environ.get("CF_MODE", "push"); DT = 0.05; NB, PER_BIN = 6, 300


def windows(state):
    touch, obj, fin = state["touch"], state["obj"], state["finger"]
    N, T = touch.shape[:2]; cb = touch[..., 6] > 0.5; out = []
    for e in range(N):
        for t0 in range(0, T - T_CTX - 16 + 1, 2):
            tc = t0 + T_CTX - 1
            if CF_MODE == "release":
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


def build(seed):
    sim = pw.simulate(48000, 64, seed=seed)
    W = windows(sim)
    par = sim["props"][W[:, 0], 1] if CF_MODE == "release" else np.log(sim["props"][W[:, 0], 0])
    edges = np.linspace(0.5, 4.0, NB + 1) if CF_MODE == "release" else np.linspace(np.log(0.5), np.log(3.0), NB + 1)
    b = np.clip(np.digitize(par, edges) - 1, 0, NB - 1); rng = np.random.default_rng(seed)
    keep = np.concatenate([rng.permutation(np.where(b == i)[0])[:PER_BIN] for i in range(NB)])
    keep = rng.permutation(keep); W = W[keep]      # shuffled: gru_probe holds out the last 15% for early stopping
    sel = np.unique(W[:, 0]); remap = {e: i for i, e in enumerate(sel)}
    img = pw.render(sim["finger"][sel], sim["obj"][sel]); st = {k: v[sel] for k, v in sim.items()}
    ep = np.array([remap[e] for e in W[:, 0]]); t0 = W[:, 1]
    fr = np.stack([img[e, t:t + T_CTX] for e, t in zip(ep, t0)]) / 255.0
    cen = image_centroids(fr); v = np.zeros_like(cen); v[:, 1:-1] = (cen[:, 2:] - cen[:, :-2]) / (2 * DT); v[:, 0], v[:, -1] = v[:, 1], v[:, -2]
    bad = np.isnan(cen).any((1, 2)); cen, v = np.nan_to_num(cen), np.nan_to_num(v)
    g = lambda k: np.stack([st[k][e, t:t + T_CTX] for e, t in zip(ep, t0)])
    X = {"V+P+T+A": np.concatenate([cen, v, g("finger"), g("touch"), g("act")], -1).astype(np.float32),
         "state+A": np.concatenate([g("obj"), g("finger"), g("touch"), g("act")], -1).astype(np.float32)}
    y = np.stack([np.log(st["props"][ep, 0]), st["props"][ep, 1], np.log(st["props"][ep, 2])], 1)
    return X, y, bad


Xa, ya, ba = build(790); Xb, yb, bb = build(780)
print(f"[{CF_MODE}] fit windows {len(ya)} (bad {ba.sum()}), test windows {len(yb)} (bad {bb.sum()})", flush=True)
res = {}
for name in ("V+P+T+A", "state+A"):
    g = gru_probe(Xa[name][~ba], ya[~ba], Xb[name][~bb], yb[~bb])
    res[name] = [round(float(x), 3) for x in g]
    print(f"[{CF_MODE}] {name:8s} GRU recovery on intervention histories  m/g/k {np.round(g, 3)}", flush=True)
json.dump(res, open(RESULTS / f"cert_cf_{CF_MODE}_v2.json", "w"), indent=1); print("saved")
