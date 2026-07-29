"""e011: cross-embodiment joint training (cfg7 KUKA + cfg1 Flexiv).

One config-blind model over both corpora with per-config whitening; evaluation
per config (IID + held-out-task OOD) against the single-config specialists.
Usage: python joint_train.py --variant VFX [--steps 30000]
"""
import argparse, json, time, pathlib
import numpy as np
import torch

import model as M
from model import XJEPA

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
T_CTX, HORIZON, WIN, ROLL = 16, 8, 24, 16


def clean(a):
    a = np.nan_to_num(a.astype(np.float64), posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(a, 0.1, axis=0), np.percentile(a, 99.9, axis=0)
    return np.clip(a, lo, hi).astype(np.float32)


def load_config(dirname):
    d = HERE / dirname
    ix = np.load(d / "index.npz")
    sen = np.load(d / "sensors.npz")
    c = {
        "img": np.load(d / "images.npy", mmap_mode="r"),
        "state": clean(sen["state"]), "touch": clean(sen["touch"]),
        "action": clean(sen["action"]),
        "start": ix["start"], "length": ix["length"],
        "splits": {k[6:]: ix[k] for k in ix.files if k.startswith("split_")},
    }
    rows = np.concatenate([np.arange(c["start"][e], c["start"][e] + c["length"][e])
                           for e in c["splits"]["train"][:200]])
    c["stats"] = {
        "s_mu": c["state"][rows].mean(0), "s_sd": c["state"][rows].std(0) + 1e-6,
        "t_mu": c["touch"][rows].mean(0), "t_sd": c["touch"][rows].std(0) + 1e-6,
        "a_mu": c["action"][rows].mean(0), "a_sd": c["action"][rows].std(0) + 1e-6,
    }
    return c


CFGS = {}


def window_list(cfg_id, split, min_len=WIN):
    c = CFGS[cfg_id]
    out = []
    for e in c["splits"][split]:
        for t0 in range(0, c["length"][e] - min_len + 1, 4):
            out.append((cfg_id, e, t0))
    return np.array(out, np.int64)


def make_batch(triples, device, T=WIN):
    parts = {"img": [], "prop": [], "touch": [], "act": []}
    rows_all = []
    for cfg_id in np.unique(triples[:, 0]):
        c = CFGS[cfg_id]
        sub = triples[triples[:, 0] == cfg_id]
        rows = np.stack([np.arange(c["start"][e] + t0, c["start"][e] + t0 + T)
                         for _, e, t0 in sub])
        frames = torch.from_numpy(c["img"][rows]).to(device).float().div_(255)
        motion = torch.zeros_like(frames)
        motion[:, 1:] = frames[:, 1:] - frames[:, :-1]
        st = (c["state"][rows] - c["stats"]["s_mu"]) / c["stats"]["s_sd"]
        to = (c["touch"][rows] - c["stats"]["t_mu"]) / c["stats"]["t_sd"]
        ac = (c["action"][rows] - c["stats"]["a_mu"]) / c["stats"]["a_sd"]
        parts["img"].append(torch.stack([frames, motion], 2))
        parts["prop"].append(torch.from_numpy(st).to(device).float())
        parts["touch"].append(torch.from_numpy(to).to(device).float())
        parts["act"].append(torch.from_numpy(ac).to(device).float())
        rows_all.append((cfg_id, rows))
    return {k: torch.cat(v) for k, v in parts.items()}, rows_all


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)


def ridge(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-6
    cut = int(len(Xtr) * 0.8)
    best = (10.0, -1e9)
    for a in (1e-1, 1.0, 10.0, 100.0):
        w = np.linalg.solve(Xtr[:cut].T @ Xtr[:cut] + a * np.eye(Xtr.shape[1]),
                            Xtr[:cut].T @ ((ytr[:cut] - ymu) / ysd))
        v = r2((ytr[cut:] - ymu) / ysd, Xtr[cut:] @ w).mean()
        if np.isfinite(v) and v > best[1]:
            best = (a, v)
    w = np.linalg.solve(Xtr.T @ Xtr + best[0] * np.eye(Xtr.shape[1]),
                        Xtr.T @ ((ytr - ymu) / ysd))
    return r2(yte, (Xte @ w) * ysd + ymu), (w, mu, sd, ymu, ysd)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0.5
    if pos.sum() in (0, len(labels)):
        return float("nan")
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


@torch.no_grad()
def encode_split(model, cfg_id, split, device, bs=192, min_len=T_CTX):
    wl = window_list(cfg_id, split, min_len=max(min_len, WIN))
    Z, H, R = [], [], []
    for s in range(0, len(wl), bs):
        b, rows_all = make_batch(wl[s:s + bs], device, T=T_CTX)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"], b["prop"], b["touch"])
            h = model.pred.trunk(z[:, :-1].float(), b["act"][:, :-1])
        Z.append(z.float().cpu().numpy())
        H.append(h[:, -1].float().cpu().numpy())
        R.append(rows_all[0][1])
    return np.concatenate(Z), np.concatenate(H), np.concatenate(R)


def evaluate_cfg(model, cfg_id, device, split_te):
    c = CFGS[cfg_id]
    out = {}
    Ztr, Htr, Rtr = encode_split(model, cfg_id, "probe_tr", device)
    Zte, Hte, Rte = encode_split(model, cfg_id, split_te, device)
    Zf_tr, Zf_te = Ztr.reshape(-1, M.D), Zte.reshape(-1, M.D)
    sub = np.random.default_rng(0).choice(len(Zf_tr), min(60000, len(Zf_tr)), False)
    r2_t, _ = ridge(Zf_tr[sub], c["touch"][Rtr.reshape(-1)][:, :3][sub],
                    Zf_te, c["touch"][Rte.reshape(-1)][:, :3])
    out["force_r2"] = round(float(r2_t.mean()), 4)
    r2_p, probe = ridge(Zf_tr[sub], c["state"][Rtr.reshape(-1)][:, :3][sub],
                        Zf_te, c["state"][Rte.reshape(-1)][:, :3])
    out["pos_r2"] = round(float(r2_p.mean()), 4)
    w, mu, sd, ymu, ysd = probe
    wl = window_list(cfg_id, split_te, min_len=T_CTX + ROLL)
    d16, stat = [], []
    with torch.no_grad():
        for s in range(0, len(wl), 192):
            b, rows_all = make_batch(wl[s:s + 192], device, T=T_CTX + ROLL)
            rows = rows_all[0][1]
            with torch.autocast("cuda", torch.bfloat16):
                z = model.encode(b["img"][:, :T_CTX], b["prop"][:, :T_CTX],
                                 b["touch"][:, :T_CTX])
                h = model.pred.trunk(z.float(), b["act"][:, :T_CTX])
                aw = b["act"][:, T_CTX - 1:T_CTX - 1 + 16].reshape(len(rows), 1, -1)
                z16 = model.dheads["16"](h[:, -1:].float(), aw)[:, 0].float()
            dec = ((z16.cpu().numpy() - mu) / sd) @ w * ysd + ymu
            true = c["state"][rows[:, T_CTX - 1 + 16], :3]
            base = c["state"][rows[:, T_CTX - 1], :3]
            d16.append(np.linalg.norm(dec - true, axis=1))
            stat.append(np.linalg.norm(base - true, axis=1))
    out["direct16"] = round(float(np.concatenate(d16).mean()), 4)
    out["static16"] = round(float(np.concatenate(stat).mean()), 4)
    fm = np.linalg.norm(c["touch"][:, :3], axis=1)
    lab_tr = np.array([1.0 if (fm[r] < 10 and fm[r + 1:r + 5].max() > 10) else 0.0
                       for r in Rtr[:, -1]])
    lab_te = np.array([1.0 if (fm[r] < 10 and fm[r + 1:r + 5].max() > 10) else 0.0
                       for r in Rte[:, -1]])
    _, (wo, muo, sdo, ymuo, ysdo) = ridge(Htr, lab_tr[:, None], Hte, lab_te[:, None])
    sc = ((Hte - muo) / sdo) @ wo * ysdo + ymuo
    out["onset_auc"] = round(float(auc(sc[:, 0], lab_te)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="VFX")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    RESULTS.mkdir(exist_ok=True)
    print("loading cfg7 + cfg1 ...", flush=True)
    CFGS[0] = load_config("data")
    CFGS[1] = load_config("data_cfg1")
    tag = f"{args.variant}_joint"
    model = XJEPA(args.variant).to(device)
    ckpt = RESULTS / f"{tag}.pt"
    resume = RESULTS / f"{tag}.resume.pt"
    t0 = time.time()
    if not args.eval_only:
        wl = np.concatenate([window_list(0, "train"), window_list(1, "train")])
        rng = np.random.default_rng(0)
        opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 500,
                               0.5 * (1 + np.cos(np.pi * s / args.steps)) + 1e-2))
        start = 0
        if resume.exists():
            try:
                rs = torch.load(resume, weights_only=False)
                model.load_state_dict(rs["model"]); opt.load_state_dict(rs["opt"])
                sched.load_state_dict(rs["sched"]); start = rs["step"] + 1
                print(f"resuming from {start}", flush=True)
            except Exception as e:
                print(f"corrupt resume ({type(e).__name__}), restarting", flush=True)
                resume.unlink()
        model.train()
        for step in range(start, args.steps):
            batch, _ = make_batch(wl[rng.integers(0, len(wl), args.bs)], device)
            with torch.autocast("cuda", torch.bfloat16):
                losses = model.loss(batch, T_CTX, HORIZON, global_step=step,
                                    lam=args.lam)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            if step % 500 == 0 or step == args.steps - 1:
                print(f"[{tag}] {step} total={float(losses['total']):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if step % 1000 == 0 and step > 0:
                tmp = resume.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step}, tmp)
                tmp.replace(resume)
        torch.save(model.state_dict(), ckpt)
        if resume.exists():
            resume.unlink()
    else:
        model.load_state_dict(torch.load(ckpt, weights_only=True))

    model.eval()
    res = {"variant": args.variant, "steps": args.steps,
           "minutes": round((time.time() - t0) / 60, 1)}
    for cid, name in ((0, "cfg7"), (1, "cfg1")):
        res[name] = {"iid": evaluate_cfg(model, cid, device, "probe_te"),
                     "ood": evaluate_cfg(model, cid, device, "ood")}
    (RESULTS / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
