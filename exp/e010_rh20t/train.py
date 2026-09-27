"""Train X-JEPA on RH20T with training-only clipping and evaluate:
  (a) vision-only latent -> current force/torque readout (Kepler contrast)
  (b) multi-horizon ee-position prediction vs static baseline
  (c) contact-onset anticipation AUC (|F|>10N within next 4 steps)
All repeated on the held-out-task OOD split.
Usage: python train.py --variant V|VF|VX|VFX [--seed 0] [--steps 20000]
"""
import argparse, json, os, time, pathlib
import numpy as np
import torch

import model as M
from model import XJEPA, action_windows

HERE = pathlib.Path(__file__).parent
DATA = HERE / os.environ.get("E010_DATA", "data")
TAG_SUFFIX = {"data": "", "data_master": "_m",
              "data_cfg1": "_c1"}.get(os.environ.get("E010_DATA", "data"), "_x")
RESULTS = HERE / "results"
T_CTX, HORIZON, WIN = 16, 8, 24
ROLL = 16

_ix = np.load(DATA / "index.npz")
START, LENGTH = _ix["start"], _ix["length"]
SPLITS = {k[6:]: _ix[k] for k in _ix.files if k.startswith("split_")}
_sen = np.load(DATA / "sensors.npz")


def _clean(a):
    """Real sensor streams contain glitch spikes (>1e19 on cfg1) that overflow
    float32 stats and blow up whitening/ridge; clip per-column to robust range."""
    a = np.nan_to_num(a.astype(np.float64), posinf=0.0, neginf=0.0)
    # clipping thresholds from TRAINING episodes only (the original computed them on all
    # data including held-out tasks: a preprocessing leak flagged in the 2026-09-14 review)
    lo = np.percentile(a[_TRAIN_ROWS], 0.1, axis=0)
    hi = np.percentile(a[_TRAIN_ROWS], 99.9, axis=0)
    return np.clip(a, lo, hi).astype(np.float32)


# data-scale subsets are fixed here, before any statistic is computed, so that clipping and whitening use only the
# subset's own episodes. E010_NEP selects a task-stratified subset of the training split (seeded, shared by all arms).
_NEP = int(os.environ.get("E010_NEP", "0") or 0)
if _NEP:
    _task = _ix["task"] if "task" in _ix.files else None
    _tr = np.array(SPLITS["train"]); _rs = np.random.default_rng(1234)
    if _task is not None:
        _t = _task[_tr]; _keep = []
        for _u in np.unique(_t):
            _idx = _tr[_t == _u]; _n = max(1, int(round(len(_idx) * _NEP / len(_tr))))
            _keep.append(_rs.permutation(_idx)[:_n])
        _keep = np.concatenate(_keep)
        _keep = _rs.permutation(_keep)[:_NEP] if len(_keep) > _NEP else _keep
    else:
        _keep = _rs.permutation(_tr)[:_NEP]
    SPLITS["train"] = np.sort(_keep)
    print(f"[E010_NEP] training subset of {len(SPLITS['train'])} episodes (task-stratified: {_task is not None})", flush=True)
_TRAIN_ROWS = np.concatenate([np.arange(START[e], START[e] + LENGTH[e]) for e in SPLITS["train"]])
STATE, TOUCH, ACTION = (_clean(_sen[k]) for k in ("state", "touch", "action"))
IMG = np.load(DATA / "images.npy", mmap_mode="r")
IMG2 = None      # second-view memmap, set in main() with --cam2


def window_list(split, min_len=WIN):
    out = []
    for e in SPLITS[split]:
        L = LENGTH[e]
        for t0 in range(0, L - min_len + 1, 4):
            out.append((e, t0))
    return np.array(out, np.int64)


def whiten_stats():
    rows = np.concatenate([np.arange(START[e], START[e] + LENGTH[e])
                           for e in SPLITS["train"][:200]])
    return {"s_mu": STATE[rows].mean(0), "s_sd": STATE[rows].std(0) + 1e-6,
            "t_mu": TOUCH[rows].mean(0), "t_sd": TOUCH[rows].std(0) + 1e-6,
            "a_mu": ACTION[rows].mean(0), "a_sd": ACTION[rows].std(0) + 1e-6}


def make_batch(pairs, stats, device, T=WIN):
    rows = np.stack([np.arange(START[e] + t0, START[e] + t0 + T) for e, t0 in pairs])
    im = IMG[rows]                                   # [B,T,96,96] uint8
    frames = torch.from_numpy(im).to(device).float().div_(255)
    motion = torch.zeros_like(frames)
    motion[:, 1:] = frames[:, 1:] - frames[:, :-1]
    chans = [frames, motion]
    if IMG2 is not None:
        f2 = torch.from_numpy(IMG2[rows]).to(device).float().div_(255)
        m2 = torch.zeros_like(f2)
        m2[:, 1:] = f2[:, 1:] - f2[:, :-1]
        chans += [f2, m2]
    st = (STATE[rows] - stats["s_mu"]) / stats["s_sd"]
    to = (TOUCH[rows] - stats["t_mu"]) / stats["t_sd"]
    ac = (ACTION[rows] - stats["a_mu"]) / stats["a_sd"]
    return {"img": torch.stack(chans, 2),
            "prop": torch.from_numpy(st).to(device).float(),
            "touch": torch.from_numpy(to).to(device).float(),
            "act": torch.from_numpy(ac).to(device).float()}, rows


def r2(y, p):
    sse = ((y - p) ** 2).sum(0)
    sst = ((y - y.mean(0)) ** 2).sum(0) + 1e-12
    return 1 - sse / sst


def ridge(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-6
    cut = int(len(Xtr) * 0.8)
    best = (None, -1e9)
    for a in (1e-1, 1.0, 10.0, 100.0):
        w = np.linalg.solve(Xtr[:cut].T @ Xtr[:cut] + a * np.eye(Xtr.shape[1]),
                            Xtr[:cut].T @ ((ytr[:cut] - ymu) / ysd))
        v = r2((ytr[cut:] - ymu) / ysd, Xtr[cut:] @ w).mean()
        if np.isfinite(v) and v > best[1]:
            best = (a, v)
    if best[0] is None:
        best = (10.0, float("nan"))
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
def encode_split(model, split, stats, device, bs=192, min_len=32):
    wl = window_list(split, min_len=min_len)
    Z, H, ROWS = [], [], []
    for s in range(0, len(wl), bs):
        b, rows = make_batch(wl[s:s + bs], stats, device, T=T_CTX)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"], b["prop"], b["touch"])
            h = model.pred.trunk(z[:, :-1].float(), b["act"][:, :-1])
        Z.append(z.float().cpu().numpy())
        H.append(h[:, -1].float().cpu().numpy())
        ROWS.append(rows)
    return np.concatenate(Z), np.concatenate(H), np.concatenate(ROWS), wl


def evaluate(model, stats, device, split_tr="probe_tr", split_te="probe_te"):
    out = {}
    Ztr, Htr, Rtr, _ = encode_split(model, split_tr, stats, device)
    Zte, Hte, Rte, wl_te = encode_split(model, split_te, stats, device)
    # (a) per-frame z -> current touch readout
    Zf_tr, Zf_te = Ztr.reshape(-1, M.D), Zte.reshape(-1, M.D)
    ttr, tte = TOUCH[Rtr.reshape(-1)], TOUCH[Rte.reshape(-1)]
    sub = np.random.default_rng(0).choice(len(Zf_tr), min(60000, len(Zf_tr)), False)
    r2_touch, _ = ridge(Zf_tr[sub], ttr[sub], Zf_te, tte)
    out["touch_readout_r2_force"] = round(float(r2_touch[:3].mean()), 4)
    out["touch_readout_r2_all"] = round(float(r2_touch.mean()), 4)
    # (b) ee-position: per-frame probe + rollout/direct16
    ptr, pte = STATE[Rtr.reshape(-1), :3], STATE[Rte.reshape(-1), :3]
    r2_pos, pos_probe = ridge(Zf_tr[sub], ptr[sub], Zf_te, pte)
    out["pos_readout_r2"] = round(float(r2_pos.mean()), 4)
    w, mu, sd, ymu, ysd = pos_probe
    wl_roll = window_list(split_te, min_len=T_CTX + ROLL)
    roll_err, dir16_err, static_err = {1: [], 4: [], 8: [], 16: []}, [], {1: [], 4: [], 8: [], 16: []}
    with torch.no_grad():
        for s in range(0, len(wl_roll), 192):
            pairs = wl_roll[s:s + 192]
            b, rows = make_batch(pairs, stats, device, T=T_CTX + ROLL)
            with torch.autocast("cuda", torch.bfloat16):
                z_ctx = model.encode(b["img"][:, :T_CTX], b["prop"][:, :T_CTX],
                                     b["touch"][:, :T_CTX])
                zh = model.pred.rollout(z_ctx.float(), b["act"][:, :T_CTX],
                                        b["act"][:, T_CTX:T_CTX + ROLL]).float()
                h = model.pred.trunk(z_ctx.float(), b["act"][:, :T_CTX])
                aw = b["act"][:, T_CTX - 1:T_CTX - 1 + 16].reshape(len(pairs), 1, -1)
                z16 = model.dheads["16"](h[:, -1:].float(), aw)[:, 0].float()
            zh, z16 = zh.cpu().numpy(), z16.cpu().numpy()
            base = STATE[rows[:, T_CTX - 1], :3]
            for dlt in roll_err:
                dec = ((zh[:, dlt - 1] - mu) / sd) @ w * ysd + ymu
                true = STATE[rows[:, T_CTX - 1 + dlt], :3]
                roll_err[dlt].append(np.linalg.norm(dec - true, axis=1))
                static_err[dlt].append(np.linalg.norm(base - true, axis=1))
            dec16 = ((z16 - mu) / sd) @ w * ysd + ymu
            dir16_err.append(np.linalg.norm(
                dec16 - STATE[rows[:, T_CTX - 1 + 16], :3], axis=1))
    out["roll_pos_err"] = {k: round(float(np.concatenate(v).mean()), 4)
                           for k, v in roll_err.items()}
    out["static_err"] = {k: round(float(np.concatenate(v).mean()), 4)
                         for k, v in static_err.items()}
    out["direct16_err"] = round(float(np.concatenate(dir16_err).mean()), 4)
    # (c) contact-onset anticipation from trunk state
    def onset_labels(R):
        fm = np.linalg.norm(TOUCH[:, :3], axis=1)
        lab = []
        for row_end in R[:, -1]:
            fut = fm[row_end + 1: row_end + 5]
            now = fm[row_end]
            lab.append(1.0 if (now < 10 and len(fut) and fut.max() > 10) else 0.0)
        return np.array(lab)
    ltr, lte = onset_labels(Rtr), onset_labels(Rte)
    _, (wo, muo, sdo, ymuo, ysdo) = ridge(Htr, ltr[:, None], Hte, lte[:, None])
    sc = ((Hte - muo) / sdo) @ wo * ysdo + ymuo
    out["contact_onset_auc"] = round(float(auc(sc[:, 0], lte)), 4)
    out["onset_rate"] = round(float(lte.mean()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--cam2", type=str, default="")
    ap.add_argument("--nep", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    if args.nep:
        assert _NEP == args.nep, "set E010_NEP=<n> in the environment so that clipping statistics use the subset (see top of file)"

    torch.manual_seed(args.seed)
    device = "cuda"
    RESULTS.mkdir(exist_ok=True)
    stats = whiten_stats()
    tag = f"{args.variant}_s{args.seed}{TAG_SUFFIX}_tc"   # _tc: train-only clipping thresholds
    in_ch = 2
    if args.cam2:
        global IMG2
        IMG2 = np.load(HERE / args.cam2 / "images.npy", mmap_mode="r")
        in_ch = 4
        tag += "_2cam"
    if args.nep:
        tag += f"_n{args.nep}"
    if args.steps != 20000:
        tag += f"_st{args.steps // 1000}k"
    model = XJEPA(args.variant, in_ch=in_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ckpt = RESULTS / f"{tag}.pt"
    log, t0 = [], time.time()
    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, weights_only=True))
    else:
        wl = window_list("train")
        rng = np.random.default_rng(args.seed)
        opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 500,
                               0.5 * (1 + np.cos(np.pi * s / args.steps)) + 1e-2))
        resume_path = RESULTS / f"{tag}.resume.pt"
        start_step = 0
        if resume_path.exists():
            rs = torch.load(resume_path, weights_only=False)
            model.load_state_dict(rs["model"])
            opt.load_state_dict(rs["opt"])
            sched.load_state_dict(rs["sched"])
            start_step, log = rs["step"] + 1, rs["log"]
            print(f"[{tag}] resuming from step {start_step}", flush=True)
        model.train()
        for step in range(start_step, args.steps):
            pairs = wl[rng.integers(0, len(wl), args.bs)]
            batch, _ = make_batch(pairs, stats, device)
            with torch.autocast("cuda", torch.bfloat16):
                losses = model.loss(batch, T_CTX, HORIZON, global_step=step, lam=args.lam)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            if step % 500 == 0 or step == args.steps - 1:
                msg = {k: round(float(v), 5) for k, v in losses.items()}
                print(f"[{tag}] step {step} {msg} ({time.time()-t0:.0f}s)", flush=True)
                log.append({"step": step, **msg})
            if step % 1000 == 0 and step > 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step, "log": log},
                           resume_path)
        torch.save(model.state_dict(), ckpt)
        if resume_path.exists():
            resume_path.unlink()

    model.eval()
    res = {"variant": args.variant, "seed": args.seed, "params": n_params,
           "minutes": round((time.time() - t0) / 60, 1),
           "iid": evaluate(model, stats, device),
           "ood": evaluate(model, stats, device, "probe_tr", "ood"),
           "log": log}
    (RESULTS / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
