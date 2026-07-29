"""Train one X-JEPA v4 variant on PokeWorld and evaluate:
  1) ridge probes for hidden props (log m, gamma, log k) on two feature sets:
     zcat  = concat of 16 per-frame embeddings  (what the encoder carries)
     trunk = predictor last hidden state        (integrated system-ID state)
  2) open-loop autoregressive rollout -> decoded object position error
Usage: python train.py --variant V|VF|VX|VFX [--seed 0] [--steps 20000]
"""
import argparse, json, time, pathlib
import numpy as np
import torch

from model import XJEPA, D, DP

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
T_CTX, HORIZON = 16, 8
WIN = T_CTX + HORIZON


def load_split(name):
    img = np.load(DATA / f"{name}_img.npy", mmap_mode="r")
    st = np.load(DATA / f"{name}_state.npz")
    return img, {k: st[k] for k in st.files}


def whiten_stats(state):
    prop = state["finger"].reshape(-1, 4)
    touch = state["touch"].reshape(-1, 7)
    return {
        "prop_mu": prop.mean(0), "prop_sd": prop.std(0) + 1e-6,
        "touch_mu": touch[:, :6].mean(0), "touch_sd": touch[:, :6].std(0) + 1e-6,
    }


def make_batch(img, state, stats, ep_idx, t0, device, T=WIN):
    im = np.stack([img[e, t:t + T] for e, t in zip(ep_idx, t0)])
    pr = np.stack([state["finger"][e, t:t + T] for e, t in zip(ep_idx, t0)])
    to = np.stack([state["touch"][e, t:t + T] for e, t in zip(ep_idx, t0)])
    ac = np.stack([state["act"][e, t:t + T] for e, t in zip(ep_idx, t0)])
    pr = (pr - stats["prop_mu"]) / stats["prop_sd"]
    to = to.copy()
    to[..., :6] = (to[..., :6] - stats["touch_mu"]) / stats["touch_sd"]
    return {
        "img": torch.from_numpy(im).to(device, non_blocking=True).float().div_(255),
        "prop": torch.from_numpy(pr).to(device).float(),
        "touch": torch.from_numpy(to).to(device).float(),
        "act": torch.from_numpy(ac).to(device).float(),
    }


@torch.no_grad()
def encode_and_trunk(model, img, state, stats, ep_idx, t0, device, bs=256):
    """Per-frame embeddings and trunk last hidden for T_CTX windows."""
    zs, hs = [], []
    for s in range(0, len(ep_idx), bs):
        b = make_batch(img, state, stats, ep_idx[s:s + bs], t0[s:s + bs], device, T=T_CTX)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"], b["prop"], b["touch"])
            h = model.pred.trunk(z, b["act"])
        zs.append(z.float().cpu().numpy())
        hs.append(h[:, -1].float().cpu().numpy())
    return np.concatenate(zs), np.concatenate(hs)


def r2(y, p):
    sse = ((y - p) ** 2).sum(0)
    sst = ((y - y.mean(0)) ** 2).sum(0) + 1e-12
    return 1 - sse / sst


def ridge_fit_eval(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-6
    n = len(Xtr)
    cut = int(n * 0.8)
    best = (None, -1e9)
    for alpha in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        w = np.linalg.solve(Xtr[:cut].T @ Xtr[:cut] + alpha * np.eye(Xtr.shape[1]),
                            Xtr[:cut].T @ ((ytr[:cut] - ymu) / ysd))
        r2v = r2((ytr[cut:] - ymu) / ysd, Xtr[cut:] @ w).mean()
        if r2v > best[1]:
            best = (alpha, r2v)
    w = np.linalg.solve(Xtr.T @ Xtr + best[0] * np.eye(Xtr.shape[1]),
                        Xtr.T @ ((ytr - ymu) / ysd))
    pred = (Xte @ w) * ysd + ymu
    return r2(yte, pred), (w, mu, sd, ymu, ysd)


def prop_labels(state, ep_idx):
    props = state["props"][ep_idx]
    return np.stack([np.log(props[:, 0]), props[:, 1], np.log(props[:, 2])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(
        __import__("model").VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    RESULTS.mkdir(exist_ok=True)

    img_tr, st_tr = load_split("train")
    stats = whiten_stats(st_tr)

    model = XJEPA(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 500, 0.5 * (1 + np.cos(np.pi * s / args.steps)) + 1e-2))

    rng = np.random.default_rng(args.seed)
    n_ep, T = img_tr.shape[:2]
    log, t_start = [], time.time()
    model.train()
    for step in range(args.steps):
        ep_idx = rng.integers(0, n_ep, args.bs)
        t0 = rng.integers(0, T - WIN + 1, args.bs)
        batch = make_batch(img_tr, st_tr, stats, ep_idx, t0, device)
        with torch.autocast("cuda", torch.bfloat16):
            losses = model.loss(batch, T_CTX, HORIZON, global_step=step)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            msg = {k: round(float(v), 5) for k, v in losses.items()}
            el = time.time() - t_start
            print(f"[{args.variant} s{args.seed}] step {step} {msg} ({el:.0f}s)", flush=True)
            log.append({"step": step, **msg})

    torch.save(model.state_dict(), RESULTS / f"{args.variant}_s{args.seed}.pt")
    model.eval()
    img_ptr, st_ptr = load_split("probe_tr")
    img_pte, st_pte = load_split("probe_te")

    def probe_windows(img, state):
        n = img.shape[0]
        t0s = np.array([0, 8, 16, 24, 32, 40])
        ep = np.repeat(np.arange(n), len(t0s))
        t0 = np.tile(t0s, n)
        z, h = encode_and_trunk(model, img, state, stats, ep, t0, device)
        contact = np.array([state["touch"][e, t:t + T_CTX, 6].sum() for e, t in zip(ep, t0)])
        y_prop = prop_labels(state, ep)
        pos_lbl = np.stack([state["obj"][e, t:t + T_CTX, :2] for e, t in zip(ep, t0)])
        return z, h, y_prop, pos_lbl, contact

    Ztr, Htr, yptr, postr, ctr = probe_windows(img_ptr, st_ptr)
    Zte, Hte, ypte, poste, cte = probe_windows(img_pte, st_pte)
    mask_tr, mask_te = ctr >= 2, cte >= 2

    out_probe = {}
    for feat_name, Ftr, Fte in (("zcat", Ztr.reshape(len(Ztr), -1), Zte.reshape(len(Zte), -1)),
                                ("trunk", Htr, Hte)):
        r2_all, _ = ridge_fit_eval(Ftr, yptr, Fte, ypte)
        r2_ct, _ = ridge_fit_eval(Ftr[mask_tr], yptr[mask_tr], Fte[mask_te], ypte[mask_te])
        out_probe[feat_name] = {
            "all": dict(zip(("log_mass", "gamma", "log_stiffness"), map(float, r2_all))),
            "contact": dict(zip(("log_mass", "gamma", "log_stiffness"), map(float, r2_ct))),
        }

    # per-frame position probe (encoder health + rollout decoder)
    Zf_tr = Ztr.reshape(-1, D)
    pos_tr = postr.reshape(-1, 2)
    sub = np.random.default_rng(0).choice(len(Zf_tr), min(60000, len(Zf_tr)), replace=False)
    r2_pos, pos_probe = ridge_fit_eval(Zf_tr[sub], pos_tr[sub],
                                       Zte.reshape(-1, D), poste.reshape(-1, 2))
    w, mu, sd, ymu, ysd = pos_probe

    # open-loop rollout: context frames 0..15, generate 16 steps with true actions
    n_te = img_pte.shape[0]
    roll_err = {dlt: [] for dlt in (1, 2, 4, 8, 16)}
    with torch.no_grad():
        for s in range(0, n_te, 256):
            idx = np.arange(s, min(s + 256, n_te))
            b = make_batch(img_pte, st_pte, stats, idx, np.zeros(len(idx), np.int64),
                           device, T=T_CTX)
            with torch.autocast("cuda", torch.bfloat16):
                z_ctx = model.encode(b["img"], b["prop"], b["touch"])
            a_fut = torch.from_numpy(st_pte["act"][idx, T_CTX:T_CTX + 16]).to(device).float()
            z_hat = model.pred.rollout(z_ctx.float(), b["act"], a_fut).cpu().numpy()
            for dlt in roll_err:
                zd = (z_hat[:, dlt - 1] - mu) / sd
                pos_pred = zd @ w * ysd + ymu
                true = st_pte["obj"][idx, T_CTX - 1 + dlt, :2]
                roll_err[dlt].append(np.linalg.norm(pos_pred - true, axis=1))
    static_err = {}
    for dlt in list(roll_err):
        roll_err[dlt] = float(np.concatenate(roll_err[dlt]).mean())
        static_err[dlt] = float(np.linalg.norm(
            st_pte["obj"][:, T_CTX - 1 + dlt, :2] - st_pte["obj"][:, T_CTX - 1, :2],
            axis=1).mean())

    out = {
        "variant": args.variant, "seed": args.seed, "steps": args.steps,
        "params": n_params, "minutes": round((time.time() - t_start) / 60, 1),
        "probe": out_probe,
        "probe_r2_pos_frame": round(float(r2_pos.mean()), 4),
        "n_contact_windows_te": int(mask_te.sum()),
        "rollout_pos_err": roll_err, "static_baseline_err": static_err,
        "log": log,
    }
    path = RESULTS / f"{args.variant}_s{args.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
