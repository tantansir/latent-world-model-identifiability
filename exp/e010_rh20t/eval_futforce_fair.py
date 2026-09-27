"""Future-force readout evaluation for RH20T Tables 4 and 16.

Every frozen representation uses the same MLP readout (256-128 units),
optimizer, and probe episodes to predict force four steps ahead. The reported
:h condition uses the latent history alone; :h+a additionally supplies the
future action window. Baselines include persistence, raw sensor inputs,
a history GRU, and an untrained fused-input model.

E010_STRICT=1 replaces the last context action with the previous one because
the recorded action contains the next end-effector pose. E010_LABFIX=1 uses
common full-training-split label statistics for the 800- and 2,947-episode
models, while model inputs retain their own training-subset preprocessing.

Usage (Bash, after training all four variants for the selected seed):
E010_DATA=data_cfg1 E010_TAGSUF=_tc E010_SEED=0 E010_STRICT=1 E010_LABFIX=1 \
    python eval_futforce_fair.py
Add E010_NEP=800 to evaluate checkpoints trained on the 800-episode subset.
"""
import json, pathlib, sys, os
import numpy as np
import torch, torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import train as TR
from model import XJEPA

device = "cuda"
T = TR.T_CTX + 4
torch.manual_seed(0)


def r2_cols(y, p):
    sse = ((y - p) ** 2).sum(0)
    sst = ((y - y.mean(0)) ** 2).sum(0) + 1e-12
    return 1 - sse / sst


STRICT = os.environ.get("E010_STRICT", "0") == "1"     # strict history: the trunk at t does not receive a_t (= recorded pose_{t+1})
# common evaluation target across training sizes (Codex round 3): the force label (and the persistence prediction) are
# clipped and whitened with the FULL training split's statistics regardless of E010_NEP, so that the 800- and 2,947-episode
# arms are scored on the same numbers; model inputs keep their own training-subset preprocessing.
LABFIX = os.environ.get("E010_LABFIX", "0") == "1"
if LABFIX:
    _ixf = np.load(TR.DATA / "index.npz"); _ftr = np.array(_ixf["split_train"])
    _rows_full = np.concatenate([np.arange(TR.START[e], TR.START[e] + TR.LENGTH[e]) for e in _ftr])
    _raw = np.nan_to_num(np.load(TR.DATA / "sensors.npz")["touch"].astype(np.float64), posinf=0.0, neginf=0.0)
    _lo, _hi = np.percentile(_raw[_rows_full], 0.1, axis=0), np.percentile(_raw[_rows_full], 99.9, axis=0)
    _tl = np.clip(_raw, _lo, _hi)
    _r200 = np.concatenate([np.arange(TR.START[e], TR.START[e] + TR.LENGTH[e]) for e in _ftr[:200]])
    TOUCH_LAB = ((_tl - _tl[_r200].mean(0)) / (_tl[_r200].std(0) + 1e-6)).astype(np.float32)
    print("[LABFIX] labels use full-split clipping/whitening", flush=True)
SEED = os.environ.get("E010_SEED", "0"); NEP = os.environ.get("E010_NEP", "")
CK_SUF = f"{TR.TAG_SUFFIX}{os.environ.get('E010_TAGSUF', '')}" + (f"_n{NEP}" if NEP else "")
OUT_SUF = "_lab" if os.environ.get("E010_LABFIX", "0") == "1" else ""


def collect(split, model=None):
    """Returns dict of numpy arrays for every window of the split."""
    wl = TR.window_list(split, min_len=T)
    stats = TR.whiten_stats()
    H, A, Y, LAST, RAW, HIST = [], [], [], [], [], []
    with torch.no_grad():
        for s in range(0, len(wl), 256):
            b, rows = TR.make_batch(wl[s:s + 256], stats, device, T=T)
            if model is not None:
                with torch.autocast("cuda", torch.bfloat16):
                    z = model.encode(b["img"][:, :TR.T_CTX], b["prop"][:, :TR.T_CTX],
                                     b["touch"][:, :TR.T_CTX])
                    a_ctx = b["act"][:, :TR.T_CTX]
                    if STRICT:
                        # strict history: the last context action a_t (= recorded pose_{t+1}) is replaced by a_{t-1} (= pose_t);
                        # earlier positions keep their actions, which are poses <= pose_t and therefore part of the history
                        a_ctx = torch.cat([a_ctx[:, :-1], a_ctx[:, -2:-1]], 1)
                    h = model.pred.trunk(z.float(), a_ctx)
                H.append(h[:, -1].float().cpu().numpy())
            A.append(b["act"][:, TR.T_CTX - 1:TR.T_CTX + 3].reshape(len(rows), -1).cpu().numpy())
            if LABFIX:
                Y.append(TOUCH_LAB[rows[:, TR.T_CTX + 3], :3]); LAST.append(TOUCH_LAB[rows[:, TR.T_CTX - 1], :3])
            else:
                Y.append(b["touch"][:, TR.T_CTX + 3, :3].cpu().numpy())
                LAST.append(b["touch"][:, TR.T_CTX - 1, :3].cpu().numpy())
            RAW.append(torch.cat([b["touch"][:, TR.T_CTX - 1], b["prop"][:, TR.T_CTX - 1]], -1).cpu().numpy())
            a_hist = b["act"][:, :TR.T_CTX]
            if STRICT:
                a_hist = torch.cat([a_hist[:, :-1], a_hist[:, -2:-1]], 1)
            HIST.append(torch.cat([b["touch"][:, :TR.T_CTX], b["prop"][:, :TR.T_CTX], a_hist], -1).cpu().numpy())
    out = {"A": np.concatenate(A), "Y": np.concatenate(Y), "LAST": np.concatenate(LAST),
           "RAW": np.concatenate(RAW), "HIST": np.concatenate(HIST), "ep": wl[:, 0]}
    if H:
        out["H"] = np.concatenate(H)
    return out


def fit_mlp(Xtr, ytr, Xte_list, steps=6000, seq=False):
    """Same head, optimiser and budget for every representation."""
    mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-6
    xtr = torch.tensor((Xtr - mu) / sd, device=device, dtype=torch.float32)
    ytr_ = torch.tensor((ytr - ymu) / ysd, device=device, dtype=torch.float32)
    cut = int(len(xtr) * 0.9)
    if seq:
        class G(nn.Module):
            def __init__(self, d_in, d_a):
                super().__init__()
                self.gru = nn.GRU(d_in, 128, num_layers=2, batch_first=True)
                self.head = nn.Sequential(nn.Linear(128 + d_a, 128), nn.GELU(), nn.Linear(128, 3))
            def forward(self, x):
                seq_, a = x
                h, _ = self.gru(seq_)
                return self.head(torch.cat([h[:, -1], a], -1))
        net = G(Xtr.shape[-1], 0).to(device)
    else:
        net = nn.Sequential(nn.Linear(Xtr.shape[-1], 256), nn.GELU(), nn.Linear(256, 128),
                            nn.GELU(), nn.Linear(128, 3)).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    best, best_state = -1e9, None
    for i in range(steps):
        idx = torch.randint(0, cut, (512,), device=device)
        xb = xtr[idx]
        pred = net((xb, xb[:, -1, :0])) if seq else net(xb)
        l = (pred - ytr_[idx]).pow(2).mean()
        opt.zero_grad(); l.backward(); opt.step()
        if i % 250 == 0:
            with torch.no_grad():
                xv = xtr[cut:]
                pv = net((xv, xv[:, -1, :0])) if seq else net(xv)
                rv = float(r2_cols(ytr_[cut:].cpu().numpy(), pv.cpu().numpy()).mean())
            if rv > best:
                best, best_state = rv, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    outs = []
    with torch.no_grad():
        for Xte in Xte_list:
            xt = torch.tensor((Xte - mu) / sd, device=device, dtype=torch.float32)
            p = net((xt, xt[:, -1, :0])) if seq else net(xt)
            outs.append(p.cpu().numpy() * ysd + ymu)
    return outs


def main():
    splits = ["probe_tr", "probe_te", "ood"]
    base = {sp: collect(sp) for sp in splits}
    res = {}; PRED = {}
    def report(name, preds):
        res[name] = {sp: round(float(r2_cols(base[sp]["Y"], p).mean()), 4)
                     for sp, p in zip(splits[1:], preds)}
        PRED[name] = {sp: np.asarray(p, dtype=np.float32) for sp, p in zip(splits[1:], preds)}
        print(name, res[name], flush=True)

    report("persist", [base["probe_te"]["LAST"], base["ood"]["LAST"]])
    Xa = {sp: np.concatenate([base[sp]["RAW"], base[sp]["A"]], 1) for sp in splits}
    report("raw+a", fit_mlp(Xa["probe_tr"], base["probe_tr"]["Y"], [Xa["probe_te"], Xa["ood"]]))
    Xr = {sp: base[sp]["RAW"] for sp in splits}
    report("raw", fit_mlp(Xr["probe_tr"], base["probe_tr"]["Y"], [Xr["probe_te"], Xr["ood"]]))
    Xh = {sp: np.concatenate([base[sp]["HIST"].reshape(len(base[sp]["HIST"]), -1), base[sp]["A"]], 1)
          for sp in splits}
    report("rawhist+a", fit_mlp(Xh["probe_tr"], base["probe_tr"]["Y"], [Xh["probe_te"], Xh["ood"]]))
    Xg = {sp: base[sp]["HIST"] for sp in splits}
    report("rawGRU", fit_mlp(Xg["probe_tr"], base["probe_tr"]["Y"], [Xg["probe_te"], Xg["ood"]], seq=True))

    for variant in ("V", "VX", "VF", "VFX"):
        for tagname, load in ((f"{variant}", True), (f"{variant}_rand", False)):
            if not load and variant != "VFX":
                continue
            model = XJEPA(variant).to(device).eval()
            if load:
                ck = TR.RESULTS / f"{variant}_s{SEED}{CK_SUF}.pt"
                model.load_state_dict(torch.load(ck, weights_only=True))
            feats = {sp: collect(sp, model) for sp in splits}
            Hh = {sp: feats[sp]["H"] for sp in splits}
            report(f"{tagname}:h", fit_mlp(Hh["probe_tr"], base["probe_tr"]["Y"], [Hh["probe_te"], Hh["ood"]]))
            Hha = {sp: np.concatenate([feats[sp]["H"], base[sp]["A"]], 1) for sp in splits}
            report(f"{tagname}:h+a", fit_mlp(Hha["probe_tr"], base["probe_tr"]["Y"], [Hha["probe_te"], Hha["ood"]]))
            del model; torch.cuda.empty_cache()
    (TR.RESULTS / f"futforce_fair{CK_SUF}_s{SEED}{'_strict' if STRICT else ''}{OUT_SUF}.json").write_text(json.dumps(res, indent=1))
    # per-window predictions for paired bootstraps across arms / scales (same window order for every model)
    np.savez_compressed(TR.RESULTS / f"futforce_pred{CK_SUF}_s{SEED}{'_strict' if STRICT else ''}{OUT_SUF}.npz",
                        Y_te=base["probe_te"]["Y"], Y_ood=base["ood"]["Y"], ep_te=base["probe_te"]["ep"], ep_ood=base["ood"]["ep"],
                        **{f"{k}__{sp}": v for k, d in PRED.items() for sp, v in d.items()})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
