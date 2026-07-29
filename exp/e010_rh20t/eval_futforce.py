"""Passthrough-immune force metric: FUTURE force prediction.
Frame-level force readout is bounded by input passthrough (RAND control), but
forecasting force at t+Delta under known actions requires dynamics knowledge a
random encoder cannot have. Evaluates head_touch (Delta=1) and
dheads_touch['4'] (Delta=4) on probe_te and ood windows against a persistence
baseline (touch stays at its last observed value). Force dims = touch[:3].
"""
import json, pathlib, sys
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import train as TR
from model import XJEPA

device = "cuda"


def r2_cols(y, p):
    sse = ((y - p) ** 2).sum(0)
    sst = ((y - y.mean(0)) ** 2).sum(0) + 1e-12
    return 1 - sse / sst


out = {}
for variant, suf in [(v, s) for v in ("VFX", "VX")
                     for s in ("_n800", "_n2000", "")]:
    tag = f"{variant}_s0_c1{suf}"
    model = XJEPA(variant).to(device).eval()
    model.load_state_dict(torch.load(TR.RESULTS / f"{tag}.pt", weights_only=True))
    stats = TR.whiten_stats()
    res = {}
    T = TR.T_CTX + 4
    for split in ("probe_te", "ood"):
        wl = TR.window_list(split, min_len=T)
        p1, p4, t1, t4, last, epids = [], [], [], [], [], []
        with torch.no_grad():
            for s in range(0, len(wl), 192):
                epids.append(wl[s:s + 192][:, 0])
                b, rows = TR.make_batch(wl[s:s + 192], stats, device, T=T)
                with torch.autocast("cuda", torch.bfloat16):
                    z = model.encode(b["img"][:, :TR.T_CTX], b["prop"][:, :TR.T_CTX],
                                     b["touch"][:, :TR.T_CTX])
                    h = model.pred.trunk(z.float(), b["act"][:, :TR.T_CTX])
                    y1 = model.head_touch(h[:, -1])
                    aw = b["act"][:, TR.T_CTX - 1:TR.T_CTX + 3].reshape(len(rows), 1, -1)
                    y4 = model.dheads_touch["4"](h[:, -1:].float(), aw)[:, 0]
                p1.append(y1.float().cpu().numpy())
                p4.append(y4.float().cpu().numpy())
                t1.append(b["touch"][:, TR.T_CTX].cpu().numpy())
                t4.append(b["touch"][:, TR.T_CTX + 3].cpu().numpy())
                last.append(b["touch"][:, TR.T_CTX - 1].cpu().numpy())
        p1, p4 = np.concatenate(p1), np.concatenate(p4)
        t1, t4, last = np.concatenate(t1), np.concatenate(t4), np.concatenate(last)
        eps = np.concatenate(epids)

        def boot_all(y, pm, pp, n_boot=1000):
            """Paired episode bootstrap: model, persistence, and their
            difference under the SAME episode resamples."""
            uq = np.unique(eps)
            idxs = {e: np.where(eps == e)[0] for e in uq}
            rng = np.random.default_rng(0)
            vm, vp, vd = [], [], []
            for _ in range(n_boot):
                sel = rng.choice(uq, len(uq), replace=True)
                ii = np.concatenate([idxs[e] for e in sel])
                a = float(r2_cols(y[ii, :3], pm[ii, :3]).mean())
                b = float(r2_cols(y[ii, :3], pp[ii, :3]).mean())
                vm.append(a); vp.append(b); vd.append(a - b)
            pct = lambda v: [round(float(np.percentile(v, q)), 4)
                             for q in (2.5, 97.5)]
            return pct(vm), pct(vp), pct(vd)

        ci_m, ci_p, ci_d = boot_all(t4, p4, last)
        res[split] = {
            "d1_model": round(float(r2_cols(t1[:, :3], p1[:, :3]).mean()), 4),
            "d1_persist": round(float(r2_cols(t1[:, :3], last[:, :3]).mean()), 4),
            "d4_model": round(float(r2_cols(t4[:, :3], p4[:, :3]).mean()), 4),
            "d4_persist": round(float(r2_cols(t4[:, :3], last[:, :3]).mean()), 4),
            "d4_model_ci95": ci_m,
            "d4_persist_ci95": ci_p,
            "d4_diff_ci95": ci_d,
            "n": int(len(t1)),
        }
        print(tag, split, res[split], flush=True)
    out[tag] = res
(TR.RESULTS / "futforce.json").write_text(json.dumps(out, indent=1))
print("done", flush=True)
