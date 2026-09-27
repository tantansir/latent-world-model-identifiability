"""Shared evaluation setup (checkpoint from argv[1], data, per-frame position readout). Extracted from analyze_gamma.py."""
import sys, pathlib
import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import model as M
from model import XJEPA
from train import (load_split, whiten_stats, make_batch, ridge_fit_eval,
                   T_CTX, RESULTS)

device = "cuda"
DT16 = 16 * 0.05
_tag_probe = sys.argv[1] if len(sys.argv) > 1 else ""
img_tr, st_tr = load_split("train2" if "_g" in _tag_probe else "trainB" if "_dB" in _tag_probe else "trainC" if "_dC" in _tag_probe else "train")
stats = whiten_stats(st_tr)
img_ptr, st_ptr = load_split("probe_tr")
img_pte, st_pte = load_split("probe_te")

TAG = sys.argv[1] if len(sys.argv) > 1 else "VFX_s0_l0.005"
FH_SUF = "_fht" if TAG.endswith("_fht") else "_fhh" if TAG.endswith("_fhh") else "_fhr" if TAG.endswith("_fhr") else "_fh" if TAG.endswith("_fh") else ""
BASE_TAG = TAG[:-len(FH_SUF)] if FH_SUF else TAG
M.ACORR = "_ac" in BASE_TAG
M.IHEAD = "_ih" in BASE_TAG or FH_SUF == "_fh"      # retrained plain heads are interaction heads
M.GHA0 = "_ga0" in BASE_TAG
AOFF = M.act_offset(T_CTX)
if "_fl" in BASE_TAG:
    import train as TR
    from flownet import FlowNet
    fn = FlowNet().to(device)
    fn.load_state_dict(torch.load(RESULTS / "flownet.pt", weights_only=True))
    fn.eval()
    TR.FLOW_NET = fn
in_ch = 4 if "_fl" in BASE_TAG else 2
if "_ls" in BASE_TAG:
    import train as TR
    TR.LOGSPEED = True
    in_ch += 1
if "_obs" in BASE_TAG:
    import train as TR
    TR.OBSIN = True
if "_cv" in BASE_TAG:
    import train as TR
    TR.OBSVEL = "cent"
model = XJEPA(BASE_TAG.split("_")[0], in_ch=in_ch, vr=("_vr" in BASE_TAG), vr_flow=("_vrf" in TAG or ("_fl" in TAG and "_vv" in BASE_TAG)),
              mul=("_mul" in BASE_TAG), vv=("_vv" in BASE_TAG),
              oracle=("_or" in BASE_TAG), objin=("_oi" in TAG or "_obs" in BASE_TAG), vrcond=("_vc" in BASE_TAG),
              vr16=("_vr16" in BASE_TAG), physhead=("_ph" in BASE_TAG),
              phys_mode=("fixg" if "_phfixg" in TAG else "truev" if "_phtruev" in TAG else "multi" if "_phmulti" in TAG else "full"),
              dispmlp=("_dm" in BASE_TAG),
              kr=("_kr" in BASE_TAG), ksep=("_ksep" in BASE_TAG), mr=("_mr" in BASE_TAG), mr_nogamma=("_mr0" in BASE_TAG),
              msep=("_msep" in BASE_TAG), vd=("_vd" in BASE_TAG), obscoord=("_oc" in BASE_TAG), offwall=("_ow" in BASE_TAG), vn=("_vn" in BASE_TAG), selfcond=("_sc" in BASE_TAG)).to(device)
if FH_SUF in ("_fht", "_fhh", "_fhr"):
    from train_heads_lib import ThetaHead
    model.dheads = torch.nn.ModuleDict({str(d): ThetaHead(d, M.D) for d in M.DELTAS}).to(device)
    model.use_theta_heads = True
    if FH_SUF in ("_fhh", "_fhr"):
        _r = torch.load(RESULTS / f"{TAG}_ridge.pt", weights_only=True)
        model.thetahat_W = (_r["mu"].to(device), _r["sd"].to(device), _r["W"].to(device))
        model.thetahat_off = _r["off"].to(device) if "off" in _r else torch.zeros(3, device=device)
        model.use_thetahat = True
elif FH_SUF == "_fh":
    model.dheads = torch.nn.ModuleDict({str(d): M.DeltaHead(d) for d in M.DELTAS}).to(device)
model.load_state_dict(torch.load(RESULTS / f"{TAG}.pt", weights_only=True), strict=False)
model.eval()
print(f"checkpoint: {TAG}", flush=True)


@torch.no_grad()
def frame_z(img, state, ep, t0, bs=256):
    zs = []
    for s in range(0, len(ep), bs):
        b = make_batch(img, state, stats, ep[s:s + bs], t0[s:s + bs], device, T=T_CTX)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"], b["prop"], b["touch"], b["obj"])
        zs.append(z.float().cpu().numpy())
    return np.concatenate(zs)


# per-frame position probe fit on probe_tr (same protocol as train.py)
n_tr = img_ptr.shape[0]
t0s = np.array([0, 8, 16, 24, 32, 40])
ep = np.repeat(np.arange(n_tr), len(t0s))
t0 = np.tile(t0s, n_tr)
Z = frame_z(img_ptr, st_ptr, ep, t0)
pos = np.stack([st_ptr["obj"][e, t:t + T_CTX, :2] for e, t in zip(ep, t0)])
Zf, posf = Z.reshape(-1, M.D), pos.reshape(-1, 2)
sub = np.random.default_rng(0).choice(len(Zf), 60000, replace=False)
_, (w, mu, sd, ymu, ysd) = ridge_fit_eval(Zf[sub], posf[sub], Zf[:100], posf[:100])
