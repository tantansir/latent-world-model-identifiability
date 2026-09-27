"""Train one X-JEPA v5 (multi-timescale) variant on PokeWorld and evaluate:
  1) ridge + MLP probes for hidden props on trunk features; subsets
     all / contact (>=2 contact frames) / moving (mean obj speed > 0.15)
  2) autoregressive rollout AND direct Delta=16 head -> decoded position error
Usage: python train.py --variant V|VX|VXt|VFX [--seed 0] [--steps 20000]
"""
import argparse, json, time, pathlib
import numpy as np
import torch

import model as M
from model import XJEPA, action_windows

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "e001_xmodal_jepa" / "data"
RESULTS = HERE / "results"
T_CTX, HORIZON = 16, 8
WIN = T_CTX + HORIZON
FLOW_NET = None      # set to a frozen FlowNet to append flow input channels
NODIFF = False       # --nodiff: grayscale frame only, no temporal-difference channel
LOGSPEED = False     # append log|flow| channel (linearizes glide decay)
OBSIN = False        # observation-derived object state (pixel centroid + mean flow vector) replaces the true state
OBSVEL = "flow"      # velocity estimate used in obsstate / objspeed: "flow" (mean flow vector) or "cent" (centroid backward difference)


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
    finger_xy = pr[..., :2].copy()                      # raw finger position (proprioception), arena units
    pr = (pr - stats["prop_mu"]) / stats["prop_sd"]
    to = to.copy()
    to[..., :6] = (to[..., :6] - stats["touch_mu"]) / stats["touch_sd"]
    pp = state["props"][ep_idx]
    props = np.stack([np.log(pp[:, 0]), pp[:, 1], np.log(pp[:, 2])], 1)
    frames = torch.from_numpy(im).to(device, non_blocking=True).float().div_(255)
    motion = torch.zeros_like(frames)
    motion[:, 1:] = frames[:, 1:] - frames[:, :-1]
    chans = [frames] if NODIFF else [frames, motion]
    if FLOW_NET is not None:
        B, T = frames.shape[:2]
        with torch.no_grad():
            f0 = frames[:, :-1].reshape(-1, 1, 64, 64)
            f1 = frames[:, 1:].reshape(-1, 1, 64, 64)
            fl = torch.nn.functional.interpolate(
                FLOW_NET(f0, f1), size=64, mode="bilinear", align_corners=False)
        fl = fl.reshape(B, T - 1, 2, 64, 64).div_(4.0)
        fl = torch.cat([torch.zeros_like(fl[:, :1]), fl], 1)
        chans += [fl[:, :, 0], fl[:, :, 1]]
        if LOGSPEED:
            chans += [torch.log(fl.pow(2).sum(2).sqrt() + 0.05)]
        # observation-derived object speed: mean flow magnitude over object pixels
        # (object rendered at 0.55, finger at 1.0; anti-aliased edges excluded)
        omask = ((frames > 0.3) & (frames < 0.8)).float()          # object pixels (finger renders at 1.0)
        # the finger's anti-aliased edge ring also falls in the 0.3-0.8 band and drags the centroid toward the
        # finger (error 0.8 px near, 3.6 px far); carve a disk around the proprioceptive finger position out
        yy, xx = torch.meshgrid(torch.arange(64, device=frames.device), torch.arange(64, device=frames.device), indexing="ij")
        fxy = torch.from_numpy(finger_xy).to(frames.device).float()                        # [B,T,2]
        far = (((xx[None, None].float() + 0.5) / 64 - fxy[..., 0, None, None]) ** 2
               + ((yy[None, None].float() + 0.5) / 64 - fxy[..., 1, None, None]) ** 2) > (0.06 + 2 / 64) ** 2
        omask = omask * far.float()
        msum = omask.sum((2, 3)) + 1e-6
        # mean flow VECTOR over the object, then its magnitude (the mean of magnitudes is biased upward by noise)
        vmean = torch.stack([(fl[:, :, c] * omask).sum((2, 3)) / msum for c in (0, 1)], -1)   # [B,T,2]
        objspeed = vmean.norm(dim=-1)
        # observation-derived object state: pixel centroid (arena units) + flow velocity
        cx = ((xx[None, None].float() + 0.5) / 64 * omask).sum((2, 3)) / msum
        cy = ((yy[None, None].float() + 0.5) / 64 * omask).sum((2, 3)) / msum
        if OBSVEL == "cent":
            # state input: centroid backward-difference velocity (arena units / s); zero at t=0 like the flow
            cpos = torch.stack([cx, cy], -1)
            vmean = torch.zeros_like(cpos)
            vmean[:, 1:] = (cpos[:, 1:] - cpos[:, :-1]) / 0.05
            # target speed: 4-frame backward displacement / (4 dt), so the 4-step log-speed ratio becomes the
            # displacement ratio log|c_{t+4}-c_t| - log|c_t-c_{t-4}| (= -4*gamma*dt on pure glide; R2 0.94 vs the
            # true target on probe_te). Zero (-> masked) for t<4 and where contact occurred in [t-4, t].
            objspeed = torch.zeros_like(cx)
            objspeed[:, 4:] = (cpos[:, 4:] - cpos[:, :-4]).norm(dim=-1) / 0.2
            contact = torch.from_numpy(np.stack([state["touch"][e, t:t + T, 6] for e, t in zip(ep_idx, t0)])).to(frames.device).float()
            back_free = (1 - contact).unfold(1, 5, 1).prod(-1)       # no contact over t-4..t, indexed by t-4
            objspeed[:, 4:] = objspeed[:, 4:] * back_free
        obsstate = torch.stack([cx, cy, vmean[..., 0], vmean[..., 1]], -1)   # [B,T,4]
    out = {
        "img": torch.stack(chans, 2),              # [B,T,C,H,W]
        "prop": torch.from_numpy(pr).to(device).float(),
        "touch": torch.from_numpy(to).to(device).float(),
        "act": torch.from_numpy(ac).to(device).float(),
        "props": torch.from_numpy(props).to(device).float(),   # (log m, g, log k)
        "obj": torch.from_numpy(np.stack([state["obj"][e, t:t + T]
                                          for e, t in zip(ep_idx, t0)])).to(device).float(),
    }
    out["obj_true"] = out["obj"]           # simulator state, always kept for the state-based targets
    out["touch_raw"] = torch.from_numpy(np.stack([state["touch"][e, t:t + T] for e, t in zip(ep_idx, t0)])).to(device).float()
    out["finger_xy"] = torch.from_numpy(finger_xy).to(device).float()
    if FLOW_NET is not None:
        out["objspeed"] = objspeed
        out["obsstate"] = obsstate
        if OBSIN:
            out["obj"] = obsstate          # the encoder's object-state input is now observation-derived
    return out


@torch.no_grad()
def encode_and_trunk(model, img, state, stats, ep_idx, t0, device, bs=256):
    """Per-frame embeddings and trunk last hidden for T_CTX windows."""
    zs, hs = [], []
    for s in range(0, len(ep_idx), bs):
        # 4 extra frames so the vr self-conditioning has future actions at the last context position
        b = make_batch(img, state, stats, ep_idx[s:s + bs], t0[s:s + bs], device, T=T_CTX + 4)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.encode(b["img"][:, :T_CTX], b["prop"][:, :T_CTX], b["touch"][:, :T_CTX], b["obj"][:, :T_CTX])
            pc = model.props_w(b) if model.use_oracle else None
            h = model.pred.trunk(z, b["act"][:, :T_CTX], pc)
            h = model.condition(h, b["act"])
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
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--motw", type=float, default=0.0)
    ap.add_argument("--flow", action="store_true")
    ap.add_argument("--nodiff", action="store_true", help="grayscale frame only (no temporal-difference input channel)")
    ap.add_argument("--data2", action="store_true")
    ap.add_argument("--dsuf", type=str, default="")
    ap.add_argument("--psuf", type=str, default="")
    ap.add_argument("--glidew", type=float, default=1.0,
                    help="loss-share reallocation: sample zero-contact (glide) "
                         "episodes with this relative weight")
    ap.add_argument("--touchw", type=float, default=4.0,
                    help="touch-loss weight at contact frames (default 1+3=4); "
                         "<4 starves the stiffness information channel")
    ap.add_argument("--recon", action="store_true",
                    help="reconstruction-objective baseline: pixel targets")
    ap.add_argument("--sysid", action="store_true",
                    help="add supervised system-ID auxiliary head (oracle control)")
    ap.add_argument("--logspeed", action="store_true",
                    help="append log-speed input channel (requires --flow)")
    ap.add_argument("--nll", action="store_true",
                    help="heteroscedastic Gaussian-NLL prediction heads")
    ap.add_argument("--oracle", action="store_true",
                    help="feed true whitened params into the predictor conditioning")
    ap.add_argument("--mul", action="store_true",
                    help="multiplicative-interaction branch in the predictor")
    ap.add_argument("--vr", action="store_true",
                    help="log-speed-ratio (Delta=4, glide frames) target head")
    ap.add_argument("--obsvel", default="flow", choices=["flow", "cent"],
                    help="observation-derived velocity: mean flow vector or centroid backward difference")
    ap.add_argument("--vrflow", action="store_true",
                    help="--vr with the object speed taken from the frozen flow sensor (requires --flow)")
    ap.add_argument("--tctx", type=int, default=16,
                    help="predictor context length (window = tctx + 8)")
    ap.add_argument("--vr16", action="store_true", help="16-step log-speed-ratio target on glide frames")
    ap.add_argument("--physhead", action="store_true", help="structured (gamma, v) glide displacement head")
    ap.add_argument("--phys_mode", default="full", choices=["full", "fixg", "truev", "multi"])
    ap.add_argument("--dispmlp", action="store_true", help="plain MLP displacement head with the physics head's supervision")
    ap.add_argument("--vv", action="store_true", help="control: plain future-speed target (Delta=4, glide frames)")
    ap.add_argument("--vd", action="store_true", help="plain speed-difference target |v(t+4)|-|v(t)| (Coulomb's first-order coordinate)")
    ap.add_argument("--kr", action="store_true", help="stiffness coordinate target log|f_last|-1.5 log overlap at t+1 (contact frames)")
    ap.add_argument("--ksep", action="store_true", help="control: log|f_last| and log overlap as two separate targets")
    ap.add_argument("--mr", action="store_true", help="mass coordinate target log|f|-log(dv/dt+gamma v)_par at t+1 (off-wall contact frames)")
    ap.add_argument("--mr0", action="store_true", help="mass coordinate without the drag term (coupled-parameter control)")
    ap.add_argument("--msep", action="store_true", help="control: log|f| and log a_par as two separate targets")
    ap.add_argument("--obscoord", action="store_true", help="k/m coordinate targets from the observation-derived state (requires --obsin)")
    ap.add_argument("--offwall", action="store_true", help="glide targets (vr/vd) only on off-wall windows (no wall bounces)")
    ap.add_argument("--vn", action="store_true", help="normalized deceleration target (|v(t)|-|v(t+4)|)/|v(t)| on glide frames")
    ap.add_argument("--selfcond", action="store_true", help="inject the model's own vr/kr/mr estimates into the trunk state (self-conditioning)")
    ap.add_argument("--acorr", action="store_true", help="corrected action windows a_{t+1..t+delta}")
    ap.add_argument("--ihead", action="store_true", help="interaction (MLP) prediction heads over [h, phi(A)]")
    ap.add_argument("--gha0", action="store_true", help="glide heads without future-action input")
    ap.add_argument("--deltas", type=str, default="4,16", help="extra latent horizons ('' = single-step)")
    ap.add_argument("--xdeltas", type=str, default="4", help="horizons of the cross-modal heads ('' = next-step only)")
    ap.add_argument("--extra1", type=int, default=0, help="horizon control: extra Delta=1 latent heads")
    ap.add_argument("--noproj", action="store_true", help="horizon control: the Delta=1 prediction uses an action-conditioned head (next action) instead of the action-free projector; needs --extra1 >= 1")
    ap.add_argument("--hnorm", action="store_true", help="horizon control: normalize total latent-prediction weight")
    ap.add_argument("--obsin", action="store_true", help="observation-derived object state as encoder input (requires --flow); implies objin")
    ap.add_argument("--oinoise", type=float, default=0.0, help="pixels of Gaussian noise on the objin state")
    ap.add_argument("--vrnoise", type=float, default=0.0, help="Gaussian noise (scaled units) on the vr target")
    ap.add_argument("--vrcond", action="store_true",
                    help="inject the vr head's estimate back into the trunk state (requires --vr)")
    ap.add_argument("--objin", action="store_true",
                    help="privileged object state (pos, vel) as an extra encoder input (precision-layer judgement)")
    ap.add_argument("--probe_every", type=int, default=0,
                    help="ridge-probe the trunk every N steps (acquisition curve)")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    global T_CTX, WIN
    T_CTX, WIN = args.tctx, args.tctx + HORIZON
    M.set_dim(args.dim)
    tag = f"{args.variant}_s{args.seed}"
    if args.dim != 128:
        tag += f"_d{args.dim}"
    if args.lam != 0.1:
        tag += f"_l{args.lam}"
    if args.motw > 0:
        tag += f"_mw{args.motw:g}"
    if args.nodiff:
        assert not args.flow, "--nodiff is for the plain frame regime"
        global NODIFF
        NODIFF = True
        tag += "_nd"
    if args.flow:
        tag += "_fl"
        global FLOW_NET
        from flownet import FlowNet
        FLOW_NET = FlowNet().to("cuda")
        FLOW_NET.load_state_dict(torch.load(RESULTS / "flownet.pt", weights_only=True))
        FLOW_NET.eval()
    if args.data2:
        tag += "_g"
    if args.dsuf:
        tag += f"_d{args.dsuf}"
    if args.glidew != 1.0:
        tag += f"_gw{args.glidew:g}"
    if args.touchw != 4.0:
        tag += f"_tw{args.touchw:g}"
    if args.logspeed:
        assert args.flow, "--logspeed requires --flow"
        global LOGSPEED
        LOGSPEED = True
        tag += "_ls"
    if args.recon:
        tag += "_rc"
    if args.sysid:
        tag += "_sid"
    if args.nll:
        tag += "_nll"
    if args.oracle:
        tag += "_or"
    if args.mul:
        tag += "_mul"
    if args.vrflow:
        assert args.flow, "--vrflow requires --flow"
        args.vr = True
        tag += "_vrf"
    elif args.vr:
        tag += "_vr"
    if args.steps != 20000:
        tag += f"_st{args.steps // 1000}k"
    if args.tctx != 16:
        tag += f"_ctx{args.tctx}"
    if args.obsvel == "cent":
        global OBSVEL
        OBSVEL = "cent"
    if args.obsin:
        assert args.flow, "--obsin requires --flow"
        global OBSIN
        OBSIN = True
        args.objin = True
        tag += "_obs"
    if args.obsvel == "cent":
        tag += "_cv"
    if args.objin and not args.obsin:
        tag += "_oi"
    if args.vrcond:
        assert args.vr, "--vrcond requires --vr"
        tag += "_vc"
    if args.vr16:
        tag += "_vr16"
    if args.physhead:
        tag += "_ph" + ("" if args.phys_mode == "full" else args.phys_mode)
    if args.dispmlp:
        tag += "_dm"
    if args.vv:
        tag += "_vv"
    if args.vd:
        tag += "_vd"
    if args.kr:
        tag += "_kr"
    if args.ksep:
        tag += "_ksep"
    if args.mr:
        tag += "_mr"
    if args.mr0:
        tag += "_mr0"
    if args.msep:
        tag += "_msep"
    if args.obscoord:
        assert args.obsin, "--obscoord requires --obsin"
        tag += "_oc"
    if args.offwall:
        tag += "_ow"
    if args.vn:
        tag += "_vn"
    if args.selfcond:
        tag += "_sc"
    if args.acorr:
        tag += "_ac"
        M.ACORR = True
    if args.ihead:
        tag += "_ih"
        M.IHEAD = True
    if args.gha0:
        tag += "_ga0"
        M.GHA0 = True
    if args.deltas != "4,16":
        M.set_deltas([int(x) for x in args.deltas.split(",") if x])
        tag += "_D" + (args.deltas.replace(",", "-") or "1")
    if args.xdeltas != "4":
        M.set_deltas_x([int(x) for x in args.xdeltas.split(",") if x])
        tag += "_Dx" + (args.xdeltas.replace(",", "-") or "1")
    if args.extra1:
        M.EXTRA1 = args.extra1
        tag += f"_e1x{args.extra1}"
    if args.noproj:
        assert args.extra1 >= 1, "--noproj needs --extra1 >= 1"
        M.NOPROJ = True
        tag += "_np"
    if args.hnorm:
        M.HNORM = True
        tag += "_hn"
    if args.oinoise > 0:
        tag += f"_oin{args.oinoise:g}"
    if args.vrnoise > 0:
        tag += f"_vrn{args.vrnoise:g}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    RESULTS.mkdir(exist_ok=True)

    train_name = f"train{args.dsuf}" if args.dsuf else ("train2" if args.data2 else "train")
    img_tr, st_tr = load_split(train_name)
    stats = whiten_stats(st_tr)

    in_ch = (4 if args.flow else 2) + (1 if args.logspeed else 0) - (1 if args.nodiff else 0)
    model = XJEPA(args.variant, in_ch=in_ch,
                  recon=args.recon, sysid=args.sysid, nll=args.nll,
                  oracle=args.oracle, mul=args.mul, vr=args.vr, vr_flow=args.vrflow,
                  objin=args.objin, vrcond=args.vrcond, vr16=args.vr16, physhead=args.physhead,
                  phys_mode=args.phys_mode, dispmlp=args.dispmlp,
                  vv=args.vv, oinoise_px=args.oinoise, vrnoise=args.vrnoise,
                  kr=args.kr, mr=(args.mr or args.mr0), mr_nogamma=args.mr0, ksep=args.ksep, msep=args.msep,
                  vd=args.vd, obscoord=args.obscoord, offwall=args.offwall, vn=args.vn, selfcond=args.selfcond).to(device)
    if args.flow and args.vv and not args.vr:
        model.vr_flow = True          # plain-speed control takes its speed from the flow sensor too
    if args.objin:
        if OBSIN:
            rs = np.random.default_rng(0)
            mb = make_batch(img_tr, st_tr, stats, rs.integers(0, img_tr.shape[0], 256),
                            rs.integers(0, img_tr.shape[1] - WIN + 1, 256), device)
            ob = mb["obj"].reshape(-1, 4).cpu().numpy()
        else:
            ob = st_tr["obj"].reshape(-1, 4)
        model.obj_mu.copy_(torch.tensor(ob.mean(0), dtype=torch.float32))
        model.obj_sd.copy_(torch.tensor(ob.std(0) + 1e-6, dtype=torch.float32))
    model.mot_w = args.motw
    model.touch_cw = args.touchw
    if args.sysid or args.oracle:
        lbl = np.stack([np.log(st_tr["props"][:, 0]), st_tr["props"][:, 1],
                        np.log(st_tr["props"][:, 2])], 1)
        model.sid_mu.copy_(torch.tensor(lbl.mean(0), dtype=torch.float32))
        model.sid_sd.copy_(torch.tensor(lbl.std(0) + 1e-6, dtype=torch.float32))
    ep_p = None
    if args.glidew != 1.0:
        csum = st_tr["touch"][:, :, 6].sum(1)
        wvec = np.where(csum == 0, args.glidew, 1.0)
        ep_p = wvec / wvec.sum()
        print(f"[glidew] {int((csum == 0).sum())}/{len(csum)} zero-contact episodes "
              f"at relative weight {args.glidew:g} -> "
              f"{ep_p[csum == 0].sum():.2%} of sampled loss mass", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    if getattr(model, "use_motion_tgt", False):
        rs = np.random.default_rng(0)
        mb = make_batch(img_tr, st_tr, stats, rs.integers(0, img_tr.shape[0], 256),
                        rs.integers(0, img_tr.shape[1] - WIN + 1, 256), device)
        mot = torch.nn.functional.avg_pool2d(
            mb["img"][:, :, 1].abs().reshape(-1, 1, 64, 64), 8).reshape(-1, 64)
        model.motion_mu.copy_(mot.mean(0))
        model.motion_sd.fill_(float(mot.std().clamp_min(0.01)))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 500, 0.5 * (1 + np.cos(np.pi * s / args.steps)) + 1e-2))

    img_ptr, st_ptr = load_split(f"probe{args.psuf}_tr")
    img_pte, st_pte = load_split(f"probe{args.psuf}_te")

    def probe_windows(img, state):
        n = img.shape[0]
        t0s = np.array([t for t in (0, 8, 16, 24, 32, 40) if t + T_CTX + 4 <= img.shape[1]])
        ep = np.repeat(np.arange(n), len(t0s))
        t0 = np.tile(t0s, n)
        z, h = encode_and_trunk(model, img, state, stats, ep, t0, device)
        contact = np.array([state["touch"][e, t:t + T_CTX, 6].sum() for e, t in zip(ep, t0)])
        speed = np.array([np.linalg.norm(state["obj"][e, t:t + T_CTX, 2:], axis=1).mean()
                          for e, t in zip(ep, t0)])
        y_prop = prop_labels(state, ep)
        pos_lbl = np.stack([state["obj"][e, t:t + T_CTX, :2] for e, t in zip(ep, t0)])
        return z, h, y_prop, pos_lbl, contact, speed

    def quick_probe():
        model.eval()
        _, Htr_, ytr_, _, ctr_, sptr_ = probe_windows(img_ptr, st_ptr)
        _, Hte_, yte_, _, cte_, spte_ = probe_windows(img_pte, st_pte)
        model.train()
        subs = {"all": (np.ones(len(Htr_), bool), np.ones(len(Hte_), bool)),
                "contact": (ctr_ >= 2, cte_ >= 2),
                "moving": (sptr_ > 0.15, spte_ > 0.15),
                "glide": ((sptr_ > 0.15) & (ctr_ == 0), (spte_ > 0.15) & (cte_ == 0))}
        res = {}
        for nm, (mtr, mte) in subs.items():
            rr, _ = ridge_fit_eval(Htr_[mtr], ytr_[mtr], Hte_[mte], yte_[mte])
            res[nm] = dict(zip(("log_mass", "gamma", "log_stiffness"), map(lambda v: round(float(v), 4), rr)))
        return res

    probe_curve = []
    rng = np.random.default_rng(args.seed)
    n_ep, T = img_tr.shape[:2]
    log, t_start = [], time.time()
    ckpt = RESULTS / f"{tag}.pt"
    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"[{tag}] loaded checkpoint, eval only", flush=True)
    model.train()
    for step in range(0 if not args.eval_only else args.steps, args.steps):
        ep_idx = (rng.choice(n_ep, args.bs, p=ep_p) if ep_p is not None
                  else rng.integers(0, n_ep, args.bs))
        t0 = rng.integers(0, T - WIN + 1, args.bs)
        batch = make_batch(img_tr, st_tr, stats, ep_idx, t0, device, T=WIN)
        with torch.autocast("cuda", torch.bfloat16):
            losses = model.loss(batch, T_CTX, HORIZON, global_step=step, lam=args.lam)
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
        if args.probe_every and step % args.probe_every == 0 and step > 0:
            pr = quick_probe()
            probe_curve.append({"step": step, **pr})
            print(f"[probe@{step}] contact {pr['contact']} glide {pr['glide']}", flush=True)

    torch.save(model.state_dict(), ckpt)
    model.eval()

    Ztr, Htr, yptr, postr, ctr, sptr = probe_windows(img_ptr, st_ptr)
    Zte, Hte, ypte, poste, cte, spte = probe_windows(img_pte, st_pte)

    def mlp_probe(Xtr, ytr, Xte, yte):
        import torch.nn as nn
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
                            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3)).to(device)
        o = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
        best_r2, best_pred = -1e9, None
        for i in range(4001):
            idx = torch.randint(0, len(xtr), (256,), device=device)
            l = (net(xtr[idx]) - ytr_[idx]).pow(2).mean()
            o.zero_grad(); l.backward(); o.step()
            if i % 200 == 0:
                with torch.no_grad():
                    pv = net(xva).cpu().numpy() * ysd + ymu
                    rv = r2(yva, pv).mean()
                    if rv > best_r2:
                        best_r2 = rv
                        best_pred = net(xte).cpu().numpy() * ysd + ymu
        return r2(yte, best_pred)

    glide_tr = (sptr > 0.15) & (ctr == 0)
    glide_te = (spte > 0.15) & (cte == 0)
    subsets = {"all": (np.ones(len(Htr), bool), np.ones(len(Hte), bool)),
               "contact": (ctr >= 2, cte >= 2),
               "moving": (sptr > 0.15, spte > 0.15),
               "glide": (glide_tr, glide_te)}
    out_probe = {"trunk": {}, "trunk_mlp": {}}
    for sub_name, (mtr, mte) in subsets.items():
        r2_r, _ = ridge_fit_eval(Htr[mtr], yptr[mtr], Hte[mte], ypte[mte])
        r2_m = mlp_probe(Htr[mtr], yptr[mtr], Hte[mte], ypte[mte])
        out_probe["trunk"][sub_name] = dict(zip(("log_mass", "gamma", "log_stiffness"),
                                                map(float, r2_r)))
        out_probe["trunk_mlp"][sub_name] = dict(zip(("log_mass", "gamma", "log_stiffness"),
                                                    map(float, r2_m)))

    # velocity probe: is object velocity even decodable from the trunk?
    def vel_labels(state, img_n):
        t0s = np.array([t for t in (0, 8, 16, 24, 32, 40) if t + T_CTX + 4 <= 64])
        ep = np.repeat(np.arange(img_n), len(t0s))
        t0 = np.tile(t0s, img_n)
        return np.stack([state["obj"][e, t + T_CTX - 1, 2:] for e, t in zip(ep, t0)])

    vtr = vel_labels(st_ptr, img_ptr.shape[0])
    vte = vel_labels(st_pte, img_pte.shape[0])
    r2_vel, _ = ridge_fit_eval(Htr, vtr, Hte, vte)
    mv_tr, mv_te = sptr > 0.15, spte > 0.15
    r2_vel_mv, _ = ridge_fit_eval(Htr[mv_tr], vtr[mv_tr], Hte[mv_te], vte[mv_te])
    out_probe["velocity_r2"] = {"all": round(float(r2_vel.mean()), 4),
                                "moving": round(float(r2_vel_mv.mean()), 4)}

    # per-frame position probe (encoder health + rollout decoder)
    Zf_tr = Ztr.reshape(-1, M.D)
    pos_tr = postr.reshape(-1, 2)
    sub = np.random.default_rng(0).choice(len(Zf_tr), min(60000, len(Zf_tr)), replace=False)
    r2_pos, pos_probe = ridge_fit_eval(Zf_tr[sub], pos_tr[sub],
                                       Zte.reshape(-1, M.D), poste.reshape(-1, 2))
    w, mu, sd, ymu, ysd = pos_probe

    # open-loop rollout: context frames 0..15, generate 16 steps with true
    # actions; plus the direct Delta=16 head from position 15. Skipped in recon
    # mode (the latent AR projector / z-Delta heads receive no gradient there).
    n_te = img_pte.shape[0]
    roll_err = {dlt: [] for dlt in (1, 2, 4, 8, 16)}
    direct16 = []
    if not args.recon:
        with torch.no_grad():
            for s in range(0, n_te, 256):
                idx = np.arange(s, min(s + 256, n_te))
                b = make_batch(img_pte, st_pte, stats, idx, np.zeros(len(idx), np.int64),
                               device, T=T_CTX)
                with torch.autocast("cuda", torch.bfloat16):
                    z_ctx = model.encode(b["img"], b["prop"], b["touch"], b["obj"])
                a_fut = torch.from_numpy(st_pte["act"][idx, T_CTX:T_CTX + 16]).to(device).float()
                pc = model.props_w(b) if model.use_oracle else None
                z_hat = model.pred.rollout(z_ctx.float(), b["act"], a_fut, pc).cpu().numpy()
                for dlt in roll_err:
                    zd = (z_hat[:, dlt - 1] - mu) / sd
                    pos_pred = zd @ w * ysd + ymu
                    true = st_pte["obj"][idx, T_CTX - 1 + dlt, :2]
                    roll_err[dlt].append(np.linalg.norm(pos_pred - true, axis=1))
                h_ctx = model.pred.trunk(z_ctx.float(), b["act"], pc)
                h_ctx = model.condition(h_ctx, torch.cat([b["act"], a_fut], 1))
                aw = torch.from_numpy(
                    st_pte["act"][idx, M.act_offset(T_CTX):M.act_offset(T_CTX) + 16]).to(device).float()
                aw = aw.reshape(len(idx), 1, 32)
                if "16" not in model.dheads:
                    direct16.append(np.full(len(idx), np.nan)); continue
                zd16 = model.dheads["16"](h_ctx[:, -1:], aw)[:, 0].cpu().numpy()
                pos16 = ((zd16 - mu) / sd) @ w * ysd + ymu
                true16 = st_pte["obj"][idx, T_CTX - 1 + 16, :2]
                direct16.append(np.linalg.norm(pos16 - true16, axis=1))
    static_err = {}
    for dlt in list(roll_err):
        roll_err[dlt] = (float(np.concatenate(roll_err[dlt]).mean())
                         if roll_err[dlt] else None)
        static_err[dlt] = float(np.linalg.norm(
            st_pte["obj"][:, T_CTX - 1 + dlt, :2] - st_pte["obj"][:, T_CTX - 1, :2],
            axis=1).mean())

    out = {
        "variant": args.variant, "seed": args.seed, "steps": args.steps,
        "params": n_params, "minutes": round((time.time() - t_start) / 60, 1),
        "probe": out_probe,
        "probe_r2_pos_frame": round(float(r2_pos.mean()), 4),
        "n_contact_windows_te": int((cte >= 2).sum()),
        "n_moving_windows_te": int((spte > 0.15).sum()),
        "rollout_pos_err": roll_err, "static_baseline_err": static_err,
        "direct16_pos_err": (float(np.concatenate(direct16).mean())
                             if direct16 else None),
        "log": log,
        "probe_curve": probe_curve,
    }
    out["dim"], out["lam"] = args.dim, args.lam
    path = RESULTS / f"{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
