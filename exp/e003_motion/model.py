"""X-JEPA v6 (e003): motion input channel on top of multi-timescale heads.

Mechanistic motivation from e002: object velocity is barely decodable from
per-frame embeddings (position residual ~ same order as one-step displacement,
velocity SNR ~ 1), so gamma - which requires velocity ratios - stays invisible
even under Delta=16 pressure. Biological fix: the retina computes motion
(magnocellular pathway). Input becomes 2 channels: [img_t, img_t - img_{t-1}].
Hypothesis: velocity R^2 jumps, gamma emerges (even for vision-only V).
"""
import torch
import torch.nn as nn

D = 128          # default latent embedding dim (overridable via set_dim)
DP = 192         # predictor width


def set_dim(d):
    """Override the module-level latent dim before constructing models."""
    global D
    D = d


def sigreg(x, global_step, num_slices=1024):
    """LeJEPA Algorithm 1: Epps-Pulley statistic on random 1-d projections."""
    x = x.float()
    g = torch.Generator(device=x.device.type)
    g.manual_seed(int(global_step))
    A = torch.randn((x.size(1), num_slices), generator=g, device=x.device)
    A = A / A.norm(p=2, dim=0, keepdim=True)
    t = torch.linspace(-5, 5, 17, device=x.device)
    exp_f = torch.exp(-0.5 * t ** 2)
    x_t = (x @ A).unsqueeze(2) * t
    ecf = (1j * x_t).exp().mean(0)
    err = (ecf - exp_f).abs().square().mul(exp_f)
    T = torch.trapezoid(err, t, dim=1) * x.size(0)
    return T.mean()


class FrameEncoder(nn.Module):
    """Per-frame multimodal encoder -> z_t via BatchNorm projector."""

    def __init__(self, use_pt, in_ch=2, use_obj=False):
        super().__init__()
        self.use_pt = use_pt
        self.use_obj = use_obj
        ch = (32, 64, 128, 256)
        layers, c_in = [], in_ch      # [frame, diff] (+2 flow channels with --flow)
        for c in ch:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.GELU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.vis_fc = nn.Sequential(nn.Flatten(), nn.Linear(256 * 4 * 4, 192), nn.GELU())
        if use_pt:
            self.enc_prop = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 96), nn.GELU())
            self.enc_touch = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 96), nn.GELU())
            fuse_in = 192 + 96 + 96
        else:
            fuse_in = 192
        if use_obj:
            # precision-layer judgement: the true object state (pos, vel) as an
            # extra per-frame input, so the encoder's sub-pixel imprecision is removed
            self.enc_obj = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 96), nn.GELU())
            fuse_in += 96
        self.fuse = nn.Sequential(nn.Linear(fuse_in, 192), nn.GELU(), nn.Linear(192, D))
        self.bn = nn.BatchNorm1d(D)

    def forward(self, img, prop, touch, obj=None):   # img [B,T,C,H,W]
        B, T = img.shape[:2]
        f = self.vis_fc(self.conv(img.reshape(B * T, img.shape[2], *img.shape[3:])))
        if self.use_pt:
            f = torch.cat([f, self.enc_prop(prop.reshape(B * T, -1)),
                           self.enc_touch(touch.reshape(B * T, -1))], -1)
        if self.use_obj:
            f = torch.cat([f, self.enc_obj(obj.reshape(B * T, -1))], -1)
        z = self.bn(self.fuse(f))
        return z.reshape(B, T, D)


class AdaLNBlock(nn.Module):
    def __init__(self, d, heads, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(4 * d, d))
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c, mask):
        s1, b1, g1, s2, b2, g2 = self.mod(c).chunk(6, -1)
        h = self.ln1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = self.ln2(x) * (1 + s2) + b2
        x = x + g2 * self.mlp(h)
        return x


class MulBranch(nn.Module):
    """Multiplicative interaction (Jayakumar et al., ICLR 2020): a low-rank
    bilinear form x -> O((U x) * (V x)), zero-initialised output so the
    predictor starts identical to the additive baseline."""

    def __init__(self, d, r=64):
        super().__init__()
        self.u = nn.Linear(d, r)
        self.v = nn.Linear(d, r)
        self.o = nn.Linear(r, d)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x):
        return self.o(self.u(x) * self.v(x))


class Predictor(nn.Module):
    """Causal transformer over embedding history; AdaLN action conditioning.
    Input positions t=0..T-1 (z_t, a_t) -> trunk h_t -> projector -> z_{t+1}.
    props (optional, [B,3] whitened true parameters) is added to the AdaLN
    conditioning: the oracle-conditioned judgement experiment."""

    def __init__(self, n_layers=4, heads=8, max_t=64, mul=False):
        super().__init__()
        self.inp = nn.Linear(D, DP)
        self.pos = nn.Parameter(torch.zeros(max_t, DP))
        nn.init.normal_(self.pos, std=0.02)
        self.a_emb = nn.Sequential(nn.Linear(2, DP), nn.GELU(), nn.Linear(DP, DP))
        self.p_emb = nn.Sequential(nn.Linear(3, DP), nn.GELU(), nn.Linear(DP, DP))
        self.blocks = nn.ModuleList([AdaLNBlock(DP, heads) for _ in range(n_layers)])
        self.muls = nn.ModuleList([MulBranch(DP) for _ in range(n_layers)]) if mul else None
        self.ln_f = nn.LayerNorm(DP, elementwise_affine=False)
        self.proj = nn.Linear(DP, D)
        self.bn = nn.BatchNorm1d(D)

    def trunk(self, z_seq, a_seq, props=None):  # [B,T,D], [B,T,2], [B,3]
        B, T = z_seq.shape[:2]
        x = self.inp(z_seq) + self.pos[None, :T]
        c = self.a_emb(a_seq)
        if props is not None:
            c = c + self.p_emb(props)[:, None, :]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=z_seq.device), 1)
        for i, blk in enumerate(self.blocks):
            x = blk(x, c, mask)
            if self.muls is not None:
                x = x + self.muls[i](x)
        return self.ln_f(x)                     # [B,T,DP]

    def project(self, h):                       # trunk -> next-frame embedding
        B, T = h.shape[:2]
        z = self.bn(self.proj(h).reshape(B * T, D))
        return z.reshape(B, T, D)

    @torch.no_grad()
    def rollout(self, z_ctx, a_ctx, a_future, props=None):
        """Autoregressive generation. z_ctx [B,Tc,D] embeddings of observed
        frames, a_ctx [B,Tc,2] their aligned actions (position t consumes
        (z_t, a_t) and emits z_{t+1}), a_future [B,H,2] actions for the
        generated steps. Returns predicted embeddings [B,H,D]."""
        z_seq = z_ctx
        a_all = torch.cat([a_ctx, a_future], 1)
        outs = []
        for _ in range(a_future.shape[1]):
            h = self.trunk(z_seq, a_all[:, :z_seq.shape[1]], props)
            z_next = self.project(h[:, -1:])
            outs.append(z_next)
            z_seq = torch.cat([z_seq, z_next], 1)
        return torch.cat(outs, 1)


ACORR = False   # corrected action windows: a_{t+1..t+delta} (act[t] produced state t); default keeps the paper's a_{t..t+delta-1}
IHEAD = False   # interaction heads: 2-layer MLP over [h, phi(A)] instead of the additive linear map (lets parameters modulate action effects)
GHA0 = False    # glide heads (vr / vd / vn) receive no future-action input (finger actions cannot affect a free-gliding object)


class DeltaHead(nn.Module):
    """Predict z_{t+delta} from trunk h_t conditioned on the action window (a_{t..t+delta-1}, or a_{t+1..t+delta} under ACORR)."""

    def __init__(self, delta, d_out=None, bn=True):
        super().__init__()
        d_out = D if d_out is None else d_out   # resolve AFTER set_dim overrides
        self.delta = delta
        self.noact = False
        self.a_mlp = nn.Sequential(nn.Linear(2 * delta, 128), nn.GELU())
        if IHEAD:
            self.out = nn.Sequential(nn.Linear(DP + 128, 256), nn.GELU(), nn.Linear(256, d_out))
        else:
            self.out = nn.Linear(DP + 128, d_out)
        self.bn = nn.BatchNorm1d(d_out) if bn else nn.Identity()

    def forward(self, h_valid, a_win):          # [B,Tv,DP], [B,Tv,2*delta]
        B, Tv = h_valid.shape[:2]
        if self.noact:
            a_win = torch.zeros_like(a_win)
        y = self.out(torch.cat([h_valid, self.a_mlp(a_win)], -1))
        return self.bn(y.reshape(B * Tv, -1)).reshape(B, Tv, -1)


def action_windows(act, delta, tv):
    """[B,T,2] -> [B,tv,2*delta]: flattened a_{t..t+delta-1} (ACORR: a_{t+1..t+delta}) for t=0..tv-1."""
    src = act[:, 1:] if ACORR else act
    w = src.unfold(1, delta, 1)                 # [B, T'-delta+1, 2, delta]
    assert w.shape[1] >= tv, f"action window too short: {w.shape[1]} < {tv}"
    return w[:, :tv].permute(0, 1, 3, 2).reshape(act.shape[0], tv, 2 * delta)


def act_offset(t_ctx):
    """index of the first action in the window that predicts from context position t_ctx-1"""
    return t_ctx if ACORR else t_ctx - 1


class PixelDecoder(nn.Module):
    """Future-frame deconv decoder for the reconstruction-objective baseline:
    trunk h_t (+ action window for delta>1) -> frame pixels at t+delta. Same
    encoder/trunk as the JEPA variants; only the prediction TARGET differs
    (pixels instead of latents), isolating the objective-family comparison."""

    def __init__(self, deltas=None):
        super().__init__()
        deltas = (1,) + DELTAS if deltas is None else deltas
        self.adapters = nn.ModuleDict()
        self.a_mlps = nn.ModuleDict()
        for d in deltas:
            if d == 1:
                self.adapters["1"] = nn.Linear(DP, 256 * 4 * 4)
            else:
                self.a_mlps[str(d)] = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU())
                self.adapters[str(d)] = nn.Linear(DP + 128, 256 * 4 * 4)
        ch = (256, 128, 64, 32)
        ups = []
        for c_in, c_out in zip(ch[:-1], ch[1:]):
            ups += [nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1), nn.GELU()]
        ups += [nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)]
        self.up = nn.Sequential(*ups)

    def forward(self, h, delta, a_win=None):    # h [N,DP], a_win [N,2*delta]
        if delta == 1:
            f = self.adapters["1"](h)
        else:
            f = self.adapters[str(delta)](
                torch.cat([h, self.a_mlps[str(delta)](a_win)], -1))
        return self.up(f.reshape(-1, 256, 4, 4))[:, 0]      # [N,64,64]


VARIANTS = {
    #        inputs V+P+T, touch target, proprio target, motion target
    "V":    (False, False, False, False),
    "VF":   (True,  False, False, False),
    "VX":   (False, True,  True,  False),
    "VFX":  (True,  True,  True,  False),
    "VXt":  (False, True,  False, False),
    "VXp":  (False, False, True,  False),
    "VM":   (False, False, False, True),    # motion-image targets only
    "VFXM": (True,  True,  True,  True),    # everything + motion targets
}


VR_SCALE = 0.25           # log-speed-ratio target scale (std of -gamma*4*DT over the prior)
VD_SCALE = 0.15           # plain speed-difference target scale
KR_MU, KR_SD = 7.351, 0.75      # stiffness coordinate stats on training contact frames
MR_MU, MR_SD = 0.283, 0.516     # mass coordinate stats on training off-wall contact frames
DELTAS = (4, 16)          # extra z-horizons on top of the default Delta=1
DELTAS_X = (4,)           # extra horizons for cross-modal heads
EXTRA1 = 0                # horizon control: number of extra Delta=1 latent heads (same total weight as the multi-horizon heads)
NOPROJ = False            # horizon control: the Delta=1 latent prediction is made by the first extra Delta=1 head, which receives
                          # the next action a_{t+1}; the action-free projector is not trained
HNORM = False             # horizon control: scale the latent-head weights so the total prediction weight equals the single-step model's


def set_deltas(d):
    global DELTAS
    DELTAS = tuple(d)


def set_deltas_x(d):
    global DELTAS_X
    DELTAS_X = tuple(d)


class XJEPA(nn.Module):
    def __init__(self, variant, in_ch=2, recon=False, sysid=False, nll=False,
                 oracle=False, mul=False, vr=False, vr_flow=False, objin=False, vrcond=False,
                 vr16=False, physhead=False, phys_mode="full", dispmlp=False,
                 vv=False, oinoise_px=0.0, vrnoise=0.0,
                 kr=False, mr=False, mr_nogamma=False, ksep=False, msep=False, vd=False, obscoord=False, offwall=False, vn=False, selfcond=False):
        super().__init__()
        # selfcond: the model's own coordinate estimates (vr, kr, mr heads, whichever exist) are injected back into
        # the trunk state that feeds the prediction heads (zero-initialized linear), so the estimated parameters can be
        # USED by the dynamics heads as conditioning variables - the oracle does this with the true parameters
        self.selfcond = selfcond
        # vn: normalized deceleration target (|v(t)| - |v(t+4)|)/|v(t)| - dimensionless like vr, first-order in the
        # parameter like vd (drag: 1 - e^{-0.2 gamma}; Coulomb: 0.2 mu / |v(t)|). Separates "log" from "scale-free".
        self.use_vn = vn
        if vn:
            self.head_vn = DeltaHead(4, 1, bn=False)
        # offwall: glide targets (vr / vd) only on windows where the object stays inside [0.13, 0.87]^2 - wall bounces
        # (damped penalty contacts, not in the touch flag) otherwise contaminate the target: on the training data the
        # log-speed ratio matches -0.2*gamma with R^2 0.35 over all free windows and 1.00 off-wall
        self.offwall = offwall
        self.obscoord = obscoord        # k/m coordinates from the observation-derived object state (centroid, centroid velocity)
        # first-order coordinate targets for the other two parameters (1-step forecasts, contact frames):
        #   kr  : log|f_last(t+1)| - 1.5 log overlap(t+1)  = log k   (Hertz penalty f = k d^1.5, R^2 0.89 in data)
        #   mr  : log|f_avg(t+1)| - log a_par(t+1), a = dv/dt + gamma*vbar projected on f  = log m (R^2 0.99 off-wall)
        #   ksep/msep : the two ingredients of the coordinate as separate targets (is the combination needed?)
        #   mr_nogamma: the mass coordinate without the drag term (coupled-parameter control)
        #   vd  : plain speed difference |v(t+4)| - |v(t)| (first-order coordinate of Coulomb friction, not of drag)
        self.use_kr, self.use_mr, self.mr_nogamma, self.use_ksep, self.use_msep, self.use_vd = kr, mr, mr_nogamma, ksep, msep, vd
        if kr:
            self.head_kr = DeltaHead(1, 1, bn=False)
        if ksep:
            self.head_ksep = DeltaHead(1, 2, bn=False)
        if mr:
            self.head_mr = DeltaHead(1, 1, bn=False)
        if msep:
            self.head_msep = DeltaHead(1, 2, bn=False)
        if vd:
            self.head_vd = DeltaHead(4, 1, bn=False)
        for nm in ("head_vr", "head_vd", "head_vn"):
            if hasattr(self, nm):
                getattr(self, nm).noact = GHA0
        if selfcond:
            n_est = int(vr) + int(kr) + int(mr)
            assert n_est > 0, "selfcond needs at least one coordinate head"
            self.sc_inj = nn.Linear(n_est, DP)
            nn.init.zeros_(self.sc_inj.weight); nn.init.zeros_(self.sc_inj.bias)
        self.variant = variant
        (self.use_pt_in, self.use_touch_tgt,
         self.use_prop_tgt, self.use_motion_tgt) = VARIANTS[variant]
        self.use_recon = recon
        self.use_sid = sysid
        self.use_nll = nll
        self.use_oracle = oracle
        self.use_vr = vr
        self.vr_flow = vr_flow   # object speed from the frozen flow sensor instead of state
        self.touch_cw = 4.0      # touch-loss weight at contact frames (1+3 default)
        self.use_objin = objin
        self.register_buffer("obj_mu", torch.zeros(4))
        self.register_buffer("obj_sd", torch.ones(4))
        self.enc = FrameEncoder(self.use_pt_in, in_ch=in_ch, use_obj=objin)
        self.pred = Predictor(mul=mul)
        self.use_vv = vv                # control: plain future-speed target |v_{t+4}| (no ratio, no log)
        self.oinoise_px = oinoise_px    # precision axis: Gaussian noise on the objin state (pixels)
        self.vrnoise = vrnoise          # target-precision axis: Gaussian noise on the vr target (scaled units)
        if vv:
            self.head_vv = DeltaHead(4, 1, bn=False)
        self.use_vr16 = vr16
        self.use_phys = physhead
        self.phys_mode = phys_mode      # full | fixg (gamma fixed at prior median) | truev (v_hat := true v) | multi (shared gamma over Delta 4/8/16)
        self.use_disp = dispmlp
        if dispmlp:
            # control for the physics head: same supervision (true 16-step object
            # displacement on glide frames, same mask, same weight, same scaling),
            # plain MLP output instead of the analytic decay integral
            self.head_disp = nn.Sequential(nn.Linear(DP, 96), nn.GELU(), nn.Linear(96, 2))
        if vr16:
            # 16-step log-speed ratio on glide frames: the first-order coordinate
            # target at the horizon where gamma's share is largest
            self.head_vr16 = DeltaHead(16, 1, bn=False)
        if physhead:
            # structured dynamics head: trunk -> (log gamma_hat, v_hat) -> analytic
            # 16-step glide displacement v (1 - exp(-gamma T)) / gamma, supervised
            # by the true object displacement on glide frames. Tests whether the
            # functional failure is the head's functional form.
            n_out = {"full": 3, "fixg": 2, "truev": 1, "multi": 3}[phys_mode]
            self.head_phys = nn.Sequential(nn.Linear(DP, 96), nn.GELU(), nn.Linear(96, n_out))
        self.vrcond = vrcond
        if vrcond:
            # self-conditioning: the vr head's own estimate is injected back into
            # the trunk state that feeds every prediction head, so the estimated
            # rate can be USED by the dynamics heads, not only decoded
            self.vr_inj = nn.Linear(1, DP)
            nn.init.zeros_(self.vr_inj.weight); nn.init.zeros_(self.vr_inj.bias)
        if vr:
            # log-speed-ratio head: log|v(t+4)| - log|v(t)| of the object on
            # glide frames, scaled by VR_SCALE. In pure glide this equals
            # -gamma * 4 * DT exactly: a linear trace of gamma at Delta=4.
            self.head_vr = DeltaHead(4, 1, bn=False)
            self.head_vr.noact = GHA0
        self.dheads = nn.ModuleDict({str(d): DeltaHead(d) for d in DELTAS})
        for i in range(EXTRA1):
            self.dheads[f"1x{i}"] = DeltaHead(1)
        if recon:
            self.pixdec = PixelDecoder()
        if sysid:
            # explicit system-identification auxiliary head: supervised
            # (log m, gamma, log k) from the trunk. Oracle-labelled CONTROL that
            # separates "architecture/data cannot carry the parameter" from
            # "the predictive objective never asks for it".
            self.head_sid = nn.Sequential(nn.Linear(DP, 96), nn.GELU(),
                                          nn.Linear(96, 3))
        self.register_buffer("sid_mu", torch.zeros(3))
        self.register_buffer("sid_sd", torch.ones(3))
        if self.use_touch_tgt:
            self.head_touch = nn.Sequential(nn.Linear(DP, 96), nn.GELU(), nn.Linear(96, 7))
            self.dheads_touch = nn.ModuleDict({str(d): DeltaHead(d, 7, bn=False) for d in DELTAS_X})
        if self.use_prop_tgt:
            self.head_prop = nn.Sequential(nn.Linear(DP, 96), nn.GELU(), nn.Linear(96, 4))
            self.dheads_prop = nn.ModuleDict({str(d): DeltaHead(d, 4, bn=False) for d in DELTAS_X})
        if self.use_motion_tgt:
            # future motion-image targets at ALL horizons: the 8x8 |frame-diff|
            # at t+16 of a gliding object is v*exp(-gamma*0.8)*dt - the only
            # observation-space quantity that depends on gamma unambiguously
            self.dheads_motion = nn.ModuleDict(
                {str(d): DeltaHead(d, 64, bn=False) for d in (1,) + DELTAS})
        self.register_buffer("motion_mu", torch.zeros(64))
        self.register_buffer("motion_sd", torch.ones(64))
        self.mot_w = 0.0
        if nll:
            # minimal belief-maintaining variant (heteroscedastic Gaussian
            # NLL): each prediction term gets an input-dependent per-dim
            # log-variance head conditioned on the trunk state
            dims = {"z1": D, "z4": D, "z16": D}
            if self.use_touch_tgt:
                dims.update({"t1": 7, "t4": 7})
            if self.use_prop_tgt:
                dims.update({"p1": 4, "p4": 4})
            self.lv_heads = nn.ModuleDict(
                {k: nn.Linear(DP, d) for k, d in dims.items()})

    def _nll(self, err2, h_valid, key):
        lv = self.lv_heads[key](h_valid).clamp(-6.0, 4.0)
        return 0.5 * (err2 * torch.exp(-lv) + lv).mean()

    def encode(self, img, prop, touch, obj=None):
        if self.use_objin:
            if self.oinoise_px > 0:
                # measurement noise on position (sigma px) and on velocity as if differenced from
                # noisy positions over one step: sigma_v = sigma * sqrt(2) / DT
                s = self.oinoise_px / 64.0
                noise = torch.randn_like(obj) * torch.tensor([s, s, s * 1.4142 / 0.05, s * 1.4142 / 0.05], device=obj.device)
                obj = obj + noise
            obj = (obj - self.obj_mu) / self.obj_sd
        return self.enc(img, prop, touch, obj)

    def props_w(self, batch):
        return (batch["props"] - self.sid_mu) / self.sid_sd

    def phys_params(self, h, obj):
        """(gamma_hat [.., 1], v_hat [.., 2]) from the trunk under the chosen phys_mode."""
        out = self.head_phys(h)
        if self.phys_mode == "fixg":
            return torch.full_like(out[..., :1], 2.25), out[..., 0:2]
        if self.phys_mode == "truev":
            return torch.exp(out[..., 0:1]).clamp(0.05, 20.0), obj[..., 2:4]
        return torch.exp(out[..., 0:1]).clamp(0.05, 20.0), out[..., 1:3]

    def vr_estimate(self, h, act):
        """vr head prediction at every position that has 4 future actions."""
        tv = min(h.shape[1], act.shape[1] - 3 - int(ACORR))   # ACORR windows start one step later
        return self.head_vr(h[:, :tv], action_windows(act, 4, tv))[..., 0]   # [B,tv]

    def condition(self, h, act):
        if getattr(self, "selfcond", False):
            ests = []
            tv = min(h.shape[1], act.shape[1] - 3 - int(ACORR))   # ACORR windows start one step later
            if self.use_vr:
                ests.append(self.head_vr(h[:, :tv], action_windows(act, 4, tv))[..., 0])
            aw1 = action_windows(act, 1, tv)
            if self.use_kr:
                ests.append(self.head_kr(h[:, :tv], aw1)[..., 0])
            if self.use_mr:
                ests.append(self.head_mr(h[:, :tv], aw1)[..., 0])
            p = torch.stack(ests, -1)
            if not hasattr(self, "sc_inj"):
                raise RuntimeError("selfcond injection layer missing")
            return torch.cat([h[:, :tv] + self.sc_inj(p), h[:, tv:]], 1)
        if not getattr(self, "vrcond", False):
            return h
        p = self.vr_estimate(h, act)
        inj = self.vr_inj(p[..., None])
        return torch.cat([h[:, :p.shape[1]] + inj, h[:, p.shape[1]:]], 1)

    def loss(self, batch, t_ctx, horizon, global_step, lam=0.1):
        img, prop, touch, act = batch["img"], batch["prop"], batch["touch"], batch["act"]
        z = self.encode(img, prop, touch, batch.get("obj"))  # [B,T,D]
        pc = self.props_w(batch) if self.use_oracle else None
        h = self.pred.trunk(z[:, :-1], act[:, :-1], pc)      # positions 0..T-2
        h_raw = h
        h = self.condition(h, act)
        T = z.shape[1]
        losses = {}
        l_pred = 0.0
        if self.use_recon:
            # reconstruction objective family: same trunk, pixel targets.
            # Decode S random valid positions per (sample, horizon) to bound
            # deconv memory; uniform subsampling keeps the loss unbiased.
            S = 8
            frames_tgt = img[:, :, 0]                        # [B,T,H,W]
            B = z.shape[0]
            for d, wgt in zip((1,) + DELTAS, (1.0, 0.5, 0.5)):
                tv = T - d
                pos = torch.randint(0, tv, (B, S), device=z.device)
                hp = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
                if d == 1:
                    y = self.pixdec(hp.reshape(B * S, -1), 1)
                else:
                    aw = action_windows(act, d, tv)
                    awp = aw.gather(1, pos[..., None].expand(-1, -1, aw.shape[-1]))
                    y = self.pixdec(hp.reshape(B * S, -1), d, awp.reshape(B * S, -1))
                tgt = frames_tgt.gather(
                    1, (pos + d)[:, :, None, None].expand(
                        -1, -1, frames_tgt.shape[-2], frames_tgt.shape[-1]))
                l_rd = (y.reshape(B, S, *y.shape[-2:]) - tgt).pow(2).mean()
                losses[f"pred_pix{d}"] = l_rd
                l_pred = l_pred + wgt * l_rd
        else:
            if NOPROJ:
                tv1 = T - 1
                e_z = (self.dheads["1x0"](h[:, :tv1], action_windows(act, 1, tv1)) - z[:, 1:]).pow(2)
            else:
                z_hat = self.pred.project(h)                 # predict z_{1..T-1}
                e_z = (z_hat - z[:, 1:]).pow(2)
            l_z = self._nll(e_z, h, "z1") if self.use_nll else e_z.mean()
            losses["pred_z"] = l_z
            w1, wd = (1.0, 0.5) if not HNORM else (1.0 / (1.0 + 0.5 * (len(DELTAS) + EXTRA1)), 0.5 / (1.0 + 0.5 * (len(DELTAS) + EXTRA1)))
            l_pred = l_pred + w1 * l_z
            for d in DELTAS:
                tv = T - d
                y = self.dheads[str(d)](h[:, :tv], action_windows(act, d, tv))
                e_d = (y - z[:, d:]).pow(2)
                l_d = (self._nll(e_d, h[:, :tv], f"z{d}") if self.use_nll
                       else e_d.mean())
                losses[f"pred_z{d}"] = l_d
                l_pred = l_pred + wd * l_d
            for i in range(1 if NOPROJ else 0, EXTRA1):
                tv = T - 1
                y = self.dheads[f"1x{i}"](h[:, :tv], action_windows(act, 1, tv))
                l_d = (y - z[:, 1:]).pow(2).mean()
                losses[f"pred_z1x{i}"] = l_d
                l_pred = l_pred + wd * l_d
        if self.use_vr:
            speed = (batch["objspeed"] if self.vr_flow
                     else batch.get("obj_true", batch["obj"])[..., 2:].norm(dim=-1))     # [B,T]
            ls = torch.log(speed + 1e-3)
            tv = T - 4
            y_vr = (ls[:, 4:] - ls[:, :tv]) / VR_SCALE           # [B,tv]
            if self.vrnoise > 0:
                y_vr = y_vr + torch.randn_like(y_vr) * self.vrnoise
            free = (touch[..., 6] < 0.5).float()                 # contact flag is unwhitened
            glide = free.unfold(1, 5, 1).prod(-1)                # no contact over t..t+4
            # the object must still be moving at t+4: under Coulomb friction it stops exactly, and a window that
            # contains the stop encodes v(t) instead of the parameter (log(1e-3/v) or -v); drag never reaches zero
            mask = glide * (speed[:, :tv] > 0.05).float() * (speed[:, 4:] > 0.02).float()
            if self.offwall:
                pos = batch.get("obj_true", batch["obj"])[..., :2]
                ins = ((pos > 0.13) & (pos < 0.87)).all(-1).float()
                mask = mask * ins.unfold(1, 5, 1).prod(-1)
            p_vr = self.head_vr(h_raw[:, :tv], action_windows(act, 4, tv))[..., 0]
            l_vr = ((p_vr - y_vr).pow(2) * mask).sum() / mask.sum().clamp(min=1.0)
            losses["pred_vr"] = l_vr
            losses["vr_frac"] = mask.mean()
            l_pred = l_pred + 0.5 * l_vr
        if self.use_vv:
            sp = batch["objspeed"] if self.vr_flow else batch.get("obj_true", batch["obj"])[..., 2:].norm(dim=-1)
            tvv = T - 4
            y_vv = sp[:, 4:] / 0.3                                 # future speed, scaled by its typical std
            fr = (touch[..., 6] < 0.5).float()
            mk = fr.unfold(1, 5, 1).prod(-1) * (sp[:, :tvv] > 0.05).float()
            p_vv = self.head_vv(h[:, :tvv], action_windows(act, 4, tvv))[..., 0]
            l_vv = ((p_vv - y_vv).pow(2) * mk).sum() / mk.sum().clamp(min=1.0)
            losses["pred_vv"] = l_vv
            l_pred = l_pred + 0.5 * l_vv
        if self.use_vd:
            sp = batch.get("obj_true", batch["obj"])[..., 2:].norm(dim=-1)
            tvd = T - 4
            y_vd = (sp[:, 4:] - sp[:, :tvd]) / VD_SCALE
            fr = (touch[..., 6] < 0.5).float()
            mk = fr.unfold(1, 5, 1).prod(-1) * (sp[:, :tvd] > 0.05).float() * (sp[:, 4:] > 0.02).float()
            if self.offwall:
                pos = batch.get("obj_true", batch["obj"])[..., :2]
                mk = mk * ((pos > 0.13) & (pos < 0.87)).all(-1).float().unfold(1, 5, 1).prod(-1)
            p_vd = self.head_vd(h_raw[:, :tvd], action_windows(act, 4, tvd))[..., 0]
            l_vd = ((p_vd - y_vd).pow(2) * mk).sum() / mk.sum().clamp(min=1.0)
            losses["pred_vd"] = l_vd
            l_pred = l_pred + 0.5 * l_vd
        if self.use_vn:
            sp = batch.get("obj_true", batch["obj"])[..., 2:].norm(dim=-1)
            tvn = T - 4
            y_vn = ((sp[:, :tvn] - sp[:, 4:]) / sp[:, :tvn].clamp(min=0.05)) / 0.3
            fr = (touch[..., 6] < 0.5).float()
            mk = fr.unfold(1, 5, 1).prod(-1) * (sp[:, :tvn] > 0.05).float() * (sp[:, 4:] > 0.02).float()
            p_vn = self.head_vn(h_raw[:, :tvn], action_windows(act, 4, tvn))[..., 0]
            l_vn = ((p_vn - y_vn).pow(2) * mk).sum() / mk.sum().clamp(min=1.0)
            losses["pred_vn"] = l_vn
            l_pred = l_pred + 0.5 * l_vn
        if self.use_kr or self.use_ksep or self.use_mr or self.use_msep:
            objT = batch["obj"] if self.obscoord else batch.get("obj_true", batch["obj"])   # object state [B,T,4]
            traw = batch["touch_raw"]                            # unwhitened touch [B,T,7]
            fxy = batch["finger_xy"]                             # finger position [B,T,2]
            cflag = (traw[..., 6] > 0.5).float()
            tv1 = T - 1
            aw1 = action_windows(act, 1, tv1)
        if self.use_kr or self.use_ksep:
            ov = 0.15 - (objT[:, 1:, :2] - fxy[:, 1:]).norm(dim=-1)
            fl = traw[:, 1:, 3:5].norm(dim=-1)
            mk = cflag[:, 1:] * (ov > 0.002).float() * (fl > 0.05).float()
            lF = torch.log(fl.clamp(min=1e-3)); lo = torch.log(ov.clamp(min=1e-4))
            if self.use_kr:
                y = (lF - 1.5 * lo - KR_MU) / KR_SD
                pk = self.head_kr(h_raw[:, :tv1], aw1)[..., 0]
                l_kr = ((pk - y).pow(2) * mk).sum() / mk.sum().clamp(min=1.0)
                losses["pred_kr"] = l_kr
                l_pred = l_pred + 0.5 * l_kr
            if self.use_ksep:
                y = torch.stack([(lF - 1.284) / 1.069, (lo + 4.044) / 0.761], -1)
                pk = self.head_ksep(h_raw[:, :tv1], aw1)
                l_ks = ((pk - y).pow(2).mean(-1) * mk).sum() / mk.sum().clamp(min=1.0)
                losses["pred_ksep"] = l_ks
                l_pred = l_pred + 0.5 * l_ks
        if self.use_mr or self.use_msep:
            f_on = -traw[:, 1:, :2]
            fn = f_on.norm(dim=-1)
            dv = (objT[:, 1:, 2:] - objT[:, :-1, 2:]) / 0.05
            vbar = 0.5 * (objT[:, 1:, 2:] + objT[:, :-1, 2:])
            gam = batch["props"][:, 1][:, None, None]
            A = dv if self.mr_nogamma else dv + gam * vbar
            a_par = (A * f_on).sum(-1) / fn.clamp(min=1e-6)
            inside = lambda q: ((q > 0.13) & (q < 0.87)).all(-1)
            offwall = (inside(objT[:, 1:, :2]) & inside(objT[:, :-1, :2])).float()
            mk = cflag[:, 1:] * offwall * (fn > 0.1).float() * (a_par > 0.02).float()
            lf = torch.log(fn.clamp(min=1e-3)); la = torch.log(a_par.clamp(min=1e-3))
            if self.use_mr:
                y = (lf - la - MR_MU) / MR_SD
                pm = self.head_mr(h_raw[:, :tv1], aw1)[..., 0]
                l_mr = ((pm - y).pow(2) * mk).sum() / mk.sum().clamp(min=1.0)
                losses["pred_mr"] = l_mr
                losses["mr_frac"] = mk.mean()
                l_pred = l_pred + 0.5 * l_mr
            if self.use_msep:
                y = torch.stack([(lf - 1.025) / 1.156, (la - 0.742) / 1.209], -1)
                pm = self.head_msep(h_raw[:, :tv1], aw1)
                l_ms = ((pm - y).pow(2).mean(-1) * mk).sum() / mk.sum().clamp(min=1.0)
                losses["pred_msep"] = l_ms
                l_pred = l_pred + 0.5 * l_ms
        if self.use_vr16 or self.use_phys or self.use_disp:
            obj = batch["obj"]
            speed = obj[..., 2:].norm(dim=-1)
            free = (touch[..., 6] < 0.5).float()
            glide16 = free.unfold(1, 17, 1).prod(-1)                # no contact over t..t+16
            tv16 = T - 16
            mask16 = glide16[:, :tv16] * (speed[:, :tv16] > 0.05).float()
        if self.use_vr16:
            ls = torch.log(speed + 1e-3)
            y16 = (ls[:, 16:] - ls[:, :tv16]) / (VR_SCALE * 4)
            p16 = self.head_vr16(h[:, :tv16], action_windows(act, 16, tv16))[..., 0]
            l16 = ((p16 - y16).pow(2) * mask16).sum() / mask16.sum().clamp(min=1.0)
            losses["pred_vr16"] = l16
            l_pred = l_pred + 0.5 * l16
        if self.use_disp:
            disp_hat = self.head_disp(h[:, :tv16])
            disp = obj[:, 16:, :2] - obj[:, :tv16, :2]
            l_d = ((disp_hat - disp).pow(2).sum(-1) * mask16).sum() / mask16.sum().clamp(min=1.0) * 64.0
            losses["pred_disp"] = l_d
            l_pred = l_pred + 0.5 * l_d
        if self.use_phys:
            g_hat, v_hat = self.phys_params(h[:, :tv16], obj[:, :tv16])
            if self.phys_mode == "multi":
                l_ph = 0.0
                for dd in (4, 8, 16):
                    tvd = T - dd
                    gd = free.unfold(1, dd + 1, 1).prod(-1)[:, :tvd] * (speed[:, :tvd] > 0.05).float()
                    gh, vh = self.phys_params(h[:, :tvd], obj[:, :tvd])
                    dh = vh * (1 - torch.exp(-gh * dd * 0.05)) / gh
                    dsp = obj[:, dd:, :2] - obj[:, :tvd, :2]
                    l_ph = l_ph + ((dh - dsp).pow(2).sum(-1) * gd).sum() / gd.sum().clamp(min=1.0) * 64.0 / 3
            else:
                disp_hat = v_hat * (1 - torch.exp(-g_hat * 0.8)) / g_hat
                disp = obj[:, 16:, :2] - obj[:, :tv16, :2]
                l_ph = ((disp_hat - disp).pow(2).sum(-1) * mask16).sum() / mask16.sum().clamp(min=1.0) * 64.0
            losses["pred_phys"] = l_ph
            l_pred = l_pred + 0.5 * l_ph
        if self.use_sid:
            y_sid = (batch["props"] - self.sid_mu) / self.sid_sd     # [B,3]
            l_sid = (self.head_sid(h) - y_sid[:, None]).pow(2).mean()
            losses["sysid"] = l_sid
            l_pred = l_pred + 0.5 * l_sid
        if self.use_touch_tgt:
            tt = touch[:, 1:]
            w = 1.0 + (self.touch_cw - 1.0) * tt[..., 6:7]
            e_t = (self.head_touch(h) - tt).pow(2)
            l_t = self._nll(e_t, h, "t1") if self.use_nll else (e_t * w).mean()
            losses["pred_touch"] = l_t
            l_pred = l_pred + 0.5 * l_t
            for d in DELTAS_X:
                tv = T - d
                y = self.dheads_touch[str(d)](h[:, :tv], action_windows(act, d, tv))
                td = touch[:, d:]
                wd = 1.0 + (self.touch_cw - 1.0) * td[..., 6:7]
                e_td = (y - td).pow(2)
                l_td = (self._nll(e_td, h[:, :tv], "t4") if self.use_nll
                        else (e_td * wd).mean())
                losses[f"pred_touch{d}"] = l_td
                l_pred = l_pred + 0.25 * l_td
        if self.use_prop_tgt:
            tp = prop[:, 1:]
            e_p = (self.head_prop(h) - tp).pow(2)
            l_p = self._nll(e_p, h, "p1") if self.use_nll else e_p.mean()
            losses["pred_prop"] = l_p
            l_pred = l_pred + 0.5 * l_p
            for d in DELTAS_X:
                tv = T - d
                y = self.dheads_prop[str(d)](h[:, :tv], action_windows(act, d, tv))
                e_pd = (y - prop[:, d:]).pow(2)
                l_pd = (self._nll(e_pd, h[:, :tv], "p4") if self.use_nll
                        else e_pd.mean())
                losses[f"pred_prop{d}"] = l_pd
                l_pred = l_pred + 0.25 * l_pd
        if self.use_motion_tgt:
            mot = torch.nn.functional.avg_pool2d(
                img[:, :, 1].abs().reshape(-1, 1, *img.shape[3:]), 8)
            mot = mot.reshape(img.shape[0], T, 64)
            mot = (mot - self.motion_mu) / self.motion_sd
            for d, wgt in zip((1,) + DELTAS, (0.5, 0.25, 0.25)):
                tv = T - d
                y = self.dheads_motion[str(d)](h[:, :tv], action_windows(act, d, tv))
                tgt = mot[:, d:]
                # magnitude-weighted cells: naive per-dim MSE lets the (easy,
                # action-predictable) finger motion swamp the gamma-carrying
                # object cells - same wash-out pattern as Finding #1
                wd = 1.0 + self.mot_w * tgt.clamp(min=0)
                l_md = ((y - tgt).pow(2) * wd).mean()
                losses[f"pred_mot{d}"] = l_md
                l_pred = l_pred + wgt * l_md
        if self.use_recon:
            # pure reconstruction family: pixel targets anchor the latent, no
            # anti-collapse regularizer needed (and none used by that family)
            losses["total"] = l_pred
        else:
            l_sig = sigreg(z.reshape(-1, D), global_step)
            losses["sigreg"] = l_sig
            losses["total"] = l_pred + lam * l_sig
        return losses
