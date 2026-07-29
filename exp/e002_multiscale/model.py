"""X-JEPA v5 (e002): multi-timescale prediction heads.

Mechanistic motivation from e001b: slow dynamics parameters (drag gamma) have
sub-pixel single-step effects (gamma*dt^2 ~ 0.009 < 1 px), so teacher-forced
1-step prediction gives them ~zero gradient pressure; they only manifest over
multi-step composition. Fix: Delta-strided prediction heads (Delta = 1, 4, 16)
conditioned on the intervening action window - multi-horizon pressure at
teacher-forcing cost (no autoregressive backprop).
"""
import torch
import torch.nn as nn

D = 128          # latent embedding dim
DP = 192         # predictor width


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

    def __init__(self, use_pt):
        super().__init__()
        self.use_pt = use_pt
        ch = (32, 64, 128, 256)
        layers, c_in = [], 1
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
        self.fuse = nn.Sequential(nn.Linear(fuse_in, 192), nn.GELU(), nn.Linear(192, D))
        self.bn = nn.BatchNorm1d(D)

    def forward(self, img, prop, touch):        # img [B,T,H,W]
        B, T = img.shape[:2]
        f = self.vis_fc(self.conv(img.reshape(B * T, 1, *img.shape[2:])))
        if self.use_pt:
            f = torch.cat([f, self.enc_prop(prop.reshape(B * T, -1)),
                           self.enc_touch(touch.reshape(B * T, -1))], -1)
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


class Predictor(nn.Module):
    """Causal transformer over embedding history; AdaLN action conditioning.
    Input positions t=0..T-1 (z_t, a_t) -> trunk h_t -> projector -> z_{t+1}."""

    def __init__(self, n_layers=4, heads=8, max_t=64):
        super().__init__()
        self.inp = nn.Linear(D, DP)
        self.pos = nn.Parameter(torch.zeros(max_t, DP))
        nn.init.normal_(self.pos, std=0.02)
        self.a_emb = nn.Sequential(nn.Linear(2, DP), nn.GELU(), nn.Linear(DP, DP))
        self.blocks = nn.ModuleList([AdaLNBlock(DP, heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(DP, elementwise_affine=False)
        self.proj = nn.Linear(DP, D)
        self.bn = nn.BatchNorm1d(D)

    def trunk(self, z_seq, a_seq):              # [B,T,D], [B,T,2]
        B, T = z_seq.shape[:2]
        x = self.inp(z_seq) + self.pos[None, :T]
        c = self.a_emb(a_seq)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=z_seq.device), 1)
        for blk in self.blocks:
            x = blk(x, c, mask)
        return self.ln_f(x)                     # [B,T,DP]

    def project(self, h):                       # trunk -> next-frame embedding
        B, T = h.shape[:2]
        z = self.bn(self.proj(h).reshape(B * T, D))
        return z.reshape(B, T, D)

    @torch.no_grad()
    def rollout(self, z_ctx, a_ctx, a_future):
        """Autoregressive generation. z_ctx [B,Tc,D] embeddings of observed
        frames, a_ctx [B,Tc,2] their aligned actions (position t consumes
        (z_t, a_t) and emits z_{t+1}), a_future [B,H,2] actions for the
        generated steps. Returns predicted embeddings [B,H,D]."""
        z_seq = z_ctx
        a_all = torch.cat([a_ctx, a_future], 1)
        outs = []
        for _ in range(a_future.shape[1]):
            h = self.trunk(z_seq, a_all[:, :z_seq.shape[1]])
            z_next = self.project(h[:, -1:])
            outs.append(z_next)
            z_seq = torch.cat([z_seq, z_next], 1)
        return torch.cat(outs, 1)


class DeltaHead(nn.Module):
    """Predict z_{t+delta} from trunk h_t conditioned on actions a_{t..t+delta-1}."""

    def __init__(self, delta, d_out=D, bn=True):
        super().__init__()
        self.delta = delta
        self.a_mlp = nn.Sequential(nn.Linear(2 * delta, 128), nn.GELU())
        self.out = nn.Linear(DP + 128, d_out)
        self.bn = nn.BatchNorm1d(d_out) if bn else nn.Identity()

    def forward(self, h_valid, a_win):          # [B,Tv,DP], [B,Tv,2*delta]
        B, Tv = h_valid.shape[:2]
        y = self.out(torch.cat([h_valid, self.a_mlp(a_win)], -1))
        return self.bn(y.reshape(B * Tv, -1)).reshape(B, Tv, -1)


def action_windows(act, delta, tv):
    """[B,T,2] -> [B,tv,2*delta]: flattened a_{t..t+delta-1} for t=0..tv-1."""
    w = act.unfold(1, delta, 1)                 # [B, T-delta+1, 2, delta]
    return w[:, :tv].permute(0, 1, 3, 2).reshape(act.shape[0], tv, 2 * delta)


VARIANTS = {
    #        inputs V+P+T, touch target, proprio target
    "V":   (False, False, False),
    "VF":  (True,  False, False),
    "VX":  (False, True,  True),
    "VFX": (True,  True,  True),
    "VXt": (False, True,  False),   # touch-only targets (control)
    "VXp": (False, False, True),    # proprio-only targets (control)
}


DELTAS = (4, 16)          # extra z-horizons on top of the default Delta=1
DELTAS_X = (4,)           # extra horizons for cross-modal heads


class XJEPA(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.use_pt_in, self.use_touch_tgt, self.use_prop_tgt = VARIANTS[variant]
        self.enc = FrameEncoder(self.use_pt_in)
        self.pred = Predictor()
        self.dheads = nn.ModuleDict({str(d): DeltaHead(d) for d in DELTAS})
        if self.use_touch_tgt:
            self.head_touch = nn.Sequential(nn.Linear(DP, 96), nn.GELU(), nn.Linear(96, 7))
            self.dheads_touch = nn.ModuleDict({str(d): DeltaHead(d, 7, bn=False) for d in DELTAS_X})
        if self.use_prop_tgt:
            self.head_prop = nn.Sequential(nn.Linear(DP, 96), nn.GELU(), nn.Linear(96, 4))
            self.dheads_prop = nn.ModuleDict({str(d): DeltaHead(d, 4, bn=False) for d in DELTAS_X})

    def encode(self, img, prop, touch):
        return self.enc(img, prop, touch)

    def loss(self, batch, t_ctx, horizon, global_step, lam=0.1):
        img, prop, touch, act = batch["img"], batch["prop"], batch["touch"], batch["act"]
        z = self.encode(img, prop, touch)                    # [B,T,D]
        h = self.pred.trunk(z[:, :-1], act[:, :-1])          # positions 0..T-2
        z_hat = self.pred.project(h)                         # predict z_{1..T-1}
        l_z = (z_hat - z[:, 1:]).pow(2).mean()
        losses = {"pred_z": l_z}
        l_pred = l_z
        T = z.shape[1]
        for d in DELTAS:
            tv = T - d
            y = self.dheads[str(d)](h[:, :tv], action_windows(act, d, tv))
            l_d = (y - z[:, d:]).pow(2).mean()
            losses[f"pred_z{d}"] = l_d
            l_pred = l_pred + 0.5 * l_d
        if self.use_touch_tgt:
            tt = touch[:, 1:]
            w = 1.0 + 3.0 * tt[..., 6:7]
            l_t = ((self.head_touch(h) - tt).pow(2) * w).mean()
            losses["pred_touch"] = l_t
            l_pred = l_pred + 0.5 * l_t
            for d in DELTAS_X:
                tv = T - d
                y = self.dheads_touch[str(d)](h[:, :tv], action_windows(act, d, tv))
                td = touch[:, d:]
                wd = 1.0 + 3.0 * td[..., 6:7]
                l_td = ((y - td).pow(2) * wd).mean()
                losses[f"pred_touch{d}"] = l_td
                l_pred = l_pred + 0.25 * l_td
        if self.use_prop_tgt:
            tp = prop[:, 1:]
            l_p = (self.head_prop(h) - tp).pow(2).mean()
            losses["pred_prop"] = l_p
            l_pred = l_pred + 0.5 * l_p
            for d in DELTAS_X:
                tv = T - d
                y = self.dheads_prop[str(d)](h[:, :tv], action_windows(act, d, tv))
                l_pd = (y - prop[:, d:]).pow(2).mean()
                losses[f"pred_prop{d}"] = l_pd
                l_pred = l_pred + 0.25 * l_pd
        l_sig = sigreg(z.reshape(-1, D), global_step)
        losses["sigreg"] = l_sig
        losses["total"] = l_pred + lam * l_sig
        return losses
