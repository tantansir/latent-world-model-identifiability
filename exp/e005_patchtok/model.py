"""X-JEPA v7 (e005): patch-token architecture.

Phase-1 found gamma capped at ~0.12 by the metric precision of a single global
frame embedding (velocity R^2 ceiling ~0.79). e005 keeps every validated design
rule (motion channel, cross-modal targets, multi-horizon heads, SIGReg at low
lambda) and replaces the global embedding with K=16 spatial tokens + a
factorized spatial/temporal predictor - the scalable (ViT-style) interface.

Delta=1 prediction is dense (per-token, spatially aligned); Delta=4/16 heads
and cross-modal heads act on the pooled trunk. VFX uses learned mask tokens +
modality dropout so autoregressive rollout (where future touch is unknown)
stays in-distribution.
"""
import torch
import torch.nn as nn

D = 128          # token embedding dim
DP = 192         # predictor width
K = 16           # spatial tokens per frame (4x4 CNN map)
DELTAS = (4, 16)
DELTAS_X = (4,)


def set_dim(d):
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


class TokenEncoder(nn.Module):
    """CNN -> 4x4 feature map -> K spatial tokens (BN'd); optional touch/prop
    tokens with learned mask embeddings for modality dropout / rollout."""

    def __init__(self, use_pt):
        super().__init__()
        self.use_pt = use_pt
        ch = (32, 64, 128, 256)
        layers, c_in = [], 2
        for c in ch:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.GELU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.tok = nn.Linear(256, D)
        self.bn = nn.BatchNorm1d(D)
        if use_pt:
            self.enc_prop = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, D))
            self.enc_touch = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, D))
            self.bn_p = nn.BatchNorm1d(D)
            self.bn_t = nn.BatchNorm1d(D)
            self.mask_p = nn.Parameter(torch.zeros(D))
            self.mask_t = nn.Parameter(torch.zeros(D))

    def forward(self, img, prop, touch, mod_drop=0.0):
        B, T = img.shape[:2]
        f = self.conv(img.reshape(B * T, 2, *img.shape[3:]))       # [BT,256,4,4]
        f = f.flatten(2).transpose(1, 2)                           # [BT,K,256]
        z = self.bn(self.tok(f).reshape(-1, D)).reshape(B, T, K, D)
        if not self.use_pt:
            return z, z
        zp = self.bn_p(self.enc_prop(prop.reshape(B * T, -1))).reshape(B, T, 1, D)
        zt = self.bn_t(self.enc_touch(touch.reshape(B * T, -1))).reshape(B, T, 1, D)
        if mod_drop > 0:
            keep = (torch.rand(B, T, 1, 1, device=img.device) > mod_drop).float()
            zp = keep * zp + (1 - keep) * self.mask_p.view(1, 1, 1, D)
            keep = (torch.rand(B, T, 1, 1, device=img.device) > mod_drop).float()
            zt = keep * zt + (1 - keep) * self.mask_t.view(1, 1, 1, D)
        return z, torch.cat([z, zp, zt], 2)                        # [B,T,Kf,D]

    def mask_frame_tokens(self, z_spatial):
        """Assemble input tokens for generated frames: predicted spatial tokens
        + mask embeddings for the unavailable touch/prop modalities."""
        if not self.use_pt:
            return z_spatial
        B = z_spatial.shape[0]
        mp = self.mask_p.view(1, 1, 1, D).expand(B, z_spatial.shape[1], 1, D)
        mt = self.mask_t.view(1, 1, 1, D).expand(B, z_spatial.shape[1], 1, D)
        return torch.cat([z_spatial, mp, mt], 2)


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


class DeltaHead(nn.Module):
    def __init__(self, delta, d_out=None, bn=True):
        super().__init__()
        d_out = D if d_out is None else d_out
        self.delta = delta
        self.a_mlp = nn.Sequential(nn.Linear(2 * delta, 128), nn.GELU())
        self.out = nn.Linear(DP + 128, d_out)
        self.bn = nn.BatchNorm1d(d_out) if bn else nn.Identity()

    def forward(self, h_valid, a_win):
        B, Tv = h_valid.shape[:2]
        y = self.out(torch.cat([h_valid, self.a_mlp(a_win)], -1))
        return self.bn(y.reshape(B * Tv, -1)).reshape(B, Tv, -1)


def action_windows(act, delta, tv):
    w = act.unfold(1, delta, 1)
    return w[:, :tv].permute(0, 1, 3, 2).reshape(act.shape[0], tv, 2 * delta)


class TokenPredictor(nn.Module):
    """Factorized spatial/temporal transformer over frame token sets."""

    def __init__(self, kf, max_t=64):
        super().__init__()
        self.kf = kf
        self.inp = nn.Linear(D, DP)
        self.tok_emb = nn.Parameter(torch.zeros(kf, DP))
        self.pos = nn.Parameter(torch.zeros(max_t, DP))
        nn.init.normal_(self.tok_emb, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        self.a_emb = nn.Sequential(nn.Linear(2, DP), nn.GELU(), nn.Linear(DP, DP))
        sp = lambda: nn.TransformerEncoderLayer(DP, 8, 384, dropout=0.1,
                                                activation="gelu", batch_first=True,
                                                norm_first=True)
        self.sp1, self.sp2 = sp(), sp()
        self.tp1, self.tp2 = AdaLNBlock(DP, 8), AdaLNBlock(DP, 8)
        self.ln_f = nn.LayerNorm(DP, elementwise_affine=False)
        self.proj1 = nn.Linear(DP, D)
        self.bn1 = nn.BatchNorm1d(D)

    def trunk(self, tokens, act):               # [B,T,Kf,D], [B,T,2]
        B, T, Kf = tokens.shape[:3]
        x = self.inp(tokens) + self.tok_emb[None, None] + self.pos[None, :T, None]
        c = self.a_emb(act)                     # [B,T,DP]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
        for sp, tp in ((self.sp1, self.tp1), (self.sp2, self.tp2)):
            x = sp(x.reshape(B * T, Kf, DP)).reshape(B, T, Kf, DP)
            xt = x.permute(0, 2, 1, 3).reshape(B * Kf, T, DP)
            ct = c[:, None].expand(B, Kf, T, DP).reshape(B * Kf, T, DP)
            xt = tp(xt, ct, causal)
            x = xt.reshape(B, Kf, T, DP).permute(0, 2, 1, 3)
        return self.ln_f(x)                     # [B,T,Kf,DP]

    def project1(self, h_spatial):              # [B,T,K,DP] -> next-frame tokens
        B, T = h_spatial.shape[:2]
        z = self.bn1(self.proj1(h_spatial).reshape(-1, D))
        return z.reshape(B, T, K, D)


VARIANTS = {
    "V":   (False, False, False),
    "VF":  (True,  False, False),
    "VX":  (False, True,  True),
    "VFX": (True,  True,  True),
    "VXt": (False, True,  False),
    "VXp": (False, False, True),
}


class XJEPA(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.use_pt_in, self.use_touch_tgt, self.use_prop_tgt = VARIANTS[variant]
        self.enc = TokenEncoder(self.use_pt_in)
        self.pred = TokenPredictor(kf=K + 2 if self.use_pt_in else K)
        self.dheads = nn.ModuleDict({str(d): DeltaHead(d) for d in DELTAS})
        if self.use_touch_tgt:
            self.head_touch = nn.Sequential(nn.Linear(DP, 96), nn.GELU(), nn.Linear(96, 7))
            self.dheads_touch = nn.ModuleDict({str(d): DeltaHead(d, 7, bn=False) for d in DELTAS_X})
        if self.use_prop_tgt:
            self.head_prop = nn.Sequential(nn.Linear(DP, 96), nn.GELU(), nn.Linear(96, 4))
            self.dheads_prop = nn.ModuleDict({str(d): DeltaHead(d, 4, bn=False) for d in DELTAS_X})

    def encode(self, img, prop, touch):
        """Pooled per-frame embedding (probe / decode space)."""
        z, _ = self.enc(img, prop, touch)
        return z.mean(2)                        # [B,T,D]

    def trunk_feature(self, img, prop, touch, act):
        """Pooled trunk hidden per frame (system-ID probe space)."""
        _, tokens = self.enc(img, prop, touch)
        h = self.pred.trunk(tokens, act)
        return h[:, :, :K].mean(2)              # [B,T,DP]

    def loss(self, batch, t_ctx, horizon, global_step, lam=0.02):
        img, prop, touch, act = batch["img"], batch["prop"], batch["touch"], batch["act"]
        z_sp, tokens = self.enc(img, prop, touch,
                                mod_drop=0.2 if self.use_pt_in else 0.0)
        h = self.pred.trunk(tokens, act)        # [B,T,Kf,DP]
        h_sp = h[:, :, :K]
        z1 = self.pred.project1(h_sp[:, :-1])
        l_z = (z1 - z_sp[:, 1:]).pow(2).mean()
        losses = {"pred_z": l_z}
        l_pred = l_z
        T = z_sp.shape[1]
        h_pool = h_sp.mean(2)                   # [B,T,DP]
        z_pool = z_sp.mean(2)
        for d in DELTAS:
            tv = T - d
            y = self.dheads[str(d)](h_pool[:, :tv], action_windows(act, d, tv))
            l_d = (y - z_pool[:, d:]).pow(2).mean()
            losses[f"pred_z{d}"] = l_d
            l_pred = l_pred + 0.5 * l_d
        if self.use_touch_tgt:
            tt = touch[:, 1:]
            w = 1.0 + 3.0 * tt[..., 6:7]
            l_t = ((self.head_touch(h_pool[:, :-1]) - tt).pow(2) * w).mean()
            losses["pred_touch"] = l_t
            l_pred = l_pred + 0.5 * l_t
            for d in DELTAS_X:
                tv = T - d
                y = self.dheads_touch[str(d)](h_pool[:, :tv], action_windows(act, d, tv))
                td = touch[:, d:]
                wd = 1.0 + 3.0 * td[..., 6:7]
                l_td = ((y - td).pow(2) * wd).mean()
                losses[f"pred_touch{d}"] = l_td
                l_pred = l_pred + 0.25 * l_td
        if self.use_prop_tgt:
            l_p = (self.head_prop(h_pool[:, :-1]) - prop[:, 1:]).pow(2).mean()
            losses["pred_prop"] = l_p
            l_pred = l_pred + 0.5 * l_p
            for d in DELTAS_X:
                tv = T - d
                y = self.dheads_prop[str(d)](h_pool[:, :tv], action_windows(act, d, tv))
                l_pd = (y - prop[:, d:]).pow(2).mean()
                losses[f"pred_prop{d}"] = l_pd
                l_pred = l_pred + 0.25 * l_pd
        # token-level rows explode SIGReg memory (B*T*K x slices x knots complex
        # tensor); subsample rows + fewer slices - per-step resampling of both
        # rows and directions accumulates full coverage across training.
        zf = z_sp.reshape(-1, D)
        idx = torch.randint(0, zf.shape[0], (4096,), device=zf.device)
        l_sig = sigreg(zf[idx], global_step, num_slices=256)
        losses["sigreg"] = l_sig
        losses["total"] = l_pred + lam * l_sig
        return losses

    @torch.no_grad()
    def rollout_pooled(self, img, prop, touch, a_ctx, a_future):
        """AR generation over token frames; returns pooled embeddings [B,H,D]."""
        _, tokens = self.enc(img, prop, touch)
        a_all = torch.cat([a_ctx, a_future], 1)
        outs = []
        for _ in range(a_future.shape[1]):
            h = self.pred.trunk(tokens, a_all[:, :tokens.shape[1]])
            z_next = self.pred.project1(h[:, -1:, :K])          # [B,1,K,D]
            outs.append(z_next.mean(2))
            tokens = torch.cat([tokens, self.enc.mask_frame_tokens(z_next)], 1)
        return torch.cat(outs, 1)

    @torch.no_grad()
    def direct16_pooled(self, img, prop, touch, a_ctx, a_win16):
        _, tokens = self.enc(img, prop, touch)
        h = self.pred.trunk(tokens, a_ctx)
        h_pool = h[:, -1:, :K].mean(2)
        return self.dheads["16"](h_pool, a_win16)[:, 0]         # [B,D]
