"""X-JEPA v4: LeWM-faithful recipe extended with cross-modal targets.

Per-frame encoder (CNN -> BatchNorm projector, NO trailing LayerNorm: LeWM
shows a final LayerNorm blocks the SIGReg anti-collapse objective) + causal
transformer predictor over the embedding history with zero-init AdaLN action
conditioning, trained teacher-forced at every position. Cross-modal variants
add heads on the predictor trunk that forecast next-frame touch/proprio.

Variants (2x2): inputs {V, V+P+T} x targets {V, V+P+T}
  V, VF, VX, VFX as before.
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
        for j in range(a_future.shape[1]):
            h = self.trunk(z_seq, a_all[:, :z_seq.shape[1]])
            z_next = self.project(h[:, -1:])
            outs.append(z_next)
            z_seq = torch.cat([z_seq, z_next], 1)
        return torch.cat(outs, 1)


VARIANTS = {
    #        inputs V+P+T, touch target, proprio target
    "V":   (False, False, False),
    "VF":  (True,  False, False),
    "VX":  (False, True,  True),
    "VFX": (True,  True,  True),
    "VXt": (False, True,  False),   # touch-only targets (control)
    "VXp": (False, False, True),    # proprio-only targets (control)
}


class XJEPA(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.use_pt_in, self.use_touch_tgt, self.use_prop_tgt = VARIANTS[variant]
        self.enc = FrameEncoder(self.use_pt_in)
        self.pred = Predictor()
        if self.use_touch_tgt:
            self.head_touch = nn.Sequential(nn.Linear(DP, 96), nn.GELU(), nn.Linear(96, 7))
        if self.use_prop_tgt:
            self.head_prop = nn.Sequential(nn.Linear(DP, 96), nn.GELU(), nn.Linear(96, 4))

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
        if self.use_touch_tgt:
            tt = touch[:, 1:]
            w = 1.0 + 3.0 * tt[..., 6:7]
            l_t = ((self.head_touch(h) - tt).pow(2) * w).mean()
            losses["pred_touch"] = l_t
            l_pred = l_pred + 0.5 * l_t
        if self.use_prop_tgt:
            tp = prop[:, 1:]
            l_p = (self.head_prop(h) - tp).pow(2).mean()
            losses["pred_prop"] = l_p
            l_pred = l_pred + 0.5 * l_p
        l_sig = sigreg(z.reshape(-1, D), global_step)
        losses["sigreg"] = l_sig
        losses["total"] = l_pred + lam * l_sig
        return losses
