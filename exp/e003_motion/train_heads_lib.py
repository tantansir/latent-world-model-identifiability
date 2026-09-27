import torch, torch.nn as nn
import model as M


class ThetaHead(nn.Module):
    """interaction head over [h, phi(A), theta]; theta_ctx [B,3] (whitened params) is set by the caller before forward"""
    def __init__(self, delta, d_out):
        super().__init__()
        self.delta = delta
        self.a_mlp = nn.Sequential(nn.Linear(2 * delta, 128), nn.GELU())
        self.out = nn.Sequential(nn.Linear(M.DP + 128 + 3, 256), nn.GELU(), nn.Linear(256, d_out))
        self.bn = nn.BatchNorm1d(d_out)
        self.theta_ctx = None

    def forward(self, h, a_win):
        B, Tv = h.shape[:2]
        th = self.theta_ctx if self.theta_ctx.dim() == 3 else self.theta_ctx[:, None, :].expand(-1, Tv, -1)
        th = th[:, :Tv]
        y = self.out(torch.cat([h, self.a_mlp(a_win), th], -1))
        return self.bn(y.reshape(B * Tv, -1)).reshape(B, Tv, -1)
