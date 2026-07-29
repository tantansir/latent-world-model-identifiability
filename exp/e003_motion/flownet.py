"""Tiny self-supervised flow net for PokeWorld: warp-consistency training on
consecutive frame pairs (MC-JEPA's mechanism, minimal form). The frozen net
then serves as a visual VELOCITY SENSOR feeding flow input channels - velocity
becomes computable instead of hoped-for.

Convention: f1(p) ~= f0(p + flow(p)), so flow(p) = -v_px on moving content.
Sanity metric (eval only, privileged): correlation of object-region mean flow
against true object velocity.
"""
import pathlib, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "e001_xmodal_jepa" / "data"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 2, 3, padding=1))

    def forward(self, f0, f1):                   # [B,1,64,64] x2 -> [B,2,16,16]
        return self.net(torch.cat([f0, f1], 1))


def warp(img, flow16):
    """Backward-warp img by flow (pixel units at 64 scale)."""
    B = img.shape[0]
    flow = F.interpolate(flow16, size=64, mode="bilinear", align_corners=False)
    ar = torch.arange(64, device=img.device).float()
    yy, xx = torch.meshgrid(ar, ar, indexing="ij")
    gx = (xx[None] + flow[:, 0]) * (2 / 63) - 1
    gy = (yy[None] + flow[:, 1]) * (2 / 63) - 1
    grid = torch.stack([gx, gy], -1)
    return F.grid_sample(img, grid, align_corners=True, padding_mode="border")


def train_flownet(steps=4000, bs=256, seed=0):
    device = "cuda"
    img = np.load(DATA / "train_img.npy", mmap_mode="r")
    n_ep, T = img.shape[:2]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = FlowNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        ep = rng.integers(0, n_ep, bs)
        t = rng.integers(0, T - 1, bs)
        f0 = torch.from_numpy(np.stack([img[e, u] for e, u in zip(ep, t)])
                              ).to(device).float().div_(255).unsqueeze(1)
        f1 = torch.from_numpy(np.stack([img[e, u + 1] for e, u in zip(ep, t)])
                              ).to(device).float().div_(255).unsqueeze(1)
        flow = net(f0, f1)
        l_photo = (warp(f0, flow) - f1).abs().mean()
        l_smooth = (flow[:, :, 1:] - flow[:, :, :-1]).abs().mean() + \
                   (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
        loss = l_photo + 0.02 * l_smooth
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 500 == 0 or s == steps - 1:
            print(f"flownet step {s}: photo {l_photo.item():.5f} "
                  f"smooth {l_smooth.item():.4f}", flush=True)
    torch.save(net.state_dict(), HERE / "results" / "flownet.pt")

    # privileged sanity: object-region mean flow vs true velocity
    st = np.load(DATA / "val_state.npz")
    obj = st["obj"]
    vimg = np.load(DATA / "val_img.npy", mmap_mode="r")
    rng = np.random.default_rng(1)
    ep = rng.integers(0, vimg.shape[0], 512)
    t = rng.integers(1, T - 1, 512)
    f0 = torch.from_numpy(np.stack([vimg[e, u - 1] for e, u in zip(ep, t)])
                          ).to(device).float().div_(255).unsqueeze(1)
    f1 = torch.from_numpy(np.stack([vimg[e, u] for e, u in zip(ep, t)])
                          ).to(device).float().div_(255).unsqueeze(1)
    with torch.no_grad():
        flow = F.interpolate(net(f0, f1), size=64, mode="bilinear",
                             align_corners=False).cpu().numpy()
    est, true = [], []
    for i, (e, u) in enumerate(zip(ep, t)):
        c = obj[e, u, :2] * 64
        yy, xx = np.mgrid[0:64, 0:64]
        mask = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 < (0.09 * 64 * 0.8) ** 2
        if mask.sum() < 10 or np.linalg.norm(obj[e, u, 2:]) < 0.05:
            continue
        est.append([-flow[i, 0][mask].mean(), -flow[i, 1][mask].mean()])
        true.append(obj[e, u, 2:] * 0.05 * 64)
    est, true = np.array(est), np.array(true)
    corr = [np.corrcoef(est[:, d], true[:, d])[0, 1] for d in (0, 1)]
    epe = np.linalg.norm(est - true, axis=1).mean()
    print(f"sanity (n={len(est)}): flow-vs-true-velocity corr x/y = "
          f"{corr[0]:.3f}/{corr[1]:.3f}, EPE {epe:.2f}px "
          f"(median true speed {np.linalg.norm(true, axis=1).mean():.2f}px)", flush=True)


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    train_flownet()
