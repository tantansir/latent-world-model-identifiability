"""Project-page videos: the same push-to-goal task planned on two frozen
latents. Trajectories come from dump_plan_trajectories.py."""
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import imageio.v2 as imageio

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "video"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GOLD, GREEN = "#3B6FB6", "#A07B2A", "#3A923A"
OBJ_FILL, FIN_FILL = "#C9D7EC", "#F2E3C8"
R_OBJ, R_FINGER, GOAL_R = 0.09, 0.06, 0.12
SEED, FPS = 1018, 20

JOBS = [("plan_vision_only", "plan_traj_V_s0_l0.02.npz"),
        ("plan_full", "plan_traj_VFX_s0_l0.005.npz")]


def write(tag, npz):
    d = np.load(HERE / npz)
    obj, fing, goal = d[f"obj_{SEED}"], d[f"fing_{SEED}"], d[f"goal_{SEED}"]
    T = len(obj)
    fig = plt.figure(figsize=(4.0, 4.0), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    path = OUT / f"{tag}.mp4"
    w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                           macro_block_size=1,
                           ffmpeg_params=["-pix_fmt", "yuv420p"])
    for t in range(T):
        ax.clear()
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white",
                               edgecolor="#1a1a1a", lw=2.5))
        ax.add_patch(Circle(goal, GOAL_R, facecolor=GREEN, alpha=0.16, lw=0))
        ax.add_patch(Circle(goal, 0.014, color=GREEN, zorder=5))
        lo = max(0, t - 40)
        seg = obj[lo:t + 1]
        for i in range(len(seg) - 1):
            a = 0.12 + 0.55 * (i / max(len(seg) - 1, 1))
            ax.plot(seg[i:i + 2, 0], seg[i:i + 2, 1], color=BLUE, lw=3.2,
                    alpha=a, solid_capstyle="round")
        segf = fing[lo:t + 1]
        ax.plot(segf[:, 0], segf[:, 1], color="#9E9E9E", lw=1.4, ls=":",
                alpha=0.85)
        ax.add_patch(Circle(obj[t], R_OBJ, facecolor=OBJ_FILL, edgecolor=BLUE,
                            lw=2.2))
        ax.add_patch(Circle(fing[t], R_FINGER, facecolor=FIN_FILL,
                            edgecolor=GOLD, lw=2.2))
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal"); ax.axis("off")
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    w.close()
    plt.close(fig)
    dist = np.linalg.norm(obj - goal, axis=1)
    print(f"wrote {path.name} ({path.stat().st_size // 1024} KB) "
          f"init {dist[0]:.3f} -> min {dist.min():.3f}")


if __name__ == "__main__":
    for tag, npz in JOBS:
        write(tag, npz)
