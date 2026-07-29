"""Project-page videos: matched PokeWorld episodes that start from identical
pixels and identical actions and differ only in one hidden parameter.

sample_props is patched so it still consumes the generator exactly as before
(keeping initial positions and the action stream identical across a pair) and
only the returned physical parameters are overridden.
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import imageio.v2 as imageio

HERE = pathlib.Path(__file__).resolve().parent          # docs/assets
sys.path.insert(0, str(HERE.parents[1] / "exp" / "e001_xmodal_jepa"))
import pokeworld as pw                                          # noqa: E402

OUT = HERE / "video"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GOLD = "#3B6FB6", "#A07B2A"
OBJ_FILL, FIN_FILL = "#C9D7EC", "#F2E3C8"
FPS = 20
N_EP, T = 64, 64

_orig_props = pw.sample_props


def run(props, seed, glide_frac=0.0):
    """One simulation whose episodes all carry the given (m, gamma, k)."""
    def patched(n, rng):
        p = _orig_props(n, rng)          # consume the stream identically
        p[:] = np.asarray(props, float)
        return p

    pw.sample_props = patched
    try:
        sim = pw.simulate(N_EP, T, seed=seed, glide_frac=glide_frac)
    finally:
        pw.sample_props = _orig_props
    return sim


def pick_episode(sim, glide):
    """An episode whose event happens early enough that its consequences play
    out on screen: a launch at t=0 for glides, an early strike otherwise."""
    best, score = 0, -1e9
    for e in range(N_EP):
        obj = sim["obj"][e]
        speed = np.linalg.norm(obj[:, 2:], axis=1)
        inside = ((obj[:, :2] > 0.10) & (obj[:, :2] < 0.90)).all()
        s = 0.5 * speed.max() + (1.0 if inside else -3.0)
        if glide:
            s += 2.0 * speed[0] - np.linalg.norm(obj[0, :2] - 0.5)
        else:
            contact = np.where(sim["touch"][e, :, 6] > 0.5)[0]
            if len(contact) == 0 or contact[0] > 18:
                continue                     # nothing to see, or too late
            t_c = contact[0]
            after = np.linalg.norm(obj[-1, :2] - obj[t_c, :2])
            s += 3.0 * after - 0.05 * t_c
        if s > score:
            best, score = e, s
    return best


def write_video(sim, e, path, trail=32):
    frames_obs = pw.render(sim["finger"][e:e + 1], sim["obj"][e:e + 1])[0]
    fig = plt.figure(figsize=(4.0, 4.0), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=1,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
    for t in range(T):
        ax.clear()
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white",
                               edgecolor="#1a1a1a", lw=2.5))
        lo = max(0, t - trail)
        if t > lo:
            seg = sim["obj"][e, lo:t + 1, :2]
            for i in range(len(seg) - 1):
                a = 0.12 + 0.55 * (i / max(len(seg) - 1, 1))
                ax.plot(seg[i:i + 2, 0], seg[i:i + 2, 1], color=BLUE,
                        lw=3.2, alpha=a, solid_capstyle="round")
        ax.add_patch(Circle(sim["obj"][e, t, :2], pw.R_OBJ,
                            facecolor=OBJ_FILL, edgecolor=BLUE, lw=2.2))
        ax.add_patch(Circle(sim["finger"][e, t, :2], pw.R_FINGER,
                            facecolor=FIN_FILL, edgecolor=GOLD, lw=2.2))
        # what the model actually sees, in the corner
        inset = fig.add_axes([0.735, 0.035, 0.225, 0.225])
        inset.imshow(frames_obs[t], cmap="gray", vmin=0, vmax=255,
                     origin="lower")
        inset.set_xticks([]); inset.set_yticks([])
        for s in inset.spines.values():
            s.set_color("#8C8C8C"); s.set_linewidth(1.2)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal"); ax.axis("off")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(buf)
        inset.remove()
    writer.close()
    plt.close(fig)
    print("wrote", path.name, f"({path.stat().st_size // 1024} KB)")


# (name, props, seed, glide) -- each pair shares its seed, so the pair starts
# from the same pixels and is driven by the same actions
JOBS = [
    ("drag_low",       (1.2, 0.5, 2000.0), 7, True),
    ("drag_high",      (1.2, 4.0, 2000.0), 7, True),
    ("mass_light",     (0.5, 1.5, 2000.0), 3, False),
    ("mass_heavy",     (3.0, 1.5, 2000.0), 3, False),
    ("stiff_soft",     (1.2, 1.5, 500.0),  11, False),
    ("stiff_hard",     (1.2, 1.5, 6000.0), 11, False),
]

if __name__ == "__main__":
    chosen = {}
    for name, props, seed, glide in JOBS:
        sim = run(props, seed, glide_frac=1.0 if glide else 0.0)
        key = (seed, glide)
        if key not in chosen:                    # the pair shares an episode
            chosen[key] = pick_episode(sim, glide)
        write_video(sim, chosen[key], OUT / f"{name}.mp4")
    print("done")
