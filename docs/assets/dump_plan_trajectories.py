"""Run the frozen-latent MPC controller on a fixed set of seeds and dump the
resulting trajectories, so the project page can show the same task solved (or
not) by two representations. Usage: python dump_plan_trajectories.py <TAG>"""
import pathlib
import sys

import numpy as np

TAG = sys.argv[1]
SEEDS = ([int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2
         else list(range(1000, 1030)))

HERE = pathlib.Path(__file__).resolve().parent
E003 = HERE.parents[1] / "exp" / "e003_motion"
sys.argv = ["plan_eval.py", TAG]
sys.path.insert(0, str(E003))
import plan_eval as pe                                            # noqa: E402

out = {}
for seed in SEEDS:
    pe.run_episode(seed)
    obj = np.array([t[0] for t in pe.traj])
    fing = np.array([t[1] for t in pe.traj])
    goal = pe.viz_env.goal
    d = np.linalg.norm(obj - goal, axis=1)
    out[f"obj_{seed}"] = obj
    out[f"fing_{seed}"] = fing
    out[f"goal_{seed}"] = goal
    print(f"seed {seed}: init {d[0]:.3f} min {d.min():.3f} end {d[-1]:.3f}",
          flush=True)

np.savez(HERE / f"plan_traj_{TAG}.npz", **out)
print("saved", f"plan_traj_{TAG}.npz")
