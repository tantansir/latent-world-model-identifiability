"""e012: closed-loop planning in PokeWorld with the frozen world model.

Task: push the object so its center reaches a random goal. Controller: MPC with
CEM in action space, scoring 16-step action candidates by the model's direct
Delta=16 head decoded to object position. Baselines: random actions; the
vision-only model as planner. Usage: python plan_eval.py [V_s0_l0.02|VFX_s0_l0.005|random]
"""
import sys, json, pathlib
import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "e001_xmodal_jepa"))
sys.path.insert(0, str(HERE))                    # HERE first: its model.py must win
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pokeworld as pw
import model as M
from model import XJEPA

DATA = HERE.parent / "e001_xmodal_jepa" / "data"
device = "cuda"
TAG = sys.argv[1] if len(sys.argv) > 1 else "VFX_s0_l0.005"
T_CTX, H, REPLAN, T_MAX, GOAL_R = 16, 16, 4, 64, 0.12
N_EP, N_CEM, N_ELITE, CEM_IT = 100, 256, 32, 3

st_tr = np.load(DATA / "train_state.npz")
stats = {
    "prop_mu": st_tr["finger"].reshape(-1, 4).mean(0),
    "prop_sd": st_tr["finger"].reshape(-1, 4).std(0) + 1e-6,
    "touch_mu": st_tr["touch"].reshape(-1, 7)[:, :6].mean(0),
    "touch_sd": st_tr["touch"].reshape(-1, 7)[:, :6].std(0) + 1e-6,
}

model = None
if TAG != "random":
    model = XJEPA(TAG.split("_")[0]).to(device)
    model.load_state_dict(torch.load(HERE / "results" / f"{TAG}.pt", weights_only=True),
                          strict=False)   # motion_mu/sd buffers postdate these ckpts
    model.eval()

    # per-frame position probe fit on probe_tr
    img_p = np.load(DATA / "probe_tr_img.npy", mmap_mode="r")
    st_p = np.load(DATA / "probe_tr_state.npz")
    Z, Y = [], []
    with torch.no_grad():
        for s in range(0, 600, 64):
            ep = np.arange(s, s + 64)
            fr = torch.from_numpy(img_p[ep, :T_CTX]).to(device).float().div_(255)
            mo = torch.zeros_like(fr); mo[:, 1:] = fr[:, 1:] - fr[:, :-1]
            pr = torch.from_numpy((st_p["finger"][ep, :T_CTX] - stats["prop_mu"])
                                  / stats["prop_sd"]).to(device).float()
            to = st_p["touch"][ep, :T_CTX].copy()
            to[..., :6] = (to[..., :6] - stats["touch_mu"]) / stats["touch_sd"]
            to = torch.from_numpy(to).to(device).float()
            with torch.autocast("cuda", torch.bfloat16):
                z = model.encode(torch.stack([fr, mo], 2), pr, to)
            Z.append(z.float().cpu().numpy().reshape(-1, M.D))
            Y.append(np.concatenate([st_p["obj"][ep, :T_CTX, :2],
                                     st_p["finger"][ep, :T_CTX, :2]],
                                    -1).reshape(-1, 4))
    Z, Y = np.concatenate(Z), np.concatenate(Y)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    Wp = np.linalg.solve(((Z - mu) / sd).T @ ((Z - mu) / sd) + 10 * np.eye(M.D),
                         ((Z - mu) / sd).T @ ((Y - ymu) / ysd))
    print("object+finger position probe fitted", flush=True)


class SingleEnv:
    """Single-episode PokeWorld stepper mirroring pokeworld.simulate physics."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.props = pw.sample_props(1, rng)[0]
        self.p_f = rng.uniform(0.15, 0.85, 2)
        self.p_o = rng.uniform(0.25, 0.75, 2)
        while np.linalg.norm(self.p_f - self.p_o) < 0.2:
            self.p_o = rng.uniform(0.25, 0.75, 2)
        self.v_f = np.zeros(2); self.v_o = np.zeros(2)
        self.goal = rng.uniform(0.2, 0.8, 2)
        while np.linalg.norm(self.goal - self.p_o) < 0.25:
            self.goal = rng.uniform(0.2, 0.8, 2)

    def step(self, a):
        m_o, gamma_o, k_o = self.props
        dt_s = pw.DT / pw.SUBSTEPS
        m_red = np.array([m_o * pw.M_FINGER / (m_o + pw.M_FINGER)])
        acc = np.zeros(2); peak = 0.0; csub = 0; last = np.zeros(2)
        pf, po, vf, vo = (x.reshape(1, 2).copy() for x in
                          (self.p_f, self.p_o, self.v_f, self.v_o))
        for _ in range(pw.SUBSTEPS):
            f_on_obj = pw._contact_force(pf, po, vf, vo, pw.R_FINGER + pw.R_OBJ,
                                         np.array([k_o]), m_red)
            f_obj = f_on_obj - gamma_o * m_o * vo + pw._wall_force(po, vo, pw.R_OBJ, pw.K_WALL)
            f_fin = (pw.F_MAX * a.reshape(1, 2) - f_on_obj
                     - pw.GAMMA_FINGER * pw.M_FINGER * vf
                     + pw._wall_force(pf, vf, pw.R_FINGER, pw.K_WALL))
            fr = -f_on_obj[0]
            acc += fr * dt_s; mag = np.linalg.norm(fr)
            peak = max(peak, mag); csub += mag > 1e-6; last = fr
            vo += dt_s * f_obj / m_o; vf += dt_s * f_fin / pw.M_FINGER
            for v in (vo, vf):
                sp = np.linalg.norm(v)
                if sp > pw.V_CAP: v *= pw.V_CAP / sp
            po += dt_s * vo; pf += dt_s * vf
        self.p_f, self.p_o = pf[0], po[0]; self.v_f, self.v_o = vf[0], vo[0]
        favg = acc / pw.DT
        cf = csub / pw.SUBSTEPS
        touch = np.concatenate([favg, [peak], last, [cf], [float(cf > 0)]])
        touch[:5] += self.rng.normal(0, 0.02, 5)
        frame = pw.render(np.concatenate([self.p_f, self.v_f]).reshape(1, 1, 4),
                          np.concatenate([self.p_o, self.v_o]).reshape(1, 1, 4))[0, 0]
        return frame, np.concatenate([self.p_f, self.v_f]), touch


def run_episode(seed):
    global traj, viz_env
    env = SingleEnv(seed)
    viz_env = env
    traj = []
    frames, props_, touches, acts = [], [], [], []
    for _ in range(T_CTX):                       # settle-in with zero actions
        f, p, t = env.step(np.zeros(2))
        frames.append(f); props_.append(p); touches.append(t); acts.append(np.zeros(2))
    plan = np.zeros((H, 2))
    dists = []
    for t_step in range(T_MAX):
        if TAG != "random" and t_step % REPLAN == 0:
            fr = torch.from_numpy(np.array(frames[-T_CTX:])[None]).to(device).float().div_(255)
            mo = torch.zeros_like(fr); mo[:, 1:] = fr[:, 1:] - fr[:, :-1]
            pr = torch.from_numpy(((np.array(props_[-T_CTX:]) - stats["prop_mu"])
                                   / stats["prop_sd"])[None]).to(device).float()
            to = np.array(touches[-T_CTX:]).copy()
            to[:, :6] = (to[:, :6] - stats["touch_mu"]) / stats["touch_sd"]
            to = torch.from_numpy(to[None]).to(device).float()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                z = model.encode(torch.stack([fr, mo], 2), pr, to)
                h = model.pred.trunk(z.float(),
                                     torch.from_numpy(np.array(acts[-T_CTX:])[None]
                                                      ).to(device).float())
            h_last = h[:, -1:].float()
            mean = np.zeros((2, 2)); std = np.full((2, 2), 0.7)
            for _ in range(CEM_IT):
                knots = np.clip(mean + std * np.random.randn(N_CEM, 2, 2), -1, 1)
                w_lin = np.linspace(0, 1, H)[None, :, None]
                cand = knots[:, :1] * (1 - w_lin) + knots[:, 1:] * w_lin
                aw16 = torch.from_numpy(cand.reshape(N_CEM, 1, H * 2)).to(device).float()
                aw4 = torch.from_numpy(
                    np.ascontiguousarray(cand[:, :4]).reshape(N_CEM, 1, 8)
                ).to(device).float()
                with torch.no_grad():
                    z16 = model.dheads["16"](h_last.expand(N_CEM, 1, -1), aw16)[:, 0]
                    z4 = model.dheads["4"](h_last.expand(N_CEM, 1, -1), aw4)[:, 0]
                score = 0.0
                for zt, wgt in ((z16, 1.0), (z4, 0.5)):
                    dec = ((zt.cpu().numpy() - mu) / sd) @ Wp * ysd + ymu
                    obj, fing = dec[:, :2], dec[:, 2:]
                    away = obj - env.goal
                    away /= np.linalg.norm(away, axis=1, keepdims=True) + 1e-8
                    prepush = obj + 0.18 * away          # behind object w.r.t. goal
                    score = score - wgt * (np.linalg.norm(obj - env.goal, axis=1)
                                           + 0.4 * np.linalg.norm(fing - prepush, axis=1))
                elite = knots[np.argsort(score)[-N_ELITE:]]
                mean, std = elite.mean(0), elite.std(0) + 0.05
            wl = np.linspace(0, 1, H)[:, None]
            plan = np.clip(mean[0] * (1 - wl) + mean[1] * wl, -1, 1)
        a = plan[min(t_step % REPLAN, H - 1)] if TAG != "random" \
            else np.clip(np.random.randn(2) * 0.6, -1, 1)
        f, p, t = env.step(np.clip(a, -1, 1))
        frames.append(f); props_.append(p); touches.append(t); acts.append(a)
        dists.append(np.linalg.norm(env.p_o - env.goal))
        traj.append((env.p_o.copy(), env.p_f.copy()))
    return min(dists[-8:]), dists[-1], min(dists), dists[0]


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "viz":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for seed in range(1000, 1300):
        _, _, mind, initd = run_episode(seed)
        d = np.array([np.linalg.norm(o - viz_env.goal) for o, _ in traj])
        if mind < 0.08 and initd > 0.3 and d.max() < initd + 0.06:
            print(f"viz episode seed {seed}: min {mind:.3f}, init {initd:.3f}, "
                  f"max {d.max():.3f}", flush=True)
            break
    obj = np.array([t[0] for t in traj]); fing = np.array([t[1] for t in traj])
    goal = viz_env.goal
    np.savez(HERE / "results" / "plan_viz.npz", obj=obj, fing=fing, goal=goal)
    print("plan_viz.npz saved", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    results = [run_episode(1000 + i) for i in range(N_EP)]
    endd = np.array([r[0] for r in results])
    anyd = np.array([r[2] for r in results])
    initd = np.array([r[3] for r in results])
    out = {"tag": TAG, "n": N_EP,
           "success_end": float((endd < GOAL_R).mean()),
           "success_any": float((anyd < GOAL_R).mean()),
           "success_any15": float((anyd < 0.15).mean()),
           "median_end_dist": float(np.median(endd)),
           "median_any_dist": float(np.median(anyd)),
           "median_init_dist": float(np.median(initd)),
           "median_improvement": float(np.median(initd - anyd))}
    print(json.dumps(out), flush=True)
    (HERE / "results" / f"plan_{TAG}.json").write_text(json.dumps(out, indent=1))
