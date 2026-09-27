"""Fisher-information (Cram茅r鈥揜ao) recoverability certificate for PokeWorld.

For each 16-step window of a probe episode we replay the recorded actions
open-loop from the true state at the window start, with the hidden parameters
theta = (log m, gamma, log k) perturbed one at a time (central differences),
and build the Jacobian J = d obs / d theta of the window's observation vector.
With an observation-noise model Sigma, the Fisher information is
F = J^T Sigma^-1 J, and the Bayesian Cram茅r鈥揜ao bound on the per-window
posterior variance is diag((F + P)^-1) where P = prior precision
(1/Var(theta) under the sampling distribution). The best achievable R^2 of any
estimator that sees only that window is then  1 - E[CRLB_theta] / Var(theta).

Observation sets (what the estimator is allowed to see), mirroring the paper:
  PT     : finger proprioception (pos, vel) + 7-d touch + actions  (VX-style, no object)
  PT+O   : PT + object position observed through vision at a given pixel noise
  STATE  : PT + full object state (pos+vel) with small noise  (privileged, like the GRU certificate)
Noise model: touch force channels sigma=0.02 (as in the simulator); finger
state sigma=1e-3; object position sigma = px * (1/64); object velocity (STATE
only) sigma = 1e-3.
"""
import sys, json, pathlib, argparse
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent / "e001_xmodal_jepa"
sys.path.insert(0, str(REPO))
import pokeworld as pw  # constants + force functions only; we re-implement the step loop

T_CTX = 16


def replay(p_f0, v_f0, p_o0, v_o0, acts, m_o, gamma_o, k_o):
    """Open-loop replay for T steps. Inputs are [n,...] arrays; acts [T,n,2].
    Returns finger [T,n,4], obj [T,n,4], touch [T,n,7] (noise-free)."""
    n, T = acts.shape[1], acts.shape[0]
    p_f, v_f, p_o, v_o = p_f0.copy(), v_f0.copy(), p_o0.copy(), v_o0.copy()
    fin = np.zeros((T, n, 4)); obj = np.zeros((T, n, 4)); tou = np.zeros((T, n, 7))
    dt_s = pw.DT / pw.SUBSTEPS
    m_red = m_o * pw.M_FINGER / (m_o + pw.M_FINGER)
    for t in range(T):
        a = acts[t]
        touch_accum = np.zeros((n, 2)); f_peak = np.zeros(n); contact_sub = np.zeros(n)
        f_last = np.zeros((n, 2))
        for _ in range(pw.SUBSTEPS):
            f_on_obj = pw._contact_force(p_f, p_o, v_f, v_o, pw.R_FINGER + pw.R_OBJ, k_o, m_red)
            f_obj = (f_on_obj - gamma_o[:, None] * m_o[:, None] * v_o
                     + pw._wall_force(p_o, v_o, pw.R_OBJ, pw.K_WALL))
            f_fin = (pw.F_MAX * a - f_on_obj - pw.GAMMA_FINGER * pw.M_FINGER * v_f
                     + pw._wall_force(p_f, v_f, pw.R_FINGER, pw.K_WALL))
            f_react = -f_on_obj
            touch_accum += f_react * dt_s
            mag_now = np.linalg.norm(f_react, axis=1)
            f_peak = np.maximum(f_peak, mag_now)
            contact_sub += (mag_now > 1e-6)
            f_last = f_react
            v_o += dt_s * f_obj / m_o[:, None]
            v_f += dt_s * f_fin / pw.M_FINGER
            sp_o = np.linalg.norm(v_o, axis=1, keepdims=True)
            v_o *= np.minimum(1.0, pw.V_CAP / np.maximum(sp_o, 1e-8))
            sp_f = np.linalg.norm(v_f, axis=1, keepdims=True)
            v_f *= np.minimum(1.0, pw.V_CAP / np.maximum(sp_f, 1e-8))
            p_o += dt_s * v_o
            p_f += dt_s * v_f
        f_avg = touch_accum / pw.DT
        contact_frac = contact_sub / pw.SUBSTEPS
        tou[t] = np.concatenate([f_avg, f_peak[:, None], f_last, contact_frac[:, None],
                                 (contact_frac > 0)[:, None].astype(float)], 1)
        fin[t] = np.concatenate([p_f, v_f], 1)
        obj[t] = np.concatenate([p_o, v_o], 1)
    return fin, obj, tou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="probe_te")
    ap.add_argument("--n_ep", type=int, default=800)
    ap.add_argument("--eps", type=float, default=0.02, help="relative FD step in theta")
    ap.add_argument("--out", default="fim_certificate.json")
    ap.add_argument("--T", type=int, default=16)
    args = ap.parse_args()
    global T_CTX
    T_CTX = args.T

    st = np.load(REPO / "data" / f"{args.split}_state.npz")
    finger, obj, touch, act, props = (st[k] for k in ("finger", "obj", "touch", "act", "props"))
    n_all = min(args.n_ep, finger.shape[0])
    t0s = np.arange(0, pw.T_EP - T_CTX + 1, 8)
    # theta and prior precision from the sampling distribution
    m, g, k = props[:n_all].T.astype(float)
    theta = np.stack([np.log(m), g, np.log(k)], 1)           # [n,3]
    # exact prior variances: log-uniform -> uniform in log; gamma uniform
    var_prior = np.array([(np.log(3.0) - np.log(0.5)) ** 2 / 12, (4.0 - 0.5) ** 2 / 12,
                          (np.log(6000.0) - np.log(500.0)) ** 2 / 12])
    steps = np.array([args.eps * np.log(3.0 / 0.5), args.eps * 3.5, args.eps * np.log(6000 / 500)])

    results = {}
    # noise levels for the object position (vision), in pixels of a 64-px arena
    px_levels = [0.25, 0.5, 1.0, 2.0]
    sig_touch = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])  # last two are flags; keep small var
    sig_touch[5:] = 0.02
    sig_fin = 1e-3
    sig_ov = 1e-3

    acc = {}  # key -> list of per-window CRLB rows [3]
    meta = {"contact": [], "moving": []}
    for t0 in t0s:
        idx = np.arange(n_all)
        p_f0, v_f0 = finger[idx, t0, :2].astype(float), finger[idx, t0, 2:].astype(float)
        p_o0, v_o0 = obj[idx, t0, :2].astype(float), obj[idx, t0, 2:].astype(float)
        acts = act[idx, t0:t0 + T_CTX].transpose(1, 0, 2).astype(float)
        base = replay(p_f0, v_f0, p_o0, v_o0, acts, m, g, k)
        # Jacobians by central differences, one parameter at a time
        J = {}  # name -> [n, T, dim, 3]
        outs = {"fin": [], "obj": [], "tou": []}
        for j in range(3):
            th_p, th_m = theta.copy(), theta.copy()
            th_p[:, j] += steps[j]; th_m[:, j] -= steps[j]
            def unpack(th):
                return np.exp(th[:, 0]), th[:, 1], np.exp(th[:, 2])
            fp, op_, tp = replay(p_f0, v_f0, p_o0, v_o0, acts, *unpack(th_p))
            fm, om, tm = replay(p_f0, v_f0, p_o0, v_o0, acts, *unpack(th_m))
            for name, (a_, b_) in zip(("fin", "obj", "tou"), ((fp, fm), (op_, om), (tp, tm))):
                outs[name].append(((a_ - b_) / (2 * steps[j])).transpose(1, 0, 2))  # [n,T,dim]
        for name in outs:
            J[name] = np.stack(outs[name], -1)  # [n,T,dim,3]
        # window classes from the NOMINAL open-loop replay itself (float32 rounding of the recorded
        # state can flip grazing contacts relative to the stored touch flag; using the replay keeps
        # the mask consistent with the Jacobians, so zero-information cells come out at exactly 0)
        contact = (base[2][:, :, 6] > 0.5).sum(0)
        speed = np.linalg.norm(base[1][:, :, 2:], axis=2).mean(0)
        meta["contact"].append(contact); meta["moving"].append(speed)

        def fim_from(parts):
            # parts: list of (J [n,T,dim,3], sigma [dim])
            F = np.zeros((n_all, 3, 3))
            for Jp, sg in parts:
                Jw = Jp / sg[None, None, :, None]
                Jw = Jw.reshape(n_all, -1, 3)
                F += np.einsum("nti,ntj->nij", Jw, Jw)
            return F

        def crlb(F):
            P = np.diag(1.0 / var_prior)
            return np.stack([np.diag(np.linalg.inv(F[i] + P)) for i in range(n_all)])

        # PT: finger state + touch + (actions carry no theta info)
        J_fin, J_tou, J_obj = J["fin"], J["tou"], J["obj"]
        sg_fin = np.full(4, sig_fin)
        F_pt = fim_from([(J_fin, sg_fin), (J_tou, sig_touch)])
        acc.setdefault("PT", []).append(crlb(F_pt))
        for px in px_levels:
            sg_op = np.full(2, px / pw.RES)
            F = F_pt + fim_from([(J_obj[:, :, :2], sg_op)])
            acc.setdefault(f"PT+O@{px}px", []).append(crlb(F))
            # vision only + touch-free? (V-style: object + finger position only, no touch)
            F_v = fim_from([(J_fin[:, :, :2], sg_fin[:2]), (J_obj[:, :, :2], sg_op)])
            acc.setdefault(f"V@{px}px", []).append(crlb(F_v))
        F_state = F_pt + fim_from([(J_obj, np.array([sig_fin, sig_fin, sig_ov, sig_ov]))])
        acc.setdefault("STATE", []).append(crlb(F_state))
        print(f"t0={t0} done", flush=True)

    contact = np.concatenate(meta["contact"]); speed = np.concatenate(meta["moving"])
    masks = {"all": np.ones(len(contact), bool), "contact": contact >= 2,
             "glide": (contact == 0) & (speed > 0.05), "moving": speed > 0.05}
    for key, rows in acc.items():
        C = np.concatenate(rows)  # [N,3]
        results[key] = {}
        for mk, msk in masks.items():
            r2max = 1 - C[msk].mean(0) / var_prior
            results[key][mk] = {"log_m": round(float(r2max[0]), 3), "gamma": round(float(r2max[1]), 3),
                                "log_k": round(float(r2max[2]), 3), "n": int(msk.sum())}
    json.dump(results, open(args.out, "w"), indent=1)
    for key in results:
        print(key, {mk: (v["log_m"], v["gamma"], v["log_k"]) for mk, v in results[key].items()})


if __name__ == "__main__":
    main()
