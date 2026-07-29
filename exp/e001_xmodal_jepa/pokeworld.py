"""PokeWorld-2D: a minimal interactive multimodal physics world.

A force-controlled finger pokes one visually-identical object whose hidden
physical properties (mass m, drag rate gamma, contact stiffness k) are
resampled every episode. Sensors: 64x64 grayscale vision, 4-d proprioception
(finger pos+vel), 7-d touch with sub-step (kHz-like) statistics:
[Fx_avg, Fy_avg, |F|_peak, Fx_end, Fy_end, contact_frac, contact_bin].
Peak force and contact duration carry the stiffness signature that plain
frame-averaged force washes out (quasi-static push force is k-independent).

Physics: semi-implicit Euler, Hertzian-style penalty contacts with normal
damping. Drag force = -gamma * m * v so the visual glide decay rate gamma is
mass-independent: gamma is visually observable, m only via force response.
Behavior policy mixes pursuit (sustained pushes), strike (hit-retreat cycles
producing clean collisions and free glides) and OU noise.
"""
import numpy as np
import pathlib

RES = 64
DT = 0.05
SUBSTEPS = 20
T_EP = 64
ZETA = 0.25
R_FINGER = 0.06
R_OBJ = 0.09
M_FINGER = 1.0
GAMMA_FINGER = 2.0
F_MAX = 6.0
K_WALL = 3000.0
V_CAP = 3.0

DATA_DIR = pathlib.Path(__file__).parent / "data"


def sample_props(n, rng):
    m = np.exp(rng.uniform(np.log(0.5), np.log(3.0), n))
    gamma = rng.uniform(0.5, 4.0, n)
    k = np.exp(rng.uniform(np.log(500.0), np.log(6000.0), n))
    return np.stack([m, gamma, k], 1).astype(np.float64)


def _contact_force(p_a, p_b, v_a, v_b, r_sum, k, m_red):
    """Force on body b from penalty contact with body a. Shapes [n,2].
    Underdamped (ZETA) so peak force stays spring-dominated and collisions
    bounce - the stiffness signature survives in the tactile transient."""
    d = p_b - p_a
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    dist = np.maximum(dist, 1e-8)
    n_hat = d / dist
    overlap = np.maximum(r_sum - dist, 0.0)
    active = overlap[:, 0] > 0
    f_spring = k[:, None] * overlap ** 1.5
    v_rel = (v_b - v_a)
    v_n = np.sum(v_rel * n_hat, axis=1, keepdims=True)
    c = 2.0 * ZETA * np.sqrt(k * m_red)[:, None]
    f_damp = -c * v_n * overlap ** 0.5
    f = np.where(active[:, None], (f_spring + f_damp), 0.0) * n_hat
    return f


def _wall_force(p, v, r, k_wall):
    """Penalty force from the four walls of [0,1]^2. p,v: [n,2]."""
    f = np.zeros_like(p)
    for axis in range(2):
        low = np.maximum(r - p[:, axis], 0.0)
        f[:, axis] += k_wall * low ** 1.5 - 2.0 * np.sqrt(k_wall) * v[:, axis] * (low > 0) * low ** 0.5
        high = np.maximum(p[:, axis] - (1 - r), 0.0)
        f[:, axis] -= k_wall * high ** 1.5 + 2.0 * np.sqrt(k_wall) * v[:, axis] * (high > 0) * high ** 0.5
    return f


def simulate(n_ep, T, seed, glide_frac=0.0, drift=False, gain=False, mgain=False,
             bias_range=0.5):
    """Vectorized simulation of n_ep episodes for T steps. glide_frac episodes
    are pure-glide: finger parked in a corner with weak drift, object launched
    fast - supplies the gamma-informative gradient mass that pursuit-heavy
    data starves (e008 data-coverage hypothesis)."""
    rng = np.random.default_rng(seed)
    props = sample_props(n_ep, rng)
    if drift:
        # fix gamma so the slow-LINEAR drift is not confounded with per-episode
        # drag in dv/dt = a_d - gamma*v (certificate gate caught this: with
        # gamma varying, a_d is unrecoverable from 16-step windows)
        props[:, 1] = 1.5
    m_o, gamma_o, k_o = props[:, 0], props[:, 1], props[:, 2]

    p_f = rng.uniform(0.15, 0.85, (n_ep, 2))
    p_o = rng.uniform(0.2, 0.8, (n_ep, 2))
    bad = np.linalg.norm(p_f - p_o, axis=1) < (R_FINGER + R_OBJ + 0.05)
    while bad.any():
        p_o[bad] = rng.uniform(0.2, 0.8, (bad.sum(), 2))
        bad = np.linalg.norm(p_f - p_o, axis=1) < (R_FINGER + R_OBJ + 0.05)
    v_f = np.zeros((n_ep, 2))
    v_o = np.zeros((n_ep, 2))
    moving = rng.random(n_ep) < 0.6
    ang = rng.uniform(0, 2 * np.pi, n_ep)
    spd = rng.uniform(0.15, 0.9, n_ep) * moving
    v_o[:, 0] = spd * np.cos(ang)
    v_o[:, 1] = spd * np.sin(ang)

    # slow-LINEAR hidden parameter for the blind-class attribute matrix (e013):
    # a constant per-episode drift acceleration on the object. Per-step effect is
    # sub-pixel (slow) but cumulative displacement is LINEAR in a_d (not a ratio).
    a_d = rng.uniform(-0.15, 0.15, (n_ep, 2)) if drift else np.zeros((n_ep, 2))
    # slow-LINEAR, ungated parameter (attribute matrix, 3rd construction): a
    # per-episode actuation gain on the finger. The finger is always driven, so
    # no temporal gating is needed to read g; the response is linear in g; and
    # the per-step effect (g-1)*F*dt^2 is sub-pixel.
    # 4th construction: multiplicative gains are inherently ratio-type
    # (g = dv/(F*a) divides by the action), so the slow-LINEAR cell must be
    # ADDITIVE: a constant per-episode force bias on the finger.
    # dv = (F*a + b - c*v)*dt is linear in (a, v, b).
    # mgain: the multiplicative version g in [0.85, 1.15], kept as its own cell
    # of the attribute matrix (slow x RATIO-type x UNGATED x high loss share):
    # the finger is always driven, so no temporal gate is needed to read g, but
    # g = dv/(F*a) divides by the action - a ratio-type readout. Its linear
    # certificate fails BY DESIGN (that failure certifies the ratio attribute);
    # recoverability is certified by a GRU sequence probe instead.
    # bias_range sets the linear family's SNR: 0.5 (original slow cell), 1.5
    # (slow but with a strong certificate: per-step displacement effect
    # ~b*dt^2/2 = 0.002 stays sub-pixel), 2.5 (the fast x linear cell).
    g_act = rng.uniform(0.85, 1.15, n_ep) if mgain else np.ones(n_ep)
    b_bias = (rng.uniform(-bias_range, bias_range, (n_ep, 2)) if gain
              else np.zeros((n_ep, 2)))

    a = np.zeros((n_ep, 2))
    mode = rng.integers(0, 3, n_ep)          # 0=pursuit, 1=strike, 2=OU, 3=glide
    glide_ep = rng.random(n_ep) < glide_frac
    if glide_ep.any():
        mode[glide_ep] = 3
        ng = glide_ep.sum()
        corner = rng.integers(0, 4, ng)
        p_f[glide_ep, 0] = np.where(corner % 2 == 0, 0.12, 0.88) + rng.normal(0, 0.02, ng)
        p_f[glide_ep, 1] = np.where(corner < 2, 0.12, 0.88) + rng.normal(0, 0.02, ng)
        ang_g = rng.uniform(0, 2 * np.pi, ng)
        spd_g = rng.uniform(0.3, 1.2, ng)
        v_o[glide_ep, 0] = spd_g * np.cos(ang_g)
        v_o[glide_ep, 1] = spd_g * np.sin(ang_g)
    phase_off = rng.integers(0, 10, n_ep)

    finger_traj = np.zeros((T, n_ep, 4), np.float32)
    obj_traj = np.zeros((T, n_ep, 4), np.float32)
    touch_traj = np.zeros((T, n_ep, 7), np.float32)
    act_traj = np.zeros((T, n_ep, 2), np.float32)

    dt_s = DT / SUBSTEPS
    for t in range(T):
        toggle = (rng.random(n_ep) < 0.03) & (mode != 3)
        mode = np.where(toggle, rng.integers(0, 3, n_ep), mode)
        noise = rng.normal(0, 1, (n_ep, 2))
        to_obj = p_o - p_f
        to_obj /= np.linalg.norm(to_obj, axis=1, keepdims=True) + 1e-8
        a_pursuit = 1.2 * to_obj + 0.4 * noise
        approach = ((t + phase_off) % 10) < 6
        a_strike = np.where(approach[:, None], 1.4 * to_obj, -1.0 * to_obj) + 0.3 * noise
        a_ou = 0.85 * a + 0.45 * noise
        a = np.where(mode[:, None] == 0, a_pursuit,
                     np.where(mode[:, None] == 1, a_strike,
                              np.where(mode[:, None] == 2, a_ou, 0.15 * noise)))
        a = np.clip(a, -1, 1)
        act_traj[t] = a

        touch_accum = np.zeros((n_ep, 2))
        f_peak = np.zeros(n_ep)
        contact_sub = np.zeros(n_ep)
        f_last = np.zeros((n_ep, 2))
        m_red = m_o * M_FINGER / (m_o + M_FINGER)
        for _ in range(SUBSTEPS):
            f_on_obj = _contact_force(p_f, p_o, v_f, v_o, R_FINGER + R_OBJ, k_o, m_red)
            f_obj = (f_on_obj - gamma_o[:, None] * m_o[:, None] * v_o
                     + m_o[:, None] * a_d + _wall_force(p_o, v_o, R_OBJ, K_WALL))
            f_fin = (g_act[:, None] * F_MAX * a + b_bias - f_on_obj
                     - GAMMA_FINGER * M_FINGER * v_f
                     + _wall_force(p_f, v_f, R_FINGER, K_WALL))
            f_react = -f_on_obj
            touch_accum += f_react * dt_s
            mag_now = np.linalg.norm(f_react, axis=1)
            f_peak = np.maximum(f_peak, mag_now)
            contact_sub += (mag_now > 1e-6)
            f_last = f_react
            v_o += dt_s * f_obj / m_o[:, None]
            v_f += dt_s * f_fin / M_FINGER
            sp_o = np.linalg.norm(v_o, axis=1, keepdims=True)
            v_o *= np.minimum(1.0, V_CAP / np.maximum(sp_o, 1e-8))
            sp_f = np.linalg.norm(v_f, axis=1, keepdims=True)
            v_f *= np.minimum(1.0, V_CAP / np.maximum(sp_f, 1e-8))
            p_o += dt_s * v_o
            p_f += dt_s * v_f

        f_avg = touch_accum / DT
        contact_frac = contact_sub / SUBSTEPS
        contact_bin = (contact_frac > 0).astype(np.float32)
        touch = np.concatenate([f_avg, f_peak[:, None], f_last,
                                contact_frac[:, None], contact_bin[:, None]], 1)
        touch[:, :5] += rng.normal(0, 0.02, (n_ep, 5))

        finger_traj[t] = np.concatenate([p_f, v_f], 1)
        obj_traj[t] = np.concatenate([p_o, v_o], 1)
        touch_traj[t] = touch

    return {
        "finger": finger_traj.transpose(1, 0, 2),
        "obj": obj_traj.transpose(1, 0, 2),
        "touch": touch_traj.transpose(1, 0, 2),
        "act": act_traj.transpose(1, 0, 2),
        "props": props.astype(np.float32),
        "drift": a_d.astype(np.float32),
        "gain": b_bias.astype(np.float32),
        "mgain": g_act.astype(np.float32),
    }


def render(finger, obj, chunk=512):
    """finger, obj: [n, T, 4] -> uint8 [n, T, RES, RES]."""
    n, T = finger.shape[:2]
    lin = (np.arange(RES) + 0.5) / RES
    yy, xx = np.meshgrid(lin, lin, indexing="ij")
    aa = 1.0 / RES
    out = np.zeros((n, T, RES, RES), np.uint8)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        img = np.zeros((e - s, T, RES, RES), np.float32)
        for pos, r, val in ((obj[s:e, :, :2], R_OBJ, 0.55), (finger[s:e, :, :2], R_FINGER, 1.0)):
            dx = xx[None, None] - pos[:, :, 0, None, None]
            dy = yy[None, None] - pos[:, :, 1, None, None]
            dist = np.sqrt(dx * dx + dy * dy)
            alpha = np.clip((r - dist) / aa + 0.5, 0.0, 1.0) * val
            img = np.maximum(img, alpha)
        out[s:e] = (img * 255).astype(np.uint8)
    return out


def generate_split(name, n_ep, seed, glide_frac=0.0, drift=False, gain=False,
                   mgain=False, bias_range=0.5):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = simulate(n_ep, T_EP, seed, glide_frac=glide_frac, drift=drift, gain=gain,
                    mgain=mgain, bias_range=bias_range)
    img = render(data["finger"], data["obj"])
    np.save(DATA_DIR / f"{name}_img.npy", img)
    np.savez(DATA_DIR / f"{name}_state.npz", **data)
    contact_frac = (data["touch"][:, :, 6] > 0.5).mean()
    print(f"{name}: {n_ep} ep, contact fraction {contact_frac:.3f}", flush=True)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "glide":
        generate_split("train2", 4000, 20, glide_frac=0.3)
        generate_split("val2", 400, 21, glide_frac=0.3)
    elif len(_sys.argv) > 1 and _sys.argv[1] == "drift":
        generate_split("train3", 4000, 30, glide_frac=0.3, drift=True)
        generate_split("val3", 400, 31, glide_frac=0.3, drift=True)
        generate_split("probe3_tr", 1600, 32, glide_frac=0.3, drift=True)
        generate_split("probe3_te", 800, 33, glide_frac=0.3, drift=True)
    elif len(_sys.argv) > 1 and _sys.argv[1] == "gain":
        generate_split("train5", 4000, 50, glide_frac=0.3, gain=True)
        generate_split("val5", 400, 51, glide_frac=0.3, gain=True)
        generate_split("probe5_tr", 1600, 52, glide_frac=0.3, gain=True)
        generate_split("probe5_te", 800, 53, glide_frac=0.3, gain=True)
    elif len(_sys.argv) > 1 and _sys.argv[1] == "gain2":
        for nm, n, sd in (("train7", 4000, 70), ("val7", 400, 71),
                          ("probe7_tr", 1600, 72), ("probe7_te", 800, 73)):
            generate_split(nm, n, sd, glide_frac=0.3, gain=True, bias_range=1.5)
    elif len(_sys.argv) > 1 and _sys.argv[1] == "gain3":
        for nm, n, sd in (("train8", 4000, 80), ("val8", 400, 81),
                          ("probe8_tr", 1600, 82), ("probe8_te", 800, 83)):
            generate_split(nm, n, sd, glide_frac=0.3, gain=True, bias_range=2.5)
    elif len(_sys.argv) > 1 and _sys.argv[1] == "mgain":
        generate_split("train6", 4000, 60, glide_frac=0.3, mgain=True)
        generate_split("val6", 400, 61, glide_frac=0.3, mgain=True)
        generate_split("probe6_tr", 1600, 62, glide_frac=0.3, mgain=True)
        generate_split("probe6_te", 800, 63, glide_frac=0.3, mgain=True)
    elif len(_sys.argv) > 1 and _sys.argv[1] == "poorcontact":
        generate_split("train4", 4000, 40, glide_frac=0.95)
        generate_split("val4", 400, 41, glide_frac=0.95)
    else:
        generate_split("train", 4000, 10)
        generate_split("val", 400, 11)
        generate_split("probe_tr", 1600, 12)
        generate_split("probe_te", 800, 13)
    print("done", flush=True)
