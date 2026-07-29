"""Orchestrate experiment e001: generate data, oracle identifiability check,
train all four variants sequentially, print the comparison table."""
import json, pathlib, subprocess, sys
import numpy as np

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def oracle_check():
    """GRU sequence probe from raw 16-step windows -> hidden props.
    Identifiability sanity. The properties are NONLINEAR, temporally GATED
    functions of the observables (glide-frame velocity ratios for gamma,
    contact-frame impulse/dv for m, onset transients for k), so the oracle
    needs a recurrent probe; a flat MLP provably underperforms (analytic
    gamma estimator hits R2=0.89 where the MLP scored 0.0)."""
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(HERE))
    from train import prop_labels, T_CTX

    def build(name):
        st = np.load(DATA / f"{name}_state.npz")
        state = {k: st[k] for k in st.files}
        n = state["finger"].shape[0]
        t0s = np.array([0, 8, 16, 24, 32, 40])
        ep = np.repeat(np.arange(n), len(t0s))
        t0 = np.tile(t0s, n)
        feats = np.stack([np.concatenate(
            [state["finger"][e, t:t + T_CTX], state["obj"][e, t:t + T_CTX],
             state["touch"][e, t:t + T_CTX], state["act"][e, t:t + T_CTX]],
            1) for e, t in zip(ep, t0)])                    # [N, T_CTX, 17]
        contact = np.array([state["touch"][e, t:t + T_CTX, 6].sum()
                            for e, t in zip(ep, t0)])
        return feats.astype(np.float32), prop_labels(state, ep).astype(np.float32), contact

    def r2(y, p):
        return 1 - ((y - p) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)

    class GRUProbe(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.gru = nn.GRU(d_in, 96, num_layers=2, batch_first=True)
            self.head = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 3))

        def forward(self, x):
            h, _ = self.gru(x)
            return self.head(torch.cat([h[:, -1], h.mean(1)], -1))

    Xtr, ytr, ctr = build("probe_tr")
    Xte, yte, cte = build("probe_te")
    mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-6
    out = {}
    for label, mtr, mte in (("all", np.ones(len(Xtr), bool), np.ones(len(Xte), bool)),
                            ("contact2", ctr >= 2, cte >= 2),
                            ("contact5", ctr >= 5, cte >= 5)):
        xtr = torch.tensor((Xtr[mtr] - mu) / sd, device="cuda")
        ytr_ = torch.tensor((ytr[mtr] - ymu) / ysd, device="cuda")
        xte = torch.tensor((Xte[mte] - mu) / sd, device="cuda")
        torch.manual_seed(0)
        net = GRUProbe(Xtr.shape[-1]).cuda()
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4000)
        for i in range(4000):
            idx = torch.randint(0, len(xtr), (256,), device="cuda")
            loss = (net(xtr[idx]) - ytr_[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            pred = torch.cat([net(xte[s:s + 2048]) for s in range(0, len(xte), 2048)])
        pred = pred.cpu().numpy() * ysd + ymu
        out[f"oracle_{label}"] = dict(zip(("log_mass", "gamma", "log_stiffness"),
                                          map(float, r2(yte[mte], pred))))
        out[f"n_{label}"] = int(mte.sum())
    (RESULTS / "oracle.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


def main():
    RESULTS.mkdir(exist_ok=True)
    if not (DATA / "train_img.npy").exists():
        print("=== generating data ===", flush=True)
        subprocess.run([sys.executable, str(HERE / "pokeworld.py")], check=True)
    if not (RESULTS / "oracle.json").exists():
        print("=== oracle identifiability check ===", flush=True)
        oracle_check()
    orc = json.loads((RESULTS / "oracle.json").read_text())
    # Judge identifiability on contact windows: a resting object reveals
    # nothing about any of its dynamical properties (including gamma).
    gate = {"gamma(contact2)": orc["oracle_contact2"]["gamma"],
            "log_mass(contact2)": orc["oracle_contact2"]["log_mass"],
            "log_stiffness(contact2)": orc["oracle_contact2"]["log_stiffness"]}
    if gate["gamma(contact2)"] < 0.4 or gate["log_mass(contact2)"] < 0.4 \
            or gate["log_stiffness(contact2)"] < 0.25:
        print(f"ABORT: env not identifiable enough {gate}; "
              "fix the environment before training.", flush=True)
        sys.exit(2)
    VARIANTS = ("V", "VF", "VX", "VFX", "VXt", "VXp")
    for variant in VARIANTS:
        if (RESULTS / f"{variant}_s0.json").exists():
            print(f"=== {variant} already done, skip ===", flush=True)
            continue
        print(f"=== training {variant} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / "train.py"),
                        "--variant", variant, "--seed", "0"], check=True)

    print("\n=== SUMMARY (contact windows) ===", flush=True)
    rows = []
    for variant in VARIANTS:
        r = json.loads((RESULTS / f"{variant}_s0.json").read_text())
        rows.append(r)
    hdr = ("variant | trunk m/g/k | zcat m/g/k | posR2 | roll@4 | roll@8 | roll@16 | static@8")
    print(hdr, flush=True)
    for r in rows:
        tk, zc = r["probe"]["trunk"]["contact"], r["probe"]["zcat"]["contact"]
        ro, st = r["rollout_pos_err"], r["static_baseline_err"]
        print(f"{r['variant']:7s} | {tk['log_mass']:.3f}/{tk['gamma']:.3f}/{tk['log_stiffness']:.3f}"
              f" | {zc['log_mass']:.3f}/{zc['gamma']:.3f}/{zc['log_stiffness']:.3f}"
              f" | {r['probe_r2_pos_frame']:.3f} | {ro['4']:.4f} | {ro['8']:.4f}"
              f" | {ro['16']:.4f} | {st['8']:.4f}", flush=True)


if __name__ == "__main__":
    main()
