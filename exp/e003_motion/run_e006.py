"""e006: future motion-image prediction targets (Delta 1/4/16) — the gamma
attack. VM = vision-only + motion targets; VFXM = everything + motion targets.
Compare against V_s0_l0.02 / VFX_s0_l0.02 baselines. Then run the functional
glide test (analyze_gamma.py) on each new checkpoint."""
import json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

JOBS = [
    {"variant": "VM", "seed": 0, "lam": 0.02},
    {"variant": "VFXM", "seed": 0, "lam": 0.02},
    # e006b: magnitude-weighted motion cells (anti-wash-out, cf. Finding #1)
    {"variant": "VM", "seed": 0, "lam": 0.02, "motw": 3},
    {"variant": "VFXM", "seed": 0, "lam": 0.02, "motw": 3},
]


def tag_of(j):
    t = f"{j['variant']}_s{j['seed']}"
    if j.get("dim", 128) != 128:
        t += f"_d{j['dim']}"
    if j.get("lam", 0.1) != 0.1:
        t += f"_l{j['lam']}"
    if j.get("motw", 0) > 0:
        t += f"_mw{j['motw']:g}"
    return t


def main():
    for j in JOBS:
        tag = tag_of(j)
        if (RESULTS / f"{tag}.json").exists():
            print(f"=== {tag} done, skip ===", flush=True)
            continue
        cmd = [sys.executable, str(HERE / "train.py"), "--variant", j["variant"],
               "--seed", str(j["seed"]), "--lam", str(j["lam"])]
        if j.get("motw", 0) > 0:
            cmd += ["--motw", str(j["motw"])]
        if (RESULTS / f"{tag}.pt").exists():
            cmd.append("--eval_only")
        print(f"=== {tag} ===", flush=True)
        subprocess.run(cmd, check=True)

    print("\n=== SUMMARY e006 (motion targets) vs lambda=0.02 baselines ===", flush=True)
    print("tag | m/g/k (contact) | g(moving) | velR2 mov | posR2 | roll@8 | direct@16",
          flush=True)
    for tag in ("V_s0_l0.02", "VM_s0_l0.02", "VM_s0_l0.02_mw3",
                "VFX_s0_l0.02", "VFXM_s0_l0.02", "VFXM_s0_l0.02_mw3"):
        r = json.loads((RESULTS / f"{tag}.json").read_text())
        tc = r["probe"]["trunk"]["contact"]
        tm = r["probe"]["trunk"]["moving"]
        v = r["probe"]["velocity_r2"]
        print(f"{tag:14s} | {tc['log_mass']:.3f}/{tc['gamma']:.3f}/{tc['log_stiffness']:.3f}"
              f" | {tm['gamma']:.3f} | {v['moving']:.3f} | {r['probe_r2_pos_frame']:.3f}"
              f" | {r['rollout_pos_err']['8']:.4f} | {r['direct16_pos_err']:.4f}", flush=True)

    for j in JOBS:
        tag = tag_of(j)
        print(f"\n=== functional glide test: {tag} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / "analyze_gamma.py"), tag], check=True)


if __name__ == "__main__":
    main()
