"""e007: frozen warp-consistency flow net as a visual VELOCITY SENSOR
(input channels), completing the sensor-derivative principle for vision.
Estimation-at-current-frame, not prediction-of-future-motion (which failed
twice in e006). Criteria: velocity R2 >= 0.9, gamma(moving) >= 0.3,
glide test < 0.049 (median-gamma baseline), k/m maintained."""
import json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

JOBS = [
    {"variant": "V", "seed": 0, "lam": 0.02, "flow": True},
    {"variant": "VFX", "seed": 0, "lam": 0.02, "flow": True},
]


def tag_of(j):
    t = f"{j['variant']}_s{j['seed']}"
    if j.get("lam", 0.1) != 0.1:
        t += f"_l{j['lam']}"
    if j.get("flow"):
        t += "_fl"
    return t


def main():
    for j in JOBS:
        tag = tag_of(j)
        if (RESULTS / f"{tag}.json").exists():
            print(f"=== {tag} done, skip ===", flush=True)
            continue
        cmd = [sys.executable, str(HERE / "train.py"), "--variant", j["variant"],
               "--seed", str(j["seed"]), "--lam", str(j["lam"]), "--flow"]
        if (RESULTS / f"{tag}.pt").exists():
            cmd.append("--eval_only")
        print(f"=== {tag} ===", flush=True)
        subprocess.run(cmd, check=True)

    print("\n=== SUMMARY e007 (flow sensor) ===", flush=True)
    print("tag | m/g/k (contact) | g(moving)/g(glide) | velR2 mov | posR2 "
          "| roll@8 | direct@16", flush=True)
    for tag in ("V_s0_l0.02", "V_s0_l0.02_fl", "VFX_s0_l0.02", "VFX_s0_l0.02_fl"):
        r = json.loads((RESULTS / f"{tag}.json").read_text())
        tc = r["probe"]["trunk"]["contact"]
        tm = r["probe"]["trunk"]["moving"]
        tg = r["probe"]["trunk"]["glide"]
        v = r["probe"]["velocity_r2"]
        print(f"{tag:15s} | {tc['log_mass']:.3f}/{tc['gamma']:.3f}/{tc['log_stiffness']:.3f}"
              f" | {tm['gamma']:.3f}/{tg['gamma']:.3f} | {v['moving']:.3f}"
              f" | {r['probe_r2_pos_frame']:.3f} | {r['rollout_pos_err']['8']:.4f}"
              f" | {r['direct16_pos_err']:.4f}", flush=True)

    for j in JOBS:
        tag = tag_of(j)
        print(f"\n=== functional glide test: {tag} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / "analyze_gamma.py"), tag], check=True)


if __name__ == "__main__":
    main()
