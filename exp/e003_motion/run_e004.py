"""e004 overnight: (a) seed-1 replication of all e003 variants; (b) lambda/dim
sweep on VFX and V to test the Gaussian-fill precision-ceiling hypothesis."""
import json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

JOBS = [
    # seed replication
    {"variant": "V", "seed": 1}, {"variant": "VX", "seed": 1},
    {"variant": "VXt", "seed": 1}, {"variant": "VFX", "seed": 1},
    # dim sweep (lam fixed 0.1)
    {"variant": "VFX", "seed": 0, "dim": 64}, {"variant": "VFX", "seed": 0, "dim": 256},
    {"variant": "V", "seed": 0, "dim": 256},
    # lambda sweep (dim fixed 128)
    {"variant": "VFX", "seed": 0, "lam": 0.02}, {"variant": "VFX", "seed": 0, "lam": 0.3},
    {"variant": "V", "seed": 0, "lam": 0.02},
    # e004b: all variants at optimal lambda + cliff search + seed check
    {"variant": "VX", "seed": 0, "lam": 0.02}, {"variant": "VXt", "seed": 0, "lam": 0.02},
    {"variant": "VFX", "seed": 1, "lam": 0.02}, {"variant": "V", "seed": 1, "lam": 0.02},
    {"variant": "VFX", "seed": 0, "lam": 0.01}, {"variant": "VFX", "seed": 0, "lam": 0.005},
]


def tag_of(j):
    t = f"{j['variant']}_s{j['seed']}"
    if j.get("dim", 128) != 128:
        t += f"_d{j['dim']}"
    if j.get("lam", 0.1) != 0.1:
        t += f"_l{j['lam']}"
    return t


def main():
    RESULTS.mkdir(exist_ok=True)
    for j in JOBS:
        tag = tag_of(j)
        if (RESULTS / f"{tag}.json").exists():
            print(f"=== {tag} done, skip ===", flush=True)
            continue
        cmd = [sys.executable, str(HERE / "train.py"), "--variant", j["variant"],
               "--seed", str(j["seed"])]
        if "dim" in j:
            cmd += ["--dim", str(j["dim"])]
        if "lam" in j:
            cmd += ["--lam", str(j["lam"])]
        if (RESULTS / f"{tag}.pt").exists():
            cmd.append("--eval_only")
        print(f"=== {tag} ===", flush=True)
        subprocess.run(cmd, check=True)

    print("\n=== SUMMARY e004 ===", flush=True)
    print("tag | m/g/k (contact) | velR2 mov | posR2 | roll@8 | direct@16", flush=True)
    for j in JOBS:
        tag = tag_of(j)
        r = json.loads((RESULTS / f"{tag}.json").read_text())
        tc = r["probe"]["trunk"]["contact"]
        v = r["probe"]["velocity_r2"]
        print(f"{tag:16s} | {tc['log_mass']:.3f}/{tc['gamma']:.3f}/{tc['log_stiffness']:.3f}"
              f" | {v['moving']:.3f} | {r['probe_r2_pos_frame']:.3f}"
              f" | {r['rollout_pos_err']['8']:.4f} | {r['direct16_pos_err']:.4f}", flush=True)


if __name__ == "__main__":
    main()
