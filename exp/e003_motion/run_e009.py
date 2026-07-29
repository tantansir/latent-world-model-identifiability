"""e009 consolidation: complete 2-seed coverage of the full variant family at
the optimal lambda=0.02 (paper-grade statistics for every core contrast)."""
import json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

JOBS = [
    {"variant": "VX", "seed": 1, "lam": 0.02},
    {"variant": "VXt", "seed": 1, "lam": 0.02},
    {"variant": "VXp", "seed": 0, "lam": 0.02},
    {"variant": "VXp", "seed": 1, "lam": 0.02},
]


def tag_of(j):
    return f"{j['variant']}_s{j['seed']}_l{j['lam']}"


def main():
    for j in JOBS:
        tag = tag_of(j)
        if (RESULTS / f"{tag}.json").exists():
            print(f"=== {tag} done, skip ===", flush=True)
            continue
        cmd = [sys.executable, str(HERE / "train.py"), "--variant", j["variant"],
               "--seed", str(j["seed"]), "--lam", str(j["lam"])]
        if (RESULTS / f"{tag}.pt").exists():
            cmd.append("--eval_only")
        print(f"=== {tag} ===", flush=True)
        subprocess.run(cmd, check=True)

    print("\n=== FULL 2-SEED MATRIX (lambda=0.02) ===", flush=True)
    print("tag | m/g/k (contact) | velR2 mov | posR2 | roll@8 | direct@16", flush=True)
    for v in ("V", "VXp", "VXt", "VX", "VFX"):
        for s in (0, 1):
            p = RESULTS / f"{v}_s{s}_l0.02.json"
            if not p.exists():
                print(f"{v}_s{s}: missing", flush=True)
                continue
            r = json.loads(p.read_text())
            tc = r["probe"]["trunk"]["contact"]
            vel = r["probe"]["velocity_r2"]
            print(f"{v}_s{s}  | {tc['log_mass']:.3f}/{tc['gamma']:.3f}/{tc['log_stiffness']:.3f}"
                  f" | {vel['moving']:.3f} | {r['probe_r2_pos_frame']:.3f}"
                  f" | {r['rollout_pos_err']['8']:.4f} | {r['direct16_pos_err']:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
