"""Orchestrate e002: multi-timescale variants on validated e001 v3.1 data."""
import json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
VARIANTS = ("V", "VX", "VXt", "VFX")


def main():
    RESULTS.mkdir(exist_ok=True)
    for variant in VARIANTS:
        if (RESULTS / f"{variant}_s0.json").exists():
            print(f"=== {variant} already done, skip ===", flush=True)
            continue
        cmd = [sys.executable, str(HERE / "train.py"), "--variant", variant, "--seed", "0"]
        if (RESULTS / f"{variant}_s0.pt").exists():
            cmd.append("--eval_only")
            print(f"=== {variant}: checkpoint found, eval only ===", flush=True)
        else:
            print(f"=== training {variant} ===", flush=True)
        subprocess.run(cmd, check=True)

    print("\n=== SUMMARY e002 ===", flush=True)
    print("variant | ridge m/g/k (contact) | ridge g (moving) | mlp m/g/k (contact) "
          "| posR2 | roll@8 | roll@16 | direct@16 | static@16", flush=True)
    for variant in VARIANTS:
        r = json.loads((RESULTS / f"{variant}_s0.json").read_text())
        tc = r["probe"]["trunk"]["contact"]
        tm = r["probe"]["trunk"]["moving"]
        mc = r["probe"]["trunk_mlp"]["contact"]
        ro, st = r["rollout_pos_err"], r["static_baseline_err"]
        print(f"{variant:5s} | {tc['log_mass']:.3f}/{tc['gamma']:.3f}/{tc['log_stiffness']:.3f}"
              f" | {tm['gamma']:.3f}"
              f" | {mc['log_mass']:.3f}/{mc['gamma']:.3f}/{mc['log_stiffness']:.3f}"
              f" | {r['probe_r2_pos_frame']:.3f} | {ro['8']:.4f} | {ro['16']:.4f}"
              f" | {r['direct16_pos_err']:.4f} | {st['16']:.4f}", flush=True)


if __name__ == "__main__":
    main()
