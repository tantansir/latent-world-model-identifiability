"""Orchestrate e005: patch-token architecture, key contrast V vs VFX first."""
import json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
VARIANTS = ("V", "VFX")


def main():
    RESULTS.mkdir(exist_ok=True)
    for variant in VARIANTS:
        if (RESULTS / f"{variant}_s0.json").exists():
            print(f"=== {variant} already done, skip ===", flush=True)
            continue
        cmd = [sys.executable, str(HERE / "train.py"), "--variant", variant, "--seed", "0"]
        if (RESULTS / f"{variant}_s0.pt").exists():
            cmd.append("--eval_only")
        print(f"=== {variant} ===", flush=True)
        subprocess.run(cmd, check=True)

    print("\n=== SUMMARY e005 (patch tokens) ===", flush=True)
    print("variant | ridge m/g/k (contact) | g(moving) | velR2 mov | posR2 "
          "| roll@8 | roll@16 | direct@16", flush=True)
    for variant in VARIANTS:
        r = json.loads((RESULTS / f"{variant}_s0.json").read_text())
        tc = r["probe"]["trunk"]["contact"]
        tm = r["probe"]["trunk"]["moving"]
        v = r["probe"]["velocity_r2"]
        ro = r["rollout_pos_err"]
        print(f"{variant:5s} | {tc['log_mass']:.3f}/{tc['gamma']:.3f}/{tc['log_stiffness']:.3f}"
              f" | {tm['gamma']:.3f} | {v['moving']:.3f} | {r['probe_r2_pos_frame']:.3f}"
              f" | {ro['8']:.4f} | {ro['16']:.4f} | {r['direct16_pos_err']:.4f}", flush=True)


if __name__ == "__main__":
    main()
