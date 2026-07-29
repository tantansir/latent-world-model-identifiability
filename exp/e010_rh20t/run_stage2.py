"""Idempotent stage-2 runner: skips any step whose output artifact exists."""
import os, pathlib, subprocess, sys

E010 = pathlib.Path(r"C:\Users\Kaizh\Desktop\physical representation\exp\e010_rh20t")
E003 = pathlib.Path(r"C:\Users\Kaizh\Desktop\physical representation\exp\e003_motion")
E001 = pathlib.Path(r"C:\Users\Kaizh\Desktop\physical representation\exp\e001_xmodal_jepa")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

env_c1 = dict(os.environ, E010_DATA="data_cfg1")
STEPS = [
    (E010, ["python", "train.py", "--variant", "V", "--seed", "0", "--nep", "800",
            "--eval_only"], E010 / "results" / "V_s0_c1_n800.json", env_c1),
    (E010, ["python", "train.py", "--variant", "VFX", "--seed", "0", "--nep", "800"],
     E010 / "results" / "VFX_s0_c1_n800.json", env_c1),
    (E010, ["python", "train.py", "--variant", "V", "--seed", "0", "--nep", "2000"],
     E010 / "results" / "V_s0_c1_n2000.json", env_c1),
    (E010, ["python", "train.py", "--variant", "VFX", "--seed", "0", "--nep", "2000"],
     E010 / "results" / "VFX_s0_c1_n2000.json", env_c1),
    (E010, ["python", "train.py", "--variant", "V", "--seed", "0", "--steps", "30000"],
     E010 / "results" / "V_s0_c1_st30k.json", env_c1),
    (E001, ["python", "pokeworld.py", "gain"],
     E001 / "data" / "probe5_te_state.npz", None),
    (E003, ["python", "train.py", "--variant", "VFX", "--seed", "0", "--lam", "0.02",
            "--dsuf", "5", "--psuf", "5"],
     E003 / "results" / "VFX_s0_l0.02_d5.json", None),
    (E003, ["python", "review_fixes.py", "gain"],
     E003 / "results" / "gain_probe.json", None),
]

for wd, cmd, done, env in STEPS:
    if done.exists():
        print(f"skip (done): {' '.join(cmd)}", flush=True)
        continue
    print(f"RUN: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(wd), env=env)
    if r.returncode != 0:
        print(f"step failed rc={r.returncode}, aborting chain", flush=True)
        sys.exit(r.returncode)
print("STAGE2 ALL DONE", flush=True)
