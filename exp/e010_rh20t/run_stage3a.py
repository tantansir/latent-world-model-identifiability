"""Stage-3a: complete the objective-structure factorial on the cfg1 scaling
curve (review point: V vs VFX confounds input set with target set).
Adds VF (fused inputs, latent-only target) and VX (vision input, cross-modal
targets) at nep in {800, 2000, full}. Idempotent; continues past failures.
"""
import os, pathlib, subprocess, sys, time

E010 = pathlib.Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
env_c1 = dict(os.environ, E010_DATA="data_cfg1")

STEPS = [
    (["python", "train.py", "--variant", "VF", "--seed", "0"],
     E010 / "results" / "VF_s0_c1.json"),
    (["python", "train.py", "--variant", "VX", "--seed", "0"],
     E010 / "results" / "VX_s0_c1.json"),
    (["python", "train.py", "--variant", "VF", "--seed", "0", "--nep", "800"],
     E010 / "results" / "VF_s0_c1_n800.json"),
    (["python", "train.py", "--variant", "VX", "--seed", "0", "--nep", "800"],
     E010 / "results" / "VX_s0_c1_n800.json"),
    (["python", "train.py", "--variant", "VF", "--seed", "0", "--nep", "2000"],
     E010 / "results" / "VF_s0_c1_n2000.json"),
    (["python", "train.py", "--variant", "VX", "--seed", "0", "--nep", "2000"],
     E010 / "results" / "VX_s0_c1_n2000.json"),
]

failures = []
for cmd, done in STEPS:
    if done.exists():
        print(f"skip (done): {' '.join(cmd)}", flush=True)
        continue
    print(f"RUN: {' '.join(cmd)}  [{time.strftime('%H:%M:%S')}]", flush=True)
    r = subprocess.run(cmd, cwd=str(E010), env=env_c1)
    if r.returncode != 0:
        print(f"STEP FAILED rc={r.returncode}: {' '.join(cmd)} - continuing", flush=True)
        failures.append(" ".join(cmd))
print("STAGE3A DONE", "failures:", failures or "none", flush=True)
