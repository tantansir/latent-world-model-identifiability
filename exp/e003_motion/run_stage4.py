"""Stage-4: the linear-family certificate sweep (review round 4, point 1).
Bias ranges 0.5 (existing d5) / 1.5 (d7, slow with a strong certificate) /
2.5 (d8, the fast x linear cell), each with linear + GRU certificates and
trunk probes. Tests whether the model tracks its certificate across the
linear family. Idempotent; continues past failures.
"""
import pathlib, subprocess, sys, time

E003 = pathlib.Path(__file__).parent
E001 = E003.parent / "e001_xmodal_jepa"
R = E003 / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = ["python", "train.py", "--seed", "0", "--lam", "0.02", "--variant", "VFX"]
STEPS = [
    (E001, ["python", "pokeworld.py", "gain2"], E001 / "data" / "probe7_te_state.npz"),
    (E001, ["python", "pokeworld.py", "gain3"], E001 / "data" / "probe8_te_state.npz"),
    (E003, TRAIN + ["--dsuf", "7", "--psuf", "7"], R / "VFX_s0_l0.02_d7.json"),
    (E003, TRAIN + ["--dsuf", "8", "--psuf", "8"], R / "VFX_s0_l0.02_d8.json"),
    (E003, ["python", "review_fixes.py", "gain"], R / "gain_probe.json"),
    (E003, ["python", "review_fixes.py", "gain2"], R / "gain2_probe.json"),
    (E003, ["python", "review_fixes.py", "gain3"], R / "gain3_probe.json"),
]

failures = []
for wd, cmd, done in STEPS:
    if done.exists():
        print(f"skip (done): {' '.join(cmd)}", flush=True)
        continue
    print(f"RUN: {' '.join(cmd)}  [{time.strftime('%H:%M:%S')}]", flush=True)
    rc = subprocess.run(cmd, cwd=str(wd)).returncode
    if rc != 0:
        print(f"STEP FAILED rc={rc}: {' '.join(cmd)} - continuing", flush=True)
        failures.append(" ".join(cmd))
print("STAGE4 DONE", "failures:", failures or "none", flush=True)
