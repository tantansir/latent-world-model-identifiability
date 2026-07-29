"""Stage-3b: attribute-matrix completion + objective-family coverage.
  1) RAND: untrained-encoder readout control on cfg1 (input-passthrough bound)
  2) sysID auxiliary head control (objective-attribution for the blind region)
  3) reconstruction-objective baselines V/VFX (family coverage)
  4) gamma loss-share sweep gw in {4,16,64} + functional glide tests
  5) multiplicative-gain cell (slow x ratio x UNGATED): data + train + GRU cert
  6) stiffness loss-share starvation tw in {1, 0.25, 0} (fast x ratio x low share)
  7) VF at lambda=0.02 (completes the toy 2x2 input/target factorial)
Idempotent; continues past failures.
"""
import os, pathlib, subprocess, sys, time

E003 = pathlib.Path(__file__).parent
E001 = E003.parent / "e001_xmodal_jepa"
E010 = E003.parent / "e010_rh20t"
R = E003 / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
env_c1 = dict(os.environ, E010_DATA="data_cfg1")

TRAIN = ["python", "train.py", "--seed", "0", "--lam", "0.02", "--variant"]
STEPS = [
    (E010, ["python", "rand_baseline.py"], E010 / "results" / "RAND_c1.json",
     env_c1, None),
    (E003, TRAIN + ["VFX", "--sysid"], R / "VFX_s0_l0.02_sid.json", None, None),
    (E003, TRAIN + ["V", "--recon"], R / "V_s0_l0.02_rc.json", None, None),
    (E003, TRAIN + ["VFX", "--recon"], R / "VFX_s0_l0.02_rc.json", None, None),
]
for w in ("4", "16", "64"):
    tag = f"VFX_s0_l0.02_fl_g_gw{w}"
    STEPS += [
        (E003, TRAIN + ["VFX", "--flow", "--data2", "--glidew", w],
         R / f"{tag}.json", None, None),
        (E003, ["python", "analyze_gamma.py", tag],
         R / f"glide_{tag}.txt", None, R / f"glide_{tag}.txt"),
    ]
STEPS += [
    (E001, ["python", "pokeworld.py", "mgain"],
     E001 / "data" / "probe6_te_state.npz", None, None),
    (E003, TRAIN + ["VFX", "--dsuf", "6", "--psuf", "6"],
     R / "VFX_s0_l0.02_d6.json", None, None),
    (E003, ["python", "review_fixes.py", "mgain"], R / "mgain_probe.json",
     None, None),
    (E003, TRAIN + ["VFX", "--touchw", "1"], R / "VFX_s0_l0.02_tw1.json",
     None, None),
    (E003, TRAIN + ["VFX", "--touchw", "0.25"], R / "VFX_s0_l0.02_tw0.25.json",
     None, None),
    (E003, TRAIN + ["VFX", "--touchw", "0"], R / "VFX_s0_l0.02_tw0.json",
     None, None),
    (E003, TRAIN + ["VF"], R / "VF_s0_l0.02.json", None, None),
    (E003, ["python", "train.py", "--seed", "1", "--lam", "0.02", "--variant", "VF"],
     R / "VF_s1_l0.02.json", None, None),
]

failures = []
for wd, cmd, done, env, capture in STEPS:
    if done.exists():
        print(f"skip (done): {' '.join(cmd)}", flush=True)
        continue
    print(f"RUN: {' '.join(cmd)}  [{time.strftime('%H:%M:%S')}]", flush=True)
    if capture is not None:
        tmp = capture.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=str(wd), env=env, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        if rc == 0:
            tmp.replace(capture)
        else:
            print(tmp.read_text(encoding="utf-8")[-2000:], flush=True)
    else:
        rc = subprocess.run(cmd, cwd=str(wd), env=env).returncode
    if rc != 0:
        print(f"STEP FAILED rc={rc}: {' '.join(cmd)} - continuing", flush=True)
        failures.append(" ".join(cmd))
print("STAGE3B DONE", "failures:", failures or "none", flush=True)
