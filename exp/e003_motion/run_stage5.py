"""Stage-5 (review round 5): the coordinate-change prediction test (log-speed
channel), the minimal belief-maintaining variant (Gaussian NLL), pixel-derived
certificates, target-composition probes, and futforce bootstrap CIs.
Idempotent; continues past failures."""
import os, pathlib, subprocess, sys, time

E003 = pathlib.Path(__file__).parent
E010 = E003.parent / "e010_rh20t"
R = E003 / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
env_c1 = dict(os.environ, E010_DATA="data_cfg1")

TRAIN = ["python", "train.py", "--seed", "0", "--lam", "0.02", "--variant", "VFX"]
STEPS = [
    (E003, TRAIN + ["--flow", "--data2", "--logspeed"],
     R / "VFX_s0_l0.02_fl_g_ls.json", None, None),
    (E003, ["python", "analyze_gamma.py", "VFX_s0_l0.02_fl_g_ls"],
     R / "glide_VFX_s0_l0.02_fl_g_ls.txt", None,
     R / "glide_VFX_s0_l0.02_fl_g_ls.txt"),
    (E003, TRAIN + ["--nll"], R / "VFX_s0_l0.02_nll.json", None, None),
    (E003, ["python", "round5.py", "logspeed_cert"],
     R / "logspeed_cert.json", None, None),
    (E003, ["python", "round5.py", "pixel_certs"],
     R / "pixel_certs.json", None, None),
    (E003, ["python", "round5.py", "composition"],
     R / "composition_probe.json", None, None),
    (E010, ["python", "eval_futforce.py"],
     E010 / "results" / "futforce.json", env_c1, None),
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
print("STAGE5 DONE", "failures:", failures or "none", flush=True)
