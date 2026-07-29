"""e010 preprocessing: RH20T cfg7 pilot corpus -> compact training tensors.

Input : data_rh20t/cfg7 (LeRobot v3: one parquet, 34 AV1 video files, meta)
Output: exp/e010_rh20t/data/
  images.npy   uint8 [total_frames, 96, 96]   (episode-sorted global order)
  sensors.npz  state[.,15] touch[.,12] action[.,8] episode_index[.]
  index.npz    per-episode: start, length, task_id, rating + split assignments
Splits: episode-level, stratified by task; 6 held-out tasks -> ood split.
"""
import json, sys, time, pathlib
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else \
    pathlib.Path(r"C:\Users\Kaizh\Desktop\physical representation\data_rh20t\cfg7")
CAM = sys.argv[1] if len(sys.argv) > 1 else "observation.images.cam_037522061512"
OUT = pathlib.Path(__file__).parent / (sys.argv[2] if len(sys.argv) > 2 else "data")
OUT.mkdir(parents=True, exist_ok=True)
RES = 96

print("== loading parquet", flush=True)
dp = sorted((SRC / "data").rglob("*.parquet"))
print(f"   {len(dp)} parquet files", flush=True)
df = pd.concat([pd.read_parquet(p) for p in dp], ignore_index=True)
df = df.sort_values("index").reset_index(drop=True)
n_total = len(df)
ep_meta = json.loads((SRC / "meta" / "rh20t_episodes.json").read_text(encoding="utf-8"))

state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
touch = np.concatenate([np.stack(df["observation.force"].to_numpy()),
                        np.stack(df["observation.torque"].to_numpy()),
                        np.stack(df["observation.robot_ft"].to_numpy())], 1).astype(np.float32)
action = np.stack(df["action"].to_numpy()).astype(np.float32)
ep_idx = df["episode_index"].to_numpy()
task_idx = df["task_index"].to_numpy()

eps, starts, lengths = np.unique(ep_idx, return_index=True, return_counts=True)
assert (eps == np.arange(len(eps))).all()
tasks = np.array([task_idx[s] for s in starts])
ratings = np.array([ep_meta[str(int(e))]["rating"] for e in eps])
for e in (0, len(eps) // 2, len(eps) - 1):
    assert lengths[e] == ep_meta[str(e)]["n_frames"], f"length mismatch ep {e}"
print(f"episodes {len(eps)}, frames {n_total}, tasks {len(np.unique(tasks))}", flush=True)

np.savez(OUT / "sensors.npz", state=state, touch=touch, action=action,
         episode_index=ep_idx.astype(np.int32))

# ---- decode videos sequentially into the global-order image memmap ----
print("== decoding videos", flush=True)
import imageio.v3 as iio
prog_path = OUT / "decode_progress.json"
done_files = {}
if prog_path.exists() and (OUT / "images.npy").exists():
    done_files = {r["file"]: r["n"] for r in
                  json.loads(prog_path.read_text())}
    img_mm = np.load(OUT / "images.npy", mmap_mode="r+")
    print(f"   resuming: {len(done_files)} files already decoded", flush=True)
else:
    img_mm = np.lib.format.open_memmap(OUT / "images.npy", mode="w+",
                                       dtype=np.uint8, shape=(n_total, RES, RES))
device = "cuda" if torch.cuda.is_available() else "cpu"
vids = sorted((SRC / "videos" / CAM).rglob("*.mp4"))
print(f"camera {CAM}: {len(vids)} video files", flush=True)
g = 0
t0 = time.time()
buf = []


def flush(buf, g0):
    x = torch.from_numpy(np.stack(buf)).to(device).float()          # [n,H,W,3]
    x = x.mean(-1, keepdim=True).permute(0, 3, 1, 2)                # gray [n,1,H,W]
    x = torch.nn.functional.interpolate(x, size=(RES, RES), mode="area")
    img_mm[g0:g0 + len(buf)] = x[:, 0].clamp(0, 255).byte().cpu().numpy()


prog = [{"file": f, "n": n} for f, n in done_files.items()]
for vi, vp in enumerate(vids):
    if vp.name in done_files:
        g += done_files[vp.name]
        continue
    n_file = 0
    for frame in iio.imiter(vp, plugin="pyav"):
        if g + len(buf) >= n_total:
            break
        buf.append(frame)
        n_file += 1
        if len(buf) == 512:
            flush(buf, g)
            g += 512
            buf = []
    if buf:
        flush(buf, g)
        g += len(buf)
        buf = []
    prog.append({"file": vp.name, "n": n_file})
    prog_path.write_text(json.dumps(prog))
    el = time.time() - t0
    print(f"[{vi+1}/{len(vids)}] {vp.name}: +{n_file} frames, global {g}/{n_total} "
          f"({(g - sum(done_files.values()))/max(el,1):.0f} fps)", flush=True)
    if g >= n_total:
        break
img_mm.flush()
print(f"decoded {g} frames vs expected {n_total}", flush=True)
assert abs(g - n_total) <= 34, "frame count mismatch beyond per-file tolerance"

# ---- splits: 6 held-out tasks = ood; rest stratified by task ----
rng = np.random.default_rng(0)
uniq_tasks = np.unique(tasks)
ood_tasks = rng.choice(uniq_tasks, max(6, len(uniq_tasks) // 10), replace=False)
is_ood = np.isin(tasks, ood_tasks)
rest = np.where(~is_ood)[0]
rng.shuffle(rest)
n = len(rest)
split = {"train": rest[: int(n * 0.78)], "val": rest[int(n * 0.78): int(n * 0.83)],
         "probe_tr": rest[int(n * 0.83): int(n * 0.94)],
         "probe_te": rest[int(n * 0.94):], "ood": np.where(is_ood)[0]}
np.savez(OUT / "index.npz", start=starts.astype(np.int64),
         length=lengths.astype(np.int64), task=tasks.astype(np.int32),
         rating=ratings.astype(np.int32), ood_tasks=ood_tasks.astype(np.int32),
         **{f"split_{k}": v.astype(np.int32) for k, v in split.items()})
fmag = np.linalg.norm(touch[:, :3], axis=1)
print("splits:", {k: len(v) for k, v in split.items()}, flush=True)
print(f"|force| p50/p90/p99: {np.percentile(fmag, [50, 90, 99]).round(2)}", flush=True)
print("PREP DONE", flush=True)
