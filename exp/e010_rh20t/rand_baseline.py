"""Untrained-encoder readout control for the cfg1 factorial: how much of a
fused-input model's force/position readout is INPUT PASSTHROUGH that ridge can
extract from a randomly initialized encoder, with no training at all? This
bounds the trivial component of readout metrics for fused-input variants
(review point: force/proprioception sit directly in the inputs)."""
import json, os, pathlib, sys
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import XJEPA
import train as TR

RESULTS = pathlib.Path(__file__).parent / "results"
device = "cuda"
out = {}
for variant in ("VFX", "V"):
    torch.manual_seed(0)
    model = XJEPA(variant).to(device).eval()
    stats = TR.whiten_stats()
    res = {"iid": TR.evaluate(model, stats, device),
           "ood": TR.evaluate(model, stats, device, "probe_tr", "ood")}
    out[variant] = res
    print(variant, json.dumps(res), flush=True)
suffix = {"data": "", "data_master": "_m", "data_cfg1": "_c1"}.get(
    os.environ.get("E010_DATA", "data"), "_x")
(RESULTS / f"RAND{suffix}.json").write_text(json.dumps(out, indent=1))
print("rand baseline done", flush=True)
