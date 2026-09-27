"""Export the replay-array statistics used in Tables 3, 10, and 13.

Run after func_cf.py has written its JSON and NPZ outputs. The formulas match
the manuscript's gen_tabs.py: release slopes control the two other parameters,
distance, finger speed, projected velocities, and direction when available.
The exported JSON allows tables to be inspected without distributing checkpoints.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def partial_slope(y, x, nuisances):
    design = np.c_[np.ones(len(y)), x, nuisances]
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(weights[1])


def summarize(path):
    with np.load(path) as data:
        r = data["R"]
    params = r[:, :3]
    metrics = {"n_windows": len(r), "source": path.name}
    if path.stem.endswith("_release"):
        direction = r[:, 16:18] if r.shape[1] >= 18 else np.zeros((len(r), 0))
        nuisances = np.c_[r[:, [3, 4, 12, 13]], direction, params[:, [0, 2]]]
        observed = partial_slope(r[:, 6], params[:, 1], nuisances)
        predicted = partial_slope(r[:, 8], params[:, 1], nuisances)
        metrics["drag_release_slope_ratio"] = predicted / observed
    else:
        observed_effect = r[:, 6] - r[:, 14]
        predicted_effect = r[:, 8] - r[:, 15]
        metrics["mass_action_effect_relative_error"] = float(
            ((predicted_effect - observed_effect) ** 2).sum()
            / (observed_effect ** 2).sum()
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "exp/e003_motion/results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stems", nargs="+", help="Selected func_cf result stems, without extension")
    args = parser.parse_args()
    stems = args.stems or sorted(p.stem for p in args.results_dir.glob("func_cf_*.npz"))
    if not stems:
        parser.error("No func_cf_*.npz arrays found; run func_cf.py before exporting replay statistics.")
    metrics = {stem: summarize(args.results_dir / (stem + ".npz")) for stem in stems}
    output = args.output or args.results_dir / "intervention_metrics.json"
    output.write_text(json.dumps({"metrics": metrics}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(metrics)} replay summaries to {output}")


if __name__ == "__main__":
    main()
