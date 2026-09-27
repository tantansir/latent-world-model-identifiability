"""Extract Figure 3 measurements from analyze_gamma/probe_targets output.

Each arm/seed needs one gl_fc_<arm>_s<seed>*.out forecast log and one
gl_rd_<arm>_s<seed>*.out probe log. Only measured numerical values and source
basenames are exported; machine paths and scheduler headers are omitted.
"""
import argparse
import json
from pathlib import Path
import re


def unique_log(directory, pattern):
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern}; found {len(paths)}")
    return paths[0]


def extract(directory):
    models = []
    for arm in ("ac", "vr_oi_kr_mr_ac_ih"):
        for seed in (0, 1, 2):
            forecast = unique_log(directory, f"gl_fc_{arm}_s{seed}*.out")
            readout = unique_log(directory, f"gl_rd_{arm}_s{seed}*.out")
            text = forecast.read_text(encoding="utf-8")
            block = text.split("implied factor f_hat")[1].split("bin-mean slope")[0]
            rows = re.findall(
                r"\[([\d.]+),([\d.]+)\)\s+([-\d.]+) ± ([\d.]+) \| ([\d.]+) \| [\d.]+\s+\((\d+)\)", block
            )
            if not rows:
                raise ValueError(f"No forecast bins in {forecast}")
            models.append({
                "tag": f"VFX_s{seed}_l0.02_{arm}", "seed": seed,
                "forecast_log": forecast.name, "readout_log": readout.name,
                "gamma_bins": [(float(r[0]) + float(r[1])) / 2 for r in rows],
                "forecast_factors": [float(r[2]) for r in rows],
                "true_factors": [float(r[4]) for r in rows],
                "bin_mean_slope": float(re.search(r"bin-mean slope of f_hat on f_true: ([-\d.]+)", text)[1]),
                "partial_slope": float(re.search(r"\+ ([-\d.]+)\*f_true", text)[1]),
                "partial_slope_se": float(re.search(r"f_true \(se ([\d.]+)\)", text)[1]),
                "glide_readout_r2": float(re.search(r"\n\s+gamma\s+h: ([-\d.]+)", readout.read_text(encoding="utf-8"))[1]),
            })
    return {"evaluation_seed": 778, "glide_fraction": 1.0, "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "exp/e003_motion/results/glide_forecasts.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(extract(args.logs_dir), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported Figure 3 measurements to {args.output}")


if __name__ == "__main__":
    main()
