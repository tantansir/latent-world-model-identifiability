# What Can Latent World Models Know?

**Physical Information in Multimodal Predictive Representations**

**Kaizhen Tan<sup>1,4,&#42;</sup>, Sizhe Xu<sup>1,4,&#42;</sup>, Xin Xu<sup>2</sup>,
Siru Tao<sup>2</sup>, Yixiao Li<sup>2</sup>, Hanzhe Hong<sup>2</sup>,
Yang Feng<sup>3</sup>, Heqing Du<sup>3</sup>, Zhaonan Wang<sup>4</sup>**

<sup>1</sup>New York University · <sup>2</sup>Carnegie Mellon University ·
<sup>3</sup>Columbia University · <sup>4</sup>NYU Shanghai<br>
<sup>&#42;</sup>Equal contribution.

[![Project Page](https://img.shields.io/badge/Project-Page-2b5fa8?style=flat-square)](https://tantansir.github.io/latent-world-model-identifiability/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-2b5fa8?style=flat-square)](docs/assets/paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2607.27017-b31b1b?style=flat-square)](https://arxiv.org/abs/2607.27017)
[![License](https://img.shields.io/badge/License-MIT-6b7280?style=flat-square)](LICENSE)

We study which physical quantities are accessible in learned latent states,
how this depends on training, and how those quantities relate to future
predictions. Experiments use PokeWorld, a simulated pushing environment with
hidden mass, drag, and contact stiffness, and real multimodal robot data from RH20T.

The [paper PDF](docs/assets/paper.pdf) and
[project page](https://tantansir.github.io/latent-world-model-identifiability/)
reflect the current manuscript. The linked arXiv record currently contains an
earlier revision with the previous title and author list.

![Action-conditioned multimodal world model, prediction targets, and frozen latent probes](docs/assets/img/fig_overview.png)

## Measurements and Findings

The study separates three measurements:

| Measurement | What is evaluated |
| --- | --- |
| Recovery from observations | A supervised two-layer GRU estimates physical parameters from observation and action windows. Its held-out performance provides an empirical recovery reference. |
| Latent readout | Probes estimate physical quantities from a frozen model's representations. Weak linear readout alone does not establish that all parameter information has been discarded. |
| Forecast dependence | Tests with prescribed actions measure how forecasts vary with a physical parameter and compare that response with the simulator. |

The recovery reference is the performance of a particular estimator, not an
information-theoretic upper bound.

- **Prediction targets affect parameter readout.** On contact windows, stiffness
  readout is −0.01 for VF, which receives touch but predicts only latent targets,
  and 0.50 for VFX, which also predicts touch (Table 1, mean R² over three seeds).
- **Longer horizons improve motion readout.** Position and velocity readout improve
  across all three observation and regularization settings in Table 2. For
  frame-plus-difference inputs at λ = 0.02, position R² increases from 0.67 to
  0.95 and velocity R² from 0.40 to 0.78.
- **Readout and forecast dependence can diverge.** On free glides, VFX forecasts
  shorten with increasing drag, with a slope ratio of 1.04 relative to the true
  dynamics. A coordinate-target model reads drag almost perfectly but forecasts
  displacement that increases with drag (Figure 3).
- **RH20T shows the same input–target dependence.** On held-out Flexiv tasks,
  VFX reaches current-force R² of 0.88 and future-force R² of 0.47. An untrained
  fused-input encoder already has current-force R² of 0.91; predicting force
  preserves this readout, while training VF reduces it below zero (Table 4).

Here, V uses visual input; F adds proprioception and touch or force to the input;
X adds prediction targets for those sensor streams. All variants predict latent
targets. The paper also separates proprioception-only and touch-only target
variants.

![PokeWorld matched physical-parameter pairs and empirical recovery references](docs/assets/img/fig_pokeworld.png)

PokeWorld objects look identical while their physical parameters vary. The
[project page](https://tantansir.github.io/latent-world-model-identifiability/)
includes paired simulations, glide forecasts, and force traces from real robots.

## Code Release Scope

This repository contains the PokeWorld simulator, earlier model and training
implementations, saved experiment results, and the project-page assets. The
commands below run these released implementations.

The current manuscript also uses updates that are **not yet included in this
code release**: corrected action alignment, matched recovery references and
horizon controls, the intervention and frozen-head diagnostics in Table 3 and
Figure 4, and updated RH20T preprocessing and evaluation. The released scripts
therefore do not
reproduce every table in the current PDF. JSON results in `exp/` include earlier
experiments and should be interpreted with their model settings and data splits.

Raw and preprocessed datasets and trained checkpoints are not included. PokeWorld
data can be generated locally; RH20T preprocessing requires the converted dataset
layout described below.

## Repository Layout

| Path | Contents |
| --- | --- |
| [`exp/e001_xmodal_jepa/`](exp/e001_xmodal_jepa/) | PokeWorld simulator, single-step models, training, and parameter probes |
| [`exp/e002_multiscale/`](exp/e002_multiscale/) | Multi-horizon prediction with Δ ∈ {1, 4, 16} |
| [`exp/e003_motion/`](exp/e003_motion/) | Frame-difference inputs, objective controls, parameter probes, and CEM planning |
| [`exp/e005_patchtok/`](exp/e005_patchtok/) | Patch-token architecture experiments |
| [`exp/e010_rh20t/`](exp/e010_rh20t/) | RH20T preprocessing, training, force evaluation, and cross-embodiment experiments |
| [`docs/`](docs/) | Project website, current paper PDF, figures, and videos |

## Running the Released Code

### Environment

Clone the repository and run the commands below from its root:

```bash
git clone https://github.com/tantansir/latent-world-model-identifiability.git
cd latent-world-model-identifiability
```

The training scripts use CUDA and bfloat16 autocasting. Install a CUDA-enabled
[PyTorch build](https://pytorch.org/get-started/locally/) compatible with your GPU.
Core training and evaluation use NumPy and PyTorch.

```bash
python -m pip install numpy
```

RH20T preprocessing additionally needs pandas, a Parquet engine, ImageIO, and PyAV:

```bash
python -m pip install pandas pyarrow imageio av
```

The release does not include a pinned environment specification.

### PokeWorld Data and Training

Generate the default 64-step episodes: 4,000 training, 400 validation, 1,600
probe-training, and 800 probe-test episodes. This writes image arrays and state
files under `exp/e001_xmodal_jepa/data/`.

```bash
python exp/e001_xmodal_jepa/pokeworld.py

# Single-step model with visual input and proprioception/touch targets.
python exp/e001_xmodal_jepa/train.py --variant VX --seed 0

# Multi-horizon model with frame-difference and multimodal inputs.
python exp/e003_motion/train.py --variant VFX --seed 0 --lam 0.02

# Re-evaluate the checkpoint created by the preceding command.
python exp/e003_motion/train.py --variant VFX --seed 0 --lam 0.02 --eval_only
```

Training writes checkpoints (`.pt`) and evaluation summaries (`.json`) into each
experiment's `results/` directory. Reusing a variant/seed/configuration tag
overwrites its saved output. The older `run_e001.py` orchestrator skips variants
whose result JSON already exists, including the results committed here; use the
explicit training commands above to run a new experiment.

### RH20T Preprocessing and Training

The study uses [RH20T](https://rh20t.github.io/) configurations 1 (Flexiv) and 7
(KUKA). The released `prep.py` expects a **converted LeRobot-style dataset**, with
`data/**/*.parquet`, `videos/<camera>/**/*.mp4`, and
`meta/rh20t_episodes.json`. The conversion from the original RH20T files is not
included in this repository.

For Flexiv, replace `/path/to/converted/cfg1` with that dataset directory:

```bash
python exp/e010_rh20t/prep.py observation.images.cam_035622060973 data_cfg1 /path/to/converted/cfg1
```

This creates `images.npy`, `sensors.npz`, and `index.npz` in
`exp/e010_rh20t/data_cfg1/`. Preprocessing must finish before training or evaluation,
because the training module reads these files when imported.

```bash
# Bash
E010_DATA=data_cfg1 python exp/e010_rh20t/train.py --variant VFX --seed 0
```

```powershell
# PowerShell
$env:E010_DATA = "data_cfg1"
python exp/e010_rh20t/train.py --variant VFX --seed 0
```

For KUKA, use camera `observation.images.cam_037522061512`, output directory
`data`, and set `E010_DATA=data`. The output JSON separates within-task (`iid`)
and held-out-task (`ood`) evaluations.

### Planning and Videos

The supplementary planning demonstration uses a separate VFX checkpoint with
λ = 0.005. Install the visualization dependencies first:

```bash
python -m pip install matplotlib imageio imageio-ffmpeg
python exp/e003_motion/train.py --variant VFX --seed 0 --lam 0.005
python exp/e003_motion/plan_eval.py VFX_s0_l0.005 viz
```

Existing videos are available on the project page. To regenerate the PokeWorld
paired simulations:

```bash
python docs/assets/make_pokeworld_videos.py
```

Planning and real-robot video builders under `docs/assets/` additionally require
the corresponding trajectories, prepared data, and checkpoints. Some older
orchestration and analysis scripts retain machine-specific paths; inspect those
paths before running them.

## Citation

The following entry describes the current manuscript linked above:

```bibtex
@misc{tan2026latentworldmodels,
  title         = {What Can Latent World Models Know? Physical Information
                   in Multimodal Predictive Representations},
  author        = {Tan, Kaizhen and Xu, Sizhe and Xu, Xin and Tao, Siru and
                   Li, Yixiao and Hong, Hanzhe and Feng, Yang and Du, Heqing
                   and Wang, Zhaonan},
  year          = {2026},
  url           = {https://tantansir.github.io/latent-world-model-identifiability/}
}
```

## License

Code is distributed under the [MIT license](LICENSE). RH20T data remains subject
to its own dataset terms.
