# Sphere-traced 1-Lipschitz MLPs

Self-contained implementation of multi-view photoconsistency training with 1-Lipschitz sphere tracing.

Minimises

$$E(\theta) = \sum_{o,u} \sum_{o' \in V(x_\theta(o,u)) ,\ o' \neq o} \|I_{o'}(\pi_{o'}(x_\theta)) - I_o(\pi_o(x_\theta))\|_1$$

where $x_\theta = o + t_\theta(o,u)\u$ is the surface hit found by differentiable sphere tracing through a **1-Lipschitz CPL network** $f_\theta$.

## Structure

The core package:

```
lip_tracer/
├── config.py          # all hyper-parameters (ModelConfig, TraceConfig, TrainConfig, …)
├── model.py           # FTheta: CPL stack with MaxMin / GroupSort activation
├── sphere_tracing.py  # differentiable trace_unrolled + fast trace_nograd
├── data.py            # data loaders (DTU/SDFStudio, NeRF Blender)
├── loss.py            # photo_loss, silhouette, eikonal, free-space, MVS depth
├── train.py           # fit_sphere_init, train(), __main__ entry point
└── visualize.py       # marching-cubes PNG, Viser viewer, Chamfer / geom metrics
```

Supporting scripts are grouped by role at the repo root:

```
experiments/   # .slurm cluster launchers (train / fit / render / eval / carve …)
analysis/      # diagnostics, plots, metrics, comparisons, paper-figure renders
tools/         # reusable helpers: viz, render, precompute, hull / mesh utilities
figures/       # diagnostic PNGs (gitignored)
archive/       # stale one-off scripts kept for reference
baselines/     # NeuS / Geo-Neus / ProbeSDF submodules
```

Conventions: `.slurm` scripts `cd` to the repo root, so submit them by path
(`sbatch experiments/foo.slurm`) and any paths inside resolve from the root.
Import-coupled `.py` modules are kept in the same folder (a script's own directory
is on `sys.path`).

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# DTU skull (default)
python -m lip_tracer.train --steps 500 --w-photo 1.0 --w-sil 0.1

# NeRF Blender lego
python -m lip_tracer.train --dataset lego --steps 500 --w-photo 1.0 --w-sil 0.2

# eval only from a checkpoint
python -m lip_tracer.train --no-train --pt /path/to/checkpoint.pt

# interactive Viser viewer
python -m lip_tracer.train --no-train --pt /path/to/checkpoint.pt --viewer

# full-resolution sphere-traced reference + novel views
python -m lip_tracer.render_views --pt /path/to/checkpoint.pt --dataset lego
```

## Key hyper-parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 500 | training steps |
| `--hidden` | 256 | CPL network width |
| `--depth` | 12 | number of CPL layers |
| `--w-photo` | 1.0 | multi-view photo loss weight |
| `--w-sil` | 0.1 | silhouette BCE weight |
| `--w-eikonal` | 0.0 | eikonal regularisation weight |
| `--w-mvs` | 0.0 | MVS depth prior weight |
| `--dataset` | `skull` | `skull` (DTU) or `lego` (Blender) |

Set scene paths in `lip_tracer/config.py` (`SCENE`, `BLENDER_SCENE`, `OUT_DIR`).
