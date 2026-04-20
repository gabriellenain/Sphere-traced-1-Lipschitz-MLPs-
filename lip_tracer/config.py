"""All hyper-parameters and paths for the 1-Lip sphere-tracing pipeline."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------ paths ----

SCENE         = Path("/Data/gabriel.lenain/nerfstudio/sdfstudio/data/sdfstudio/sdfstudio-demo-data/dtu-scan65")
BLENDER_SCENE = Path("/Data/gabriel.lenain/nerfstudio/data/blender/nerf_synthetic/lego")
OUT_DIR       = Path("/Data/gabriel.lenain/nerfstudio/outputs")


# ----------------------------------------------------------------- model ----

@dataclass
class ModelConfig:
    hidden:     int = 256   # network width (must be divisible by group_size)
    depth:      int = 12    # number of CPL layers
    group_size: int = 2     # 2 → MaxMin, >2 → GroupSort-N (Prach & Lampert 2022)


# --------------------------------------------------------- sphere tracing ----

@dataclass
class TraceConfig:
    iters:      int   = 64     # max sphere-tracing iterations
    eps:        float = 1e-3   # convergence threshold |f(x)| < eps → hit
    t_far:      float = 10.0   # ray cut-off distance
    eik_stride: int   = 8      # collect one eikonal sample every N iters


# ------------------------------------------------------- initialisation -----

@dataclass
class InitConfig:
    """Hyper-params for the sphere warm-start (fit_sphere_init)."""
    steps:  int         = 2000   # gradient steps
    batch:  int         = 8192   # points per step
    lr:     float       = 1e-3
    radius: float | None = None  # None → auto-detect from COLMAP p80


# ----------------------------------------------------------------- train ----

@dataclass
class TrainConfig:
    # optimiser
    steps: int   = 500
    batch: int   = 1024
    lr:    float = 5e-4
    down:  int   = 2      # pixel-grid downsample factor for deterministic rays

    # visibility filter
    n_alt:      int   = 4     # nearest-neighbour cameras per ray
    cos_thresh: float = 0.1   # |cos(n, view)| threshold for valid reprojections

    # loss weights
    w_photo:     float = 1.0
    w_ncc:       float = 0.0
    ncc_patch:   int   = 7
    w_cam_free:  float = 0.0
    w_sfm:       float = 0.0
    w_free:      float = 0.0   # free-space along SFM camera→point rays
    w_mvs:       float = 0.0
    w_behind_hit: float = 0.0
    behind_eps:  float = 1e-2
    w_sil:       float = 0.1
    sil_s:       float = 20.0  # sigmoid sharpness for silhouette
    sil_k:       int   = 16    # stratified samples per ray for silhouette
    sil_t_near:  float = 0.5
    sil_t_far:   float = 6.0
    w_eikonal:   float = 0.0
    n_free:      int   = 32    # free-space samples per ray pair

    use_blender: bool = False


# ------------------------------------------------------------------ eval ----

@dataclass
class EvalConfig:
    """Params for marching cubes, chamfer, and rendering evaluations."""
    mc_res:        int   = 256   # marching-cubes voxel resolution
    bound_dtu:     float = 2.0   # SDF grid half-extent for DTU scenes
    bound_blender: float = 1.5   # SDF grid half-extent for Blender scenes
    render_down:   int   = 2     # downsample factor for normal/render overlays
    nn_chunk:      int   = 4096  # chunk size for chamfer nearest-neighbour search
    n_views:       int   = 6     # number of reference views in the visualize grid

    def bound(self, use_blender: bool) -> float:
        return self.bound_blender if use_blender else self.bound_dtu


# --------------------------------------------------------------- master -----

@dataclass
class Config:
    """Single entry point for all pipeline hyper-parameters."""
    model: ModelConfig = field(default_factory=ModelConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    init:  InitConfig  = field(default_factory=InitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval:  EvalConfig  = field(default_factory=EvalConfig)
    scene:   Path = SCENE
    out_dir: Path = OUT_DIR

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["scene"]   = str(self.scene)
        d["out_dir"] = str(self.out_dir)
        return d
