"""All hyper-parameters and paths for the 1-Lip sphere-tracing pipeline."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------ paths ----

SCENE         = Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65")
BLENDER_SCENE = Path("/scratch/_projets_/willow/1-lip-tracer/data/nerf_synthetic/lego")
OUT_DIR       = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs")


# ----------------------------------------------------------------- model ----

@dataclass
class ModelConfig:
    hidden:     int =  256# network width (must be divisible by group_size)
    depth:      int = 8   # number of CPL layers
    group_size: int = 2     # 2 → MaxMin, >2 → GroupSort-N (ignored when activation="nact")
    activation: str = "groupsort"  # "groupsort" | "nact"
    input_encoding: str = "neus"  # "identity" | "neus"
    multires: int = 2 # NeuS-style positional encoding frequencies when input_encoding="neus"
    architecture: str = "cpl"  # "cpl" | "mlp"


# --------------------------------------------------------- sphere tracing ----

@dataclass
class TraceConfig:
    iters:        int   = 24     # max sphere-tracing iterations
    eps:          float = 1e-3   # convergence threshold |f(x)| < eps → hit
    t_far:        float = 10.0   # ray cut-off distance
    eik_stride:   int   = 8      # collect one eikonal sample every N iters
    newton_steps: int   = 0      # Newton refinement iters on hit rays after tracing


# ------------------------------------------------------- initialisation -----

@dataclass
class InitConfig:
    """Hyper-params for the warm-start (sphere or visual hull)."""
    init:     str          = "hull"  # "sphere" | "hull"
    steps:    int          = 20000    # gradient steps
    batch:    int          = 4096      # hull: small batch keeps GPU footprint <200 MB
    lr:       float        = 1e-1
    radius:   float | None = None      # sphere only: None → auto-detect from COLMAP p80
    hull_res: int          = 256      # hull only: voxel carving resolution
    hull_mode: str         = "grid"   # "grid" | "octree"
    octree_min_depth: int  = 5        # octree only: stop early for fully-inside cells at/after this depth
    w_depth_surface: float = 0.0  # blender hull-init only: weight on GT depth surface samples


# ------------------------------------------------------- MVSDF schedule ------

@dataclass
class MvsdfScheduleConfig:
    """Three-phase weight schedule from the MVSDF paper.

    Fractions are of total training steps:
      phase 1: [0,          phase1_end)  — depth carving only
      phase 2: [phase1_end, phase2_end)  — appearance added
      phase 3: [phase2_end, 1.0]         — fine-tune at low weights

    Default boundaries: 1/6 | 1/3 | 1/2  (paper values).
    """
    enabled:      bool  = False

    # ---- phase boundaries (fraction of total steps) ----
    phase1_end:   float = 1 / 6   # end of phase 1
    phase2_end:   float = 1 / 2   # end of phase 2  (= 1/6 + 1/3)

    # wR (w_photo) and wE (w_eikonal) are NOT scheduled — set them in TrainConfig directly.
    # wD and wF only:
    # ---- phase 1: depth carving only ----
    p1_w_msdf:    float = 1.0
    p1_w_feat:    float = 0.0

    # ---- phase 2: add feature appearance ----
    p2_w_msdf:    float = 0.1
    p2_w_feat:    float = 0.1

    # ---- phase 3: fine-tune ----
    p3_w_msdf:    float = 0.01
    p3_w_feat:    float = 0.01

    def weights(self, progress: float) -> tuple[float, float]:
        """Return (w_msdf, w_feat) for the given training progress ∈ [0, 1]."""
        if progress < self.phase1_end:
            return self.p1_w_msdf, self.p1_w_feat
        elif progress < self.phase2_end:
            return self.p2_w_msdf, self.p2_w_feat
        else:
            return self.p3_w_msdf, self.p3_w_feat

    def phase(self, progress: float) -> int:
        if progress < self.phase1_end:   return 1
        elif progress < self.phase2_end: return 2
        else:                            return 3


# ----------------------------------------------------------------- train ----

@dataclass
class TrainConfig:
    # optimiser
    steps: int   = 50000
    batch: int   = 4096
    lr:    float = 1e-3
    down:  int   = 1      # pixel-grid downsample factor for deterministic rays

    # visibility filter
    n_alt:      int   = 6      # nearest-neighbour cameras per ray
    cos_thresh: float = 0.1   # |cos(n, view)| threshold for valid reprojections

    # loss weights
    w_photo:     float = 1.0
    w_feature:   float = 0.0  # cosine distance on precomputed feature maps
    feature_maps: Path | None = None
    w_ncc:       float = 0.0
    ncc_patch:   int   = 7
    ncc_half_pix: float = 3.0   # PMVS patch half-width in reference-view pixels
    sample_mode:  str   = "bilinear"  # "bilinear" | "gaussian"
    gaussian_sigma:  float = 0.5
    gaussian_radius: int   = 1
    w_cam_free:  float = 0.0
    w_sfm:       float = 0.0
    w_free:      float = 0.0   # free-space along SFM camera→point rays
    w_surf:      float = 0.0
    w_mvs:       float = 0.0
    mvs_depth_dir: Path | None = None   # if set, load MASt3R depths from this dir (IDR scans)
    mvsdf_schedule: MvsdfScheduleConfig = field(default_factory=MvsdfScheduleConfig)
    w_mvs_sdf:   float = 0.0  # MVSDF carving loss (volumetric SDF from depth consensus)
    n_mvs_sdf:   int   = 4096
    mvs_sdf_out_thresh: float = 0.5   # outside vote threshold (MVSDF out_thresh_perc)
    mvs_sdf_trunc: float = 1.25
    mvs_sdf_smooth: float = 0.0
    mvs_sdf_far_thresh: float = 1.25
    mvs_sdf_far_att: float = 1.0
    mvs_sdf_near_thresh: float = 0.1   # 5% of bbox size 2 → points closer than this get lower weight
    mvs_sdf_near_att: float = 0.1      # weight for near-surface uncertain points
    w_behind_hit: float = 0.0
    behind_eps:  float = 1e-2
    w_sil:          float = 0.0
    sil_s:          float = 20.0  # sigmoid sharpness for silhouette
    sil_fg_offset:  float = 0.1   # shift sdf_min for fg rays so σ(−α·(sdf_min−offset))→1 on hits
    sil_k:          int   = 16    # stratified samples per ray for silhouette
    sil_t_near:     float = 0.5
    sil_t_far:      float = 6.0
    w_eikonal:   float = 0.1    # wE in MVSDF paper (always fixed)
    n_eik_vol:   int   = 512    # random volume points for eikonal (separate from ray samples)
    w_normal:    float = 0.0  # GT normal supervision (requires DTU normal maps)
    n_free:      int   = 32    # free-space samples per ray pair
    w_ray_free:  float = 0.0  # free-space along training rays before the hit
    n_ray_free:  int   = 8     # samples per hit ray

    use_blender: bool = False
    single_view: int = -1  # >=0: restrict training to this one camera (overfit diagnostic)


# ------------------------------------------------------------------ eval ----

@dataclass
class EvalConfig:
    """Params for marching cubes, chamfer, and rendering evaluations."""
    mc_res:        int   = 256   # marching-cubes voxel resolution
    bound_dtu:     float = 1.5   # SDF grid half-extent for DTU scenes (cameras at r~2.7 must stay outside)
    bound_blender: float = 1.5   # SDF grid half-extent for Blender scenes
    render_down:   int   = 2     # downsample factor for normal/render overlays
    nn_chunk:      int   = 4096  # chunk size for chamfer nearest-neighbour search
    n_views:       int   = 6     # number of reference views in the visualize grid
    dtu_eval_dir:  Path | None = None   # path to DTU SampleSet/ + ObsMask/ for official Chamfer
    dtu_chamfer_freq:  int   = 500   # log official DTU Chamfer every N train steps (0 = off)
    dtu_chamfer_res:   int   = 256   # MC resolution for in-training DTU Chamfer (fast)
    dtu_chamfer_bound: float = 0.8   # MC bound for chamfer only — tighter than SDF grid to maximise voxel precision

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
        if d["train"]["feature_maps"] is not None:
            d["train"]["feature_maps"] = str(d["train"]["feature_maps"])
        if d["train"]["mvs_depth_dir"] is not None:
            d["train"]["mvs_depth_dir"] = str(d["train"]["mvs_depth_dir"])
        if d["eval"]["dtu_eval_dir"] is not None:
            d["eval"]["dtu_eval_dir"] = str(d["eval"]["dtu_eval_dir"])
        return d
