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
    depth:      int = 8  # number of CPL layers
    group_size: int = 2     # 2 → MaxMin, >2 → GroupSort-N (ignored when activation="nact")
    activation: str = "groupsort"  # "groupsort" | "nact"
    input_encoding: str = "pe"  # "identity" | "pe"
    multires: int = 6  # PE frequencies when input_encoding="pe"
    architecture: str = "cpl"  # "cpl" | "neus" | "mlp"
    lipschitz_mode: str = "none"  # "none" | "uniform" | "per_band" — rescale PE so γ is 1-Lipschitz


# --------------------------------------------------------- sphere tracing ----

@dataclass
class TraceConfig:
    iters:          int   =  64  # max sphere-tracing iterations
    eps:            float = 1e-3   # convergence threshold |f(x)| < eps → hit
    t_far:          float = 5.0   # ray cut-off distance (fallback when bsphere_radius=0)
    eik_stride:     int   = 8      # collect one eikonal sample every N iters
    newton_steps:   int   = 2      # Newton refinement iters on hit rays after tracing
    # --- occlusion-trace overrides (photo loss visibility test) ---
    # The occlusion trace produces a thresholded visibility boolean, so it does
    # not need primary-trace precision. Sentinels (<0) fall back to the primary
    # iters/newton_steps/eps above, so defaults reproduce the old behaviour.
    occ_iters:        int   = -1   # max iters for the occ trace (-1 → use `iters`)
    occ_newton_steps: int   = -1   # Newton steps for the occ trace (-1 → use `newton_steps`; 0 = none)
    occ_eps:          float = -1.0 # hit threshold for the occ trace (<0 → use `eps`)
    grad_mode:      str   = "idr"  # "backprop" | "idr"
    bsphere_radius: float = 0.0    # >0: use per-ray exit depth of this bounding sphere as t_far;
                                   # rays that reach the sphere exit are marked hit_bg=True so the
                                   # photo loss applies there (photometric inconsistency → gradient
                                   # pulls surface inward, eliminating fg-miss holes)
    sdf_min_beta:   float = 200.0   # 0: hard min over trace iters for sdf_min (gradient through argmin only);
                                   # >0: soft-min via -1/β·logsumexp(-β·sdf_k) — gradient flows through every
                                   # iteration weighted by proximity to the minimum. β≈50 ≈ hard min but
                                   # smooth, so mask_loss_min_sdf gradient pulls multiple trace points toward
                                   # the surface, much better signal for thin missing pieces.


# ------------------------------------------------------- initialisation -----

@dataclass
class InitConfig:
    """Hyper-params for the warm-start (sphere or visual hull)."""
    init:     str          = "hull"  # "sphere" | "hull"
    steps:    int          = 40000    # gradient steps
    batch:    int          = 4096      # hull: small batch keeps GPU footprint <200 MB
    lr:       float        = 1e-1
    radius:   float | None = None      # sphere only: None → auto-detect from COLMAP p80
    hull_res: int          = 256      # hull only: voxel carving resolution
    hull_sfm_roi: bool      = False    # hull only: crop unconstrained outer volume to padded SFM AABB
    hull_min_views: int     = 0        # hull only: require projection inside this many views
    hull_border_aware: bool = False    # hull only: carve off-frame voxels through clear image edges
                                       # (removes silhouette bloat when the object spills past the frame;
                                       # switches cleanup to largest-component). See visual_hull.carve.
    w_sfm_free: float       = 0.0      # hull only: enforce f>0 along camera→COLMAP-point sight-lines
    sfm_free_eps: float     = 0.02     # hull only: stop the free-space ray this far before the point
    w_depth_surface: float = 0.0  # blender hull-init only: weight on GT depth surface samples
    init_mesh: str | None  = None  # hull-init source: silhouette carving (default) or
                                   # voxelisation of this PLY mesh (e.g. COLMAP poisson.ply,
                                   # in NSVF-COLMAP / un-normalised frame).
    init_sdf_grid: str | None = None  # hull-init source: precomputed signed-distance
                                      # grid on [-bound,bound]^3; inside is sdf < 0.
    # --- "points" init: reconstruct a surface directly from the COLMAP sparse
    # cloud (sparse_sfm_points.txt, already in the normalized training frame).
    # No masks, no dense MVS — kNN-PCA normals oriented toward the cameras, a
    # screened-Poisson surface, then SDF regression (see fit_points_init).
    points_poisson_depth: int   = 9     # octree depth for screened Poisson (8=coarse, 9=fine)
    points_trim_quantile: float = 0.02  # drop Poisson vertices below this density quantile
                                        # (removes balloon extrapolation in unseen regions)
    points_normal_knn:    int   = 16    # kNN for PCA normal estimation + orientation MST


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


# ---------------------------------------------------- bundle adjustment -----

@dataclass
class BundleAdjustConfig:
    """Block-coordinate photometric bundle adjustment.

    Refines the per-camera extrinsics φ jointly with the SDF θ by alternating
    minimisation of the photometric objective E(θ, φ), starting from a converged
    checkpoint. See bundle_adjustment.py (run_bundle_adjustment). Intrinsics stay
    fixed by default (DTU / MVMannequin calibration is trusted); set opt_intrinsics
    to refine the per-camera K (fx, fy, cx, cy) jointly in the φ-block."""
    enabled:      bool  = False
    lock_first:   bool  = True     # fix camera 0 (gauge lock — prevents global drift)

    # --- block coordinate descent (block-diagonal alternation) ---------------
    # One BA "cycle" runs a φ-block (refine extrinsics, SDF frozen) followed by a
    # θ-block (refine SDF, extrinsics frozen) — or the reverse when phi_first is
    # False. The φ-block minimises the bare photometric E; the θ-block minimises
    # E plus the same SDF regularisers the model was trained with (eikonal /
    # silhouette / behind-hit), so f stays a valid 1-Lipschitz SDF.
    cycles:       int   = 20       # number of (φ-block, θ-block) alternations
    block_phi:    int   = 200      # optimiser steps per φ-block (θ frozen)
    block_theta:  int   = 200      # optimiser steps per θ-block (φ frozen)
    phi_first:    bool  = True     # start each cycle with the φ-block
    lr:           float = 1e-5     # φ (extrinsics) learning rate
    lr_theta:     float = 1e-5     # θ (SDF) learning rate during BA — small, refinement only
    opt_intrinsics: bool  = False  # also refine per-camera intrinsics K (fx,fy,cx,cy) in the φ-block
    lr_intrinsics:  float = 1e-4   # intrinsics learning rate (dimensionless delta; see CameraParams)
    batch:        int   = 0        # rays per BA step; 0 → reuse TrainConfig.batch
    ckpt_every:   int   = 1        # save a BA checkpoint every N cycles
    log_every:    int   = 20       # stdout per-step log cadence (CSV logs every step)
    w_eikonal:    float = 0.0      # >0: eikonal weight for the θ-block ONLY (overrides
                                   # TrainConfig.w_eikonal during BA). Keeps f a valid
                                   # SDF (|∇f|≈1) when refining θ jointly with poses;
                                   # required for a sound joint BA when the run trained
                                   # with w_eikonal=0 (else θ overfits bare photo E).


# ----------------------------------------------------------------- train ----

@dataclass
class TrainConfig:
    # optimiser
    steps: int   = 50000
    batch: int   = 16384
    lr:    float = 1e-3
    down:  int   = 1      # pixel-grid downsample factor for deterministic rays

    profile: bool = False  # one-shot compute/memory breakdown at startup

    # torch.compile the SDF network used in the hot path (trace + photo loss).
    # dynamic=True because the compacted trace feeds variable-size batches each
    # iteration. fp32-exact w.r.t. the eager model up to op-fusion reordering
    # (no precision change — does NOT enable TF32/autocast). ~1.3-2x on the
    # matmul/kernel-launch-bound CPL forward. Off → plain eager module.
    compile: bool = True

    # mask handling / ray sampling
    use_masks: bool = True          # False: ignore loaded masks during training;
                                    # no fg/bg split, no photo-mask gate

    # ray sampling: uniform within fg/bg strata, OR image-gradient-weighted (fg only)
    grad_weighted_sampling: bool =  False  # True: sample fg rays ∝ image-gradient
                                          # magnitude → more samples on edges/texture,
                                          # better fine-detail capture. bg stays uniform.
    grad_sampling_alpha:    float = 0.8   # mix: p = α·grad + (1−α)·uniform. The uniform
                                          # floor keeps smooth-but-real regions (flat
                                          # surfaces, silhouette interiors) from starving.
    fg_fraction:             float = 0.7   # foreground-ray share when sampling both strata
    force_fg_bg_split:       bool = False # preserve fg/bg split without a bg-sensitive loss
    init_hit_sampling:       bool = False # mask-free fg: trace det rays vs the init SDF once
                                          # and set fg = hit. Concentrates sampling on the
                                          # object (init hull is a conservative superset) with
                                          # no segmentation mask. Pair with fg_fraction<1 +
                                          # force_fg_bg_split for a full-frame escape valve.

    # alternative-view selection (which source cameras each reference view pairs
    # with). "nearest": the n_alt nearest camera centres (default, unchanged).
    # "pairs_file": read a precomputed MVSNet/NeuralWarp pair.txt and take the
    # first n_alt ranked source ids per reference view (scores ignored).
    # "uniform": pick n_alt source cameras uniformly at random (excluding self).
    # "arccos": sort by arccos(dot(d_i,d_j)) where d_k=norm(scene_centre−cam_k),
    #   i.e. angular separation of viewing directions; take n_alt smallest angles.
    #   NOTE: "arccos" ALSO switches per-step ray sampling to the 2-level
    #   uniform-cam scheme (one ref cam ~ Uniform(V) per step, all rays from it).
    # "arccos_nn": MINIMAL variant — identical arccos angular-distance neighbour
    #   ranking, but reference-view and ray selection are left EXACTLY as
    #   "nearest" (default flat sampling). Use this to isolate the effect of the
    #   neighbour-ranking change alone. Only the source-view *set* changes;
    #   visibility filtering, ZNCC, init, and sampling are untouched.
    view_selection: str         = "nearest"   # "nearest" | "pairs_file" | "uniform" | "arccos" | "arccos_nn"
    pairs_path:     Path | None = None         # required when view_selection=="pairs_file"

    # visibility filter
    n_alt:      int   = 6     # nearest-neighbour cameras per ray
    cos_thresh: float = 0.1   # |cos(n, view)| threshold for valid reprojections
    occ_mode:   str   = "from_hit"  # "pinhole": trace from alt cam origin | "from_hit": trace from hit point toward alt cam

    # loss weights
    w_photo:     float = 0.0
    w_feature:   float = 0.0  # cosine distance on precomputed feature maps
    feature_maps: Path | None = None
    w_ncc:       float = 1.0
    # PMVS NCC on the normal branch: L = w_ncc·NCC(x_θ, detach(n)) +
    # w_ncc_normal·NCC(detach(x_θ), n_θ). >0 enables a differentiable normal
    # (double-backward through ∇f) so the photometric loss reshapes local
    # curvature/orientation independently of the level-set position.
    w_ncc_normal: float = 0.0
    ncc_patch:   int   = 5
    ncc_half_pix: float = 2.0   # PMVS patch half-width in reference-view pixels
    # Normal-branch patch geometry, decoupled from the position branch. The
    # normal-branch ZNCC leverage scales with the patch half-extent in pixels
    # (a tangent-plane tilt only differentially reshapes the patch), so it
    # needs a longer lever arm than the position branch tolerates. Sentinel
    # <0 → fall back to ncc_patch / ncc_half_pix (backward-compatible).
    ncc_normal_patch:    int   = 9
    ncc_normal_half_pix: float = 4
    ncc_color:    str   = "rgb"  # "gray": Rec.601 luminance (DTU default — robust to
                                #   per-channel exposure/WB drift, 3x cheaper).
                                # "rgb": per-channel ZNCC averaged over R,G,B (legacy).
    ncc_grad_alpha: float = 0.0  # 0: intensity ZNCC only. >0: blend a second ZNCC on
                                # the patch gradient magnitude (Gipuma-style edge term,
                                # ZNCC-consistent): zncc←(1-α)·zncc_I+α·zncc_∇. Sharpens
                                # the depth minimum on fine texture (feathers). ~0.3-0.5.
    ncc_min:      float = 0.0   # PMVS photometric gate: drop pairs with ZNCC below this
    ncc_topk:     int   = 0     # 0: mean (1−ZNCC) over ALL valid alt views (legacy).
                                # >0: per-surface-point top-K best ZNCC across the n_alt
                                # pool (robust MVS aggregation à la PMVS/COLMAP — rejects
                                # occluded/grazing views). Applied from step 0 (safe with
                                # visual-hull init: x_θ already near the surface so ZNCC
                                # is meaningful, no warm-up needed). Use 3–4 with n_alt≈10.
    sample_mode:  str   = "bilinear"  # "bilinear" | "gaussian"
    gaussian_sigma:  float = 2.0
    gaussian_sigma_end: float = 0.5   # if > 0 and != gaussian_sigma: anneal sigma→this over training
    gaussian_radius: int   = 1
    # Spatial patch weighting: instead of a small hard P×P window, use a larger
    # patch with per-grid-point weight w = exp(-r/α), r = distance to patch
    # centre in grid units, α = ncc_patch_wsigma. Weighted ZNCC (weighted
    # mean/std/cov). 0 → uniform weights (legacy, exactly unchanged). Enables a
    # robust large window early + localised detail by annealing α↓.
    ncc_patch_wsigma:     float = 0.0
    ncc_patch_wsigma_end: float = 0.0  # if >0 and != ncc_patch_wsigma: anneal α→this over training
    # Gipuma (Galliani et al. 2015) bilateral / adaptive-support-weight NCC:
    # weight each patch pixel by photometric similarity to the centre in the
    # REFERENCE view, w = exp(-|I_p - I_q| / γ) (intensities in [0,1]). With a
    # FIXED large patch, annealing γ↓ shrinks the *effective* support window over
    # training (large/robust early → small/edge-hugging late) with no geometry
    # change. 0 → disabled. Combines multiplicatively with ncc_patch_wsigma.
    ncc_bilateral_gamma:     float = 0.0
    ncc_bilateral_gamma_end: float = 0.0  # if >0 and != ncc_bilateral_gamma: anneal γ→this over training
    w_cam_free:  float = 0.0
    w_sfm:       float = 0.0
    w_geo_sdf:   float = 0.0   # pure Geo-Neus L1 SDF loss on COLMAP pts (surface term ONLY,
                               # no free-space/behind bundle). Independent of w_sfm.
    sfm_min_views: int = 30   # filter COLMAP pts visible in fewer cameras than this
    sfm_behind_eps: float = 0.01  # step behind SFM point along camera ray; require f <= 0 there
    w_free:      float = 0.0   # free-space along SFM camera→point rays
    w_surf:      float = 0.0
    w_mvs:       float = 0.0
    mvs_depth_dir: Path | None = None           # if set, load MASt3R depths from this dir (IDR scans)
    mvsformer_depth_dir: Path | None = None     # if set, load MVSFormer++ depths instead of MASt3R
    mvsformer_conf_thr: float = 0.7            # confidence threshold for MVSFormer++ valid mask
    mvsdf_schedule: MvsdfScheduleConfig = field(default_factory=MvsdfScheduleConfig)
    bundle: BundleAdjustConfig = field(default_factory=BundleAdjustConfig)
    w_mvs_sdf:   float = 0.0  # MVSDF carving loss (volumetric SDF from depth consensus)
    n_mvs_sdf:   int   = 4096
    mvs_sdf_out_thresh: float = 0.7   # outside vote threshold (MVSDF out_thresh_perc)
    mvs_sdf_trunc: float = 1.25
    mvs_sdf_smooth: float = 0.0
    mvs_sdf_far_thresh: float = 1.25
    mvs_sdf_far_att: float = 1.0
    mvs_sdf_near_thresh: float = 0.1   # 5% of bbox size 2 → points closer than this get lower weight
    mvs_sdf_near_att: float = 0.1      # weight for near-surface uncertain points
    w_behind_hit: float = 0.0
    behind_eps:  float = 0.05
    w_mask_fg: float = 0.0        # superseded by w_sil (mask_loss_min_sdf with focal weighting)
    w_mask_bg: float = 0.0
    mask_fg_margin: float = 0.0
    mask_bg_margin: float = 1e-2
    n_mask_fg: int = 32
    n_mask_bg: int = 32
    debug_views:  str = ""    # opt-in: e.g. "16,32" → thorough per-view hole diagnostics
    debug_every:  int = 500   # steps between debug-view dumps
    debug_zoom_json: str = "" # opt-in JSON {"16":{"nose":[x0,y0,x1,y1],...}} → region crops
    w_idr_mask:     float = 0.0  # IDR mask loss weight (Yariv 2020); ρ=100 in the paper
    idr_n_samples:  int   = 32   # uniform ray samples for the hard min_t f (IDR uses 100;
                                 # 32 locates the argmin bucket just as well at ~1/3 the cost —
                                 # only t* gets the single grad eval, so precision is unaffected)
    w_sil:          float = 0.0   # mask_loss_min_sdf weight. 0.5 puts it on par with a typical NCC=0.5
                                  # signal so missing-piece gradient actually competes. Bump higher
                                  # (1.0–2.0) if pieces still missing after a few thousand steps.
    sil_s:          float = 50.0  # starting α; IDR: 50 → ×2 every 250 epochs up to 5 doublings → 1600
    sil_s_interval: int   = 6000  # steps between α doublings (0 = fixed). 6000 → 50,100,200,400,
                                  # 800,1600 at 0/6k/12k/18k/24k/≥30k (IDR-faithful spacing)
    sil_s_max_mults: int  = 5     # max doublings: α_max = sil_s * 2^max_mults
    sil_fg_offset:  float = 0.1   # shift sdf_min for fg rays so σ(−α·(sdf_min−offset))→1 on hits
    sil_bg_offset:  float = 0.05  # symmetric offset for bg: σ(−α·(sdf_min+offset))→0 on near-surface bg
                                  # rays. Tolerates ~bg_offset of mask noise at silhouette edges.
    sil_focal_gamma: float = 3.0  # focal weighting (1−p_correct)^γ. γ=3: easy hits ≈0.1% weight,
                                  # hard misses ≈100%. ~1000× contrast — concentrates gradient
                                  # almost entirely on missing-piece rays. (γ=2 standard, γ=3 for
                                  # the small-pieces-missing case where you want maximum focus.)
    sil_balance:    bool  = True  # rescale fg/bg contributions to 1/class_frac so a mostly-bg batch
                                  # doesn't dilute the fg signal.
    sil_norm_alpha: bool  = False # divide loss by α (legacy IDR scaling). False: scheduling α actually
                                  # sharpens gradient.
    sil_k:          int   = 16    # stratified samples per ray for silhouette
    sil_t_near:     float = 0.5
    sil_t_far:      float = 6.0
    w_eikonal:   float = 0.0    # wE in MVSDF paper (always fixed)
    n_eik_vol:   int   = 4096   # random volume points for eikonal (separate from ray samples)
    w_normal:    float = 0.0  # GT normal supervision (requires DTU normal maps)
    n_free:      int   = 32    # free-space samples per ray pair
    w_ray_free:  float = 0.0  # free-space along training rays before the hit
    n_ray_free:  int   = 8     # samples per hit ray

    # ---- soft-argmin photo-coherence (method.pdf §2) ----
    w_soft_argmin:   float = 0.0   # weight for soft-argmin depth pull (hit rays only)
    sa_n_candidates: int   = 16    # with sphere-trace occlusion on hit rays only; 16×6×B_hit ≈ photo_loss trace budget
    sa_tau_start:    float = 0.02  # initial Boltzmann temperature — must be < 0.09 for signal to dominate with 256 candidates
    sa_tau_end:      float = 0.005  # final temperature (sharp, local pick)
    sa_use_bg:       bool  = False  # background candidate misleads on bright DTU regions (cost≈0 for near-white pixels)
    sa_bg_color:     float = 1.0   # background colour (1.0 = white for DTU masks)
    sa_t_near:       float = 0.1   # near clamp for candidate depth interval
    sa_t_far:        float = 0.0   # far clamp (0.0 = inherit trace_cfg.t_far) — occlusion check handles interior candidates
    sa_start_step:   int   = 0     # step at which soft-argmin is first applied (warm-up)

    use_blender: bool = False
    single_view: int = -1  # >=0: restrict training to this one camera (overfit diagnostic)

    # debug region monitoring: semicolon-separated "name,view,u0,v0,u1,v1" in full-res px
    # example: "zygo,16,900,380,1060,500;mand,16,620,760,760,880"
    debug_regions: str = ""
    debug_region_freq: int = 500


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
    dtu_chamfer_freq:  int   = 0     # cadence (steps) for the cheap in-training sfm_surf diagnostic (0 = off)
    dtu_chamfer_res:   int   = 256   # MC resolution for the sfm_surf diagnostic
    blender_chamfer_freq: int = 0    # cadence (steps) for in-training Blender GT chamfer (0 = off)
    dtu_official_freq: int   = 0     # run DTUeval-python every N train steps (0 = off)
    dtu_official_res:  int   = 384   # MC resolution for periodic official eval
    dtu_official_bound: float = 1.0  # MC bound for periodic official eval
    dtu_official_mask_crop: bool = True  # NeuralWarp-style eval crop with dilated DTU object masks
    dtu_official_mask_dilate_px: int = 12
    dtu_official_mask_crop_min_ratio: float = 1.0
    dtu_official_mask_crop_min_views: int = 1
    tnt_eval_dir:  Path | None = None   # root or scene dir with TnT official GT assets
    tnt_official_scene: str | None = None  # official TnT scene name when scene path is staged/symlinked
    tnt_official_frame: str = "colmap-pose"  # "colmap-pose" | "colmap-local" | "colmap-sfm"
    tnt_official_freq: int = 0      # run official TnT F-score every N train steps (0 = off)
    tnt_official_res:  int = 512    # MC resolution for periodic official TnT eval
    tnt_official_bound: float = 1.5 # MC bound for periodic official TnT eval
    tnt_official_n_samples: int = 2_000_000  # area-uniform points sampled from MC mesh
    mc_level: float = 0.0            # marching-cubes isovalue; slightly >0 (e.g. 0.005) trims
                                     # noisy near-zero wandering in under-supervised pockets

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
        if d["train"]["pairs_path"] is not None:
            d["train"]["pairs_path"] = str(d["train"]["pairs_path"])
        if d["train"]["mvs_depth_dir"] is not None:
            d["train"]["mvs_depth_dir"] = str(d["train"]["mvs_depth_dir"])
        if d["train"]["mvsformer_depth_dir"] is not None:
            d["train"]["mvsformer_depth_dir"] = str(d["train"]["mvsformer_depth_dir"])
        if d["eval"]["dtu_eval_dir"] is not None:
            d["eval"]["dtu_eval_dir"] = str(d["eval"]["dtu_eval_dir"])
        if d["eval"]["tnt_eval_dir"] is not None:
            d["eval"]["tnt_eval_dir"] = str(d["eval"]["tnt_eval_dir"])
        return d
