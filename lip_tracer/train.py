"""Training loop and sphere-SDF initialisation for the 1-Lip tracer."""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import SCENE, BLENDER_SCENE, OUT_DIR, Config, ModelConfig, InitConfig, TrainConfig, TraceConfig, EvalConfig
from .data import (load_colmap_points, load_camera_centers, load_sfm_pairs,
                   load_views, load_blender_views,
                   make_deterministic_rays, precompute_alt_cameras)
from .loss import (photo_loss, silhouette_loss, eikonal_loss, cam_free_loss,
                   sfm_sdf_loss, free_space_loss, mvs_depth_loss, behind_hit_loss)
from .model import FTheta, ConvexPotentialLayer
from .sphere_tracing import trace_unrolled, trace_nograd


# ---------- sphere initialisation ----------

def fit_sphere_init(
    model_cfg: ModelConfig = None,
    init_cfg:  InitConfig  = None,
    scene:     Path        = SCENE,
) -> tuple[FTheta, float]:
    """Train f_θ ≈ ‖x‖ − r as a warm start.

    If init_cfg.radius is None, uses 1.1 × p80 of COLMAP point distances so the
    sphere encloses the scene while cameras remain outside.
    """
    model_cfg = model_cfg or ModelConfig()
    init_cfg  = init_cfg  or InitConfig()
    radius    = init_cfg.radius

    if radius is None:
        pts    = load_colmap_points(scene)
        r      = pts.norm(dim=-1)
        radius = 1.1 * r.quantile(0.80).item()
        print(f"  sphere init  auto radius={radius:.3f}  "
              f"(p80={r.quantile(0.80):.3f} p99={r.quantile(0.99):.3f} max={r.max():.3f})")
    bound  = radius * 2.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f      = FTheta(hidden=model_cfg.hidden, depth=model_cfg.depth,
                    group_size=model_cfg.group_size).to(device)
    opt    = torch.optim.Adam(f.parameters(), lr=init_cfg.lr)
    for step in range(init_cfg.steps):
        x      = (2 * torch.rand(init_cfg.batch, 3, device=device) - 1) * bound
        target = x.norm(dim=-1) - radius
        loss   = F.mse_loss(f(x), target)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"  sphere init  step {step:5d}  mse {loss.item():.6f}")
    with torch.no_grad():
        f_origin = f(torch.zeros(1, 3, device=device)).item()
        frac_out = (f(load_camera_centers(scene).to(device)) > 0).float().mean().item()
    print(f"  sphere init done  r={radius:.3f}  f(0)={f_origin:.4f}  cams_outside={frac_out:.0%}")
    return f, radius


# ---------- training ----------

def train(cfg: Config = None) -> Path:
    """Train the 1-Lip SDF and save checkpoints to a timestamped run directory.

    Returns the path to the final checkpoint.
    """
    cfg = cfg or Config()
    # unpack for convenience
    model_cfg = cfg.model
    trace_cfg = cfg.trace
    init_cfg  = cfg.init
    train_cfg = cfg.train
    scene     = cfg.scene
    out_dir   = cfg.out_dir

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = "blender" if train_cfg.use_blender else "dtu"
    run_dir = out_dir / f"run_{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    OUT         = run_dir / "checkpoint.pt"
    RENDER_OUT  = run_dir / "render.png"
    SURFACE_OUT = run_dir / "surface.png"

    full_cfg = {**cfg.to_dict(), "cmd": " ".join(sys.argv)}
    (run_dir / "config.json").write_text(json.dumps(full_cfg, indent=2))
    print(f"  run dir → {run_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data ---
    if train_cfg.use_blender:
        views = load_blender_views(down=train_cfg.down)
    else:
        views = load_views(scene)
    images      = views["images"].to(device)
    masks       = views["masks"].to(device) if "masks" in views else None
    c2w_all     = views["c2w"].to(device)
    K_all       = views["K"].to(device)
    H, W        = views["H"], views["W"]
    V           = images.shape[0]
    origins_all = c2w_all[:, :3, 3]
    w2c_all     = torch.linalg.inv(c2w_all)

    if train_cfg.use_blender:
        sfm_pts      = torch.zeros(1, 3, device=device)
        sfm_origins  = torch.zeros(1, 3, device=device)
        sfm_targets  = torch.zeros(1, 3, device=device)
        n_sfm_pairs  = 1
        mvs_data     = None
    else:
        sfm_pts                 = load_colmap_points(scene).to(device)
        sfm_origins, sfm_targets = load_sfm_pairs(scene)
        sfm_origins, sfm_targets = sfm_origins.to(device), sfm_targets.to(device)
        n_sfm_pairs = sfm_origins.shape[0]
        print(f"  {n_sfm_pairs} (cam, sfm_point) pairs for free-space loss")
        from .geomvs import load_aligned_depths
        mvs_data = load_aligned_depths()
        print(f"  loaded {len(mvs_data['depths'])} aligned depth maps")

    # --- model ---
    _init_cfg = InitConfig(radius=1.0) if train_cfg.use_blender else init_cfg
    if train_cfg.use_blender:
        print("  [blender] sphere-init radius=1.0")
    f, _ = fit_sphere_init(model_cfg=model_cfg, init_cfg=_init_cfg, scene=scene)
    f = f.to(device)
    total_params = sum(p.numel() for p in f.parameters())
    n_cpl = sum(1 for m in f.net if isinstance(m, ConvexPotentialLayer))
    print(f"  model: hidden={f.hidden}  depth={n_cpl}  params={total_params:,}  gs={f.group_size}")

    # --- deterministic rays ---
    det        = make_deterministic_rays(views, down=train_cfg.down, device=device)
    total_rays = det["o"].shape[0]
    rpv        = det["rays_per_view"]
    print(f"  det rays: {total_rays} total, {rpv}/view, {V} views, down={train_cfg.down}")
    fg_frac = det["fg"].float().mean().item()
    print(f"  fg rays: {det['fg'].sum():.0f}/{total_rays} ({fg_frac:.1%})"
          f"{'  [NO MASKS]' if masks is None else ''}")

    # --- MVS depth alignment ---
    if mvs_data is None:
        mvs_depth_flat = torch.zeros(total_rays, device=device)
        mvs_valid_flat = torch.zeros(total_rays, dtype=torch.bool, device=device)
    else:
        H_d, W_d = H // train_cfg.down, W // train_cfg.down
        dep_all, val_all = [], []
        for v in range(V):
            d_full = mvs_data["depths"][v]; v_full = mvs_data["valid"][v]
            ys_d = torch.arange(H_d); xs_d = torch.arange(W_d)
            ys_f = ((ys_d.float() + 0.5) * train_cfg.down - 0.5).long().clamp(0, H - 1)
            xs_f = ((xs_d.float() + 0.5) * train_cfg.down - 0.5).long().clamp(0, W - 1)
            yy, xx = torch.meshgrid(ys_f, xs_f, indexing="ij")
            dep_all.append(d_full[yy, xx].reshape(-1))
            val_all.append(v_full[yy, xx].reshape(-1))
        mvs_depth_flat = torch.cat(dep_all).to(device)
        mvs_valid_flat = torch.cat(val_all).to(device)
        depth_p95      = mvs_depth_flat[mvs_valid_flat].quantile(0.95)
        mvs_valid_flat = mvs_valid_flat & (mvs_depth_flat < depth_p95)
        print(f"  MVS depth: {mvs_valid_flat.sum()}/{total_rays} valid, p95={depth_p95:.3f}")

    alt_nn = precompute_alt_cameras(views, train_cfg.n_alt).to(device)
    print(f"  alt cameras: {train_cfg.n_alt} NN per view")

    # --- held-out log subset ---
    n_log   = min(4096, total_rays)
    log_idx = torch.randperm(total_rays, device=device)[:n_log]
    log_o   = det["o"][log_idx];  log_d  = det["d"][log_idx]
    log_vi  = det["vi"][log_idx]; log_fg = det["fg"][log_idx]
    log_mvs_d = mvs_depth_flat[log_idx]; log_mvs_v = mvs_valid_flat[log_idx]
    z_cams_all  = c2w_all[:, :3, 2]
    log_cos     = (log_d * z_cams_all[log_vi]).sum(-1).abs().clamp(min=1e-6)
    log_t_target = log_mvs_d / log_cos

    # --- post-init sanity ---
    with torch.no_grad():
        test_o = origins_all[:5]
        print(f"  POST-INIT  f(cams[:5])={f(test_o).tolist()}")
        print(f"  POST-INIT  f(0,0,0)={f(torch.zeros(1,3,device=device)).item():.4f}")
        o_dbg = origins_all[0:1]
        d_dbg = -o_dbg / o_dbg.norm(dim=-1, keepdim=True)
        t_dbg = torch.tensor([0.0], device=device)
        print(f"  TRACE-DBG  o={o_dbg[0].tolist()}  f(o)={f(o_dbg).item():.4f}")
        for i in range(20):
            p = o_dbg + t_dbg.unsqueeze(-1) * d_dbg
            sv = f(p).item()
            print(f"    iter {i:2d}  t={t_dbg.item():.4f}  f(p)={sv:.4f}")
            if abs(sv) < 1e-3: print(f"    HIT"); break
            t_dbg = t_dbg + abs(sv)
            if t_dbg.item() > 10.0: print("    MISS"); break

    # --- optimiser ---
    opt       = torch.optim.Adam(f.parameters(), lr=train_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg.steps,
                                                             eta_min=train_cfg.lr / 10)
    best_score        = float("inf"); best_step = -1
    best_photo_score  = float("inf")
    BEST_OUT       = run_dir / "checkpoint_best_geo.pt"
    BEST_PHOTO_OUT = run_dir / "checkpoint_best_photo.pt"
    photo_history: list[tuple[int, float]] = []

    # ------------------------------------------------------------------ loop --
    for step in range(train_cfg.steps):
        idx      = torch.randint(0, total_rays, (train_cfg.batch,), device=device)
        o        = det["o"][idx];  u  = det["d"][idx]
        vi       = det["vi"][idx]; gt = det["gt"][idx]
        fg_self  = det["fg"][idx]

        x_theta, t, hit, eik_pts = trace_unrolled(f, o, u, trace_cfg)

        # normals — one f evaluation, reused for both |∇f| logging and photo loss
        with torch.enable_grad():
            xr    = x_theta.detach().clone().requires_grad_(True)
            n_raw = torch.autograd.grad(f(xr).sum(), xr, create_graph=False)[0].detach()
        if step % 50 == 0:
            print(f"step {step:5d}  |∇f| mean={n_raw.norm(dim=-1).mean():.4f}")
        n = n_raw / n_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # primary-camera uv (for NCC and photo)
        w2c_self  = w2c_all[vi]
        xc_self   = torch.einsum("bij,bj->bi", w2c_self[:, :3, :3], x_theta) + w2c_self[:, :3, 3]
        uv_h_self = torch.einsum("bij,bj->bi", K_all[vi], xc_self)
        uv_self   = uv_h_self[:, :2] / uv_h_self[:, 2:3].clamp(min=1e-6)

        # --- losses ---
        ph, ph_stats = photo_loss(
            f, x_theta, hit, n,
            vi, alt_nn, origins_all,
            images, K_all, w2c_all, masks, fg_self,
            H, W, uv_self,
            train_cfg.n_alt, train_cfg.cos_thresh,
            train_cfg.w_photo, train_cfg.w_ncc, train_cfg.ncc_patch,
            step,
        )

        sil  = (silhouette_loss(f, o, u, fg_self,
                                train_cfg.sil_k, train_cfg.sil_t_near, train_cfg.sil_t_far, train_cfg.sil_s)
                if train_cfg.w_sil > 0 else torch.zeros(1, device=device).squeeze())
        eik  = (eikonal_loss(f, eik_pts, train_cfg.batch, device)
                if train_cfg.w_eikonal > 0 else torch.zeros(1, device=device).squeeze())
        cfr  = (cam_free_loss(f, o)
                if train_cfg.w_cam_free > 0 else torch.zeros(1, device=device).squeeze())
        sfm  = (sfm_sdf_loss(f, sfm_pts, train_cfg.batch)
                if train_cfg.w_sfm > 0 else torch.zeros(1, device=device).squeeze())
        fs   = (free_space_loss(f, sfm_origins, sfm_targets, n_sfm_pairs,
                                train_cfg.batch, train_cfg.n_free)
                if train_cfg.w_free > 0 else torch.zeros(1, device=device).squeeze())
        mvs  = (mvs_depth_loss(x_theta, o, u, vi, c2w_all,
                               mvs_depth_flat, mvs_valid_flat, idx, step)
                if train_cfg.w_mvs > 0 else torch.zeros(1, device=device).squeeze())
        beh  = (behind_hit_loss(f, x_theta, hit, u, train_cfg.behind_eps)
                if train_cfg.w_behind_hit > 0 else torch.zeros(1, device=device).squeeze())

        loss = (ph + train_cfg.w_sil * sil + train_cfg.w_eikonal * eik
                + train_cfg.w_cam_free * cfr + train_cfg.w_sfm * sfm + train_cfg.w_free * fs
                + train_cfg.w_mvs * mvs + train_cfg.w_behind_hit * beh)
        photo_history.append((step, ph.item()))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=1.0)
        opt.step(); scheduler.step()

        # --- logging every 50 steps ---
        if step % 50 == 0:
            hw_grad    = f.head_weight.grad
            grad_norm  = hw_grad.norm().item() if hw_grad is not None else float("nan")
            with torch.no_grad():
                frac_fo_neg = (f(o) < 0).float().mean().item()
                frac_t_far  = (t >= 10.0 - 1e-3).float().mean().item()
                if hit.any():
                    _r = x_theta[hit].detach().norm(dim=-1)
                    xh_str = f"[{_r.min():.3f},{_r.mean():.3f},{_r.max():.3f}]"
                else:
                    xh_str = "[-]"
            pm = ph_stats
            print(f"step {step:5d}  cams {vi.unique().numel():2d}  "
                  f"loss {loss.item():.4f}  photo {ph.item():.4f}  sil {sil.item():.4f}  "
                  f"hit {hit.sum()}/{train_cfg.batch}  x_r {xh_str}  "
                  f"mask {pm['n_mask']}/{pm['n_in_frame']}/{pm['n_not_occl']}/"
                  f"{pm['n_cos_ok']}/{pm['n_total']}  "
                  f"f(o)<0 {frac_fo_neg:.2f}  t_far {frac_t_far:.2f}  "
                  f"sfm {sfm.item():.4f}  free {fs.item():.4f}  "
                  f"mvs {mvs.item():.4f}  beh {beh.item():.4f}  "
                  f"eik {eik.item():.4f}  ∇head {grad_norm:.6f}")

            # geometry metrics on held-out subset
            with torch.no_grad():
                f_sfm_all = f(sfm_pts).abs()
                sfm_mean  = f_sfm_all.mean().item()
                sfm_p90   = torch.quantile(f_sfm_all, 0.9).item()
                _, t_log, hit_log = trace_nograd(f, log_o, log_d, trace_cfg)
                hr_log    = hit_log.float().mean().item()
                valid     = log_mvs_v & hit_log
                dt_str    = "n=0"
                dt_mean   = float("inf")
                if valid.any():
                    dt      = (t_log[valid] - log_t_target[valid]).abs()
                    dt_mean = dt.mean().item()
                    dt_str  = f"mean={dt_mean:.4f} p90={dt.quantile(.9):.4f} n={int(valid.sum())}"
                cams_out  = (f(origins_all) > 0).float().mean().item()

            sil_str = ""
            if log_fg.any():
                tp_s = (hit_log & log_fg).float().sum()
                fp_s = (hit_log & ~log_fg).float().sum()
                fn_s = (~hit_log & log_fg).float().sum()
                sil_iou  = (tp_s / (tp_s + fp_s + fn_s).clamp(1)).item()
                sil_prec = (tp_s / (tp_s + fp_s).clamp(1)).item()
                sil_rec  = (tp_s / (tp_s + fn_s).clamp(1)).item()
                sil_f1   = 2 * sil_prec * sil_rec / max(sil_prec + sil_rec, 1e-6)
                vol = (2 * torch.rand(1024, 3, device=device) - 1) * 2.0
                vol.requires_grad_(True)
                with torch.enable_grad():
                    gv = torch.autograd.grad(f(vol).sum(), vol)[0]
                gnorm     = gv.norm(dim=-1)
                grad_mean = gnorm.mean().item()
                grad_std  = gnorm.std().item()
                r_str = "n/a"; normal_cons = 0.0
                if hit_log.any():
                    x_h  = log_o[hit_log] + t_log[hit_log].unsqueeze(-1) * log_d[hit_log]
                    r_str = f"{x_h.norm(dim=-1).mean():.3f}"
                    xr_h = x_h.detach().requires_grad_(True)
                    with torch.enable_grad():
                        nh = torch.autograd.grad(f(xr_h).sum(), xr_h)[0]
                    nh = nh / nh.norm(dim=-1, keepdim=True).clamp(1e-6)
                    ia = torch.randint(0, nh.shape[0], (min(512, nh.shape[0]),), device=device)
                    ib = torch.randint(0, nh.shape[0], (min(512, nh.shape[0]),), device=device)
                    normal_cons = (nh[ia] * nh[ib]).sum(-1).abs().mean().item()
                sil_str = (f"  sil_IoU={sil_iou:.3f} P={sil_prec:.3f} R={sil_rec:.3f} "
                           f"F1={sil_f1:.3f}  |∇f|={grad_mean:.3f}±{grad_std:.3f}  "
                           f"n_cons={normal_cons:.3f}  r={r_str}")

            print(f"  [geom@{step:5d}]  |f(sfm)| mean={sfm_mean:.4f} p90={sfm_p90:.4f}  "
                  f"hit={hr_log:.2%}  mvs_|Δt| {dt_str}  cams_out={cams_out:.1%}{sil_str}")

            score = float("inf")
            if train_cfg.use_blender and log_fg.any():
                score = 1.0 - sil_iou
            elif valid.any():
                score = dt_mean

            if score < best_score:
                best_score = score; best_step = step
                torch.save({"f": f.state_dict(), "step": step, "score": score,
                            "group_size": f.group_size, "depth": f.depth}, BEST_OUT)
                print(f"  [best_geo@{step}] score={score:.4f} → {BEST_OUT.name}")

            if ph.item() > 1e-6 and ph.item() < best_photo_score:
                best_photo_score = ph.item()
                torch.save({"f": f.state_dict(), "step": step, "photo": ph.item(),
                            "group_size": f.group_size, "depth": f.depth}, BEST_PHOTO_OUT)
                print(f"  [best_photo@{step}] photo={ph.item():.4f} → {BEST_PHOTO_OUT.name}")

    # --- save final ---
    torch.save({"f": f.state_dict(), "group_size": f.group_size, "depth": f.depth}, OUT)
    print(f"saved → {OUT}")
    if best_step >= 0:
        print(f"best_geo:   step {best_step}  score {best_score:.4f}  → {BEST_OUT.name}")
        print(f"best_photo: photo {best_photo_score:.4f}              → {BEST_PHOTO_OUT.name}")

    # --- photo loss curve ---
    if photo_history:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps_h, vals_h = zip(*photo_history)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps_h, vals_h, linewidth=0.8)
        ax.set_xlabel("step"); ax.set_ylabel("photo loss")
        ax.set_title(f"photo loss — hidden={f.hidden} depth={f.depth} gs={f.group_size}")
        ax.grid(True, alpha=0.3)
        plot_out = run_dir / "photo_loss.png"
        fig.savefig(plot_out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"photo loss plot → {plot_out}")

    return OUT


# ---------- entry point ----------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="1-Lip sphere-tracing trainer")
    ap.add_argument("--no-train",   action="store_true", help="skip training")
    ap.add_argument("--steps",      type=int,   default=500)
    ap.add_argument("--batch",      type=int,   default=1024)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--down",       type=int,   default=2)
    ap.add_argument("--blender",    action="store_true", help="use Blender lego dataset")
    ap.add_argument("--dataset",    choices=["skull", "lego"], default=None)
    ap.add_argument("--hidden",     type=int,   default=256)
    ap.add_argument("--depth",      type=int,   default=12)
    ap.add_argument("--group-size", type=int,   default=2)
    ap.add_argument("--w-photo",    type=float, default=1.0)
    ap.add_argument("--w-ncc",      type=float, default=0.0)
    ap.add_argument("--w-sil",      type=float, default=0.1)
    ap.add_argument("--w-eikonal",  type=float, default=0.0)
    ap.add_argument("--w-sfm",      type=float, default=0.0)
    ap.add_argument("--w-free",     type=float, default=0.0)
    ap.add_argument("--w-mvs",      type=float, default=0.0)
    ap.add_argument("--pt",         default=None, help="checkpoint to evaluate")
    ap.add_argument("--viewer",     action="store_true")
    ap.add_argument("--viewer-res", type=int, default=256)
    ap.add_argument("--viewer-port",type=int, default=8080)
    args = ap.parse_args()

    if args.dataset == "lego":
        args.blender = True
    elif args.dataset == "skull":
        args.blender = False

    run_cfg = Config(
        model=ModelConfig(hidden=args.hidden, depth=args.depth, group_size=args.group_size),
        train=TrainConfig(
            steps=args.steps, batch=args.batch, lr=args.lr, down=args.down,
            use_blender=args.blender,
            w_photo=args.w_photo, w_ncc=args.w_ncc, w_sil=args.w_sil,
            w_eikonal=args.w_eikonal, w_sfm=args.w_sfm, w_free=args.w_free,
            w_mvs=args.w_mvs,
        ),
    )

    ckpt_path: Path | None = None
    if not args.no_train:
        ckpt_path = train(run_cfg)

    if args.pt is not None:
        ckpt_path = Path(args.pt)
    elif ckpt_path is None:
        # fall back to best available in output dir
        for name in ("checkpoint_best_photo.pt", "checkpoint_best_geo.pt", "checkpoint.pt"):
            cand = OUT_DIR / name
            if cand.exists():
                ckpt_path = cand; break

    if ckpt_path is None:
        print("No checkpoint found — run without --no-train first."); sys.exit(1)

    print(f"loading {ckpt_path}")
    ckpt       = torch.load(ckpt_path, map_location="cpu")
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    hidden     = ckpt["f"]["head_weight"].shape[0]
    group_size = args.group_size or ckpt.get("group_size", 2)
    depth      = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                        if k.startswith("net.") and k.endswith(".weight")
                                        and "_u" not in k))
    print(f"  hidden={hidden}  depth={depth}  group_size={group_size}")
    f = FTheta(hidden=hidden, depth=depth, group_size=group_size).to(device)
    f.load_state_dict(ckpt["f"])

    from .visualize import visualize, render_vs_reference, view_in_viser, lego_stats, geom_stats, chamfer_stats

    eval_cfg = EvalConfig()
    if args.viewer:
        view_in_viser(f, res=args.viewer_res, bound=eval_cfg.bound(args.blender),
                      port=args.viewer_port, use_blender=args.blender)
        sys.exit(0)

    visualize(f, eval_cfg=eval_cfg, use_blender=args.blender)
    if args.blender:
        lego_stats(f, label=ckpt_path.name)
    else:
        render_vs_reference(f)
        geom_stats(f)
        chamfer_stats(f)
