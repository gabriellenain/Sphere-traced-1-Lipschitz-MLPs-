"""Refine per-image depth maps from a checkpoint: NCC + 4-neighbour smoothness."""

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.config import TraceConfig
from lip_tracer.data import load_views, precompute_alt_cameras
from lip_tracer.loss import pmvs_ncc_loss
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd
from lip_tracer.train import load_config_json


def build_rays(views, view_idx, down, device):
    H, W = views["H"], views["W"]
    H_d, W_d = H // down, W // down
    K   = views["K"][view_idx].numpy()
    c2w = views["c2w"][view_idx].numpy()
    ys, xs = np.meshgrid(np.arange(H_d), np.arange(W_d), indexing="ij")
    xs_f = (xs + 0.5) * down - 0.5
    ys_f = (ys + 0.5) * down - 0.5
    d_cam = np.stack([(xs_f - K[0, 2]) / K[0, 0],
                      (ys_f - K[1, 2]) / K[1, 1],
                      np.ones_like(xs_f)], axis=-1)
    dirs = d_cam @ c2w[:3, :3].T
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    o = torch.from_numpy(np.broadcast_to(c2w[:3, 3], dirs.shape).copy()).float().to(device)
    d = torch.from_numpy(dirs.reshape(-1, 3)).float().to(device)
    yi = torch.from_numpy((ys_f + 0.5).astype(np.int64).ravel()).clamp(0, H - 1)
    xi = torch.from_numpy((xs_f + 0.5).astype(np.int64).ravel()).clamp(0, W - 1)
    fg = views["masks"][view_idx][yi, xi].to(device)
    return o.reshape(-1, 3), d, fg, H_d, W_d


def normals_from_depth(depth, o, d, H, W):
    pts = (o + depth.unsqueeze(-1) * d).reshape(H, W, 3)
    dx  = torch.roll(pts, -1, dims=1) - torch.roll(pts, 1, dims=1)
    dy  = torch.roll(pts, -1, dims=0) - torch.roll(pts, 1, dims=0)
    return F.normalize(torch.cross(dx, dy, dim=-1), dim=-1).reshape(-1, 3)


def depth_to_mesh(depth, o, d, valid, H, W):
    pts   = (o + depth.detach().unsqueeze(-1) * d).reshape(H, W, 3)
    v     = valid.reshape(H, W)
    verts_full = pts.reshape(-1, 3).cpu().numpy()
    idx   = np.arange(H * W).reshape(H, W)
    faces = []
    for y in range(H - 1):
        for x in range(W - 1):
            a, b, c, e = idx[y,x], idx[y,x+1], idx[y+1,x], idx[y+1,x+1]
            if v[y,x] and v[y,x+1] and v[y+1,x]:   faces.append([a, b, c])
            if v[y,x+1] and v[y+1,x+1] and v[y+1,x]: faces.append([b, e, c])
    if not faces:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    faces_full = np.array(faces, dtype=np.int64)
    used, inverse = np.unique(faces_full.reshape(-1), return_inverse=True)
    verts = verts_full[used]
    faces_compact = inverse.reshape(-1, 3).astype(np.int32)
    return verts, faces_compact


def depth_to_points(depth, o, d, valid):
    depth_det = depth.detach()
    keep = valid & torch.isfinite(depth_det) & (depth_det > 0.0)
    if not keep.any():
        return np.zeros((0, 3), dtype=np.float32)
    pts = o[keep] + depth_det[keep].unsqueeze(-1) * d[keep]
    return pts.detach().cpu().numpy().astype(np.float32)


def save_ply(path, verts, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in verts:  f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:  f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def extract_marching_cubes_mesh(model, bound, res, device, chunk=65536):
    from skimage import measure

    grid_t = torch.linspace(-bound, bound, res, device=device)
    xs, ys, zs = torch.meshgrid(grid_t, grid_t, grid_t, indexing="ij")
    pts = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)
    vals = []
    print(f"[mc] evaluating {len(pts):,} grid points (res={res}, bound={bound})", flush=True)
    with torch.no_grad():
        for i in range(0, len(pts), chunk):
            vals.append(model(pts[i:i + chunk]).detach().cpu())
    vol = torch.cat(vals).reshape(res, res, res).numpy()
    print(f"[mc] SDF range=[{vol.min():.4f},{vol.max():.4f}]", flush=True)
    if vol.min() > 0 or vol.max() < 0:
        raise ValueError("No zero crossing in marching-cubes grid. Increase --mc_bound.")
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0)
    verts = (verts / (res - 1) * (2 * bound) - bound).astype(np.float32)
    return verts, faces.astype(np.int32)


def smoothness_loss(depth, valid, H, W):
    t = depth.reshape(H, W)
    m = valid.reshape(H, W)

    mh = m[:, :-1] & m[:, 1:]
    mv = m[:-1, :] & m[1:, :]
    loss_h = (t[:, :-1] - t[:, 1:]).abs()[mh].sum()
    loss_v = (t[:-1, :] - t[1:, :]).abs()[mv].sum()
    denom = (mh.sum() + mv.sum()).clamp(min=1)
    return (loss_h + loss_v) / denom


def checkpoint_state_dict(ckpt):
    for key in ("model", "f", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            return key, ckpt[key]
    return "raw", ckpt


@torch.no_grad()
def reprojection_ncc_metrics(depth, valid, o, d, H_d, W_d, view_idx, alt_v,
                             views, images, K_all, w2c, device,
                             patch=7, half_pix=3.0):
    if valid.sum() == 0 or len(alt_v) == 0:
        return {
            "pairs": 0, "zncc_mean": None, "loss_mean": None,
            "kept_zncc_mean": None, "kept_loss_mean": None,
            "n_textured": 0, "n_kept": 0, "n_valid_patches": 0,
        }

    N = H_d * W_d
    vi_a = torch.full((N,), view_idx, dtype=torch.long, device=device)
    x3d = o + depth.unsqueeze(-1) * d
    nrm = normals_from_depth(depth, o, d, H_d, W_d)

    zncc_vals, kept_vals = [], []
    n_valid_patches = 0
    n_kept = 0
    pairs = 0
    for cam in alt_v:
        vi_b = torch.full((N,), int(cam.item()), dtype=torch.long, device=device)
        zncc, keep, n_valid = pmvs_ncc_loss(
            images, x3d[valid], nrm[valid], vi_a[valid], vi_b[valid],
            K_all, w2c, views["H"], views["W"],
            patch=patch, half_pix=half_pix, sample_mode="bilinear", ncc_min=0.0)
        n_valid_patches += int(n_valid)
        if zncc.numel() == 0:
            continue
        pairs += 1
        zncc_vals.append(zncc.detach())
        if keep.any():
            kept_vals.append(zncc[keep].detach())
            n_kept += int(keep.sum())

    if not zncc_vals:
        return {
            "pairs": 0, "zncc_mean": None, "loss_mean": None,
            "kept_zncc_mean": None, "kept_loss_mean": None,
            "n_textured": 0, "n_kept": n_kept, "n_valid_patches": n_valid_patches,
        }

    z = torch.cat(zncc_vals)
    if kept_vals:
        zk = torch.cat(kept_vals)
        kept_zncc = float(zk.mean().item())
    else:
        kept_zncc = None
    zncc_mean = float(z.mean().item())
    return {
        "pairs": pairs,
        "zncc_mean": zncc_mean,
        "loss_mean": 1.0 - zncc_mean,
        "kept_zncc_mean": kept_zncc,
        "kept_loss_mean": None if kept_zncc is None else 1.0 - kept_zncc,
        "n_textured": int(z.numel()),
        "n_kept": n_kept,
        "n_valid_patches": n_valid_patches,
    }


def _fmt_metric(m):
    if m["loss_mean"] is None:
        return "n/a"
    kept = "n/a" if m["kept_loss_mean"] is None else f"{m['kept_loss_mean']:.4f}"
    return (f"loss={m['loss_mean']:.4f} zncc={m['zncc_mean']:.4f} "
            f"kept_loss={kept} pairs={m['pairs']} "
            f"textured={m['n_textured']} kept={m['n_kept']}")


def refine_view(view_idx, model, views, alt_nn, device,
                down=4, n_iters=50, lr=5e-3, w_smooth=0.05, w_anchor=1.0,
                n_alt=4, n_eval_alt=4, chunk=2048, ncc_accum_backward=False,
                out_dir=Path("outputs/depth_refine"),
                ncc_patch=5, ncc_half_pix=2.0):

    trace_cfg = TraceConfig(newton_steps=2, iters=24)
    o, d, fg, H_d, W_d = build_rays(views, view_idx, down, device)
    N = H_d * W_d

    K_all  = views["K"].to(device)
    w2c    = torch.linalg.inv(views["c2w"].to(device))
    images = views["images"].to(device)

    print(f"  [view {view_idx:02d}] sphere tracing {N} rays ...", flush=True)
    with torch.no_grad():
        ts, hs = [], []
        for i in range(0, N, chunk):
            _, t_c, h_c = trace_nograd(model, o[i:i+chunk], d[i:i+chunk], trace_cfg)
            ts.append(t_c);  hs.append(h_c)
    t_init = torch.cat(ts);  hit = torch.cat(hs)
    with torch.no_grad():
        sdf0 = model(o[:8])
        sdf_hit = model(o[hit[:len(o)]][:4] + t_init[hit[:len(o)]][:4].unsqueeze(-1) * d[hit[:len(o)]][:4]) if hit[:len(o)].any() else torch.tensor([])
    print(f"  DEBUG sdf@origin[0:8]={sdf0.tolist()}", flush=True)
    print(f"  DEBUG hit={hit.sum()}/{N} fg={fg.sum()}/{N} t_range=[{t_init.min():.3f},{t_init.max():.3f}]", flush=True)
    print(f"  DEBUG sdf@hit_pts={sdf_hit.tolist()}", flush=True)
    valid  = hit & fg

    t_mean = t_init[valid].mean().clamp(min=0.1) if valid.any() else torch.tensor(1.0)
    depth  = t_init.clone();  depth[~valid] = t_mean.item()
    depth  = torch.nn.Parameter(depth.detach().clone())
    t_anchor   = t_init[valid].detach()
    x3d_anchor = (o[valid] + t_anchor.unsqueeze(-1) * d[valid]).detach()
    optim  = torch.optim.Adam([depth], lr=lr)
    vi_a   = torch.full((N,), view_idx, dtype=torch.long, device=device)
    pool   = alt_nn[view_idx].cpu().numpy()
    rng    = np.random.default_rng(view_idx)
    rng.shuffle(pool)
    alt_v      = torch.from_numpy(pool[:n_alt]).to(device)
    eval_alt_v = torch.from_numpy(pool[n_alt:n_alt + n_eval_alt]).to(device)

    print(f"  [view {view_idx:02d}] optimising depth ({valid.sum().item()} valid px) ...", flush=True)
    loss0 = 0.0
    for it in range(n_iters):
        optim.zero_grad()

        if ncc_accum_backward:
            ncc_sum, n_pairs = 0.0, 0
            for cam in alt_v:
                # Build a fresh graph per camera and backprop immediately. This
                # keeps peak memory almost independent of n_alt.
                x3d = o + depth.unsqueeze(-1) * d
                nrm = normals_from_depth(depth, o, d, H_d, W_d)
                vi_b = torch.full((N,), cam.item(), dtype=torch.long, device=device)
                zncc, _, _ = pmvs_ncc_loss(
                    images, x3d[valid], nrm[valid], vi_a[valid], vi_b[valid],
                    K_all, w2c, views["H"], views["W"],
                    patch=ncc_patch, half_pix=ncc_half_pix,
                    sample_mode="bilinear", ncc_min=0.0)
                if zncc.numel() > 0:
                    cam_loss = (1.0 - zncc).mean()
                    (cam_loss / max(1, len(alt_v))).backward()
                    ncc_sum += float(cam_loss.detach().item())
                    n_pairs += 1
            ncc_total = torch.tensor(ncc_sum / max(1, n_pairs), device=device)
        else:
            x3d = o + depth.unsqueeze(-1) * d
            nrm = normals_from_depth(depth, o, d, H_d, W_d)
            ncc_total, n_pairs = torch.tensor(0.0, device=device), 0
            for cam in alt_v:
                vi_b = torch.full((N,), cam.item(), dtype=torch.long, device=device)
                zncc, _, _ = pmvs_ncc_loss(
                    images, x3d[valid], nrm[valid], vi_a[valid], vi_b[valid],
                    K_all, w2c, views["H"], views["W"],
                    patch=ncc_patch, half_pix=ncc_half_pix,
                    sample_mode="bilinear", ncc_min=0.0)
                if zncc.numel() > 0:
                    ncc_total = ncc_total + (1.0 - zncc).mean();  n_pairs += 1
            if n_pairs: ncc_total = ncc_total / n_pairs
        sm = smoothness_loss(depth, valid, H_d, W_d)
        x3d_valid = o[valid] + depth[valid].unsqueeze(-1) * d[valid]
        anc = (x3d_valid - x3d_anchor).pow(2).sum(-1).mean()
        reg_loss = w_smooth * sm + w_anchor * anc
        if ncc_accum_backward:
            reg_loss.backward()
            loss_value = float(ncc_total.item() + reg_loss.detach().item())
        else:
            loss = ncc_total + reg_loss
            loss.backward()
            loss_value = float(loss.detach().item())
        with torch.no_grad():
            depth.grad[~valid] = 0.0
        optim.step()
        with torch.no_grad():
            depth.clamp_(min=0.01)
        if it == 0: loss0 = loss_value
        if (it+1) % 10 == 0 or it == n_iters-1:
            drift_norm = anc.item() ** 0.5  # RMS 3D displacement in normalised scene units
            print(f"    iter {it+1:3d}/{n_iters}  loss={loss_value:.4f}  "
                  f"ncc={ncc_total.item():.4f}  sm={sm.item():.5f}  "
                  f"anc={w_anchor*anc.item():.5f}(raw={anc.item():.5f},drift3d={drift_norm:.4f})  pairs={n_pairs}",
                  flush=True)

    print(f"  [view {view_idx:02d}] loss {loss0:.4f} → {loss_value:.4f}")

    train_before = reprojection_ncc_metrics(
        t_init, valid, o, d, H_d, W_d, view_idx, alt_v,
        views, images, K_all, w2c, device,
        patch=ncc_patch, half_pix=ncc_half_pix)
    train_after = reprojection_ncc_metrics(
        depth.detach(), valid, o, d, H_d, W_d, view_idx, alt_v,
        views, images, K_all, w2c, device,
        patch=ncc_patch, half_pix=ncc_half_pix)
    held_before = reprojection_ncc_metrics(
        t_init, valid, o, d, H_d, W_d, view_idx, eval_alt_v,
        views, images, K_all, w2c, device,
        patch=ncc_patch, half_pix=ncc_half_pix)
    held_after = reprojection_ncc_metrics(
        depth.detach(), valid, o, d, H_d, W_d, view_idx, eval_alt_v,
        views, images, K_all, w2c, device,
        patch=ncc_patch, half_pix=ncc_half_pix)
    print(f"  [view {view_idx:02d}] reproj train before: {_fmt_metric(train_before)}")
    print(f"  [view {view_idx:02d}] reproj train after:  {_fmt_metric(train_after)}")
    print(f"  [view {view_idx:02d}] reproj held  before: {_fmt_metric(held_before)}")
    print(f"  [view {view_idx:02d}] reproj held  after:  {_fmt_metric(held_after)}")

    with torch.no_grad():
        vb, fb = depth_to_mesh(t_init, o, d, valid, H_d, W_d)
        va, fa = depth_to_mesh(depth.detach(), o, d, valid, H_d, W_d)
        pb = depth_to_points(t_init, o, d, valid)
        pa = depth_to_points(depth.detach(), o, d, valid)
    tag = f"view{view_idx:02d}_down{down}"
    save_ply(out_dir / f"{tag}_before.ply", vb, fb)
    save_ply(out_dir / f"{tag}_after.ply",  va, fa)
    save_ply(out_dir / f"{tag}_points_before.ply", pb, np.zeros((0, 3), dtype=np.int32))
    save_ply(out_dir / f"{tag}_points_after.ply",  pa, np.zeros((0, 3), dtype=np.int32))
    print(f"  saved {tag}: {len(fb)} / {len(fa)} tris, {len(pb)} / {len(pa)} pts")

    metrics = {
        "view": int(view_idx),
        "down": int(down),
        "valid_pixels": int(valid.sum().item()),
        "train_alt_views": [int(x) for x in alt_v.detach().cpu().tolist()],
        "heldout_alt_views": [int(x) for x in eval_alt_v.detach().cpu().tolist()],
        "train_before": train_before,
        "train_after": train_after,
        "heldout_before": held_before,
        "heldout_after": held_after,
        "loss_before": float(loss0),
        "loss_after": float(loss_value),
        "ncc_patch": int(ncc_patch),
        "ncc_half_pix": float(ncc_half_pix),
        "ncc_accum_backward": bool(ncc_accum_backward),
    }
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))

    for t_flat, name in [(t_init, "before"), (depth.detach(), "after")]:
        t_map = t_flat.reshape(H_d, W_d).cpu().numpy()
        v_map = valid.reshape(H_d, W_d).cpu().numpy()
        tv = t_map[v_map]
        if len(tv) == 0: continue
        img = np.zeros((H_d, W_d), dtype=np.uint8)
        lo, hi = tv.min(), tv.max()
        if hi > lo: img[v_map] = ((t_map[v_map]-lo)/(hi-lo)*255).astype(np.uint8)
        imageio.imwrite(out_dir / f"{tag}_depth_{name}.png", img)

    return metrics


def summarize_metrics(all_metrics, out_dir):
    if not all_metrics:
        return

    def mean_metric(section, key):
        vals = [m[section][key] for m in all_metrics if m[section][key] is not None]
        return None if not vals else float(np.mean(vals))

    summary = {
        "n_views": len(all_metrics),
        "train_before_zncc_mean": mean_metric("train_before", "zncc_mean"),
        "train_after_zncc_mean": mean_metric("train_after", "zncc_mean"),
        "heldout_before_zncc_mean": mean_metric("heldout_before", "zncc_mean"),
        "heldout_after_zncc_mean": mean_metric("heldout_after", "zncc_mean"),
        "train_before_loss_mean": mean_metric("train_before", "loss_mean"),
        "train_after_loss_mean": mean_metric("train_after", "loss_mean"),
        "heldout_before_loss_mean": mean_metric("heldout_before", "loss_mean"),
        "heldout_after_loss_mean": mean_metric("heldout_after", "loss_mean"),
    }
    for prefix in ("train", "heldout"):
        zb = summary[f"{prefix}_before_zncc_mean"]
        za = summary[f"{prefix}_after_zncc_mean"]
        lb = summary[f"{prefix}_before_loss_mean"]
        la = summary[f"{prefix}_after_loss_mean"]
        summary[f"{prefix}_zncc_delta"] = None if zb is None or za is None else za - zb
        summary[f"{prefix}_loss_delta"] = None if lb is None or la is None else la - lb

    print("\n=== reprojection summary (mean over views) ===")
    def fmt(x, signed=False):
        if x is None:
            return "n/a"
        return f"{x:+.4f}" if signed else f"{x:.4f}"

    print(f"  train   zncc {fmt(summary['train_before_zncc_mean'])} → "
          f"{fmt(summary['train_after_zncc_mean'])} "
          f"(Δ {fmt(summary['train_zncc_delta'], signed=True)})")
    print(f"  heldout zncc {fmt(summary['heldout_before_zncc_mean'])} → "
          f"{fmt(summary['heldout_after_zncc_mean'])} "
          f"(Δ {fmt(summary['heldout_zncc_delta'], signed=True)})")
    print(f"  train   loss {fmt(summary['train_before_loss_mean'])} → "
          f"{fmt(summary['train_after_loss_mean'])} "
          f"(Δ {fmt(summary['train_loss_delta'], signed=True)})")
    print(f"  heldout loss {fmt(summary['heldout_before_loss_mean'])} → "
          f"{fmt(summary['heldout_after_loss_mean'])} "
          f"(Δ {fmt(summary['heldout_loss_delta'], signed=True)})")

    (out_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2))


def _chamfer_from_pts(pts_world: np.ndarray, gt_pts: np.ndarray, gt_info: dict,
                      scene_path: Path, device: str, label: str,
                      n_sample: int = 1_000_000,
                      max_dist: float = 20.0,
                      gt_crop_padding: float = 0.02) -> dict | None:
    """Match the pipeline of analysis/compare_dtu_chamfer.py:
    pred ObsMask → GT mask-filter → GT pred-bbox crop → GT pred-dist filter → Chamfer.
    """
    from compare_dtu_chamfer import _in_obs, _nn_metrics, _mask_filter_gt, _crop_to_bbox
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    if len(pts_world) > n_sample:
        pts_world = pts_world[rng.choice(len(pts_world), n_sample, replace=False)]

    # 1. filter pred by ObsMask
    pred = pts_world[_in_obs(pts_world, gt_info["ObsMask"], gt_info["BB"], gt_info["Res"])]
    if len(pred) < 100:
        print(f"  [chamfer {label}] only {len(pred)} pts in ObsMask — skipping", flush=True)
        return None

    # 2. filter GT by scene masks
    keep_mf, _ = _mask_filter_gt(gt_pts, scene_path)
    gt_eval = gt_pts[keep_mf]

    # 3. crop GT to pred bounding box
    keep_crop, _ = _crop_to_bbox(gt_eval, pred, gt_crop_padding)
    gt_eval = gt_eval[keep_crop]

    # 4. remove GT points farther than max_dist from any pred point
    d_gt_to_pred, _ = cKDTree(pred).query(gt_eval, k=1, workers=-1)
    gt_eval = gt_eval[d_gt_to_pred <= max_dist]

    if len(gt_eval) == 0:
        print(f"  [chamfer {label}] no GT points remain after filtering — skipping", flush=True)
        return None

    m, _, _ = _nn_metrics(pred, gt_eval, max_dist=max_dist, device=device)
    print(f"  [chamfer {label}]  accuracy={m['accuracy']:.4f}  "
          f"completeness={m['completeness']:.4f}  chamfer={m['chamfer']:.4f}  "
          f"(pred={len(pred):,}  gt={len(gt_eval):,})", flush=True)
    return m


def _load_ply_verts(path: Path) -> np.ndarray:
    """Read vertex xyz from an ASCII PLY saved by save_ply()."""
    lines = path.read_text().splitlines()
    n_verts = 0
    for line in lines:
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
            break
    start = lines.index("end_header") + 1
    rows = [list(map(float, lines[start + i].split())) for i in range(n_verts)]
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run",      default="outputs/run_20260510_094943_scan65")
    p.add_argument("--down",     type=int,   default=4)
    p.add_argument("--iters",    type=int,   default=50)
    p.add_argument("--lr",       type=float, default=5e-3)
    p.add_argument("--w_smooth",  type=float, default=0.05)
    p.add_argument("--w_anchor",  type=float, default=1.0,
                   help="L2 anchor weight (relative units) keeping depth near SDF init")
    p.add_argument("--views",    type=int,   nargs="+")
    p.add_argument("--n_alt",    type=int,   default=8)
    p.add_argument("--n_eval_alt", type=int, default=4,
                   help="views used only for held-out reprojection metrics")
    p.add_argument("--n_pool",   type=int,   default=16,
                   help="candidate pool of nearest cameras; randomly split into n_alt train + n_eval_alt heldout")
    p.add_argument("--ncc_patch", type=int, default=5)
    p.add_argument("--ncc_half_pix", type=float, default=2.0)
    p.add_argument("--ncc_accum_backward", action="store_true",
                   help="backpropagate NCC camera-by-camera to reduce peak memory for large n_alt")
    p.add_argument("--out",      default="outputs/depth_refine")
    p.add_argument("--save_mc_mesh", action="store_true",
                   help="extract and save the checkpoint SDF mesh with marching cubes")
    p.add_argument("--mc_mesh_path", default=None,
                   help="existing MC mesh PLY to evaluate at startup instead of extracting it")
    p.add_argument("--mc_res", type=int, default=512)
    p.add_argument("--mc_bound", type=float, default=1.0)
    p.add_argument("--dtu_eval_dir", default=None,
                   help="path to DTU evaluation data (Points/stl + ObsMask); enables Chamfer eval")
    p.add_argument("--n_chamfer_pts", type=int, default=1_000_000,
                   help="points sampled per cloud for Chamfer evaluation")
    args = p.parse_args()

    run_dir  = Path(args.run)
    out_dir  = Path(args.out);  out_dir.mkdir(parents=True, exist_ok=True)
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config_json(run_dir / "config.json")
    mc  = cfg.model
    model = make_model(hidden=mc.hidden, depth=mc.depth, group_size=mc.group_size,
                       activation=mc.activation, input_encoding=mc.input_encoding,
                       multires=mc.multires, architecture=mc.architecture).to(device)
    ckpt = torch.load(run_dir / "checkpoint_final.pt", map_location=device)
    state_key, state = checkpoint_state_dict(ckpt)
    load_info = model.load_state_dict(state, strict=False)
    model_keys = set(model.state_dict().keys())
    matched = len(model_keys & set(state.keys())) if isinstance(state, dict) else 0
    if matched == 0:
        raise RuntimeError(
            f"No model weights matched in checkpoint {run_dir / 'checkpoint_final.pt'} "
            f"(selected key: {state_key}). Check checkpoint format."
        )
    model.eval()
    print(f"loaded {run_dir}/checkpoint_final.pt [{state_key}] "
          f"matched={matched} missing={len(load_info.missing_keys)} "
          f"unexpected={len(load_info.unexpected_keys)}")

    if args.save_mc_mesh:
        verts, faces = extract_marching_cubes_mesh(model, args.mc_bound, args.mc_res, device)
        save_ply(out_dir / f"mc_mesh_res{args.mc_res}_bound{args.mc_bound:g}.ply", verts, faces)
        print(f"[mc] saved {out_dir / f'mc_mesh_res{args.mc_res}_bound{args.mc_bound:g}.ply'} "
              f"({len(verts):,} verts, {len(faces):,} faces)", flush=True)
        if args.dtu_eval_dir is not None:
            import re
            from compare_dtu_chamfer import _load_dtu_gt, _sample_surface, _to_world, _infer_scan_id
            scene_path_ = Path(cfg.scene)
            cam_dict_   = np.load(scene_path_ / "cameras.npz")
            scale_mat_  = cam_dict_["scale_mat_0"].astype(np.float64)
            gt_pts_, gt_info_ = _load_dtu_gt(Path(args.dtu_eval_dir), _infer_scan_id(scene_path_))
            mc_world = _to_world(_sample_surface(verts, faces, args.n_chamfer_pts, seed=0), scale_mat_)
            print("[chamfer] MC mesh baseline (sanity check):", flush=True)
            _chamfer_from_pts(mc_world, gt_pts_, gt_info_, scene_path_,
                              device, label="MC mesh", n_sample=args.n_chamfer_pts)
    elif args.mc_mesh_path is not None and args.dtu_eval_dir is not None:
        import trimesh
        from compare_dtu_chamfer import _load_dtu_gt, _sample_surface, _to_world, _infer_scan_id
        scene_path_ = Path(cfg.scene)
        cam_dict_   = np.load(scene_path_ / "cameras.npz")
        scale_mat_  = cam_dict_["scale_mat_0"].astype(np.float64)
        gt_pts_, gt_info_ = _load_dtu_gt(Path(args.dtu_eval_dir), _infer_scan_id(scene_path_))
        m_ = trimesh.load(str(args.mc_mesh_path), process=False)
        mc_world = _to_world(
            _sample_surface(np.asarray(m_.vertices, dtype=np.float32),
                            np.asarray(m_.faces, dtype=np.int32),
                            args.n_chamfer_pts, seed=0),
            scale_mat_)
        print(f"[chamfer] MC mesh baseline from {args.mc_mesh_path}:", flush=True)
        _chamfer_from_pts(mc_world, gt_pts_, gt_info_, scene_path_,
                          device, label="MC mesh", n_sample=args.n_chamfer_pts)

    views  = load_views(cfg.scene, down=1)
    n_pool = max(args.n_pool, args.n_alt + args.n_eval_alt)
    alt_nn = precompute_alt_cameras(views, n_alt=n_pool).to(device)
    V      = views["c2w"].shape[0]
    print(f"scene: {cfg.scene}  views: {V}  H={views['H']} W={views['W']}")

    metrics = []
    for vi in (args.views or range(V)):
        metrics.append(refine_view(vi, model, views, alt_nn, device,
                                   down=args.down, n_iters=args.iters, lr=args.lr,
                                   w_smooth=args.w_smooth, w_anchor=args.w_anchor,
                                   n_alt=args.n_alt,
                                   n_eval_alt=args.n_eval_alt, out_dir=out_dir,
                                   ncc_patch=args.ncc_patch,
                                   ncc_half_pix=args.ncc_half_pix,
                                   ncc_accum_backward=args.ncc_accum_backward))

    summarize_metrics(metrics, out_dir)

    if args.dtu_eval_dir is not None:
        import re
        from compare_dtu_chamfer import _load_dtu_gt, _sample_surface, _to_world, _infer_scan_id

        scene_path = Path(cfg.scene)
        scan_id    = _infer_scan_id(scene_path)
        cam_dict   = np.load(scene_path / "cameras.npz")
        scale_mat  = cam_dict["scale_mat_0"].astype(np.float64)
        dtu_dir    = Path(args.dtu_eval_dir)

        print(f"\n[chamfer] scan{scan_id}  dtu_eval_dir={dtu_dir}", flush=True)
        gt_pts, gt_info = _load_dtu_gt(dtu_dir, scan_id)
        print(f"[chamfer] GT: {len(gt_pts):,} pts in ObsMask", flush=True)

        tag   = f"down{args.down}"
        views_done = args.views or list(range(views["c2w"].shape[0]))

        # --- before: aggregate sphere-traced (unrefined) depth points ---
        before_verts = []
        for vi in views_done:
            p_ = out_dir / f"view{vi:02d}_{tag}_points_before.ply"
            if not p_.exists():
                p_ = out_dir / f"view{vi:02d}_{tag}_before.ply"
            if p_.exists():
                before_verts.append(_load_ply_verts(p_))
        if before_verts:
            pts_before_obj = np.concatenate(before_verts)
            pts_before = _to_world(pts_before_obj, scale_mat)
            save_ply(out_dir / f"depth_points_before_{tag}.ply",
                     pts_before_obj.astype(np.float32),
                     np.zeros((0, 3), dtype=np.int32))
            save_ply(out_dir / f"depth_points_before_{tag}_world.ply",
                     pts_before.astype(np.float32),
                     np.zeros((0, 3), dtype=np.int32))
            chamfer_before = _chamfer_from_pts(pts_before, gt_pts, gt_info, scene_path,
                                               device, label="depth before",
                                               n_sample=args.n_chamfer_pts)
        else:
            chamfer_before = None

        # --- after: aggregate refined depth points ---
        after_verts = []
        for vi in views_done:
            p_ = out_dir / f"view{vi:02d}_{tag}_points_after.ply"
            if not p_.exists():
                p_ = out_dir / f"view{vi:02d}_{tag}_after.ply"
            if p_.exists():
                after_verts.append(_load_ply_verts(p_))
        if after_verts:
            pts_after_obj = np.concatenate(after_verts)
            pts_after = _to_world(pts_after_obj, scale_mat)
            save_ply(out_dir / f"depth_points_after_{tag}.ply",
                     pts_after_obj.astype(np.float32),
                     np.zeros((0, 3), dtype=np.int32))
            save_ply(out_dir / f"depth_points_after_{tag}_world.ply",
                     pts_after.astype(np.float32),
                     np.zeros((0, 3), dtype=np.int32))
            chamfer_after = _chamfer_from_pts(pts_after, gt_pts, gt_info, scene_path,
                                              device, label="depth after ",
                                              n_sample=args.n_chamfer_pts)
        else:
            chamfer_after = None

        # --- mc mesh baseline if it was saved ---
        mc_path = out_dir / f"mc_mesh_res{args.mc_res}_bound{args.mc_bound:g}.ply"
        if mc_path.exists():
            import trimesh
            m_ = trimesh.load(str(mc_path), process=False)
            mc_pts_world = _to_world(
                _sample_surface(np.asarray(m_.vertices, dtype=np.float32),
                                np.asarray(m_.faces, dtype=np.int32),
                                args.n_chamfer_pts, seed=0),
                scale_mat)
            _chamfer_from_pts(mc_pts_world, gt_pts, gt_info, scene_path,
                              device, label="MC mesh     ",
                              n_sample=args.n_chamfer_pts)

        # summary
        if chamfer_before is not None and chamfer_after is not None:
            delta = chamfer_after["chamfer"] - chamfer_before["chamfer"]
            print(f"\n[chamfer] before={chamfer_before['chamfer']:.4f}  "
                  f"after={chamfer_after['chamfer']:.4f}  delta={delta:+.4f}", flush=True)
            chamfer_summary = {"before": chamfer_before, "after": chamfer_after, "delta": delta}
            if mc_path.exists():
                chamfer_summary["mc"] = {"chamfer": None}  # filled above if needed
            (out_dir / "chamfer_metrics.json").write_text(json.dumps(chamfer_summary, indent=2))

    print(f"\ndone — {out_dir}/")


if __name__ == "__main__":
    main()
