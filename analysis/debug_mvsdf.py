"""Debug script: verify MVSFormer++ depth coherence for scan65.

Three checks:
  1. Per-view depth stats (coverage, range, confidence)
  2. Cross-view depth consistency  (project view i depths into view j)
  3. Carving target slice  (2D XY slice at z=0 coloured by inside/outside target)

Run:
  python debug_mvsdf.py
  python debug_mvsdf.py --slice-axis z --slice-val 0.0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT  = Path(__file__).parent
SCENE = Path("/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan122")
MVSF  = SCENE / "mvsformer_depth_1536x1152_1536x1152"
OUT   = ROOT / "outputs"

sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────── load data ──

def load_mvs(scene: Path, depth_dir: Path, conf_thresh: float = 0.7):
    from lip_tracer.geomvs import load_mvsformer_depths_idr
    data = load_mvsformer_depths_idr(scene, depth_dir, conf_thresh=conf_thresh)
    assert data is not None, "Failed to load MVS data"
    depths = [d.numpy() if hasattr(d, "numpy") else np.asarray(d) for d in data["depths"]]
    valids = [v.numpy() if hasattr(v, "numpy") else np.asarray(v) for v in data["valid"]]
    c2ws   = [c.numpy() if hasattr(c, "numpy") else np.asarray(c) for c in data["c2w"]]
    Ks     = [k.numpy() if hasattr(k, "numpy") else np.asarray(k) for k in data["K"]]
    return depths, valids, c2ws, Ks


# ──────────────────────────────────────────────────── check 1: per-view stats

def check_per_view(depths, valids):
    print("\n=== Check 1: per-view depth stats ===")
    print(f"{'view':>4}  {'valid%':>7}  {'depth_min':>9}  {'depth_max':>9}  {'depth_mean':>10}")
    for i, (d, v) in enumerate(zip(depths, valids)):
        if v.any():
            print(f"{i:4d}  {v.mean():7.1%}  {d[v].min():9.3f}  {d[v].max():9.3f}  {d[v].mean():10.3f}")
        else:
            print(f"{i:4d}  {'NO VALID':>7}")


# ──────────────────────────────────────────── check 2: cross-view consistency

def project_depth_map(d_src, v_src, c2w_src, K_src, c2w_tgt, K_tgt):
    """Back-project valid pixels from src, re-project into tgt, compare depths."""
    H, W = d_src.shape
    ys, xs = np.where(v_src)
    if len(xs) == 0:
        return np.array([])

    # back-project to 3D (world)
    zs = d_src[ys, xs]
    # src cam coords
    x_cam = (xs - K_src[0, 2]) / K_src[0, 0] * zs
    y_cam = (ys - K_src[1, 2]) / K_src[1, 1] * zs
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)   # (N,3)

    R_src = c2w_src[:3, :3]
    t_src = c2w_src[:3, 3]
    pts_world = pts_cam @ R_src.T + t_src             # (N,3)

    # project into tgt
    R_tgt = c2w_tgt[:3, :3]
    t_tgt = c2w_tgt[:3, 3]
    w2c_tgt = np.eye(4)
    w2c_tgt[:3, :3] = R_tgt.T
    w2c_tgt[:3, 3]  = -R_tgt.T @ t_tgt
    pts_c  = pts_world @ w2c_tgt[:3, :3].T + w2c_tgt[:3, 3]  # (N,3)
    z_tgt  = pts_c[:, 2]

    # pixel coords in tgt
    u = pts_c[:, 0] / pts_c[:, 2] * K_tgt[0, 0] + K_tgt[0, 2]
    v = pts_c[:, 1] / pts_c[:, 2] * K_tgt[1, 1] + K_tgt[1, 2]
    H_t, W_t = d_src.shape  # same res assumed
    ok = (z_tgt > 0) & (u >= 0) & (u < W_t) & (v >= 0) & (v < H_t)
    if not ok.any():
        return np.array([])

    u_i = np.round(u[ok]).astype(int).clip(0, W_t - 1)
    v_i = np.round(v[ok]).astype(int).clip(0, H_t - 1)
    d_tgt_sampled = depths_global[view_tgt_global][v_i, u_i]  # filled by caller
    valid_tgt     = valids_global[view_tgt_global][v_i, u_i]

    err = np.abs(z_tgt[ok] - d_tgt_sampled)
    mask = valid_tgt & (d_tgt_sampled > 0)
    return err[mask]


# simpler version without globals
def cross_view_consistency(depths, valids, c2ws, Ks, n_pairs: int = 20):
    """Sample n_pairs adjacent-view pairs, measure mean abs depth error."""
    print("\n=== Check 2: cross-view depth consistency ===")
    V = len(depths)
    results = []
    pairs = [(i, (i + 4) % V) for i in range(0, V, V // n_pairs)][:n_pairs]
    for i, j in pairs:
        H, W = depths[i].shape
        ys, xs = np.where(valids[i])
        if len(xs) < 10:
            continue
        zs = depths[i][ys, xs]
        # back-project to world
        x_c = (xs - Ks[i][0, 2]) / Ks[i][0, 0] * zs
        y_c = (ys - Ks[i][1, 2]) / Ks[i][1, 1] * zs
        pts_c = np.stack([x_c, y_c, zs], axis=1)
        pts_w = pts_c @ c2ws[i][:3, :3].T + c2ws[i][:3, 3]
        # project into view j
        R_j = c2ws[j][:3, :3].T
        t_j = -R_j @ c2ws[j][:3, 3]
        pts_j = pts_w @ R_j.T + t_j
        z_j = pts_j[:, 2]
        u_j = pts_j[:, 0] / pts_j[:, 2] * Ks[j][0, 0] + Ks[j][0, 2]
        v_j = pts_j[:, 1] / pts_j[:, 2] * Ks[j][1, 1] + Ks[j][1, 2]
        ok = (z_j > 0) & (u_j >= 0) & (u_j < W) & (v_j >= 0) & (v_j < H)
        if not ok.any():
            continue
        ui = np.round(u_j[ok]).astype(int).clip(0, W - 1)
        vi = np.round(v_j[ok]).astype(int).clip(0, H - 1)
        d_sampled = depths[j][vi, ui]
        valid_j   = valids[j][vi, ui]
        mask = valid_j & (d_sampled > 0)
        if mask.sum() < 5:
            continue
        err = np.abs(z_j[ok][mask] - d_sampled[mask])
        results.append((i, j, mask.sum(), err.mean(), np.percentile(err, 90)))
        print(f"  views {i:2d}→{j:2d}: overlap={mask.sum():5d}  "
              f"mean_err={err.mean():.4f}  p90_err={np.percentile(err, 90):.4f}")
    if results:
        all_errs = [r[3] for r in results]
        print(f"  → overall mean consistency error: {np.mean(all_errs):.4f}  "
              f"(good < 0.05, bad > 0.15)")
    return results


# ───────────────────────────────────────── check 3: carving target 2D slice

def carving_slice(depths, valids, c2ws, Ks,
                  axis: str = "z", val: float = 0.0,
                  res: int = 128, out_thresh: float = 0.7,
                  trunc: float = 1.25,
                  bounds=(-0.65, 0.65)):
    """Sample a 2D grid of points at a fixed axis value and compute carving targets."""
    print(f"\n=== Check 3: carving target slice at {axis}={val:.2f} ===")
    lo, hi = bounds
    lin = np.linspace(lo, hi, res)
    g1, g2 = np.meshgrid(lin, lin)
    g1f, g2f = g1.ravel(), g2.ravel()
    z_fix = np.full_like(g1f, val)

    if axis == "z":
        pts = np.stack([g1f, g2f, z_fix], axis=1)   # X-Y slice
        ax_labels = ("X", "Y")
    elif axis == "y":
        pts = np.stack([g1f, z_fix, g2f], axis=1)   # X-Z slice
        ax_labels = ("X", "Z")
    else:
        pts = np.stack([z_fix, g1f, g2f], axis=1)   # Y-Z slice
        ax_labels = ("Y", "Z")

    N = pts.shape[0]
    V = len(depths)
    BIG = 1e6

    total_valid  = np.zeros(N)
    total_inside = np.zeros(N)
    best_inside  = np.full(N,  BIG)
    best_outside = np.full(N, -BIG)

    for v in range(V):
        H, W = depths[v].shape
        R = c2ws[v][:3, :3].T
        t = -R @ c2ws[v][:3, 3]
        pts_c = pts @ R.T + t
        z_c = pts_c[:, 2]

        u = pts_c[:, 0] / np.maximum(pts_c[:, 2], 1e-6) * Ks[v][0, 0] + Ks[v][0, 2]
        vp = pts_c[:, 1] / np.maximum(pts_c[:, 2], 1e-6) * Ks[v][1, 1] + Ks[v][1, 2]
        in_frame = (z_c > 0) & (u >= 0) & (u < W) & (vp >= 0) & (vp < H)

        ui = np.round(u).astype(int).clip(0, W - 1)
        vi = np.round(vp).astype(int).clip(0, H - 1)
        d_s = depths[v][vi, ui]
        v_s = valids[v][vi, ui]

        valid   = in_frame & v_s & (d_s > 0)
        inside  = valid & (z_c > d_s * 0.99)
        outside = valid & ~inside
        dist    = z_c - d_s

        total_valid  += valid.astype(float)
        total_inside += inside.astype(float)
        best_inside  = np.where(inside  & (dist < best_inside),  dist, best_inside)
        best_outside = np.where(outside & (dist > best_outside), dist, best_outside)

    scene_valid   = total_valid > 0
    outside_perc  = (total_valid - total_inside) / np.maximum(total_valid, 1e-9)
    scene_outside = (outside_perc > out_thresh) & scene_valid
    scene_inside  = scene_valid & ~scene_outside

    safe_in  = np.clip(best_inside,  -BIG, trunc)
    safe_out = np.clip(best_outside, -trunc, BIG)
    ave_dist = safe_in * scene_inside + safe_out * scene_outside
    target   = np.clip(-ave_dist, -trunc, trunc)
    target[~scene_valid] = np.nan

    # ─── stats ───
    n_sv  = scene_valid.sum()
    n_si  = scene_inside.sum()
    n_so  = scene_outside.sum()
    print(f"  supervised: {n_sv}/{N} ({n_sv/N:.1%})")
    print(f"  inside labels: {n_si} ({n_si/max(n_sv,1):.1%})   "
          f"outside labels: {n_so} ({n_so/max(n_sv,1):.1%})")
    tgt_valid = target[scene_valid]
    print(f"  target range: [{tgt_valid.min():.3f}, {tgt_valid.max():.3f}]  "
          f"mean={tgt_valid.mean():.3f}  std={tgt_valid.std():.3f}")
    near_frac = (np.abs(tgt_valid) < 0.1).mean()
    far_frac  = (np.abs(tgt_valid) > 0.5).mean()
    print(f"  near_frac (|tgt|<0.1): {near_frac:.2f}   far_frac (|tgt|>0.5): {far_frac:.2f}")

    # ─── plot ───
    tgt_img = target.reshape(res, res)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"MVS carving targets — {axis}={val:.2f} slice")

    # target value (continuous)
    im0 = axes[0].imshow(tgt_img, origin="lower", extent=[lo, hi, lo, hi],
                         cmap="RdBu", vmin=-trunc, vmax=trunc)
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("SDF target (R=outside/+, B=inside/−)")
    axes[0].set_xlabel(ax_labels[0]); axes[0].set_ylabel(ax_labels[1])

    # inside / outside / unseen labels
    label_img = np.zeros((res, res))   # 0=unseen, 1=inside, 2=outside
    label_img.ravel()[scene_inside]  = 1
    label_img.ravel()[scene_outside] = 2
    axes[1].imshow(label_img, origin="lower", extent=[lo, hi, lo, hi],
                   cmap=matplotlib.colors.ListedColormap(["gray", "blue", "red"]),
                   vmin=0, vmax=2)
    axes[1].set_title("Labels: gray=unseen, blue=inside, red=outside")
    axes[1].set_xlabel(ax_labels[0])

    # outside_perc (confidence the point is outside)
    op_img = outside_perc.reshape(res, res)
    op_img[~scene_valid.reshape(res, res)] = np.nan
    im2 = axes[2].imshow(op_img, origin="lower", extent=[lo, hi, lo, hi],
                         cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im2, ax=axes[2])
    axes[2].axhline(y=val if axis != "y" else 0, color="white", lw=0.5, ls="--")
    axes[2].set_title(f"outside_perc (threshold={out_thresh})")
    axes[2].set_xlabel(ax_labels[0])

    out_path = OUT / f"debug_mvsdf_slice_{axis}{val:+.2f}_out{out_thresh:.3f}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"  → saved {out_path}")
    plt.close()

    return target, scene_inside, scene_outside


# ─────────────────────────────────────────────────────────────────── main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",       type=Path, default=SCENE)
    ap.add_argument("--depth-dir",   type=Path, default=MVSF)
    ap.add_argument("--conf-thresh", type=float, default=0.7)
    ap.add_argument("--out-thresh",  type=float, default=0.7)
    ap.add_argument("--slice-axis",  default="z", choices=["x", "y", "z"])
    ap.add_argument("--slice-val",   type=float, default=0.0)
    ap.add_argument("--res",         type=int,   default=128)
    ap.add_argument("--bound",       type=float, default=0.65)
    ap.add_argument("--skip-consistency", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    depths, valids, c2ws, Ks = load_mvs(args.scene, args.depth_dir,
                                        conf_thresh=args.conf_thresh)
    print(f"Loaded {len(depths)} views, conf_thresh={args.conf_thresh}")

    check_per_view(depths, valids)

    if not args.skip_consistency:
        cross_view_consistency(depths, valids, c2ws, Ks)

    carving_slice(depths, valids, c2ws, Ks,
                  axis=args.slice_axis, val=args.slice_val,
                  res=args.res, out_thresh=args.out_thresh,
                  bounds=(-args.bound, args.bound))


if __name__ == "__main__":
    main()
