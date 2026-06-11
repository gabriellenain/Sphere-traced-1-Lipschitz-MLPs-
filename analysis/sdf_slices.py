"""Conference-quality SDF slice figure.

N uniform cuts along one axis.  Each row:
  left  — SDF colormap on that slice (diverging, zero-contour in black)
  right — GT skull mesh (fixed camera) with blue cutting plane at that offset

Usage
-----
  python sdf_slices.py CKPT --gt_mesh MESH.ply
  python sdf_slices.py CKPT --gt_mesh MESH.ply --axis xz --n_cuts 4
  python sdf_slices.py CKPT --gt_mesh MESH.ply --axis xy --n_cuts 5 --cut_range 0.5
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from render_paper_marching import _checkpoint_model_kwargs, render_one, _make_intersector
from lip_tracer.model import make_model


# ------------------------------------------------------------------ model ----

def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    f = make_model(**_checkpoint_model_kwargs(ckpt)).to(device).eval()
    f.load_state_dict(ckpt["f"], strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    return f


# --------------------------------------------------------------- SDF slice ---

@torch.no_grad()
def eval_slice(f, axis: str, offset: float, bound: float, res: int,
               device: str, chunk: int = 65536) -> tuple[np.ndarray, str, str]:
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    a, b = np.meshgrid(lin, lin, indexing="ij")
    c = np.full_like(a, offset)
    if axis == "xy":
        pts = np.stack([a, b, c], -1); xl, yl = "x", "y"
    elif axis == "xz":
        pts = np.stack([a, c, b], -1); xl, yl = "x", "z"
    else:
        pts = np.stack([c, a, b], -1); xl, yl = "y", "z"
    flat = pts.reshape(-1, 3)
    out = [f(torch.from_numpy(flat[i:i+chunk]).to(device)).cpu().numpy()
           for i in range(0, len(flat), chunk)]
    return np.concatenate(out).reshape(res, res), xl, yl


# ------------------------------------------------------- 3-D context view ---

def _plane_corners(axis: str, offset: float, bound: float) -> np.ndarray:
    lo, hi = -bound, bound
    if axis == "xy":
        return np.array([[lo,lo,offset],[hi,lo,offset],[hi,hi,offset],[lo,hi,offset]], np.float64)
    elif axis == "xz":
        return np.array([[lo,offset,lo],[hi,offset,lo],[hi,offset,hi],[lo,offset,hi]], np.float64)
    else:
        return np.array([[offset,lo,lo],[offset,hi,lo],[offset,hi,hi],[offset,lo,hi]], np.float64)


def _project_pts(pts3d: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """World (N,3) → pixel (N,2).  c2w is camera-to-world 4×4."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    cam = (pts3d - t) @ R          # world→cam (row-vector convention)
    z = cam[:, 2].clip(1e-6)
    u = cam[:, 0] / z * K[0, 0] + K[0, 2]
    v = cam[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v], -1)


def render_skull_base(mesh, intersector, c2w, K, H, W, ao_rays=16, ao_radius=0.06):
    """Render skull mesh once (fixed camera) → (H,W,3) float32, white bg."""
    shaded, _, _, _, hit_alpha = render_one(
        mesh, intersector, c2w, K, H, W, ss=1,
        ao_rays=ao_rays, ao_radius=ao_radius,
    )
    a = hit_alpha[:, :, None]
    return (shaded * a + np.ones_like(shaded) * (1 - a)).astype(np.float32)


def overlay_plane(skull_img: np.ndarray, K, c2w, H, W,
                  axis, offset, bound,
                  color=(0.20, 0.47, 0.67), alpha=0.52) -> np.ndarray:
    """Composite semi-transparent plane quad onto skull_img → (H,W,3)."""
    corners2d = _project_pts(_plane_corners(axis, offset, bound), K, c2w)
    pil = Image.fromarray(np.clip(skull_img * 255, 0, 255).astype(np.uint8)).convert("RGBA")
    ov  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
    poly = [(float(x), float(y)) for x, y in corners2d]
    draw.polygon(poly, fill=(r, g, b, int(alpha * 255)))
    draw.polygon(poly, outline=(max(0,r-30), max(0,g-30), max(0,b-30), 230), width=2)
    return np.array(Image.alpha_composite(pil, ov).convert("RGB"), dtype=np.float32) / 255.0


# ------------------------------------------------------------------ figure ---

def make_figure(
    f, gt_mesh, intersector,
    context_c2w, context_K, context_H, context_W,
    axis, offsets, bound, res, device,
    vmax, cmap, contour_lw, tick_every,
    panel_size, fontsize, dpi, out_path,
    ao_rays=16,
):
    n = len(offsets)
    fixed_ax_name = {"xy": "z", "xz": "y", "yz": "x"}[axis]

    # --- evaluate all SDF slices ---
    print(f"evaluating {n} SDF slices ({axis}-plane, res={res})…", flush=True)
    slices = []
    for off in offsets:
        sdf, xl, yl = eval_slice(f, axis, off, bound, res, device)
        slices.append((sdf, xl, yl))
        print(f"  {fixed_ax_name}={off:+.3f}  SDF=[{sdf.min():.3f}, {sdf.max():.3f}]", flush=True)

    if vmax is None:
        all_abs = np.array([np.abs(s[0]).max() for s in slices])
        vmax = round(float(np.percentile(all_abs, 95)), 2) or 0.5
        print(f"  auto vmax={vmax}", flush=True)

    # --- render skull base once ---
    print("rendering skull base (fixed camera)…", flush=True)
    skull_base = render_skull_base(
        gt_mesh, intersector, context_c2w, context_K,
        context_H, context_W, ao_rays=ao_rays,
    )

    # --- build figure: n rows × 2 cols ---
    fig_w = panel_size * 2 + 0.7   # +0.7 for colorbar
    fig_h = panel_size * n
    fig, axes = plt.subplots(
        n, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.06, "hspace": 0.12},
        squeeze=False,
    )

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    last_im = None

    for i, (off, (sdf, xl, yl)) in enumerate(zip(offsets, slices)):
        # ---- left: SDF slice ----
        ax_sdf = axes[i, 0]
        ext = [-bound, bound, -bound, bound]
        last_im = ax_sdf.imshow(sdf.T, origin="lower", extent=ext,
                                cmap=cmap, norm=norm,
                                interpolation="bilinear", aspect="equal")
        lin = np.linspace(-bound, bound, res)
        ax_sdf.contour(lin, lin, sdf.T, levels=[0.0],
                       colors="black", linewidths=contour_lw)
        ax_sdf.set_xlim(-bound, bound); ax_sdf.set_ylim(-bound, bound)
        ax_sdf.xaxis.set_major_locator(ticker.MultipleLocator(tick_every))
        ax_sdf.yaxis.set_major_locator(ticker.MultipleLocator(tick_every))
        ax_sdf.tick_params(labelsize=fontsize - 3)
        ax_sdf.set_xlabel(xl, fontsize=fontsize - 1, labelpad=3)
        ax_sdf.set_ylabel(yl, fontsize=fontsize - 1, labelpad=3)
        ax_sdf.set_title(f"{fixed_ax_name} = {off:+.2f}",
                         fontsize=fontsize, fontweight="semibold", pad=5)

        # ---- right: skull + plane ----
        ax_ctx = axes[i, 1]
        ctx_img = overlay_plane(skull_base, context_K, context_c2w,
                                context_H, context_W,
                                axis, off, bound)
        ax_ctx.imshow(ctx_img, aspect="equal")
        ax_ctx.axis("off")
        if i == 0:
            ax_ctx.set_title("cut plane", fontsize=fontsize,
                             fontweight="semibold", pad=5)

    # shared colorbar
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.895, 0.08, 0.018, 0.84])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label("SDF", fontsize=fontsize, labelpad=8)
    cb.ax.tick_params(labelsize=fontsize - 3)
    cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cb.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved → {out_path}", flush=True)


# ----------------------------------------------------------------------- CLI --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt",     type=Path)
    ap.add_argument("--gt_mesh", type=Path, required=True,
                    help="skull mesh for the 3-D context view (GT or best trained)")
    ap.add_argument("--axis",   type=str, default="xz",
                    choices=["xy", "xz", "yz"],
                    help="slice orientation (default: xz — horizontal cuts)")
    ap.add_argument("--n_cuts", type=int,   default=4)
    ap.add_argument("--cut_range", type=float, default=None,
                    help="offsets span [-cut_range, +cut_range] (default: 0.6*bound)")
    ap.add_argument("--offsets", nargs="+", type=float, default=None,
                    help="explicit offsets (overrides --n_cuts / --cut_range)")
    ap.add_argument("--bound",  type=float, default=1.0)
    ap.add_argument("--res",    type=int,   default=1024)
    ap.add_argument("--vmax",   type=float, default=None)
    ap.add_argument("--cmap",   type=str,   default="RdBu_r")
    ap.add_argument("--contour_lw", type=float, default=2.0)
    ap.add_argument("--tick_every", type=float, default=0.5)
    ap.add_argument("--panel_size", type=float, default=4.5)
    ap.add_argument("--fontsize",   type=int,   default=13)
    ap.add_argument("--dpi",    type=int,   default=300)
    ap.add_argument("--ao_rays", type=int,  default=16)
    ap.add_argument("--context_view", type=int, default=8)
    ap.add_argument("--context_down", type=int, default=2)
    ap.add_argument("--scene",  type=Path,  default=None)
    ap.add_argument("--out",    type=Path,  default=None)
    args = ap.parse_args()

    # offsets
    if args.offsets is not None:
        offsets = args.offsets
    else:
        r = args.cut_range if args.cut_range is not None else 0.6 * args.bound
        offsets = np.linspace(-r, r, args.n_cuts).tolist()
    print(f"cuts: axis={args.axis}  offsets={[f'{o:+.3f}' for o in offsets]}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or args.ckpt.parent / "sdf_slices.png"

    # SDF model
    print(f"loading model {args.ckpt}", flush=True)
    f = load_model(args.ckpt, device)

    # GT mesh
    import trimesh
    print(f"loading mesh {args.gt_mesh}", flush=True)
    gt_mesh = trimesh.load(args.gt_mesh, process=False)
    intersector = _make_intersector(gt_mesh)

    # camera
    scene = args.scene
    if scene is None:
        cfg = args.ckpt.parent / "config.json"
        if cfg.exists():
            scene = Path(json.loads(cfg.read_text()).get("scene", ""))
    if scene is None or not scene.exists():
        ap.error("Could not resolve scene path — pass --scene explicitly.")

    from lip_tracer.data import load_views, load_blender_views
    use_blender = (scene / "transforms_train.json").exists()
    views = (load_blender_views(scene, split="train", down=1)
             if use_blender else load_views(scene, down=1))
    vi = min(args.context_view, views["c2w"].shape[0] - 1)
    d  = args.context_down
    H, W = views["H"] // d, views["W"] // d
    K = views["K"][vi].numpy().copy()
    K[0] /= d; K[1] /= d
    c2w = views["c2w"][vi].numpy()
    print(f"context: view={vi}  {H}×{W}", flush=True)

    make_figure(
        f, gt_mesh, intersector, c2w, K, H, W,
        axis=args.axis,
        offsets=offsets,
        bound=args.bound,
        res=args.res,
        device=device,
        vmax=args.vmax,
        cmap=args.cmap,
        contour_lw=args.contour_lw,
        tick_every=args.tick_every,
        panel_size=args.panel_size,
        fontsize=args.fontsize,
        dpi=args.dpi,
        out_path=out,
        ao_rays=args.ao_rays,
    )


if __name__ == "__main__":
    main()
