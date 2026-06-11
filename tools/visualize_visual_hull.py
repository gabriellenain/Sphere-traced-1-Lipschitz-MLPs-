"""Carve a visual hull from NSVF cameras + per-image masks, then render a
normal-map PNG of the hull so we can see whether it's a reasonable init.

Usage:
    python tools/visualize_visual_hull.py \
        --scene data/tnt/Barn \
        --out   outputs/.../hull.png \
        [--res 256] [--bound 1.0] [--view 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lip_tracer.visual_hull import _points_inside_masks  # noqa: E402
from render_paper_marching import (_mesh_from_volume, _make_intersector,    # noqa: E402
                                   render_normals_only)


def _load_nsvf_with_masks(scene: Path, down: int = 2) -> dict:
    """NSVF loader + per-image masks from <scene>/mask/<stem>.png (255=fg)."""
    from PIL import Image as _PIL

    K = np.loadtxt(scene / "intrinsics.txt", dtype=np.float32)[:3, :3].copy()
    bbox = np.loadtxt(scene / "bbox.txt", dtype=np.float32)
    center = 0.5 * (bbox[:3] + bbox[3:6])
    scale  = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
    pose_paths = sorted((scene / "pose").glob("0_*.txt"))
    mask_dir = scene / "mask"
    assert mask_dir.exists(), f"{mask_dir} missing — run tools/sky_segment.py first"

    masks, c2ws, Ks = [], [], []
    H = W = None
    for pp in pose_paths:
        mp = mask_dir / (pp.stem + ".png")
        if not mp.exists():
            continue
        msk = (np.asarray(imageio.imread(mp)) > 127).astype(np.float32)
        if down > 1:
            H0, W0 = msk.shape
            H1, W1 = H0 // down, W0 // down
            msk = np.asarray(_PIL.fromarray((msk * 255).astype(np.uint8)).resize(
                (W1, H1), _PIL.NEAREST)).astype(np.float32) / 255.0
        c2w = np.loadtxt(pp, dtype=np.float32).reshape(4, 4)
        c2w[:3, 3] = (c2w[:3, 3] - center) / scale
        Ki = K.copy()
        if down > 1:
            Ki[0] /= down; Ki[1] /= down
        masks.append(msk); c2ws.append(c2w); Ks.append(Ki)
        if H is None: H, W = msk.shape
    return {
        "masks": np.stack(masks),
        "c2w":   np.stack(c2ws),
        "K":     np.stack(Ks),
        "H": H, "W": W,
        "center": center, "scale": scale,
    }


def carve_voxels(views: dict, res: int, bound: float) -> np.ndarray:
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    print(f"[hull] carving {res}^3 = {pts.shape[0]} points "
          f"vs {views['masks'].shape[0]} masks  (H,W={views['H']},{views['W']})")
    t0 = time.time()
    inside = _points_inside_masks(pts, views["masks"], views["c2w"], views["K"],
                                  views["H"], views["W"])
    print(f"[hull] carved in {time.time()-t0:.1f}s  occ={inside.mean():.1%}")
    return inside.reshape(res, res, res)


def render_hull(occ: np.ndarray, views: dict, bound: float,
                view_idx: int, out_png: Path) -> None:
    # Convert bool occupancy → signed scalar: -1 inside, +1 outside.
    # Then marching cubes at level 0.0 gives the surface.
    vol = np.where(occ, -1.0, 1.0).astype(np.float32)
    print(f"[hull] mc on {vol.shape} grid …")
    mesh = _mesh_from_volume(vol, bound=bound, res=vol.shape[0], level=0.0)
    print(f"[hull] mesh: {len(mesh.vertices)} verts  {len(mesh.faces)} faces")
    intersector = _make_intersector(mesh)

    H, W = views["H"], views["W"]
    c2w  = views["c2w"][view_idx]
    K    = views["K"][view_idx]
    img  = render_normals_only(mesh, intersector, c2w, K, H, W, ss=1)

    # contact-sheet: rgb | mask | normal-rendered hull
    rgb_path = Path(out_png).parent.parent.parent  # don't use, just show overlay
    rgb_img_path = (Path(__file__).resolve().parent.parent / "data" / "tnt" / "Barn"
                    / "rgb" / (sorted((Path(__file__).resolve().parent.parent /
                                       "data" / "tnt" / "Barn" / "pose").glob("0_*.txt"))[view_idx].stem + ".png"))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    if rgb_img_path.exists():
        rgb_full = imageio.imread(rgb_img_path)
        axes[0].imshow(rgb_full); axes[0].set_title(f"rgb view {view_idx}"); axes[0].axis("off")
    msk = views["masks"][view_idx]
    axes[1].imshow(msk, cmap="gray"); axes[1].set_title("not-sky mask (this view)"); axes[1].axis("off")
    axes[2].imshow(img.clip(0, 1)); axes[2].set_title("visual-hull normal map"); axes[2].axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[hull] -> {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out",   type=Path, required=True)
    ap.add_argument("--res",   type=int, default=192)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--down",  type=int, default=2,
                    help="downsample factor for carving (image-side)")
    ap.add_argument("--view",  type=int, default=0)
    args = ap.parse_args()

    views = _load_nsvf_with_masks(args.scene.resolve(), down=args.down)
    occ = carve_voxels(views, res=args.res, bound=args.bound)
    if occ.sum() == 0:
        print("[hull] empty hull — masks may be too restrictive")
        return
    render_hull(occ, views, bound=args.bound, view_idx=args.view, out_png=args.out)


if __name__ == "__main__":
    main()
