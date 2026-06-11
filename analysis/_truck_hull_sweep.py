#!/usr/bin/env python3
"""GPU percentile-hull sweep for the clean Truck scene.

The expensive part of a visibility-aware percentile hull is projecting every
voxel into every view to accumulate (n_view, n_fg). That is INDEPENDENT of the
percentile p, so we do it ONCE on the GPU, then threshold cheaply for each p.
Each candidate is clipped to the sparse-SfM ROI and reduced to its largest
connected component (the truck), then saved as top/front/side projections.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from lip_tracer.data import load_views
from lip_tracer.visual_hull import keep_largest_component


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("data/tnt/Truck"))
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--percentiles", type=float, nargs="+",
                    default=[0.85, 0.90, 0.95, 0.98, 0.99])
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=Path("_diagnostics/mvsformer_truck_gsam_clean/hull_sweep"))
    ap.add_argument("--vox-chunk", type=int, default=4_000_000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} res={args.res} bound={args.bound} min_views={args.min_views}")

    v = load_views(args.scene, down=1)
    masks = v["masks"].bool()                 # (V,H,W)
    c2w = v["c2w"].float(); K = v["K"].float()
    H, W = int(v["H"]), int(v["W"])
    nonblank = masks.view(masks.shape[0], -1).any(dim=1)
    print(f"views={masks.shape[0]}  blank(dropped)={int((~nonblank).sum())}  HxW={H}x{W}")

    lin = torch.linspace(-args.bound, args.bound, args.res)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)        # (N,3) world xyz
    N = pts.shape[0]
    n_view = torch.zeros(N, dtype=torch.int32, device=dev)
    n_fg = torch.zeros(N, dtype=torch.int32, device=dev)

    idx = torch.nonzero(nonblank).squeeze(1).tolist()
    for s in range(0, N, args.vox_chunk):
        e = min(s + args.vox_chunk, N)
        P = pts[s:e].to(dev)                                  # (c,3)
        nv = torch.zeros(e - s, dtype=torch.int32, device=dev)
        nf = torch.zeros(e - s, dtype=torch.int32, device=dev)
        for i in idx:
            R = c2w[i, :3, :3].to(dev); t = c2w[i, :3, 3].to(dev); Ki = K[i].to(dev)
            cam = (P - t) @ R                                 # world->cam (R orthonormal)
            z = cam[:, 2]
            valid = z > 1e-3
            zc = torch.where(valid, z, torch.ones_like(z))
            px = cam[:, 0] / zc * Ki[0, 0] + Ki[0, 2]
            py = cam[:, 1] / zc * Ki[1, 1] + Ki[1, 2]
            xi = torch.floor(px).long(); yi = torch.floor(py).long()
            inb = valid & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            nv += inb.int()
            if inb.any():
                m = masks[i].to(dev)
                fg = torch.zeros(e - s, dtype=torch.bool, device=dev)
                fg[inb] = m[yi[inb].clamp(0, H - 1), xi[inb].clamp(0, W - 1)]
                nf += fg.int()
        n_view[s:e] = nv; n_fg[s:e] = nf
    n_view = n_view.cpu().numpy(); n_fg = n_fg.cpu().numpy()
    print(f"max n_view={n_view.max()} median(seen)={np.median(n_view[n_view>0]):.0f}")

    # sparse-SfM ROI (truck AABB + 15% pad), same normalised frame
    roi = None
    sp = args.scene / "sparse_sfm_points.txt"
    if sp.exists():
        q = np.loadtxt(sp); q = q[np.all(np.isfinite(q), 1)]
        lo, hi = q.min(0), q.max(0); pad = np.maximum(0.15, 0.15 * (hi - lo))
        lo = np.maximum(lo - pad, -args.bound); hi = np.minimum(hi + pad, args.bound)
        L = np.linspace(-args.bound, args.bound, args.res)
        roi = ((L[:, None, None] >= lo[2]) & (L[:, None, None] <= hi[2]) &
               (L[None, :, None] >= lo[1]) & (L[None, :, None] <= hi[1]) &
               (L[None, None, :] >= lo[0]) & (L[None, None, :] <= hi[0]))
        print(f"SFM ROI lo={lo.round(3)} hi={hi.round(3)}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cfgs = args.percentiles
    fig, ax = plt.subplots(len(cfgs), 3, figsize=(11, 3 * len(cfgs)), squeeze=False)
    rows = []
    for r, p in enumerate(cfgs):
        inside = (n_view >= args.min_views) & (n_fg >= p * np.maximum(n_view, 1))
        occ = inside.reshape(args.res, args.res, args.res)
        if roi is not None:
            occ = occ & roi
        lc = keep_largest_component(occ)
        rec = dict(percentile=float(p), min_views=args.min_views,
                   occ_pct=float(100 * occ.mean()), lc_pct=float(100 * lc.mean()),
                   lc_voxels=int(lc.sum()))
        rows.append(rec)
        print(f"p={p:.2f}: occ={rec['occ_pct']:.2f}%  largest_comp={rec['lc_pct']:.2f}%  ({rec['lc_voxels']} vox)")
        for k in range(3):
            ax[r, k].imshow(lc.max(axis=k).astype(float), cmap="gray"); ax[r, k].axis("off")
        ax[r, 0].set_title(f"p={p} mv={args.min_views}  lc={rec['lc_pct']:.2f}%  [top]", fontsize=9)
        ax[r, 1].set_title("front", fontsize=9); ax[r, 2].set_title("side", fontsize=9)
    fig.suptitle("Truck clean hull sweep — largest component, SFM-ROI clipped, blanks dropped")
    fig.tight_layout(); fig.savefig(args.out / "hull_sweep.png", dpi=85, bbox_inches="tight")
    (args.out / "hull_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"saved {args.out/'hull_sweep.png'}")


if __name__ == "__main__":
    main()
