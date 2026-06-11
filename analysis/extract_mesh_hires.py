#!/usr/bin/env python3
"""Hi-res marching-cubes extraction cropped to the object AABB.

Crops to the normalized GT bbox (+ margin) and uses an ~isotropic target voxel
size (default 1 mm), so the grid is spent on the object instead of empty air.
Evaluates the SDF in x-slabs to avoid materializing the full coordinate grid.
Writes the predicted mesh in the CALIBRATION WORLD frame (m), matching
eval_mvmannequin_official.extract_pred_mesh, so the result can be fed to that
script via --mesh and run through the exact official protocol.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def load_model(ckpt_path: Path, device: str):
    import torch
    sys.path.insert(0, str(REPO))
    from lip_tracer.model import make_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["f"]
    arch = ckpt.get("architecture", "cpl")
    hidden = ckpt.get("hidden", 256)
    for k, v in state.items():
        if "weight" in k and v.ndim >= 2 and "head" not in k and "encoder" not in k:
            hidden = v.shape[-1]; break
    f = make_model(hidden=hidden, depth=ckpt.get("depth", 8),
                   group_size=ckpt.get("group_size", 2),
                   activation=ckpt.get("activation", "groupsort"),
                   input_encoding=ckpt.get("input_encoding", "pe"),
                   multires=ckpt.get("multires", 6),
                   architecture=arch).to(device)
    f.load_state_dict(state, strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    f.eval()
    return f


def extract(ckpt_path: Path, scene: Path, out_ply: Path,
            voxel_mm: float = 1.0, margin: float = 0.10,
            batch: int = 1 << 19, device: str = "auto") -> Path:
    import torch
    from skimage.measure import marching_cubes
    import trimesh

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    f = load_model(ckpt_path, device)

    scale_mat = np.load(scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
    mm_per_unit = float(np.linalg.norm(scale_mat[:3, :3], axis=0)[0]) * 1000.0

    # crop box = normalized GT bbox + margin (comfortably contains the prediction)
    gt = trimesh.load(str(scene / "gt_mesh.ply"), process=False)
    lo, hi = np.asarray(gt.bounds, dtype=np.float64)
    pad = (hi - lo) * margin
    lo, hi = lo - pad, hi + pad

    voxel_unit = voxel_mm / mm_per_unit
    res = np.maximum(np.ceil((hi - lo) / voxel_unit).astype(int) + 1, 2)
    axes = [np.linspace(lo[i], hi[i], res[i], dtype=np.float32) for i in range(3)]
    spacing = [float((hi[i] - lo[i]) / (res[i] - 1)) for i in range(3)]
    print(f"[grid] crop lo={lo.round(3).tolist()} hi={hi.round(3).tolist()}", flush=True)
    print(f"[grid] res={res.tolist()}  voxel~{voxel_mm}mm  total={np.prod(res.astype(float)):.3e}", flush=True)

    ys, zs = np.meshgrid(axes[1], axes[2], indexing="ij")
    yz = np.stack([ys.ravel(), zs.ravel()], axis=-1)          # (Ny*Nz, 2)
    yz_t = torch.from_numpy(yz).to(device)
    vol = np.empty((res[0], res[1] * res[2]), dtype=np.float32)
    with torch.no_grad():
        for i, x in enumerate(axes[0]):
            xcol = torch.full((yz_t.shape[0], 1), float(x), device=device)
            pts = torch.cat([xcol, yz_t], dim=1)
            out = torch.cat([f(pts[j:j + batch]) for j in range(0, pts.shape[0], batch)])
            vol[i] = out.squeeze(-1).float().cpu().numpy()
            if i % 50 == 0:
                print(f"[grid]   x-slab {i}/{res[0]}", flush=True)
    vol = vol.reshape(res[0], res[1], res[2])
    if vol.min() > 0 or vol.max() < 0:
        raise RuntimeError(f"surface not in crop box (vol range [{vol.min():.3f},{vol.max():.3f}])")

    v_grid, faces, *_ = marching_cubes(vol, level=0.0, spacing=tuple(spacing))
    v_norm = v_grid.astype(np.float64) + lo                    # back to normalized frame
    v_h = np.concatenate([v_norm, np.ones((len(v_norm), 1))], axis=1)
    v_world = (scale_mat @ v_h.T).T[:, :3]

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=v_world, faces=faces, process=False).export(str(out_ply))
    print(f"[mesh] verts={len(v_world):,} faces={len(faces):,} -> {out_ply}", flush=True)
    return out_ply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--voxel-mm", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    extract(args.ckpt, args.scene, args.out,
            voxel_mm=args.voxel_mm, margin=args.margin, device=args.device)


if __name__ == "__main__":
    main()
