"""Phong-render the *target* SDF that hull-init regresses to.

Target = distance transform of the visual-hull occupancy grid
(see occ_to_sdf in lip_tracer/visual_hull.py). Wraps the grid as a
trilinearly-interpolated torch callable so the existing sphere-tracer
+ Phong-shader can render it from training cameras.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.config import TraceConfig
import lip_tracer.sphere_tracing as st
_orig_trace_nograd = st.trace_nograd
_PATCHED_CFG = TraceConfig(t_far=10.0, iters=128)  # cameras at ||c||=4.3..6.7, need t_far>=6.7

def _patched_trace_nograd(f, o, d, cfg=None, **kw):
    return _orig_trace_nograd(f, o, d, cfg=_PATCHED_CFG, **kw)

st.trace_nograd = _patched_trace_nograd

from lip_tracer.data import load_views
from lip_tracer.train import _render_poses
from lip_tracer.visual_hull import carve, occ_to_sdf


class GridSDF(torch.nn.Module):
    """f(x) via trilinear sampling of a (z,y,x) SDF grid on [-bound, bound]^3."""

    def __init__(self, sdf_grid: np.ndarray, bound: float, device: str) -> None:
        super().__init__()
        # sdf_grid axes are (z, y, x); torch grid_sample treats input dims as (D, H, W)
        # and grid's last axis as (x_norm, y_norm, z_norm). So shape (1,1,D=z,H=y,W=x)
        # pairs with coords passed as (x, y, z). Both match our world axes directly.
        self.register_buffer("vol",
            torch.from_numpy(sdf_grid.astype(np.float32)).view(1, 1, *sdf_grid.shape))
        self.to(device)
        self.bound = float(bound)
        # _render_poses pokes these attributes for introspection logs.
        self.architecture = "grid"; self.group_size = 1; self.depth = 0
        self.activation = "n/a"; self.input_encoding = "identity"; self.multires = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        coords = (x / self.bound).clamp(-1.0, 1.0)            # (N, 3) order (x,y,z)
        grid = coords.view(1, n, 1, 1, 3)                     # (1, D_out=n, 1, 1, 3)
        out = F.grid_sample(self.vol, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.view(n)


def main() -> None:
    scene = Path("data/mvmannequin_neus/kinette-cos-hx")
    out_dir = Path("outputs/_hull_target_kinette")
    (out_dir / "render").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bound = 1.5
    hull_res = 256

    print(f"device={device}  carving at res={hull_res}, bound={bound}")
    occ = carve(scene=scene, res=hull_res, bound=bound)
    print(f"  occupancy: {occ.sum()}/{occ.size}  ({100*occ.mean():.3f}%)")

    print("computing distance-transform SDF ...")
    _, sdf_flat = occ_to_sdf(occ, bound)
    sdf_grid = sdf_flat.reshape(occ.shape)
    print(f"  sdf grid range: [{sdf_grid.min():+.4f}, {sdf_grid.max():+.4f}]  "
          f"voxel={2*bound/(hull_res-1):.4f}")

    f = GridSDF(sdf_grid, bound, device)
    f.eval()
    with torch.no_grad():
        print(f"  f(0,0,0) = {f(torch.zeros(1,3,device=device)).item():+.4f}  (neg=inside)")
        rng = np.random.default_rng(0)
        xs = rng.normal(size=(1024, 3)).astype(np.float32)
        xs /= np.linalg.norm(xs, axis=-1, keepdims=True) + 1e-9
        for r in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
            v = f(torch.from_numpy(xs * r).to(device))
            print(f"  shell r={r:.2f}  f mean={v.mean().item():+.4f}  std={v.std().item():.4f}")

    views = load_views(scene)
    print(f"\nrendering 4 training views ...")
    _render_poses(f, views, step=0, run_dir=out_dir, device=device, res=400)
    print(f"\nrender → {out_dir/'render'}/render_00000.png")


if __name__ == "__main__":
    main()
