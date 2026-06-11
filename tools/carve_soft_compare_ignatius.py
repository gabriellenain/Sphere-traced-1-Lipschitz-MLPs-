"""Hard-vs-soft visual hull on Ignatius using the GSAM mask_statue/ masks.

Compares:
  (a) hard intersection      p=1.00          (one bad mask erodes geometry)
  (b) soft voting            p=0.92          (tolerates ~8% of views wrong)
  (c) soft voting + dilation p=0.92, dil 6px (also fixes boundary under-seg)
Renders the three hulls into the same reference views for visual comparison.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi

from lip_tracer.data import load_views
from lip_tracer.visual_hull import _percentile_inside_masks
from _hull_depth_carve_scan24 import occ_to_mesh_world, render_mesh, default_ref_views, save_ply

SCENE = Path("data/tnt/Ignatius")
MASK_DIR = SCENE / "mask_statue"
OUT = Path("_diagnostics/ignatius_soft_hull"); OUT.mkdir(parents=True, exist_ok=True)
RES, BOUND, MIN_VIEWS = 256, 1.5, 8


def load_statue_masks(c2w_order_stems, dilate_px=0):
    masks = []
    for st in c2w_order_stems:
        m = np.array(Image.open(MASK_DIR / f"{st}.png").convert("L")) > 127
        if dilate_px > 0:
            m = ndi.binary_dilation(m, iterations=dilate_px)
        masks.append(m)
    return np.stack(masks).astype(np.float32)


def main():
    views = load_views(SCENE)                       # c2w/K/H/W in sorted-pose order
    c2ws, Ks = views["c2w"].numpy(), views["K"].numpy()
    H, W = views["H"], views["W"]
    stems = [p.stem for p in sorted((SCENE / "pose").glob("0_*.txt"))]
    assert len(stems) == len(c2ws), (len(stems), len(c2ws))

    lin = np.linspace(-BOUND, BOUND, RES, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx, yy, zz], -1).reshape(-1, 3)

    configs = [("hard_p100", 1.00, 0), ("soft_p92", 0.92, 0), ("soft_p92_dil6", 0.92, 6)]
    occs = {}
    for name, p, dil in configs:
        masks = load_statue_masks(stems, dilate_px=dil)
        inside = _percentile_inside_masks(pts, masks, c2ws, Ks, H, W, percentile=p, min_views=MIN_VIEWS)
        occ = inside.reshape(RES, RES, RES)
        occs[name] = occ
        v, f = occ_to_mesh_world(occ, BOUND)
        save_ply(v, f, OUT / f"hull_{name}.ply")
        print(f"  {name}: p={p} dil={dil}  occupied voxels={int(occ.sum())}  verts={len(v)}", flush=True)

    # render the three hulls into the same reference views
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rv = {"H": H, "W": W, "K": Ks, "c2w": c2ws}
    ref = default_ref_views(c2ws)
    cols = [(occs["hard_p100"], "hard  p=1.00"), (occs["soft_p92"], "soft  p=0.92"),
            (occs["soft_p92_dil6"], "soft p=0.92 + dil6")]
    rendered = []
    for occ, lab in cols:
        v, f = occ_to_mesh_world(occ, BOUND)
        rendered.append((render_mesh(v, f, rv, ref), lab))
    fig, ax = plt.subplots(len(ref), 3, figsize=(12, 4*len(ref)), squeeze=False)
    for c, (imgs, lab) in enumerate(rendered):
        for r in range(len(ref)):
            ax[r, c].imshow(np.clip(imgs[r], 0, 1)); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(lab, fontsize=11)
    fig.suptitle("Ignatius visual hull: hard intersection vs soft voting (263 GSAM masks)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "soft_vs_hard.png", dpi=130, bbox_inches="tight")
    print(f"\n[done] {OUT}/soft_vs_hard.png")
    for name in occs:
        print(f"  {name}: {int(occs[name].sum())} voxels")


if __name__ == "__main__":
    main()
