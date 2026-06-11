"""Render the same view with three normal-shading pipelines and stack them
side-by-side: MC + vertex-normals  |  MC + analytic ∇fθ  |  sphere-tracing.

This isolates whether noisy sphere-traced normals come from a noisy SDF
(test 2 will also be noisy) or from the sphere-tracer itself (test 2 stays
clean while test 3 is noisy).
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO = Path(__file__).resolve().parent


def _run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[fail] {label} exited rc={r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--view", type=int, default=23)
    ap.add_argument("--ss",   type=int, default=2)
    ap.add_argument("--mc_res", type=int, default=512)
    ap.add_argument("--out_dir", type=Path, default=Path("paper_render_test"))
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    d_mc      = out / "mc_vertex"
    d_mc_ana  = out / "mc_analytic"
    d_st      = out / "st"
    view = args.view
    name = f"view{view:03d}_normals.png"

    py = sys.executable
    common = ["--ckpt", str(args.ckpt), "--views", str(view), "--ss", str(args.ss)]
    # AO disabled — we only diff normal maps, AO is heavy and OOM-prone
    mc_extra = ["--mc_res", str(args.mc_res), "--ao_rays", "0"]

    _run([py, "-u", str(REPO / "analysis/render_paper_marching.py"),
          *common, *mc_extra,
          "--out_dir", str(d_mc)], "MC + vertex normals")

    _run([py, "-u", str(REPO / "analysis/render_paper_marching.py"),
          *common, *mc_extra, "--analytic_normals",
          "--out_dir", str(d_mc_ana)], "MC + analytic normals")

    _run([py, "-u", str(REPO / "analysis/render_paper.py"),
          *common, "--out_dir", str(d_st)], "sphere-tracing")

    panels = [imageio.imread(d / name) for d in (d_mc, d_mc_ana, d_st)]
    H = min(p.shape[0] for p in panels)
    W = min(p.shape[1] for p in panels)
    panels = [p[:H, :W] for p in panels]

    # 12 px white separator + label band
    sep = np.full((H, 12, panels[0].shape[2]), 255, dtype=panels[0].dtype)
    row = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)

    band_h = 28
    band = np.full((band_h, row.shape[1], row.shape[2]), 255, dtype=row.dtype)
    out_img = np.concatenate([band, row], axis=0)

    try:
        import cv2
        labels = ["MC | vertex normals", "MC | analytic grad", "sphere-trace"]
        col_w = H + 12  # placeholder; recompute below
        col_w = panels[0].shape[1]
        for i, txt in enumerate(labels):
            x = i * (col_w + 12) + 12
            cv2.putText(out_img, txt, (x, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        pass

    out_path = out / f"compare_view{view:03d}.png"
    imageio.imwrite(out_path, out_img)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
