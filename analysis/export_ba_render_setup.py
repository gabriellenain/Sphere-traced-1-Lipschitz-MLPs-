"""Bake bundle-adjusted camera poses into a cameras.npz so the existing
sphere-trace + Blender pipeline can render the post-BA reconstruction with the
SAME poses BA co-optimised (not the original calibration).

Reconstructs each adjusted world_mat by INVERTING data.py's decomposition:
  world_mat -> (K, R_w2c, t_w2c) -> c2w (R_base=R^T, t_base=normalized centre)
BA gives c2w deltas:  R_c2w = rodrigues(log_rot) @ R_base ,  t_c2w = t_base + dt
then we go back  world_mat_adj = K @ [ (R_c2w)^T | -(R_c2w)^T @ (scale_mat @ [t_c2w,1]) ].

Writes:
  <out>/scene_ba/cameras.npz   (adjusted) + image/ mask/ symlinks to the orig scene
  <out>/run_ba/config.json     (copy, scene -> scene_ba) + ckpt/ba_final.pt symlink

Usage:
  python export_ba_render_setup.py --run-dir <run> --ba <ba_final.pt> [--ckpt-name ba_final.pt]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import rq

from lip_tracer.bundle_adjustment import rodrigues


def decompose(world_mat, scale_mat):
    """world_mat[:3,:4] -> (K, R_w2c, cam_center_normalized) matching data.py."""
    P = world_mat[:3, :4].astype(np.float64)
    M = P[:, :3]
    K, R = rq(M)
    sign = np.sign(np.diag(K)); sign[sign == 0] = 1.0
    T = np.diag(sign)
    K = K @ T
    R = T @ R
    if np.linalg.det(R) < 0:
        K[:, 2] *= -1.0
        R[2, :] *= -1.0
    K = K / K[2, 2]
    t = np.linalg.solve(K, P[:, 3])
    cam_center = -R.T @ t                                   # DTU world (mm)
    cc_h = np.concatenate([cam_center, [1.0]])
    cc_norm = (np.linalg.inv(scale_mat) @ cc_h)[:3]         # normalized frame
    return K, R, cc_norm


def recompose(K, R_c2w_adj, cc_norm_adj, scale_mat):
    """adjusted c2w (normalized) -> world_mat[:3,:4] in DTU world convention."""
    R_w2c = R_c2w_adj.T
    cc_world = (scale_mat @ np.concatenate([cc_norm_adj, [1.0]]))[:3]
    t_w2c = -R_w2c @ cc_world
    P = np.concatenate([R_w2c, t_w2c[:, None]], axis=1)     # (3,4)
    return K @ P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ba", type=Path, required=True, help="ba_final.pt")
    ap.add_argument("--ckpt-name", default="ba_final.pt")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <run-dir>/ba_render")
    args = ap.parse_args()

    cfg = json.loads((args.run_dir / "config.json").read_text())
    scene = Path(cfg["scene"])
    out = args.out or (args.run_dir / "ba_render")
    scene_ba = out / "scene_ba"
    run_ba = out / "run_ba"
    (scene_ba).mkdir(parents=True, exist_ok=True)
    (run_ba / "ckpt").mkdir(parents=True, exist_ok=True)

    cams = np.load(scene / "cameras.npz")
    V = sum(k.startswith("world_mat_") and "inv" not in k for k in cams.files)

    ck = torch.load(args.ba, map_location="cpu", weights_only=False)
    cp = ck["cam_params"]
    log_rot = (cp["log_rot"] * cp["free_mask"]).numpy().astype(np.float64)
    dt      = (cp["dt"]      * cp["free_mask"]).numpy().astype(np.float64)
    R_base  = cp["R_base"].numpy().astype(np.float64)     # = R_w2c^T (c2w rot)
    t_base  = cp["t_base"].numpy().astype(np.float64)     # = normalized centre
    dR_all  = rodrigues(torch.from_numpy(log_rot)).numpy()  # (V,3,3)
    assert R_base.shape[0] == V, f"cam_params V={R_base.shape[0]} != scene V={V}"

    npz = {}
    max_err = 0.0
    for i in range(V):
        wm = cams[f"world_mat_{i}"].astype(np.float64)
        sm = cams[f"scale_mat_{i}"].astype(np.float64)
        K, R_w2c, cc_norm = decompose(wm, sm)

        # sanity: zero-delta recompose must reproduce the original world_mat
        wm_check = recompose(K, R_w2c.T, cc_norm, sm)
        max_err = max(max_err, float(np.abs(wm_check - wm[:3, :4]).max()))

        R_c2w_adj = dR_all[i] @ R_base[i]                 # rodrigues @ R_base
        cc_norm_adj = t_base[i] + dt[i]
        wm_adj = np.eye(4)
        wm_adj[:3, :4] = recompose(K, R_c2w_adj, cc_norm_adj, sm)
        npz[f"world_mat_{i}"] = wm_adj.astype(np.float64)
        npz[f"scale_mat_{i}"] = sm
        if f"scale_mat_inv_{i}" in cams.files:
            npz[f"scale_mat_inv_{i}"] = cams[f"scale_mat_inv_{i}"]
        # also verify R_base matches our decomposition (convention check)
        assert np.abs(R_base[i] - R_w2c.T).max() < 1e-4, \
            f"view {i}: R_base != decomposed c2w rotation ({np.abs(R_base[i]-R_w2c.T).max():.2e})"

    print(f"[check] zero-delta world_mat reconstruction max abs err = {max_err:.3e}")
    np.savez(scene_ba / "cameras.npz", **npz)
    print(f"[write] {scene_ba/'cameras.npz'}  ({V} views)")

    # symlink image/ + mask/ (and eval_mask if present) into scene_ba
    for sub in ("image", "mask", "eval_mask"):
        src = scene / sub
        if src.is_dir():
            link = scene_ba / sub
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
            print(f"[link] {link} -> {src}")

    # tiny run dir: same config but scene -> scene_ba, ckpt -> ba_final.pt
    cfg_ba = dict(cfg)
    cfg_ba["scene"] = str(scene_ba.resolve())
    (run_ba / "config.json").write_text(json.dumps(cfg_ba, indent=2))
    ckpt_link = run_ba / "ckpt" / args.ckpt_name
    if ckpt_link.is_symlink() or ckpt_link.exists():
        ckpt_link.unlink()
    ckpt_link.symlink_to(args.ba.resolve())
    print(f"[write] {run_ba/'config.json'}  (scene -> scene_ba)")
    print(f"[link] {ckpt_link} -> {args.ba.resolve()}")
    print()
    print("Now render with the existing pipeline:")
    print(f"  sbatch render_st_blender.slurm {run_ba} {args.ckpt_name} <views> <down> <res> <out>")


if __name__ == "__main__":
    main()
