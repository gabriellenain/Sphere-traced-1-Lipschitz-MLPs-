#!/usr/bin/env python3
"""Generate MASt3R depth maps for an IDR-style DTU scan.

The MASt3R/DUST3R optimizer works at its model input resolution, so this
script upsamples the optimized z-depth maps back to the requested target
resolution before saving them.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)


MAST3R_ROOT = Path("/scratch/_projets_/willow/1-lip-tracer/mast3r")
if MAST3R_ROOT.exists():
    sys.path.insert(0, str(MAST3R_ROOT))
    sys.path.insert(0, str(MAST3R_ROOT / "dust3r"))


def _load_mast3r(weights: str, device: str):
    try:
        from mast3r.model import AsymmetricMASt3R
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MASt3R is not importable. Run with the project venv at "
            "/scratch/_projets_/willow/1-lip-tracer/.venv, or install MASt3R."
        ) from exc
    return AsymmetricMASt3R.from_pretrained(weights).to(device).eval()


def _save_depth_png(path: Path, depth: np.ndarray, scale: float) -> None:
    depth_u16 = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_u16 = np.clip(depth_u16 * scale, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(path), depth_u16)


def _save_vis_png(path: Path, depth: np.ndarray, valid: np.ndarray) -> None:
    vals = depth[valid & np.isfinite(depth)]
    if vals.size == 0:
        vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        lo, hi = np.percentile(vals, [2.0, 98.0])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[~valid] = 0
    cv2.imwrite(str(path), vis)


@torch.no_grad()
def _inference_flush(pairs, model, device: str, batch_size: int, verbose: bool = True):
    """Run DUST3R inference while aggressively returning cached VRAM.

    At 1600x1200 the attention allocations are close to the limit of 22 GB
    GPUs. The stock inference loop can leave enough reserved memory between
    pairs to OOM on the next pair, so this mirrors it with explicit cleanup.
    """
    import gc
    import tqdm
    from dust3r.inference import check_if_same_size, loss_of_one_batch
    from dust3r.utils.device import collate_with_cat, to_cpu

    if verbose:
        print(f">> Inference with model on {len(pairs)} image pairs (flush-cache mode)", flush=True)
    result = []
    multiple_shapes = not check_if_same_size(pairs)
    if multiple_shapes:
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))
        del res
        gc.collect()
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    return collate_with_cat(result, lists=multiple_shapes)


def _load_idr_cameras(scene_path: Path, n_imgs: int, imshapes: list, first_img_path: Path):
    """Read cameras.npz and return cam2world (4x4), focal (px), pp (px) at processing resolution.

    Poses are in the IDR *normalised* world frame (unit-sphere), which is the
    coordinate system used both by the 1-Lip training code and by the depth
    values the global aligner will output after presetting these cameras.
    """
    from scipy.linalg import rq
    import PIL.Image

    cam_dict = np.load(scene_path / "cameras.npz")

    # Original image resolution — read from disk once (all views share the same size)
    W_orig, H_orig = PIL.Image.open(first_img_path).size  # (width, height)

    cam2worlds, focals, pps = [], [], []
    for i, (H_proc, W_proc) in enumerate(imshapes):
        # Projection matrix in IDR normalised world coords → original pixel coords
        P_metric = cam_dict[f"world_mat_{i}"][:3, :4].astype(np.float64)
        scale_mat = cam_dict[f"scale_mat_{i}"].astype(np.float64)
        P_norm = P_metric @ scale_mat  # 3×4

        # RQ decomposition → K (pixel, orig res) and R
        K, R = rq(P_norm[:, :3])
        sign = np.sign(np.diag(K)); sign[sign == 0] = 1.0
        K = K @ np.diag(sign); R = np.diag(sign) @ R
        if np.linalg.det(R) < 0:
            K[:, 2] *= -1; R[2, :] *= -1
        K /= K[2, 2]

        # Translation and camera centre in normalised world coords
        t_norm = np.linalg.solve(K, P_norm[:, 3])
        cam_center = -R.T @ t_norm

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.T.astype(np.float32)
        c2w[:3, 3]  = cam_center.astype(np.float32)
        cam2worlds.append(c2w)

        # Scale intrinsics to the MASt3R processing resolution
        sx = W_proc / W_orig
        sy = H_proc / H_orig
        fx = float(K[0, 0] * sx)
        fy = float(K[1, 1] * sy)
        cx = float(K[0, 2] * sx)
        cy = float(K[1, 2] * sy)

        focals.append(float(np.sqrt(fx * fy)))  # geometric mean (optimizer uses one focal)
        pps.append([cx, cy])

    return cam2worlds, focals, pps


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute MASt3R depth maps for DTU IDR scans.")
    ap.add_argument("--scene", type=Path, default=Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu_idr/scan122"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--weights", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--target-width", type=int, default=1600)
    ap.add_argument("--target-height", type=int, default=1200)
    ap.add_argument("--scene-graph", default="logwin-4",
                    help="MASt3R/DUST3R pair graph, e.g. complete, swin-5, logwin-4, oneref-0.")
    ap.add_argument("--prefilter", default=None,
                    help="Optional pair prefilter understood by MASt3R, e.g. cyc3 or seq4.")
    ap.add_argument("--no-symmetrize", action="store_true",
                    help="Do not add reverse duplicate image pairs; useful at high input resolution.")
    ap.add_argument("--flush-cache", action="store_true",
                    help="Use a slower inference loop that clears CUDA cache between pair batches.")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="Debug/probe mode: keep only the first N pairs after graph construction.")
    ap.add_argument("--max-images", type=int, default=0,
                    help="Debug/probe mode: load only the first N images.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--niter", type=int, default=300)
    ap.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--min-conf-thr", type=float, default=3.0)
    ap.add_argument("--clean-depth", action="store_true")
    ap.add_argument("--use-idr-mask", action="store_true",
                    help="Intersect MASt3R confidence masks with scene/mask/*.png DTU object masks.")
    ap.add_argument("--use-known-cameras", action="store_true",
                    help="Preset camera poses/intrinsics from cameras.npz before global alignment. "
                         "Locks the coordinate frame to the IDR normalised world and eliminates "
                         "scale drift. Requires scene/cameras.npz (standard for DTU IDR).")
    ap.add_argument("--png-scale", type=float, default=1000.0,
                    help="Multiplier for 16-bit PNG depth export. NPY files always keep float32 depth.")
    args = ap.parse_args()

    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from mast3r.image_pairs import make_pairs

    img_dir = args.scene / "image"
    image_paths = sorted(p for p in img_dir.glob("*.png") if not p.name.startswith("._"))
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"no PNG images found under {img_dir}")

    out_dir = args.out_dir or (args.scene / "mast3r_depth_1600x1200")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"scene:          {args.scene}", flush=True)
    print(f"images:         {len(image_paths)}", flush=True)
    print(f"out:            {out_dir}", flush=True)
    print(f"target res:     {args.target_width}x{args.target_height}", flush=True)
    print(f"scene graph:    {args.scene_graph}", flush=True)
    print(f"known cameras:  {args.use_known_cameras}", flush=True)
    print(f"weights:        {args.weights}", flush=True)

    model = _load_mast3r(args.weights, args.device)
    square_ok = bool(getattr(model, "square_ok", False))
    imgs = load_images([str(p) for p in image_paths], size=args.image_size,
                       verbose=True, patch_size=model.patch_size, square_ok=square_ok)
    pairs = make_pairs(imgs, scene_graph=args.scene_graph, prefilter=args.prefilter,
                       symmetrize=not args.no_symmetrize)
    pairs = sorted(pairs, key=lambda pair: (int(pair[0]["idx"]), int(pair[1]["idx"])))
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
    print(f"pairs:          {len(pairs)}", flush=True)

    if args.flush_cache:
        output = _inference_flush(pairs, model, args.device, batch_size=args.batch, verbose=True)
    else:
        output = inference(pairs, model, args.device, batch_size=args.batch, verbose=True)
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=args.device, mode=mode, verbose=True)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        if args.use_known_cameras:
            cam_npz = args.scene / "cameras.npz"
            if not cam_npz.exists():
                raise FileNotFoundError(f"--use-known-cameras requires {cam_npz}")
            cam2worlds, focals, pps = _load_idr_cameras(
                args.scene, len(imgs), scene.imshapes, image_paths[0])
            print("Presetting known cameras from cameras.npz ...", flush=True)
            for i, (c2w, f, pp) in enumerate(zip(cam2worlds, focals, pps)):
                print(f"  cam {i:3d}: focal={f:.1f}px  pp=({pp[0]:.1f},{pp[1]:.1f})  "
                      f"center=({c2w[0,3]:.3f},{c2w[1,3]:.3f},{c2w[2,3]:.3f})", flush=True)
            scene.preset_pose(cam2worlds)
            scene.preset_focal(focals)
            # preset_principal_point requires im_pp.requires_grad=True (optimize_pp mode).
            # Since we use optimize_pp=False (default), directly overwrite the _pp buffer
            # instead. get_principal_points() = _pp + 10*im_pp, and im_pp is frozen at 0.
            scene._pp.data[:] = torch.tensor(pps, dtype=torch.float32, device=args.device)
            init_mode = "known_poses"
        else:
            init_mode = "mst"

        loss = scene.compute_global_alignment(init=init_mode, niter=args.niter,
                                              schedule=args.schedule, lr=0.01)
        print(f"alignment loss: {float(loss):.6f}", flush=True)

    if args.clean_depth:
        scene = scene.clean_pointcloud()
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(args.min_conf_thr, device=args.device)))

    depths = [d.detach().cpu().numpy().astype(np.float32) for d in scene.get_depthmaps()]
    confs = [c.detach().cpu().numpy().astype(np.float32) for c in scene.im_conf]
    masks = [m.detach().cpu().numpy().astype(bool) for m in scene.get_masks()]

    manifest = {
        "scene": str(args.scene),
        "weights": args.weights,
        "image_size": args.image_size,
        "target_width": args.target_width,
        "target_height": args.target_height,
        "scene_graph": args.scene_graph,
        "niter": args.niter,
        "schedule": args.schedule,
        "min_conf_thr": args.min_conf_thr,
        "clean_depth": args.clean_depth,
        "prefilter": args.prefilter,
        "symmetrize": not args.no_symmetrize,
        "use_idr_mask": args.use_idr_mask,
        "use_known_cameras": args.use_known_cameras,
        "flush_cache": args.flush_cache,
        "max_pairs": args.max_pairs,
        "max_images": args.max_images,
        "png_scale": args.png_scale,
        "frames": [],
    }

    mask_paths = sorted(p for p in (args.scene / "mask").glob("*.png") if not p.name.startswith("._"))
    for idx, (src, depth, conf, valid) in enumerate(zip(image_paths, depths, confs, masks)):
        stem = src.stem
        d_hi = cv2.resize(depth, (args.target_width, args.target_height), interpolation=cv2.INTER_LINEAR)
        c_hi = cv2.resize(conf, (args.target_width, args.target_height), interpolation=cv2.INTER_LINEAR)
        v_hi = cv2.resize(valid.astype(np.uint8), (args.target_width, args.target_height),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        if args.use_idr_mask and idx < len(mask_paths):
            obj = cv2.imread(str(mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            if obj is not None:
                obj = cv2.resize(obj, (args.target_width, args.target_height),
                                 interpolation=cv2.INTER_NEAREST) > 0
                v_hi &= obj
        d_hi = np.where(v_hi, d_hi, 0.0).astype(np.float32)

        np.save(out_dir / f"{stem}_depth.npy", d_hi)
        np.save(out_dir / f"{stem}_conf.npy", c_hi.astype(np.float32))
        cv2.imwrite(str(out_dir / f"{stem}_valid.png"), (v_hi.astype(np.uint8) * 255))
        _save_depth_png(out_dir / f"{stem}_depth_u16.png", d_hi, args.png_scale)
        _save_vis_png(out_dir / f"{stem}_depth_vis.png", d_hi, v_hi)
        manifest["frames"].append({
            "image": str(src.relative_to(args.scene)),
            "depth_npy": f"{stem}_depth.npy",
            "confidence_npy": f"{stem}_conf.npy",
            "valid_mask": f"{stem}_valid.png",
            "depth_u16_png": f"{stem}_depth_u16.png",
            "depth_vis_png": f"{stem}_depth_vis.png",
            "shape": [int(args.target_height), int(args.target_width)],
        })

        vals = d_hi[v_hi & np.isfinite(d_hi)]
        if vals.size:
            print(f"{stem}: valid={v_hi.mean():.1%} depth=[{vals.min():.4f}, {vals.max():.4f}]",
                  flush=True)
        else:
            print(f"{stem}: no valid depth", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {len(depths)} depth maps -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
