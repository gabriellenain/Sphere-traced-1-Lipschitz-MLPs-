"""Render full-resolution reference and novel views from a checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from .config import BLENDER_SCENE, OUT_DIR, SCENE, TraceConfig
from .data import load_blender_views, load_colmap_points, load_views
from .model import FTheta, make_model
from .sphere_tracing import trace_nograd


def _load_checkpoint(
    ckpt_path: Path,
    device: str,
    hidden_override: int | None = None,
    depth_override: int | None = None,
    group_size_override: int | None = None,
    activation_override: str | None = None,
    input_encoding_override: str | None = None,
    multires_override: int | None = None,
) -> FTheta:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["f"]
    architecture = ckpt.get("architecture", "cpl")
    if hidden_override is not None:
        hidden = hidden_override
    elif architecture == "neus":
        hidden = state["layers.0.weight"].shape[0]
    elif "head_weight" in state:
        hidden = state["head_weight"].shape[0]
    else:
        hidden = next(
            v.shape[1] for k, v in state.items()
            if k.endswith(".weight") and v.ndim == 2 and v.shape[0] != 1
            and not k.startswith("encoder")
        )
    if depth_override is not None:
        depth = depth_override
    elif "depth" in ckpt:
        depth = ckpt["depth"]
    elif architecture == "neus":
        depth = sum(
            1
            for k in state
            if k.startswith("layers.") and k.endswith(".weight")
        )
    else:
        depth = sum(
            1
            for k in state
            if k.startswith("net.") and k.endswith(".weight") and "_u" not in k
        )
    group_size = group_size_override or ckpt.get("group_size", 2)
    activation = activation_override or ckpt.get("activation", "groupsort")
    input_encoding = (
        input_encoding_override if input_encoding_override is not None else ckpt.get("input_encoding", "identity")
    )
    multires = multires_override if multires_override is not None else ckpt.get("multires", 6)
    print(
        f"loading {ckpt_path}\n"
        f"  arch={architecture} hidden={hidden} depth={depth} group_size={group_size} activation={activation} "
        f"input_encoding={input_encoding} multires={multires}"
    )
    f = make_model(
        hidden=hidden,
        depth=depth,
        group_size=group_size,
        activation=activation,
        input_encoding=input_encoding,
        multires=multires,
        architecture=architecture,
    ).to(device)
    f.load_state_dict(state, strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    f.eval()
    return f


def _render_buffers(
    f: FTheta,
    c2w: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    device: str,
    trace_cfg: TraceConfig,
    ray_chunk: int,
    ss: int = 1,
) -> dict[str, np.ndarray]:
    # Supersample: cast ss*ss sub-pixel rays per pixel, average the shaded
    # buffers at the end (SSAA). Sub-pixel col c -> pixel coord (c + 0.5) / ss,
    # so c = x*ss + i lands at the center of sub-cell i within pixel x.
    Hs, Ws = H * ss, W * ss
    ys, xs = np.meshgrid(np.arange(Hs), np.arange(Ws), indexing="ij")
    d_cam = np.stack(
        [
            ((xs + 0.5) / ss - K[0, 2]) / K[0, 0],
            ((ys + 0.5) / ss - K[1, 2]) / K[1, 1],
            np.ones_like(xs, dtype=np.float64),
        ],
        axis=-1,
    )
    dirs = d_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape).copy().reshape(-1, 3)
    dirs = dirs.reshape(-1, 3)

    x_hit_all: list[torch.Tensor] = []
    hit_all: list[torch.Tensor] = []
    for start in range(0, origins.shape[0], ray_chunk):
        end = min(start + ray_chunk, origins.shape[0])
        o_t = torch.from_numpy(origins[start:end]).float().to(device)
        d_t = torch.from_numpy(dirs[start:end]).float().to(device)
        x_hit, _, hit = trace_nograd(f, o_t, d_t, trace_cfg)
        x_hit_all.append(x_hit)
        hit_all.append(hit)

    x_hit = torch.cat(x_hit_all, dim=0)
    hit = torch.cat(hit_all, dim=0)

    normals_all: list[torch.Tensor] = []
    for start in range(0, x_hit.shape[0], ray_chunk):
        end = min(start + ray_chunk, x_hit.shape[0])
        with torch.enable_grad():
            xr = x_hit[start:end].detach().clone().requires_grad_(True)
            n = torch.autograd.grad(f(xr).sum(), xr)[0]
        normals_all.append(n.detach())
    normals = torch.cat(normals_all, dim=0)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # World-space axes from c2w
    cam_right = torch.from_numpy(c2w[:3, 0].astype("float32")).to(device)
    cam_up    = torch.from_numpy(c2w[:3, 1].astype("float32")).to(device)
    cam_fwd   = torch.from_numpy(-c2w[:3, 2].astype("float32")).to(device)

    # 3-point lighting in world space
    key_light  = F.normalize( cam_fwd + 0.6 * cam_up + 0.3 * cam_right, dim=0)
    fill_light = F.normalize( cam_fwd - 0.2 * cam_up - 0.8 * cam_right, dim=0)
    back_light = F.normalize(-cam_fwd + 0.4 * cam_up,                   dim=0)

    # Surface-to-camera direction per pixel for specular
    cam_pos = torch.from_numpy(c2w[:3, 3].astype("float32")).to(device)
    view = F.normalize(cam_pos.unsqueeze(0) - x_hit.reshape(-1, 3), dim=-1)

    key_d  = (normals * key_light ).sum(-1, keepdim=True).clamp(min=0.0)
    fill_d = (normals * fill_light).sum(-1, keepdim=True).clamp(min=0.0)
    back_d = (normals * back_light).sum(-1, keepdim=True).clamp(min=0.0)

    # Blinn-Phong specular on key light
    half_key = F.normalize(key_light.unsqueeze(0) + view, dim=-1)
    spec = (normals * half_key).sum(-1, keepdim=True).clamp(min=0.0) ** 40

    albedo = torch.tensor([0.92, 0.90, 0.88], device=device)
    shading = (0.08                          # ambient
               + 0.70 * key_d               # key diffuse
               + 0.18 * fill_d              # fill diffuse
               + 0.10 * back_d              # rim
               + 0.25 * spec)               # specular highlight
    phong = (shading * albedo).clamp(0.0, 1.0)
    normals_rgb = (0.5 * (normals + 1.0)).clamp(0.0, 1.0)
    white = torch.ones_like(phong)
    phong = torch.where(hit.unsqueeze(-1), phong, white)
    normals_rgb = torch.where(hit.unsqueeze(-1), normals_rgb, white)

    # Box-downsample the supersampled buffers (H*ss, W*ss) -> (H, W).
    def _downsample(t: torch.Tensor) -> np.ndarray:
        c = t.shape[-1] if t.dim() == 2 else 1
        img = t.reshape(H, ss, W, ss, c) if c > 1 else t.reshape(H, ss, W, ss)
        img = img.float().mean(dim=(1, 3))
        return img.cpu().numpy()

    hit_cov = _downsample(hit.float())  # fractional pixel coverage in [0, 1]
    return {
        "hit": hit_cov > 0.5,
        "phong": _downsample(phong),
        "normals": _downsample(normals_rgb),
    }


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))


def _make_tile(label: str, img: np.ndarray, tile_size: tuple[int, int]) -> Image.Image:
    rgb = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    tile = Image.fromarray(rgb, mode="RGB")
    tile = ImageOps.contain(tile, tile_size)
    canvas = Image.new("RGB", (tile_size[0], tile_size[1] + 28), (245, 245, 245))
    canvas.paste(tile, ((tile_size[0] - tile.width) // 2, (tile_size[1] - tile.height) // 2))
    ImageDraw.Draw(canvas).text((8, tile_size[1] + 6), label, fill=(20, 20, 20))
    return canvas


def _save_contact_sheet(path: Path, labeled_images: list[tuple[str, np.ndarray]], cols: int = 4) -> None:
    if not labeled_images:
        return
    tile_size = (320, 320)
    rows = (len(labeled_images) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * tile_size[0], rows * (tile_size[1] + 28)),
        (245, 245, 245),
    )
    for idx, (label, img) in enumerate(labeled_images):
        tile = _make_tile(label, img, tile_size)
        x = (idx % cols) * tile.width
        y = (idx // cols) * tile.height
        sheet.paste(tile, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _look_at_c2w(
    eye: np.ndarray,
    target: np.ndarray,
    up_guess: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_guess)
    if np.linalg.norm(right) < 1e-6:
        up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_guess)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def _novel_orbit_poses(views: dict, use_blender: bool, n_views: int) -> list[np.ndarray]:
    cams = views["c2w"].numpy()[:, :3, 3]
    if use_blender:
        target = np.zeros(3, dtype=np.float32)
    else:
        target = load_colmap_points().numpy().mean(axis=0).astype(np.float32)
    rel = cams - target[None]
    radius_xy = np.linalg.norm(rel[:, :2], axis=1)
    radius = float(np.median(radius_xy[radius_xy > 1e-6])) if np.any(radius_xy > 1e-6) else 1.0
    height = float(np.median(rel[:, 2]))
    poses = []
    for i in range(n_views):
        theta = 2.0 * np.pi * i / n_views
        eye = target + np.array(
            [radius * np.cos(theta), radius * np.sin(theta), height],
            dtype=np.float32,
        )
        poses.append(_look_at_c2w(eye, target))
    return poses


def _novel_training_poses(views: dict, n_views: int) -> list[np.ndarray]:
    """Use evenly spaced training cameras as stable 'novel' viewpoints.

    For datasets like Blender, these poses already form a smooth orbit and are
    often more reliable than synthesizing a new orbit around an assumed center.
    """
    ids = _select_reference_ids(int(views["c2w"].shape[0]), n_views)
    return [views["c2w"][vi].numpy() for vi in ids]


def _select_reference_ids(total_views: int, n_views: int) -> list[int]:
    if n_views >= total_views:
        return list(range(total_views))
    return [int(round(i * (total_views - 1) / (n_views - 1))) for i in range(n_views)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Render reference and novel views from a checkpoint")
    ap.add_argument("--pt", type=Path, required=True, help="checkpoint path")
    ap.add_argument("--dataset", choices=["dtu", "skull", "lego"], default="lego")
    ap.add_argument("--scene", type=Path, default=None, help="override scene path")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--out-dir", type=Path, default=None, help="output directory")
    ap.add_argument("--ref-views", type=int, default=8, help="number of reference views to render")
    ap.add_argument("--novel-views", type=int, default=60, help="number of novel orbit views to render")
    ap.add_argument(
        "--novel-mode",
        choices=["train", "orbit"],
        default="train",
        help="novel pose source: sampled training cameras or synthesized orbit",
    )
    ap.add_argument("--ray-chunk", type=int, default=65536, help="rays per tracing chunk")
    ap.add_argument("--ss", type=int, default=1, help="SSAA factor: ss*ss sub-pixel rays per pixel, averaged (2-3 removes silhouette aliasing)")
    ap.add_argument("--trace-iters", type=int, default=64, help="sphere-tracing iterations")
    ap.add_argument("--eps", type=float, default=1e-3, help="sphere-tracing hit epsilon")
    ap.add_argument("--t-far", type=float, default=10.0, help="sphere-tracing far cutoff")
    ap.add_argument("--newton-steps", type=int, default=0, help="optional Newton refinement steps")
    ap.add_argument("--hidden", type=int, default=None, help="override checkpoint hidden width")
    ap.add_argument("--depth", type=int, default=None, help="override checkpoint depth")
    ap.add_argument("--group-size", type=int, default=None, help="override checkpoint group size")
    ap.add_argument(
        "--activation",
        choices=["groupsort", "nact"],
        default=None,
        help="override checkpoint activation",
    )
    ap.add_argument(
        "--input-encoding",
        choices=["identity", "pe"],
        default=None,
        help="override checkpoint input encoding",
    )
    ap.add_argument("--multires", type=int, default=None, help="override checkpoint positional encoding frequencies")
    args = ap.parse_args()

    use_blender = args.dataset == "lego"
    scene = args.scene or (BLENDER_SCENE if use_blender else SCENE)
    out_dir = args.out_dir or (OUT_DIR / f"renders_{args.pt.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trace_cfg = TraceConfig(
        iters=args.trace_iters,
        eps=args.eps,
        t_far=args.t_far,
        newton_steps=args.newton_steps,
    )
    f = _load_checkpoint(
        args.pt,
        device=device,
        hidden_override=args.hidden,
        depth_override=args.depth,
        group_size_override=args.group_size,
        activation_override=args.activation,
        input_encoding_override=args.input_encoding,
        multires_override=args.multires,
    )

    if use_blender:
        views = load_blender_views(scene=scene, split=args.split, down=1)
    else:
        views = load_views(scene=scene)
    H, W = views["H"], views["W"]
    print(f"rendering at dataset resolution {W}x{H}")

    ref_ids = _select_reference_ids(int(views["images"].shape[0]), args.ref_views)
    ref_sheet: list[tuple[str, np.ndarray]] = []
    for idx, vi in enumerate(ref_ids):
        print(f"[ref {idx + 1}/{len(ref_ids)}] view={vi}")
        K = views["K"][vi].numpy()
        c2w = views["c2w"][vi].numpy()
        gt = views["images"][vi].numpy()
        buffers = _render_buffers(f, c2w, K, H, W, device, trace_cfg, args.ray_chunk, args.ss)
        overlay = gt.copy()
        overlay[buffers["hit"]] = 0.5 * gt[buffers["hit"]] + 0.5 * buffers["normals"][buffers["hit"]]

        stem = f"ref_{vi:03d}"
        _write_png(out_dir / "reference" / f"{stem}_gt.png", gt)
        _write_png(out_dir / "reference" / f"{stem}_phong.png", buffers["phong"])
        _write_png(out_dir / "reference" / f"{stem}_normals.png", buffers["normals"])
        _write_png(out_dir / "reference" / f"{stem}_overlay.png", overlay)
        ref_sheet.append((stem, np.concatenate([gt, buffers["phong"], buffers["normals"], overlay], axis=1)))

    _save_contact_sheet(out_dir / "reference_sheet.png", ref_sheet, cols=2)

    K_novel = views["K"][0].numpy()
    novel_sheet: list[tuple[str, np.ndarray]] = []
    if args.novel_mode == "train":
        novel_poses = _novel_training_poses(views, n_views=args.novel_views)
    else:
        novel_poses = _novel_orbit_poses(views, use_blender=use_blender, n_views=args.novel_views)
    for idx, c2w in enumerate(novel_poses):
        print(f"[novel {idx + 1}/{len(novel_poses)}]")
        buffers = _render_buffers(f, c2w, K_novel, H, W, device, trace_cfg, args.ray_chunk, args.ss)
        hit_rate = float(buffers["hit"].mean())
        eye = c2w[:3, 3]
        eye_t = torch.from_numpy(eye[None]).float().to(device)
        with torch.no_grad():
            eye_sdf = float(f(eye_t).item())
        print(
            f"  eye=({eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f})"
            f"  f(eye)={eye_sdf:.4f}  hit_rate={hit_rate:.2%}"
        )
        stem = f"novel_{idx:03d}"
        _write_png(out_dir / "novel" / f"{stem}_phong.png", buffers["phong"])
        _write_png(out_dir / "novel" / f"{stem}_normals.png", buffers["normals"])
        if idx < 12:
            novel_sheet.append((stem, np.concatenate([buffers["phong"], buffers["normals"]], axis=1)))

    _save_contact_sheet(out_dir / "novel_sheet.png", novel_sheet, cols=3)
    print(f"renders written to {out_dir}")


if __name__ == "__main__":
    main()
