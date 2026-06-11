"""Coarse-to-fine refinement: freeze fθ, learn δψ on normal-deformed surface.

Uses the same photo_loss as training (nearest-neighbour views + occlusion test).
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from lip_tracer.data import load_views, load_blender_views, precompute_alt_cameras
from lip_tracer.deformation_field import DeformationField, deform_hits
from lip_tracer.loss import photo_loss, bilinear_sample
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt",        required=True,  help="coarse checkpoint (.pt)")
    ap.add_argument("--scene",     default=None,   help="scene directory (default: read from coarse config.json)")
    ap.add_argument("--out",       default="outputs/refined.pt")
    ap.add_argument("--steps",     type=int,   default=10_000)
    ap.add_argument("--batch",     type=int,   default=1024)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--delta-max", type=float, default=0.01)
    ap.add_argument("--hidden",    type=int,   default=256)
    ap.add_argument("--depth",     type=int,   default=8)
    ap.add_argument("--down",      type=int,   default=1)
    ap.add_argument("--blender",   action="store_true")
    ap.add_argument("--n-alt",     type=int,   default=4,   help="number of alternate views")
    ap.add_argument("--w-photo",      type=float, default=0.0)
    ap.add_argument("--w-ncc",        type=float, default=1.0)
    ap.add_argument("--ncc-patch",    type=int,   default=5,   help="NCC patch side length")
    ap.add_argument("--ncc-half-pix", type=float, default=2.0, help="NCC reprojection half-window (pixels)")
    ap.add_argument("--ncc-min",      type=float, default=0.0, help="PMVS NCC gate threshold")
    ap.add_argument("--w-reg",        type=float, default=1e-3, help="δ² regularisation weight")
    ap.add_argument("--cos-thresh",   type=float, default=0.1)
    ap.add_argument("--render-freq", type=int,   default=500)
    ap.add_argument("--out-dir",    default="outputs", help="root output directory")
    ap.add_argument("--tag",        default="",        help="extra tag for run dir name")
    return ap.parse_args()


def main() -> None:
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Resolve scene from coarse config.json if not given ──────────────────
    if args.scene is None:
        coarse_cfg = Path(args.pt).parent / "config.json"
        if not coarse_cfg.exists():
            raise FileNotFoundError(
                f"--scene not given and no config.json found next to {args.pt}"
            )
        args.scene = json.loads(coarse_cfg.read_text())["scene"]
        print(f"[info] scene from coarse config: {args.scene}")

    # ── Run directory (one per run, like train.py) ──────────────────────────
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scene   = Path(args.scene)
    tag     = args.tag or scene.name
    run_dir = Path(args.out_dir) / f"refine_{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out      = run_dir / "refined.pt"
    BEST_PT  = run_dir / "refined_best_photo.pt"
    best_photo = float("inf")

    cfg = vars(args)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"run dir → {run_dir}")

    # ── Load coarse field (frozen) ──────────────────────────────────────────
    ckpt = torch.load(args.pt, map_location="cpu")
    arch       = ckpt.get("architecture", "cpl")
    group_size = ckpt.get("group_size", 2)
    activation = ckpt.get("activation", "groupsort")
    input_enc  = ckpt.get("input_encoding", "identity")
    multires   = ckpt.get("multires", 4)
    if arch == "neus":
        hidden = ckpt["f"]["layers.0.weight"].shape[0]
        depth  = ckpt.get("depth", sum(1 for k in ckpt["f"]
                           if k.startswith("layers.") and k.endswith(".weight")))
    elif "head_weight" in ckpt["f"]:
        hidden = ckpt["f"]["head_weight"].shape[0]
        depth  = ckpt.get("depth", 8)
    else:
        hidden = next(v.shape[1] for k, v in ckpt["f"].items()
                      if k.endswith(".weight") and v.ndim == 2 and v.shape[0] != 1
                      and not k.startswith("encoder"))
        depth  = ckpt.get("depth", 8)

    f = make_model(hidden=hidden, depth=depth, group_size=group_size,
                   activation=activation, input_encoding=input_enc,
                   multires=multires, architecture=arch).to(device)
    f.load_state_dict(ckpt["f"], strict=False)
    f.eval()
    for p in f.parameters():
        p.requires_grad_(False)
    print(f"loaded coarse field: arch={arch} hidden={hidden} depth={depth}")

    # ── Deformation field ───────────────────────────────────────────────────
    psi = DeformationField(hidden=args.hidden, depth=args.depth,
                           delta_max=args.delta_max).to(device)
    opt = torch.optim.Adam(psi.parameters(), lr=args.lr)

    # ── Scene data ──────────────────────────────────────────────────────────
    scene = Path(args.scene)
    views = (load_blender_views(scene, down=args.down) if args.blender
             else load_views(scene, down=args.down))
    images   = views["images"].to(device)        # (V, H, W, 3)
    K_all    = views["K"].to(device)             # (V, 3, 3)
    c2w_all  = views["c2w"].to(device)           # (V, 4, 4)
    w2c_all  = torch.linalg.inv(c2w_all)
    masks    = views["masks"].to(device) if "masks" in views else None
    H, W     = views["H"], views["W"]
    V        = images.shape[0]
    origins  = c2w_all[:, :3, 3]                 # (V, 3)
    alt_nn   = precompute_alt_cameras(views, args.n_alt).to(device)  # (V, n_alt)

    print(f"scene: {V} views  H={H} W={W}  n_alt={args.n_alt}")
    print(f"δmax={args.delta_max}  steps={args.steps}  batch={args.batch}  lr={args.lr}")

    # ── Training loop ────────────────────────────────────────────────────────
    for step in range(args.steps):
        vi = torch.randint(0, V, (args.batch,), device=device)
        px = (torch.rand(args.batch, device=device) * W).clamp(0, W - 1)
        py = (torch.rand(args.batch, device=device) * H).clamp(0, H - 1)

        d_cam = torch.stack([
            (px - K_all[vi, 0, 2]) / K_all[vi, 0, 0],
            (py - K_all[vi, 1, 2]) / K_all[vi, 1, 1],
            torch.ones(args.batch, device=device),
        ], dim=-1)
        d_world = F.normalize(
            torch.einsum("bij,bj->bi", w2c_all[vi, :3, :3].transpose(-1, -2), d_cam), dim=-1
        )
        o = origins[vi]

        with torch.no_grad():
            x_coarse, _, hit = trace_nograd(f, o, d_world)

        if not hit.any():
            continue

        # deform hit points — gradient only through δψ
        x_psi, delta, n_theta = deform_hits(f, x_coarse[hit], psi)

        # uv of x_psi in source view (for photo_loss c_self sampling)
        xc_self  = torch.einsum("bij,bj->bi", w2c_all[vi[hit], :3, :3], x_psi) + w2c_all[vi[hit], :3, 3]
        uvh_self = torch.einsum("bij,bj->bi", K_all[vi[hit]], xc_self)
        uv_self  = uvh_self[:, :2] / uvh_self[:, 2:3].clamp(min=1e-6)

        # fg_self: pixels in mask (or all True if no masks)
        if masks is not None:
            uv_c = uv_self.long().clamp(0)
            uv_c[:, 0].clamp_(max=W - 1); uv_c[:, 1].clamp_(max=H - 1)
            fg_self = masks[vi[hit], uv_c[:, 1], uv_c[:, 0]]
        else:
            fg_self = hit.new_ones(hit.sum())

        ph, ph_stats = photo_loss(
            f,
            x_psi, hit.new_ones(hit.sum()),   # all deformed points are "hits"
            n_theta,
            vi[hit], alt_nn, origins,
            images, K_all, w2c_all,
            None,           # no feature maps
            masks, fg_self,
            H, W, uv_self,
            n_alt=args.n_alt, cos_thresh=args.cos_thresh,
            w_photo=args.w_photo, w_feature=0.0, w_ncc=args.w_ncc,
            ncc_patch=args.ncc_patch, ncc_half_pix=args.ncc_half_pix, ncc_min=args.ncc_min,
            sample_mode="bilinear", gaussian_sigma=0.8, gaussian_radius=2,
            step=step,
        )

        reg  = delta.pow(2).mean()
        loss = ph + args.w_reg * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  "
                  f"photo {ph.item():.4f}  reg {reg.item():.4f}  "
                  f"hits {hit.sum().item()}/{args.batch}  "
                  f"n_mask {ph_stats['n_mask']}  "
                  f"|δ| {delta.abs().mean().item():.5f}")

        # ── best photo checkpoint + render ──────────────────────────────────
        ph_val = ph.item()
        if ph_val > 1e-6 and ph_val < best_photo:
            best_photo = ph_val
            payload = {"psi": psi.state_dict(), "coarse_pt": args.pt,
                       "delta_max": args.delta_max, "step": step, "photo": ph_val}
            tmp = BEST_PT.with_suffix(".pt.tmp")
            torch.save(payload, tmp); tmp.replace(BEST_PT)
            print(f"  [best_photo@{step}] photo={ph_val:.4f} → {BEST_PT.name}")
            render_src = run_dir / f"render_{step:05d}.png"
            if render_src.exists():
                shutil.copy(render_src, run_dir / "render_best_photo.png")

        # ── periodic render on Sψ : trace g(x) = fθ(x) - δψ(x) ────────────
        if step % args.render_freq == 0:
            from lip_tracer.train import _render_poses

            class _GField(torch.nn.Module):
                def forward(self_, x):  # noqa: N805
                    return f(x) - psi(x)

            _render_poses(_GField().to(device), views, step, run_dir, device)

    payload = {"psi": psi.state_dict(), "coarse_pt": args.pt,
               "delta_max": args.delta_max, "step": args.steps}
    torch.save(payload, out)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
