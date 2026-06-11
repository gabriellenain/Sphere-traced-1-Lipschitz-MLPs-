#!/usr/bin/env python3
"""Precompute frozen MASt3R feature maps for the 1-Lip trainer.

This is intentionally standalone: it only extracts and saves feature tensors.
Training can later load the resulting .pt file and use feature cosine loss in
place of, or in addition to, RGB photometric loss.

Example:
    python precompute_mast3r_features.py \
        --dataset skull \
        --model naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
        --out outputs/skull_mast3r_feats.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lip_tracer.config import BLENDER_SCENE, SCENE
from lip_tracer.data import load_blender_views, load_views


def _load_mast3r(model_name: str, device: str):
    try:
        from mast3r.model import AsymmetricMASt3R
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MASt3R is not importable in this environment. Install MASt3R/DUST3R "
            "in the active venv, or run this script from an environment where "
            "`from mast3r.model import AsymmetricMASt3R` works."
        ) from exc

    model = AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()
    return model


def _to_mast3r_input(images: torch.Tensor, size: int | None) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert repo images (V,H,W,3) in [0,1] to MASt3R tensors in [-1,1]."""
    imgs = images.permute(0, 3, 1, 2).float()
    h0, w0 = imgs.shape[-2:]
    if size is not None and max(h0, w0) != size:
        scale = size / max(h0, w0)
        h1 = max(16, int(round(h0 * scale / 16)) * 16)
        w1 = max(16, int(round(w0 * scale / 16)) * 16)
        imgs = F.interpolate(imgs, size=(h1, w1), mode="bilinear", align_corners=False)
    else:
        h1 = max(16, (h0 // 16) * 16)
        w1 = max(16, (w0 // 16) * 16)
        if (h1, w1) != (h0, w0):
            imgs = F.interpolate(imgs, size=(h1, w1), mode="bilinear", align_corners=False)
    imgs = imgs * 2.0 - 1.0
    return imgs.contiguous(), (h0, w0)


def _flatten_tensors(obj: Any) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(_flatten_tensors(value))
    elif isinstance(obj, (tuple, list)):
        for value in obj:
            out.extend(_flatten_tensors(value))
    return out


def _as_feature_map(t: torch.Tensor, h_img: int, w_img: int) -> torch.Tensor | None:
    """Return feature maps as (B,C,Hf,Wf), or None if tensor is not suitable."""
    if t.ndim == 4:
        # Accept either (B,C,H,W) or (B,H,W,C).
        if t.shape[1] >= 8 and t.shape[1] >= t.shape[-1]:
            return t.float()
        if t.shape[-1] >= 8:
            return t.permute(0, 3, 1, 2).float()
        return None
    if t.ndim != 3:
        return None

    b, n, c = t.shape
    if c < 8:
        return None
    # MASt3R/CroCo encoder tokens are normally one per 16x16 patch.
    h16, w16 = h_img // 16, w_img // 16
    if n == h16 * w16:
        return t.reshape(b, h16, w16, c).permute(0, 3, 1, 2).float()

    # Some ViT outputs include a CLS token. Drop it if that makes the grid fit.
    if n - 1 == h16 * w16:
        return t[:, 1:].reshape(b, h16, w16, c).permute(0, 3, 1, 2).float()
    return None


@torch.no_grad()
def _encode_batch(model, imgs: torch.Tensor, device: str) -> torch.Tensor:
    imgs = imgs.to(device)
    true_shape = torch.tensor([[imgs.shape[-2], imgs.shape[-1]]] * imgs.shape[0],
                              device=device)

    if hasattr(model, "_encode_image"):
        try:
            encoded = model._encode_image(imgs, true_shape)
        except TypeError:
            encoded = model._encode_image(imgs)
    else:
        raise RuntimeError("MASt3R model has no _encode_image method; this script expects the MASt3R/DUST3R API.")

    for tensor in _flatten_tensors(encoded):
        fmap = _as_feature_map(tensor, imgs.shape[-2], imgs.shape[-1])
        if fmap is not None and fmap.shape[0] == imgs.shape[0]:
            return F.normalize(fmap, dim=1).cpu()
    raise RuntimeError("Could not find an encoder tensor that can be reshaped into a feature map.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute MASt3R feature maps.")
    ap.add_argument("--dataset", choices=["skull", "lego"], default="skull")
    ap.add_argument("--scene", type=Path, default=None)
    ap.add_argument("--model", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=512,
                    help="resize longest image side before encoding; use 0 to keep/crop to multiple of 16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    scene = args.scene or (BLENDER_SCENE if args.dataset == "lego" else SCENE)
    views = load_blender_views(scene=scene, down=1) if args.dataset == "lego" else load_views(scene)
    imgs, orig_hw = _to_mast3r_input(views["images"], None if args.size <= 0 else args.size)

    model = _load_mast3r(args.model, args.device)
    feats = []
    for start in range(0, imgs.shape[0], args.batch):
        end = min(start + args.batch, imgs.shape[0])
        fmap = _encode_batch(model, imgs[start:end], args.device)
        feats.append(fmap)
        print(f"encoded views {start:03d}-{end - 1:03d}: {tuple(fmap.shape)}")
    feat_maps = torch.cat(feats, dim=0).contiguous()

    out = args.out or (Path("outputs") / f"{args.dataset}_mast3r_features.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": feat_maps.half(),
        "feature_hw": tuple(feat_maps.shape[-2:]),
        "encoded_hw": tuple(imgs.shape[-2:]),
        "orig_hw": orig_hw,
        "scene": str(scene),
        "dataset": args.dataset,
        "model": args.model,
        "c2w": views["c2w"].cpu(),
        "K": views["K"].cpu(),
    }
    torch.save(payload, out)
    print(f"saved {tuple(feat_maps.shape)} features -> {out}")
    print(f"uv scale image->feature: sx={feat_maps.shape[-1] / imgs.shape[-1]:.6f}, "
          f"sy={feat_maps.shape[-2] / imgs.shape[-2]:.6f}")


if __name__ == "__main__":
    main()
