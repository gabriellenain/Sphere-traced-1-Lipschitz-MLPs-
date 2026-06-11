"""Minimal MVSFormer++ feature extractor for MVSDF-style feature-consistency loss.

Returns per-view multi-scale feature maps (stages 1..4) from the pretrained
DINOv2-B + FPN + FMT pipeline, *before* cost-volume regularisation. Sample
these at re-projected surface points and minimise cross-view L1.

Setup
-----
1. git clone https://github.com/maybeLx/MVSFormerPlusPlus  (default: ~/scratch)
2. DINOv2-B backbone -> ``MVSFormerPlusPlus/pretrained_models/dinov2_vitb14_pretrain.pth``
3. MVSFormer++ checkpoint (``model_best.pth``) from the repo's OneDrive link.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_DIR = Path(os.environ.get("MVSFORMER_REPO",
                               Path.home() / "scratch" / "MVSFormerPlusPlus"))
CKPT_DIR    = Path(__file__).parent / "dtu_ckpt"
DEFAULT_CKPT   = CKPT_DIR / "model_best.pth"
DEFAULT_CONFIG = CKPT_DIR / "config.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _load_model_args(config_path: Path = DEFAULT_CONFIG) -> dict:
    args = json.loads(Path(config_path).read_text())["arch"]["args"]
    args["vit_path"] = str(REPO_DIR / "pretrained_models" / "dinov2_vitb14_pretrain.pth")
    return args


class MVSFormerFeatures(nn.Module):
    """Wraps MVSFormer++ feature extraction (DINOv2-B + FPN + FMT)."""

    def __init__(self, ckpt_path: str | Path = DEFAULT_CKPT, device: str = "cuda"):
        super().__init__()
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))
        from models.networks.DINOv2_mvsformer_model import DINOv2MVSNet

        self.net = DINOv2MVSNet(_load_model_args())
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.net.load_state_dict(state, strict=False)
        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1))
        self.rescale = self.net.vit_args["rescale"]

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        """imgs: (V, 3, H, W) in [0, 1]. Returns {stage1..4: (V, C, h, w)}."""
        V, _, H, W = imgs.shape
        x = (imgs - self.mean.to(imgs.device)) / self.std.to(imgs.device)

        vit_h = int(H * self.rescale // 14 * 14)
        vit_w = int(W * self.rescale // 14 * 14)

        conv01, conv11, conv21, conv31 = self.net.encoder(x)
        vit_imgs = F.interpolate(x, (vit_h, vit_w), mode="bicubic", align_corners=False)
        vit_feat = self.net.vit_forward(vit_imgs, B=1, V=V, vit_h=vit_h, vit_w=vit_w)
        if vit_feat.shape[-2:] != conv31.shape[-2:]:
            vit_feat = F.interpolate(vit_feat, size=conv31.shape[-2:],
                                     mode="bilinear", align_corners=False)
        conv31 = conv31 + vit_feat.squeeze(0) if vit_feat.dim() == 5 else conv31 + vit_feat
        f1, f2, f3, f4 = self.net.decoder(conv01, conv11, conv21, conv31)

        feats = {"stage1": f1.unsqueeze(0), "stage2": f2.unsqueeze(0),
                 "stage3": f3.unsqueeze(0), "stage4": f4.unsqueeze(0)}  # add B=1
        feats = self.net.FMT_module(feats)
        return {k: v.squeeze(0) for k, v in feats.items()}             # (V, C, h, w)


def sample_features(feat: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample a feature map at pixel coords of the *input* image.

    feat: (V, C, h, w).  uv: (V, N, 2) in input-image pixels. Returns (V, N, C).
    """
    V, C, h, w = feat.shape
    norm = uv.clone()
    norm[..., 0] = norm[..., 0] / (uv.new_tensor(w).clamp(min=1)) * 2 - 1
    norm[..., 1] = norm[..., 1] / (uv.new_tensor(h).clamp(min=1)) * 2 - 1
    grid = norm.unsqueeze(2)                                            # (V,N,1,2)
    out = F.grid_sample(feat, grid, mode="bilinear",
                        padding_mode="border", align_corners=False)
    return out.squeeze(-1).permute(0, 2, 1)                             # (V,N,C)
