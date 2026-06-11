"""Shape-only probe for MVSFormer++. Skips DINOv2 (uses zero vit_feat)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/scratch/MVSFormerPlusPlus"))

import torch
from lip_tracer.MVS_former import _load_model_args
from models.networks.DINOv2_mvsformer_model import DINOv2MVSNet


def main():
    args = _load_model_args()
    args["vit_path"] = ""                      # skip ViT weight load
    net = DINOv2MVSNet(args).eval()
    for p in net.parameters(): p.requires_grad_(False)

    V, H, W = 2, 128, 192                      # tiny test
    imgs = torch.zeros(V, 3, H, W)

    conv01, conv11, conv21, conv31 = net.encoder(imgs)
    print("encoder strides:",
          [(t.shape[-2], t.shape[-1]) for t in (conv01, conv11, conv21, conv31)])

    vit_feat = torch.zeros_like(conv31)        # bypass DINOv2
    conv31 = conv31 + vit_feat
    f1, f2, f3, f4 = net.decoder(conv01, conv11, conv21, conv31)
    feats = {"stage1": f1.unsqueeze(0), "stage2": f2.unsqueeze(0),
             "stage3": f3.unsqueeze(0), "stage4": f4.unsqueeze(0)}
    feats = net.FMT_module(feats)
    for k, v in feats.items():
        v = v.squeeze(0)
        print(f"{k}: {tuple(v.shape)}  stride≈{H//v.shape[-2]}x{W//v.shape[-1]}")


if __name__ == "__main__":
    main()
