"""Text-prompted (open-vocabulary concept) masks via SAM 3.

For each image, SAM 3's Promptable Concept Segmentation finds all instances of
the given noun-phrase prompt(s); we union them into one foreground mask.
Saves <out>/<stem>.png (255 fg / 0 bg) + a 5-view _overview.png.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import torch


def to_mask_array(masks):
    """Normalize SAM3 mask output to a list of bool (H,W) arrays."""
    if masks is None:
        return []
    if torch.is_tensor(masks):
        m = masks.detach().float().cpu().numpy()
    else:
        m = np.asarray(masks)
    m = np.squeeze(m)
    if m.ndim == 2:
        m = m[None]
    return [(mi > 0.5) for mi in m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prompt", action="append", required=True,
                    help="noun-phrase concept; repeat to union (e.g. --prompt statue --prompt pedestal)")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--largest-cc", action="store_true")
    ap.add_argument("--fill-holes", action="store_true")
    ap.add_argument("--overview", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # SAM3 runs in bfloat16 autocast (matches the official example notebook).
    if dev == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print(f"[sam3] building model on {dev} ...", flush=True)
    model = build_sam3_image_model(device=dev, load_from_HF=True)
    proc = Sam3Processor(model, confidence_threshold=args.score_thresh)
    print("[sam3] model ready", flush=True)

    paths = sorted(args.image_dir.glob(args.glob))
    print(f"[sam3] {len(paths)} images  prompts={args.prompt}", flush=True)
    overview = []
    for i, ip in enumerate(paths):
        img = Image.open(ip).convert("RGB")
        W, H = img.size
        state = proc.set_image(img)
        union = np.zeros((H, W), bool)
        ninst = 0
        for pr in args.prompt:
            out = proc.set_text_prompt(state=state, prompt=pr)
            masks, scores = out.get("masks"), out.get("scores")
            sc = (scores.detach().float().cpu().numpy().ravel() if torch.is_tensor(scores)
                  else (np.asarray(scores).ravel() if scores is not None else None))
            for j, mb in enumerate(to_mask_array(masks)):
                if sc is not None and j < len(sc) and sc[j] < args.score_thresh:
                    continue
                if mb.shape != (H, W):
                    mb = np.array(Image.fromarray(mb).resize((W, H), Image.NEAREST))
                union |= mb; ninst += 1
        if args.largest_cc and union.any():
            from scipy.ndimage import label
            lab, n = label(union)
            if n > 1:
                s = np.bincount(lab.ravel()); s[0] = 0; union = lab == s.argmax()
        if args.fill_holes and union.any():
            from scipy.ndimage import binary_fill_holes
            union = binary_fill_holes(union)
        imageio.imwrite(args.out_dir / f"{ip.stem}.png", (union.astype(np.uint8) * 255))
        if i in args.overview:
            overview.append((ip.stem, np.asarray(img), union, ninst))
        if (i + 1) % 25 == 0 or (i + 1) == len(paths):
            print(f"  [sam3] {i+1}/{len(paths)}", flush=True)

    if overview:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(overview), 3, figsize=(13, 4*len(overview)), squeeze=False)
        for r, (st, rgb, msk, n) in enumerate(overview):
            ax[r,0].imshow(rgb); ax[r,0].set_title(f"{st} ({n} inst)", fontsize=9); ax[r,0].axis('off')
            ax[r,1].imshow(msk, cmap='gray'); ax[r,1].set_title(f"mask ({msk.mean():.0%})", fontsize=9); ax[r,1].axis('off')
            ov = rgb.astype(np.float32).copy(); ov[~msk] = ov[~msk]*0.4 + np.array([200,30,30])*0.6
            ax[r,2].imshow(ov.clip(0,255).astype(np.uint8)); ax[r,2].set_title("overlay", fontsize=9); ax[r,2].axis('off')
        fig.suptitle(f"SAM 3 masks  prompts={args.prompt}", fontsize=11); fig.tight_layout()
        fig.savefig(args.out_dir / "_overview.png", dpi=120, bbox_inches='tight')
        print(f"[sam3] overview -> {args.out_dir/'_overview.png'}", flush=True)


if __name__ == "__main__":
    main()
