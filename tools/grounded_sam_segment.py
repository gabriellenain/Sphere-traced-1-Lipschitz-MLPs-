"""Text-prompted foreground masks via Grounded-SAM (GroundingDINO + SAM).

For each RGB image:
  1. GroundingDINO with text prompt (e.g. "a barn.") -> N detection boxes.
  2. SAM segments each box and we union all resulting masks.
  3. Save uint8 PNG (255 = foreground, 0 = background) to scene/mask/<stem>.png,
     or to --out-dir when --image-dir is used.

Also writes a 4-view overview contact-sheet to scene/mask/_overview.png.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          SamProcessor, SamModel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path)
    ap.add_argument("--image-dir", type=Path,
                    help="override input image directory; otherwise uses <scene>/rgb")
    ap.add_argument("--out-dir", type=Path,
                    help="override mask output directory; otherwise uses <scene>/mask")
    ap.add_argument("--glob", type=str, default=None,
                    help="input glob relative to image dir; defaults to 0_*.png for scene/rgb, else all common image extensions")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N images after sorting")
    ap.add_argument("--prompt", type=str, action="append",
                    help="text prompt; can be passed multiple times and masks are unioned")
    ap.add_argument("--box-threshold",  type=float, default=0.40)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--dino", type=str, default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam",  type=str, default="facebook/sam-vit-large")
    ap.add_argument("--keep-top", type=int, default=1,
                    help="keep only the N highest-confidence detection boxes per image")
    ap.add_argument("--box-expand", type=float, default=0.0,
                    help="expand each detection box by this fraction of its width/height before SAM")
    ap.add_argument("--largest-cc", action=argparse.BooleanOptionalAction, default=True,
                    help="restrict SAM mask union to its single largest connected component")
    ap.add_argument("--fill-holes", action="store_true",
                    help="fill enclosed holes in the final foreground mask")
    ap.add_argument("--close-px", type=int, default=0,
                    help="binary closing radius in pixels for tiny gaps/noise before saving")
    ap.add_argument("--min-cc-area", type=int, default=0,
                    help="remove connected foreground components smaller than this many pixels")
    ap.add_argument("--overview-views", type=int, nargs="+", default=[0, 96, 192, 288])
    args = ap.parse_args()
    prompts = args.prompt or ["a barn."]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[gsam] loading detector  {args.dino}")
    dino_proc = AutoProcessor.from_pretrained(args.dino)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino).to(device).eval()
    print(f"[gsam] loading segmenter {args.sam}")
    sam_proc = SamProcessor.from_pretrained(args.sam)
    sam = SamModel.from_pretrained(args.sam).to(device).eval()

    if args.scene is None and (args.image_dir is None or args.out_dir is None):
        raise ValueError("provide --scene, or both --image-dir and --out-dir")
    image_dir = args.image_dir if args.image_dir is not None else args.scene / "rgb"
    mask_dir = args.out_dir if args.out_dir is not None else args.scene / "mask"
    mask_dir.mkdir(exist_ok=True)
    if args.glob is not None:
        rgb_paths = sorted(image_dir.glob(args.glob))
    elif args.image_dir is None:
        rgb_paths = sorted(image_dir.glob("0_*.png"))
    else:
        rgb_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )
    if args.limit > 0:
        rgb_paths = rgb_paths[:args.limit]
    print(f"[gsam] {len(rgb_paths)} images  prompts={prompts!r}")

    overview = []
    n_empty = 0
    t0 = time.time()
    for i, ip in enumerate(rgb_paths):
        img = Image.open(ip).convert("RGB")
        W, H = img.size

        # GroundingDINO: text -> boxes. Multiple prompts are kept independently
        # so a high-confidence base/fountain box cannot suppress the statue box.
        prompt_boxes = []
        for prompt in prompts:
            inputs = dino_proc(images=img, text=prompt,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = dino(**inputs)
            det = dino_proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[(H, W)],
            )[0]
            boxes_p = det["boxes"]                       # (N, 4) xyxy
            scores = det.get("scores")
            if scores is not None and len(boxes_p) > args.keep_top:
                top = torch.topk(scores, args.keep_top).indices
                boxes_p = boxes_p[top]
            if len(boxes_p) > 0:
                prompt_boxes.append(boxes_p)
        boxes = torch.cat(prompt_boxes, dim=0) if prompt_boxes else torch.empty((0, 4), device=device)
        if len(boxes) > 0 and args.box_expand > 0:
            wh = boxes[:, 2:4] - boxes[:, 0:2]
            delta = 0.5 * args.box_expand * wh
            boxes = torch.cat([boxes[:, 0:2] - delta, boxes[:, 2:4] + delta], dim=1)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H - 1)

        if len(boxes) == 0:
            n_empty += 1
            mask = np.zeros((H, W), dtype=np.uint8)
        else:
            sam_inputs = sam_proc(img, input_boxes=[boxes.cpu().tolist()],
                                  return_tensors="pt").to(device)
            with torch.no_grad():
                sam_out = sam(**sam_inputs, multimask_output=False)
            sam_masks = sam_proc.image_processor.post_process_masks(
                sam_out.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu(),
            )[0]                                          # (N_boxes, 1, H, W) bool
            mask_bool = sam_masks.any(dim=0).squeeze(0).numpy()
            if args.largest_cc and mask_bool.any():
                from scipy.ndimage import label
                lab, n_cc = label(mask_bool)
                if n_cc > 1:
                    sizes = np.bincount(lab.ravel())
                    sizes[0] = 0          # background
                    keep_id = int(sizes.argmax())
                    mask_bool = (lab == keep_id)
            if (args.fill_holes or args.close_px > 0) and mask_bool.any():
                from scipy import ndimage as ndi
                if args.close_px > 0:
                    yy, xx = np.ogrid[-args.close_px:args.close_px + 1,
                                      -args.close_px:args.close_px + 1]
                    struct = (xx * xx + yy * yy) <= args.close_px * args.close_px
                    mask_bool = ndi.binary_closing(mask_bool, structure=struct)
                if args.fill_holes:
                    mask_bool = ndi.binary_fill_holes(mask_bool)
            if args.min_cc_area > 0 and mask_bool.any():
                from scipy.ndimage import label
                lab, n_cc = label(mask_bool)
                if n_cc > 0:
                    sizes = np.bincount(lab.ravel())
                    keep = sizes >= args.min_cc_area
                    keep[0] = False
                    mask_bool = keep[lab]
            mask = (mask_bool.astype(np.uint8)) * 255

        imageio.imwrite(mask_dir / (ip.stem + ".png"), mask)
        if i in args.overview_views:
            overview.append((ip.stem, np.asarray(img), mask, len(boxes)))
        if (i + 1) % 50 == 0 or (i + 1) == len(rgb_paths):
            dt = time.time() - t0
            print(f"  [gsam] {i+1}/{len(rgb_paths)}  "
                  f"({dt/(i+1)*1000:.0f} ms/img, empty={n_empty})", flush=True)

    print(f"[gsam] {n_empty}/{len(rgb_paths)} images had zero detections")

    if overview:
        fig, axes = plt.subplots(len(overview), 3,
                                 figsize=(12, 3.5 * len(overview)),
                                 squeeze=False)
        for r, (stem, rgb, msk, nb) in enumerate(overview):
            axes[r, 0].imshow(rgb)
            axes[r, 0].set_title(f"{stem}  ({nb} box{'es' if nb != 1 else ''})", fontsize=9)
            axes[r, 0].axis("off")
            axes[r, 1].imshow(msk, cmap="gray")
            axes[r, 1].set_title(f"fg mask ({(msk > 0).mean():.0%})", fontsize=9)
            axes[r, 1].axis("off")
            overlay = rgb.copy().astype(np.float32)
            overlay[msk == 0] = overlay[msk == 0] * 0.4 + np.array([200, 30, 30]) * 0.6
            axes[r, 2].imshow(overlay.clip(0, 255).astype(np.uint8))
            axes[r, 2].set_title("non-fg tinted red", fontsize=9)
            axes[r, 2].axis("off")
        fig.suptitle(f"Grounded-SAM masks  prompts={prompts!r}", fontsize=11)
        fig.tight_layout()
        out = mask_dir / "_overview.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[gsam] overview -> {out}")


if __name__ == "__main__":
    main()
