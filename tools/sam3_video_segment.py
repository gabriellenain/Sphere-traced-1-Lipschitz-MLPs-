"""SAM 3 VIDEO predictor: seed a concept on frame 0, propagate across the orbit.

Frames are treated as an ordered video (capture order). One text/concept prompt
on the first frame is tracked through all frames via SAM 3's memory — giving
cross-view-consistent masks (the property a visual hull wants).

Saves <out>/<orig_stem>.png masks + <out>/_overview.png.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", type=Path, required=True)
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prompt", action="append", required=True,
                    help="concept to track; repeat to union (each gets its own propagation)")
    ap.add_argument("--seed-frame", type=int, default=0)
    ap.add_argument("--overview", type=int, nargs="+", default=[0, 52, 105, 158, 210, 262])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    framedir = args.out_dir / "frames"; framedir.mkdir(exist_ok=True)

    rgb = sorted(args.rgb_dir.glob(args.glob))
    stems = [p.stem for p in rgb]
    W, H = Image.open(rgb[0]).size
    print(f"[sam3v] {len(rgb)} frames {W}x{H}; staging integer-named jpgs ...", flush=True)
    for i, p in enumerate(rgb):
        fp = framedir / f"{i}.jpg"
        if not fp.exists():
            Image.open(p).convert("RGB").save(fp, quality=95)

    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(gpus_to_use=[0])
    print("[sam3v] predictor built; starting session ...", flush=True)
    sid = predictor.handle_request(dict(type="start_session", resource_path=str(framedir)))["session_id"]

    # union one propagation per concept (text query needs a session reset to switch)
    def collect_binary(o):
        if o is None or "out_binary_masks" not in o:
            return np.zeros((H, W), bool), 0
        bm = o["out_binary_masks"]
        bm = bm.detach().float().cpu().numpy() if torch.is_tensor(bm) else np.asarray(bm)
        bm = np.squeeze(bm)
        if bm.ndim == 2: bm = bm[None]
        fg = np.zeros((H, W), bool); n = 0
        for m in bm:
            mb = m > 0.5
            if mb.shape != (H, W):
                mb = np.array(Image.fromarray(mb).resize((W, H), Image.NEAREST))
            fg |= mb; n += 1
        return fg, n

    union_fg = {i: np.zeros((H, W), bool) for i in range(len(stems))}
    nobj_f = {i: 0 for i in range(len(stems))}
    for pr in args.prompt:
        predictor.handle_request(dict(type="reset_session", session_id=sid))
        predictor.handle_request(dict(type="add_prompt", session_id=sid,
                                      frame_index=args.seed_frame, text=pr))
        print(f"[sam3v] seeded '{pr}' on frame {args.seed_frame}; propagating ...", flush=True)
        for resp in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
            i = resp["frame_index"]
            fg, n = collect_binary(resp["outputs"])
            union_fg[i] |= fg; nobj_f[i] += n
    outs = None  # masks already unioned in union_fg
    print(f"[sam3v] unioned {len(args.prompt)} concept propagations", flush=True)

    overview = []
    for i, st in enumerate(stems):
        fg = union_fg[i]; nobj = nobj_f[i]
        from scipy.ndimage import binary_fill_holes, label
        if fg.any():
            lab, n = label(fg)
            if n > 1:
                s = np.bincount(lab.ravel()); s[0] = 0; fg = lab == s.argmax()
            fg = binary_fill_holes(fg)
        imageio.imwrite(args.out_dir / f"{st}.png", fg.astype(np.uint8) * 255)
        if i in args.overview:
            overview.append((st, np.asarray(Image.open(rgb[i]).convert("RGB")), fg, nobj))

    if overview:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(overview), 3, figsize=(13, 4*len(overview)), squeeze=False)
        for r, (st, im, msk, n) in enumerate(overview):
            ax[r,0].imshow(im); ax[r,0].set_title(f"{st} ({n} obj)", fontsize=9); ax[r,0].axis('off')
            ax[r,1].imshow(msk, cmap='gray'); ax[r,1].set_title(f"mask ({msk.mean():.0%})", fontsize=9); ax[r,1].axis('off')
            ov = im.astype(np.float32).copy(); ov[~msk] = ov[~msk]*0.4 + np.array([200,30,30])*0.6
            ax[r,2].imshow(ov.clip(0,255).astype(np.uint8)); ax[r,2].set_title("overlay", fontsize=9); ax[r,2].axis('off')
        fig.suptitle(f"SAM 3 video (seed '{args.prompt}' frame {args.seed_frame} -> propagate)", fontsize=11)
        fig.tight_layout(); fig.savefig(args.out_dir / "_overview.png", dpi=120, bbox_inches='tight')
        print(f"[sam3v] overview -> {args.out_dir/'_overview.png'}", flush=True)


if __name__ == "__main__":
    main()
