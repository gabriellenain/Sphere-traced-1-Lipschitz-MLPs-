"""Masks for *exactly what the TnT GT wants reconstructed*.

Pipeline (no text prompt — the prompt comes from the GT geometry):
  1. Register our cameras -> TnT reference cameras (Umeyama on centres) and
     compose with the provided <scene>_trans.txt -> ours_frame -> GT frame.
  2. Crop the GT cloud with <scene>.json (SelectionPolygonVolume).
  3. Project the cropped GT cloud into each view -> a box (+ centre point).
  4. Prompt SAM with that box/point -> clean dense mask.

Output: <out>/<stem>.png masks + <out>/_overview.png (image | GT-proj | SAM mask).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MplPath
from plyfile import PlyData
from PIL import Image
import torch
import imageio.v2 as imageio


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    U, d, Vt = np.linalg.svd((Dc.T @ Sc) / len(src))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = U @ Vt
    c = np.trace(np.diag(d)) / ((Sc ** 2).sum() / len(src))
    T = np.eye(4); T[:3, :3] = c * R; T[:3, 3] = mu_d - c * R @ mu_s
    return T


def read_log(path):
    L = [l for l in Path(path).read_text().split('\n')]
    mats, i = [], 0
    while i < len(L):
        if L[i].strip() == '': i += 1; continue
        mats.append(np.array([[float(x) for x in L[i+1+r].split()] for r in range(4)]))
        i += 5
    return np.stack(mats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("data/tnt/Ignatius"))
    ap.add_argument("--gt-dir", type=Path, default=Path("data/tnt_gt/Ignatius"))
    ap.add_argument("--gt-name", type=str, default="Ignatius")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stems", type=str, nargs="+", default=None,
                    help="pose stems to process; default = all 0_*.txt")
    ap.add_argument("--box-pad", type=float, default=0.04)
    ap.add_argument("--sam", type=str, default="facebook/sam-vit-large")
    ap.add_argument("--overview-stems", type=str, nargs="+", default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- registration ours -> GT ---
    ref = read_log(args.gt_dir / f"{args.gt_name}_COLMAP_SfM.log")
    poses = sorted((args.scene / "pose").glob("0_*.txt"))
    ours = np.stack([np.loadtxt(p).reshape(4, 4) for p in poses])
    S = umeyama(ours[:, :3, 3], ref[:, :3, 3])
    T = np.loadtxt(args.gt_dir / f"{args.gt_name}_trans.txt")
    Minv = np.linalg.inv(T @ S)                      # GT -> ours

    # --- crop GT cloud ---
    v = PlyData.read(args.gt_dir / f"{args.gt_name}.ply")['vertex']
    G = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    cj = json.load(open(args.gt_dir / f"{args.gt_name}.json"))
    poly = np.array(cj['bounding_polygon'])[:, :2]
    m = ((G[:, 2] >= cj['axis_min']) & (G[:, 2] <= cj['axis_max'])
         & MplPath(poly).contains_points(G[:, :2]))
    Po = (Minv @ np.c_[G[m], np.ones(m.sum())].T).T[:, :3]
    print(f"[gt_sam] cropped GT statue pts: {len(Po)}", flush=True)

    # --- SAM ---
    from transformers import SamProcessor, SamModel
    sam_proc = SamProcessor.from_pretrained(args.sam)
    sam = SamModel.from_pretrained(args.sam).to(dev).eval()

    K = np.loadtxt(args.scene / "intrinsics.txt")[:3, :3]
    stems = args.stems or [p.stem for p in poses]
    ov_stems = set(args.overview_stems or stems[:5])
    overview = []
    from scipy.ndimage import label
    for st in stems:
        c2w = np.loadtxt(args.scene / "pose" / f"{st}.txt").reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        Xc = w2c[:3, :3] @ Po.T + w2c[:3, 3:4]
        front = Xc[2] > 1e-6
        uv = K @ Xc; u = uv[0] / uv[2]; vv = uv[1] / uv[2]
        img = Image.open(args.scene / "rgb" / f"{st}.png").convert("RGB")
        W, H = img.size
        ok = front & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        if ok.sum() < 20:
            imageio.imwrite(args.out_dir / f"{st}.png", np.zeros((H, W), np.uint8)); continue
        u_i, v_i = u[ok], vv[ok]
        x0, x1, y0, y1 = u_i.min(), u_i.max(), v_i.min(), v_i.max()
        pw, ph = (x1 - x0) * args.box_pad, (y1 - y0) * args.box_pad
        box = [max(0, x0 - pw), max(0, y0 - ph), min(W - 1, x1 + pw), min(H - 1, y1 + ph)]
        cpt = [[float(np.median(u_i)), float(np.median(v_i))]]
        inp = sam_proc(img, input_boxes=[[box]], input_points=[[cpt]],
                       input_labels=[[[1]]], return_tensors="pt").to(dev)
        with torch.no_grad():
            out = sam(**inp, multimask_output=False)
        msk = sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu())[0][0, 0].numpy()
        if msk.any():
            lab, n = label(msk)
            if n > 1:
                s = np.bincount(lab.ravel()); s[0] = 0; msk = lab == s.argmax()
        imageio.imwrite(args.out_dir / f"{st}.png", (msk.astype(np.uint8) * 255))
        if st in ov_stems:
            overview.append((st, np.asarray(img), (u_i, v_i), msk, box))
    print(f"[gt_sam] wrote {len(stems)} masks -> {args.out_dir}", flush=True)

    if overview:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(overview), 3, figsize=(13, 4 * len(overview)), squeeze=False)
        for r, (st, im, (uu, vv), msk, box) in enumerate(overview):
            ax[r, 0].imshow(im); ax[r, 0].set_title(st, fontsize=9); ax[r, 0].axis('off')
            ax[r, 1].imshow(im); ax[r, 1].scatter(uu, vv, s=0.3, c='lime', marker='.', linewidths=0)
            import matplotlib.patches as mp
            ax[r, 1].add_patch(mp.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                               fill=False, ec='yellow', lw=1.2))
            ax[r, 1].set_title('GT proj + box prompt', fontsize=9); ax[r, 1].axis('off')
            ov = im.astype(np.float32).copy(); ov[~msk] = ov[~msk]*0.4 + np.array([200,30,30])*0.6
            ax[r, 2].imshow(ov.clip(0,255).astype(np.uint8))
            ax[r, 2].set_title(f'SAM mask ({msk.mean():.0%} fg)', fontsize=9); ax[r, 2].axis('off')
        fig.suptitle('Masks of what TnT GT wants: GT crop -> box prompt -> SAM', fontsize=11)
        fig.tight_layout(); fig.savefig(args.out_dir / "_overview.png", dpi=120, bbox_inches='tight')
        print(f"[gt_sam] overview -> {args.out_dir/'_overview.png'}", flush=True)


if __name__ == "__main__":
    main()
