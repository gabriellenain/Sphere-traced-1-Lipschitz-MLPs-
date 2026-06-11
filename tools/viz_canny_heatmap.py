"""Minimal: visualise a Canny-edge ray-weighting heatmap on one view.

Canny analog of tools/viz_grad_heatmap.py: instead of the central-difference
luminance gradient used by _image_grad_ray_weights, the per-ray weight is the
Canny edge map of the GT image (Gaussian smoothing -> Sobel gradients ->
non-maximum suppression -> double-threshold hysteresis). Canny/Sobel are
implemented here in torch since cv2/skimage are unavailable.

Usage: python viz_canny_heatmap.py <scene> <view_id> [down] [alpha] [lo] [hi] [method]
       method in {canny, harris, sobel, all}  (default: all)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.data import load_views, make_deterministic_rays


def _gaussian_blur(img: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    r = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-r, r + 1, dtype=img.dtype, device=img.device)
    k1 = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k1 = k1 / k1.sum()
    kx = k1.view(1, 1, 1, -1)
    ky = k1.view(1, 1, -1, 1)
    img = F.conv2d(img, kx, padding=(0, r))
    img = F.conv2d(img, ky, padding=(r, 0))
    return img


def canny(lum: torch.Tensor, sigma: float, lo: float, hi: float) -> torch.Tensor:
    """lum: (V,H,W) luminance in [0,1]. Returns (V,H,W) binary edge map and
    the NMS-thinned gradient magnitude (for a soft weight)."""
    x = lum.unsqueeze(1)                                   # (V,1,H,W)
    x = _gaussian_blur(x, sigma)

    sob = x.new_tensor([[1.0, 0.0, -1.0],
                        [2.0, 0.0, -2.0],
                        [1.0, 0.0, -1.0]])
    gx = F.conv2d(x, sob.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(x, sob.t().contiguous().view(1, 1, 3, 3), padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    ang = torch.atan2(gy, gx)                              # (-pi, pi]

    # Quantise orientation to 0/45/90/135 deg and compare with the two
    # neighbours along the gradient direction (non-maximum suppression).
    sector = (torch.round(ang / (np.pi / 4.0)) % 4).long()  # 0..3
    m = F.pad(mag, (1, 1, 1, 1))                            # (V,1,H+2,W+2)
    c = m[..., 1:-1, 1:-1]
    nb = {
        0: (m[..., 1:-1, 2:],  m[..., 1:-1, :-2]),          # E / W
        1: (m[..., :-2, 2:],   m[..., 2:, :-2]),            # NE / SW
        2: (m[..., :-2, 1:-1], m[..., 2:, 1:-1]),           # N / S
        3: (m[..., :-2, :-2],  m[..., 2:, 2:]),             # NW / SE
    }
    keep = torch.ones_like(c, dtype=torch.bool)
    for s, (a, b) in nb.items():
        sel = sector == s
        keep &= ~sel | ((c >= a) & (c >= b))
    thin = torch.where(keep, c, torch.zeros_like(c))

    strong = thin >= hi
    weak = (thin >= lo) & ~strong

    # Hysteresis: iteratively promote weak pixels touching strong ones.
    edges = strong.clone()
    pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    for _ in range(64):
        grown = (pool(edges.float()) > 0) & weak
        new = edges | grown
        if torch.equal(new, edges):
            break
        edges = new
    return edges.squeeze(1).float(), thin.squeeze(1)


def _dilate(mask: np.ndarray, it: int = 2) -> np.ndarray:
    """Grow a sparse binary map by `it` pixels so 1px features are visible
    when the panel is downscaled. Uses torch maxpool (cv2/skimage absent)."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    for _ in range(it):
        t = F.max_pool2d(t, 3, stride=1, padding=1)
    return t[0, 0].numpy() > 0


def harris(lum: torch.Tensor, sigma: float, k: float,
           rel_thresh: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Harris corner response. lum: (V,H,W) in [0,1].

    Returns (corner_mask, response): response is the (normalised) Harris R map,
    corner_mask is its NMS + relative-threshold peaks (analog of Canny edges).
    """
    x = lum.unsqueeze(1)
    sob = x.new_tensor([[1.0, 0.0, -1.0],
                        [2.0, 0.0, -2.0],
                        [1.0, 0.0, -1.0]])
    gx = F.conv2d(x, sob.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(x, sob.t().contiguous().view(1, 1, 3, 3), padding=1)
    # Structure-tensor entries, Gaussian-windowed.
    sxx = _gaussian_blur(gx * gx, sigma)
    syy = _gaussian_blur(gy * gy, sigma)
    sxy = _gaussian_blur(gx * gy, sigma)
    det = sxx * syy - sxy * sxy
    tr = sxx + syy
    r = det - k * tr * tr
    r = torch.clamp(r, min=0.0)

    # NMS: keep response only at 3x3 local maxima. Thresholding is left to
    # the caller (fg-relative percentile) so a few huge peaks don't crush it.
    peak = F.max_pool2d(r, 5, stride=1, padding=2)
    nms = torch.where(r >= peak, r, torch.zeros_like(r))
    return nms.squeeze(1), r.squeeze(1)


def main() -> None:
    scene = Path(sys.argv[1])
    vi    = int(sys.argv[2])
    down  = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    lo    = float(sys.argv[5]) if len(sys.argv) > 5 else 0.06
    hi    = float(sys.argv[6]) if len(sys.argv) > 6 else 0.15

    method = sys.argv[7].lower() if len(sys.argv) > 7 else "all"
    valid = {"canny", "harris", "sobel"}
    methods = sorted(valid) if method == "all" else [method]
    if any(m not in valid for m in methods):
        raise SystemExit(f"method must be one of {sorted(valid)} or 'all', got {method!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    views = load_views(scene=scene)
    n_views = views["images"].shape[0]
    if not 0 <= vi < n_views:
        print(f"skip {scene.name} v{vi}: only {n_views} views (0..{n_views-1})")
        return
    det   = make_deterministic_rays(views, down, "cpu")
    H_d, W_d = views["H"] // down, views["W"] // down
    rpv   = H_d * W_d
    sl    = slice(vi * rpv, (vi + 1) * rpv)

    gt  = det["gt"][sl].reshape(H_d, W_d, 3)
    lum = (gt * gt.new_tensor([0.299, 0.587, 0.114])).sum(-1)[None].to(device)
    img = gt.numpy()
    fg  = det["fg"][sl].reshape(H_d, W_d).numpy().astype(bool)

    # Crop to the fg bounding box (+margin): the object fills only a fraction
    # of the 1600x1200 frame, so the full frame wastes most of every panel.
    ys, xs = np.where(fg)
    pad = 24
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H_d)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W_d)
    crop = (slice(y0, y1), slice(x0, x1))
    fg_c = fg[crop]

    # raw: continuous response; w: the sampling weight training would use.
    # is_pts marks sparse point/edge maps that need dilation to stay visible.
    def features(m: str):
        if m == "canny":
            edge, thin = canny(lum, sigma=1.0, lo=lo, hi=hi)
            return (thin[0].cpu().numpy(), "Canny NMS mag",
                    edge[0].cpu().numpy(), "Canny edges", 2)
        if m == "harris":
            nms, hresp = harris(lum, sigma=1.5, k=0.04, rel_thresh=0.0)
            nms = nms[0].cpu().numpy(); hresp = hresp[0].cpu().numpy()
            # fg-relative threshold: keep the strongest ~1.5% of fg pixels
            # (global-max normalisation collapsed everything before).
            cand = nms[fg & (nms > 0)]
            thr = np.percentile(cand, 90.0) if cand.size else np.inf
            corners = (nms >= thr) & fg
            return (hresp, "Harris response",
                    corners.astype(np.float32), "Harris corners", 3)
        # sobel: central-difference luminance gradient (same as the training
        # _image_grad_ray_weights), used directly as the soft weight.
        l = lum[0]
        gx = torch.zeros_like(l); gy = torch.zeros_like(l)
        gx[:, 1:-1] = 0.5 * (l[:, 2:] - l[:, :-2])
        gy[1:-1, :] = 0.5 * (l[2:, :] - l[:-2, :])
        g = torch.sqrt(gx * gx + gy * gy + 1e-12).cpu().numpy()
        return (g, "Sobel |grad lum|", g, "Sobel weight", 0)

    def heat(ax, field, title, stretch=np.sqrt, pct=99.5):
        fc = field[crop]
        m = np.where(fg_c, stretch(np.maximum(fc, 0.0)), np.nan)
        vmax = np.nanpercentile(m, pct) or 1.0
        cmap = plt.cm.magma.copy(); cmap.set_bad("0.12")
        im = ax.imshow(m, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=15); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    u = fg.astype(np.float64) / max(fg.sum(), 1)
    ar = (y1 - y0) / (x1 - x0)
    for m in methods:
        raw, raw_t, w, w_t, dil = features(m)
        g = w.astype(np.float64).copy(); g[~fg] = 0.0
        g = g / max(g.sum(), 1e-12)
        prob = alpha * g + (1.0 - alpha) * u
        # Dilated copy for *display only*; stats/prob use the true w.
        w_disp = _dilate(w > 0, dil).astype(np.float32) if dil else w

        panel_w = 6.0
        fig, ax = plt.subplots(1, 4, figsize=(4 * panel_w, panel_w * ar + 1.4))
        ax[0].imshow(img[crop])
        ax[0].set_title(f"GT {scene.name} v{vi}", fontsize=15); ax[0].axis("off")
        heat(ax[1], raw,    raw_t, stretch=np.sqrt)
        heat(ax[2], w_disp, w_t,   stretch=(lambda z: z) if m != "sobel" else np.sqrt)
        heat(ax[3], prob,   f"sampling prob (α={alpha:g})", pct=99.95)
        fig.suptitle(f"{m.capitalize()} — {scene.name} view {vi}", fontsize=17)

        out = f"{m}_heatmap_{scene.name}_v{vi}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}  [{device}]  (H_d={H_d} W_d={W_d}  α={alpha:g}  "
              f"lo={lo:g} hi={hi:g}  fg={fg.mean():.1%}  "
              f"w>0 frac (fg)={(w[fg] > 0).mean():.2%}  "
              f"prob[min/mean/max over fg]="
              f"{prob[fg].min():.2e}/{prob[fg].mean():.2e}/{prob[fg].max():.2e})")


if __name__ == "__main__":
    main()
