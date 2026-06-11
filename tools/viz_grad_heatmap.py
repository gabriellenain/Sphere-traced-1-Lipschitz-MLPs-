"""Minimal: visualise the grad-weighted-sampling heatmap on one view.

Usage: python viz_grad_heatmap.py <scene> <view_id> [down] [alpha]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.data import load_views, make_deterministic_rays
from lip_tracer.train import _image_grad_ray_weights

scene = Path(sys.argv[1])
vi    = int(sys.argv[2])
down  = int(sys.argv[3]) if len(sys.argv) > 3 else 1
alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8

views = load_views(scene=scene)
det   = make_deterministic_rays(views, down, "cpu")
H_d, W_d = views["H"] // down, views["W"] // down
rpv   = H_d * W_d
sl    = slice(vi * rpv, (vi + 1) * rpv)

w   = _image_grad_ray_weights(det, H_d, W_d)[sl].reshape(H_d, W_d).numpy()
img = det["gt"][sl].reshape(H_d, W_d, 3).numpy()
fg  = det["fg"][sl].reshape(H_d, W_d).numpy().astype(bool)

# Effective per-pixel sampling probability over the fg set: p = α·grad + (1−α)·unif
g = w.copy(); g[~fg] = 0.0
g = g / max(g.sum(), 1e-12)
u = fg.astype(np.float64) / max(fg.sum(), 1)
prob = alpha * g + (1.0 - alpha) * u

def heat(ax, field, title):
    """Heatmap on fg only; bg shown white; vmax=p99 so a few hot edges
    don't crush the contrast; sqrt stretch for the peaked distribution."""
    m = np.where(fg, np.sqrt(np.maximum(field, 0.0)), np.nan)
    vmax = np.nanpercentile(m, 99) or 1.0
    cmap = plt.cm.magma.copy(); cmap.set_bad("white")
    im = ax.imshow(m, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

fig, ax = plt.subplots(1, 4, figsize=(24, 6))
ax[0].imshow(img); ax[0].set_title(f"GT view {vi}"); ax[0].axis("off")
heat(ax[1], w,    "grad weight")
heat(ax[2], prob, f"sampling prob (α={alpha:g})")
ax[3].imshow(img)
ax[3].imshow(np.where(fg, np.sqrt(prob), np.nan),
             cmap=plt.cm.magma.copy(), alpha=0.55)
ax[3].set_title("overlay"); ax[3].axis("off")

out = f"grad_heatmap_v{vi}.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}  (H_d={H_d} W_d={W_d}  α={alpha:g}  "
      f"fg={fg.mean():.1%}  w[max]={w.max():.4f}  "
      f"prob[min/mean/max over fg]="
      f"{prob[fg].min():.2e}/{prob[fg].mean():.2e}/{prob[fg].max():.2e})")
