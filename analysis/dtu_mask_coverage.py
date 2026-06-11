#!/usr/bin/env python3
"""Per-scan distribution of the % of each image the object occupies (DTU masks).

For every DTU scan we have locally, read every view's mask, threshold the
antialiased RGB mask at >127, and compute foreground_pixels / total_pixels * 100.
Emit:
  - a per-view CSV,
  - a console summary (mean/std/min/median/max per scan),
  - dtu_mask_coverage.png: violin+box distribution across views per scan,
    plus a bar chart of the mean coverage % per scan.
"""
import os
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DTU_ROOT = "/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr"
OUT_PNG = "dtu_mask_coverage.png"
OUT_CSV = "dtu_mask_coverage.csv"

scans = sorted(
    [os.path.basename(p) for p in glob.glob(os.path.join(DTU_ROOT, "scan*"))],
    key=lambda s: int(s.replace("scan", "")),
)

per_scan = {}          # scan -> np.array of per-view coverage %
rows = []              # (scan, view_idx, coverage%)

for scan in scans:
    mask_files = sorted(glob.glob(os.path.join(DTU_ROOT, scan, "mask", "*.png")))
    cov = []
    for mf in mask_files:
        m = np.asarray(Image.open(mf))
        if m.ndim == 3:
            m = m[..., 0]               # channels identical for DTU masks
        fg = m > 127
        pct = 100.0 * fg.sum() / fg.size
        cov.append(pct)
        rows.append((scan, int(os.path.splitext(os.path.basename(mf))[0]), pct))
    per_scan[scan] = np.array(cov)
    c = per_scan[scan]
    print(f"{scan:>8}  n={len(c):3d}  mean={c.mean():6.2f}%  std={c.std():5.2f}  "
          f"min={c.min():6.2f}  med={np.median(c):6.2f}  max={c.max():6.2f}")

# ---- CSV ----
with open(OUT_CSV, "w") as f:
    f.write("scan,view,coverage_pct\n")
    for scan, v, p in rows:
        f.write(f"{scan},{v},{p:.4f}\n")
print(f"\nwrote {OUT_CSV}")

# ---- figure ----
labels = scans
data = [per_scan[s] for s in labels]
means = [d.mean() for d in data]
x = np.arange(len(labels))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9),
                               gridspec_kw={"height_ratios": [2, 1]})

# violin + box: distribution across views
vp = ax1.violinplot(data, positions=x, showextrema=False, widths=0.8)
for b in vp["bodies"]:
    b.set_facecolor("#4C72B0")
    b.set_alpha(0.35)
ax1.boxplot(data, positions=x, widths=0.25, showfliers=True,
            medianprops=dict(color="#C44E52", lw=2),
            flierprops=dict(marker="o", ms=3, mfc="#888", mec="none", alpha=0.6))
# jittered raw points
for i, d in enumerate(data):
    jit = (np.random.RandomState(0).rand(len(d)) - 0.5) * 0.18
    ax1.scatter(x[i] + jit, d, s=8, color="#2A2A2A", alpha=0.5, zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Object coverage per view (%)")
ax1.set_title("DTU: distribution of % of image occupied by the object (mask > 127)")
ax1.grid(axis="y", ls=":", alpha=0.5)

# bar: mean per scan
bars = ax2.bar(x, means, color="#4C72B0")
for i, m in enumerate(means):
    ax2.text(x[i], m + 0.3, f"{m:.1f}%", ha="center", va="bottom", fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{s}\n(n={len(per_scan[s])})" for s in labels])
ax2.set_ylabel("Mean coverage (%)")
ax2.set_ylim(0, max(means) * 1.25)
ax2.grid(axis="y", ls=":", alpha=0.5)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140)
print(f"wrote {OUT_PNG}")
