#!/usr/bin/env bash
# Render NeuS reconstruction from a DTU view + mask + crop to match GT object_only.
# Usage: ./render_view_then_mask.sh <scan> <view> <mesh.ply> <out_prefix>
set -euo pipefail
SCAN="${1:-scan122}"
VIEW="${2:-50}"
MESH="${3:-scan122_owl_only.ply}"
PREFIX="${4:-neus_${SCAN}_v${VIEW}}"
SCENE="baselines/NeuS/public_data/${SCAN}"

/home/glenain/blender/blender -b -P tools/render_neus_match_gt.py -- \
    --mesh "${MESH}" \
    --cameras "${SCENE}/cameras_sphere.npz" \
    --view "${VIEW}" --width 1600 --height 1200 --samples 32 \
    --out "${PREFIX}_render.png"

/usr/bin/python3.9 - << EOF
import numpy as np
from PIL import Image
img  = np.array(Image.open("${PREFIX}_render.png").convert("RGB"))
mask = np.array(Image.open("${SCENE}/mask/$(printf '%03d' ${VIEW}).png").convert("L")) > 127
out = img.copy()
out[~mask] = 255
ys, xs = np.where(mask)
pad = 16
y0,y1 = max(ys.min()-pad,0), min(ys.max()+pad+1, out.shape[0])
x0,x1 = max(xs.min()-pad,0), min(xs.max()+pad+1, out.shape[1])
Image.fromarray(out[y0:y1, x0:x1]).save("${PREFIX}_object.png")
print("wrote ${PREFIX}_object.png", (x1-x0), "x", (y1-y0))
EOF
