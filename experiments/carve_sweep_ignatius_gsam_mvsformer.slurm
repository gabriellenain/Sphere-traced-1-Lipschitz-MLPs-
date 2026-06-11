#!/bin/bash
#SBATCH --job-name=carve-ign-gsam-p
#SBATCH --output=outputs/carve_ign_gsam_p_%j.log
#SBATCH --error=outputs/carve_ign_gsam_p_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu004

set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
SRC_SCENE=$ROOT/data/tnt/Ignatius
MASK_SOURCE=$SRC_SCENE/mask_statue
STAGED_SCENE=$ROOT/_diagnostics/ignatius_gsam_statue_lowthr_scene
DEPTH_DIR=$ROOT/_diagnostics/mvsformer_ignatius/depths_1152x640
BASE=$ROOT/_diagnostics/mvsformer_ignatius_gsam_statue_lowthr_p_sweep

RES=${RES:-256}
P_VALUES=${P_VALUES:-"0.99 0.97 0.95 0.92 0.90 0.85 0.80 0.75 0.70"}

cd "$ROOT"
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib-ignatius-gsam-p
export MVSFORMER_REPO=/scratch/glenain/MVSFormerPlusPlus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$STAGED_SCENE" "$BASE"
ln -sfn "$SRC_SCENE/rgb" "$STAGED_SCENE/rgb"
ln -sfn "$SRC_SCENE/pose" "$STAGED_SCENE/pose"
ln -sfn "$SRC_SCENE/intrinsics.txt" "$STAGED_SCENE/intrinsics.txt"
ln -sfn "$SRC_SCENE/bbox.txt" "$STAGED_SCENE/bbox.txt"
ln -sfn "$MASK_SOURCE" "$STAGED_SCENE/mask"

N_POSES=$(find "$SRC_SCENE/pose" -maxdepth 1 -type f -name '0_*.txt' | wc -l)
N_MASKS=$(find "$MASK_SOURCE" -maxdepth 1 -type f -name '0_*.png' | wc -l)
N_DEPTHS=$(find "$DEPTH_DIR/Ignatius/depth_est" -maxdepth 1 -type f -name '*.pfm' | wc -l)
echo "GSAM masks: $N_MASKS/$N_POSES  MVSFormer depths: $N_DEPTHS/$N_POSES"
if [ "$N_MASKS" -lt "$N_POSES" ]; then
  echo "ERROR: expected full GSAM mask_statue set at $MASK_SOURCE" >&2
  exit 1
fi
if [ "$N_DEPTHS" -lt "$N_POSES" ]; then
  echo "ERROR: expected full MVSFormer depth set at $DEPTH_DIR" >&2
  exit 1
fi

for P in $P_VALUES; do
  TAG=$(printf "p%03d" "$(python -c "print(round(float('$P') * 100))")")
  OUT=$BASE/${TAG}_res${RES}
  if [ -f "$OUT/summary.json" ]; then
    echo "skip $P -> $OUT/summary.json exists"
    continue
  fi
  echo
  echo "=== p=$P  res=$RES  out=$OUT ==="
  python -u tools/carve_visual_hull_with_mvsformer_scan24.py \
    --scene "$STAGED_SCENE" \
    --depth-dir "$DEPTH_DIR" \
    --out-dir "$OUT" \
    --res "$RES" --bound 1.5 \
    --conf-thr 0.5 --votes-req 3 \
    --vh-percentile "$P" --vh-min-views 8
done

python - "$BASE" <<'PY'
import json, sys
from pathlib import Path

base = Path(sys.argv[1])
rows = []
for sp in sorted(base.glob("p*_res*/summary.json")):
    s = json.loads(sp.read_text())
    st = s["settings"]
    rows.append((
        st["vh_percentile"],
        s["hull_voxels"],
        s["voxels_final"],
        s["pct_removed_total"],
        s["outputs"]["compare_png"],
        s["outputs"]["carved_sdf"],
    ))

print("\nSummary (lower p = higher mask-hull recall, lower precision):")
print("p      hull_voxels  final_voxels  removed%   compare_png")
for p, hull, final, removed, png, _sdf in rows:
    print(f"{p:0.2f}  {hull:11d}  {final:12d}  {removed:8.2f}  {png}")

if rows:
    p, hull, final, removed, _png, sdf = min(rows, key=lambda r: r[0])
    print(f"\nLowest swept p for maximum recall: p={p:0.2f}")
    print(f"Carved SDF: {sdf}")
PY
