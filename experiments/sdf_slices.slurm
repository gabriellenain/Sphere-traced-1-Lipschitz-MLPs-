#!/bin/bash
#SBATCH --job-name=sdf-slices
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/sdf_slices_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/sdf_slices_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

PT=${PT:-outputs/run_20260510_094943_scan65/checkpoint_final.pt}
GT_MESH=${GT_MESH:-outputs/paper_render_marching_scan65_4898606/mesh.ply}
AXIS=${AXIS:-xz}          # xz = horizontal cuts through skull
N_CUTS=${N_CUTS:-4}
CUT_RANGE=${CUT_RANGE:-0.4}
BOUND=${BOUND:-1.0}
RES=${RES:-1024}
VMAX=${VMAX:-0.3}
CONTEXT_VIEW=${CONTEXT_VIEW:-8}
CONTEXT_DOWN=${CONTEXT_DOWN:-2}
AO_RAYS=${AO_RAYS:-16}
PANEL_SIZE=${PANEL_SIZE:-4.5}
FONTSIZE=${FONTSIZE:-13}
OUT=${OUT:-}

OFFSETS=${OFFSETS:-}   # e.g. OFFSETS="-0.38 -0.15 0.10 0.35" to pick exact planes

extra_args=()
[[ -n "$OUT"     ]] && extra_args+=(--out "$OUT")
if [[ -n "$OFFSETS" ]]; then
  extra_args+=(--offsets $OFFSETS)
else
  extra_args+=(--n_cuts "$N_CUTS" --cut_range "$CUT_RANGE")
fi

python -u analysis/sdf_slices.py "$PT" \
  --gt_mesh "$GT_MESH" \
  --axis "$AXIS" \
  --bound "$BOUND" \
  --res "$RES" \
  --vmax "$VMAX" \
  --context_view "$CONTEXT_VIEW" \
  --context_down "$CONTEXT_DOWN" \
  --ao_rays "$AO_RAYS" \
  --panel_size "$PANEL_SIZE" \
  --fontsize "$FONTSIZE" \
  --dpi 300 \
  "${extra_args[@]}"
