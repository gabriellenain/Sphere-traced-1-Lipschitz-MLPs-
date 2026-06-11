#!/bin/bash
#SBATCH --job-name=grad-map
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/grad_map_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/grad_map_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_homogen
#SBATCH --time=02:00:00
#SBATCH --exclude=node001


source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

PT_A=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260430_164901_dtu/checkpoint_best_photo.pt
PT_B=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260430_171055_dtu/checkpoint_best_photo.pt
OUT=${OUT:-outputs/grad_map_compare_${SLURM_JOB_ID}}
BOUND=${BOUND:-1.5}
RES=${RES:-512}
AXIS=${AXIS:-all}
SLICES=${SLICES:-"-0.4 0.0 0.4"}
VOLUME_RES=${VOLUME_RES:-96}
CHUNK=${CHUNK:-4096}
VMAX=${VMAX:-2}
HIST_XMAX=${HIST_XMAX:-2}
HIST_YMAX=${HIST_YMAX:-40000}

echo "checkpoint A: $PT_A"
echo "checkpoint B: $PT_B"
echo "out:          $OUT"
echo "bound:        $BOUND"
echo "res:          $RES"
echo "axis:         $AXIS"
echo "slices:       $SLICES"
echo "volume_res:   $VOLUME_RES"
echo "chunk:        $CHUNK"
echo "vmax:         $VMAX"
echo "hist_xmax:    $HIST_XMAX"
echo "hist_ymax:    $HIST_YMAX"

MPLCONFIGDIR=/tmp/mpl python -u archive/grad_map.py \
  --ckpt "$PT_A" "$PT_B" \
  --labels weik=0.0 weik=0.1 \
  --out "$OUT" \
  --bound "$BOUND" \
  --res "$RES" \
  --axis "$AXIS" \
  --slice $SLICES \
  --volume-res "$VOLUME_RES" \
  --chunk "$CHUNK" \
  --vmax "$VMAX" \
  --hist-xmax "$HIST_XMAX" \
  --hist-ymax "$HIST_YMAX" \
  --hist-counts
