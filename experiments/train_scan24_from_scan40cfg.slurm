#!/bin/bash
#SBATCH --job-name=1lip-s24-s40cfg
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-s40cfg_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-s40cfg_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# scan24 with the SAME config/params as the scan40 run
# (run_20260602_150702_scan40_4951716): hull init + border-aware, gray NCC
# (default patch 5), w_sfm=0, w_geo_sdf=0, batch 4096, 300k steps. Only the
# scene (and the unused feature-maps path, w_feature=0) differ from scan40.

set -euo pipefail
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24
DTU_EVAL="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data"
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_scan24_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
LOG=$RUN_DIR/train.log
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"
echo "Run dir: $RUN_DIR  (scan24, scan40 config/params)"

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --run-dir "$RUN_DIR" \
    --init hull \
    --steps 300000 \
    --batch 4096 \
    --trace-iters 64 \
    --down 1 \
    --dtu-eval-dir "$DTU_EVAL" \
    --dtu-official-freq 10000 \
    --dtu-official-res 384 \
    --sfm-min-views 30 \
    --w-sfm 0 \
    --w-geo-sdf 0 \
    --feature-maps outputs/scan24_mast3r_features_idr.pt \
    --hull-border-aware \
    --ncc-color gray \
    --wandb 2>&1 | tee "$LOG"
