#!/bin/bash
#SBATCH --job-name=pv-chamfer
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/pv_chamfer_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/pv_chamfer_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

python=/home/glenain/nerfstudio/.pixi/envs/default/bin/python

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

OUT_DIR=${OUT_DIR:?set OUT_DIR=outputs/depth_refine_JOBID}
RUN=${RUN:-outputs/run_20260510_094943_scan65}
DTU_EVAL_DIR=${DTU_EVAL_DIR:-/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS Data}

$python analysis/per_view_chamfer_analysis.py \
  --out-dir "$OUT_DIR" \
  --run "$RUN" \
  --dtu-eval-dir "$DTU_EVAL_DIR"
