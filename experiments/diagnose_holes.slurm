#!/bin/bash
#SBATCH --job-name=diagnose-holes
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/diagnose_holes_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/diagnose_holes_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=gpu004
#SBATCH --time=00:15:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

RUN_DIR=${RUN_DIR:-outputs/run_20260506_200907_lego}
VIEW=${VIEW:-0}
DOWN=${DOWN:-4}

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "=========================================="
echo "run_dir: $RUN_DIR"
echo "view:    $VIEW"
echo "down:    $DOWN"

python -u analysis/diagnose_holes.py "$RUN_DIR" \
  --view "$VIEW" \
  --down "$DOWN" \
  --device cpu \
  --out "${RUN_DIR}/diagnose_holes.png"
