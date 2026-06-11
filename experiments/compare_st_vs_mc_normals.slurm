#!/bin/bash
#SBATCH --job-name=cmp-normals
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/cmp_normals_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/cmp_normals_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004

# ---------- params ----------
CKPT=${CKPT:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260519_203314_scan65/checkpoint_latest.pt}
VIEW=${VIEW:-23}
SS=${SS:-1}    # 1 suffit pour comparer les normales ; 2 = OOM à 32G de RAM
MC_RES=${MC_RES:-512}
OUT=${OUT:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/paper_render_test}
# ----------------------------

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python -u analysis/compare_st_vs_mc_normals.py \
    --ckpt "$CKPT" \
    --view "$VIEW" \
    --ss "$SS" \
    --mc_res "$MC_RES" \
    --out_dir "$OUT"
