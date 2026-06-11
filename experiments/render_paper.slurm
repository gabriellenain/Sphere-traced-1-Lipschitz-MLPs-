#!/bin/bash
#SBATCH --job-name=render-paper
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_paper_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_paper_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
DTU_ROOT=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr
LEGO_SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/nerf_synthetic/lego

N_VIEWS=${N_VIEWS:-6}
SS=${SS:-1}

run() {
  local ckpt="$1" tag="$2" views="$3"
  local out="$ROOT/outputs/paper_render_${tag}_${SLURM_JOB_ID}"
  echo "=== rendering $tag ==="
  local view_args=()
  if [[ -n "$views" ]]; then
    view_args=(--views "$views")
  else
    view_args=(--n_views "$N_VIEWS")
  fi
  python -u analysis/render_paper.py \
    --ckpt "$ckpt" \
    "${view_args[@]}" \
    --ss "$SS" \
    --zoom "${ZOOM:-0.7}" \
    --normal-blur-sigma "${NORMAL_BLUR:-1.0}" \
    --out_dir "$out"
}

run "$ROOT/outputs/run_20260519_203314_scan65/checkpoint_latest.pt" scan65
