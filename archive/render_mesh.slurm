#!/bin/bash
#SBATCH --job-name=render-mesh
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_mesh_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_mesh_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu_homogen
#SBATCH --time=12:00:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
DTU_ROOT=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr
LEGO_SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/nerf_synthetic/lego

N_VIEWS=${N_VIEWS:-6}
SS=${SS:-1}
MC_RES=${MC_RES:-512}
AO_RAYS=${AO_RAYS:-24}

run() {
  local ckpt="$1" scene="$2" dataset="$3" tag="$4" bound="$5" ao_radius="$6"
  local out="$ROOT/outputs/mesh_render_${tag}_${SLURM_JOB_ID}"
  echo "=== rendering $tag ==="
  python -u analysis/render_mesh.py \
    --ckpt "$ckpt" \
    --scene "$scene" \
    --dataset "$dataset" \
    --n_views "$N_VIEWS" \
    --ss "$SS" \
    --bound "$bound" \
    --mc_res "$MC_RES" \
    --ao_rays "$AO_RAYS" \
    --ao_radius "$ao_radius" \
    --save_mesh \
    --out_dir "$out"
}

run "$ROOT/outputs/L=6_100K_weik=0.0_lego/checkpoint_best_photo.pt"    "$LEGO_SCENE"        lego lego_weik0.0    1.5 0.05
run "$ROOT/outputs/L=6_100K_weik=0.1_lego/checkpoint_best_photo.pt"    "$LEGO_SCENE"        lego lego_weik0.1    1.5 0.05
run "$ROOT/outputs/L=6_100K_weik=0.1_scan122/checkpoint_best_photo.pt" "$DTU_ROOT/scan122"  dtu  scan122_weik0.1  1.1 0.03
