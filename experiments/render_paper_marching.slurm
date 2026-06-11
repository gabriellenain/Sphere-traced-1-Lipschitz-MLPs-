#!/bin/bash
#SBATCH --job-name=render-mc
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_paper_marching_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_paper_marching_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-

N_VIEWS=${N_VIEWS:-5}
SS=${SS:-1}
ZOOM=${ZOOM:-1.0}
MC_RES=${MC_RES:-1024}
BOUND=${BOUND:-1}
LEVEL=${LEVEL:-0.0}
MESH_LEVELS=${MESH_LEVELS:--0.005,0.000,0.005,0.010}
AO_RAYS=${AO_RAYS:-64}
AO_RADIUS=${AO_RADIUS:-0.06}
DOWN=${DOWN:-1}
COLOR_NEIGHBORS=${COLOR_NEIGHBORS:-3}
SAVE_MESH=${SAVE_MESH:-1}
HULL=${HULL:-0}
HULL_RES=${HULL_RES:-256}
PT=${PT:-$ROOT/outputs/run_20260519_203314_scan65/checkpoint_latest.pt}
TAG=${TAG:-lego}
VIEWS=${VIEWS:-0,8,16,32,40}
LAYOUT=${LAYOUT:-full}
ANALYTIC_NORMALS=${ANALYTIC_NORMALS:-0}

run() {
  local ckpt="$1" tag="$2" views="$3"
  local out="$ROOT/outputs/paper_render_marching_${tag}_${SLURM_JOB_ID}"
  echo "=== marching-cubes rendering $tag ==="
  echo "ckpt:     $ckpt"
  echo "out:      $out"
  echo "mc_res:   $MC_RES"
  echo "bound:    $BOUND"
  echo "level:    $LEVEL"
  echo "levels:   $MESH_LEVELS"
  echo "ss:       $SS"
  echo "zoom:     $ZOOM"
  echo "ao_rays:  $AO_RAYS"
  echo "layout:   $LAYOUT"

  local view_args=()
  if [[ -n "$views" ]]; then
    view_args=(--views "$views")
  else
    view_args=(--n_views "$N_VIEWS")
  fi

  local extra_args=()
  if [[ "$SAVE_MESH" = "1" ]]; then
    extra_args+=(--save_mesh)
  fi
  if [[ "$HULL" = "1" ]]; then
    extra_args+=(--hull --hull_res "$HULL_RES")
  fi
  if [[ "$ANALYTIC_NORMALS" = "1" ]]; then
    extra_args+=(--analytic_normals)
  fi

  python -u analysis/render_paper_marching.py \
    --ckpt "$ckpt" \
    "${view_args[@]}" \
    --ss "$SS" \
    --zoom "$ZOOM" \
    --mc_res "$MC_RES" \
    --bound "$BOUND" \
    --level "$LEVEL" \
    --mesh_levels="$MESH_LEVELS" \
    --ao_rays "$AO_RAYS" \
    --ao_radius "$AO_RADIUS" \
    --down "$DOWN" \
    --color_neighbors "$COLOR_NEIGHBORS" \
    --layout "$LAYOUT" \
    --out_dir "$out" \
    "${extra_args[@]}"
}

run "$PT" "$TAG" "$VIEWS"
