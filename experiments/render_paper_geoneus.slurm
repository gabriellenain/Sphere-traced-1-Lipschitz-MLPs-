#!/bin/bash
#SBATCH --job-name=render-geoneus
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_paper_geoneus_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_paper_geoneus_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
# CPU-only: --mesh path skips SDF eval; ray casting runs on pyembree.

# Renders a GeoNeuS-trained mesh through analysis/render_paper_marching.py --mesh so the
# tones / convention / grid / outputs match our own paper figures byte-for-byte.
#
# Usage: sbatch render_paper_geoneus.slurm <scan_number> [run_dir]
#   <scan_number>  : e.g. 65 122 — picks the latest run matching that scan.
#   [run_dir]      : optional explicit outputs/geoneus_*_* run dir to render.

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
DTU_ROOT=${DTU_ROOT:-/scratch/_projets_/willow/1-lip-tracer/data/dtu}

# ---------- shared render knobs (keep in sync with render_paper_marching.slurm) ----------
N_VIEWS=${N_VIEWS:-5}
SS=${SS:-1}
ZOOM=${ZOOM:-1.0}
AO_RAYS=${AO_RAYS:-64}
AO_RADIUS=${AO_RADIUS:-0.06}
DOWN=${DOWN:-1}
COLOR_NEIGHBORS=${COLOR_NEIGHBORS:-3}
HULL=${HULL:-0}
HULL_RES=${HULL_RES:-256}
VIEWS=${VIEWS:-0,8,16,32,40}
LAYOUT=${LAYOUT:-full}
# ----------------------------------------------------------------------------------------

SCAN_NUM="${1:?usage: sbatch render_paper_geoneus.slurm <scan_number> [run_dir]}"
SCAN="scan${SCAN_NUM}"
SCENE="$DTU_ROOT/$SCAN"
RUN_DIR="${2:-}"

if [[ -z "$RUN_DIR" ]]; then
    # Pick the most recently modified GeoNeuS run that trained on this scan.
    RUN_DIR=$(ls -dt "$ROOT"/outputs/geoneus_*_* 2>/dev/null | while read d; do
        grep -lq "$SCAN" "$d/run.conf" 2>/dev/null && { echo "$d"; break; }
    done)
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "ERROR: no GeoNeuS run dir found for $SCAN under outputs/geoneus_*"; exit 1
fi
echo "run_dir: $RUN_DIR"

# Latest cleaned mesh (matches official DTU eval). Falls back to raw mesh.
MESH=$(ls -1 "$RUN_DIR"/meshes/*_clean.ply 2>/dev/null | sort | tail -n1)
if [[ -z "$MESH" ]]; then
    MESH=$(ls -1 "$RUN_DIR"/meshes/*.ply 2>/dev/null | sort | tail -n1)
fi
if [[ -z "$MESH" || ! -f "$MESH" ]]; then
    echo "ERROR: no mesh found under $RUN_DIR/meshes/"; exit 1
fi
echo "mesh:    $MESH"
echo "scene:   $SCENE"

OUT="$ROOT/outputs/paper_render_geoneus_${SCAN}_${SLURM_JOB_ID}"
echo "out:     $OUT"

view_args=()
if [[ -n "$VIEWS" ]]; then view_args=(--views "$VIEWS")
else view_args=(--n_views "$N_VIEWS"); fi

extra_args=()
if [[ "$HULL" = "1" ]]; then extra_args+=(--hull --hull_res "$HULL_RES"); fi

python -u analysis/render_paper_marching.py \
    --mesh "$MESH" \
    --scene "$SCENE" \
    "${view_args[@]}" \
    --ss "$SS" \
    --zoom "$ZOOM" \
    --ao_rays "$AO_RAYS" \
    --ao_radius "$AO_RADIUS" \
    --down "$DOWN" \
    --color_neighbors "$COLOR_NEIGHBORS" \
    --layout "$LAYOUT" \
    --out_dir "$OUT" \
    "${extra_args[@]}"
