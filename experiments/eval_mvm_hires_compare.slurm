#!/bin/bash
#SBATCH --job-name=mvm-hires-cmp
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvm_hires_cmp_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvm_hires_cmp_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

PT=outputs/run_20260530_091942_mvm_kinette-cos-hx_4927852/ckpt/checkpoint_step_080000.pt
SCENE=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/mvmannequin_neus/kinette-cos-hx
RUNDIR=outputs/run_20260530_091942_mvm_kinette-cos-hx_4927852

echo "############ BASELINE  res=512  bound=1.0  (~1.96 mm/voxel) ############"
python -u analysis/eval_mvmannequin_official.py \
    --ckpt "$PT" --scene "$SCENE" --res 512 --bound 1.0 \
    --out "$RUNDIR/mvmannequin_official_res512"

echo "############ HI-RES cropped  ~1.0 mm/voxel  (object AABB +10%) ############"
HIRES_PLY="$RUNDIR/mvmannequin_official_hires1mm/pred_world_mesh.ply"
mkdir -p "$RUNDIR/mvmannequin_official_hires1mm"
python -u analysis/extract_mesh_hires.py \
    --ckpt "$PT" --scene "$SCENE" --out "$HIRES_PLY" --voxel-mm 1.0 --margin 0.10
python -u analysis/eval_mvmannequin_official.py \
    --mesh "$HIRES_PLY" --scene "$SCENE" \
    --out "$RUNDIR/mvmannequin_official_hires1mm"

echo "############ DONE ############"
grep -H chamfer_mm "$RUNDIR"/mvmannequin_official_res512/mvmannequin_official.json \
                   "$RUNDIR"/mvmannequin_official_hires1mm/mvmannequin_official.json
