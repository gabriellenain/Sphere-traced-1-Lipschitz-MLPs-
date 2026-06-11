#!/bin/bash
#SBATCH --job-name=mvm-2mm-fair
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvm_2mm_fair_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvm_2mm_fair_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

PT=outputs/run_20260530_091942_mvm_kinette-cos-hx_4927852/ckpt/checkpoint_step_080000.pt
SCENE=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/mvmannequin_neus/kinette-cos-hx
RUNDIR=outputs/run_20260530_091942_mvm_kinette-cos-hx_4927852
OUTDIR="$RUNDIR/mvmannequin_official_2mm"
GT=/scratch/_projets_/willow/1-lip-tracer-new/probesdf_release/meshes/MVMannequin/kinette-cos-hx/GroundTruth.ply

echo "########## Ours at 2mm/voxel (matches ProbeSDF extraction resolution) ##########"
mkdir -p "$OUTDIR"
# pre-place the shared GT so eval uses it directly
cp "$GT" "$OUTDIR/gt_world_mesh.ply"

python -u analysis/extract_mesh_hires.py \
    --ckpt "$PT" --scene "$SCENE" \
    --out "$OUTDIR/pred_world_mesh.ply" \
    --voxel-mm 2.0 --margin 0.10

python -u analysis/eval_mvmannequin_official.py \
    --mesh "$OUTDIR/pred_world_mesh.ply" \
    --scene "$SCENE" \
    --out "$OUTDIR"

echo "########## DONE ##########"
python3 -c "
import json
r=json.load(open('$OUTDIR/mvmannequin_official.json'))
print(f'  accuracy     = {r[\"accuracy_mm\"]:.4f} mm')
print(f'  completeness = {r[\"completeness_mm\"]:.4f} mm')
print(f'  chamfer      = {r[\"chamfer_mm\"]:.4f} mm')
"
