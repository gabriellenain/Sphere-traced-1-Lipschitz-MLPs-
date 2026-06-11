#!/bin/bash
#SBATCH --job-name=gsam-sweep
#SBATCH --output=outputs/gsam_sweep_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu004
set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export HF_HOME=/scratch/_projets_/willow/1-lip-tracer-new/hf_cache
export MPLCONFIGDIR=/tmp/mpl-sweep
IN=_diagnostics/ignatius_gsam_undist/input
run(){ python -u tools/grounded_sam_segment.py --image-dir "$IN" --out-dir "$1" \
        --keep-top 1 --largest-cc --fill-holes --overview-views 0 1 2 3 4 "${@:2}"; }
run _diagnostics/sweep/A_statue            --prompt "statue."
run _diagnostics/sweep/B_statue_pedestal   --prompt "statue." --prompt "pedestal."
run _diagnostics/sweep/C_statue_base        --prompt "statue." --prompt "stone base."
run _diagnostics/sweep/D_statue_fountain    --prompt "statue." --prompt "fountain."
echo DONE
