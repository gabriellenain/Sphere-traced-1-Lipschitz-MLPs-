#!/bin/bash
#SBATCH --job-name=gsam-sweep3
#SBATCH --output=outputs/gsam_sweep3_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu004
set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export HF_HOME=/scratch/_projets_/willow/1-lip-tracer-new/hf_cache
export MPLCONFIGDIR=/tmp/mpl-sweep3
IN=_diagnostics/ignatius_gsam_undist/input
G(){ python -u tools/grounded_sam_segment.py --image-dir "$IN" --out-dir "$1" \
       --largest-cc --fill-holes --overview-views 0 1 2 3 4 "${@:2}"; }
G _diagnostics/sweep3/I_statue_book_pedestal --prompt "statue." --prompt "book." --prompt "pedestal." --keep-top 1
G _diagnostics/sweep3/J_sbp_lowthr          --prompt "statue." --prompt "book." --prompt "pedestal." --keep-top 1 --box-threshold 0.28 --text-threshold 0.18
echo DONE
