#!/bin/bash
#SBATCH --job-name=gsam-sweep2
#SBATCH --output=outputs/gsam_sweep2_%j.log
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
export MPLCONFIGDIR=/tmp/mpl-sweep2
IN=_diagnostics/ignatius_gsam_undist/input
G(){ python -u tools/grounded_sam_segment.py --image-dir "$IN" --out-dir "$1" \
       --largest-cc --fill-holes --overview-views 0 1 2 3 4 "${@:2}"; }
# E: statue+pedestal, expand boxes to grab base
G _diagnostics/sweep2/E_pedestal_expand --prompt "statue." --prompt "pedestal." --keep-top 1 --box-expand 0.10
# F: statue+pedestal, lower thresholds so pedestal fires on far/oblique views
G _diagnostics/sweep2/F_pedestal_lowthr --prompt "statue." --prompt "pedestal." --keep-top 1 --box-threshold 0.28 --text-threshold 0.18
# G: single combined phrase, keep two boxes
G _diagnostics/sweep2/G_single_top2 --prompt "statue. pedestal." --keep-top 2
# H: descriptive phrase
G _diagnostics/sweep2/H_phrase --prompt "a bronze statue on a stone pedestal." --keep-top 2 --box-threshold 0.30 --text-threshold 0.20
echo DONE
