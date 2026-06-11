#!/bin/bash
#SBATCH --job-name=gsam-truck-probe
#SBATCH --output=outputs/gsam_truck_probe_%j.log
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
export MPLCONFIGDIR=/tmp/mpl-gsam-truck-probe

IN=_diagnostics/truck_gsam_probe/input
BASE=_diagnostics/truck_gsam_probe

run() {
  python -u tools/grounded_sam_segment.py \
    --image-dir "$IN" \
    --out-dir "$1" \
    --keep-top 1 \
    --largest-cc \
    --fill-holes \
    --overview-views 0 1 2 3 4 5 6 7 8 9 \
    "${@:2}"
}

run "$BASE/A_truck_default" \
  --prompt "truck."

run "$BASE/B_truck_lowthr" \
  --prompt "truck." \
  --box-threshold 0.28 --text-threshold 0.18

run "$BASE/C_pickup_lowthr" \
  --prompt "pickup truck." \
  --box-threshold 0.28 --text-threshold 0.18

run "$BASE/D_truck_wheels_lowthr" \
  --prompt "truck." --prompt "wheel." \
  --box-threshold 0.28 --text-threshold 0.18

echo "DONE -> $BASE/{A_truck_default,B_truck_lowthr,C_pickup_lowthr,D_truck_wheels_lowthr}/_overview.png"
