#!/bin/bash
#SBATCH --job-name=1lip-s24-hp1
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-hp1_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-hp1_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# Exact replica of run_20260603_180638_scan24_mvsformer_carved_init_4965606,
# with the single change ncc_half_pix 2.0 -> 1.0 (does a tighter PMVS patch
# half-width help the official DTU Chamfer?). All other model/trace/train/init
# params (incl. the MVSFormer-carved init_sdf_grid) are reused verbatim.

set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs
REFERENCE_RUN=$OUTPUTS/run_20260603_180638_scan24_mvsformer_carved_init_4965606
SOURCE_CONFIG=$REFERENCE_RUN/source_config_scan24_mvsformer_carved_init.json
DTU_EVAL="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data"

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_scan24_mvsfinit_halfpix1_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_scan24_mvsfinit_halfpix1.json
LOG=$RUN_DIR/train.log
mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

# --- derive config: change ONLY train.ncc_half_pix 2.0 -> 1.0 ---
python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" <<'PY'
import copy, json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
source = json.loads(src.read_text())
derived = copy.deepcopy(source); derived.pop("cmd", None)
derived["train"]["ncc_half_pix"] = 1.0

def flatten(v, p=""):
    if isinstance(v, dict):
        o = {}
        for k, it in v.items():
            o.update(flatten(it, f"{p}.{k}" if p else k))
        return o
    return {p: v}

before, after = flatten(source), flatten(derived)
changes = sorted((k, before.get(k), after.get(k))
                 for k in before.keys() | after.keys() if before.get(k) != after.get(k))
expected = sorted([("cmd", source.get("cmd"), None),
                   ("train.ncc_half_pix", source["train"].get("ncc_half_pix"), 1.0)])
# drop the cmd entry if source had no cmd key
expected = [e for e in expected if not (e[0] == "cmd" and e[1] is None)]
if changes != expected:
    raise SystemExit(f"unexpected config changes:\n{changes!r}\nexpected:\n{expected!r}")
dst.write_text(json.dumps(derived, indent=2) + "\n")
print("Derived config differs only at:")
for c in expected: print("  ", c)
PY

echo "Reference run:  $REFERENCE_RUN"
echo "Change:         train.ncc_half_pix 2.0 -> 1.0"
echo "Run dir:        $RUN_DIR"

python -u -m lip_tracer.train \
    --config "$DERIVED_CONFIG" \
    --run-dir "$RUN_DIR" \
    --occ-iters 32 \
    --occ-newton-steps 0 \
    --occ-eps 1e-3 \
    --dtu-eval-dir "$DTU_EVAL" \
    --dtu-official-freq 10000 \
    --dtu-official-res 384 \
    --dtu-official-bound 1.0 \
    --wandb 2>&1 | tee "$LOG"
