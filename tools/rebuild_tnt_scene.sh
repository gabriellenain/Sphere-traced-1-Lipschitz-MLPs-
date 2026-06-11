#!/bin/bash
# Replace one NSVF-masked TnT scene with COLMAP-on-raw-frames data, coherently.
#
# Prerequisites (you do these):
#   1. download the scene image set from tanksandtemples.org and unpack frames
#      into  data/tnt/<Scene>_raw/images/
#   2. run SfM:  sbatch colmap_sfm_raw.slurm <Scene> sequential
#
# Then this driver: convert -> coherence-check -> (only if it passes) archive the
# old NSVF scene to <Scene>_nsvf and move the COLMAP scene into <Scene>, so every
# existing config/slurm that names the scene now uses the better cameras.
# It NEVER deletes the NSVF data and NEVER swaps a scene that fails the checks.
#
# Usage:  tools/rebuild_tnt_scene.sh <Scene> [--clip P]

set -euo pipefail

SCENE=${1:?usage: tools/rebuild_tnt_scene.sh <Scene> [--clip P]}
shift || true
CLIP=2
if [ "${1:-}" = "--clip" ]; then CLIP=${2:?--clip needs a value}; fi

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
cd "$ROOT"
PY=/scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/python

RAW=$ROOT/data/tnt/${SCENE}_raw/images
WS=$ROOT/data/tnt/${SCENE}_raw/colmap
NEW=$ROOT/data/tnt/${SCENE}_colmap
CUR=$ROOT/data/tnt/${SCENE}
BAK=$ROOT/data/tnt/${SCENE}_nsvf

# --- preconditions ---------------------------------------------------------
if [ ! -d "$RAW" ] || [ -z "$(ls -A "$RAW" 2>/dev/null)" ]; then
    echo "ERROR: no raw frames at $RAW" >&2
    echo "  download the image set from https://www.tanksandtemples.org/download/ first." >&2
    exit 1
fi
if [ ! -f "$WS/sparse_txt/cameras.txt" ]; then
    echo "ERROR: no SfM result at $WS/sparse_txt" >&2
    echo "  run:  sbatch colmap_sfm_raw.slurm $SCENE sequential" >&2
    exit 1
fi
if [ -e "$BAK" ]; then
    echo "ERROR: $BAK already exists -- scene looks already swapped. Refusing to clobber." >&2
    exit 1
fi

# --- convert ---------------------------------------------------------------
echo "[1/4] converting COLMAP -> NSVF format ($NEW)"
rm -rf "$NEW"
"$PY" tools/colmap_to_nsvf.py --colmap "$WS" --out "$NEW" --clip "$CLIP"

# --- object bbox: transfer the curated NSVF bbox into the new gauge --------
# A raw SfM cloud spans the whole scene, so the auto bbox bounds background, not
# the object. If the current scene is the NSVF release, align cameras (Umeyama)
# and map its hand-tuned object bbox into the COLMAP gauge. Falls back to the
# auto bbox (with a warning) if there's no NSVF scene to borrow from.
if [ -f "$CUR/bbox.txt" ] && [ -d "$CUR/pose" ]; then
    echo "[2/4] transferring NSVF object bbox -> COLMAP gauge"
    "$PY" tools/transfer_nsvf_bbox.py --nsvf "$CUR" --colmap "$WS" --out "$NEW"
else
    echo "[2/4] no NSVF scene at $CUR -- keeping auto bbox (may bound whole scene)"
fi

# --- coherence gate --------------------------------------------------------
echo "[3/4] coherence check"
if ! "$PY" tools/check_tnt_scene.py --scene "$NEW"; then
    echo "ERROR: coherence check failed -- NOT swapping. $CUR left untouched." >&2
    echo "  inspect $NEW (try a larger --clip or hand-edit bbox.txt), then re-run." >&2
    exit 1
fi

# --- swap (archive old, never delete) --------------------------------------
echo "[4/4] swapping: $CUR -> $BAK, $NEW -> $CUR"
if [ -e "$CUR" ]; then mv "$CUR" "$BAK"; fi
mv "$NEW" "$CUR"

echo "done. ${SCENE} now uses COLMAP-on-raw cameras."
echo "  old NSVF data archived at: $BAK"
echo "  reminder: fresh COLMAP gauge -> old ${SCENE}_trans.txt no longer aligns to GT;"
echo "            re-estimate recon->GT before analysis/eval_tnt_official.py."
