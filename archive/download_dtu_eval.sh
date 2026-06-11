#!/usr/bin/env bash
# Downloads the official DTU evaluation files needed by --dtu-eval-dir:
#   SampleSet/MVS Data/ObsMask/*.mat and Points/stl/*.ply.
# Usage: bash download_dtu_eval.sh [data_root]
#   default data_root: /scratch/_projets_/willow/1-lip-tracer-new/data
set -euo pipefail

DATA_ROOT=${1:-/scratch/_projets_/willow/1-lip-tracer-new/data}
OUT_ROOT="${DATA_ROOT}/dtu_eval"
TMP_DIR="/tmp/dtu_eval_download"
SAMPLE_ZIP="${TMP_DIR}/SampleSet.zip"
POINTS_ZIP="${TMP_DIR}/Points.zip"

SAMPLE_URL="http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip"
POINTS_URL="http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip"

mkdir -p "$OUT_ROOT" "$TMP_DIR"

echo "Downloading DTU SampleSet to $SAMPLE_ZIP ..."
wget --show-progress -O "$SAMPLE_ZIP" "$SAMPLE_URL"

echo "Extracting SampleSet into $OUT_ROOT ..."
unzip -q "$SAMPLE_ZIP" -d "$OUT_ROOT"

echo "Downloading DTU Points to $POINTS_ZIP ..."
wget --show-progress -O "$POINTS_ZIP" "$POINTS_URL"

echo "Extracting Points into SampleSet/MVS Data ..."
rm -rf "${OUT_ROOT}/SampleSet/MVS Data/Points"
unzip -q "$POINTS_ZIP" -d "${OUT_ROOT}/SampleSet/MVS Data"

rm -rf "$TMP_DIR"

DTU_EVAL="${OUT_ROOT}/SampleSet/MVS Data"
echo "Done. DTU eval data in $DTU_EVAL"
echo "Expected layout: $DTU_EVAL/{ObsMask,Points/stl}"
ls "$DTU_EVAL"
