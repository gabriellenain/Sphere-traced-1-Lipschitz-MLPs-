#!/usr/bin/env bash
# Downloads IDR-preprocessed DTU scans in original IDR format (cameras.npz + image/ + mask/).
# Images are at full resolution (~1600x1200), unlike the 384x384 custom format in data/dtu/.
# Usage: bash download_dtu_idr.sh [scan_id] [data_root]
#   default scan_id: 65
#   default data_root: /scratch/_projets_/willow/1-lip-tracer-new/data
#
# Source: Yariv et al. IDR — https://github.com/lioryariv/idr
set -e

SCAN=${1:-65}
DATA_ROOT=${2:-/scratch/_projets_/willow/1-lip-tracer-new/data}
OUT_DIR="${DATA_ROOT}/dtu_idr/scan${SCAN}"
TMP="/tmp/dtu_idr_scan${SCAN}.zip"
mkdir -p "$OUT_DIR"

echo "Downloading IDR DTU scan${SCAN} to $OUT_DIR ..."

# Official IDR Dropbox mirror (Yariv et al.)
URL="https://www.dropbox.com/sh/5tam07ai8ch90pf/AADniBT3dmAexvm_J1oL__uoa?dl=1"

wget --show-progress -O "$TMP" "$URL"

echo "Extracting DTU.zip ..."
unzip -q "$TMP" "DTU.zip" -d /tmp/dtu_idr_extract/

echo "Extracting scan${SCAN} ..."
unzip -q /tmp/dtu_idr_extract/DTU.zip "DTU/scan${SCAN}/*" -d /tmp/dtu_idr_extract2/
mv /tmp/dtu_idr_extract2/DTU/scan${SCAN}/* "$OUT_DIR/"
rm -rf "$TMP" /tmp/dtu_idr_extract /tmp/dtu_idr_extract2

echo "Done. Data in $OUT_DIR/"
echo "Expected layout: $OUT_DIR/{cameras.npz, image/*.png, mask/*.png}"
ls "$OUT_DIR/"
