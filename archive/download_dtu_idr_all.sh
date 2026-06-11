#!/usr/bin/env bash
# Downloads the IDR-preprocessed DTU archive once and extracts all scan folders.
#
# Usage:
#   bash archive/download_dtu_idr_all.sh [data_root]
#
# Example on Jean Zay:
#   bash archive/download_dtu_idr_all.sh "$WORK/1-lip-tracer/data"
#
# Result:
#   [data_root]/dtu_idr/scanXX/{cameras.npz,image/,mask/,...}
#
# Source: Yariv et al. IDR — https://github.com/lioryariv/idr
set -euo pipefail

DATA_ROOT=${1:-/scratch/_projets_/willow/1-lip-tracer-new/data}
OUT_ROOT="${DATA_ROOT}/dtu_idr"
TMP_ROOT="${TMPDIR:-${DATA_ROOT}/.tmp_dtu_idr_download}"
DROPBOX_URL="https://www.dropbox.com/sh/5tam07ai8ch90pf/AADniBT3dmAexvm_J1oL__uoa?dl=1"

OUTER_ZIP="${TMP_ROOT}/idr_data.zip"
INNER_ZIP="${TMP_ROOT}/DTU.zip"
EXTRACT_DIR="${TMP_ROOT}/extract"

mkdir -p "$OUT_ROOT" "$TMP_ROOT" "$EXTRACT_DIR"

echo "Downloading IDR data archive to ${OUTER_ZIP} ..."
wget --continue --show-progress -O "$OUTER_ZIP" "$DROPBOX_URL"

echo "Extracting DTU.zip from IDR archive ..."
unzip -q "$OUTER_ZIP" "DTU.zip" -d "$TMP_ROOT"

echo "Extracting all DTU scan folders ..."
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
unzip -q "$INNER_ZIP" "DTU/scan*/*" -d "$EXTRACT_DIR"

echo "Installing scans into ${OUT_ROOT} ..."
for scan_dir in "$EXTRACT_DIR"/DTU/scan*; do
    [ -d "$scan_dir" ] || continue
    scan_name=$(basename "$scan_dir")
    mkdir -p "${OUT_ROOT}/${scan_name}"
    cp -a "${scan_dir}/." "${OUT_ROOT}/${scan_name}/"
done

echo "Verifying extracted scans ..."
bad=0
for scan_dir in "$OUT_ROOT"/scan*; do
    [ -d "$scan_dir" ] || continue
    for required in cameras.npz image mask; do
        if [ ! -e "${scan_dir}/${required}" ]; then
            echo "Missing ${required} in ${scan_dir}"
            bad=1
        fi
    done
done

if [ "$bad" -ne 0 ]; then
    echo "Some scans are incomplete; leaving temporary files in ${TMP_ROOT} for inspection."
    exit 1
fi

echo "Done. Extracted scans:"
find "$OUT_ROOT" -maxdepth 1 -type d -name "scan*" -printf "%f\n" | sort -V

echo "Cleaning temporary extraction directory ..."
rm -rf "$EXTRACT_DIR" "$INNER_ZIP"
echo "Kept downloaded archive at ${OUTER_ZIP} for resume/reuse."
