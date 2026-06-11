#!/bin/bash
# Fetch RAW (unmasked) Tanks & Temples frames for one scene and lay them out the
# way the COLMAP migration expects:  data/tnt/<Scene>_raw/images/000001.jpg ...
#
# This fills step 1 of tools/rebuild_tnt_scene.sh. It does NOT touch any existing
# NSVF scene -- it only populates <Scene>_raw/images/ from a gated download you
# point it at. Truck lives in the TnT *intermediate* group (Family, Francis,
# Horse, Lighthouse, M60, Panther, Playground, Train, Truck); Barn/Ignatius/...
# live in *training*. The official sets are distributed from
# https://www.tanksandtemples.org/download/ (Google Drive).
#
# You supply the source via exactly one of these env vars:
#   TNT_GDRIVE_ID=<file id>       a single Google Drive file (e.g. intermediate.zip)
#   TNT_GDRIVE_FOLDER=<folder id> a Google Drive folder
#   TNT_URL=<direct http(s) url>  a direct, resumable archive URL
#
# Usage:
#   TNT_GDRIVE_ID=xxxx tools/fetch_tnt_raw.sh Truck
#   TNT_URL=https://... tools/fetch_tnt_raw.sh Truck
#   tools/fetch_tnt_raw.sh Truck --list-only   # just inspect the archive, no extract
#
# Re-running is safe: the download resumes, and extraction refuses to clobber a
# non-empty images/ dir unless you pass --force.

set -euo pipefail

SCENE=${1:?usage: [TNT_GDRIVE_ID=..|TNT_GDRIVE_FOLDER=..|TNT_URL=..] tools/fetch_tnt_raw.sh <Scene> [--list-only] [--force]}
shift || true

LIST_ONLY=0
FORCE=0
for a in "$@"; do
    case "$a" in
        --list-only) LIST_ONLY=1 ;;
        --force)     FORCE=1 ;;
        *) echo "unknown flag: $a" >&2; exit 2 ;;
    esac
done

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
PY=/scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/python
DL=/scratch/_projets_/willow/1-lip-tracer-new/tnt-raw/_dl
OUT=$ROOT/data/tnt/${SCENE}_raw/images

mkdir -p "$DL" "$OUT"

# --- pick the source -------------------------------------------------------
n_src=0
[ -n "${TNT_GDRIVE_ID:-}" ]     && n_src=$((n_src+1))
[ -n "${TNT_GDRIVE_FOLDER:-}" ] && n_src=$((n_src+1))
[ -n "${TNT_URL:-}" ]           && n_src=$((n_src+1))
if [ "$n_src" -ne 1 ]; then
    echo "ERROR: set exactly one of TNT_GDRIVE_ID / TNT_GDRIVE_FOLDER / TNT_URL." >&2
    echo "  Get the link from https://www.tanksandtemples.org/download/ (Truck is in 'intermediate')." >&2
    exit 1
fi

# --- download (resumable) --------------------------------------------------
ARCHIVE=""
if [ -n "${TNT_URL:-}" ]; then
    ARCHIVE="$DL/$(basename "${TNT_URL%%\?*}")"
    [ "${ARCHIVE##*.}" = "$ARCHIVE" ] && ARCHIVE="$DL/${SCENE}_src.zip"
    echo "[dl] curl -> $ARCHIVE"
    curl -L -C - --fail --retry 5 --retry-delay 10 -o "$ARCHIVE" "$TNT_URL"
elif [ -n "${TNT_GDRIVE_FOLDER:-}" ]; then
    echo "[dl] gdown folder $TNT_GDRIVE_FOLDER -> $DL/${SCENE}_folder"
    "$PY" -m gdown --folder -O "$DL/${SCENE}_folder" "$TNT_GDRIVE_FOLDER"
    ARCHIVE=$(find "$DL/${SCENE}_folder" -maxdepth 2 -iname '*.zip' | head -1)
    [ -z "$ARCHIVE" ] && { echo "ERROR: no .zip found in downloaded folder" >&2; exit 1; }
    echo "[dl] using archive $ARCHIVE"
else
    ARCHIVE="$DL/${SCENE}_gid_${TNT_GDRIVE_ID}.zip"
    echo "[dl] gdown id $TNT_GDRIVE_ID -> $ARCHIVE"
    "$PY" -m gdown --id "$TNT_GDRIVE_ID" -O "$ARCHIVE" --continue
fi

echo "[dl] done: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

# --- verify the scene is actually in this archive --------------------------
echo "[verify] scenes present in archive:"
unzip -l "$ARCHIVE" 2>/dev/null | awk '{print $4}' | grep -v '^$' | cut -d/ -f1 | sort -u | sed 's/^/    /'

n_scene=$(unzip -l "$ARCHIVE" 2>/dev/null | awk '{print $4}' | grep -E "^${SCENE}/" | grep -icE '\.(jpg|jpeg|png)$' || true)
if [ "$n_scene" -eq 0 ]; then
    echo "ERROR: archive contains no ${SCENE}/*.{jpg,png} frames." >&2
    echo "  Truck is in the INTERMEDIATE set; training.zip (Barn/Church/.../Meetingroom) will NOT have it." >&2
    exit 1
fi
echo "[verify] ${SCENE}: $n_scene image frames in archive."

if [ "$LIST_ONLY" -eq 1 ]; then
    echo "[list-only] stopping before extract."
    exit 0
fi

# --- extract just this scene, flattened into images/ -----------------------
if [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && [ "$FORCE" -eq 0 ]; then
    echo "ERROR: $OUT is not empty. Re-run with --force to overwrite." >&2
    exit 1
fi

TMP=$(mktemp -d "$DL/${SCENE}_extract.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
echo "[extract] unzip ${SCENE}/* -> $TMP"
unzip -q "$ARCHIVE" "${SCENE}/*" -d "$TMP"

# flatten: the canonical layout is data/tnt/<Scene>_raw/images/000001.jpg
echo "[extract] flattening into $OUT"
find "$TMP/$SCENE" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    -exec mv -t "$OUT" {} +

n_out=$(find "$OUT" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)
echo "[extract] $n_out frames now in $OUT"
echo "  sample: $(ls "$OUT" | head -3 | tr '\n' ' ')"

cat <<NEXT

Next steps (the migration pipeline you already have):
  sbatch colmap_sfm_raw.slurm $SCENE sequential
  tools/rebuild_tnt_scene.sh $SCENE        # archives NSVF -> ${SCENE}_nsvf, gated on coherence check
  # then re-run GSAM + MVSFormer++ on the clean frames before the conservative hull carve
NEXT
