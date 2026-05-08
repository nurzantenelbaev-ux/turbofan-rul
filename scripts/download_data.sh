#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download NASA C-MAPSS (Turbofan Engine Degradation Simulation Dataset)
# into data/raw/. The dataset is hosted on the NASA Open Data Portal and is
# distributed as a zip archive containing train/test/RUL files for FD001..FD004.
#
# Usage:
#     bash scripts/download_data.sh
# ---------------------------------------------------------------------------
set -euo pipefail

DATA_DIR="data/raw"
mkdir -p "${DATA_DIR}"

# Mirror — replace if NASA changes the URL.
URL="https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"
ARCHIVE="${DATA_DIR}/CMAPSSData.zip"

if [ -f "${DATA_DIR}/train_FD001.txt" ]; then
  echo "C-MAPSS already present in ${DATA_DIR} — nothing to do."
  exit 0
fi

echo "Downloading C-MAPSS from ${URL} ..."
if command -v curl >/dev/null 2>&1; then
  curl -L "${URL}" -o "${ARCHIVE}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${ARCHIVE}" "${URL}"
else
  echo "ERROR: neither curl nor wget is installed." >&2
  exit 1
fi

echo "Unzipping ..."
unzip -o "${ARCHIVE}" -d "${DATA_DIR}"
rm -f "${ARCHIVE}"

echo "Done. Contents of ${DATA_DIR}:"
ls -1 "${DATA_DIR}"
