#!/bin/bash
# Download the checkpoint bundle from the Google Drive mirror
# and place it under ./checkpoints/.
#
# After this script completes, ./checkpoints/ should contain:
#   checkpoints/committed_action/{pacman,tetris_rt,snake}/{base,gating}/...
#   checkpoints/clock/{hex,go}/{base,gating}/...
#
# Override DRIVE_FOLDER_ID if you have been pointed at a different bundle.

set -euo pipefail
cd "$(dirname "$0")/.."

DRIVE_FOLDER_ID=${DRIVE_FOLDER_ID:-1Le7rZy1pPxGL021hviyet04ggH-xha78}

if ! command -v gdown >/dev/null 2>&1; then
  echo "[download_checkpoints] gdown is not installed. Install with:" >&2
  echo "    pip install gdown" >&2
  exit 1
fi

mkdir -p checkpoints
gdown --folder "${DRIVE_FOLDER_ID}" -O checkpoints --remaining-ok

echo "[download_checkpoints] checkpoints/ populated:"
find checkpoints -maxdepth 4 -type f -name '*.pkl' -o -name '*.ckpt' | sort
