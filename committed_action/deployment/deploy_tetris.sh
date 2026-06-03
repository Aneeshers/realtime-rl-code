#!/bin/bash
# real-time deployment for tetris; wraps deploy.sh (needs two gpus)
#   FPS=9 bash committed_action/deployment/deploy_tetris.sh

set -euo pipefail
GAME=tetris exec "$(dirname "$0")/deploy.sh" "$@"
