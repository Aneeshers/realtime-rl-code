#!/bin/bash
# real-time deployment for pacman; wraps deploy.sh (needs two gpus)
#   FPS=9 bash committed_action/deployment/deploy_pacman.sh

set -euo pipefail
GAME=pacman exec "$(dirname "$0")/deploy.sh" "$@"
