#!/bin/bash
# real-time deployment for snake; wraps deploy.sh (needs two gpus)
#   FPS=9 bash committed_action/deployment/deploy_snake.sh

set -euo pipefail
GAME=snake exec "$(dirname "$0")/deploy.sh" "$@"
