#!/bin/bash
# Tournament evaluation of trained Speed Hex / Speed Go base planners.
# Plays them against each other across seeds and MCTS-sim budgets.

set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHONPATH=clock/networks:. python clock/eval/base_planner/tournament.py "$@"
