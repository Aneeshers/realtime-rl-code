# Checkpoint manifest

One base planner + one gating policy checkpoint per environment, about 200 MB
total, stored with git lfs. A clone (with git lfs installed) populates this
directory as follows.

## Committed-action (Jumanji)

| Path | Phase | Notes |
|--|--|--|
| `committed_action/pacman/base/training_state_best.pkl` | AlphaZero | PacManKT, action delay K=1 |
| `committed_action/pacman/gating/gating_state_best.pkl` | PPO gating | sims = {32,64,96,128} |
| `committed_action/tetris_rt/base/training_state_best.pkl` | AlphaZero | TetrisRTKT, K=1 |
| `committed_action/tetris_rt/gating/gating_state_best.pkl` | PPO gating | sims = {32,64,96,128} |
| `committed_action/snake/base/training_state_best.pkl` | AlphaZero | SnakeKT, K=3 |
| `committed_action/snake/gating/gating_state_best.pkl` | PPO gating | sims = {32,64,96,128} |

## Clock (pgx)

| Path | Phase | Notes |
|--|--|--|
| `clock/hex/base/base_planner.ckpt` | AlphaZero | 11x11, nsim=32 |
| `clock/hex/gating/gate.pkl` | PPO gating (GRU) | options = {2,8,32,128} |
| `clock/hex/gating_timeout/gate.pkl` | PPO gating (GRU, strict-timeout) | options = {2,8,32,128}; pairs with `clock/hex/base`; backs Appendix H |
| `clock/go/base/base_planner.ckpt` | AlphaZero | 9x9, nsim=16 |
| `clock/go/gating/gate.pkl` | PPO gating (GRU) | options = {16,32,64,96} |

## Notes

* The launcher scripts default to these exact paths. Override via
  `AZ_CKPT`, `GATING_CKPT`, `CKPT_ROOT`, `GATE_ROOT` environment variables.
* Other seeds and intermediate epoch checkpoints used in the paper's
  variance and ablation studies are not included to keep the bundle small;
  they can be regenerated from the included training scripts.
