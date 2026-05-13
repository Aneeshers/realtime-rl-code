# Checkpoint manifest

The released bundle ships **one canonical (base planner + gating policy)
checkpoint per environment**, totalling ≈ 200 MB. After running
`scripts/download_checkpoints.sh`, this directory is populated as follows.

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
| `clock/hex/base/000800.ckpt` | AlphaZero | nsim=32, seed=3 |
| `clock/hex/gating/gate_001000.pkl` | PPO gating | options = {2,8,32,128}, seed=1 |
| `clock/go/base/000800.ckpt` | AlphaZero | 9×9, nsim=16, seed=0 |
| `clock/go/gating/gate_periodic_000275.pkl` | PPO gating | options = {16,32,64,96}, seed=0 |

## Notes

* The launcher scripts default to these exact paths. Override via
  `AZ_CKPT`, `GATING_CKPT`, `CKPT_ROOT`, `GATE_ROOT` environment variables.
* Other seeds and intermediate epoch checkpoints used in the paper's
  variance and ablation studies are not included to keep the bundle small;
  they can be regenerated from the included training scripts.
