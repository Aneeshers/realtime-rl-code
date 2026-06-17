# Learning Planning Budgets for Real-Time RL: Code Release

Anonymous NeurIPS 2026 supplementary code for the paper
**"Learning Planning Budgets for Real-Time RL."**

The paper trains a lightweight **gating policy** on top of a frozen
AlphaZero-style planner that selects state-dependent MCTS planning budgets
at each decision point. Five environments split across two regimes:

| Regime | Environments | Source |
|--|--|--|
| Committed-action (real-time) | Pac-Man, Tetris (real-time), Snake | Jumanji |
| Clock | Speed Hex (11x11), Speed Go (9x9) | pgx |

Plus a two-GPU real-time deployment for the committed-action environments
(paper Section 6).

---

## Repository layout

```
.
├── committed_action/          # Pac-Man, real-time Tetris, Snake (Jumanji)
│   ├── jumanji/               # vendored Jumanji library (Apache-2.0)
│   ├── train/
│   │   ├── base_planner/      # AlphaZero base-planner training launchers
│   │   └── gating_policy/     # PPO gating-policy training launchers
│   ├── eval/
│   │   ├── base_planner/      # always-K cross-evaluation launchers
│   │   └── gating_policy/     # gating-policy evaluation launchers
│   └── deployment/            # two-GPU real-time deployment (Section 6)
│
├── clock/                     # Speed Hex, Speed Go (pgx)
│   ├── envs/                  # speed_hex.py, speed_go.py, speed_hex_timeout.py
│   ├── networks/              # AZNet variants
│   ├── train/{base_planner, gating_policy}
│   └── eval/{base_planner,  gating_policy}
│
├── checkpoints/               # model checkpoints (git lfs)
├── scripts/
│   └── download_checkpoints.sh  # optional google drive mirror
├── requirements.txt
└── README.md  (this file)
```

`committed_action/jumanji/` is a vendored snapshot of the upstream
[Jumanji](https://github.com/instadeepai/jumanji) library extended with the
real-time Pac-Man (`PacManKT`), real-time Tetris (`TetrisRTKT`), and Snake
(`SnakeKT`) variants. The pgx side imports
[pgx](https://github.com/sotetsuk/pgx) as a normal pip dependency.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Hardware: paper experiments used H100 / A100 / A40 GPUs. Single-GPU is
sufficient for everything except the two-GPU real-time deployment. CPU-only
execution has not been tested.

---

## Checkpoints

One base planner + one gating policy checkpoint per environment (about 200 MB
total), stored in the repo with [Git LFS](https://git-lfs.com). A normal clone
pulls them:

```bash
git lfs install        # one-time
git clone git@github.com:Aneeshers/realtime-rl-code.git
```

If you cloned before installing LFS (so `checkpoints/*.pkl` are pointer stubs),
run `git lfs pull`.

`./checkpoints/` then matches the layout the launcher scripts expect. See
`checkpoints/MANIFEST.md` for the file list. `scripts/download_checkpoints.sh`
is an optional Google Drive mirror of the same files.

---

## Reproducing main results

All launcher scripts are run from the repo root and accept overrides via
environment variables (no SLURM dependency). Defaults reproduce the paper's
numbers when paired with the shipped checkpoints.

### Pac-Man (committed-action)

```bash
bash committed_action/eval/gating_policy/eval_pacman_gating.sh
```

Expected: ~2370 mean episode return (vs. 2149 for the best fixed-K
baseline; paper Section 5).

### Real-time Tetris

```bash
bash committed_action/eval/gating_policy/eval_tetris_gating.sh
```

Expected: ~45.6 mean (vs. 27.6 fixed-K).

### Snake

```bash
bash committed_action/eval/gating_policy/eval_snake_gating.sh
```

Expected: ~16.5 mean (vs. 14.9 fixed-K).

### Speed Hex (clock)

```bash
bash clock/eval/gating_policy/eval_hex_gating.sh
```

Expected: averaged over `T in {300, 1200, 2300, 3500, 4100}`, expected
score ~0.58 (vs. 0.43 fixed-K, 0.46 greedy).

### Speed Go (clock)

```bash
bash clock/eval/gating_policy/eval_go_gating.sh
```

Expected: same clock budgets, expected score ~0.59 (vs. 0.51 fixed-K, 0.50
midpeak).

Fixed-budget baseline bars for both clock games come from
`clock/eval/gating_policy/eval_clock_baselines.sh` (`ENV_KIND=hex` or `go`),
which re-runs the eval with `--force_gate_choice` set to each budget. The
greedy/midpeak heuristics run as opponents only
(`--opponents ...,midpeak,proportional`), not as the evaluated player.

### Speed Hex strict-timeout (appendix)

```bash
bash clock/eval/gating_policy/eval_hex_gating_timeout.sh
```

Backs the appendix control (Appendix H) where running out of time = immediate
loss. Uses the shipped strict-timeout gate
(`clock/hex/gating_timeout/gate.pkl`) paired with the
same `clock/hex/base` AlphaZero planner, and evaluates against the
`midpeak` / `proportional` heuristic opponents reported in the appendix.

### Two-GPU real-time deployment (Section 6)

Requires two visible GPUs. Each committed-action game has a per-game launcher
(thin wrappers over `deploy.sh`):

```bash
FPS=9 bash committed_action/deployment/deploy_tetris.sh
FPS=9 bash committed_action/deployment/deploy_pacman.sh
FPS=9 bash committed_action/deployment/deploy_snake.sh
```

Equivalently, drive the generic launcher directly:

```bash
GAME=tetris FPS=9 bash committed_action/deployment/deploy.sh
GAME=pacman FPS=9 bash committed_action/deployment/deploy.sh
GAME=snake  FPS=9 bash committed_action/deployment/deploy.sh
```

The single Python entrypoint `jumanji.training.deploy_tetris_rt_realtime`
handles all three games via `--game` (`TetrisRTKT-v0`, `PacManKT-v1`,
`SnakeKT-v1`).

The released deployment is configured and verified for **H100** GPUs (the
default `GPU_TYPE=h100`). The paper also reports A100 and A40 results; if you
want to reproduce those, you can configure other GPU classes yourself by
overriding `GPU_TYPE` (and adjusting the `FPS` sweep to taste) - those paths
are left unsupported here.

---

## Training from scratch

Base AlphaZero planners (multi-day on a single H100):

```bash
K=1 bash committed_action/train/base_planner/train_pacman.sh
K=1 bash committed_action/train/base_planner/train_tetris.sh
K=3 bash committed_action/train/base_planner/train_snake.sh
NUM_SIMULATIONS=32 SEED=3 bash clock/train/base_planner/train_hex.sh
NUM_SIMULATIONS=16 SEED=0 bash clock/train/base_planner/train_go.sh
```

Gating policies (frozen base planner; faster than base):

```bash
bash committed_action/train/gating_policy/train_pacman_gating.sh
bash committed_action/train/gating_policy/train_tetris_gating.sh
bash committed_action/train/gating_policy/train_snake_gating.sh
bash clock/train/gating_policy/train_hex_gating.sh
bash clock/train/gating_policy/train_go_gating.sh
bash clock/train/gating_policy/train_hex_gating_timeout.sh   # appendix
```

Each launcher exposes hyperparameters via env vars (`SEED`, `NUM_EPOCHS`,
`SIM_OPTIONS`, etc.). See the script bodies for the full list and defaults.

---

## Caveats & known limitations

* Reproducibility is GPU-class-sensitive. Expect ~1 SE noise across H100 /
  A100 / A40; deployment latencies tighten on H100.
* The committed-action evaluators emit some matplotlib side-effect figures
  that are not used in the paper. They are harmless and write to
  `--output_dir`.
* `clock/eval/base_planner/tournament.py` performs a head-to-head sweep
  rather than reproducing a specific paper table; it is included for
  completeness of base-planner evaluation.
* W&B logging is on by default. Committed-action evaluators and deploy scripts
  log unless you pass `--no_wandb`; clock evaluators default to
  `--wandb_mode online` (pass `--wandb_mode disabled` to turn off). Set the
  `WANDB_ENTITY` / `WANDB_PROJECT` env vars to control where runs land.

---

## Citation

Withheld for the anonymous review period.

## Licenses

The vendored `committed_action/jumanji/` is released under Apache-2.0 by
its upstream authors; see the in-tree license header. All paper-specific
code is released under the same license.

