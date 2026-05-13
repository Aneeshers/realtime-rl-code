"""Gating network for adaptive MCTS depth selection.

GatingNet is a CNN-based policy/value network that chooses between 4 MCTS depth
options (K=1,2,3,4 corresponding to sim=32,64,96,128) at each meta-decision point.

Architecture follows the AZNet backbone pattern (ResBlocks + LayerNorm) but is
lighter: 64 channels, 3 blocks (vs AZNet's 128ch, 6 blocks). The gating decision
only needs to detect position difficulty, not predict full policy/value, so lower
capacity suffices. Also takes frozen AZNet intermediate features as input.

Inputs:
  - obs_grid:        (B, 31, 28, 6)  — PacMan encoded observation
  - time_vec:        (B, 2)          — [time_seen_frac, time_left_frac]
  - az_trunk_feats:  (B, 128)        — global-avg-pooled AZNet trunk features (frozen)
  - az_value:        (B, 1)          — AZNet value estimate (frozen)

Outputs:
  - logits: (B, 4)  — unnormalized log-probabilities over {K=1, K=2, K=3, K=4}
  - value:  (B,)    — meta-level value estimate V_gating(s)
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Building block: ResBlockV2 with LayerNorm
# ---------------------------------------------------------------------------

class _ResBlockV2_LN(hk.Module):
    """Pre-activation ResBlock with LayerNorm (no running statistics).

    LayerNorm normalizes across the channel dimension at each spatial location,
    giving consistent behavior between single-sample rollout collection and
    mini-batch PPO updates (unlike BatchNorm which requires running statistics).
    """

    def __init__(self, num_channels: int, name: str = "resblock_v2_ln"):
        super().__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        i = x
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        return x + i


# ---------------------------------------------------------------------------
# Gating network
# ---------------------------------------------------------------------------

class GatingNet(hk.Module):
    """CNN gating network that chooses MCTS depth option K ∈ {1, 2, 3, 4}.

    Lighter than AZNet: 64 channels, 3 ResBlocks (vs 128ch, 6 blocks).
    Uses LayerNorm (no running stats). Fuses frozen AZNet trunk features
    and value estimate to inform the gating decision.
    """

    NUM_GATING_OPTIONS = 4   # K ∈ {1, 2, 3, 4}

    def __init__(
        self,
        num_channels: int = 64,
        num_blocks: int = 3,
        time_embed_dim: int = 32,
        az_feature_dim: int = 128,
        name: str = "gating_net",
    ):
        super().__init__(name=name)
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.time_embed_dim = time_embed_dim
        self.az_feature_dim = az_feature_dim

    def __call__(
        self,
        obs_grid: jnp.ndarray,        # (B, H, W, C)
        time_vec: jnp.ndarray,         # (B, 2)
        az_trunk_feats: jnp.ndarray,   # (B, az_feature_dim)
        az_value: jnp.ndarray,         # (B,) or (B, 1)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            logits: (B, 4) — one logit per K option
            value:  (B,)   — gating value estimate
        """
        obs_grid = obs_grid.astype(jnp.float32)
        time_vec = time_vec.astype(jnp.float32)
        az_trunk_feats = az_trunk_feats.astype(jnp.float32)
        az_value = az_value.astype(jnp.float32)

        # -- Own spatial backbone --
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(obs_grid)
        for i in range(self.num_blocks):
            x = _ResBlockV2_LN(self.num_channels, name=f"block_{i}")(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)

        # Time embedding: (B, 2) → (B, num_channels), broadcast + add to spatial trunk
        t = hk.Linear(self.time_embed_dim)(time_vec)
        t = jax.nn.relu(t)
        t = hk.Linear(self.num_channels)(t)
        t = jax.nn.relu(t)
        x = x + t[:, None, None, :]  # (B, H, W, num_channels)

        # Global average pool own spatial features → (B, num_channels)
        x_pool = x.mean(axis=(1, 2))

        # Ensure az_value is (B, 1)
        if az_value.ndim == 1:
            az_value = az_value[:, None]

        # Fuse own features + frozen AZNet features + AZNet value
        # fused: (B, num_channels + az_feature_dim + 1)
        fused = jnp.concatenate([x_pool, az_trunk_feats, az_value], axis=-1)

        # -- Policy head: logits over 4 K-options --
        p = hk.Linear(256)(fused)
        p = jax.nn.relu(p)
        logits = hk.Linear(self.NUM_GATING_OPTIONS)(p)  # (B, 4)

        # -- Value head: scalar meta-value estimate --
        v = hk.Linear(256)(fused)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        value = v.reshape((-1,))  # (B,)

        return logits, value


# ---------------------------------------------------------------------------
# Training-state container
# ---------------------------------------------------------------------------

class GatingParamsState(NamedTuple):
    """Training state for the PPO gating policy.

    No net_state field: GatingNet uses LayerNorm which has no running statistics,
    so hk.transform (not hk.transform_with_state) is sufficient.
    """
    params: hk.Params
    opt_state: optax.OptState
