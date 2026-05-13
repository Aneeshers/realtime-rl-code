# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
import wandb
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from network import AZNet

devices = jax.local_devices()
num_devices = len(devices)


class Config(BaseModel):
    env_id: pgx.EnvId = "go_9x9"
    seed: int = 0
    max_num_iters: int = 800
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5
    save_interval: int = 200
    save_dir: str = "./checkpoints/clock/go/base"

    class Config:
        extra = "forbid"


def read_hyperparameters_from_csv(task_id: int, csv_path: str) -> dict:
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader, start=1):
            if i == task_id:
                return {k: int(v) for k, v in row.items()}
    return None


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_simulations",
    type=int,
    default=None,
    help="Number of MCTS simulations used in self-play.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility.",
)
parser.add_argument(
    "--task_id",
    type=int,
    default=None,
    help="SLURM array task ID for reading hyperparameters from CSV.",
)
parser.add_argument(
    "--hyperparameters_csv",
    type=str,
    default="./train/train_base_model/slurm_scripts/hyperparameters_base.csv",
    help="Path to CSV file containing hyperparameters.",
)
args, unknown = parser.parse_known_args()

conf_dict = OmegaConf.from_cli(unknown)

if args.task_id is not None:
    hyperparameters = read_hyperparameters_from_csv(args.task_id, args.hyperparameters_csv)
    if hyperparameters is not None:
        conf_dict["num_simulations"] = hyperparameters["num_simulations"]
        conf_dict["seed"] = hyperparameters["seed"]
        print(f"Loaded hyperparameters from CSV for task {args.task_id}: {hyperparameters}")
    else:
        raise ValueError(f"No hyperparameters found for task ID {args.task_id}")

if args.num_simulations is not None:
    conf_dict["num_simulations"] = args.num_simulations
if args.seed is not None:
    conf_dict["seed"] = args.seed

config: Config = Config(**conf_dict)
print(config)

env = pgx.make(config.env_id)
baseline = pgx.make_baseline_model(config.env_id + "_v0")


def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
optimizer = optax.adam(learning_rate=config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(
        model_params, model_state, state.observation, is_eval=True
    )
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray
    step_count: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray):
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    step_env = auto_reset(env.step, env.init)

    def step_fn(state, key) -> tuple[pgx.State, SelfplayOutput]:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)

        state = jax.vmap(step_env)(state, policy_output.action, keys)

        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
            step_count=state._step_count,
        )

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    final_state, data = jax.lax.scan(step_fn, state, key_seq)

    return data, final_state


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)

    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss


@jax.pmap
def evaluate(rng_key, my_model):
    my_player = 0
    my_model_params, my_model_state = my_model

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, r_sum = val
        (my_logits, _), _ = forward.apply(
            my_model_params, my_model_state, state.observation, is_eval=True
        )
        opp_logits, _ = baseline(state.observation)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state = jax.vmap(env.step)(state, action)
        r_sum = r_sum + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, r_sum)

    _, _, r_sum = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return r_sum


if __name__ == "__main__":
    now_dt = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now_dt.strftime("%Y%m%d%H%M%S")

    run_name = f"{config.env_id}_nsim{config.num_simulations}_{now}"
    wandb.init(project="go9x9-base", config=config.model_dump(), name=run_name, dir=config.save_dir)

    print("JAX devices:", jax.local_devices())
    print("num_devices:", num_devices)

    master_rng_key = jax.random.PRNGKey(config.seed)

    master_rng_key, init_key1, init_key2 = jax.random.split(master_rng_key, 3)
    dummy_state = jax.vmap(env.init)(jax.random.split(init_key1, 2))
    dummy_input = dummy_state.observation
    model = forward.init(init_key2, dummy_input)
    opt_state = optimizer.init(params=model[0])
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    ckpt_dir = os.path.join(
        config.save_dir,
        f"nsim_{config.num_simulations}",
        str(config.seed),
    )
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    master_rng_key, rng_key = jax.random.split(master_rng_key)
    while True:
        if iteration % config.eval_interval == 0:
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            r_sum = evaluate(keys, model)
            log.update(
                {
                    "eval/vs_baseline/avg_R": r_sum.mean().item(),
                    "eval/vs_baseline/win_rate": ((r_sum == 1).sum() / r_sum.size).item(),
                    "eval/vs_baseline/draw_rate": ((r_sum == 0).sum() / r_sum.size).item(),
                    "eval/vs_baseline/lose_rate": ((r_sum == -1).sum() / r_sum.size).item(),
                }
            )

        if iteration % config.save_interval == 0:
            model_0, opt_state_0 = jax.tree_util.tree_map(
                lambda x: x[0], (model, opt_state)
            )
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "pgx.__version__": pgx.__version__,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)
            print(f"Saved checkpoint to {ckpt_dir}/{iteration:06d}.ckpt")

        print(log)
        wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data, final_state = selfplay(model, keys)
        samples: Sample = compute_loss_input(data)

        data_host: SelfplayOutput = jax.device_get(data)
        t_max = config.max_num_steps
        last_t = t_max - 1

        terminated = data_host.terminated[:, last_t, :]
        reward_last = data_host.reward[:, last_t, :]
        step_count_last = data_host.step_count[:, last_t, :]

        term_flat = terminated.reshape(-1).astype(jnp.bool_)
        reward_flat = reward_last.reshape(-1)
        step_count_flat = step_count_last.reshape(-1)
        total_envs = term_flat.shape[0]

        term_rate = float(jnp.mean(term_flat.astype(jnp.float32)))
        trunc_rate = 1.0 - term_rate

        timed_out_mask = jnp.zeros_like(term_flat, dtype=jnp.bool_)
        env_max_mask = term_flat & (step_count_flat >= config.max_num_steps)

        non_timeout_non_max = term_flat & ~timed_out_mask & ~env_max_mask
        win_mask = non_timeout_non_max & (reward_flat > 0.0)
        loss_mask = non_timeout_non_max & (reward_flat < 0.0)
        draw_mask = non_timeout_non_max & jnp.isclose(reward_flat, 0.0)

        def frac(mask):
            return float(jnp.sum(mask.astype(jnp.float32)) / float(total_envs))

        timeout_rate = frac(timed_out_mask)
        env_max_rate = frac(env_max_mask)
        board_win_rate = frac(win_mask)
        board_loss_rate = frac(loss_mask)
        board_draw_rate = frac(draw_mask)

        avg_step_count = float(jnp.mean(final_state._step_count))

        num_players = final_state.rewards.shape[-1]
        avg_move_count_per_player = avg_step_count / float(num_players)

        term_float = term_flat.astype(jnp.float32)
        non_term_float = 1.0 - term_float

        avg_step_count_terminated = float(
            jnp.sum(step_count_flat * term_float) / jnp.maximum(jnp.sum(term_float), 1.0)
        )
        avg_step_count_truncated = float(
            jnp.sum(step_count_flat * non_term_float) / jnp.maximum(jnp.sum(non_term_float), 1.0)
        )

        samples = jax.device_get(samples)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[3:])), samples
        )
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600

        log.update(
            {
                "hours": hours,
                "frames": frames,
                "loss/policy": policy_loss,
                "loss/value": value_loss,
                "selfplay/term_rate": term_rate,
                "selfplay/trunc_rate": trunc_rate,
                "selfplay/timeout_rate": timeout_rate,
                "selfplay/env_max_rate": env_max_rate,
                "selfplay/board_win_rate": board_win_rate,
                "selfplay/board_loss_rate": board_loss_rate,
                "selfplay/board_draw_rate": board_draw_rate,
                "selfplay/avg_step_count": avg_step_count,
                "selfplay/avg_move_count_per_player": avg_move_count_per_player,
                "selfplay/avg_step_count_terminated": avg_step_count_terminated,
                "selfplay/avg_step_count_truncated": avg_step_count_truncated,
            }
        )
