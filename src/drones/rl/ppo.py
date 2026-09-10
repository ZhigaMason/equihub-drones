"""Proximal policy optimisation, compiled end to end with JAX.

Rollout, advantage estimation and every update of an iteration run inside one jitted function, so
training has no Python in the inner loop. Works with any env exposing `num_envs`, `action_size`,
`image_shape`, `reset(key)` and `step(state, action)` with dict observations, as
`drones.sim.hover_env.HoverEnv` does.
"""
import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import serialization, struct

from drones.rl.networks import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 50_000_000
    rollout_steps: int = 64
    epochs: int = 4
    minibatches: int = 8
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: tuple[int, ...] = (256, 256)
    seed: int = 0


class Transition(struct.PyTreeNode):
    obs: dict
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


@struct.dataclass
class TrainState:
    params: dict
    opt_state: optax.OptState
    env_state: object
    obs: dict
    key: jax.Array
    iteration: jax.Array


def gaussian_log_prob(action, mean, log_std):
    return jnp.sum(-0.5 * jnp.square((action - mean) / jnp.exp(log_std))
                   - log_std - 0.5 * math.log(2 * math.pi), -1)


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimates and returns, over time-major arrays (T, n).

    `dones[t]` means the episode ended with the reward at t, so the value of the next observation
    (already the start of a new episode) is not bootstrapped across it.
    """
    def step(carry, x):
        advantage, next_value = carry
        reward, value, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        advantage = delta + gamma * lam * (1 - done) * advantage
        return (advantage, value), advantage

    _, advantages = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value),
                                 (rewards, values, dones.astype(rewards.dtype)), reverse=True)
    return advantages, advantages + values


class PPO:
    def __init__(self, env, config: PPOConfig = PPOConfig()):
        self.env = env
        self.config = config
        self.model = ActorCritic(env.action_size, config.hidden,
                                 use_image=env.image_shape is not None)
        self.batch_size = env.num_envs * config.rollout_steps
        if self.batch_size % config.minibatches:
            raise ValueError('num_envs * rollout_steps must divide into minibatches')
        self.iterations = max(1, config.total_steps // self.batch_size)
        updates = self.iterations * config.epochs * config.minibatches
        lr = (optax.linear_schedule(config.learning_rate, 0.0, updates)
              if config.anneal_lr else config.learning_rate)
        self.optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                     optax.adam(lr, eps=1e-5))
        self.iterate = jax.jit(self._iterate)

    def init(self, key):
        key, k_env, k_model = jax.random.split(key, 3)
        env_state, obs = self.env.reset(k_env)
        params = self.model.init(k_model, obs)
        return TrainState(params=params, opt_state=self.optimizer.init(params),
                          env_state=env_state, obs=obs, key=key,
                          iteration=jnp.zeros((), jnp.int32))

    def _iterate(self, ts):
        """One iteration: collect a rollout, then update. Returns (train state, statistics)."""
        cfg = self.config

        def env_step(ts, _):
            key, k_action = jax.random.split(ts.key)
            mean, log_std, value = self.model.apply(ts.params, ts.obs)
            action = mean + jnp.exp(log_std) * jax.random.normal(k_action, mean.shape)
            env_state, obs, reward, done, info = self.env.step(ts.env_state, action)
            transition = Transition(ts.obs, action, gaussian_log_prob(action, mean, log_std),
                                    value, reward, done, info)
            return ts.replace(env_state=env_state, obs=obs, key=key), transition

        ts, traj = jax.lax.scan(env_step, ts, None, length=cfg.rollout_steps)
        last_value = self.model.apply(ts.params, ts.obs, method=ActorCritic.value)
        advantages, returns = gae(traj.reward, traj.value, traj.done, last_value,
                                  cfg.gamma, cfg.gae_lambda)

        batch = (traj.obs, traj.action, traj.log_prob, traj.value, advantages, returns)
        batch = jax.tree.map(lambda x: x.reshape(self.batch_size, *x.shape[2:]), batch)

        def epoch(carry, _):
            params, opt_state, key = carry
            key, k_perm = jax.random.split(key)
            order = jax.random.permutation(k_perm, self.batch_size)
            shuffled = jax.tree.map(
                lambda x: x[order].reshape(cfg.minibatches, -1, *x.shape[1:]), batch)

            def minibatch(carry, mb):
                params, opt_state = carry
                grads, metrics = jax.grad(self._loss, has_aux=True)(params, mb)
                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), metrics

            (params, opt_state), metrics = jax.lax.scan(minibatch, (params, opt_state), shuffled)
            return (params, opt_state, key), metrics

        (params, opt_state, key), metrics = jax.lax.scan(
            epoch, (ts.params, ts.opt_state, ts.key), None, length=cfg.epochs)

        finished = traj.done.sum()
        per_episode = lambda x: jnp.where(finished > 0, x.sum() / jnp.maximum(finished, 1), jnp.nan)
        stats = {k: v.mean() for k, v in metrics.items()}
        stats.update(
            episodes=finished,
            episode_return=per_episode(traj.info['episode_return']),
            episode_length=per_episode(traj.info['episode_length'].astype(jnp.float32)),
            crash_rate=per_episode(traj.info['crashed'].astype(jnp.float32)),
            reward=traj.reward.mean(),
            height_error=traj.info['height_error'].mean(),
            speed=traj.info['speed'].mean(),
            tilt=traj.info['tilt'].mean(),
            action_std=jnp.exp(params['params']['log_std']).mean(),
        )
        return ts.replace(params=params, opt_state=opt_state, key=key,
                          iteration=ts.iteration + 1), stats

    def _loss(self, params, minibatch):
        cfg = self.config
        obs, action, old_log_prob, old_value, advantage, target = minibatch
        mean, log_std, value = self.model.apply(params, obs)
        log_ratio = gaussian_log_prob(action, mean, log_std) - old_log_prob
        ratio = jnp.exp(log_ratio)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        policy_loss = -jnp.minimum(
            ratio * advantage, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * advantage).mean()
        value_clipped = old_value + jnp.clip(value - old_value, -cfg.clip, cfg.clip)
        value_loss = 0.5 * jnp.maximum(jnp.square(value - target),
                                       jnp.square(value_clipped - target)).mean()
        entropy = jnp.sum(log_std + 0.5 * math.log(2 * math.pi * math.e))
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
        return loss, {
            'policy_loss': policy_loss, 'value_loss': value_loss, 'entropy': entropy,
            'approx_kl': ((ratio - 1) - log_ratio).mean(),
            'clip_fraction': (jnp.abs(ratio - 1) > cfg.clip).mean(),
        }


# ---------------------------------------------------------------------- checkpoints
def save_params(path, params):
    Path(path).write_bytes(serialization.to_bytes(params))


def load_params(path, template):
    return serialization.from_bytes(template, Path(path).read_bytes())


def config_to_json(config):
    return json.dumps(asdict(config), indent=2)


def config_from_dict(cls, data):
    """Rebuild a (possibly nested) frozen dataclass from JSON, restoring tuples and sub-configs."""
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        default = getattr(cls(), f.name)
        if is_dataclass(default):
            value = config_from_dict(type(default), value)
        elif isinstance(default, tuple):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)
