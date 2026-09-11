"""Short-Horizon Actor-Critic, compiled end to end with JAX.

Xu et al., "Accelerated Policy Learning with Parallel Differentiable Simulation", ICLR 2022. The
actor is trained by backpropagating the return of a short rollout, `horizon` steps, through the
simulator, with a critic's value at the end of the window standing in for the rest of the episode.
Short windows keep the gradients from exploding through long rollouts; the critic keeps the policy
from being short-sighted. The critic is trained on TD(lambda) targets from the same rollout,
bootstrapped from a slowly updated target copy.

Works with any env that has `num_envs`, `action_size`, `reset(key)` and a differentiable
`step(state, action)` reporting `crashed`, `truncated` and `final_critic` (the critic observation of
the state an episode ended in) in its info, as drones.sim.square_env.SquareEnv does.
"""
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

from drones.rl.networks import MLP


@dataclass(frozen=True)
class SHACConfig:
    iterations: int = 2000
    horizon: int = 32
    gamma: float = 0.99
    td_lambda: float = 0.95
    actor_lr: float = 2e-3
    critic_lr: float = 5e-4
    lr_decay: bool = True
    betas: tuple[float, float] = (0.7, 0.95)
    max_grad_norm: float = 1.0
    critic_epochs: int = 16
    critic_minibatches: int = 4
    target_alpha: float = 0.2      # target <- alpha * target + (1 - alpha) * critic
    init_log_std: float = -1.0
    hidden: tuple[int, ...] = (128, 128)
    remat: bool = True             # recompute env steps in the backward pass to save memory
    seed: int = 0


class Actor(nn.Module):
    action_size: int
    hidden: tuple[int, ...]
    init_log_std: float = -1.0

    def setup(self):
        # A small output scale starts the policy near zero action, which is calibrated hover.
        self.net = MLP(self.hidden, self.action_size, out_scale=0.01)
        self.log_std = self.param('log_std', nn.initializers.constant(self.init_log_std),
                                  (self.action_size,))

    def __call__(self, obs):
        """(action mean, log std) from the policy observation."""
        return self.net(obs), self.log_std


class Critic(nn.Module):
    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, obs):
        return MLP(self.hidden, 1, name='net')(obs)[..., 0]


@struct.dataclass
class SHACState:
    actor: dict
    critic: dict
    target: dict
    actor_opt: optax.OptState
    critic_opt: optax.OptState
    env_state: object
    obs: dict
    key: jax.Array
    iteration: jax.Array


def params_of(state):
    """What a checkpoint holds: the actor, the critic and the target critic."""
    return {'actor': state.actor, 'critic': state.critic, 'target': state.target}


def td_lambda_returns(rewards, next_values, dones, crashed, gamma, lam):
    """TD(lambda) targets over time-major arrays (T, n).

    `next_values[t]` is the value of the state reached by step t, before any restart. `dones[t]`
    means the episode ended there, and `crashed[t]` that it ended in a crash, which has no future.
    The last step bootstraps fully from its next value.
    """
    def step(next_return, x):
        reward, next_value, done, crash = x
        blended = (1.0 - lam) * next_value + lam * next_return
        ret = reward + gamma * jnp.where(done, next_value * (1.0 - crash), blended)
        return ret, ret

    _, returns = jax.lax.scan(step, next_values[-1],
                              (rewards, next_values, dones, crashed.astype(rewards.dtype)),
                              reverse=True)
    return returns


class SHAC:
    def __init__(self, env, config: SHACConfig = SHACConfig()):
        self.env = env
        self.config = config
        self.actor = Actor(env.action_size, config.hidden, config.init_log_std)
        self.critic = Critic(config.hidden)
        self.batch_size = env.num_envs * config.horizon
        if self.batch_size % config.critic_minibatches:
            raise ValueError('num_envs * horizon must divide into critic_minibatches')
        critic_updates = config.iterations * config.critic_epochs * config.critic_minibatches
        actor_lr, critic_lr = config.actor_lr, config.critic_lr
        if config.lr_decay:
            actor_lr = optax.linear_schedule(actor_lr, 0.0, config.iterations)
            critic_lr = optax.linear_schedule(critic_lr, 0.0, critic_updates)
        b1, b2 = config.betas
        self.actor_opt = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                     optax.adam(actor_lr, b1=b1, b2=b2))
        self.critic_opt = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                      optax.adam(critic_lr, b1=b1, b2=b2))
        self._env_step = jax.checkpoint(env.step) if config.remat else env.step
        self.iterate = jax.jit(self._iterate)

    def init(self, key):
        key, k_env, k_actor, k_critic = jax.random.split(key, 4)
        env_state, obs = self.env.reset(k_env)
        actor = self.actor.init(k_actor, obs['policy'])
        critic = self.critic.init(k_critic, obs['critic'])
        return SHACState(actor=actor, critic=critic, target=critic,
                         actor_opt=self.actor_opt.init(actor),
                         critic_opt=self.critic_opt.init(critic), env_state=env_state, obs=obs,
                         key=key, iteration=jnp.zeros((), jnp.int32))

    def act(self, actor_params, obs):
        """Deterministic action, the policy mean. Reads only obs['policy']: deployable."""
        return self.actor.apply(actor_params, obs['policy'])[0]

    def _rollout(self, actor_params, target_params, env_state, obs, key):
        """The actor loss of one window, differentiable in `actor_params`.

        Returns (loss, (env_state, obs, trajectory)); the trajectory is gradient-stopped.
        Each world's return is summed from the window start or from its latest restart. When an
        episode ends in the window, the target critic's value of the state it ended in closes it,
        unless it crashed. The window's end is closed the same way.
        """
        cfg, n = self.config, self.env.num_envs

        def value(o):
            return self.critic.apply(target_params, o)

        def body(carry, eps):
            env_state, obs, discount, running, total = carry
            mean, log_std = self.actor.apply(actor_params, obs['policy'])
            action = mean + jnp.exp(log_std) * eps
            env_state, next_obs, reward, done, info = self._env_step(env_state, action)
            crashed = info['crashed'].astype(reward.dtype)
            next_value = value(info['final_critic'])
            running = running + discount * reward
            discount = discount * cfg.gamma
            closed = running + discount * next_value * (1.0 - crashed)
            total = total + jnp.sum(jnp.where(done, closed, 0.0))
            running = jnp.where(done, 0.0, running)
            discount = jnp.where(done, 1.0, discount)
            record = {'critic_obs': obs['critic'], 'reward': reward, 'done': done,
                      'crashed': info['crashed'], 'next_value': next_value,
                      'info': {k: v for k, v in info.items() if k != 'final_critic'}}
            return (env_state, next_obs, discount, running, total), record

        noise = jax.random.normal(key, (cfg.horizon, n, self.env.action_size))
        init = (env_state, obs, jnp.ones(n), jnp.zeros(n), jnp.zeros(()))
        (env_state, obs, discount, running, total), traj = jax.lax.scan(body, init, noise)
        total = total + jnp.sum(running + discount * value(obs['critic']))
        loss = -total / (n * cfg.horizon)
        return loss, (env_state, obs, jax.lax.stop_gradient(traj))

    def _iterate(self, ts):
        """One iteration: a differentiable window, an actor step, then critic fitting."""
        cfg = self.config
        key, k_noise, k_critic = jax.random.split(ts.key, 3)
        # The window starts from where the last one ended, but no gradient flows back across it.
        env_state = jax.lax.stop_gradient(ts.env_state)
        obs = jax.lax.stop_gradient(ts.obs)
        (loss, (env_state, obs, traj)), grads = jax.value_and_grad(self._rollout, has_aux=True)(
            ts.actor, ts.target, env_state, obs, k_noise)

        grad_norm = optax.global_norm(grads)
        finite = jnp.isfinite(grad_norm)
        updates, actor_opt = self.actor_opt.update(grads, ts.actor_opt, ts.actor)
        actor = optax.apply_updates(ts.actor, updates)

        def keep_if_finite(new, old):
            return jax.tree.map(lambda a, b: jnp.where(finite, a, b), new, old)

        actor, actor_opt = keep_if_finite(actor, ts.actor), keep_if_finite(actor_opt, ts.actor_opt)

        returns = td_lambda_returns(traj['reward'], traj['next_value'], traj['done'],
                                    traj['crashed'], cfg.gamma, cfg.td_lambda)
        critic, critic_opt, critic_loss = self._fit_critic(ts.critic, ts.critic_opt,
                                                           traj['critic_obs'], returns, k_critic)
        target = jax.tree.map(lambda t, c: cfg.target_alpha * t + (1 - cfg.target_alpha) * c,
                              ts.target, critic)

        info, done = traj['info'], traj['done']
        finished = done.sum()

        def per_episode(x):
            return jnp.where(finished > 0, x.sum() / jnp.maximum(finished, 1), jnp.nan)

        stats = {
            'actor_loss': loss,
            'grad_norm': grad_norm,
            'skipped': (~finite).astype(jnp.float32),
            'critic_loss': critic_loss,
            'episodes': finished,
            'episode_return': per_episode(info['episode_return']),
            'episode_length': per_episode(info['episode_length'].astype(jnp.float32)),
            'crash_rate': per_episode(info['crashed'].astype(jnp.float32)),
            'reward': traj['reward'].mean(),
            'pos_error': info['pos_error'].mean(),
            'speed': info['speed'].mean(),
            'tilt': info['tilt'].mean(),
            'action_std': jnp.exp(actor['params']['log_std']).mean(),
        }
        return ts.replace(actor=actor, critic=critic, target=target, actor_opt=actor_opt,
                          critic_opt=critic_opt, env_state=env_state, obs=obs, key=key,
                          iteration=ts.iteration + 1), stats

    def _fit_critic(self, params, opt_state, obs, returns, key):
        """Regress the critic onto the TD(lambda) targets: epochs of shuffled minibatches."""
        cfg = self.config
        batch = (obs.reshape(self.batch_size, -1), returns.reshape(-1))

        def epoch(carry, key):
            order = jax.random.permutation(key, self.batch_size)
            shuffled = jax.tree.map(
                lambda x: x[order].reshape(cfg.critic_minibatches, -1, *x.shape[1:]), batch)

            def minibatch(carry, mb):
                params, opt_state = carry
                o, target = mb
                loss, grads = jax.value_and_grad(
                    lambda p: jnp.mean(jnp.square(self.critic.apply(p, o) - target)))(params)
                updates, opt_state = self.critic_opt.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), loss

            carry, losses = jax.lax.scan(minibatch, carry, shuffled)
            return carry, losses.mean()

        (params, opt_state), losses = jax.lax.scan(epoch, (params, opt_state),
                                                   jax.random.split(key, cfg.critic_epochs))
        return params, opt_state, losses[-1]
