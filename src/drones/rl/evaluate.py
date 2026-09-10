"""Evaluate a trained hover policy against open-loop hover.

    uv run drones-eval-hover runs/<name>

Runs every world for one full episode with deterministic actions, then again with zero action (the
calibrated hover thrust and no feedback), and prints both. A policy that has learned anything
should crash less, drift less and hold height better than the baseline.
"""
import argparse
import json
import sys
from pathlib import Path


def evaluate(env, act, key):
    """Run each world for one episode. `act(obs)` returns actions. Returns summary numbers."""
    import jax
    import jax.numpy as jnp

    state, obs = env.reset(key)

    def body(carry, _):
        state, obs, alive = carry
        state, obs, reward, done, info = env.step(state, act(obs))
        # Only the first episode of each world counts: stop recording once it ends.
        record = alive
        alive = alive & ~done
        return (state, obs, alive), (record, reward, info)

    alive = jnp.ones(env.num_envs, bool)
    _, (record, reward, info) = jax.lax.scan(body, (state, obs, alive), None,
                                             length=env.config.episode_steps)
    settled = jnp.arange(env.config.episode_steps)[:, None] >= 2 * env.config.control_freq
    mask = record & settled   # ignore the first 2 s while the start disturbance is recovered
    weight = mask / jnp.maximum(mask.sum(), 1)
    return {
        'crash_rate': float((info['crashed'] & record).any(0).mean()),
        'survived_seconds': float(record.sum(0).mean() / env.config.control_freq),
        'return': float((reward * record).sum(0).mean()),
        'height_error_m': float((info['height_error'] * weight).sum()),
        'speed_m_s': float((info['speed'] * weight).sum()),
        'tilt_deg': float((info['tilt'] * weight).sum() * 180 / 3.141592653589793),
    }


def load_run(run, num_envs, device):
    """The env, model and trained parameters of a run directory. Call under jax.default_device."""
    import jax

    from drones.rl.networks import ActorCritic
    from drones.rl.ppo import PPOConfig, config_from_dict, load_params
    from drones.sim.hover_env import HoverConfig, HoverEnv

    saved = json.loads((Path(run) / 'config.json').read_text())
    env = HoverEnv(config_from_dict(HoverConfig, {**saved['env'], 'num_envs': num_envs,
                                                  'device': device}))
    ppo_config = config_from_dict(PPOConfig, saved['ppo'])
    model = ActorCritic(env.action_size, ppo_config.hidden, use_image=env.image_shape is not None)
    _, obs = env.reset(jax.random.key(0))
    params = load_params(Path(run) / 'params.msgpack', model.init(jax.random.key(0), obs))
    return env, model, params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path, help='run directory written by drones-train-hover')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
    except ImportError:
        sys.exit('Evaluation needs the sim extra:  uv sync --extra sim')
    import jax
    import jax.numpy as jnp

    from drones.rl.networks import ActorCritic

    with jax.default_device(jax.devices(args.device)[0]):
        env, model, params = load_run(args.run, args.num_envs, args.device)
        policy = jax.jit(lambda o: model.apply(params, o, method=ActorCritic.act))
        key = jax.random.key(args.seed)
        results = {
            'policy': evaluate(env, policy, key),
            'open-loop hover': evaluate(env, lambda o: jnp.zeros((env.num_envs, 4)), key),
        }

    print(f'Sensors: {", ".join(env.config.sensors.enabled)}')
    names = list(results['policy'])
    print(f'{"":18s}' + ''.join(f'{n:>17s}' for n in names))
    for label, row in results.items():
        print(f'{label:18s}' + ''.join(f'{row[n]:17.3f}' for n in names))


if __name__ == '__main__':
    main()
