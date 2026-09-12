"""Evaluate a trained square policy against open-loop hover.

    uv run --extra sim drones-eval-square runs/<name>
    uv run --extra sim drones-eval-square runs/<name>-ft --uncorrected   # no fitted residual

Runs every world for one full episode with deterministic actions, then again with zero action, and
prints both. Tracking error is measured after the first second, once the start error is recovered.
"""
import argparse
import json
import sys
from pathlib import Path


def evaluate_square(env, act, key):
    """Run each world for one episode. `act(obs)` returns actions. Returns summary numbers."""
    import jax
    import jax.numpy as jnp

    state, obs = env.reset(key)
    lap_time = state.lap_time

    def body(carry, _):
        state, obs, alive = carry
        state, obs, reward, done, info = env.step(state, act(obs))
        # Only the first episode of each world counts: stop recording once it ends.
        record = alive
        return (state, obs, alive & ~done), (record, reward, info['crashed'], info['pos_error'])

    steps, freq = env.config.episode_steps, env.config.control_freq
    _, (record, reward, crashed, error) = jax.lax.scan(
        body, (state, obs, jnp.ones(env.num_envs, bool)), None, length=steps)
    mask = record & (jnp.arange(steps)[:, None] >= freq)
    survived = record.sum(0) / freq
    return {
        'crash_rate': float((crashed & record).any(0).mean()),
        'survived_seconds': float(survived.mean()),
        # Survival time over lap time, not measured path progress: a world that crashes off-course
        # still counts the seconds it stayed up, whichever way round the square it drifted.
        'laps': float((survived / lap_time).mean()),
        'pos_rmse_m': float(jnp.sqrt((jnp.square(error) * mask).sum()
                                     / jnp.maximum(mask.sum(), 1))),
        'max_error_m': float(jnp.where(mask, error, 0.0).max()),
        'return': float((reward * record).sum(0).mean()),
    }


def load_square_run(run, num_envs, device, corrected=True):
    """The env, SHAC agent and trained parameters of a run directory.

    A run finetuned by drones-finetune-square carries residual.msgpack; the env uses it unless
    `corrected` is False. Call under jax.default_device.
    """
    import jax

    from drones.rl.ppo import config_from_dict, load_params
    from drones.rl.shac import SHAC, SHACConfig, params_of
    from drones.sim.residual import init_residual
    from drones.sim.square_env import SquareConfig, SquareEnv

    run = Path(run)
    saved = json.loads((run / 'config.json').read_text())
    env_config = config_from_dict(SquareConfig, {**saved['env'], 'num_envs': num_envs,
                                                 'device': device})
    residual = None
    if corrected and (run / 'residual.msgpack').exists():
        residual = load_params(run / 'residual.msgpack', init_residual(jax.random.key(0)))
    env = SquareEnv(env_config, residual=residual)
    agent = SHAC(env, config_from_dict(SHACConfig, saved['shac']))
    params = load_params(run / 'params.msgpack', params_of(agent.init(jax.random.key(0))))
    return env, agent, params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path, help='run directory written by drones-train-square')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--uncorrected', action='store_true',
                        help='ignore a fitted residual (runs from drones-finetune-square)')
    args = parser.parse_args(argv)
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
    except ImportError:
        sys.exit('Evaluation needs the sim extra:  uv sync --extra sim')
    import jax
    import jax.numpy as jnp

    with jax.default_device(jax.devices(args.device)[0]):
        env, agent, params = load_square_run(args.run, args.num_envs, args.device,
                                             corrected=not args.uncorrected)
        policy = jax.jit(lambda o: agent.act(params['actor'], o))
        key = jax.random.key(args.seed)
        results = {
            'policy': evaluate_square(env, policy, key),
            'open-loop hover': evaluate_square(env, lambda o: jnp.zeros((env.num_envs, 4)), key),
        }
    print_table(results)


def print_table(results):
    names = list(next(iter(results.values())))
    print(f'{"":18s}' + ''.join(f'{n:>17s}' for n in names))
    for label, row in results.items():
        print(f'{label:18s}' + ''.join(f'{row[n]:17.3f}' for n in names))


if __name__ == '__main__':
    main()
