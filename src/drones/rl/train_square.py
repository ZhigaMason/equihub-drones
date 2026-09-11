"""Train a square-flying policy with SHAC on CrazyFlow, from a YAML experiment config.

    uv run --extra sim drones-train-square --preset cpu-test           # a quick check
    uv run --extra sim drones-train-square --preset cpu
    uv run --extra sim --extra gpu drones-train-square --device gpu
    uv run --extra sim drones-train-square --set shac.horizon=16 --set env.lap_time=[8,8]

The config defaults to configs/square/shac.yaml. Each run writes to runs/<name>/:
    config.yaml     the fully resolved config; pass it back in to rerun exactly
    config.json     the same, read by drones-eval-square and drones-finetune-square
    metrics.csv     training curves
    params.msgpack  actor, critic and target critic (every --save-every iterations and at the end)
    policy/         the numpy artifact drones-fly-square flies
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

PRESET_NAMES = ('cpu-test', 'cpu', 'gpu')
LOGGED = ('episode_return', 'episode_length', 'crash_rate', 'pos_error', 'speed', 'tilt',
          'reward', 'action_std', 'actor_loss', 'critic_loss', 'grad_norm', 'skipped')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('config', nargs='?', type=Path,
                        help='experiment YAML (default: configs/square/shac.yaml)')
    parser.add_argument('--preset', choices=PRESET_NAMES, help='resize the experiment')
    parser.add_argument('--device', help='cpu or gpu (default: from the config, else cpu)')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='SECTION.KEY=VALUE', help='override one setting; repeatable')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--name', help='run directory name; defaults to a timestamp')
    parser.add_argument('--log-every', type=int, default=10, help='iterations between log lines')
    parser.add_argument('--save-every', type=int, default=100,
                        help='iterations between checkpoints')
    return parser.parse_args(argv)


def write_config(run, env_config, shac_config):
    from drones.rl import square_experiment
    from drones.rl.ppo import config_to_json

    square_experiment.dump(env_config, shac_config, run / 'config.yaml')
    (run / 'config.json').write_text(json.dumps({
        'env': json.loads(config_to_json(env_config)),
        'shac': json.loads(config_to_json(shac_config)),
    }, indent=2))


def train(agent, state, run, log_every=10, save_every=100):
    """Run the agent's configured iterations, logging to run/metrics.csv and checkpointing to
    run/params.msgpack. Returns the final state."""
    import jax

    from drones.rl.ppo import save_params
    from drones.rl.shac import params_of

    iterations = agent.config.iterations
    with open(run / 'metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('iteration', 'env_steps', 'seconds') + LOGGED)
        start = time.time()
        for i in range(iterations):
            state, stats = agent.iterate(state)
            last = i == iterations - 1
            if i % log_every == 0 or last:
                stats = jax.device_get(stats)
                elapsed = time.time() - start
                steps = (i + 1) * agent.batch_size
                writer.writerow((i, steps, round(elapsed, 1))
                                + tuple(float(stats[k]) for k in LOGGED))
                f.flush()
                print(f'it {i:5d} | {steps / max(elapsed, 1e-9):8.0f} steps/s | '
                      f'return {stats["episode_return"]:7.1f} | '
                      f'crash {stats["crash_rate"]:4.2f} | '
                      f'err {stats["pos_error"]:.3f} m | reward {stats["reward"]:.3f} | '
                      f'|g| {stats["grad_norm"]:.2f}', flush=True)
            if i % save_every == 0 or last:
                save_params(run / 'params.msgpack', params_of(state))
    return state


def main(argv=None):
    args = parse_args(argv)
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
    except ImportError:
        sys.exit('Training needs the sim extra:  uv sync --extra sim')
    import jax

    from drones.rl import square_experiment
    from drones.rl.export import export_square_policy
    from drones.rl.shac import SHAC
    from drones.sim.square_env import SquareEnv

    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f'shac.seed={args.seed}')
    try:
        env_config, shac_config = square_experiment.resolve(args.config, args.preset, overrides,
                                                            args.device)
    except (ValueError, OSError) as exc:
        sys.exit(f'Config error: {exc}')

    with jax.default_device(jax.devices(env_config.device)[0]):
        run = args.runs / (args.name or time.strftime('square-%Y%m%d-%H%M%S'))
        run.mkdir(parents=True, exist_ok=True)
        write_config(run, env_config, shac_config)
        print(f'Building {env_config.num_envs} worlds on {env_config.device} ...', flush=True)
        env = SquareEnv(env_config)
        agent = SHAC(env, shac_config)
        state = agent.init(jax.random.key(shac_config.seed))
        print(f'Hover thrust {env.hover_thrust:.4f} N | observation {env.policy_size} values | '
              f'{shac_config.iterations} iterations of {agent.batch_size} steps | run dir {run}',
              flush=True)
        state = train(agent, state, run, args.log_every, args.save_every)
        print(f'Saved {run / "params.msgpack"}')
        print(f'Policy artifact: {export_square_policy(env, state.actor, run / "policy")}')


if __name__ == '__main__':
    main()
