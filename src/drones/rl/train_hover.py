"""Train a hover-stabilising policy on CrazyFlow, from a YAML experiment config.

    uv run --extra sim drones-train-hover --preset cpu-test                  # baseline, a quick check
    uv run --extra sim drones-train-hover configs/hover/imu.yaml --preset cpu
    uv run --extra sim --extra gpu drones-train-hover configs/hover/baseline.yaml --device gpu
    uv run --extra sim drones-train-hover --set sensors.enabled=[multiranger] --set ppo.total_steps=2e6

The config defaults to configs/hover/baseline.yaml. --preset resizes it for the machine, then each
--set SECTION.KEY=VALUE overrides a single setting. Each run writes to runs/<name>/:
    config.yaml     the fully resolved config; pass it back in to rerun exactly
    config.json     the same, read by drones-eval-hover
    metrics.csv     training curves
    params.msgpack  the trained networks (every --save-every iterations and at the end)
    policy/         the numpy artifact drones-fly-policy flies (not for camera policies)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

PRESET_NAMES = ('cpu-test', 'cpu', 'gpu')
LOGGED = ('episode_return', 'episode_length', 'crash_rate', 'height_error', 'speed', 'tilt',
          'reward', 'action_std', 'policy_loss', 'value_loss', 'approx_kl', 'clip_fraction')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('config', nargs='?', type=Path,
                        help='experiment YAML (default: configs/hover/baseline.yaml)')
    parser.add_argument('--preset', choices=PRESET_NAMES, help='resize the experiment')
    parser.add_argument('--device', help='cpu or gpu (default: from the config, else cpu)')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='SECTION.KEY=VALUE', help='override one setting; repeatable')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--name', help='run directory name; defaults to a timestamp')
    parser.add_argument('--log-every', type=int, default=10, help='iterations between log lines')
    parser.add_argument('--save-every', type=int, default=100, help='iterations between checkpoints')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
    except ImportError:
        sys.exit('Training needs the sim extra:  uv sync --extra sim')
    import jax

    from drones.rl import experiment
    from drones.rl.export import export_policy
    from drones.rl.ppo import PPO, config_to_json, save_params
    from drones.sim.hover_env import HoverEnv

    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f'ppo.seed={args.seed}')
    try:
        env_config, ppo_config = experiment.resolve(args.config, args.preset, overrides,
                                                     args.device)
    except (ValueError, OSError) as exc:
        sys.exit(f'Config error: {exc}')

    with jax.default_device(jax.devices(env_config.device)[0]):
        run = args.runs / (args.name or time.strftime('hover-%Y%m%d-%H%M%S'))
        run.mkdir(parents=True, exist_ok=True)
        experiment.dump(env_config, ppo_config, run / 'config.yaml')
        (run / 'config.json').write_text(json.dumps({
            'env': json.loads(config_to_json(env_config)),
            'ppo': json.loads(config_to_json(ppo_config)),
        }, indent=2))

        print(f'Sensors: {", ".join(env_config.sensors.enabled)} | building '
              f'{env_config.num_envs} worlds on {env_config.device} ...', flush=True)
        env = HoverEnv(env_config)
        ppo = PPO(env, ppo_config)
        state = ppo.init(jax.random.key(ppo_config.seed))
        print(f'Hover thrust {env.hover_thrust:.4f} N | observation {env.policy_size} values | '
              f'{ppo.iterations} iterations of {ppo.batch_size} steps | run dir {run}', flush=True)

        with open(run / 'metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('iteration', 'env_steps', 'seconds') + LOGGED)
            start = time.time()
            for i in range(ppo.iterations):
                state, stats = ppo.iterate(state)
                last = i == ppo.iterations - 1
                if i % args.log_every == 0 or last:
                    stats = jax.device_get(stats)
                    elapsed = time.time() - start
                    steps = (i + 1) * ppo.batch_size
                    writer.writerow((i, steps, round(elapsed, 1))
                                    + tuple(float(stats[k]) for k in LOGGED))
                    f.flush()
                    print(f'it {i:5d} | {steps / max(elapsed, 1e-9):9.0f} steps/s | '
                          f'return {stats["episode_return"]:7.1f} | '
                          f'len {stats["episode_length"]:5.0f} | '
                          f'crash {stats["crash_rate"]:4.2f} | '
                          f'h.err {stats["height_error"]:.3f} m | '
                          f'speed {stats["speed"]:.3f} m/s', flush=True)
                if i % args.save_every == 0 or last:
                    save_params(run / 'params.msgpack', state.params)

        print(f'Saved {run / "params.msgpack"}')
        if env.image_shape is None:
            print(f'Policy artifact: {export_policy(env, state.params, run / "policy")}')
        else:
            print('Camera policy: no flight artifact exported (see configs/hover/camera.yaml).')


if __name__ == '__main__':
    main()
