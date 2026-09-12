"""Correct the simulator with real square flights, then finetune a trained policy in it.

    uv run --extra sim drones-finetune-square runs/<name> \
        --flights runs/<name>/flights/*-square*.csv

1. System identification (drones.rl.sysid) fits the thrust gain, latency and a residual wrench to
   the flights, and compares the result with the uncorrected simulator on held-out flights. If the
   corrected model is not better, it stops there.
2. SHAC continues from runs/<name> in the corrected simulator: the fitted thrust gain and latency,
   half the randomisation, the residual on, and learning rates at a quarter.
3. Everything goes to runs/<name>-ft/: sysid.json, residual.msgpack, and all a training run writes.
   The finetuned policy is then evaluated in the corrected and the uncorrected simulator.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path


def finetuned_configs(env_config, shac_config, result, iterations, num_envs, device):
    """The env and SHAC settings to finetune with, given a system-ID result."""
    env = dataclasses.replace(env_config, num_envs=num_envs, device=device,
                              thrust_gain=result.thrust_gain,
                              thrust_gain_range=env_config.thrust_gain_range / 2,
                              latency_steps=(result.latency,))
    shac = dataclasses.replace(shac_config, iterations=iterations,
                               actor_lr=shac_config.actor_lr / 4,
                               critic_lr=shac_config.critic_lr / 4)
    return env, shac


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path, help='run directory written by drones-train-square')
    parser.add_argument('--flights', type=Path, nargs='+', required=True,
                        help='square flight logs from drones-fly-square')
    parser.add_argument('--out', type=Path, help='run directory to write (default: RUN-ft)')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=50)
    args = parser.parse_args(argv)
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
    except ImportError:
        sys.exit('Finetuning needs the sim extra:  uv sync --extra sim')
    import jax

    from drones.rl.evaluate_square import evaluate_square, load_square_run, print_table
    from drones.rl.export import export_square_policy
    from drones.rl.ppo import save_params
    from drones.rl.shac import SHAC
    from drones.rl.sysid import identify, load_flights
    from drones.rl.train_square import train, write_config
    from drones.sim.square_env import SquareEnv

    out = args.out or args.run.with_name(args.run.name + '-ft')
    out.mkdir(parents=True, exist_ok=True)
    with jax.default_device(jax.devices(args.device)[0]):
        # corrected=False: if `run` was itself already finetuned, its residual must not be stacked
        # into what this pass treats as the uncorrected baseline.
        base_env, base_agent, params = load_square_run(args.run, args.num_envs, args.device,
                                                        corrected=False)
        try:
            segments = load_flights(args.flights)
        except (OSError, ValueError) as exc:
            sys.exit(f'Cannot read the flights: {exc}')
        result = identify(segments, base_env.config)
        (out / 'sysid.json').write_text(json.dumps(result.report, indent=2))
        print(f'Thrust gain {result.thrust_gain:.3f} | latency {result.latency} steps | '
              f'held-out error {result.report["uncorrected"]["score_horizon"]:.2f} -> '
              f'{result.report["full"]["score_horizon"]:.2f}')
        if not result.improved:
            sys.exit(f'The corrected simulator is no better than the uncorrected one on held-out '
                     f'flights; not finetuning. See {out / "sysid.json"}')
        save_params(out / 'residual.msgpack', result.residual)

        env_config, shac_config = finetuned_configs(base_env.config, base_agent.config, result,
                                                    args.iterations, args.num_envs, args.device)
        write_config(out, env_config, shac_config)
        env = SquareEnv(env_config, residual=result.residual)
        agent = SHAC(env, shac_config)
        state = agent.init(jax.random.key(args.seed))
        state = state.replace(actor=params['actor'], critic=params['critic'],
                              target=params['target'])
        state = train(agent, state, out, args.log_every, args.save_every)
        print(f'Policy artifact: {export_square_policy(env, state.actor, out / "policy")}')

        policy = jax.jit(lambda o: agent.act(state.actor, o))
        key = jax.random.key(args.seed + 1)
        print_table({'corrected sim': evaluate_square(env, policy, key),
                     'uncorrected sim': evaluate_square(base_env, policy, key)})


if __name__ == '__main__':
    main()
