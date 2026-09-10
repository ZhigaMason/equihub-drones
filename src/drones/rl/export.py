"""Export trained parameters as a policy artifact the drone can fly.

    uv run --extra sim drones-export-policy runs/<name>     # writes runs/<name>/policy/

Training exports automatically when it finishes; this is for older runs or other checkpoints.
The artifact is read by drones.policy.runtime, with numpy alone.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from drones.policy.runtime import Policy, PolicySpec


def export_policy(env, params, directory):
    """Write the actor of `params`, with `env`'s observation and action spec, to `directory`."""
    if env.image_shape is not None:
        raise ValueError('camera policies cannot be exported for the drone yet')
    actor = params['params']['actor']
    names = sorted(actor, key=lambda name: int(name.rsplit('_', 1)[1]))
    layers = [(np.asarray(actor[n]['kernel']), np.asarray(actor[n]['bias'])) for n in names]
    return Policy(PolicySpec(**env.policy_spec()), layers).save(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path, help='run directory written by drones-train-hover')
    parser.add_argument('--out', type=Path, help='artifact directory (default: RUN/policy)')
    args = parser.parse_args(argv)
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
    except ImportError:
        sys.exit('Exporting needs the sim extra:  uv sync --extra sim')
    import jax

    from drones.rl.evaluate import load_run

    with jax.default_device(jax.devices('cpu')[0]):
        env, _, params = load_run(args.run, num_envs=2, device='cpu')
        out = export_policy(env, params, args.out or args.run / 'policy')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
