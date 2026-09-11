"""Square experiments as YAML: the task and the SHAC settings.

    extends: shac.yaml     # optional, relative to this file; merged key by key
    env:                   # drones.sim.square_env.SquareConfig
      num_envs: 4096
    shac:                  # drones.rl.shac.SHACConfig
      iterations: 4000

The rules are drones.rl.experiment's: unknown keys are errors, and values are coerced to the field's
type.
"""
from dataclasses import asdict
from pathlib import Path

import yaml

from drones.rl import experiment
from drones.rl.shac import SHACConfig
from drones.sim.square_env import SquareConfig

CONFIG_DIR = experiment.CONFIG_DIR.parent / 'square'
DEFAULT_CONFIG = CONFIG_DIR / 'shac.yaml'
SECTIONS = ('env', 'shac')

# Scale overlays: the same experiment, sized for the machine it runs on.
PRESETS = {
    # Checks that everything runs end to end on a laptop; will not learn much.
    'cpu-test': {'env': {'num_envs': 32}, 'shac': {'iterations': 20}},
    # Enough to see the square taking shape on a multi-core CPU.
    'cpu': {'env': {'num_envs': 256}, 'shac': {'iterations': 1000}},
    # A full run on one GPU.
    'gpu': {'env': {'num_envs': 4096}, 'shac': {'iterations': 4000}},
}


def build(data):
    """(SquareConfig, SHACConfig) from a loaded config dict."""
    env = experiment._build(SquareConfig, data.get('env') or {}, 'env')
    shac = experiment._build(SHACConfig, data.get('shac') or {}, 'shac')
    return env, shac


def resolve(path=None, preset=None, assignments=(), device=None):
    """Load a config, then apply a preset, `--set` assignments and a device, in that order."""
    data = experiment.load(path or DEFAULT_CONFIG, SECTIONS)
    if preset:
        data = experiment.merge(data, PRESETS[preset])
    data = experiment.override(data, assignments, SECTIONS)
    if device:
        data = experiment.merge(data, {'env': {'device': device}})
    return build(data)


def dump(env, shac, path):
    """Write the resolved config in file form; loading it back gives the same dataclasses."""
    data = {'env': experiment._plain(asdict(env)), 'shac': experiment._plain(asdict(shac))}
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))
