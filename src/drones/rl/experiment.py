"""Hover experiments as YAML: which sensors the policy gets, the task, and the PPO settings.

A config has three sections, each mapping onto a dataclass, and may extend another file:

    extends: baseline.yaml        # optional, relative to this file; merged key by key
    sensors:                      # drones.sim.sensors.SensorConfig
      enabled: [multiranger, optical_flow]
    env:                          # drones.sim.hover_env.HoverConfig, everything but sensors
      num_envs: 4096
    ppo:                          # drones.rl.ppo.PPOConfig
      total_steps: 200_000_000

Unknown keys are errors, so a typo cannot silently fall back to a default. Values are coerced to
the field's type, so `2e8` works for an integer field.
"""
from dataclasses import asdict, fields
from pathlib import Path

import yaml

from drones.rl.ppo import PPOConfig
from drones.sim.hover_env import HoverConfig
from drones.sim.sensors import SensorConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / 'configs' / 'hover'
DEFAULT_CONFIG = CONFIG_DIR / 'baseline.yaml'
SECTIONS = ('sensors', 'env', 'ppo')

# Scale overlays: the same experiment, sized for the machine it runs on.
PRESETS = {
    # Checks that everything runs end to end on a laptop; will not learn much.
    'cpu-test': {'env': {'num_envs': 32},
                 'ppo': {'total_steps': 100_000, 'rollout_steps': 32, 'minibatches': 4}},
    # Enough to see learning on a multi-core CPU, in minutes.
    'cpu': {'env': {'num_envs': 256},
            'ppo': {'total_steps': 5_000_000, 'rollout_steps': 64, 'minibatches': 8}},
    # A full run on one GPU.
    'gpu': {'env': {'num_envs': 4096},
            'ppo': {'total_steps': 200_000_000, 'rollout_steps': 64, 'minibatches': 16}},
}


def load(path):
    """Read a config file into a nested dict, resolving any `extends` chain."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a mapping at the top level')
    parent = data.pop('extends', None)
    if parent is not None:
        data = merge(load(path.parent / parent), data)
    unknown = sorted(set(data) - set(SECTIONS))
    if unknown:
        raise ValueError(f'{path}: unknown sections {unknown}; expected {list(SECTIONS)}')
    return data


def merge(base, override):
    """Merge nested dicts key by key. Anything that is not a dict, lists included, is replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def override(data, assignments):
    """Apply `section.key=value` assignments; each value is parsed as YAML."""
    for assignment in assignments:
        path, sep, raw = assignment.partition('=')
        section, dot, key = path.partition('.')
        if not sep or not dot or section not in SECTIONS or not key:
            raise ValueError(f'expected SECTION.KEY=VALUE with SECTION one of {list(SECTIONS)}, '
                             f'got {assignment!r}')
        data = merge(data, {section: {key: yaml.safe_load(raw)}})
    return data


def build(data):
    """(HoverConfig, PPOConfig) from a loaded config dict."""
    env_data = data.get('env') or {}
    if 'sensors' in env_data:
        raise ValueError('sensor settings belong in the top-level sensors: section')
    sensors = _build(SensorConfig, data.get('sensors') or {}, 'sensors')
    env = _build(HoverConfig, env_data, 'env', fixed={'sensors': sensors})
    ppo = _build(PPOConfig, data.get('ppo') or {}, 'ppo')
    return env, ppo


def resolve(path=None, preset=None, assignments=(), device=None):
    """Load a config, then apply a preset, `--set` assignments and a device, in that order."""
    data = load(path or DEFAULT_CONFIG)
    if preset:
        data = merge(data, PRESETS[preset])
    data = override(data, assignments)
    if device:
        data = merge(data, {'env': {'device': device}})
    return build(data)


def to_dict(env, ppo):
    """The resolved config in file form; loading it back gives the same dataclasses."""
    env_data = asdict(env)
    sensors = env_data.pop('sensors')
    return {'sensors': _plain(sensors), 'env': _plain(env_data), 'ppo': _plain(asdict(ppo))}


def dump(env, ppo, path):
    Path(path).write_text(yaml.safe_dump(to_dict(env, ppo), sort_keys=False))


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _build(cls, data, where, fixed=None):
    fixed = fixed or {}
    known = {f.name for f in fields(cls)} - set(fixed)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f'unknown {where} keys {unknown}; valid keys: {sorted(known)}')
    defaults = cls()
    kwargs = {name: _coerce(value, getattr(defaults, name), f'{where}.{name}')
              for name, value in data.items()}
    return cls(**kwargs, **fixed)


def _coerce(value, default, where):
    """Coerce a YAML value to the type of the field's default."""
    try:
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise TypeError
            return value
        if isinstance(default, int):
            number = float(value)
            if not number.is_integer():
                raise TypeError
            return int(number)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return str(value)
        if isinstance(default, tuple):
            if not isinstance(value, (list, tuple)):
                raise TypeError
            element = default[0] if default else None
            return tuple(v if element is None else _coerce(v, element, where) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f'{where}: expected {_describe(default)}, got {value!r}') from None
    return value


def _describe(default):
    for kind, words in ((bool, 'true or false'), (int, 'an integer'), (float, 'a number'),
                        (str, 'text'), (tuple, 'a list')):
        if isinstance(default, kind):
            return words
    return type(default).__name__
