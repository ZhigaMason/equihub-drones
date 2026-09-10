"""YAML experiment configs: sensor selection, extends, overrides and strictness."""
import pytest

pytest.importorskip('crazyflow')

import yaml

from drones.policy.interface import BASELINE
from drones.rl import experiment
from drones.rl.experiment import CONFIG_DIR


def test_the_default_config_is_the_baseline_sensors():
    env, ppo = experiment.resolve()
    assert env.sensors.enabled == BASELINE
    assert env.num_envs == 4096 and ppo.total_steps == 200_000_000


def test_extends_chains_and_merges_key_by_key():
    env, _ = experiment.resolve(CONFIG_DIR / 'camera.yaml')
    assert env.sensors.enabled == ('multiranger', 'optical_flow', 'imu', 'camera')
    assert env.sensors.camera and env.sensors.camera_resolution == (32, 24)
    assert env.num_envs == 1024
    assert env.history == 3  # inherited from baseline.yaml through imu.yaml


def test_preset_then_overrides_are_applied_in_order():
    env, ppo = experiment.resolve(CONFIG_DIR / 'imu.yaml', 'cpu',
                                  ['ppo.total_steps=2e6', 'sensors.enabled=[multiranger]',
                                   'env.num_envs=64'], device='cpu')
    assert env.num_envs == 64 and ppo.total_steps == 2_000_000
    assert ppo.minibatches == 8  # from the preset
    assert env.sensors.enabled == ('multiranger',)
    assert env.device == 'cpu'


@pytest.mark.parametrize('bad, message', [
    ({'ppo': {'totl_steps': 5}}, 'unknown ppo keys'),
    ({'sensors': {'enabled': ['lidar']}}, 'unknown sensors'),
    ({'env': {'sensors': {'enabled': ['imu']}}}, 'top-level sensors'),
    ({'ppo': {'total_steps': 1.5}}, 'integer'),
    ({'model': {}}, 'unknown sections'),
])
def test_mistakes_are_errors_not_silent_defaults(tmp_path, bad, message):
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match=message):
        experiment.resolve(path)


def test_malformed_override_is_rejected():
    with pytest.raises(ValueError, match='SECTION.KEY=VALUE'):
        experiment.resolve(assignments=['total_steps=5'])


def test_resolved_config_reloads_to_the_same_dataclasses(tmp_path):
    env, ppo = experiment.resolve(CONFIG_DIR / 'camera.yaml', 'cpu-test')
    experiment.dump(env, ppo, tmp_path / 'config.yaml')
    assert experiment.resolve(tmp_path / 'config.yaml') == (env, ppo)
